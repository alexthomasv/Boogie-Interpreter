"""
Unified interpreter runner — dispatches between Python and Rust native engines.

Usage:
    python -m interpreter.runner <test_pkg_path> [--engine=python|native|both] [--force] [--full-trace]

Architecture:
    The native engine uses a 2-phase approach:
    1. Python handles initialization + concretization (AST walking, I/O, <1s)
    2. Rust handles execution (the hot loop that takes minutes on PyPy)
"""
import argparse
import hashlib
import json
import multiprocessing
import pickle
import struct
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import functools
import os

from interpreter.python.interpreter import BoogieInterpreter, find_entry_point


# ---------------------------------------------------------------------------
# Shared state for fork-based COW sharing across worker processes.
# Loaded once in the parent; forked children inherit via COW pages.
# ---------------------------------------------------------------------------
_SHARED_PROGRAM = None
_SHARED_COMPILED = None   # CompiledProgramWrapper from swoosh_interp.lower()
_SHARED_FIELD_SIZES = None


def _load_shared(test_path, engine):
    """Load program + compile bytecode once in the parent process."""
    global _SHARED_PROGRAM, _SHARED_COMPILED, _SHARED_FIELD_SIZES

    with open(test_path, 'rb') as f:
        _SHARED_PROGRAM = pickle.load(f)

    from interpreter.utils.input_parser import get_bpl_field_sizes
    _SHARED_FIELD_SIZES = get_bpl_field_sizes(test_path.parent, program=_SHARED_PROGRAM)

    if engine in ('native', 'both'):
        try:
            import swoosh_interp
            _SHARED_COMPILED = swoosh_interp.lower(_SHARED_PROGRAM)
        except ImportError:
            _SHARED_COMPILED = None


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_reachable_blocks(program):
    """Compute the set of blocks reachable from the entry block via the CFG.

    After inlining, some blocks become unreachable dead code. This function
    walks the CFG from the entry block to find only reachable blocks.
    Returns (reachable_set, entry_block_name) or (None, None) if no entry.
    """
    entry = find_entry_point(program)
    if entry is None:
        return None, None

    blocks = entry.body.blocks
    if not blocks:
        return set(), None

    from interpreter.parser.statement import GotoStatement

    # Build adjacency from goto statements
    successors = {}
    for block in blocks:
        targets = set()
        if block.statements:
            last = block.statements[-1]
            if isinstance(last, GotoStatement):
                targets = {ident.name for ident in last.identifiers}
        successors[block.name] = targets

    # BFS from entry block
    entry_block = blocks[0].name
    visited = set()
    queue = [entry_block]
    while queue:
        b = queue.pop()
        if b in visited:
            continue
        visited.add(b)
        for s in successors.get(b, ()):
            if s not in visited:
                queue.append(s)

    return visited, entry_block


def compute_coverage(program, explored_blocks):
    """Compute block and statement coverage from explored blocks.

    Reports both raw totals (all blocks in the procedure) and reachable
    totals (blocks reachable from entry via CFG). After inlining, some
    blocks are dead code — reachable coverage is the meaningful metric.
    """
    entry = find_entry_point(program)
    if entry is None:
        return None
    blocks = entry.body.blocks
    block_stmts = {b.name: len(b.statements) for b in blocks}
    total_blocks = len(blocks)
    total_stmts = sum(block_stmts.values())
    covered_blocks = sum(1 for b in explored_blocks if b in block_stmts)
    covered_stmts = sum(block_stmts[b] for b in explored_blocks if b in block_stmts)

    # Compute reachable subset
    reachable, _ = compute_reachable_blocks(program)
    if reachable is not None:
        reachable_blocks_total = len(reachable & block_stmts.keys())
        reachable_stmts_total = sum(block_stmts[b] for b in reachable if b in block_stmts)
        unreachable_blocks = total_blocks - reachable_blocks_total
        # Coverage of reachable blocks only
        reachable_covered = sum(1 for b in explored_blocks if b in reachable and b in block_stmts)
        reachable_stmts_covered = sum(block_stmts[b] for b in explored_blocks if b in reachable and b in block_stmts)
    else:
        reachable_blocks_total = total_blocks
        reachable_stmts_total = total_stmts
        unreachable_blocks = 0
        reachable_covered = covered_blocks
        reachable_stmts_covered = covered_stmts

    return {
        "blocks_covered": covered_blocks,
        "blocks_total": total_blocks,
        "stmts_covered": covered_stmts,
        "stmts_total": total_stmts,
        "block_pct": round(100.0 * covered_blocks / total_blocks, 1) if total_blocks else 0,
        "stmt_pct": round(100.0 * covered_stmts / total_stmts, 1) if total_stmts else 0,
        "reachable_blocks_total": reachable_blocks_total,
        "reachable_stmts_total": reachable_stmts_total,
        "unreachable_blocks": unreachable_blocks,
        "reachable_block_pct": round(100.0 * reachable_covered / reachable_blocks_total, 1) if reachable_blocks_total else 0,
        "reachable_stmt_pct": round(100.0 * reachable_stmts_covered / reachable_stmts_total, 1) if reachable_stmts_total else 0,
        "explored_blocks": sorted(explored_blocks & block_stmts.keys()),
    }


# ---------------------------------------------------------------------------
# Python engine
# ---------------------------------------------------------------------------

def run_python(program, program_inputs, test_name, input_name, full_trace=False, no_read_trace=False):
    """Run the Python interpreter. Returns (explored_blocks, compact_path)."""
    from interpreter.python.Environment import Environment

    environment = Environment(test_name, input_name)
    environment.full_trace = full_trace
    if no_read_trace:
        environment.LOG_READ = False

    interp = BoogieInterpreter(environment, program_inputs, input_name)
    interp.preprocess(program)

    entry = find_entry_point(program)
    assert entry is not None, "No entry point found"

    interp.execute_procedure(entry)

    return interp.explored_blocks, environment._compact_path


# ---------------------------------------------------------------------------
# Native engine — 2 phase: Python init → Rust execution
# ---------------------------------------------------------------------------

def _init_python_state(program, program_inputs, test_name, input_name, no_read_trace=False):
    """Phase 1: Python initialization + concretization.
    Returns initialized BoogieInterpreter (ready to execute, but not yet running)."""
    from interpreter.python.Environment import Environment

    environment = Environment(test_name, input_name)
    environment.full_trace = False
    if no_read_trace:
        environment.LOG_READ = False

    interp = BoogieInterpreter(environment, program_inputs, input_name)
    interp.preprocess(program)

    entry = find_entry_point(program)
    assert entry is not None, "No entry point found"

    # Do all the initialization that execute_procedure does BEFORE the hot loop
    interp._initialize_vars(entry.body.locals, kind="local")
    interp.concretize_proc_inputs(entry)
    interp.concretize_input_memory()

    return interp, entry


def _extract_state(interp):
    """Extract initialized state from Python interpreter as plain dicts for Rust."""
    from interpreter.python.MemoryMap import MemoryMap

    env = interp.env
    var_store = {}
    memory_maps = {}
    mem_map_info = []

    MASK_64 = (1 << 64) - 1
    I64_MAX = (1 << 63) - 1

    for key, value in env.last_frame().var_store.items():
        if isinstance(value, MemoryMap):
            memory_maps[key] = dict(value.memory)
            mem_map_info.append((key, value.index_bit_width, value.element_bit_width))
        elif isinstance(value, (int, bool)):
            v = int(value)
            # Clamp to signed i64 range for Rust
            if v > I64_MAX or v < -(I64_MAX + 1):
                v = v & MASK_64
                if v > I64_MAX:
                    v -= (1 << 64)
            var_store[key] = v

    # Capture $CurrAddr state
    if env.alloc_addr:
        var_store["$CurrAddr"] = env.alloc_addr
    if env.alloc_addr_shadow:
        var_store["$CurrAddr.shadow"] = env.alloc_addr_shadow

    return var_store, memory_maps, mem_map_info


def _extract_init_trace(interp):
    """Extract trace entries accumulated during Python's init/concretization phase.
    Returns dict in native trace format: {section: {key: set_of_raw_values}}."""
    env = interp.env
    trace = {'pc_values': {}, 'block_values': {}, 'op_values': {},
             'pc_registry': {}, 'block_registry': {}, 'total': env._agg_total}

    for (k, pc), vals in env._agg_pc_values.items():
        key = f"positive_examples_{k}_{pc}"
        trace['pc_values'][key] = {v[0] if isinstance(v, tuple) else v for v in vals}
    for (k, blk), vals in env._agg_block_values.items():
        key = f"positive_examples_{k}_{blk}"
        trace['block_values'][key] = {v[0] if isinstance(v, tuple) else v for v in vals}
    for (k, pc, op), vals in env._agg_op_values.items():
        key = f"positive_examples_to_op_type_{k}_{pc}_{op}"
        trace['op_values'][key] = {v[0] if isinstance(v, tuple) else v for v in vals}
    for k, pcs in env._agg_pc_registry.items():
        key = f"positive_examples_to_pc_{k}"
        trace['pc_registry'][key] = set(pcs)
    for k, blks in env._agg_block_registry.items():
        key = f"positive_examples_to_block_{k}"
        trace['block_registry'][key] = set(blks)

    return trace


def _merge_traces(init_trace, native_trace, pickled=False):
    """Merge init-phase trace into native trace.
    If pickled=True, convert init trace values to pickled bytes first."""
    merged = {}
    for section in ('pc_values', 'block_values', 'op_values', 'pc_registry', 'block_registry'):
        merged[section] = dict(native_trace.get(section, {}))
        for key, vals in init_trace.get(section, {}).items():
            if pickled:
                # Convert raw init trace values to pickled bytes
                if section in ('pc_values', 'block_values', 'op_values'):
                    vals = {pickle.dumps((v, "")) for v in vals}
                elif section == 'pc_registry':
                    vals = {pickle.dumps(v) for v in vals}
                elif section == 'block_registry':
                    vals = {pickle.dumps(v) for v in vals}
            if key in merged[section]:
                merged[section][key] = merged[section][key] | vals
            else:
                merged[section][key] = vals
    merged['total'] = native_trace.get('total', 0) + init_trace.get('total', 0)
    return merged


def run_native(program, program_inputs, test_name, input_name, extra_data=None, log_read=True, pickled=False, compiled=None):
    """Run the Rust native interpreter. Returns (explored_blocks, compact_trace_dict).
    If pickled=True, value sets contain pre-pickled bytes (ready for Redis/disk).
    If compiled is provided, skips the lower() step (reuses pre-compiled bytecode)."""
    try:
        import swoosh_interp
    except ImportError:
        raise ImportError(
            "Native interpreter not built. Run: cd interpreter/native && maturin develop --release"
        )

    # Phase 1: Python initialization + concretization
    interp, entry = _init_python_state(program, program_inputs, test_name, input_name,
                                        no_read_trace=not log_read)
    var_store, memory_maps, mem_map_info = _extract_state(interp)

    # Capture init-phase trace before handing off to Rust
    init_trace = _extract_init_trace(interp)

    # Phase 2: Lower AST to Rust bytecode (skip if pre-compiled)
    if compiled is None:
        compiled = swoosh_interp.lower(program)

    # Phase 3: Execute in Rust
    ext_data = extra_data
    if ext_data is None and hasattr(interp, 'external_buffer') and interp.external_buffer.unread_bytes() > 0:
        ext_data = bytes(interp.external_buffer._buf[interp.external_buffer._pos:])

    result = swoosh_interp.execute(
        compiled,
        var_store,
        memory_maps,
        mem_map_info,
        extra_data=ext_data,
        log_read=log_read,
        pickled=pickled,
    )

    # Merge init-phase trace with native execution trace
    merged_trace = _merge_traces(init_trace, result['compact_trace'], pickled=pickled)

    return result['explored_blocks'], merged_trace


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def run_both(program, program_inputs, test_name, input_name, full_trace=False,
             no_read_trace=False, extra_data=None):
    """Run both interpreters and assert identical output."""
    print(f"[both] Running Python interpreter...")
    t0 = time.time()
    py_blocks, py_trace_path = run_python(
        program, program_inputs, test_name, input_name,
        full_trace=full_trace, no_read_trace=no_read_trace
    )
    py_time = time.time() - t0
    print(f"[both] Python done in {py_time:.1f}s, {len(py_blocks)} blocks explored")

    print(f"[both] Running native interpreter...")
    t0 = time.time()
    native_blocks, native_trace = run_native(
        program, program_inputs, test_name, input_name,
        extra_data=extra_data, log_read=not no_read_trace
    )
    native_time = time.time() - t0
    print(f"[both] Native done in {native_time:.1f}s, {len(native_blocks)} blocks explored")

    # Compare explored blocks
    if py_blocks != native_blocks:
        py_only = py_blocks - native_blocks
        native_only = native_blocks - py_blocks
        msg = f"Explored blocks differ!\n  Python-only: {py_only}\n  Native-only: {native_only}"
        raise AssertionError(msg)

    # Compare compact traces
    if py_trace_path.exists():
        with open(py_trace_path, 'rb') as f:
            py_compact = pickle.load(f)

        py_pv = py_compact.get('pc_values', {})
        nat_pv = native_trace.get('pc_values', {})
        _compare_trace_dicts(py_pv, nat_pv, "pc_values")

    speedup = py_time / native_time if native_time > 0 else float('inf')
    print(f"[both] PASS — identical output. Speedup: {speedup:.1f}x")

    return py_blocks, py_trace_path


def write_trace_binary(path, compact_trace):
    """Write a merged compact trace dict to the streaming binary format with zstd compression.

    The dict has keys: pc_values, block_values, op_values, pc_registry, block_registry, total.
    Each section maps string keys to sets of pre-pickled bytes.
    """
    import zstandard as zstd

    categories = [
        (0, 'pc_values'),
        (1, 'block_values'),
        (2, 'op_values'),
        (3, 'pc_registry'),
        (4, 'block_registry'),
    ]

    total = compact_trace.get('total', 0)
    cctx = zstd.ZstdCompressor(level=3, threads=-1)
    with open(path, 'wb') as fh:
        with cctx.stream_writer(fh) as writer:
            writer.write(b"SWTR")
            writer.write(struct.pack('<BQ', 1, total))
            for cat_id, section_name in categories:
                section = compact_trace.get(section_name, {})
                writer.write(struct.pack('<BI', cat_id, len(section)))
                for key_str, members in section.items():
                    key_bytes = key_str.encode() if isinstance(key_str, str) else key_str
                    writer.write(struct.pack('<H', len(key_bytes)))
                    writer.write(key_bytes)
                    writer.write(struct.pack('<I', len(members)))
                    for m in members:
                        writer.write(struct.pack('<H', len(m)))
                        writer.write(m)
            writer.write(b"DONE")


def _pickle_native_trace(native_trace):
    """Convert native raw-value trace to pickled-bytes format for verifier compatibility.
    Native returns {key: set_of_raw_values}; verifier expects {key: set_of_pickled_bytes}.
    Value sections store (int, "") tuples; registry sections store raw ints/strings."""
    result = {}
    # Value sections: wrap each int as (value, "") tuple before pickling
    for section in ('pc_values', 'block_values', 'op_values'):
        d = native_trace.get(section, {})
        result[section] = {
            key: {pickle.dumps((v, "")) for v in vals}
            for key, vals in d.items()
        }
    # Registry sections: pickle raw values directly (ints for pc, strings for block)
    for section in ('pc_registry', 'block_registry'):
        d = native_trace.get(section, {})
        result[section] = {
            key: {pickle.dumps(v) for v in vals}
            for key, vals in d.items()
        }
    result['total'] = native_trace.get('total', 0)
    return result


def _unpickle_value_set(value_set):
    """Convert a set of pickled (value, "") bytes to a set of raw ints."""
    result = set()
    for pickled_val in value_set:
        val = pickle.loads(pickled_val)
        if isinstance(val, tuple):
            val = val[0]  # Extract int from (int, "") tuple
        if isinstance(val, bool):
            val = int(val)
        result.add(val)
    return result


def _compare_trace_dicts(py_dict, native_dict, name):
    """Compare two trace dictionaries for equality.
    Python dict has pickled (value,"") bytes; native dict has raw ints."""
    py_keys = set(py_dict.keys())
    nat_keys = set(native_dict.keys())
    if py_keys != nat_keys:
        py_only = py_keys - nat_keys
        nat_only = nat_keys - py_keys
        if py_only:
            print(f"[{name}] Python-only keys (first 5): {list(py_only)[:5]}")
        if nat_only:
            print(f"[{name}] Native-only keys (first 5): {list(nat_only)[:5]}")
    for key in py_keys & nat_keys:
        py_vals = _unpickle_value_set(py_dict[key])
        nat_vals = native_dict[key]
        if py_vals != nat_vals:
            raise AssertionError(
                f"[{name}] Value mismatch at key '{key}':\n"
                f"  Python:  {py_vals}\n"
                f"  Native: {nat_vals}"
            )


# ---------------------------------------------------------------------------
# Single-input processing
# ---------------------------------------------------------------------------

def process_single_input(input_file, test_name, test_path, engine='python',
                         force=False, full_trace=False, no_read_trace=False,
                         program=None, field_sizes=None, compiled=None):
    """Process a single input file with the chosen engine.
    Returns (input_name, coverage_dict, explored_blocks) or None if skipped.
    If program/field_sizes/compiled are provided, uses them instead of loading from disk."""
    try:
        if program is None:
            with open(test_path, 'rb') as file:
                program = pickle.load(file)

        print(f"Processing input file: {input_file}")
        from interpreter.utils.input_parser import parse_input_file, get_bpl_field_sizes
        if field_sizes is None:
            field_sizes = get_bpl_field_sizes(test_path.parent, program=program)
        program_inputs = parse_input_file(input_file, field_sizes=field_sizes)
        input_name = Path(str(input_file)).stem

        trace_dir = Path("positive_examples") / test_name
        trace_dir.mkdir(parents=True, exist_ok=True)
        explored_path = trace_dir / f"{input_name}.explored_blocks.txt"
        compact_path = trace_dir / f"{input_name}.trace.compact.pkl"
        bin_path = trace_dir / f"{input_name}.trace.compact.bin.zst"

        has_trace = (compact_path.exists() and compact_path.stat().st_size > 0) or \
                    (bin_path.exists() and bin_path.stat().st_size > 0)
        if not force and explored_path.exists() and has_trace:
            print(f"Skipping input file: {input_file} (already explored)")
            # Still compute coverage from existing explored_blocks file
            with open(explored_path) as f:
                explored = {line.strip() for line in f if line.strip()}
            cov = compute_coverage(program, explored)
            if cov:
                unreachable = cov.get('unreachable_blocks', 0)
                rblk = f" (reachable: {cov['reachable_block_pct']}%)" if unreachable else ""
                print(f"[coverage] {input_name}: {cov['blocks_covered']}/{cov['blocks_total']} blocks ({cov['block_pct']}%){rblk}, "
                      f"{cov['stmts_covered']}/{cov['stmts_total']} stmts ({cov['stmt_pct']}%)")
            return (input_name, cov, explored)

        if compact_path.exists() and not explored_path.exists():
            compact_path.unlink()
        if bin_path.exists() and not explored_path.exists():
            bin_path.unlink()

        if engine == 'python':
            explored, _ = run_python(program, program_inputs, test_name, input_name,
                                     full_trace=full_trace, no_read_trace=no_read_trace)
        elif engine == 'native':
            explored, compact_trace = run_native(
                program, program_inputs, test_name, input_name,
                extra_data=program_inputs.extra_data,
                log_read=not no_read_trace,
                pickled=True,  # Pre-pickled bytes, ready for verifier
                compiled=compiled,
            )
            with open(compact_path, 'wb') as f:
                pickle.dump(compact_trace, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Also write streaming binary format
            write_trace_binary(bin_path, compact_trace)
        elif engine == 'both':
            explored, _ = run_both(program, program_inputs, test_name, input_name,
                                   full_trace=full_trace, no_read_trace=no_read_trace,
                                   extra_data=program_inputs.extra_data)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        with open(explored_path, "w") as f:
            for block in explored:
                f.write(f"{block}\n")

        cov = compute_coverage(program, explored)
        if cov:
            print(f"[coverage] {input_name}: {cov['blocks_covered']}/{cov['blocks_total']} blocks ({cov['block_pct']}%), "
                  f"{cov['stmts_covered']}/{cov['stmts_total']} stmts ({cov['stmt_pct']}%)")
        return (input_name, cov, explored)

    except Exception:
        import traceback
        traceback.print_exc()
        raise


def _process_input_shared(input_file, test_name, test_path, engine='python',
                          force=False, full_trace=False, no_read_trace=False):
    """Worker function that uses fork-inherited _SHARED_* globals."""
    return process_single_input(
        input_file, test_name=test_name, test_path=test_path, engine=engine,
        force=force, full_trace=full_trace, no_read_trace=no_read_trace,
        program=_SHARED_PROGRAM, field_sizes=_SHARED_FIELD_SIZES,
        compiled=_SHARED_COMPILED,
    )


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def _print_coverage_summary(results, test_name, test_path):
    """Print aggregate coverage and write coverage.json."""
    per_input = {}
    all_explored = set()

    for input_name, cov, explored in results:
        if cov:
            per_input[input_name] = {k: v for k, v in cov.items() if k != "explored_blocks"}
        if explored:
            all_explored |= explored

    if not per_input:
        return

    # Compute aggregate coverage using the union of all explored blocks
    program = _SHARED_PROGRAM
    if program is None:
        with open(test_path, 'rb') as f:
            program = pickle.load(f)
    agg = compute_coverage(program, all_explored)
    if not agg:
        return

    n_inputs = len(per_input)
    unreachable = agg.get('unreachable_blocks', 0)
    rblk = f", {unreachable} unreachable" if unreachable else ""
    rblk_pct = f" (reachable: {agg['reachable_block_pct']}%)" if unreachable else ""
    print(f"\n[coverage] aggregate ({n_inputs} input{'s' if n_inputs != 1 else ''}): "
          f"{agg['blocks_covered']}/{agg['blocks_total']} blocks ({agg['block_pct']}%){rblk_pct}{rblk}, "
          f"{agg['stmts_covered']}/{agg['stmts_total']} stmts ({agg['stmt_pct']}%)")

    # Write coverage.json
    trace_dir = Path("positive_examples") / test_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    coverage_data = {
        "per_input": per_input,
        "aggregate": {k: v for k, v in agg.items() if k != "explored_blocks"},
    }
    with open(trace_dir / "coverage.json", "w") as f:
        json.dump(coverage_data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Boogie interpreter runner')
    parser.add_argument('test_pkg_path', type=str, help='Path to the test package')
    parser.add_argument('--engine', choices=['python', 'native', 'both'], default='python',
                        help='Interpreter engine to use')
    parser.add_argument('--force', action='store_true', help='Force re-interpretation')
    parser.add_argument('--full-trace', action='store_true', help='Write full text trace')
    parser.add_argument('--no-read-trace', action='store_true', help='Skip read tracing')
    args = parser.parse_args()

    test_pkg_dir = Path(args.test_pkg_path)
    test_name = test_pkg_dir.name.removesuffix("_pkg")
    test_path = test_pkg_dir / f"{test_name}.pkl"
    assert test_path.exists(), f"Test path does not exist: {test_path}"

    inline_bpl = Path("bpl_out") / test_name / f"{test_name}_inline.bpl"
    hash_file = Path("positive_examples") / test_name / f"{test_name}.interp.hash"
    trace_dir = Path("positive_examples") / test_name

    if inline_bpl.exists():
        bpl_hash = hashlib.sha256(inline_bpl.read_bytes()).hexdigest()
    else:
        bpl_hash = None

    # Engine mismatch check: clear traces if engine changed
    engine_file = trace_dir / f"{test_name}.engine"
    if trace_dir.exists() and engine_file.exists():
        stored_engine = engine_file.read_text().strip()
        if stored_engine != args.engine and args.engine != "both":
            print(f"Engine changed ({stored_engine} → {args.engine}) — clearing old traces in {trace_dir}")
            shutil.rmtree(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            mem_trace_dir = Path("mem_ops_traces") / test_name
            if mem_trace_dir.exists():
                shutil.rmtree(mem_trace_dir)

    if not args.force and bpl_hash and hash_file.exists() and trace_dir.exists():
        stored_hash = hash_file.read_text().strip()
        if stored_hash == bpl_hash:
            input_directory_check = Path("test_input") / test_name
            input_files_check = list(input_directory_check.glob("*.input"))
            all_done = all(
                (trace_dir / f"{p.stem}.explored_blocks.txt").exists()
                for p in input_files_check
            )
            if all_done and input_files_check:
                print(f"Skipping interpretation — {inline_bpl} unchanged and all inputs complete")
                exit(0)
        elif stored_hash != bpl_hash:
            print(f"Hash mismatch — trashing old trace files in {trace_dir}")
            shutil.rmtree(trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            mem_trace_dir = Path("mem_ops_traces") / test_name
            if mem_trace_dir.exists():
                shutil.rmtree(mem_trace_dir)

    input_directory = Path("test_input") / test_name
    input_files = list(input_directory.glob("*.input"))
    assert len(input_files) > 0, f"No input files found in {input_directory}"

    max_workers = min(max(1, os.cpu_count() - 1), len(input_files))
    print(f"Using {max_workers} workers for {len(input_files)} inputs (engine={args.engine})")

    # Load program + compile bytecode once, then fork workers to share via COW
    assert hasattr(os, 'fork'), "Fork-based multiprocessing required (Linux only)"
    t0 = time.time()
    _load_shared(test_path, args.engine)
    print(f"Loaded program + compiled bytecode in {time.time() - t0:.1f}s (shared via fork)")
    mp_context = multiprocessing.get_context('fork')
    worker_func = functools.partial(
        _process_input_shared,
        test_name=test_name,
        test_path=test_path,
        engine=args.engine,
        force=args.force,
        full_trace=args.full_trace,
        no_read_trace=args.no_read_trace,
    )

    failed = False
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        results_iterator = executor.map(worker_func, input_files)
        try:
            for result in results_iterator:
                if result is not None:
                    results.append(result)
        except Exception:
            import traceback
            print("A worker failed!")
            traceback.print_exc()
            failed = True

    if failed:
        print("Interpretation did not complete — run again to resume")
        exit(1)

    # Coverage summary
    if results:
        _print_coverage_summary(results, test_name, test_path)

    if bpl_hash:
        trace_dir.mkdir(parents=True, exist_ok=True)
        hash_file.write_text(bpl_hash + "\n")

    # Record which engine produced these traces
    trace_dir.mkdir(parents=True, exist_ok=True)
    engine_file.write_text(args.engine + "\n")

    print(f"Done. Engine={args.engine}")


if __name__ == '__main__':
    main()
