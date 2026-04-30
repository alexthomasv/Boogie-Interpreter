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
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import functools
import os

from interpreter.python.interpreter import BoogieInterpreter, find_entry_point


def _build_loop_header_live(test_path):
    """Load loop header → live variable names from compiled package metadata.

    Returns dict {block_name: [var_name, ...]} suitable for swoosh_interp.lower().
    """
    pkg_dir = test_path.parent
    name = test_path.stem
    try:
        live_in_path = pkg_dir / f"{name}_live_in.pkl"
        loops_path = pkg_dir / f"{name}_loops.pkl"
        if not live_in_path.exists() or not loops_path.exists():
            return None
        live_in = pickle.load(open(live_in_path, 'rb'))
        loops = pickle.load(open(loops_path, 'rb'))
        result = {}
        for proc, proc_loops in loops.items():
            proc_live = live_in.get(proc, {})
            for header_block in proc_loops:
                live_vars = proc_live.get(header_block, set())
                if live_vars:
                    result[header_block] = sorted(live_vars)
        return result if result else None
    except Exception:
        return None


def _build_loop_metadata(test_path):
    """Load loop nesting metadata from the compiled package.

    Returns a dict with three keys ready for ``swoosh_interp.lower``:

      * ``is_loop_header``: list of block NAMES that are loop headers.
      * ``block_innermost_header``: dict block_name -> innermost header name.
      * ``loop_parent_header``: dict inner_header -> parent header name.

    The Rust side resolves names to block ids via its own label map, so
    no block-id numbering has to agree across the FFI boundary.
    Returns None if any required .pkl is missing.
    """
    pkg_dir = test_path.parent
    name = test_path.stem
    try:
        loops_path = pkg_dir / f"{name}_loops.pkl"
        parents_path = pkg_dir / f"{name}_loop_parents.pkl"
        btl_path = pkg_dir / f"{name}_block_to_loop.pkl"
        for p in (loops_path, parents_path, btl_path):
            if not p.exists():
                return None
        loops = pickle.load(open(loops_path, 'rb'))
        parents = pickle.load(open(parents_path, 'rb'))
        btl = pickle.load(open(btl_path, 'rb'))

        # Flatten across procedures.  The package only ever has one
        # entry procedure after inlining, but the dicts are still
        # keyed by proc name, so we union them.
        headers = set()
        for proc_headers in loops.values():
            headers.update(proc_headers)

        block_to_header = {}
        for proc_map in btl.values():
            block_to_header.update(proc_map)

        parent_map = {}
        for proc_parents in parents.values():
            for inner, outer in proc_parents.items():
                if outer is not None:
                    parent_map[inner] = outer

        return {
            "is_loop_header": sorted(headers),
            "block_innermost_header": block_to_header,
            "loop_parent_header": parent_map,
        }
    except Exception:
        return None


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
            lh_live = _build_loop_header_live(test_path)
            loop_meta = _build_loop_metadata(test_path)
            _SHARED_COMPILED = swoosh_interp.lower(
                _SHARED_PROGRAM, lh_live, loop_meta)
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

def _init_python_state(program, program_inputs, test_name, input_name,
                       no_read_trace=False, init_raw_log_path=None):
    """Phase 1: Python initialization + concretization.
    Returns initialized BoogieInterpreter (ready to execute, but not yet running)."""
    from interpreter.python.Environment import Environment
    from interpreter.parser.declaration import StorageDeclaration, ConstantDeclaration

    environment = Environment(test_name, input_name)
    environment.full_trace = False
    if no_read_trace:
        environment.LOG_READ = False

    interp = BoogieInterpreter(environment, program_inputs, input_name)
    interp.preprocess(program)

    entry = find_entry_point(program)
    assert entry is not None, "No entry point found"

    # Warn about parameter mismatches before concretization
    input_keys = set(program_inputs.keys()) if hasattr(program_inputs, 'keys') else set()
    if not input_keys and hasattr(program_inputs, 'variables'):
        input_keys = set(program_inputs.variables.keys())
    expected_params = []
    for param in entry.parameters:
        assert len(param.names) == 1
        name = param.names[0]
        if not name.endswith('.shadow'):
            expected_params.append(name)
    missing = [p for p in expected_params if p not in input_keys]
    if missing:
        import sys
        print(f"WARNING: input file '{input_name}' is missing parameters: {missing}. "
              f"Available: {sorted(input_keys)}. "
              f"Check @params mapping in the .input file (use BPL names like $p0, $i2).",
              file=sys.stderr)

    # Optionally stream the init-phase trace to a separate raw log. The
    # driver loads this file alongside the main ``.trace.raw.zst`` at
    # verify time.
    if init_raw_log_path is not None:
        var_names = []
        seen = set()
        def _add(n):
            if n and not n.endswith('.shadow') and n not in seen:
                seen.add(n)
                var_names.append(n)
        for d in program.declarations:
            if isinstance(d, StorageDeclaration):
                for n in d.names:
                    _add(n)
            elif isinstance(d, ConstantDeclaration):
                for n in d.names:
                    _add(n)
        if entry.body:
            for local_decl in entry.body.locals:
                if isinstance(local_decl, StorageDeclaration):
                    for n in local_decl.names:
                        _add(n)
        for p in entry.parameters:
            for n in p.names:
                _add(n)
        block_names = ['GLOBAL']
        seen_b = {'GLOBAL'}
        if entry.body:
            for b in entry.body.blocks:
                if b.name not in seen_b:
                    seen_b.add(b.name)
                    block_names.append(b.name)
        environment._raw_log_path = Path(init_raw_log_path)
        environment.enable_raw_log(var_names, block_names)

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


def _extract_havoc_sequences(program_inputs):
    if not hasattr(program_inputs, "variables"):
        return {}
    out = {}
    for name, inp in program_inputs.variables.items():
        seq = getattr(inp, "havoc_seq", None)
        if seq is not None:
            out[name] = [int(v) for v in seq]
    return out




def _validate_state_no_aliasing(interp):
    """Defense-in-depth: verify no two buffers in the same memory map overlap.

    Called after Python initialization, before handing state to Rust.
    Re-derives ranges from interp.arr_inputs and current variable values.
    Pointers declared equivalent via ``{:ptr_alias}`` are collapsed.
    """
    from collections import defaultdict
    env = interp.env
    aliases = getattr(interp, "ptr_aliases", {}) or {}

    def canon(p):
        seen = set()
        while p in aliases and p not in seen:
            seen.add(p)
            p = aliases[p]
        return p

    # Build map: mem_map -> {canonical_ptr: (base_addr, size)}; keep the max
    # size per canonical pointer so aliased pointers do not appear twice.
    per_map = defaultdict(dict)
    for var_name, arr_infos in interp.arr_inputs.items():
        if '.shadow' in var_name:
            continue
        for arr_info in arr_infos:
            if '.shadow' in arr_info.mem_map:
                continue
            base_addr = env.get_concrete_value(arr_info.base_ptr)
            if base_addr is None:
                continue
            if not isinstance(base_addr, int):
                continue
            size = arr_info.elem_size * arr_info.num_elements
            c_ptr = canon(arr_info.base_ptr)
            existing = per_map[arr_info.mem_map].get(c_ptr)
            if existing is None or size > existing[1]:
                per_map[arr_info.mem_map][c_ptr] = (base_addr, size)

    for mem_map, ptr_dict in per_map.items():
        ranges = [(p, a, a + s) for p, (a, s) in ptr_dict.items()]
        ranges.sort(key=lambda x: x[1])
        for i in range(len(ranges) - 1):
            ptr_a, _, end_a = ranges[i]
            ptr_b, start_b, _ = ranges[i + 1]
            assert end_a <= start_b, (
                f"BUFFER ALIASING at Rust handoff in {mem_map}: "
                f"{ptr_a}[..{end_a}) overlaps {ptr_b}[{start_b}..)"
            )


def run_native(program, program_inputs, test_name, input_name, raw_log_path,
               extra_data=None, log_read=True, compiled=None, no_trace=False,
               init_raw_log_path=None, return_status=False):
    """Run the Rust native interpreter.

    ``raw_log_path`` is the ``.trace.raw.zst`` the Rust VM writes its
    execution records to.  ``init_raw_log_path`` (optional) is a second
    ``.init.raw.zst`` file the Python init phase streams concretization
    records to — the driver loads both files into Redis at verify time.

    Returns ``explored_blocks`` by default. With ``return_status=True``, returns
    the native result dict including ``status`` and any violation metadata.
    """
    try:
        import swoosh_interp
    except ImportError:
        raise ImportError(
            "Native interpreter not built. Run: cd interpreter/native && maturin develop --release"
        )

    # Phase 1: Python initialization + concretization
    interp, entry = _init_python_state(
        program, program_inputs, test_name, input_name,
        no_read_trace=not log_read,
        init_raw_log_path=init_raw_log_path if not no_trace else None,
    )
    _validate_state_no_aliasing(interp)
    var_store, memory_maps, mem_map_info = _extract_state(interp)
    if not no_trace and init_raw_log_path is not None:
        interp.env.flush_raw_log()

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
        str(raw_log_path),
        extra_data=ext_data,
        log_read=log_read,
        no_trace=no_trace,
        havoc_sequences=_extract_havoc_sequences(program_inputs),
    )

    if return_status:
        return result

    status = result.get('status', 'ok')
    if status == 'assert_violation':
        from interpreter.python.interpreter import AssertViolation
        raise AssertViolation(
            None,
            pc=result.get('violation_pc'),
            block=result.get('violation_block'),
            expr_str='<native assertion>',
        )
    if status == 'assume_violation':
        raise AssertionError(
            "concrete assume failed at "
            f"pc={result.get('violation_pc')} "
            f"block={result.get('violation_block')!r}"
        )

    return result['explored_blocks']


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------



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
        raw_log_path = trace_dir / f"{input_name}.trace.raw.zst"
        init_raw_log_path = trace_dir / f"{input_name}.init.raw.zst"

        has_trace = raw_log_path.exists() and raw_log_path.stat().st_size > 0
        if not force and explored_path.exists() and has_trace:
            print(f"Skipping input file: {input_file} (already explored)")
            with open(explored_path) as f:
                explored = {line.strip() for line in f if line.strip()}
            cov = compute_coverage(program, explored)
            if cov:
                unreachable = cov.get('unreachable_blocks', 0)
                rblk = f" (reachable: {cov['reachable_block_pct']}%)" if unreachable else ""
                print(f"[coverage] {input_name}: {cov['blocks_covered']}/{cov['blocks_total']} blocks ({cov['block_pct']}%){rblk}, "
                      f"{cov['stmts_covered']}/{cov['stmts_total']} stmts ({cov['stmt_pct']}%)")
            return (input_name, cov, explored)

        for p in (raw_log_path, init_raw_log_path):
            if p.exists() and not explored_path.exists():
                p.unlink()

        from interpreter.python.interpreter import AssertViolation
        try:
            if engine == 'python':
                explored, _ = run_python(program, program_inputs, test_name, input_name,
                                         full_trace=full_trace, no_read_trace=no_read_trace)
            elif engine == 'native':
                explored = run_native(
                    program, program_inputs, test_name, input_name,
                    raw_log_path=raw_log_path,
                    init_raw_log_path=init_raw_log_path,
                    extra_data=program_inputs.extra_data,
                    log_read=not no_read_trace,
                    compiled=compiled,
                )
            else:
                raise ValueError(f"Unknown engine: {engine}")
        except AssertViolation as v:
            # Structured line for agent_loop's subprocess parser.  The
            # parent process reads child stdout to collect per-input
            # violations; expression is JSON-quoted so embedded quotes
            # / newlines in the Boogie expr don't break the line.
            import json as _json
            print(f"[ASSERT_VIOLATION] "
                  f"input={input_name} pc={v.pc} block={v.block!r} "
                  f"expr={_json.dumps(v.expr_str)}")
            return (input_name, None, set())

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

    per_input_timeout = int(os.environ.get("SWOOSH_INTERP_TIMEOUT", "60"))
    # A value of 0 (or negative) means "run until completion, no timeout".
    if per_input_timeout <= 0:
        per_input_timeout = None
    failed = False
    skipped = 0
    results = []
    for input_file in input_files:
        input_name = os.path.basename(input_file) if isinstance(input_file, str) else str(input_file)
        p = mp_context.Process(target=worker_func, args=(input_file,))
        p.start()
        p.join(timeout=per_input_timeout)
        if p.is_alive():
            print(f"TIMEOUT ({per_input_timeout}s) on {input_name} — killing (likely infinite loop)")
            p.kill()
            p.join(timeout=5)
            skipped += 1
        elif p.exitcode != 0 and p.exitcode is not None:
            print(f"Worker failed on {input_name} (exit={p.exitcode})")
            failed = True

    if failed and not results:
        print("Interpretation did not complete — run again to resume")
        exit(1)
    if skipped:
        print(f"Skipped {skipped} non-terminating input(s)")

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
