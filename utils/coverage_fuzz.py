"""Coverage-guided input generator using Hypothesis.

Generates .input files that maximize Boogie program block coverage.

Usage:
    python3 -m interpreter.utils.coverage_fuzz <name> [--max-examples N] [--target-pct P]

Example:
    python3 -m interpreter.utils.coverage_fuzz aead --max-examples 200
    python3 -m interpreter.utils.coverage_fuzz bearssl_test_pkcs1_i15 --target-pct 95.0

Architecture:
    1. Load the compiled program from test_packages/<name>_pkg/
    2. Extract parameter metadata (types, sizes, privacy) from BPL annotations
    3. Build Hypothesis strategies for each parameter (scalars, buffers, structs)
    4. Run Hypothesis with target(block_coverage) to guide exploration
    5. Save inputs that increase cumulative coverage as .input files
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

from hypothesis import given, settings, target, HealthCheck, Phase, Verbosity
from hypothesis import strategies as st

from interpreter.runner import prepare_native, run_native
from interpreter.utils.inputs import Input, ProgramInputs, preprocess_external_inputs
from interpreter.utils.input_parser import get_bpl_field_sizes
from interpreter.utils.program import find_entry_point


# ---------------------------------------------------------------------------
# Parameter metadata extraction
# ---------------------------------------------------------------------------

def _extract_param_info(program):
    """Extract parameter metadata from a compiled Boogie program.

    Returns list of dicts:
        [{"name": "$p0", "kind": "scalar"|"buffer"|"struct", "private": bool,
          "buffers": [{"size": N}], "struct": [{"name":..,"size":N,"is_ptr":bool}]}]
    """
    from interpreter.parser.declaration import (
        ImplementationDeclaration, ProcedureDeclaration,
    )
    from interpreter.parser.statement import CallStatement
    from interpreter.utils.program import RE_SMACK

    entry = find_entry_point(program)
    assert entry is not None, "No entrypoint found"

    # Find proc_decl for parameter info
    proc_decl = None
    for d in program.declarations:
        if (isinstance(d, ProcedureDeclaration)
                and not isinstance(d, ImplementationDeclaration)
                and d.name == entry.name):
            proc_decl = d
            break

    source = proc_decl if proc_decl and proc_decl.parameters else entry
    params = []
    for p in source.parameters:
        assert len(p.names) == 1
        name = p.names[0]
        if '.shadow' in name:
            continue
        params.append(name)

    # Get array and field annotations
    arr_map, field_map = preprocess_external_inputs(entry)

    # Detect private parameters from requires annotations
    private_params = set()
    if proc_decl and hasattr(proc_decl, 'requires'):
        for req in (proc_decl.requires or []):
            if hasattr(req, 'attributes'):
                for attr in req.attributes:
                    if attr.name == 'private_in':
                        # Extract param name from annotation
                        for arg in attr.arguments:
                            name_str = str(arg)
                            for pname in params:
                                if pname in name_str:
                                    private_params.add(pname)

    result = []
    for name in params:
        arr_infos = [a for a in arr_map.get(name, []) if '.shadow' not in a.mem_map]
        fld_infos = [f for f in field_map.get(name, []) if '.shadow' not in f.mem_map]
        is_private = name in private_params

        if fld_infos:
            # Struct parameter
            fields = []
            for fi in fld_infos:
                # Check if this field has array annotations (pointer field)
                has_arr = any(a.base_ptr == fi.base_ptr for a in arr_infos)
                fields.append({
                    "name": fi.var_name,
                    "size": fi.size,
                    "is_ptr": has_arr,
                })
            result.append({
                "name": name, "kind": "struct",
                "private": is_private, "struct": fields,
            })
        elif arr_infos:
            # Buffer parameter
            bufs = []
            for ai in arr_infos:
                bufs.append({"size": ai.elem_size * ai.num_elements})
            result.append({
                "name": name, "kind": "buffer",
                "private": is_private, "buffers": bufs,
            })
        else:
            # Scalar parameter
            result.append({
                "name": name, "kind": "scalar",
                "private": is_private,
            })

    # Also detect nondet constants ($u0, $u1, ...) — these are
    # uninitialized local variables in the C source that SMACK
    # promotes to global constants.  They need fuzz values for
    # coverage-guided input generation (Code2Inv / SV-COMP benchmarks).
    from interpreter.parser.declaration import ConstantDeclaration
    import re
    for d in program.declarations:
        if isinstance(d, ConstantDeclaration):
            for name in d.names:
                if re.match(r'\$u\d+$', name):
                    result.append({
                        "name": name, "kind": "scalar",
                        "private": False,
                    })

    return result


def _extract_param_info_from_seed(seed_path):
    """Extract parameter metadata from a seed .input file.

    Used for void-wrapper benchmarks where the entry point has no formal
    parameters — the input file declares local buffers/scalars to initialize.
    """
    import re
    text = Path(seed_path).read_text()
    lines = text.split('\n')

    # Parse @params line
    params_line = None
    for line in lines:
        m = re.match(r'^//\s*@params\s+(.+)$', line.strip())
        if m:
            params_line = m.group(1)
            break
    if not params_line:
        return []

    # Build ordered list of (c_name, bpl_name)
    param_pairs = []
    for pair in params_line.split():
        if ':' in pair:
            c_name, bpl_name = pair.split(':', 1)
            param_pairs.append((c_name.strip(), bpl_name.strip()))

    # Parse declarations to get types and sizes
    result = []
    private_next = False
    for line in lines:
        stripped = line.strip()
        if '@private' in stripped:
            private_next = True
            continue
        if stripped.startswith('//') or not stripped:
            continue

        # Match buffer: unsigned char name[N] = ...;
        m = re.match(r'unsigned\s+char\s+(\w+)\[(\d+)\]\s*=', stripped)
        if m:
            c_name, size = m.group(1), int(m.group(2))
            bpl_name = None
            for cn, bn in param_pairs:
                if cn == c_name:
                    bpl_name = bn
                    break
            if bpl_name:
                result.append({
                    "name": bpl_name, "kind": "buffer",
                    "private": private_next,
                    "buffers": [{"size": size}],
                })
            private_next = False
            continue

        # Match scalar: size_t name = N;
        m = re.match(r'size_t\s+(\w+)\s*=\s*(\d+)', stripped)
        if m:
            c_name = m.group(1)
            bpl_name = None
            for cn, bn in param_pairs:
                if cn == c_name:
                    bpl_name = bn
                    break
            if bpl_name:
                result.append({
                    "name": bpl_name, "kind": "scalar",
                    "private": private_next,
                })
            private_next = False
            continue

        # Match struct start: struct auto_name name = {
        m = re.match(r'struct\s+\w+\s+(\w+)\s*=\s*\{', stripped)
        if m:
            c_name = m.group(1)
            bpl_name = None
            for cn, bn in param_pairs:
                if cn == c_name:
                    bpl_name = bn
                    break
            if bpl_name:
                # For structs in seed files, mark as struct with empty fields
                # (the seed corpus loader will parse the full struct)
                result.append({
                    "name": bpl_name, "kind": "struct",
                    "private": private_next, "struct": [],
                })
            private_next = False

    return result


# ---------------------------------------------------------------------------
# Base Hypothesis strategies
# ---------------------------------------------------------------------------

def _scalar_base_strategy():
    """Strategy for scalar values — weighted toward small values."""
    return st.one_of(
        st.sampled_from([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=0xFFFF),
        st.integers(min_value=0, max_value=(1 << 32) - 1),
    )


@st.composite
def _mutate_buffer_strategy(draw, seed_buf, size):
    """Flip 1-8 random bytes in a seed buffer."""
    buf = bytearray(seed_buf)
    n_mutations = draw(st.integers(1, min(8, size)))
    for _ in range(n_mutations):
        idx = draw(st.integers(0, size - 1))
        buf[idx] = draw(st.integers(0, 255))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Input construction from Hypothesis data
# ---------------------------------------------------------------------------

def _build_program_inputs(param_info, drawn_values):
    """Convert Hypothesis-drawn values into a ProgramInputs object."""
    variables = {}
    for p in param_info:
        name = p["name"]
        val = drawn_values[name]

        if p["kind"] == "scalar":
            variables[name] = Input(name=name, private=p["private"], value=int(val))

        elif p["kind"] == "buffer":
            size = p["buffers"][0]["size"] if p["buffers"] else 64
            if isinstance(val, bytes):
                hex_str = "0x" + val.hex()
            else:
                hex_str = f"0x{'00' * size}"
            variables[name] = Input(
                name=name, private=p["private"],
                buffers=[{"contents": hex_str, "size": size}],
            )

        elif p["kind"] == "struct":
            fields = []
            for i, finfo in enumerate(p["struct"]):
                key = f"field_{i}"
                fval = val[key]
                if finfo["is_ptr"]:
                    if isinstance(fval, bytes):
                        hex_str = "0x" + fval.hex()
                    else:
                        hex_str = f"0x{'00' * finfo['size']}"
                    fields.append({
                        "name": finfo["name"],
                        "size": finfo["size"],
                        "buffer": {"contents": hex_str, "size": finfo["size"]},
                    })
                else:
                    sz = finfo["size"]
                    int_val = int(fval) & ((1 << (sz * 8)) - 1)
                    hex_val = f"0x{int_val:0{sz * 2}x}"
                    fields.append({
                        "name": finfo["name"],
                        "size": sz,
                        "value": hex_val,
                    })
            variables[name] = Input(
                name=name, private=p["private"],
                struct=fields,
            )

    return ProgramInputs(variables=variables)


# ---------------------------------------------------------------------------
# Input serialization to .input format
# ---------------------------------------------------------------------------

def _inputs_to_input_str(param_info, program_inputs):
    """Serialize ProgramInputs to C-style .input format string."""
    lines = []
    params_parts = []
    for p in param_info:
        c_name = p["name"].lstrip("$")
        params_parts.append(f"{c_name}:{p['name']}")
    lines.append(f"// @params {' '.join(params_parts)}")
    lines.append("")

    for p in param_info:
        inp = program_inputs.variables.get(p["name"])
        if inp is None:
            continue
        c_name = p["name"].lstrip("$")

        if inp.private:
            lines.append("// @private")

        if inp.struct is not None:
            lines.append(f"struct auto_{c_name} {c_name} = {{")
            for field in inp.struct:
                fname = field["name"]
                if "buffer" in field:
                    buf = field["buffer"]
                    contents = buf.get("contents", "")
                    raw = contents.replace("0x", "")
                    size = buf["size"]
                    byte_set = set(raw[i:i+2] for i in range(0, len(raw), 2)) if raw else {"00"}
                    if byte_set == {"00"}:
                        lines.append(f"    .{fname} = zeros({size}),")
                    elif byte_set == {"ff"}:
                        lines.append(f"    .{fname} = ones({size}),")
                    else:
                        lines.append(f'    .{fname} = "{contents}",')
                else:
                    val = field.get("value", "0x0")
                    try:
                        int_val = int(val, 16) if isinstance(val, str) else int(val)
                    except (ValueError, TypeError):
                        int_val = 0
                    lines.append(f"    .{fname} = {int_val},")
            lines.append("};")

        elif inp.buffers:
            buf = inp.buffers[0]
            size = buf["size"]
            contents = buf.get("contents", "")
            raw = contents.replace("0x", "")
            byte_set = set(raw[i:i+2] for i in range(0, len(raw), 2)) if raw else {"00"}
            if byte_set == {"00"}:
                expr = f"zeros({size})"
            elif byte_set == {"ff"}:
                expr = f"ones({size})"
            else:
                expr = f'"{contents}"'
            lines.append(f"unsigned char {c_name}[{size}] = {expr};")

        elif inp.value is not None:
            lines.append(f"size_t {c_name} = {inp.value};")

        else:
            lines.append(f"size_t {c_name} = 0;")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core fuzzing loop
# ---------------------------------------------------------------------------

class CoverageFuzzer:
    """Coverage-guided input generator using Hypothesis target()."""

    def __init__(self, name, max_examples=200, target_pct=100.0, engine='native'):
        if engine != "native":
            raise ValueError("coverage_fuzz is Rust-only; use engine='native'")
        self.name = name
        self.max_examples = max_examples
        self.target_pct = target_pct
        self.engine = engine

        self.pkg_path = Path("test_packages") / f"{name}_pkg"
        assert self.pkg_path.is_dir(), f"Package not found: {self.pkg_path}"

        self.output_dir = Path("test_input") / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load program
        program_pkl = self.pkg_path / f"{name}.pkl"
        with open(program_pkl, 'rb') as f:
            self.program = pickle.load(f)

        # Extract parameter metadata
        self.param_info = _extract_param_info(self.program)

        # Void-wrapper fallback: extract params from seed input file
        if not self.param_info:
            seed = self.output_dir / "input_0.input"
            if seed.exists():
                self.param_info = _extract_param_info_from_seed(seed)
                if self.param_info:
                    print(f"[coverage-fuzz] Void wrapper detected, "
                          f"extracted {len(self.param_info)} params from {seed.name}")

        # Seed corpus (populated after _seed_from_existing)
        self.seed_corpus = []

        # Coverage state
        self.cumulative_blocks = set()
        self.total_blocks = 0
        self.best_pct = 0.0
        self.saved_count = 0
        self.run_count = 0
        self._last_new_block_at = 0
        self.start_time = None

        # Pre-compile bytecode and static input metadata once.
        self._prepared = prepare_native(self.program)
        self._compiled = self._prepared.compiled

        # Compute total block count
        entry = find_entry_point(self.program)
        if entry:
            self.total_blocks = len(entry.body.blocks)

    def _run_interpreter(self, program_inputs):
        """Run the interpreter and return explored blocks."""
        try:
            input_name = f"fuzz_{self.run_count}"
            result = run_native(
                self.program, program_inputs,
                self.name, input_name,
                raw_log_path=self.output_dir / f"{input_name}.trace.raw.zst",
                log_read=False,
                compiled=self._compiled,
                prepared=self._prepared,
                no_trace=True,
                return_status=True,
                return_memory_summary=False,
                validate_handoff=False,
                quiet=True,
            )
            if result.get("status") == "ok":
                return set(result.get("explored_blocks") or [])
            return set()
        except Exception:
            # Some inputs may cause assertion failures — that's fine, skip
            return set()

    def _save_input(self, program_inputs, new_blocks):
        """Save an input that increased coverage."""
        self.saved_count += 1
        name = f"fuzz_{self.saved_count:04d}"
        path = self.output_dir / f"{name}.input"
        content = _inputs_to_input_str(self.param_info, program_inputs)
        path.write_text(content)
        pct = round(100.0 * len(self.cumulative_blocks) / self.total_blocks, 1)
        print(f"[coverage-fuzz] Saved {path.name}: +{len(new_blocks)} blocks "
              f"({len(self.cumulative_blocks)}/{self.total_blocks} = {pct}%)")

    def _seed_from_existing(self):
        """Seed cumulative coverage from existing .input files (avoid re-discovering)."""
        from interpreter.utils.input_parser import parse_input_file, get_bpl_field_sizes
        existing = sorted(self.output_dir.glob("*.input"))
        if not existing:
            return
        field_sizes = get_bpl_field_sizes(self.pkg_path, program=self.program)
        for path in existing:
            try:
                program_inputs = parse_input_file(path, field_sizes=field_sizes)
                explored = self._run_interpreter(program_inputs)
                self.cumulative_blocks |= explored
            except Exception:
                continue
        if self.cumulative_blocks:
            pct = 100.0 * len(self.cumulative_blocks) / self.total_blocks if self.total_blocks else 0
            self.best_pct = pct
            print(f"[coverage-fuzz] Seeded from {len(existing)} existing inputs: "
                  f"{len(self.cumulative_blocks)}/{self.total_blocks} ({pct:.1f}%)")

    def _load_seed_corpus(self):
        """Parse existing .input files into raw dicts for seed mutation."""
        from interpreter.utils.input_parser import parse_input_file, get_bpl_field_sizes
        corpus = []
        field_sizes = get_bpl_field_sizes(self.pkg_path, program=self.program)
        for path in sorted(self.output_dir.glob("*.input")):
            try:
                pi = parse_input_file(path, field_sizes=field_sizes)
                corpus.append(self._program_inputs_to_raw(pi))
            except Exception:
                continue
        return corpus

    def _program_inputs_to_raw(self, pi):
        """Convert ProgramInputs to raw dict matching strategy output format."""
        raw = {}
        for p in self.param_info:
            name = p["name"]
            inp = pi.variables.get(name)
            if inp is None:
                continue
            if p["kind"] == "scalar":
                raw[name] = inp.value if inp.value is not None else 0
            elif p["kind"] == "buffer":
                size = p["buffers"][0]["size"] if p["buffers"] else 64
                if inp.buffers:
                    hex_str = inp.buffers[0].get("contents", "").replace("0x", "")
                    raw[name] = bytes.fromhex(hex_str) if hex_str else bytes(size)
                else:
                    raw[name] = bytes(size)
            elif p["kind"] == "struct":
                struct_raw = {}
                if inp.struct:
                    for i, field in enumerate(inp.struct):
                        key = f"field_{i}"
                        if "buffer" in field:
                            buf = field["buffer"]
                            hex_str = buf.get("contents", "").replace("0x", "")
                            struct_raw[key] = bytes.fromhex(hex_str) if hex_str else bytes(field["size"])
                        else:
                            val = field.get("value", "0x0")
                            struct_raw[key] = int(val, 16) if isinstance(val, str) else int(val)
                raw[name] = struct_raw
        return raw

    def _build_seed_aware_strategies(self):
        """Build a composite strategy that mutates from seed corpus."""
        seed_corpus = self.seed_corpus
        param_info = self.param_info

        # Build base strategies for each parameter
        base = {}
        for p in param_info:
            name = p["name"]
            if p["kind"] == "scalar":
                base[name] = _scalar_base_strategy()
            elif p["kind"] == "buffer":
                size = p["buffers"][0]["size"] if p["buffers"] else 64
                seed_bufs = [s[name] for s in seed_corpus
                             if name in s and isinstance(s[name], bytes)
                             and len(s[name]) == size]
                options = [
                    st.just(bytes(size)),
                    st.just(bytes([0xFF] * size)),
                    st.binary(min_size=size, max_size=size),
                ]
                for sb in seed_bufs[:5]:
                    options.append(_mutate_buffer_strategy(sb, size))
                base[name] = st.one_of(*options)
            elif p["kind"] == "struct":
                field_strats = {}
                for i, field in enumerate(p["struct"]):
                    key = f"field_{i}"
                    if field["is_ptr"]:
                        size = field["size"]
                        seed_bufs = []
                        for s in seed_corpus:
                            sv = s.get(name)
                            if (isinstance(sv, dict) and key in sv
                                    and isinstance(sv[key], bytes)
                                    and len(sv[key]) == size):
                                seed_bufs.append(sv[key])
                        options = [
                            st.just(bytes(size)),
                            st.just(bytes([0xFF] * size)),
                            st.binary(min_size=size, max_size=size),
                        ]
                        for sb in seed_bufs[:5]:
                            options.append(_mutate_buffer_strategy(sb, size))
                        field_strats[key] = st.one_of(*options)
                    else:
                        field_strats[key] = _scalar_base_strategy()
                base[name] = st.fixed_dictionaries(field_strats)

        @st.composite
        def combined(draw):
            result = {}
            use_seed = seed_corpus and draw(st.booleans())
            seed = draw(st.sampled_from(seed_corpus)) if use_seed else {}
            for p in param_info:
                name = p["name"]
                seed_val = seed.get(name)
                if seed_val is not None and draw(st.integers(0, 9)) < 7:
                    result[name] = seed_val
                else:
                    result[name] = draw(base[name])
            return result

        return combined()

    def run(self):
        """Run the coverage-guided fuzzing loop."""
        self.start_time = time.time()

        # Seed from existing inputs so we only save inputs that add NEW coverage
        self._seed_from_existing()
        if self.best_pct >= self.target_pct:
            print(f"[coverage-fuzz] Already at {self.best_pct:.1f}% — target {self.target_pct}% met.")
            return self.best_pct

        # Load seed corpus for mutation-based strategies
        self.seed_corpus = self._load_seed_corpus()

        print(f"[coverage-fuzz] Target: {self.name} (engine={self.engine})")
        print(f"[coverage-fuzz] Parameters: {len(self.param_info)}")
        print(f"[coverage-fuzz] Total blocks: {self.total_blocks}")
        print(f"[coverage-fuzz] Max examples: {self.max_examples}")
        print(f"[coverage-fuzz] Target coverage: {self.target_pct}%")
        print(f"[coverage-fuzz] Seed corpus: {len(self.seed_corpus)} inputs")
        print()

        # Composite strategy: mutate from seeds + fresh generation
        combined = self._build_seed_aware_strategies()

        # Track target reached / stagnation
        target_reached = [False]
        stagnation_limit = max(self.max_examples // 3, 20)

        @settings(
            max_examples=self.max_examples,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
            phases=[Phase.generate, Phase.target],
            verbosity=Verbosity.quiet,
            database=None,  # no persistence — we save .input files directly
        )
        @given(data=combined)
        def fuzz_step(data):
            if target_reached[0]:
                return

            self.run_count += 1
            program_inputs = _build_program_inputs(self.param_info, data)
            explored = self._run_interpreter(program_inputs)

            new_blocks = explored - self.cumulative_blocks
            if new_blocks:
                self.cumulative_blocks |= new_blocks
                self._last_new_block_at = self.run_count
                self._save_input(program_inputs, new_blocks)

            pct = 100.0 * len(self.cumulative_blocks) / self.total_blocks if self.total_blocks else 0
            self.best_pct = pct

            # Guide Hypothesis toward higher coverage
            target(pct, label="block_coverage")

            if pct >= self.target_pct:
                target_reached[0] = True
            elif self.run_count - self._last_new_block_at >= stagnation_limit:
                target_reached[0] = True

        try:
            fuzz_step()
        except Exception as e:
            # Hypothesis may raise if target reached or other issues
            if "target_reached" not in str(e):
                print(f"[coverage-fuzz] Note: {e}")

        elapsed = time.time() - self.start_time
        print()
        print(f"[coverage-fuzz] Done: {self.run_count} runs in {elapsed:.1f}s")
        print(f"[coverage-fuzz] Coverage: {len(self.cumulative_blocks)}/{self.total_blocks} "
              f"= {self.best_pct:.1f}%")
        print(f"[coverage-fuzz] Saved: {self.saved_count} inputs to {self.output_dir}/")

        return self.best_pct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Coverage-guided input generator for Boogie programs"
    )
    parser.add_argument("name", help="Benchmark name (e.g. aead, bearssl_test_pkcs1_i15)")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Max Hypothesis examples to try (default: 200)")
    parser.add_argument("--target-pct", type=float, default=100.0,
                        help="Target coverage percentage (default: 100.0)")
    args = parser.parse_args()

    fuzzer = CoverageFuzzer(
        name=args.name,
        max_examples=args.max_examples,
        target_pct=args.target_pct,
    )
    fuzzer.run()


if __name__ == '__main__':
    main()
