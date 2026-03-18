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

from interpreter.runner import compute_coverage, run_python, find_entry_point
from interpreter.utils.inputs import Input, ProgramInputs, preprocess_external_inputs
from interpreter.utils.input_parser import get_bpl_field_sizes


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

    return result


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def _scalar_strategy():
    """Strategy for scalar values (integers)."""
    return st.one_of(
        st.just(0),
        st.just(1),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=65535),
        st.integers(min_value=0, max_value=(1 << 32) - 1),
    )


def _buffer_strategy(size, private=False):
    """Strategy for byte buffers of a given size."""
    return st.one_of(
        # Common patterns
        st.just(bytes(size)),                          # zeros
        st.just(bytes([0xFF] * size)),                 # ones
        st.just(bytes(range(size)) if size <= 256      # sequential
                else bytes(i % 256 for i in range(size))),
        # Random bytes
        st.binary(min_size=size, max_size=size),
    )


def _struct_strategy(fields):
    """Strategy for struct inputs."""
    field_strats = {}
    for i, field in enumerate(fields):
        key = f"field_{i}"
        if field["is_ptr"]:
            # Pointer field — generate buffer data
            field_strats[key] = _buffer_strategy(field["size"])
        else:
            # Scalar field
            field_strats[key] = _scalar_strategy()
    return st.fixed_dictionaries(field_strats)


def _build_strategies(param_info):
    """Build a dict of Hypothesis strategies from parameter metadata."""
    strats = {}
    for p in param_info:
        name = p["name"]
        if p["kind"] == "scalar":
            strats[name] = _scalar_strategy()
        elif p["kind"] == "buffer":
            size = p["buffers"][0]["size"] if p["buffers"] else 64
            strats[name] = _buffer_strategy(size, p["private"])
        elif p["kind"] == "struct":
            strats[name] = _struct_strategy(p["struct"])
    return strats


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

    def __init__(self, name, max_examples=200, target_pct=100.0, engine='python'):
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

        # Build strategies
        self.strategies = _build_strategies(self.param_info)

        # Coverage state
        self.cumulative_blocks = set()
        self.total_blocks = 0
        self.best_pct = 0.0
        self.saved_count = 0
        self.run_count = 0
        self.start_time = None

        # Compute total block count
        entry = find_entry_point(self.program)
        if entry:
            self.total_blocks = len(entry.body.blocks)

    def _run_interpreter(self, program_inputs):
        """Run the interpreter and return explored blocks."""
        try:
            input_name = f"fuzz_{self.run_count}"
            explored, _ = run_python(
                self.program, program_inputs,
                self.name, input_name,
                full_trace=False, no_read_trace=True,
            )
            return explored
        except Exception as e:
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

    def run(self):
        """Run the coverage-guided fuzzing loop."""
        self.start_time = time.time()

        print(f"[coverage-fuzz] Target: {self.name}")
        print(f"[coverage-fuzz] Parameters: {len(self.param_info)}")
        print(f"[coverage-fuzz] Total blocks: {self.total_blocks}")
        print(f"[coverage-fuzz] Max examples: {self.max_examples}")
        print(f"[coverage-fuzz] Target coverage: {self.target_pct}%")
        print()

        # Combined strategy for all parameters
        combined = st.fixed_dictionaries(self.strategies)

        # Track target reached
        target_reached = [False]

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
                self._save_input(program_inputs, new_blocks)

            pct = 100.0 * len(self.cumulative_blocks) / self.total_blocks if self.total_blocks else 0
            self.best_pct = pct

            # Guide Hypothesis toward higher coverage
            target(pct, label="block_coverage")

            if pct >= self.target_pct:
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
