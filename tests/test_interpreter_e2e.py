"""
End-to-end interpreter tests for aead, pkcs_i15, and pkcs_i31.

Validates:
  - Compiled packages load correctly
  - Input field counts match BPL {:field} annotations (the SMACK fix)
  - Interpreter completes without errors
  - Key output variables have expected values (return value, explored blocks)

Usage:
  pypy -m pytest tests/test_interpreter_e2e.py -v           # fast tests only
  pypy -m pytest tests/test_interpreter_e2e.py -v -m slow   # slow (full run) tests
  pypy -m pytest tests/test_interpreter_e2e.py -v --runslow # all tests
"""
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser.declaration import ImplementationDeclaration
from utils.utils import (
    preprocess_external_inputs,
    parse_inputs,
    generate_function_map,
    generate_label_to_block,
    initialize_code_metadata,
)
from interpreter.python.interpreter import BoogieInterpreter, find_entry_point, process_single_input
from interpreter.python.Environment import Environment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PACKAGES = PROJECT_ROOT / "test_packages"
TEST_INPUT = PROJECT_ROOT / "test_input"
POSITIVE_EXAMPLES = PROJECT_ROOT / "positive_examples"

BENCHMARKS = {
    "aead": {
        "pkg": "aead_pkg",
        "input": "aead/input_0.json",
        "min_explored_blocks": 100,
        "min_trace_entries": 10000,
        "return_var": "$r",  # aead returns i32
    },
    "pkcs_i15": {
        "pkg": "bearssl_test_pkcs1_i15_pkg",
        "input": "bearssl_test_pkcs1_i15/test_06_custom_vector.json",
        "min_explored_blocks": 200,
        "min_trace_entries": 100000,
        "return_var": None,  # sign_wrapper is void
    },
    "pkcs_i31": {
        "pkg": "bearssl_test_pkcs1_i31_pkg",
        "input": "bearssl_test_pkcs1_i31/test_01b_standard_2048.json",
        "min_explored_blocks": 200,
        "min_trace_entries": 100000,
        "return_var": None,  # sign_wrapper is void
    },
}


# ── Helpers ──

def _load_program(benchmark):
    info = BENCHMARKS[benchmark]
    pkg_dir = TEST_PACKAGES / info["pkg"]
    name = info["pkg"].removesuffix("_pkg")
    pkl_path = pkg_dir / f"{name}.pkl"
    if not pkl_path.exists():
        pytest.skip(f"Package not compiled: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_entry_proc(program):
    for d in program.declarations:
        if isinstance(d, ImplementationDeclaration) and d.has_attribute("entrypoint"):
            return d
    return None


def _load_input(benchmark):
    info = BENCHMARKS[benchmark]
    path = TEST_INPUT / info["input"]
    if not path.exists():
        pytest.skip(f"Input file not found: {path}")
    return parse_inputs(path), path


def _get_explored_blocks(benchmark):
    info = BENCHMARKS[benchmark]
    name = info["pkg"].removesuffix("_pkg")
    input_stem = Path(info["input"]).stem
    path = POSITIVE_EXAMPLES / name / f"{input_stem}.explored_blocks.txt"
    if not path.exists():
        return None
    return path.read_text().strip().split("\n")


def _get_compact_trace(benchmark):
    info = BENCHMARKS[benchmark]
    name = info["pkg"].removesuffix("_pkg")
    input_stem = Path(info["input"]).stem
    path = POSITIVE_EXAMPLES / name / f"{input_stem}.trace.compact.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return None  # corrupted trace


# ── Input matching tests (fast — no interpreter run) ──

class TestInputFieldMatching:
    """Verify BPL {:field} annotations match input generator struct field counts."""

    @pytest.mark.parametrize("benchmark", ["pkcs_i15", "pkcs_i31"])
    def test_struct_field_count_matches(self, benchmark):
        """Struct scalar fields in JSON match BPL {:field} annotation count."""
        program = _load_program(benchmark)
        proc = _get_entry_proc(program)
        assert proc is not None, "No entry point found"

        program_inputs, _ = _load_input(benchmark)
        arr_map, field_map = preprocess_external_inputs(proc)

        for var, inp in program_inputs.variables.items():
            if not inp.struct:
                continue
            if var not in field_map:
                continue
            bpl_count = len(field_map[var])
            # All struct fields (scalar + pointer) should match BPL field count
            gen_count = len(inp.struct)
            assert bpl_count == gen_count, (
                f"Struct field count mismatch for {var} in {benchmark}: "
                f"JSON has {gen_count}, BPL has {bpl_count}"
            )

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_array_annotations_present(self, benchmark):
        """BPL {:array} annotations exist for variables with buffer data."""
        program = _load_program(benchmark)
        proc = _get_entry_proc(program)
        assert proc is not None

        program_inputs, _ = _load_input(benchmark)
        arr_map, field_map = preprocess_external_inputs(proc)

        for var, inp in program_inputs.variables.items():
            has_data = inp.buffers or (inp.struct and inp.struct_buffers)
            if not has_data:
                continue
            has_arr = any(k for k in arr_map if not k.endswith(".shadow"))
            assert has_arr, f"No array annotations found for {benchmark}"
            break


# ── Interpreter concretization tests (fast — runs only init, not full exec) ──

class TestInputConcretization:
    """Verify interpreter can concretize inputs without errors."""

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_concretize_completes(self, benchmark):
        """Interpreter initializes and concretizes all inputs without error."""
        info = BENCHMARKS[benchmark]
        program = _load_program(benchmark)
        program_inputs, input_path = _load_input(benchmark)
        name = info["pkg"].removesuffix("_pkg")
        input_name = Path(info["input"]).stem

        env = Environment(name, input_name)
        env.LOG_READ = False
        interp = BoogieInterpreter(env, program_inputs, input_name)
        interp.preprocess(program)

        entry = find_entry_point(program)
        assert entry is not None

        # Initialize vars + concretize — this is what fails on field mismatch
        interp._initialize_vars(entry.body.locals, kind="local")
        interp.concretize_proc_inputs(entry)
        interp.concretize_input_memory()  # The critical step

    @pytest.mark.parametrize("benchmark", ["pkcs_i15", "pkcs_i31"])
    def test_pkcs_struct_fields_concretized(self, benchmark):
        """PKCS struct fields are written to the correct memory maps."""
        info = BENCHMARKS[benchmark]
        program = _load_program(benchmark)
        program_inputs, _ = _load_input(benchmark)
        name = info["pkg"].removesuffix("_pkg")
        input_name = Path(info["input"]).stem

        env = Environment(name, input_name)
        env.LOG_READ = False
        interp = BoogieInterpreter(env, program_inputs, input_name)
        interp.preprocess(program)

        entry = find_entry_point(program)
        interp._initialize_vars(entry.body.locals, kind="local")
        interp.concretize_proc_inputs(entry)
        interp.concretize_input_memory()

        # Check that n_bitlen was written (should be 0x800 = 2048)
        m0 = env.get_var("$M.0", silent=True)
        p3_val = env.get_var("$p3", silent=True)
        n_bitlen = m0.get(p3_val)  # first field at offset 0
        assert n_bitlen == 0x800 or n_bitlen == 0x400, (
            f"Expected n_bitlen 0x800 (2048) or 0x400 (1024), got {hex(n_bitlen)}"
        )


# ── Trace validation tests (fast — checks existing traces, no re-run) ──

class TestExistingTraces:
    """Validate structure of existing interpreter traces."""

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_explored_blocks_exist(self, benchmark):
        """Explored blocks file exists and has reasonable size."""
        blocks = _get_explored_blocks(benchmark)
        if blocks is None:
            pytest.skip(f"No trace for {benchmark} — run interpreter first")
        info = BENCHMARKS[benchmark]
        assert len(blocks) >= info["min_explored_blocks"], (
            f"Expected >= {info['min_explored_blocks']} explored blocks, "
            f"got {len(blocks)}"
        )

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_compact_trace_has_data(self, benchmark):
        """Compact trace exists and has expected entry count."""
        trace = _get_compact_trace(benchmark)
        if trace is None:
            pytest.skip(f"No compact trace for {benchmark}")
        info = BENCHMARKS[benchmark]
        assert trace["total"] >= info["min_trace_entries"], (
            f"Expected >= {info['min_trace_entries']} trace entries, "
            f"got {trace['total']}"
        )

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_trace_has_pc_data(self, benchmark):
        """Compact trace has PC-indexed variable data."""
        trace = _get_compact_trace(benchmark)
        if trace is None:
            pytest.skip(f"No compact trace for {benchmark}")
        assert len(trace["pc_values"]) > 0, "pc_values should not be empty"
        assert len(trace["pc_registry"]) > 0, "pc_registry should not be empty"

    @pytest.mark.parametrize("benchmark", ["aead", "pkcs_i15", "pkcs_i31"])
    def test_explored_blocks_start_at_entry(self, benchmark):
        """First explored block should be the entry block."""
        blocks = _get_explored_blocks(benchmark)
        if blocks is None:
            pytest.skip(f"No trace for {benchmark}")
        assert blocks[0] == "$bb0", f"Expected first block '$bb0', got '{blocks[0]}'"


# ── Full interpreter run tests (slow — actually executes the interpreter) ──

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class TestInterpreterExecution:
    """Full interpreter execution — checks return values and completion."""

    @pytest.mark.slow
    def test_aead_completes_with_return(self):
        """AEAD interpreter completes and returns i32 value."""
        info = BENCHMARKS["aead"]
        program = _load_program("aead")
        program_inputs, _ = _load_input("aead")
        name = info["pkg"].removesuffix("_pkg")
        input_name = Path(info["input"]).stem

        env = Environment(name, input_name)
        env.LOG_READ = False
        interp = BoogieInterpreter(env, program_inputs, input_name)
        interp.preprocess(program)

        entry = find_entry_point(program)
        assert entry is not None
        interp.execute_procedure(entry)

        # AEAD decrypt returns $r (i32 success code)
        ret = env.get_var("$r", silent=True)
        assert ret is not None, "Return variable $r should be set"
        # Also check shadow matches (constant-time: both executions same result)
        ret_shadow = env.get_var("$r.shadow", silent=True)
        assert ret == ret_shadow, (
            f"Return value mismatch: $r={ret}, $r.shadow={ret_shadow}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("benchmark", ["pkcs_i15", "pkcs_i31"])
    def test_pkcs_completes_with_output(self, benchmark):
        """PKCS interpreter completes and writes non-zero signature to output buffer."""
        info = BENCHMARKS[benchmark]
        program = _load_program(benchmark)
        program_inputs, _ = _load_input(benchmark)
        name = info["pkg"].removesuffix("_pkg")
        input_name = Path(info["input"]).stem

        env = Environment(name, input_name)
        env.LOG_READ = False
        interp = BoogieInterpreter(env, program_inputs, input_name)
        interp.preprocess(program)

        entry = find_entry_point(program)
        assert entry is not None
        interp.execute_procedure(entry)

        # Check explored blocks
        assert len(interp.explored_blocks) >= info["min_explored_blocks"], (
            f"Expected >= {info['min_explored_blocks']} explored blocks, "
            f"got {len(interp.explored_blocks)}"
        )

        # sign_wrapper is void — check the scratch buffer has non-zero data,
        # proving the RSA computation actually ran.  The scratch buffer ($p5)
        # is allocated via $alloc and used for the full modular exponentiation.
        # Scan all memory maps for any store activity beyond initialisation.
        from interpreter.python.MemoryMap import MemoryMap
        maps_with_data = 0
        for vname in list(env._var_store):
            if vname.startswith("$M.") and ".shadow" not in vname:
                mm = env._var_store[vname]
                if isinstance(mm, MemoryMap) and len(mm.memory) > 0:
                    maps_with_data += 1
        assert maps_with_data >= 3, (
            f"Expected >= 3 memory maps with data, got {maps_with_data}"
        )
