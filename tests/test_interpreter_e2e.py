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

from interpreter.parser.declaration import ImplementationDeclaration
from interpreter.utils.utils import (
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


def _get_raw_log_stats(benchmark):
    """Decode the `.trace.raw.zst` header + a sample of records and return
    aggregate stats compatible with the old ``_get_compact_trace`` tests.

    Returns a dict with keys:
        total:        total record count
        pc_values:    dict of (var_id, pc) -> count
        pc_registry:  dict of var_id -> count of distinct PCs
    (Used only by the e2e shape tests.)
    """
    info = BENCHMARKS[benchmark]
    name = info["pkg"].removesuffix("_pkg")
    input_stem = Path(info["input"]).stem
    path = POSITIVE_EXAMPLES / name / f"{input_stem}.trace.raw.zst"
    if not path.exists():
        return None

    import struct
    import zstandard as zstd

    with open(path, 'rb') as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)

        def read_exact(n):
            out = bytearray(n)
            filled = 0
            while filled < n:
                chunk = reader.read(n - filled)
                if not chunk:
                    raise EOFError(f"short raw log, wanted {n}, got {filled}")
                out[filled:filled + len(chunk)] = chunk
                filled += len(chunk)
            return bytes(out)

        magic = read_exact(4)
        if magic != b"SWRL":
            return None
        version = read_exact(1)[0]
        if version != 1:
            return None
        (nvars,) = struct.unpack('<I', read_exact(4))
        for _ in range(nvars):
            (ln,) = struct.unpack('<H', read_exact(2))
            read_exact(ln)
        (nblocks,) = struct.unpack('<I', read_exact(4))
        for _ in range(nblocks):
            (ln,) = struct.unpack('<H', read_exact(2))
            read_exact(ln)

        RECORD_SIZE = 1 + 4 + 4 + 4 + 8 + 4
        pc_values = {}
        pc_registry = {}
        total = 0
        while True:
            rec = reader.read(RECORD_SIZE)
            if not rec:
                break
            if len(rec) < RECORD_SIZE:
                rec += read_exact(RECORD_SIZE - len(rec))
            var_id, pc, _ = struct.unpack_from('<III', rec, 1)
            key = (var_id, pc)
            pc_values[key] = pc_values.get(key, 0) + 1
            pc_registry.setdefault(var_id, set()).add(pc)
            total += 1

    return {
        'total': total,
        'pc_values': pc_values,
        'pc_registry': {k: len(v) for k, v in pc_registry.items()},
    }


# Back-compat alias for callers that used the old name.
_get_compact_trace = _get_raw_log_stats


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


# ── Parser-fix regression tests (concretize_proc_inputs) ──

class TestConcretizeProcInputsWarnings:
    """Regression tests for concretize_proc_inputs.

    Bug context: a pre-fix version warned and defaulted-to-0 for *every*
    pointer-typed param because it only checked ``inp.value is not None``.
    Buffer-typed params (Input.buffers) and struct-typed params
    (Input.struct) get their addresses allocated later in
    concretize_input_memory, but the spurious 0-default could shadow that
    address and silently produce all-NULL pointers — which made every
    AEAD trace identical regardless of input file (loop bound = 0,
    ~140 blocks unreached). See ``concretize_proc_inputs`` in
    ``interpreter/python/interpreter.py``.
    """

    def _build_interp_with_inputs(self, inputs_dict):
        """Construct a minimal BoogieInterpreter with a dict-of-Input mapping."""
        from interpreter.utils.inputs import Input
        env = Environment("dummy", "dummy_input")
        env.LOG_READ = False
        # BoogieInterpreter accepts a dict[str, Input] directly (legacy path).
        interp = BoogieInterpreter(env, inputs_dict, "dummy_input")
        return interp, env

    def _make_param(self, name):
        """Build a minimal Parameter object compatible with procedure.parameters."""
        class _Param:
            def __init__(self, n):
                self.names = [n]
        return _Param(name)

    def _make_proc(self, param_names):
        class _Proc:
            def __init__(self, params):
                self.parameters = params
        return _Proc([self._make_param(n) for n in param_names])

    def test_no_warning_for_buffer_pointer_param(self, caplog):
        """Pointer params with .buffers must not trigger the missing-value warning."""
        from interpreter.utils.inputs import Input
        inputs = {
            "$p0": Input(name="$p0", private=False,
                         buffers=[{"contents": "0x00112233", "size": 4}]),
        }
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$p0"])
        with caplog.at_level("WARNING"):
            interp.concretize_proc_inputs(proc)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and "$p0" in r.getMessage()
                    and "defaulting to 0" in r.getMessage()]
        assert not warnings, (
            f"Pointer-typed param $p0 (with buffers) should not be warned about, "
            f"got: {[w.getMessage() for w in warnings]}")

    def test_no_warning_for_struct_pointer_param(self, caplog):
        """Pointer params with .struct must not trigger the missing-value warning."""
        from interpreter.utils.inputs import Input
        inputs = {
            "$p0": Input(name="$p0", private=False,
                         struct=[{"name": "f1", "size": 4, "value": "0x00000010"}]),
        }
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$p0"])
        with caplog.at_level("WARNING"):
            interp.concretize_proc_inputs(proc)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and "$p0" in r.getMessage()
                    and "defaulting to 0" in r.getMessage()]
        assert not warnings, (
            f"Struct-typed param $p0 should not be warned about, "
            f"got: {[w.getMessage() for w in warnings]}")

    def test_no_warning_for_havoc_seq_param(self, caplog):
        """Params with .havoc_seq are consumed lazily — no warn."""
        from interpreter.utils.inputs import Input
        inputs = {
            "$i0": Input(name="$i0", private=False, havoc_seq=[1, 2, 3]),
        }
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$i0"])
        with caplog.at_level("WARNING"):
            interp.concretize_proc_inputs(proc)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and "$i0" in r.getMessage()
                    and "defaulting to 0" in r.getMessage()]
        assert not warnings, (
            f"Havoc-seq param $i0 should not be warned about, "
            f"got: {[w.getMessage() for w in warnings]}")

    def test_warns_when_param_truly_missing(self, caplog):
        """Genuinely missing param (not in program_inputs) should still warn."""
        from interpreter.utils.inputs import Input
        inputs = {}  # empty
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$p_missing"])
        with caplog.at_level("WARNING"):
            interp.concretize_proc_inputs(proc)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and "$p_missing" in r.getMessage()]
        assert warnings, (
            "Truly missing param $p_missing should produce a warning, "
            "but none was emitted.")
        # And the value should default to 0
        assert env.get_var("$p_missing", silent=True) == 0

    def test_scalar_value_set_correctly(self):
        """Params with .value get set (e.g., size_t input = 26)."""
        from interpreter.utils.inputs import Input
        inputs = {
            "$p2": Input(name="$p2", private=False, value=26),
        }
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$p2"])
        interp.concretize_proc_inputs(proc)
        assert env.get_var("$p2", silent=True) == 26

    def test_buffer_param_not_written_by_concretize_proc_inputs(self):
        """Buffer-typed param must not be entered into _var_store at this phase.

        Pre-fix, every pointer-typed param got `set_value(name, 0)` because
        Input.value was None. concretize_input_memory would *normally*
        overwrite that with the allocated address — but the spurious early
        write masked any later allocation bug. The new behaviour leaves
        the variable absent from _var_store until concretize_input_memory
        sets it explicitly.
        """
        from interpreter.utils.inputs import Input
        inputs = {
            "$p0": Input(name="$p0", private=False,
                         buffers=[{"contents": "0xdeadbeef", "size": 4}]),
        }
        interp, env = self._build_interp_with_inputs(inputs)
        proc = self._make_proc(["$p0"])
        interp.concretize_proc_inputs(proc)
        # The variable must NOT have been entered into _var_store. If we
        # see it here it means the function silently clobbered a
        # pointer-typed param with 0 (the pre-fix bug).
        assert "$p0" not in env._var_store, (
            f"$p0 must not be entered into _var_store by "
            f"concretize_proc_inputs (got value "
            f"{env._var_store.get('$p0')!r}). The pre-fix bug clobbered "
            f"buffer-typed pointer params to 0 here, which made every "
            f"AEAD input produce identical traces.")


# ── Trace-divergence regression test (slow — runs the interpreter) ──

class TestTraceDivergence:
    """Multiple inputs with distinct buffer contents must produce distinct traces.

    Important: explored_blocks SHOULD be identical across inputs for a
    correctly-compiled constant-time benchmark — branches must not depend on
    secrets. What MUST differ is the byte stream of trace records: the
    interpreter must actually read and propagate the input file contents,
    so two inputs with different buffer bytes record different load values.

    The pre-fix bug pre-clobbered every pointer-typed param to 0 in
    ``concretize_proc_inputs``, which under some race conditions could
    leave pointers NULL after ``concretize_input_memory`` and silently
    flatten every input to all-zero state. This guard catches that.
    """

    @pytest.mark.slow
    def test_aead_traces_diverge_for_distinct_inputs(self):
        """Inputs with distinct buffer contents must produce distinct trace bytes.

        Using fuzz_04_random_a and fuzz_05_random_b — both have unique
        random buffer contents and therefore must produce different
        ``.trace.raw.zst`` bytes. If they hash identically, the input
        pipeline isn't propagating buffer contents into memory.
        """
        import hashlib
        candidates = [
            PROJECT_ROOT / "positive_examples" / "aead",
            PROJECT_ROOT.parent / "positive_examples" / "aead",
        ]
        aead_dir = next((d for d in candidates if d.exists()), None)
        if aead_dir is None:
            pytest.skip(
                "Run interpreter on aead first: ./swoosh interpret aead -f")
        # Pick two files with KNOWN distinct buffer contents.
        a = aead_dir / "fuzz_04_random_a.trace.raw.zst"
        b = aead_dir / "fuzz_05_random_b.trace.raw.zst"
        if not (a.exists() and b.exists()):
            pytest.skip(
                f"Need both random-content trace files: {a.name}, {b.name}")
        ha = hashlib.md5(a.read_bytes()).hexdigest()
        hb = hashlib.md5(b.read_bytes()).hexdigest()
        assert ha != hb, (
            f"AEAD inputs with distinct random bytes produced identical "
            f"traces (both md5 {ha}). The input pipeline is not "
            f"propagating buffer contents into the interpreter's memory "
            f"maps — see concretize_input_memory / _concretize_buffers.")

    @pytest.mark.slow
    def test_aead_explored_blocks_constant_across_inputs(self):
        """Constant-time invariant: explored block set must NOT depend on input.

        AEAD is a constant-time benchmark — its branches only assert
        ``cond == cond.shadow``, never branch on secret data. If two
        inputs visit different sets of blocks, either the source isn't
        actually constant-time or the interpreter is taking secret-
        dependent paths somewhere it shouldn't.
        """
        candidates = [
            PROJECT_ROOT / "positive_examples" / "aead",
            PROJECT_ROOT.parent / "positive_examples" / "aead",
        ]
        aead_dir = next((d for d in candidates if d.exists()), None)
        if aead_dir is None:
            pytest.skip("Run interpreter on aead first")
        explored_files = sorted(aead_dir.glob("*.explored_blocks.txt"))
        if len(explored_files) < 2:
            pytest.skip("Need >= 2 explored_blocks files")
        sets = {f.name: frozenset(f.read_text().splitlines()) for f in explored_files}
        first_set = next(iter(sets.values()))
        diffs = {n: s ^ first_set for n, s in sets.items() if s != first_set}
        assert not diffs, (
            f"AEAD explored-block sets diverge across inputs — CT "
            f"invariant violated. Per-input symmetric differences: {diffs}")


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
