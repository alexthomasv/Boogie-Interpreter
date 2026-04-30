"""In-process interpreter execution with coverage tracking."""

import signal
import tempfile
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from interpreter.utils.inputs import ProgramInputs
from interpreter.python.interpreter import hex_to_bytes


@contextmanager
def _alarm(seconds: int):
    """SIGALRM-based timeout (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutError(f"interpreter timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class EvaluationResult:
    covered: set
    status: str = "ok"
    violation_pc: Optional[int] = None
    violation_block: Optional[str] = None
    message: Optional[str] = None


class Evaluator:
    """Run the Boogie interpreter on a ProgramInputs and return block coverage."""

    def __init__(self, program, test_name: str, timeout: int = 30,
                 engine: str = "native"):
        self.program = program
        self.test_name = test_name
        self.timeout = timeout
        self.engine_requested = engine
        self.engine = engine
        self.native_fallback_reason = None
        self.runs = 0
        self.timeouts = 0
        self.errors = 0
        self.assertion_violations = 0
        self.assume_violations = 0
        self.last_result = EvaluationResult(set())
        self._entry = None
        self._compiled = None
        self._symbolic_state_cache = {}
        self._symbolic_state_cache_limit = 128
        self._work_dir = (
            Path(tempfile.gettempdir()) / "swoosh-gen-input" / test_name
        )

        if engine not in ("native", "python"):
            raise ValueError(f"Unknown coverage evaluator engine: {engine}")
        if engine == "native":
            try:
                import swoosh_interp
                self._compiled = swoosh_interp.lower(program)
            except Exception as exc:
                self.native_fallback_reason = (
                    f"{type(exc).__name__}: {exc}"
                )
                print("[coverage] Native evaluator unavailable; "
                      f"falling back to Python ({self.native_fallback_reason})")
                self.engine = "python"

    def _get_entry(self):
        if self._entry is None:
            from interpreter.runner import find_entry_point
            self._entry = find_entry_point(self.program)
        return self._entry

    def run(self, program_inputs: ProgramInputs, input_name: str = "_cov_eval") -> set:
        """Execute program_inputs and return the set of explored block names.

        Returns an empty set on timeout or any interpreter error.
        """
        return self.run_result(program_inputs, input_name).covered

    def run_result(self, program_inputs: ProgramInputs,
                   input_name: str = "_cov_eval") -> EvaluationResult:
        """Execute program_inputs and return coverage plus stop status."""
        self.runs += 1
        try:
            with _alarm(self.timeout):
                if self.engine == "native":
                    result = self._run_native_result(program_inputs, input_name)
                else:
                    result = self._run_python_result(program_inputs, input_name)
                self._account_result(result)
                self.last_result = result
                return result
        except TimeoutError:
            self.timeouts += 1
            result = EvaluationResult(set(), status="timeout")
        except Exception as exc:
            self.errors += 1
            result = EvaluationResult(
                set(),
                status="internal_error",
                message=f"{type(exc).__name__}: {exc}",
            )
        self.last_result = result
        return result

    def _account_result(self, result: EvaluationResult):
        if result.status == "assert_violation":
            self.assertion_violations += 1
        elif result.status == "assume_violation":
            self.assume_violations += 1
        elif result.status == "internal_error":
            self.errors += 1

    def _run_python(self, program_inputs: ProgramInputs,
                    input_name: str) -> set:
        return self._run_python_result(program_inputs, input_name).covered

    def _run_python_result(self, program_inputs: ProgramInputs,
                           input_name: str) -> EvaluationResult:
        from interpreter.python.Environment import Environment
        from interpreter.python.interpreter import BoogieInterpreter, AssertViolation

        env = Environment(self.test_name, input_name)
        env.full_trace = False
        env.LOG_READ = False

        entry = self._get_entry()
        if entry is None:
            return EvaluationResult(set())

        interp = BoogieInterpreter(env, program_inputs, input_name)
        interp.preprocess(self.program)
        try:
            interp.execute_procedure(entry)
            return EvaluationResult(set(interp.explored_blocks))
        except AssertViolation as exc:
            return EvaluationResult(
                set(interp.explored_blocks),
                status="assert_violation",
                violation_pc=exc.pc,
                violation_block=exc.block,
                message=exc.expr_str,
            )
        except AssertionError as exc:
            msg = str(exc)
            if "concrete assume failed" in msg:
                return EvaluationResult(
                    set(interp.explored_blocks),
                    status="assume_violation",
                    violation_pc=getattr(env, "pc", None),
                    violation_block=getattr(env, "curr_block", None),
                    message=msg,
                )
            raise
        finally:
            close = getattr(interp, "close", None)
            if close:
                close()

    def _run_native(self, program_inputs: ProgramInputs,
                    input_name: str) -> set:
        return self._run_native_result(program_inputs, input_name).covered

    def _run_native_result(self, program_inputs: ProgramInputs,
                           input_name: str) -> EvaluationResult:
        from interpreter.runner import run_native

        self._work_dir.mkdir(parents=True, exist_ok=True)
        raw_log_path = self._work_dir / f"{input_name}.trace.raw.zst"
        explored = run_native(
            self.program,
            program_inputs,
            self.test_name,
            input_name,
            raw_log_path=raw_log_path,
            extra_data=program_inputs.extra_data,
            log_read=False,
            compiled=self._compiled,
            no_trace=True,
            return_status=True,
        )
        return EvaluationResult(
            set(explored.get("explored_blocks") or []),
            status=explored.get("status", "ok"),
            violation_pc=explored.get("violation_pc"),
            violation_block=explored.get("violation_block"),
        )

    def concolic_suggest(self, program_inputs: ProgramInputs, input_name: str,
                         covered_blocks: set, *, loop_bound: int = 8,
                         max_path_depth: int = 512,
                         max_solver_queries: int = 10000,
                         solver_timeout_ms: int = 100,
                         havoc_bound: int = 8) -> tuple[list[ProgramInputs], dict]:
        """Ask the native concolic engine for branch-negation candidates."""
        if self.engine != "native" or self._compiled is None:
            return [], {"skipped": "native-unavailable"}

        try:
            from interpreter.runner import _extract_state, _init_python_state
            import swoosh_interp

            (var_store, memory_maps, mem_map_info, symbols, bindings,
             cache_hit) = self._prepare_symbolic_state(
                program_inputs, input_name, havoc_bound=havoc_bound,
                init_state=_init_python_state, extract_state=_extract_state)
            if not symbols:
                return [], {
                    "skipped": "no-symbolic-inputs",
                    "python_state_cache_hits": int(cache_hit),
                    "python_state_cache_misses": int(not cache_hit),
                }

            result = swoosh_interp.concolic_suggest(
                self._compiled,
                var_store,
                memory_maps,
                mem_map_info,
                symbols,
                extra_data=program_inputs.extra_data,
                covered_blocks=set(covered_blocks),
                loop_bound=loop_bound,
                max_path_depth=max_path_depth,
                max_solver_queries=max_solver_queries,
                solver_timeout_ms=solver_timeout_ms,
            )
            candidates = []
            for item in result.get("candidates", []):
                updates = item.get("updates", {})
                candidates.append(_apply_symbolic_updates(
                    program_inputs, bindings, updates, havoc_bound=havoc_bound))
            stats = dict(result.get("stats", {}))
            stats["python_state_cache_hits"] = int(cache_hit)
            stats["python_state_cache_misses"] = int(not cache_hit)
            return candidates, stats
        except Exception as exc:
            self.errors += 1
            return [], {"error": f"{type(exc).__name__}: {exc}"}

    def symbolic_explore(self, program_inputs: ProgramInputs, input_name: str,
                         covered_blocks: set, *, loop_bound: int = 8,
                         max_path_depth: int = 512,
                         max_solver_queries: int = 10000,
                         solver_timeout_ms: int = 100,
                         havoc_bound: int = 8,
                         max_states: int = 256) -> tuple[list[ProgramInputs], dict]:
        """Ask the native bounded symbolic worklist engine for path inputs."""
        if self.engine != "native" or self._compiled is None:
            return [], {"skipped": "native-unavailable"}

        try:
            from interpreter.runner import _extract_state, _init_python_state
            import swoosh_interp

            (var_store, memory_maps, mem_map_info, symbols, bindings,
             cache_hit) = self._prepare_symbolic_state(
                program_inputs, input_name, havoc_bound=havoc_bound,
                init_state=_init_python_state, extract_state=_extract_state)
            if not symbols:
                return [], {
                    "skipped": "no-symbolic-inputs",
                    "python_state_cache_hits": int(cache_hit),
                    "python_state_cache_misses": int(not cache_hit),
                }

            result = swoosh_interp.symbolic_explore(
                self._compiled,
                var_store,
                memory_maps,
                mem_map_info,
                symbols,
                extra_data=program_inputs.extra_data,
                covered_blocks=set(covered_blocks),
                loop_bound=loop_bound,
                max_path_depth=max_path_depth,
                max_solver_queries=max_solver_queries,
                solver_timeout_ms=solver_timeout_ms,
                max_states=max_states,
            )
            candidates = []
            for item in result.get("candidates", []):
                updates = item.get("updates", {})
                candidates.append(_apply_symbolic_updates(
                    program_inputs, bindings, updates, havoc_bound=havoc_bound))
            stats = dict(result.get("stats", {}))
            stats["python_state_cache_hits"] = int(cache_hit)
            stats["python_state_cache_misses"] = int(not cache_hit)
            return candidates, stats
        except Exception as exc:
            self.errors += 1
            return [], {"error": f"{type(exc).__name__}: {exc}"}

    def _prepare_symbolic_state(self, program_inputs: ProgramInputs,
                                input_name: str, *, havoc_bound: int,
                                init_state, extract_state):
        key = _program_inputs_cache_key(program_inputs, havoc_bound)
        cached = self._symbolic_state_cache.get(key)
        if cached is not None:
            var_store, memory_maps, mem_map_info, symbols, bindings = copy.deepcopy(cached)
            return var_store, memory_maps, mem_map_info, symbols, bindings, True

        interp, _entry = init_state(
            self.program, program_inputs, self.test_name, input_name,
            no_read_trace=True,
        )
        try:
            var_store, memory_maps, mem_map_info = extract_state(interp)
            symbols, bindings = _extract_symbolic_layout(
                interp, program_inputs, havoc_bound=havoc_bound)
        finally:
            close = getattr(interp, "close", None)
            if close:
                close()

        bundle = (var_store, memory_maps, mem_map_info, symbols, bindings)
        if len(self._symbolic_state_cache) >= self._symbolic_state_cache_limit:
            self._symbolic_state_cache.pop(next(iter(self._symbolic_state_cache)))
        self._symbolic_state_cache[key] = copy.deepcopy(bundle)
        return var_store, memory_maps, mem_map_info, symbols, bindings, False


def _int_to_u64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _program_inputs_cache_key(program_inputs: ProgramInputs, havoc_bound: int):
    variables = []
    for name, inp in sorted(program_inputs.variables.items()):
        variables.append((
            name,
            bool(inp.private),
            inp.value,
            copy.deepcopy(inp.buffers),
            copy.deepcopy(inp.struct),
            tuple(inp.havoc_seq) if inp.havoc_seq is not None else None,
        ))
    extra = (program_inputs.extra_data or b"").hex()
    return repr((int(havoc_bound), variables, extra))


def _bytes_from_contents(contents: str, size: int) -> bytearray:
    try:
        data = hex_to_bytes(contents)
    except Exception:
        data = bytearray()
    data = bytearray(data[:size])
    if len(data) < size:
        data.extend(b"\x00" * (size - len(data)))
    return data


def _input_arr_infos(interp, var_name: str):
    arr_infos = interp.arr_inputs.get(var_name, [])
    if var_name.endswith(".shadow"):
        return [a for a in arr_infos if ".shadow" in a.mem_map]
    return [a for a in arr_infos if ".shadow" not in a.mem_map]


def _input_field_infos(interp, var_name: str):
    field_infos = interp.field_inputs.get(var_name, [])
    if var_name.endswith(".shadow"):
        return [f for f in field_infos if ".shadow" in f.mem_map]
    return [f for f in field_infos if ".shadow" not in f.mem_map]


def _extract_symbolic_layout(interp, program_inputs: ProgramInputs,
                             *, havoc_bound: int) -> tuple[list[dict], dict]:
    symbols = []
    bindings = {}

    def add(kind, bits, value, **meta):
        name = f"s{len(symbols)}"
        spec = {"name": name, "kind": kind, "bits": int(bits),
                "value": _int_to_u64(value)}
        spec.update(meta)
        symbols.append(spec)
        bindings[name] = spec
        return name

    for var_name, inp in program_inputs.variables.items():
        if inp.value is not None:
            add("scalar", 64, inp.value, var=var_name)

        if inp.havoc_seq is not None:
            seq = list(inp.havoc_seq)
            n = max(havoc_bound, len(seq))
            for idx in range(n):
                value = seq[idx] if idx < len(seq) else 0
                add("havoc", 64, value, havoc_var=var_name,
                    havoc_index=idx)

        if inp.buffers:
            arr_infos = _input_arr_infos(interp, var_name)
            for buf_idx, (buf, arr_info) in enumerate(zip(inp.buffers, arr_infos)):
                size = int(buf.get("size", 0) or 0)
                data = _bytes_from_contents(buf.get("contents", "0x"), size)
                try:
                    base = interp.eval(arr_info.offset)
                except Exception:
                    continue
                for byte_idx, value in enumerate(data):
                    add("buffer_byte", 8, value, map=arr_info.mem_map,
                        addr=int(base) + byte_idx, input_var=var_name,
                        buffer_index=buf_idx, byte_index=byte_idx)

        if inp.struct:
            _add_struct_symbols(add, interp, var_name, inp)

    extra = program_inputs.extra_data or b""
    for idx, value in enumerate(extra):
        add("extra_byte", 8, value, extra_index=idx)

    return symbols, bindings


def _add_struct_symbols(add, interp, var_name: str, inp):
    field_infos = _input_field_infos(interp, var_name)
    arr_infos = _input_arr_infos(interp, var_name)
    field_idx = 0
    buffer_idx = 0
    for input_field_idx, field in enumerate(inp.struct):
        if field_idx >= len(field_infos):
            break
        if "value" in field:
            finfo = field_infos[field_idx]
            size = int(field.get("size", finfo.size) or finfo.size)
            try:
                base = interp.eval(finfo.base_ptr)
            except Exception:
                field_idx += 1
                continue
            data = _bytes_from_contents(field.get("value", "0x"), size)[::-1]
            elt_bytes = max(1, int(getattr(finfo, "size", 1) or 1))
            for byte_idx, value in enumerate(data):
                add("struct_scalar_byte", 8, value, map=finfo.mem_map,
                    addr=int(base) + byte_idx, input_var=var_name,
                    field_index=input_field_idx, byte_index=byte_idx)
            field_idx += 1
            continue

        if "buffer" in field:
            if buffer_idx < len(arr_infos):
                arr_info = arr_infos[buffer_idx]
                buf = field["buffer"]
                size = int(buf.get("size", 0) or 0)
                data = _bytes_from_contents(buf.get("contents", "0x"), size)
                try:
                    base = interp.eval(arr_info.offset)
                except Exception:
                    field_idx += 1
                    buffer_idx += 1
                    continue
                for byte_idx, value in enumerate(data):
                    add("struct_buffer_byte", 8, value, map=arr_info.mem_map,
                        addr=int(base) + byte_idx, input_var=var_name,
                        field_index=input_field_idx, byte_index=byte_idx)
            field_idx += 1
            buffer_idx += 1


def _apply_symbolic_updates(base: ProgramInputs, bindings: dict, updates: dict,
                            *, havoc_bound: int) -> ProgramInputs:
    out = copy.deepcopy(base)
    extra = bytearray(out.extra_data or b"")

    for sym_name, raw_value in updates.items():
        meta = bindings.get(sym_name)
        if not meta:
            continue
        value = int(raw_value)
        kind = meta.get("kind")

        if kind == "scalar":
            inp = out.variables.get(meta["var"])
            if inp is not None:
                inp.value = value if value < (1 << 63) else value - (1 << 64)
        elif kind == "havoc":
            inp = out.variables.get(meta["havoc_var"])
            if inp is not None:
                seq = list(inp.havoc_seq or [])
                idx = int(meta["havoc_index"])
                while len(seq) <= idx:
                    seq.append(0)
                seq[idx] = value if value < (1 << 63) else value - (1 << 64)
                if seq and seq[-1] != 0 and len(seq) < havoc_bound + 1:
                    seq.append(0)
                inp.havoc_seq = seq
        elif kind == "buffer_byte":
            inp = out.variables.get(meta["input_var"])
            if inp and inp.buffers:
                buf = inp.buffers[int(meta["buffer_index"])]
                data = _bytes_from_contents(buf.get("contents", "0x"),
                                            int(buf.get("size", 0) or 0))
                data[int(meta["byte_index"])] = value & 0xff
                buf["contents"] = "0x" + data.hex()
        elif kind == "struct_buffer_byte":
            inp = out.variables.get(meta["input_var"])
            if inp and inp.struct:
                field = inp.struct[int(meta["field_index"])]
                if "buffer" in field:
                    buf = field["buffer"]
                    data = _bytes_from_contents(buf.get("contents", "0x"),
                                                int(buf.get("size", 0) or 0))
                    data[int(meta["byte_index"])] = value & 0xff
                    buf["contents"] = "0x" + data.hex()
        elif kind == "struct_scalar_byte":
            inp = out.variables.get(meta["input_var"])
            if inp and inp.struct:
                field = inp.struct[int(meta["field_index"])]
                if "value" in field:
                    size = int(field.get("size", 0) or 0)
                    data = _bytes_from_contents(field.get("value", "0x"), size)
                    le = bytearray(data[::-1])
                    le[int(meta["byte_index"])] = value & 0xff
                    field["value"] = "0x" + bytes(le[::-1]).hex()
        elif kind == "extra_byte":
            idx = int(meta["extra_index"])
            while len(extra) <= idx:
                extra.append(0)
            extra[idx] = value & 0xff

    out.extra_data = bytes(extra) if extra else out.extra_data
    return out
