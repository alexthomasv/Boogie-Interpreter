"""In-process interpreter execution with coverage tracking."""

import copy
import signal
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from interpreter.utils.inputs import ProgramInputs
from interpreter.utils.debug_log import DebugLogger
from interpreter.coverage_gen.symbolic_state import (
    LazySymbolicCandidate,
    apply_symbolic_updates,
    program_inputs_cache_key,
    symbolic_candidate_meta,
)


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
    block_sequence: tuple[str, ...] = ()
    violation_pc: Optional[int] = None
    violation_block: Optional[str] = None
    message: Optional[str] = None
    invalid_input: bool = False
    invalid_reason: Optional[str] = None


DEFAULT_MAX_STEPS_PER_INPUT = 100_000_000


class Evaluator:
    """Run the Boogie interpreter on a ProgramInputs and return block coverage."""

    def __init__(self, program, test_name: str, timeout: int = 30,
                 engine: str = "native",
                 max_steps_per_input: int = DEFAULT_MAX_STEPS_PER_INPUT,
                 debug_logger=None):
        self.program = program
        self.test_name = test_name
        self.debug = (debug_logger or DebugLogger.disabled()).bind(
            component="coverage-evaluator", test_name=test_name)
        self.timeout = timeout
        self.max_steps_per_input = max(0, int(max_steps_per_input or 0))
        self.engine_requested = engine
        self.engine = engine
        self.native_fallback_reason = None
        self.runs = 0
        self.timeouts = 0
        self.errors = 0
        self.step_limits = 0
        self.assertion_violations = 0
        self.assume_violations = 0
        self.last_result = EvaluationResult(set())
        self._entry = None
        self._compiled = None
        self._prepared = None
        self._symbolic_state_cache = {}
        self._symbolic_state_cache_limit = 128
        self._work_dir = (
            Path(tempfile.gettempdir()) / "swoosh-gen-input" / test_name
        )

        if engine != "native":
            raise ValueError(
                "Coverage evaluator is Rust-only; the Python engine is archived."
            )
        from interpreter.runner import prepare_native

        self._prepared = prepare_native(program)
        self._compiled = self._prepared.compiled

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
        self.debug.event("candidate", "evaluation_start",
                         input_name=input_name, engine=self.engine,
                         run_index=self.runs,
                         timeout=self.timeout,
                         max_steps=self.max_steps_per_input)
        try:
            with _alarm(self.timeout):
                result = self._run_native_result(program_inputs, input_name)
                self._account_result(result)
                self.last_result = result
                self.debug.event(
                    "candidate",
                    "evaluation_end",
                    input_name=input_name,
                    status=result.status,
                    covered=len(result.covered),
                    violation_pc=result.violation_pc,
                    violation_block=result.violation_block,
                    message=result.message,
                )
                return result
        except TimeoutError:
            self.timeouts += 1
            result = EvaluationResult(set(), status="timeout")
            self.debug.event("candidate", "evaluation_timeout",
                             input_name=input_name, timeout=self.timeout)
        except Exception as exc:
            self.errors += 1
            result = EvaluationResult(
                set(),
                status="internal_error",
                message=f"{type(exc).__name__}: {exc}",
            )
            self.debug.event("candidate", "evaluation_error",
                             input_name=input_name, error=result.message)
        self.last_result = result
        return result

    def _account_result(self, result: EvaluationResult):
        if result.status == "assert_violation":
            self.assertion_violations += 1
        elif result.status == "assume_violation":
            self.assume_violations += 1
        elif result.status == "step_limit":
            self.step_limits += 1
        elif result.status == "internal_error":
            self.errors += 1

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
            prepared=self._prepared,
            no_trace=True,
            return_status=True,
            return_memory_summary=False,
            validate_handoff=False,
            quiet=True,
            max_steps=self.max_steps_per_input,
            debug_logger=self.debug.bind(input_name=input_name, engine="native"),
        )
        return EvaluationResult(
            set(explored.get("explored_blocks") or []),
            status=explored.get("status", "ok"),
            block_sequence=tuple(explored.get("block_sequence") or ()),
            violation_pc=explored.get("violation_pc"),
            violation_block=explored.get("violation_block"),
            message=explored.get("message"),
            invalid_input=bool(explored.get("invalid_input", False)),
            invalid_reason=explored.get("invalid_reason"),
        )

    def concolic_suggest(self, program_inputs: ProgramInputs, input_name: str,
                         covered_blocks: set, *, loop_bound: int = 8,
                         max_path_depth: int = 512,
                         max_solver_queries: int = 10000,
                         solver_timeout_ms: int = 100,
                         havoc_bound: int = 8,
                         branch_distance_policy: str = "auto") -> tuple[list[ProgramInputs], dict]:
        import swoosh_interp

        state, state_stats = self._get_symbolic_state(program_inputs, havoc_bound)
        raw = swoosh_interp.concolic_suggest(
            self._compiled,
            state["var_store"],
            state["memory_maps"],
            state["mem_map_info"],
            state["symbols"],
            extra_data=program_inputs.extra_data,
            covered_blocks=list(covered_blocks),
            loop_bound=loop_bound,
            max_path_depth=max_path_depth,
            max_solver_queries=max_solver_queries,
            solver_timeout_ms=solver_timeout_ms,
            branch_distance_policy=branch_distance_policy,
        )
        return self._materialize_symbolic_candidates(
            program_inputs,
            state,
            raw,
            havoc_bound=havoc_bound,
            state_stats=state_stats,
        )

    def symbolic_explore(self, program_inputs: ProgramInputs, input_name: str,
                         covered_blocks: set, *, loop_bound: int = 8,
                         max_path_depth: int = 512,
                         max_solver_queries: int = 10000,
                         solver_timeout_ms: int = 100,
                         havoc_bound: int = 8,
                         max_states: int = 256,
                         path_priority: str = "coverage-directed",
                         value_profile_policy: str = "late",
                         branch_distance_policy: str = "auto") -> tuple[list[ProgramInputs], dict]:
        import swoosh_interp

        state, state_stats = self._get_symbolic_state(program_inputs, havoc_bound)
        raw = swoosh_interp.symbolic_explore(
            self._compiled,
            state["var_store"],
            state["memory_maps"],
            state["mem_map_info"],
            state["symbols"],
            extra_data=program_inputs.extra_data,
            covered_blocks=list(covered_blocks),
            loop_bound=loop_bound,
            max_path_depth=max_path_depth,
            max_solver_queries=max_solver_queries,
            solver_timeout_ms=solver_timeout_ms,
            max_states=max_states,
            path_priority=path_priority,
            value_profile_policy=value_profile_policy,
            branch_distance_policy=branch_distance_policy,
        )
        return self._materialize_symbolic_candidates(
            program_inputs,
            state,
            raw,
            havoc_bound=havoc_bound,
            state_stats=state_stats,
        )

    def symbolic_explore_lazy(self, program_inputs: ProgramInputs, input_name: str,
                              covered_blocks: set, *, loop_bound: int = 8,
                              max_path_depth: int = 512,
                              max_solver_queries: int = 10000,
                              solver_timeout_ms: int = 100,
                              havoc_bound: int = 8,
                              max_states: int = 256,
                              path_priority: str = "coverage-directed",
                              value_profile_policy: str = "late",
                              branch_distance_policy: str = "auto") -> tuple[list, dict]:
        import swoosh_interp

        state, state_stats = self._get_symbolic_state(program_inputs, havoc_bound)
        raw = swoosh_interp.symbolic_explore(
            self._compiled,
            state["var_store"],
            state["memory_maps"],
            state["mem_map_info"],
            state["symbols"],
            extra_data=program_inputs.extra_data,
            covered_blocks=list(covered_blocks),
            loop_bound=loop_bound,
            max_path_depth=max_path_depth,
            max_solver_queries=max_solver_queries,
            solver_timeout_ms=solver_timeout_ms,
            max_states=max_states,
            path_priority=path_priority,
            value_profile_policy=value_profile_policy,
            branch_distance_policy=branch_distance_policy,
        )
        return self._lazy_symbolic_candidates(
            program_inputs,
            state,
            raw,
            havoc_bound=havoc_bound,
            state_stats=state_stats,
        )

    def _get_symbolic_state(self, program_inputs: ProgramInputs,
                            havoc_bound: int) -> tuple[dict, dict]:
        import swoosh_interp

        if not hasattr(swoosh_interp, "prepare_symbolic_inputs"):
            raise RuntimeError(
                "Installed Rust native module is too old: "
                "prepare_symbolic_inputs is missing. Rebuild with "
                "python -m maturin develop --release."
            )

        key = program_inputs_cache_key(program_inputs, havoc_bound)
        cached = self._symbolic_state_cache.get(key)
        if cached is not None:
            state = copy.deepcopy(cached)
            return state, {
                "symbolic_state_cache_hit": True,
                "symbolic_state_ms": state.get("state_ms", 0.0),
                "symbol_count": state.get("symbol_count", len(state.get("symbols", []))),
            }

        state = dict(swoosh_interp.prepare_symbolic_inputs(
            self._compiled,
            self._prepared.native_meta,
            program_inputs,
            extra_data=program_inputs.extra_data,
            havoc_bound=havoc_bound,
        ))
        if len(self._symbolic_state_cache) >= self._symbolic_state_cache_limit:
            self._symbolic_state_cache.pop(next(iter(self._symbolic_state_cache)))
        self._symbolic_state_cache[key] = copy.deepcopy(state)
        return state, {
            "symbolic_state_cache_hit": False,
            "symbolic_state_ms": state.get("state_ms", 0.0),
            "symbol_count": state.get("symbol_count", len(state.get("symbols", []))),
        }

    def _materialize_symbolic_candidates(self, program_inputs: ProgramInputs,
                                         state: dict, raw: dict, *,
                                         havoc_bound: int,
                                         state_stats: dict) -> tuple[list[ProgramInputs], dict]:
        raw_candidates = list(raw.get("candidates") or [])
        bindings = state.get("bindings") or {}
        candidates = []
        for raw_idx, candidate in enumerate(raw_candidates):
            meta = symbolic_candidate_meta(raw_idx, candidate)
            updates = dict(meta.get("updates") or {})
            if not updates:
                continue
            materialized = apply_symbolic_updates(
                program_inputs,
                bindings,
                updates,
                havoc_bound=havoc_bound,
            )
            materialized._swoosh_candidate_meta = meta
            materialized._swoosh_symbolic_bindings = bindings
            materialized._swoosh_havoc_bound = havoc_bound
            candidates.append(materialized)
        stats = dict(raw.get("stats") or {})
        stats.update(state_stats)
        stats["raw_candidates"] = len(raw_candidates)
        stats["candidate_count"] = len(candidates)
        return candidates, stats

    def _lazy_symbolic_candidates(self, program_inputs: ProgramInputs,
                                  state: dict, raw: dict, *,
                                  havoc_bound: int,
                                  state_stats: dict) -> tuple[list, dict]:
        raw_candidates = list(raw.get("candidates") or [])
        bindings = state.get("bindings") or {}
        candidates = []
        for raw_idx, candidate in enumerate(raw_candidates):
            meta = symbolic_candidate_meta(raw_idx, candidate)
            if not meta.get("updates"):
                continue
            candidates.append(LazySymbolicCandidate(
                program_inputs,
                bindings,
                havoc_bound,
                meta,
            ))
        stats = dict(raw.get("stats") or {})
        stats.update(state_stats)
        stats["raw_candidates"] = len(raw_candidates)
        stats["candidate_count"] = len(candidates)
        stats["lazy_candidate_count"] = len(candidates)
        return candidates, stats
