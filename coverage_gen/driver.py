"""Coverage-guided input generator for the Boogie interpreter.

Hybrid greybox mutation (Phase 1) + constraint-guided (Phase 2, future).
Follows the QSYM pattern: mutation until coverage stalls, then targeted
constraint solving.

Usage:
    python3 -m interpreter.coverage_gen.driver test_packages/aead_pkg/ \\
        [--phase greybox|hybrid] [--iters N] [--timeout S] [--out-dir PATH]
"""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from interpreter.utils.debug_log import DebugLogger
from interpreter.utils.support_matrix import support_matrix_summary

_BYTE_PATTERNS = [
    "0x00",
    "0xff",
    "0x7f",
    "0x80",
    "0xaa",
    "0x55",
]
_INT_BOUNDS = [0, 1, -1, 127, 128, 255, 256, 65535, 65536,
               2**31 - 1, 2**31, -(2**31)]
_HAVOC_SEQS = [[0], [1, 0], [1, 1, 0], [1, 1, 1, 0]]

_PROFILE_DEFAULTS = {
    "max-coverage": {
        "iters": 5000,
        "rounds": 4,
        "deterministic_budget": 500,
        "loop_bound": 16,
        "havoc_bound": 16,
        "max_path_depth": 2048,
        "max_solver_queries": 20000,
        "solver_timeout_ms": 200,
        "max_symbolic_states": 2048,
        "path_seed_limit": 0,
    },
    "balanced": {
        "iters": 1000,
        "rounds": 2,
        "deterministic_budget": 200,
        "loop_bound": 8,
        "havoc_bound": 8,
        "max_path_depth": 512,
        "max_solver_queries": 10000,
        "solver_timeout_ms": 100,
        "max_symbolic_states": 512,
        "path_seed_limit": 0,
    },
    "fast-traces": {
        "iters": 200,
        "rounds": 1,
        "deterministic_budget": 80,
        "loop_bound": 4,
        "havoc_bound": 4,
        "max_path_depth": 256,
        "max_solver_queries": 2000,
        "solver_timeout_ms": 50,
        "max_symbolic_states": 128,
        "path_seed_limit": 0,
    },
}

_MODE_TO_PROFILE = {
    "quick": "fast-traces",
    "standard": "balanced",
    "deep": "max-coverage",
}
_PROFILE_TO_MODE = {profile: mode for mode, profile in _MODE_TO_PROFILE.items()}

_WORKER_EVALUATOR = None


def _resolve_jobs(jobs: int | None) -> tuple[int, int]:
    cpu_count = os.cpu_count() or 1
    if jobs is None:
        return max(1, cpu_count), cpu_count
    return max(1, int(jobs)), cpu_count


def _init_worker():
    evaluator = _WORKER_EVALUATOR
    if evaluator is not None:
        evaluator.runs = 0
        evaluator.timeouts = 0
        evaluator.errors = 0
        evaluator.assertion_violations = 0
        evaluator.assume_violations = 0


def _run_eval(evaluator, program_inputs, input_name):
    if hasattr(evaluator, "run_result"):
        result = evaluator.run_result(program_inputs, input_name)
        return {
            "covered": set(result.covered or []),
            "status": result.status,
            "violation_pc": result.violation_pc,
            "violation_block": result.violation_block,
            "message": result.message,
        }

    covered = evaluator.run(program_inputs, input_name)
    last = getattr(evaluator, "last_result", None)
    return {
        "covered": set(covered or []),
        "status": getattr(last, "status", "ok"),
        "violation_pc": getattr(last, "violation_pc", None),
        "violation_block": getattr(last, "violation_block", None),
        "message": getattr(last, "message", None),
    }


def _eval_worker(task):
    evaluator = _WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("coverage worker evaluator was not initialized")
    index, program_inputs, input_name = task
    before_timeouts = evaluator.timeouts
    before_errors = evaluator.errors
    before_assertions = getattr(evaluator, "assertion_violations", 0)
    before_assumes = getattr(evaluator, "assume_violations", 0)
    result = _run_eval(evaluator, program_inputs, input_name)
    result.update({
        "index": index,
        "runs": 1,
        "timeouts": evaluator.timeouts - before_timeouts,
        "errors": evaluator.errors - before_errors,
        "assertion_violations": (
            getattr(evaluator, "assertion_violations", 0) - before_assertions
        ),
        "assume_violations": (
            getattr(evaluator, "assume_violations", 0) - before_assumes
        ),
    })
    return result


def _concolic_worker(task):
    evaluator = _WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("coverage worker evaluator was not initialized")
    before_errors = evaluator.errors
    candidates, stats = evaluator.concolic_suggest(**task["kwargs"])
    return {
        "index": task["index"],
        "candidates": candidates,
        "stats": stats,
        "errors": evaluator.errors - before_errors,
    }


def _symbolic_worker(task):
    evaluator = _WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("coverage worker evaluator was not initialized")
    before_errors = evaluator.errors
    candidates, stats = evaluator.symbolic_explore(**task["kwargs"])
    return {
        "index": task["index"],
        "candidates": candidates,
        "stats": stats,
        "errors": evaluator.errors - before_errors,
    }


def _path_task_progress(label: str, completed: int, total: int, result: dict):
    if total <= 1:
        return
    step = max(1, total // 4)
    if completed != total and completed % step != 0:
        return
    stats = result.get("stats", {}) or {}
    print(f"[{label}] completed {completed}/{total} seeds | "
          f"candidates={len(result.get('candidates') or [])} "
          f"states={stats.get('states_explored', 0)} "
          f"solver_queries={stats.get('solver_queries', 0)} "
          f"frontier={stats.get('frontier_ranked_objectives', 0)}")


class CoverageExecutor:
    """Evaluator facade with optional fork-backed parallel candidate execution."""

    def __init__(self, program, test_name: str, *, timeout: int,
                 engine: str, jobs: int | None, debug_logger=None):
        from interpreter.coverage_gen.evaluator import Evaluator

        self.debug = (debug_logger or DebugLogger.disabled()).bind(
            component="coverage-executor", test_name=test_name)
        self.jobs, self.cpu_count = _resolve_jobs(jobs)
        self.parallel_enabled = self.jobs > 1 and hasattr(os, "fork")
        self.evaluator = Evaluator(program, test_name, timeout=timeout,
                                   engine=engine,
                                   debug_logger=self.debug)
        self.runs = 0
        self.timeouts = 0
        self.errors = 0
        self.assertion_violations = 0
        self.assume_violations = 0
        self.assertion_pcs = {}
        self.worker_setup_seconds = 0.0
        self._executor = None

        if self.parallel_enabled:
            global _WORKER_EVALUATOR
            _WORKER_EVALUATOR = self.evaluator
            try:
                import multiprocessing
                ctx = multiprocessing.get_context("fork")
                t0 = time.perf_counter()
                self._executor = ProcessPoolExecutor(
                    max_workers=self.jobs,
                    mp_context=ctx,
                    initializer=_init_worker,
                )
                self.worker_setup_seconds = time.perf_counter() - t0
            except Exception:
                self.parallel_enabled = False
                self._executor = None

    @property
    def engine_requested(self):
        return self.evaluator.engine_requested

    @property
    def engine(self):
        return self.evaluator.engine

    @property
    def native_fallback_reason(self):
        return self.evaluator.native_fallback_reason

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def evaluate_many(self, tasks: list[tuple[int, object, str]]) -> list[dict]:
        if not tasks:
            return []
        self.debug.event("candidate", "evaluate_many_start", count=len(tasks),
                         parallel=self.parallel_enabled)
        if not self.parallel_enabled or self._executor is None:
            results = []
            for index, program_inputs, input_name in tasks:
                before_timeouts = self.evaluator.timeouts
                before_errors = self.evaluator.errors
                before_assertions = getattr(
                    self.evaluator, "assertion_violations", 0)
                before_assumes = getattr(
                    self.evaluator, "assume_violations", 0)
                result = _run_eval(self.evaluator, program_inputs, input_name)
                result.update({
                    "index": index,
                    "runs": 1,
                    "timeouts": self.evaluator.timeouts - before_timeouts,
                    "errors": self.evaluator.errors - before_errors,
                    "assertion_violations": (
                        getattr(self.evaluator, "assertion_violations", 0)
                        - before_assertions
                    ),
                    "assume_violations": (
                        getattr(self.evaluator, "assume_violations", 0)
                        - before_assumes
                    ),
                })
                results.append(result)
                self._merge_eval_stats(result)
            self.debug.event("candidate", "evaluate_many_end",
                             count=len(results),
                             runs=self.runs,
                             errors=self.errors,
                             timeouts=self.timeouts)
            return results

        futures = {
            self._executor.submit(_eval_worker, task): task[0]
            for task in tasks
        }
        results = []
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "index": futures[fut],
                    "covered": set(),
                    "runs": 1,
                    "timeouts": 0,
                    "errors": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            self._merge_eval_stats(result)
        out = sorted(results, key=lambda r: r["index"])
        self.debug.event("candidate", "evaluate_many_end",
                         count=len(out),
                         runs=self.runs,
                         errors=self.errors,
                         timeouts=self.timeouts)
        return out

    def concolic_many(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        self.debug.event("solver", "concolic_many_start",
                         count=len(tasks), parallel=self.parallel_enabled)
        if not hasattr(self.evaluator, "concolic_suggest"):
            return [
                {
                    "index": task["index"],
                    "candidates": [],
                    "stats": {"skipped": "evaluator-does-not-support-concolic"},
                    "errors": 0,
                }
                for task in tasks
            ]
        if not self.parallel_enabled or self._executor is None:
            results = []
            total = len(tasks)
            for completed, task in enumerate(tasks, start=1):
                before_errors = self.evaluator.errors
                candidates, stats = self.evaluator.concolic_suggest(
                    **task["kwargs"])
                result = {
                    "index": task["index"],
                    "candidates": candidates,
                    "stats": stats,
                    "errors": self.evaluator.errors - before_errors,
                }
                results.append(result)
                self.errors += int(result.get("errors", 0))
                _path_task_progress("concolic", completed, total, result)
            self.debug.event("solver", "concolic_many_end",
                             count=len(results), errors=self.errors)
            return results

        futures = {
            self._executor.submit(_concolic_worker, task): task["index"]
            for task in tasks
        }
        results = []
        total = len(futures)
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "index": futures[fut],
                    "candidates": [],
                    "stats": {"error": f"{type(exc).__name__}: {exc}"},
                    "errors": 1,
                }
            results.append(result)
            self.errors += int(result.get("errors", 0))
            _path_task_progress("concolic", completed, total, result)
        out = sorted(results, key=lambda r: r["index"])
        self.debug.event("solver", "concolic_many_end",
                         count=len(out), errors=self.errors)
        return out

    def symbolic_many(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        self.debug.event("solver", "symbolic_many_start",
                         count=len(tasks), parallel=self.parallel_enabled)
        if not hasattr(self.evaluator, "symbolic_explore"):
            return [
                {
                    "index": task["index"],
                    "candidates": [],
                    "stats": {"skipped": "evaluator-does-not-support-symbolic"},
                    "errors": 0,
                }
                for task in tasks
            ]
        if not self.parallel_enabled or self._executor is None:
            results = []
            total = len(tasks)
            for completed, task in enumerate(tasks, start=1):
                before_errors = self.evaluator.errors
                candidates, stats = self.evaluator.symbolic_explore(
                    **task["kwargs"])
                result = {
                    "index": task["index"],
                    "candidates": candidates,
                    "stats": stats,
                    "errors": self.evaluator.errors - before_errors,
                }
                results.append(result)
                self.errors += int(result.get("errors", 0))
                _path_task_progress("symbolic", completed, total, result)
            self.debug.event("solver", "symbolic_many_end",
                             count=len(results), errors=self.errors)
            return results

        futures = {
            self._executor.submit(_symbolic_worker, task): task["index"]
            for task in tasks
        }
        results = []
        total = len(futures)
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "index": futures[fut],
                    "candidates": [],
                    "stats": {"error": f"{type(exc).__name__}: {exc}"},
                    "errors": 1,
                }
            results.append(result)
            self.errors += int(result.get("errors", 0))
            _path_task_progress("symbolic", completed, total, result)
        out = sorted(results, key=lambda r: r["index"])
        self.debug.event("solver", "symbolic_many_end",
                         count=len(out), errors=self.errors)
        return out

    def _merge_eval_stats(self, result: dict):
        self.runs += int(result.get("runs", 0))
        self.timeouts += int(result.get("timeouts", 0))
        self.errors += int(result.get("errors", 0))
        self.assertion_violations += int(result.get("assertion_violations", 0))
        self.assume_violations += int(result.get("assume_violations", 0))
        if result.get("status") == "assert_violation":
            pc = result.get("violation_pc")
            key = str(pc) if pc is not None else "unknown"
            self.assertion_pcs[key] = self.assertion_pcs.get(key, 0) + 1


def _next_input_idx(out_dir: Path, prefix: str) -> int:
    """Find the next available <prefix>_N index in out_dir."""
    idxs = []
    for p in out_dir.glob(f"{prefix}_*.input"):
        m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", p.stem)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs, default=-1) + 1


def _next_gen_idx(out_dir: Path) -> int:
    """Find the next available gen_N index in out_dir."""
    return _next_input_idx(out_dir, "gen")


def _next_trace_idx(out_dir: Path) -> int:
    """Find the next available trace_N index in out_dir."""
    return _next_input_idx(out_dir, "trace")


def _next_assert_idx(out_dir: Path) -> int:
    """Find the next available assert_N index in out_dir."""
    return _next_input_idx(out_dir, "assert")


def _sample_blocks(blocks: set, limit: int = 4) -> str:
    sample = sorted(blocks)[:limit]
    suffix = "" if len(blocks) <= limit else ", ..."
    return ", ".join(sample) + suffix


def _record_candidate_result(corpus, out_dir: Path, candidate, covered: set,
                             params_line: str, gen_idx: int, trace_idx: int,
                             label: str, *, allow_trace: bool = True,
                             debug_candidates: bool = False,
                             status: str = "ok",
                             violation_pc=None,
                             violation_block=None,
                             debug_logger=None):
    from interpreter.coverage_gen.writer import write_input_file

    debug = debug_logger or DebugLogger.disabled()
    new_blocks = covered - corpus.covered
    text = None
    if status == "assert_violation":
        pc_key = str(violation_pc) if violation_pc is not None else "unknown"
        assertion_sig = (pc_key, corpus.trace_signature(candidate, covered))
        if assertion_sig in corpus.assertion_signatures:
            if debug_candidates:
                print(f"[{label}:assert-skip] duplicate assertion trace | "
                      f"pc={pc_key}")
            debug.event("candidate", "candidate_skip",
                        label=label, reason="duplicate_assertion",
                        status=status, violation_pc=violation_pc,
                        violation_block=violation_block)
            return None, gen_idx, trace_idx
        corpus.assertion_signatures.add(assertion_sig)

        assert_idx = _next_assert_idx(out_dir)
        out_path = out_dir / f"assert_{assert_idx}.input"
        text = write_input_file(candidate, params_line)
        out_path.write_text(text)
        corpus.assertion_inputs += 1
        corpus.assertion_pcs[pc_key] = corpus.assertion_pcs.get(pc_key, 0) + 1
        corpus.add_assertion_feedback(out_path, candidate, covered,
                                      params_line=params_line, source=label)
        loc = f"pc={violation_pc}" if violation_pc is not None else "pc=?"
        if violation_block:
            loc += f" block={violation_block}"
        if new_blocks:
            print(f"[{label}:assert] failing input saw "
                  f"{len(new_blocks):3d} otherwise-new blocks | "
                  f"wrote assert_{assert_idx}.input for feedback | {loc}")
        else:
            print(f"[{label}:assert] failing input recorded for feedback | "
                  f"wrote assert_{assert_idx}.input | {loc}")
        debug.event("candidate", "candidate_assertion",
                    label=label, output=str(out_path),
                    new_blocks=len(new_blocks),
                    violation_pc=violation_pc,
                    violation_block=violation_block)
        return "assertion", gen_idx, trace_idx

    if status == "assume_violation":
        if debug_candidates:
            loc = f"pc={violation_pc}" if violation_pc is not None else "pc=?"
            if violation_block:
                loc += f" block={violation_block}"
            print(f"[{label}:assume] invalid precondition input | {loc}")
        debug.event("candidate", "candidate_skip",
                    label=label, reason="assume_violation",
                    violation_pc=violation_pc,
                    violation_block=violation_block)
        return None, gen_idx, trace_idx

    if new_blocks:
        out_path = out_dir / f"gen_{gen_idx}.input"
        text = write_input_file(candidate, params_line)
        out_path.write_text(text)
        corpus.add(out_path, candidate, covered, params_line=params_line,
                   source=label)
        print(f"[{label}] +{len(new_blocks):3d} blocks -> "
              f"{len(corpus.covered):5d} total | wrote gen_{gen_idx}.input | "
              f"{_sample_blocks(new_blocks)}")
        debug.event("candidate", "candidate_kept",
                    label=label, reason="new_coverage",
                    output=str(out_path),
                    new_blocks=sorted(new_blocks),
                    blocks_total=len(corpus.covered))
        return "coverage", gen_idx + 1, trace_idx

    if allow_trace and corpus.should_add_trace(candidate, covered):
        out_path = out_dir / f"trace_{trace_idx}.input"
        text = text or write_input_file(candidate, params_line)
        out_path.write_text(text)
        if corpus.add(out_path, candidate, covered, params_line=params_line,
                      source=label, allow_trace=True):
            print(f"[{label}:trace] kept distinct trace | "
                  f"wrote trace_{trace_idx}.input")
            debug.event("candidate", "candidate_kept",
                        label=label, reason="trace_diversity",
                        output=str(out_path))
            return "trace", gen_idx, trace_idx + 1

    if debug_candidates:
        print(f"[{label}:skip] no new coverage or trace diversity")
    debug.event("candidate", "candidate_skip",
                label=label, reason="no_new_coverage_or_trace",
                status=status, covered=len(covered))
    return None, gen_idx, trace_idx


def _record_seed_assertion_feedback(corpus, path: Path, inputs, covered: set,
                                    params_line: str, result: dict,
                                    *, debug_candidates: bool = False,
                                    debug_logger=None):
    debug = debug_logger or DebugLogger.disabled()
    pc = result.get("violation_pc")
    pc_key = str(pc) if pc is not None else "unknown"
    assertion_sig = (pc_key, corpus.trace_signature(inputs, covered))
    if assertion_sig in corpus.assertion_signatures:
        if debug_candidates:
            print(f"[seed:assert-skip] {path.name} duplicate assertion trace | "
                  f"pc={pc_key}")
        debug.event("candidate", "seed_assertion_skip",
                    path=str(path), reason="duplicate_assertion", pc=pc_key)
        return

    corpus.assertion_signatures.add(assertion_sig)
    corpus.seed_assertion_feedback_inputs += 1
    corpus.seed_assertion_feedback_pcs[pc_key] = (
        corpus.seed_assertion_feedback_pcs.get(pc_key, 0) + 1
    )
    corpus.add_assertion_feedback(path, inputs, covered,
                                  params_line=params_line, source="seed")
    if debug_candidates:
        loc = f"pc={pc}" if pc is not None else "pc=?"
        block = result.get("violation_block")
        if block:
            loc += f" block={block}"
        print(f"[seed:assert] {path.name} recorded for feedback | {loc}")
    debug.event("candidate", "seed_assertion_feedback",
                path=str(path), pc=pc_key,
                block=result.get("violation_block"))


def _write_candidate(corpus, evaluator, out_dir: Path, candidate,
                     params_line: str, gen_idx: int, trace_idx: int,
                     label: str, *, allow_trace: bool = True,
                     debug_logger=None):
    result = _run_eval(evaluator, candidate, f"gen_{gen_idx}")
    return _record_candidate_result(
        corpus, out_dir, candidate, result["covered"], params_line, gen_idx, trace_idx,
        label, allow_trace=allow_trace,
        status=result.get("status", "ok"),
        violation_pc=result.get("violation_pc"),
        violation_block=result.get("violation_block"),
        debug_logger=debug_logger,
    )


def _write_interesting(corpus, evaluator, out_dir: Path, candidate,
                       params_line: str, gen_idx: int, label: str,
                       *, debug_logger=None) -> tuple[bool, int]:
    trace_idx = _next_trace_idx(out_dir)
    kind, gen_idx, _trace_idx = _write_candidate(
        corpus, evaluator, out_dir, candidate, params_line, gen_idx,
        trace_idx, label, allow_trace=False, debug_logger=debug_logger,
    )
    return kind == "coverage", gen_idx


def _evaluate_candidate_batch(corpus, executor: CoverageExecutor, out_dir: Path,
                              candidates: list[tuple[object, str, str]],
                              *, gen_idx: int, trace_idx: int,
                              debug_candidates: bool = False,
                              debug_logger=None):
    debug = debug_logger or DebugLogger.disabled()
    debug.event("candidate", "candidate_batch_start",
                count=len(candidates))
    tasks = [
        (idx, candidate, f"{label}_{idx}")
        for idx, (candidate, _params_line, label) in enumerate(candidates)
    ]
    results = executor.evaluate_many(tasks)
    by_index = {result["index"]: result for result in results}
    coverage_new = 0
    trace_new = 0
    for idx, (candidate, params_line, label) in enumerate(candidates):
        result = by_index.get(idx, {"covered": set()})
        kind, gen_idx, trace_idx = _record_candidate_result(
            corpus, out_dir, candidate, set(result.get("covered") or []),
            params_line, gen_idx, trace_idx, label,
            debug_candidates=debug_candidates,
            status=result.get("status", "ok"),
            violation_pc=result.get("violation_pc"),
            violation_block=result.get("violation_block"),
            debug_logger=debug,
        )
        if kind == "coverage":
            coverage_new += 1
        elif kind == "trace":
            trace_new += 1
    debug.event("candidate", "candidate_batch_end",
                count=len(candidates), coverage_new=coverage_new,
                trace_new=trace_new)
    return coverage_new, trace_new, gen_idx, trace_idx


def _load_seeds(seed_dir: Path, corpus, executor: CoverageExecutor,
                *, debug_candidates: bool = False, debug_logger=None) -> int:
    from interpreter.coverage_gen.corpus import _extract_params_line
    from interpreter.utils.input_parser import parse_input_file

    debug = debug_logger or DebugLogger.disabled()
    seed_dir = Path(seed_dir)
    debug.event("candidate", "seed_load_start", seed_dir=str(seed_dir))
    if not seed_dir.exists():
        debug.event("candidate", "seed_load_missing", seed_dir=str(seed_dir))
        return 0

    parsed = []
    for p in sorted(seed_dir.glob("*.input")):
        try:
            parsed.append((p, parse_input_file(p), _extract_params_line(p)))
        except Exception as exc:
            corpus.invalid_inputs += 1
            print(f"[corpus] Warning: could not load seed {p.name}: {exc}")
            debug.event("candidate", "seed_parse_error",
                        path=str(p), error=f"{type(exc).__name__}: {exc}")

    tasks = [
        (idx, inputs, path.stem)
        for idx, (path, inputs, _params_line) in enumerate(parsed)
    ]
    results = executor.evaluate_many(tasks)
    by_index = {result["index"]: result for result in results}
    for idx, (path, inputs, params_line) in enumerate(parsed):
        result = by_index.get(idx, {"covered": set()})
        covered = set(result.get("covered") or [])
        status = result.get("status", "ok")
        if status == "assert_violation":
            _record_seed_assertion_feedback(
                corpus, path, inputs, covered, params_line, result,
                debug_candidates=debug_candidates,
                debug_logger=debug)
            continue
        if status == "assume_violation":
            if debug_candidates:
                pc = result.get("violation_pc")
                loc = f"pc={pc}" if pc is not None else "pc=?"
                block = result.get("violation_block")
                if block:
                    loc += f" block={block}"
                print(f"[seed:assume] {path.name} invalid precondition input | "
                      f"{loc}")
            debug.event("candidate", "seed_skip", path=str(path),
                        reason="assume_violation",
                        pc=result.get("violation_pc"),
                        block=result.get("violation_block"))
            continue
        if status not in ("ok", None):
            if debug_candidates:
                print(f"[seed:skip] {path.name} status={status}")
            debug.event("candidate", "seed_skip", path=str(path),
                        reason="status", status=status)
            continue
        added = corpus.add(
            path, inputs, covered,
            params_line=params_line, source="existing", allow_trace=True,
        )
        if debug_candidates and not added:
            print(f"[seed:skip] {path.name} duplicate/no trace diversity")
        debug.event("candidate", "seed_result", path=str(path),
                    added=bool(added), covered=len(covered),
                    status=status)

    corpus.loaded_inputs += len(parsed)
    debug.event("candidate", "seed_load_end",
                parsed=len(parsed), loaded=corpus.loaded_inputs,
                corpus_entries=len(corpus.entries))
    return len(parsed)


def _repeat_hex(size: int, byte_hex: str) -> str:
    byte = byte_hex.removeprefix("0x")
    return "0x" + (byte * max(0, size))


def _mutated_input(base, var_name, mutate_fn):
    candidate = copy.deepcopy(base)
    mutate_fn(candidate.variables[var_name])
    return candidate


def _deterministic_candidates(inputs):
    """Yield bounded structure-aware probes for one ProgramInputs."""
    for var_name, inp in inputs.variables.items():
        if inp.value is not None:
            for value in _INT_BOUNDS:
                yield _mutated_input(
                    inputs, var_name,
                    lambda x, value=value: setattr(x, "value", value),
                )
            continue

        if inp.buffers:
            for buf_idx, buf in enumerate(inp.buffers):
                size = int(buf.get("size", 0) or 0)
                if size <= 0:
                    continue
                for pattern in _BYTE_PATTERNS:
                    def _apply(x, buf_idx=buf_idx, size=size, pattern=pattern):
                        x.buffers[buf_idx]["contents"] = _repeat_hex(size, pattern)
                    yield _mutated_input(inputs, var_name, _apply)
            continue

        if inp.struct:
            for field_idx, field in enumerate(inp.struct):
                if "buffer" in field:
                    size = int(field["buffer"].get("size", 0) or 0)
                    if size <= 0:
                        continue
                    for pattern in _BYTE_PATTERNS:
                        def _apply(x, field_idx=field_idx, size=size,
                                   pattern=pattern):
                            x.struct[field_idx]["buffer"]["contents"] = (
                                _repeat_hex(size, pattern)
                            )
                        yield _mutated_input(inputs, var_name, _apply)
                elif "value" in field:
                    for value in _INT_BOUNDS:
                        def _apply(x, field_idx=field_idx, value=value):
                            size = int(x.struct[field_idx].get("size", 0) or 0)
                            if size > 0:
                                masked = value & ((1 << (size * 8)) - 1)
                                x.struct[field_idx]["value"] = (
                                    "0x" + f"{masked:0{size * 2}x}"
                                )
                            else:
                                x.struct[field_idx]["value"] = value
                        yield _mutated_input(inputs, var_name, _apply)
            continue

        if inp.havoc_seq is not None:
            for seq in _HAVOC_SEQS:
                yield _mutated_input(
                    inputs, var_name,
                    lambda x, seq=seq: setattr(x, "havoc_seq", list(seq)),
                )


def run_deterministic_probes(corpus, evaluator, out_dir: Path,
                             max_candidates: int = 200,
                             *, debug_candidates: bool = False,
                             debug_logger=None) -> int:
    """Try deterministic boundary/havoc probes before random greybox fuzz."""
    gen_idx = _next_gen_idx(out_dir)
    trace_idx = _next_trace_idx(out_dir)
    candidates = []
    for entry in list(corpus.entries):
        for candidate in _deterministic_candidates(entry.inputs):
            if len(candidates) >= max_candidates:
                break
            candidates.append((
                candidate,
                entry.params_line or corpus.params_line,
                "deterministic",
            ))
        if len(candidates) >= max_candidates:
            break

    if hasattr(evaluator, "evaluate_many"):
        new_count, _trace_new, _gen_idx, _trace_idx = _evaluate_candidate_batch(
            corpus, evaluator, out_dir, candidates, gen_idx=gen_idx,
            trace_idx=trace_idx, debug_candidates=debug_candidates,
            debug_logger=debug_logger)
        return new_count

    new_count = 0
    for candidate, params_line, label in candidates:
        kind, gen_idx, trace_idx = _write_candidate(
            corpus, evaluator, out_dir, candidate, params_line,
            gen_idx, trace_idx, label,
            debug_logger=debug_logger,
        )
        if kind == "coverage":
            new_count += 1
    return new_count


def run_greybox(corpus, evaluator, out_dir: Path, iters: int,
                *, batch_size: int = 1, progress_interval: int = 200,
                debug_candidates: bool = False,
                debug_logger=None) -> int:
    """Run the greybox mutation phase. Returns the number of new inputs written."""
    from interpreter.coverage_gen.mutator import mutate, splice

    gen_idx = _next_gen_idx(out_dir)
    trace_idx = _next_trace_idx(out_dir)
    new_count = 0
    stale = 0
    stale_limit = max(50, iters // 10)

    evaluated = 0
    batch_size = max(1, int(batch_size))
    while evaluated < iters:
        batch = []
        for _ in range(min(batch_size, iters - evaluated)):
            entry = corpus.pick()
            if entry is None:
                break

            # 10% splice, 90% mutation
            if len(corpus.entries) > 1 and random.random() < 0.1:
                other = corpus.pick()
                candidate = splice(entry.inputs, other.inputs)
            else:
                candidate = mutate(entry.inputs)
            batch.append((candidate, entry.params_line or corpus.params_line,
                          "greybox"))
        if not batch:
            break

        if hasattr(evaluator, "evaluate_many"):
            batch_new, _trace_new, gen_idx, trace_idx = _evaluate_candidate_batch(
                corpus, evaluator, out_dir, batch, gen_idx=gen_idx,
                trace_idx=trace_idx, debug_candidates=debug_candidates,
                debug_logger=debug_logger)
        else:
            batch_new = 0
            for candidate, params_line, label in batch:
                kind, gen_idx, trace_idx = _write_candidate(
                    corpus, evaluator, out_dir, candidate, params_line,
                    gen_idx, trace_idx, label,
                    debug_logger=debug_logger,
                )
                if kind == "coverage":
                    batch_new += 1

        evaluated += len(batch)
        if batch_new:
            new_count += batch_new
            stale = 0
        else:
            stale += len(batch)
            if stale >= stale_limit:
                print(f"[greybox] Coverage stalled for {stale} iterations — stopping")
                break

        if progress_interval and evaluated % progress_interval == 0:
            print(f"[greybox] iter={evaluated}/{iters} | "
                  f"covered={len(corpus.covered)} blocks | "
                  f"corpus={len(corpus.entries)} entries")

    return new_count


def _merge_stats(total: dict, stats: dict):
    for key, value in (stats or {}).items():
        if isinstance(value, bool):
            if key == "complete_within_bounds":
                if key not in total:
                    total[key] = bool(value)
                else:
                    total[key] = bool(total.get(key, True) and value)
            else:
                total[key] = bool(total.get(key, False) or value)
        elif isinstance(value, int):
            total[key] = int(total.get(key, 0)) + value
        elif value and key not in total:
            total[key] = value


def _resolve_mode_profile(mode: str | None, profile: str | None) -> tuple[str, str]:
    if profile is not None:
        if profile not in _PROFILE_DEFAULTS:
            raise ValueError(f"unknown coverage generation profile: {profile}")
        return _PROFILE_TO_MODE.get(profile, mode or "custom"), profile

    mode = mode or "deep"
    if mode not in _MODE_TO_PROFILE:
        raise ValueError(f"unknown coverage generation mode: {mode}")
    return mode, _MODE_TO_PROFILE[mode]


def _engine_detail(executor: CoverageExecutor) -> str:
    if executor.engine == "native" and executor.native_fallback_reason is None:
        return "native-rust"
    if executor.native_fallback_reason:
        return "python-fallback"
    return executor.engine


def _auto_path_seed_limit(profile: str, requested_limit: int | None,
                          jobs: int, entries: int) -> int:
    if entries <= 0:
        return 0
    if requested_limit is not None and requested_limit > 0:
        return min(entries, int(requested_limit))

    scale = {
        "max-coverage": (4, 16),
        "balanced": (2, 8),
        "fast-traces": (1, 4),
    }.get(profile, (2, 8))
    factor, floor = scale
    return min(entries, max(floor, max(1, jobs) * factor))


def _rank_path_entries(corpus, entries: list) -> list[tuple[int, object, float]]:
    """Rank seeds so expensive path search starts with high-yield entries."""
    block_counts = {}
    for entry in entries:
        for block in entry.covered_blocks:
            block_counts[block] = block_counts.get(block, 0) + 1

    ranked = []
    for idx, entry in enumerate(entries):
        rare_score = sum(1.0 / block_counts.get(block, 1)
                         for block in entry.covered_blocks)
        source_bonus = {
            "symbolic": 5.0,
            "concolic": 4.0,
            "deterministic": 3.0,
            "greybox": 2.0,
            "existing": 1.0,
            "seed": 1.0,
        }.get(entry.source, 1.0)
        reason_bonus = 16.0 if entry.interesting_reason == "assertion-feedback" else 0.0
        trace_bonus = 1.0 if entry.interesting_reason == "trace" else 0.0
        score = rare_score + float(entry.energy) + source_bonus + reason_bonus + trace_bonus
        ranked.append((idx, entry, score))

    ranked.sort(key=lambda item: (
        -item[2],
        str(item[1].path or ""),
        item[0],
    ))
    return ranked


def _select_path_entries(corpus, entries: list, *, profile: str,
                         seed_limit: int | None, jobs: int) -> tuple[list, int]:
    ranked = _rank_path_entries(corpus, entries)
    limit = _auto_path_seed_limit(profile, seed_limit, jobs, len(ranked))
    return ranked[:limit], limit


def _budget_slice(remaining: int, weights: list[float], index: int) -> int:
    if remaining <= 0:
        return 0
    remaining_weight = sum(max(1.0, weight) for weight in weights[index:])
    if remaining_weight <= 0:
        return max(1, remaining // max(1, len(weights) - index))
    budget = int(remaining * max(1.0, weights[index]) / remaining_weight)
    return max(1, min(remaining, budget))


def _record_phase_decision(decisions: list[dict], *, phase: str,
                           round_idx: int | None, action: str,
                           reason: str, **metadata) -> None:
    item = {
        "phase": phase,
        "action": action,
        "reason": reason,
    }
    if round_idx is not None:
        item["round"] = round_idx + 1
    for key, value in metadata.items():
        if value is not None:
            item[key] = value
    decisions.append(item)


def _symbolic_should_stop(stats: dict, new_inputs: int) -> tuple[bool, str | None]:
    if new_inputs:
        return False, None
    if stats.get("skipped"):
        return True, str(stats.get("skipped"))
    if stats.get("error"):
        return False, None

    unknown = int(stats.get("paths_unknown", 0) or 0)
    feasible = int(stats.get("paths_feasible", 0) or 0)
    solver_sat = int(stats.get("solver_sat", 0) or 0)
    value_profile = int(stats.get("value_profile_candidates", 0) or 0)
    infeasible = (
        int(stats.get("paths_infeasible", 0) or 0)
        + int(stats.get("solver_unsat", 0) or 0)
    )
    frontier = int(stats.get("frontier_ranked_objectives", 0) or 0)

    if stats.get("complete_within_bounds") and unknown == 0:
        return True, "symbolic-complete-no-new-inputs"
    if infeasible and unknown == 0 and feasible == 0 and solver_sat == 0 and value_profile == 0:
        return True, "all-symbolic-objectives-infeasible"
    if frontier == 0 and feasible == 0 and solver_sat == 0 and value_profile == 0:
        return True, "no-symbolic-frontier"
    return False, None


def _concolic_should_stop(stats: dict, new_inputs: int,
                          no_yield_rounds: int) -> tuple[bool, str | None]:
    if new_inputs:
        return False, None
    if stats.get("skipped"):
        return True, str(stats.get("skipped"))
    if stats.get("error"):
        return False, None
    if no_yield_rounds >= 2:
        return True, "concolic-no-new-inputs"
    return False, None


def _build_recommendations(*, executor: CoverageExecutor, stall_reason: str | None,
                           total_new_inputs: int, symbolic_stats: dict,
                           phase_decisions: list[dict]) -> list[str]:
    recommendations = []
    if executor.native_fallback_reason:
        recommendations.append(
            "Native Rust engine was unavailable; rebuild/install it before long coverage runs.")
    if stall_reason == "no-seeds":
        recommendations.append(
            "No valid seed inputs were loaded; run gen-input without --seed-only or provide a seed directory.")
    if (
        total_new_inputs == 0
        and symbolic_stats.get("complete_within_bounds")
        and int(symbolic_stats.get("paths_unknown", 0) or 0) == 0
    ):
        recommendations.append(
            "Bounded symbolic search completed without new valid inputs; current seeds may already cover the bounded feasible paths it saw.")
    if any(d.get("action") == "downshift" for d in phase_decisions):
        recommendations.append(
            "Adaptive scheduling skipped later unproductive path phases so the run could spend time on remaining strategies.")
    return recommendations


def run_symbolic(corpus, evaluator, out_dir: Path, *,
                 loop_bound: int = 8,
                 max_path_depth: int = 512,
                 max_solver_queries: int = 10000,
                 solver_timeout_ms: int = 100,
                 havoc_bound: int = 8,
                 max_states: int = 256,
                 profile: str = "max-coverage",
                 seed_limit: int | None = None,
                 jobs: int = 1,
                 debug_candidates: bool = False,
                 debug_logger=None) -> tuple[int, dict]:
    """Run bounded native symbolic path exploration over current seeds."""
    debug = debug_logger or DebugLogger.disabled()
    supports_symbolic = (
        hasattr(evaluator, "symbolic_many")
        or hasattr(evaluator, "symbolic_explore")
    )
    if not supports_symbolic:
        return 0, {"skipped": "evaluator-does-not-support-symbolic"}

    gen_idx = _next_gen_idx(out_dir)
    trace_idx = _next_trace_idx(out_dir)
    new_count = 0
    stats_total = {}
    entries = list(corpus.entries) + list(corpus.assertion_feedback_entries)
    if corpus.assertion_feedback_entries:
        stats_total["assertion_feedback_seeds"] = len(
            corpus.assertion_feedback_entries)
    selected, effective_seed_limit = _select_path_entries(
        corpus, entries, profile=profile, seed_limit=seed_limit, jobs=jobs)
    stats_total["available_seed_entries"] = len(entries)
    stats_total["selected_seed_entries"] = len(selected)
    stats_total["path_seed_limit"] = effective_seed_limit
    if len(selected) < len(entries):
        stats_total["seed_limit_applied"] = True
    if not selected:
        debug.event("solver", "symbolic_no_seeds",
                    available=len(entries), seed_limit=seed_limit)
        return 0, stats_total
    limit_label = "auto" if seed_limit is None or seed_limit <= 0 else seed_limit
    print(f"[symbolic] scheduling {len(selected)}/{len(entries)} seeds "
          f"(limit={limit_label}, "
          f"jobs={jobs})")
    debug.event("solver", "symbolic_schedule",
                selected=len(selected), available=len(entries),
                seed_limit=effective_seed_limit, jobs=jobs)

    if hasattr(evaluator, "symbolic_many"):
        tasks = []
        remaining_queries = max_solver_queries
        remaining_states = max_states
        weights = [score for _entry_idx, _entry, score in selected]
        for selected_idx, (entry_idx, entry, _score) in enumerate(selected):
            if remaining_queries <= 0:
                stats_total["solver_query_bounded"] = True
                stats_total["complete_within_bounds"] = False
                break
            if remaining_states <= 0:
                stats_total["states_bounded"] = (
                    int(stats_total.get("states_bounded", 0))
                    + (len(selected) - selected_idx)
                )
                stats_total["complete_within_bounds"] = False
                break
            query_budget = _budget_slice(
                remaining_queries, weights, selected_idx)
            state_budget = _budget_slice(
                remaining_states, weights, selected_idx)
            remaining_queries -= query_budget
            remaining_states -= state_budget
            tasks.append({
                "index": entry_idx,
                "kwargs": {
                    "program_inputs": entry.inputs,
                    "input_name": f"symbolic_seed_{entry_idx}",
                    "covered_blocks": set(corpus.covered),
                    "loop_bound": loop_bound,
                    "max_path_depth": max_path_depth,
                    "max_solver_queries": query_budget,
                    "solver_timeout_ms": solver_timeout_ms,
                    "havoc_bound": havoc_bound,
                    "max_states": state_budget,
                },
            })

        candidate_batch = []
        for result in evaluator.symbolic_many(tasks):
            _merge_stats(stats_total, result.get("stats", {}))
            entry = entries[result["index"]]
            for candidate in result.get("candidates", []):
                candidate_batch.append((
                    candidate,
                    entry.params_line or corpus.params_line,
                    "symbolic",
                ))
        batch_new, _trace_new, gen_idx, trace_idx = _evaluate_candidate_batch(
            corpus, evaluator, out_dir, candidate_batch, gen_idx=gen_idx,
            trace_idx=trace_idx, debug_candidates=debug_candidates,
            debug_logger=debug_logger)
        new_count += batch_new
    else:
        for selected_idx, (entry_idx, entry, _score) in enumerate(selected):
            remaining_queries = max_solver_queries - int(stats_total.get("solver_queries", 0))
            remaining_states = max_states - int(stats_total.get("states_explored", 0))
            if remaining_queries <= 0:
                stats_total["solver_query_bounded"] = True
                stats_total["complete_within_bounds"] = False
                break
            if remaining_states <= 0:
                stats_total["states_bounded"] = (
                    int(stats_total.get("states_bounded", 0))
                    + (len(selected) - selected_idx)
                )
                stats_total["complete_within_bounds"] = False
                break
            candidates, stats = evaluator.symbolic_explore(
                entry.inputs,
                f"symbolic_seed_{entry_idx}",
                corpus.covered,
                loop_bound=loop_bound,
                max_path_depth=max_path_depth,
                max_solver_queries=remaining_queries,
                solver_timeout_ms=solver_timeout_ms,
                havoc_bound=havoc_bound,
                max_states=remaining_states,
            )
            _merge_stats(stats_total, stats)
            for candidate in candidates:
                kind, gen_idx, trace_idx = _write_candidate(
                    corpus, evaluator, out_dir, candidate,
                    entry.params_line or corpus.params_line,
                    gen_idx, trace_idx, "symbolic",
                    debug_logger=debug_logger,
                )
                if kind == "coverage":
                    new_count += 1

    return new_count, stats_total


def run_concolic(corpus, evaluator, out_dir: Path, *,
                 loop_bound: int = 8,
                 max_path_depth: int = 512,
                 max_solver_queries: int = 10000,
                 solver_timeout_ms: int = 100,
                 havoc_bound: int = 8,
                 profile: str = "max-coverage",
                 seed_limit: int | None = None,
                 jobs: int = 1,
                 debug_candidates: bool = False,
                 debug_logger=None) -> tuple[int, dict]:
    """Run native branch-negation concolic search over current corpus."""
    debug = debug_logger or DebugLogger.disabled()
    supports_concolic = (
        hasattr(evaluator, "concolic_many")
        or hasattr(evaluator, "concolic_suggest")
    )
    if not supports_concolic:
        return 0, {"skipped": "evaluator-does-not-support-concolic"}

    gen_idx = _next_gen_idx(out_dir)
    trace_idx = _next_trace_idx(out_dir)
    new_count = 0
    stats_total = {}
    entries = list(corpus.entries) + list(corpus.assertion_feedback_entries)
    if corpus.assertion_feedback_entries:
        stats_total["assertion_feedback_seeds"] = len(
            corpus.assertion_feedback_entries)
    selected, effective_seed_limit = _select_path_entries(
        corpus, entries, profile=profile, seed_limit=seed_limit, jobs=jobs)
    stats_total["available_seed_entries"] = len(entries)
    stats_total["selected_seed_entries"] = len(selected)
    stats_total["path_seed_limit"] = effective_seed_limit
    if len(selected) < len(entries):
        stats_total["seed_limit_applied"] = True
    if not selected:
        debug.event("solver", "concolic_no_seeds",
                    available=len(entries), seed_limit=seed_limit)
        return 0, stats_total
    limit_label = "auto" if seed_limit is None or seed_limit <= 0 else seed_limit
    print(f"[concolic] scheduling {len(selected)}/{len(entries)} seeds "
          f"(limit={limit_label}, "
          f"jobs={jobs})")
    debug.event("solver", "concolic_schedule",
                selected=len(selected), available=len(entries),
                seed_limit=effective_seed_limit, jobs=jobs)

    if hasattr(evaluator, "concolic_many"):
        tasks = []
        remaining = max_solver_queries
        weights = [score for _entry_idx, _entry, score in selected]
        for selected_idx, (entry_idx, entry, _score) in enumerate(selected):
            if remaining <= 0:
                stats_total["solver_query_bounded"] = True
                break
            budget = _budget_slice(remaining, weights, selected_idx)
            remaining -= budget
            tasks.append({
                "index": entry_idx,
                "kwargs": {
                    "program_inputs": entry.inputs,
                    "input_name": f"concolic_seed_{entry_idx}",
                    "covered_blocks": set(corpus.covered),
                    "loop_bound": loop_bound,
                    "max_path_depth": max_path_depth,
                    "max_solver_queries": budget,
                    "solver_timeout_ms": solver_timeout_ms,
                    "havoc_bound": havoc_bound,
                },
            })
        candidate_batch = []
        for result in evaluator.concolic_many(tasks):
            _merge_stats(stats_total, result.get("stats", {}))
            entry = entries[result["index"]]
            for candidate in result.get("candidates", []):
                candidate_batch.append((
                    candidate,
                    entry.params_line or corpus.params_line,
                    "concolic",
                ))
        batch_new, _trace_new, gen_idx, trace_idx = _evaluate_candidate_batch(
            corpus, evaluator, out_dir, candidate_batch, gen_idx=gen_idx,
            trace_idx=trace_idx, debug_candidates=debug_candidates,
            debug_logger=debug_logger)
        new_count += batch_new
    else:
        for entry_idx, entry, _score in selected:
            remaining = max_solver_queries - int(stats_total.get("solver_queries", 0))
            if remaining <= 0:
                stats_total["solver_query_bounded"] = True
                break
            candidates, stats = evaluator.concolic_suggest(
                entry.inputs,
                f"concolic_seed_{entry_idx}",
                corpus.covered,
                loop_bound=loop_bound,
                max_path_depth=max_path_depth,
                max_solver_queries=remaining,
                solver_timeout_ms=solver_timeout_ms,
                havoc_bound=havoc_bound,
            )
            _merge_stats(stats_total, stats)
            for candidate in candidates:
                kind, gen_idx, trace_idx = _write_candidate(
                    corpus, evaluator, out_dir, candidate,
                    entry.params_line or corpus.params_line,
                    gen_idx, trace_idx, "concolic",
                    debug_logger=debug_logger,
                )
                if kind == "coverage":
                    new_count += 1

    return new_count, stats_total


def _coverage_dict(program, covered_blocks: set) -> dict:
    try:
        from interpreter.runner import compute_coverage
        cov = compute_coverage(program, set(covered_blocks)) or {}
        return {k: v for k, v in cov.items() if k != "explored_blocks"}
    except Exception:
        return {}


def _write_report(out_dir: Path, report: dict):
    out_path = out_dir / "coverage_gen.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out_path


def _resolve_profile(profile: str, overrides: dict) -> dict:
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(f"unknown coverage generation profile: {profile}")
    resolved = dict(_PROFILE_DEFAULTS[profile])
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def generate_coverage_inputs(program, test_name: str, seed_dir: Path,
                             out_dir: Path, *, mode: str | None = None,
                             profile: str | None = None,
                             iters: int | None = None,
                             timeout: int = 30, engine: str = "native",
                             rng_seed: int = 0,
                             path_engine: str = "hybrid",
                             jobs: int | None = None,
                             progress_interval: int = 200,
                             debug_candidates: bool = False,
                             rounds: int | None = None,
                             deterministic_budget: int | None = None,
                             loop_bound: int | None = None,
                             havoc_bound: int | None = None,
                             max_path_depth: int | None = None,
                             max_solver_queries: int | None = None,
                             max_symbolic_states: int | None = None,
                             path_seed_limit: int | None = None,
                             solver_timeout_ms: int | None = None,
                             debug_logger=None) -> dict:
    """Run the unified deterministic/concolic/greybox input pipeline."""
    from interpreter.coverage_gen.corpus import Corpus

    mode, profile = _resolve_mode_profile(mode, profile)
    cfg = _resolve_profile(profile, {
        "iters": iters,
        "rounds": rounds,
        "deterministic_budget": deterministic_budget,
        "loop_bound": loop_bound,
        "havoc_bound": havoc_bound,
        "max_path_depth": max_path_depth,
        "max_solver_queries": max_solver_queries,
        "max_symbolic_states": max_symbolic_states,
        "path_seed_limit": path_seed_limit,
        "solver_timeout_ms": solver_timeout_ms,
    })
    iters = int(cfg["iters"])
    rounds = max(1, int(cfg["rounds"]))
    deterministic_budget = max(0, int(cfg["deterministic_budget"]))
    loop_bound = int(cfg["loop_bound"])
    havoc_bound = int(cfg["havoc_bound"])
    max_path_depth = int(cfg["max_path_depth"])
    max_solver_queries = int(cfg["max_solver_queries"])
    max_symbolic_states = int(cfg["max_symbolic_states"])
    path_seed_limit = int(cfg.get("path_seed_limit", 0))
    solver_timeout_ms = int(cfg["solver_timeout_ms"])

    debug = (debug_logger or DebugLogger.disabled()).bind(
        component="coverage-driver", test_name=test_name)

    random.seed(rng_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = CoverageExecutor(program, test_name, timeout=timeout,
                                engine=engine, jobs=jobs,
                                debug_logger=debug)
    corpus = Corpus()
    phase_timings = []
    deterministic_new = 0
    concolic_new = 0
    concolic_stats = {}
    symbolic_new = 0
    symbolic_stats = {}
    greybox_new = 0
    round_reports = []
    phase_decisions = []
    stall_reason = None
    n_seeds = 0
    t_total = time.perf_counter()

    print("[driver] gen-input pipeline")
    print(f"[driver] target={test_name} mode={mode} profile={profile} "
          f"path_engine={path_engine} engine={engine} rng_seed={rng_seed}")
    print(f"[driver] seed_dir={seed_dir} out_dir={out_dir}")
    print(f"[driver] jobs={executor.jobs}/{executor.cpu_count} "
          f"parallel={executor.parallel_enabled} "
          f"worker_setup={executor.worker_setup_seconds:.3f}s")
    print(f"[driver] engine_detail={_engine_detail(executor)}")
    print(f"[driver] budgets: deterministic={deterministic_budget}, "
          f"rounds={rounds}, iters={iters}, loop_bound={loop_bound}, "
          f"havoc_bound={havoc_bound}, depth={max_path_depth}, "
          f"symbolic_states={max_symbolic_states}, "
          f"solver_queries={max_solver_queries}, "
          f"path_seed_limit={'auto' if path_seed_limit <= 0 else path_seed_limit}, "
          f"solver_timeout_ms={solver_timeout_ms}")
    if executor.native_fallback_reason:
        print("[driver] native evaluator fallback: "
              f"{executor.native_fallback_reason}")
    debug.event(
        "exec",
        "coverage_pipeline_start",
        mode=mode,
        profile=profile,
        path_engine=path_engine,
        engine=engine,
        seed_dir=seed_dir,
        out_dir=out_dir,
        jobs=executor.jobs,
        budgets={
            "deterministic": deterministic_budget,
            "rounds": rounds,
            "iters": iters,
            "loop_bound": loop_bound,
            "havoc_bound": havoc_bound,
            "max_path_depth": max_path_depth,
            "max_solver_queries": max_solver_queries,
            "max_symbolic_states": max_symbolic_states,
            "path_seed_limit": path_seed_limit,
            "solver_timeout_ms": solver_timeout_ms,
        },
    )

    try:
        print(f"[driver] Loading seeds from {seed_dir} ...")
        t0 = time.perf_counter()
        n_seeds = _load_seeds(
            seed_dir, corpus, executor, debug_candidates=debug_candidates,
            debug_logger=debug)
        elapsed = time.perf_counter() - t0
        phase_timings.append({
            "phase": "seed-load",
            "seconds": round(elapsed, 6),
            "evaluated": n_seeds,
            "evals_per_sec": round(n_seeds / elapsed, 3) if elapsed else n_seeds,
        })
        print(f"[driver] Loaded {n_seeds} seeds in {elapsed:.2f}s | "
              f"{len(corpus.covered)} blocks covered | "
              f"{len(corpus.entries)} corpus entries | "
              f"invalid={corpus.invalid_inputs} duplicate={corpus.duplicate_inputs}")

        if corpus.entries:
            if deterministic_budget > 0:
                print(f"[driver] Starting deterministic probe phase "
                      f"({deterministic_budget} candidates max) ...")
                before_runs = executor.runs
                before_blocks = len(corpus.covered)
                before_trace = corpus.trace_interesting_inputs
                t0 = time.perf_counter()
                deterministic_new = run_deterministic_probes(
                    corpus, executor, out_dir,
                    max_candidates=deterministic_budget,
                    debug_candidates=debug_candidates,
                    debug_logger=debug)
                elapsed = time.perf_counter() - t0
                evaluated = executor.runs - before_runs
                phase_timings.append({
                    "phase": "deterministic",
                    "seconds": round(elapsed, 6),
                    "evaluated": evaluated,
                    "evals_per_sec": round(evaluated / elapsed, 3) if elapsed else evaluated,
                    "new_inputs": deterministic_new,
                    "blocks_added": len(corpus.covered) - before_blocks,
                    "trace_added": corpus.trace_interesting_inputs - before_trace,
                })
                print(f"[driver] deterministic done in {elapsed:.2f}s | "
                      f"evals={evaluated} new_inputs={deterministic_new} "
                      f"blocks={len(corpus.covered)} "
                      f"trace={corpus.trace_interesting_inputs} "
                      f"rate={evaluated / elapsed if elapsed else evaluated:.1f}/s")
            else:
                _record_phase_decision(
                    phase_decisions, phase="deterministic", round_idx=None,
                    action="skip", reason="zero-deterministic-budget")
                print("[driver] Skipping deterministic probe phase "
                      "(budget=0)")

            remaining_iters = iters
            remaining_solver_queries = max_solver_queries
            remaining_symbolic_states = max_symbolic_states
            symbolic_disabled_reason = None
            concolic_disabled_reason = None
            concolic_no_yield_rounds = 0
            for round_idx in range(rounds):
                round_symbolic = 0
                round_concolic = 0
                round_greybox = 0
                before_blocks = len(corpus.covered)
                before_trace = corpus.trace_interesting_inputs
                round_start_runs = executor.runs
                round_t0 = time.perf_counter()

                if (
                    path_engine in ("hybrid", "symbolic")
                    and symbolic_disabled_reason is not None
                ):
                    _record_phase_decision(
                        phase_decisions, phase="symbolic", round_idx=round_idx,
                        action="skip", reason=symbolic_disabled_reason)
                    print(f"[driver] Round {round_idx + 1}/{rounds}: "
                          f"skipping symbolic ({symbolic_disabled_reason})")
                elif path_engine in ("hybrid", "symbolic") and (
                    remaining_solver_queries > 0 and remaining_symbolic_states > 0
                ):
                    print(f"[driver] Round {round_idx + 1}/{rounds}: bounded symbolic "
                          f"(loop_bound={loop_bound}, depth={max_path_depth}, "
                          f"havoc_bound={havoc_bound}, "
                          f"states={remaining_symbolic_states}, "
                          f"solver_queries={remaining_solver_queries}) ...")
                    t0 = time.perf_counter()
                    before_runs = executor.runs
                    round_symbolic, stats = run_symbolic(
                        corpus, executor, out_dir,
                        loop_bound=loop_bound,
                        max_path_depth=max_path_depth,
                        max_solver_queries=remaining_solver_queries,
                        solver_timeout_ms=solver_timeout_ms,
                        havoc_bound=havoc_bound,
                        max_states=remaining_symbolic_states,
                        profile=profile,
                        seed_limit=path_seed_limit,
                        jobs=executor.jobs,
                        debug_candidates=debug_candidates,
                        debug_logger=debug,
                    )
                    elapsed = time.perf_counter() - t0
                    evaluated = executor.runs - before_runs
                    _merge_stats(symbolic_stats, stats)
                    symbolic_new += round_symbolic
                    remaining_solver_queries = max(
                        0,
                        max_solver_queries - int(symbolic_stats.get("solver_queries", 0)),
                    )
                    remaining_symbolic_states = max(
                        0,
                        max_symbolic_states - int(symbolic_stats.get("states_explored", 0)),
                    )
                    phase_timings.append({
                        "phase": "symbolic",
                        "round": round_idx + 1,
                        "seconds": round(elapsed, 6),
                        "evaluated": evaluated,
                        "evals_per_sec": round(evaluated / elapsed, 3) if elapsed else evaluated,
                        "new_inputs": round_symbolic,
                        "states_explored": stats.get("states_explored", 0),
                        "paths_feasible": stats.get("paths_feasible", 0),
                        "paths_infeasible": stats.get("paths_infeasible", 0),
                        "paths_unknown": stats.get("paths_unknown", 0),
                        "solver_queries": stats.get("solver_queries", 0),
                        "solver_cache_hits": stats.get("solver_cache_hits", 0),
                        "python_state_cache_hits": stats.get(
                            "python_state_cache_hits", 0),
                        "python_state_cache_misses": stats.get(
                            "python_state_cache_misses", 0),
                        "available_seed_entries": stats.get(
                            "available_seed_entries", 0),
                        "selected_seed_entries": stats.get(
                            "selected_seed_entries", 0),
                        "seed_limit_applied": bool(stats.get(
                            "seed_limit_applied", False)),
                        "frontier_ranked_objectives": stats.get(
                            "frontier_ranked_objectives", 0),
                        "value_profile_candidates": stats.get(
                            "value_profile_candidates", 0),
                    })
                    print(f"[driver] symbolic round {round_idx + 1} done in "
                          f"{elapsed:.2f}s | evals={evaluated} "
                          f"new_inputs={round_symbolic} "
                          f"states={stats.get('states_explored', 0)} "
                          f"unknown={stats.get('paths_unknown', 0)} "
                          f"solver_queries={stats.get('solver_queries', 0)} "
                          f"cache_hits={stats.get('solver_cache_hits', 0)} "
                          f"py_state_cache={stats.get('python_state_cache_hits', 0)}/"
                          f"{stats.get('python_state_cache_misses', 0)} "
                          f"frontier={stats.get('frontier_ranked_objectives', 0)} "
                          f"value_profile={stats.get('value_profile_candidates', 0)} "
                          f"seeds={stats.get('selected_seed_entries', 0)}/"
                          f"{stats.get('available_seed_entries', 0)}")

                    stop_symbolic, stop_reason = _symbolic_should_stop(
                        stats, round_symbolic)
                    if stop_symbolic and path_engine == "hybrid":
                        symbolic_disabled_reason = stop_reason
                        _record_phase_decision(
                            phase_decisions, phase="symbolic",
                            round_idx=round_idx, action="downshift",
                            reason=stop_reason,
                            selected_seed_entries=stats.get(
                                "selected_seed_entries", 0),
                            available_seed_entries=stats.get(
                                "available_seed_entries", 0))
                        print(f"[driver] symbolic downshift: {stop_reason}")

                if (
                    path_engine in ("hybrid", "concolic")
                    and concolic_disabled_reason is not None
                ):
                    _record_phase_decision(
                        phase_decisions, phase="concolic", round_idx=round_idx,
                        action="skip", reason=concolic_disabled_reason)
                    print(f"[driver] Round {round_idx + 1}/{rounds}: "
                          f"skipping concolic ({concolic_disabled_reason})")
                elif path_engine in ("hybrid", "concolic") and remaining_solver_queries > 0:
                    print(f"[driver] Round {round_idx + 1}/{rounds}: bounded concolic "
                          f"(loop_bound={loop_bound}, depth={max_path_depth}, "
                          f"havoc_bound={havoc_bound}, "
                          f"solver_queries={remaining_solver_queries}) ...")
                    t0 = time.perf_counter()
                    before_runs = executor.runs
                    round_concolic, stats = run_concolic(
                        corpus, executor, out_dir,
                        loop_bound=loop_bound,
                        max_path_depth=max_path_depth,
                        max_solver_queries=remaining_solver_queries,
                        solver_timeout_ms=solver_timeout_ms,
                        havoc_bound=havoc_bound,
                        profile=profile,
                        seed_limit=path_seed_limit,
                        jobs=executor.jobs,
                        debug_candidates=debug_candidates,
                        debug_logger=debug,
                    )
                    elapsed = time.perf_counter() - t0
                    evaluated = executor.runs - before_runs
                    _merge_stats(concolic_stats, stats)
                    concolic_new += round_concolic
                    remaining_solver_queries = max(
                        0,
                        max_solver_queries - int(concolic_stats.get("solver_queries", 0)),
                    )
                    phase_timings.append({
                        "phase": "concolic",
                        "round": round_idx + 1,
                        "seconds": round(elapsed, 6),
                        "evaluated": evaluated,
                        "evals_per_sec": round(evaluated / elapsed, 3) if elapsed else evaluated,
                        "new_inputs": round_concolic,
                        "solver_queries": stats.get("solver_queries", 0),
                        "solver_cache_hits": stats.get("solver_cache_hits", 0),
                        "python_state_cache_hits": stats.get(
                            "python_state_cache_hits", 0),
                        "python_state_cache_misses": stats.get(
                            "python_state_cache_misses", 0),
                        "available_seed_entries": stats.get(
                            "available_seed_entries", 0),
                        "selected_seed_entries": stats.get(
                            "selected_seed_entries", 0),
                        "seed_limit_applied": bool(stats.get(
                            "seed_limit_applied", False)),
                        "frontier_ranked_objectives": stats.get(
                            "frontier_ranked_objectives", 0),
                        "value_profile_candidates": stats.get(
                            "value_profile_candidates", 0),
                    })
                    print(f"[driver] concolic round {round_idx + 1} done in "
                          f"{elapsed:.2f}s | evals={evaluated} "
                          f"new_inputs={round_concolic} "
                          f"solver_queries={stats.get('solver_queries', 0)} "
                          f"cache_hits={stats.get('solver_cache_hits', 0)} "
                          f"py_state_cache={stats.get('python_state_cache_hits', 0)}/"
                          f"{stats.get('python_state_cache_misses', 0)} "
                          f"frontier={stats.get('frontier_ranked_objectives', 0)} "
                          f"value_profile={stats.get('value_profile_candidates', 0)} "
                          f"seeds={stats.get('selected_seed_entries', 0)}/"
                          f"{stats.get('available_seed_entries', 0)}")

                    if round_concolic:
                        concolic_no_yield_rounds = 0
                    else:
                        concolic_no_yield_rounds += 1
                    stop_concolic, stop_reason = _concolic_should_stop(
                        stats, round_concolic, concolic_no_yield_rounds)
                    if stop_concolic and path_engine == "hybrid":
                        concolic_disabled_reason = stop_reason
                        _record_phase_decision(
                            phase_decisions, phase="concolic",
                            round_idx=round_idx, action="downshift",
                            reason=stop_reason,
                            selected_seed_entries=stats.get(
                                "selected_seed_entries", 0),
                            available_seed_entries=stats.get(
                                "available_seed_entries", 0))
                        print(f"[driver] concolic downshift: {stop_reason}")

                if path_engine in ("hybrid", "fuzz") and remaining_iters > 0:
                    rounds_left = rounds - round_idx
                    fuzz_budget = max(1, (remaining_iters + rounds_left - 1) // rounds_left)
                    print(f"[driver] Round {round_idx + 1}/{rounds}: greybox "
                          f"({fuzz_budget} iterations, rng_seed={rng_seed}) ...")
                    t0 = time.perf_counter()
                    before_runs = executor.runs
                    round_greybox = run_greybox(
                        corpus, executor, out_dir, iters=fuzz_budget,
                        batch_size=max(1, executor.jobs * 4),
                        progress_interval=progress_interval,
                        debug_candidates=debug_candidates,
                        debug_logger=debug,
                    )
                    elapsed = time.perf_counter() - t0
                    evaluated = executor.runs - before_runs
                    greybox_new += round_greybox
                    remaining_iters = max(0, remaining_iters - fuzz_budget)
                    phase_timings.append({
                        "phase": "greybox",
                        "round": round_idx + 1,
                        "seconds": round(elapsed, 6),
                        "evaluated": evaluated,
                        "evals_per_sec": round(evaluated / elapsed, 3) if elapsed else evaluated,
                        "new_inputs": round_greybox,
                    })
                    print(f"[driver] greybox round {round_idx + 1} done in "
                          f"{elapsed:.2f}s | evals={evaluated} "
                          f"new_inputs={round_greybox} "
                          f"rate={evaluated / elapsed if elapsed else evaluated:.1f}/s")

                blocks_added = len(corpus.covered) - before_blocks
                trace_added = corpus.trace_interesting_inputs - before_trace
                round_elapsed = time.perf_counter() - round_t0
                round_evals = executor.runs - round_start_runs
                round_reports.append({
                    "round": round_idx + 1,
                    "seconds": round(round_elapsed, 6),
                    "evaluated": round_evals,
                    "symbolic_new_inputs": round_symbolic,
                    "concolic_new_inputs": round_concolic,
                    "greybox_new_inputs": round_greybox,
                    "blocks_added": blocks_added,
                    "trace_interesting_added": trace_added,
                    "blocks_covered": len(corpus.covered),
                    "corpus_entries": len(corpus.entries),
                })
                print(f"[driver] round {round_idx + 1} summary | "
                      f"blocks_added={blocks_added} "
                      f"trace_added={trace_added} "
                      f"corpus={len(corpus.entries)} "
                      f"covered={len(corpus.covered)} "
                      f"timeouts={executor.timeouts} errors={executor.errors}")

                if blocks_added == 0:
                    stall_reason = "no-new-coverage-round"
                    break

            if stall_reason is None:
                if remaining_iters <= 0 and path_engine in ("hybrid", "fuzz"):
                    stall_reason = "iteration-budget-exhausted"
                elif remaining_symbolic_states <= 0 and path_engine in ("hybrid", "symbolic"):
                    stall_reason = "symbolic-state-budget-exhausted"
                elif remaining_solver_queries <= 0 and path_engine in ("hybrid", "symbolic", "concolic"):
                    stall_reason = "solver-query-budget-exhausted"
                else:
                    stall_reason = "round-budget-exhausted"
        else:
            print("[driver] No seeds found — cannot search without a starting input.",
                  file=sys.stderr)
            stall_reason = "no-seeds"
    finally:
        executor.close()

    total_elapsed = time.perf_counter() - t_total
    bounded_symbolic_complete = bool(
        symbolic_stats.get("complete_within_bounds", False)
    )
    complete_within_bounds = (
        path_engine == "symbolic"
        and bounded_symbolic_complete
        and executor.errors == 0
        and executor.timeouts == 0
        and executor.native_fallback_reason is None
    )
    total_new_inputs = deterministic_new + symbolic_new + concolic_new + greybox_new
    useful_inputs = total_new_inputs + corpus.generated_trace_inputs
    recommendations = _build_recommendations(
        executor=executor,
        stall_reason=stall_reason,
        total_new_inputs=total_new_inputs,
        symbolic_stats=symbolic_stats,
        phase_decisions=phase_decisions,
    )

    report = {
        "strategy": "unified-gen-input",
        "mode": mode,
        "pipeline_profile": profile,
        "phase_order": ["existing", "seed", "deterministic", "symbolic", "concolic", "fuzz"],
        "exhaustive": False,
        "exhaustive_unbounded": False,
        "complete_within_bounds": complete_within_bounds,
        "complete_within_bounds_scope": "symbolic-seed-corpus",
        "bounded_symbolic_complete": bounded_symbolic_complete,
        "bounds": {
            "rounds": rounds,
            "deterministic_budget": deterministic_budget,
            "loop_bound": loop_bound,
            "havoc_bound": havoc_bound,
            "max_path_depth": max_path_depth,
            "max_solver_queries": max_solver_queries,
            "max_symbolic_states": max_symbolic_states,
            "path_seed_limit": path_seed_limit,
            "solver_timeout_ms": solver_timeout_ms,
        },
        "jobs": executor.jobs,
        "cpu_count": executor.cpu_count,
        "parallel_enabled": executor.parallel_enabled,
        "worker_setup_seconds": round(executor.worker_setup_seconds, 6),
        "progress_interval": progress_interval,
        "debug_candidates": debug_candidates,
        "total_seconds": round(total_elapsed, 6),
        "phase_timings": phase_timings,
        "phase_decisions": phase_decisions,
        "engine_requested": executor.engine_requested,
        "engine_used": executor.engine,
        "engine_detail": _engine_detail(executor),
        "native_fallback": executor.native_fallback_reason is not None,
        "native_fallback_reason": executor.native_fallback_reason,
        "rng_seed": rng_seed,
        "path_engine": path_engine,
        "seed_inputs": n_seeds,
        "existing_inputs": n_seeds,
        "invalid_inputs": corpus.invalid_inputs,
        "duplicate_inputs": corpus.duplicate_inputs,
        "deterministic_new_inputs": deterministic_new,
        "symbolic_new_inputs": symbolic_new,
        "concolic_new_inputs": concolic_new,
        "greybox_new_inputs": greybox_new,
        "total_new_inputs": total_new_inputs,
        "trace_interesting_inputs": corpus.trace_interesting_inputs,
        "generated_trace_inputs": corpus.generated_trace_inputs,
        "assertion_inputs": corpus.assertion_inputs,
        "generated_assertion_inputs": corpus.assertion_inputs,
        "assertion_feedback_inputs": len(corpus.assertion_feedback_entries),
        "seed_assertion_feedback_inputs": corpus.seed_assertion_feedback_inputs,
        "assertion_violations": executor.assertion_violations,
        "assume_violations": executor.assume_violations,
        "assertion_pcs": executor.assertion_pcs,
        "generated_assertion_pcs": corpus.assertion_pcs,
        "seed_assertion_feedback_pcs": corpus.seed_assertion_feedback_pcs,
        "total_written_inputs": useful_inputs,
        "useful_inputs": useful_inputs,
        "valid_written_inputs": useful_inputs,
        "corpus_entries": len(corpus.entries),
        "candidates_evaluated": executor.runs,
        "timeouts": executor.timeouts,
        "errors": executor.errors,
        "symbolic": symbolic_stats,
        "concolic": concolic_stats,
        "rounds_completed": len(round_reports),
        "rounds": round_reports,
        "stall_reason": stall_reason,
        "recommendations": recommendations,
        "blocks_covered": len(corpus.covered),
        "coverage": _coverage_dict(program, corpus.covered),
    }
    report_path = _write_report(out_dir, report)
    debug.event("exec", "coverage_pipeline_end",
                total_seconds=round(total_elapsed, 6),
                total_new_inputs=total_new_inputs,
                blocks_covered=len(corpus.covered),
                stall_reason=stall_reason,
                report_path=report_path)
    print("[driver] Coverage search is bounded. "
          "Unbounded all-path exploration is not claimed.")
    if phase_decisions:
        for decision in phase_decisions[-3:]:
            round_txt = (
                f" round={decision['round']}" if "round" in decision else ""
            )
            print(f"[driver] adaptive: {decision['phase']} "
                  f"{decision['action']}{round_txt} | {decision['reason']}")
    for recommendation in recommendations[:3]:
        print(f"[driver] recommendation: {recommendation}")
    print(f"[driver] Wrote coverage report: {report_path}")
    print(f"[driver] final: covered={len(corpus.covered)} "
          f"new_inputs={total_new_inputs} "
          f"trace_inputs={corpus.generated_trace_inputs} "
          f"assert_feedback={len(corpus.assertion_feedback_entries)} "
          f"stall={stall_reason} engine={_engine_detail(executor)} "
          f"elapsed={total_elapsed:.2f}s")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Coverage-guided input generator for the Boogie interpreter"
    )
    parser.add_argument("pkg_dir",
                        help="Compiled package dir (test_packages/<name>_pkg/)")
    parser.add_argument("--phase", choices=["greybox", "hybrid"],
                        default="hybrid",
                        help="Generation strategy (default: hybrid)")
    parser.add_argument("--profile",
                        choices=sorted(_PROFILE_DEFAULTS),
                        default=None,
                        help="Advanced budget profile override "
                             "(default: chosen by --mode)")
    parser.add_argument("--mode", choices=sorted(_MODE_TO_PROFILE),
                        default=None,
                        help="One-stop generation mode (default: deep)")
    parser.add_argument("--quick", dest="mode", action="store_const",
                        const="quick",
                        help="Alias for --mode quick")
    parser.add_argument("--standard", dest="mode", action="store_const",
                        const="standard",
                        help="Alias for --mode standard")
    parser.add_argument("--deep", dest="mode", action="store_const",
                        const="deep",
                        help="Alias for --mode deep")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override greybox mutation iteration budget")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override concolic/fuzz feedback rounds")
    parser.add_argument("--deterministic-budget", type=int, default=None,
                        help="Override deterministic probe candidate budget")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-execution timeout in seconds (default: 30)")
    parser.add_argument("--engine", choices=["native", "python"],
                        default="native",
                        help="Execution engine for coverage candidates "
                             "(default: native, falls back to python)")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for reproducible greybox mutation "
                             "(default: 0)")
    parser.add_argument("--path-engine", choices=["hybrid", "symbolic", "concolic", "fuzz"],
                        default="hybrid",
                        help="Path search pipeline after deterministic probes "
                             "(default: hybrid)")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="Worker process count (default: all logical CPUs)")
    parser.add_argument("--progress-interval", type=int, default=200,
                        help="Print progress every N candidate evaluations "
                             "(default: 200)")
    parser.add_argument("--debug-candidates", action="store_true",
                        help="Print per-candidate duplicate/skip details")
    parser.add_argument("--loop-bound", type=int, default=None,
                        help="Maximum native concolic loop iterations per header "
                             "(profile default)")
    parser.add_argument("--havoc-bound", type=int, default=None,
                        help="Override maximum symbolic nondet calls per havoc var")
    parser.add_argument("--max-path-depth", type=int, default=None,
                        help="Override maximum native concolic block depth")
    parser.add_argument("--max-solver-queries", type=int, default=None,
                        help="Override maximum concolic solver queries")
    parser.add_argument("--max-symbolic-states", type=int, default=None,
                        help="Override maximum bounded symbolic states")
    parser.add_argument("--path-seed-limit", type=int, default=None,
                        help="Advanced cap on symbolic/concolic seeds per round "
                             "(default: automatic)")
    parser.add_argument("--solver-timeout-ms", type=int, default=None,
                        help="Override per-query SMT timeout in milliseconds")
    parser.add_argument("--debug-log",
                        help="Directory for structured debug JSONL sidecar logs")
    parser.add_argument("--debug-categories", default="all",
                        help="Comma-separated debug categories "
                             "(default: all; e.g. candidate,solver)")
    parser.add_argument("--out-dir",
                        help="Output directory for generated .input files "
                             "(default: test_input/<name>/)")
    parser.add_argument("--seed",
                        help="Seed corpus directory "
                             "(default: test_input/<name>/)")
    args = parser.parse_args(argv)

    pkg_path = Path(args.pkg_dir)
    m = re.fullmatch(r"(.+)_pkg", pkg_path.name)
    if not m:
        print(f"ERROR: {pkg_path.name!r} does not match <name>_pkg pattern",
              file=sys.stderr)
        sys.exit(1)
    test_name = m.group(1)

    # Load compiled program
    program_pkl = pkg_path / f"{test_name}.pkl"
    if not program_pkl.exists():
        print(f"ERROR: {program_pkl} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[driver] Loading {test_name} ...")
    program_bytes = program_pkl.read_bytes()
    program_hash = hashlib.sha256(program_bytes).hexdigest()
    program = pickle.loads(program_bytes)

    seed_dir = Path(args.seed) if args.seed else Path(f"test_input/{test_name}")
    out_dir = Path(args.out_dir) if args.out_dir else seed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_logger = DebugLogger.from_options(
        args.debug_log,
        args.debug_categories,
        run_id=f"{test_name}-{int(time.time())}",
    ).bind(test_name=test_name, command="coverage-gen",
           program_hash=program_hash)
    debug_logger.event("unsupported", "support_matrix",
                       summary=support_matrix_summary())

    report = generate_coverage_inputs(
        program, test_name, seed_dir, out_dir,
        mode=args.mode,
        profile=args.profile,
        iters=args.iters,
        timeout=args.timeout,
        engine=args.engine,
        rng_seed=args.rng_seed,
        path_engine=args.path_engine,
        jobs=args.jobs,
        progress_interval=args.progress_interval,
        debug_candidates=args.debug_candidates,
        rounds=args.rounds,
        deterministic_budget=args.deterministic_budget,
        loop_bound=args.loop_bound,
        havoc_bound=args.havoc_bound,
        max_path_depth=args.max_path_depth,
        max_solver_queries=args.max_solver_queries,
        max_symbolic_states=args.max_symbolic_states,
        path_seed_limit=args.path_seed_limit,
        solver_timeout_ms=args.solver_timeout_ms,
        debug_logger=debug_logger,
    )
    if report["seed_inputs"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
