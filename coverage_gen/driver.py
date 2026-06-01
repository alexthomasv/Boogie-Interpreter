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
from interpreter.coverage_gen.corpus import (
    coverage_features,
    path_features_from_sequence,
)
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
    "guided-deep": {
        "iters": 5000,
        "rounds": 4,
        "deterministic_budget": 500,
        "loop_bound": 16,
        "havoc_bound": 16,
        "max_path_depth": 1024,
        "max_solver_queries": 32768,
        "solver_timeout_ms": 200,
        "max_symbolic_states": 16384,
        "path_seed_limit": 0,
        "path_priority": "coverage-directed",
        "value_profile_policy": "late",
        "branch_distance_policy": "auto",
        "max_generations": 4,
        "children_per_trace": 512,
        "objective_policy": "branch,concrete-blocker,branch-distance,compound-value-profile,value-profile",
        "score_policy": "coverage-max",
        "state_pruning": "compatible-branch",
        "rare_branch_schedule": "auto",
        "target_uncovered": "auto",
        "corpus_reduce": "coverage-setcover",
    },
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
        "path_priority": "coverage-directed",
        "value_profile_policy": "late",
        "branch_distance_policy": "auto",
        "max_generations": 3,
        "children_per_trace": 256,
        "objective_policy": "branch,branch-distance,compound-value-profile,value-profile",
        "score_policy": "coverage-max",
        "state_pruning": "compatible-branch",
        "rare_branch_schedule": "auto",
        "target_uncovered": "auto",
        "corpus_reduce": "coverage-setcover",
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
        "path_priority": "coverage-directed",
        "value_profile_policy": "late",
        "branch_distance_policy": "auto",
        "max_generations": 2,
        "children_per_trace": 128,
        "objective_policy": "branch,branch-distance,compound-value-profile,value-profile",
        "score_policy": "coverage-max",
        "state_pruning": "compatible-branch",
        "rare_branch_schedule": "auto",
        "target_uncovered": "auto",
        "corpus_reduce": "coverage-setcover",
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
        "path_priority": "coverage-directed",
        "value_profile_policy": "late",
        "branch_distance_policy": "auto",
        "max_generations": 1,
        "children_per_trace": 64,
        "objective_policy": "branch,branch-distance,compound-value-profile,value-profile",
        "score_policy": "coverage-max",
        "state_pruning": "compatible-branch",
        "rare_branch_schedule": "auto",
        "target_uncovered": "auto",
        "corpus_reduce": "coverage-setcover",
    },
}

_MODE_TO_PROFILE = {
    "quick": "fast-traces",
    "standard": "balanced",
    "deep": "max-coverage",
}
_PROFILE_TO_MODE = {profile: mode for mode, profile in _MODE_TO_PROFILE.items()}

_WORKER_EVALUATOR = None
DEFAULT_MAX_STEPS_PER_INPUT = 100_000_000


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
        evaluator.step_limits = 0
        evaluator.assertion_violations = 0
        evaluator.assume_violations = 0


def _run_eval(evaluator, program_inputs, input_name):
    if hasattr(evaluator, "run_result"):
        result = evaluator.run_result(program_inputs, input_name)
        return {
            "covered": set(result.covered or []),
            "status": result.status,
            "block_sequence": tuple(result.block_sequence or ()),
            "violation_pc": result.violation_pc,
            "violation_block": result.violation_block,
            "message": result.message,
            "invalid_input": result.invalid_input,
            "invalid_reason": result.invalid_reason,
        }

    covered = evaluator.run(program_inputs, input_name)
    last = getattr(evaluator, "last_result", None)
    return {
        "covered": set(covered or []),
        "status": getattr(last, "status", "ok"),
        "block_sequence": tuple(getattr(last, "block_sequence", ()) or ()),
        "violation_pc": getattr(last, "violation_pc", None),
        "violation_block": getattr(last, "violation_block", None),
        "message": getattr(last, "message", None),
        "invalid_input": getattr(last, "invalid_input", False),
        "invalid_reason": getattr(last, "invalid_reason", None),
    }


def _eval_worker(task):
    evaluator = _WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("coverage worker evaluator was not initialized")
    index, program_inputs, input_name = task
    before_timeouts = evaluator.timeouts
    before_errors = evaluator.errors
    before_step_limits = getattr(evaluator, "step_limits", 0)
    before_assertions = getattr(evaluator, "assertion_violations", 0)
    before_assumes = getattr(evaluator, "assume_violations", 0)
    result = _run_eval(evaluator, program_inputs, input_name)
    result.update({
        "index": index,
        "runs": 1,
        "timeouts": evaluator.timeouts - before_timeouts,
        "errors": evaluator.errors - before_errors,
        "step_limits": getattr(evaluator, "step_limits", 0) - before_step_limits,
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
                 engine: str, jobs: int | None,
                 max_steps_per_input: int = DEFAULT_MAX_STEPS_PER_INPUT,
                 debug_logger=None):
        from interpreter.coverage_gen.evaluator import Evaluator

        self.debug = (debug_logger or DebugLogger.disabled()).bind(
            component="coverage-executor", test_name=test_name)
        self.jobs, self.cpu_count = _resolve_jobs(jobs)
        self.parallel_enabled = self.jobs > 1 and hasattr(os, "fork")
        self.evaluator = Evaluator(program, test_name, timeout=timeout,
                                   engine=engine,
                                   max_steps_per_input=max_steps_per_input,
                                   debug_logger=self.debug)
        self.runs = 0
        self.timeouts = 0
        self.errors = 0
        self.step_limits = 0
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
                before_step_limits = getattr(self.evaluator, "step_limits", 0)
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
                    "step_limits": (
                        getattr(self.evaluator, "step_limits", 0)
                        - before_step_limits
                    ),
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
        self.step_limits += int(result.get("step_limits", 0))
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


def _next_path_idx(out_dir: Path) -> int:
    """Find the next available path_N index in out_dir."""
    return _next_input_idx(out_dir, "path")


def _next_assert_idx(out_dir: Path) -> int:
    """Find the next available assert_N index in out_dir."""
    return _next_input_idx(out_dir, "assert")


def _sample_blocks(blocks: set, limit: int = 4) -> str:
    sample = sorted(blocks)[:limit]
    suffix = "" if len(blocks) <= limit else ", ..."
    return ", ".join(sample) + suffix


def _result_path_features(result: dict, covered: set | None = None) -> dict:
    sequence = tuple(result.get("block_sequence") or ())
    if not sequence and covered:
        sequence = tuple(sorted(covered))
    return path_features_from_sequence(sequence)


def _record_candidate_result(corpus, out_dir: Path, candidate, covered: set,
                             params_line: str, gen_idx: int, trace_idx: int,
                             label: str, *, allow_trace: bool = True,
                             debug_candidates: bool = False,
                             status: str = "ok",
                             violation_pc=None,
                             violation_block=None,
                             path_features: dict | None = None,
                             debug_logger=None):
    from interpreter.coverage_gen.writer import write_input_file

    debug = debug_logger or DebugLogger.disabled()
    path_features = path_features or path_features_from_sequence(())
    new_blocks = covered - corpus.covered
    new_edges = set(path_features.get("edges", ())) - corpus.covered_edges
    new_context_edges = (
        set(path_features.get("context_edges", ()))
        - corpus.covered_context_edges
    )
    if corpus.coverage_metric == "context-edge":
        new_paths = new_context_edges
    elif corpus.coverage_metric in {"edge", "branch"}:
        new_paths = new_edges
    else:
        new_paths = set()
    text = None
    if status == "assert_violation":
        pc_key = str(violation_pc) if violation_pc is not None else "unknown"
        assertion_sig = (
            pc_key,
            corpus.trace_signature(candidate, covered, path_features),
        )
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
                                      params_line=params_line, source=label,
                                      path_features=path_features)
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
        corpus.invalid_inputs += 1
        if debug_candidates:
            loc = f"pc={violation_pc}" if violation_pc is not None else "pc=?"
            if violation_block:
                loc += f" block={violation_block}"
            print(f"[{label}:assume] invalid precondition input | {loc}")
        debug.event("candidate", "candidate_skip",
                    label=label, reason="assume_violation",
                    invalid_input=True,
                    violation_pc=violation_pc,
                    violation_block=violation_block)
        return None, gen_idx, trace_idx

    if status == "step_limit":
        if debug_candidates:
            loc = f"pc={violation_pc}" if violation_pc is not None else "pc=?"
            if violation_block:
                loc += f" block={violation_block}"
            print(f"[{label}:step-limit] bounded execution stopped | {loc}")
        debug.event("candidate", "candidate_skip",
                    label=label, reason="step_limit",
                    violation_pc=violation_pc,
                    violation_block=violation_block)
        return None, gen_idx, trace_idx

    if status not in ("ok", None):
        if debug_candidates:
            print(f"[{label}:skip] status={status}")
        debug.event("candidate", "candidate_skip",
                    label=label, reason="status", status=status)
        return None, gen_idx, trace_idx

    if new_blocks:
        out_path = out_dir / f"gen_{gen_idx}.input"
        text = write_input_file(candidate, params_line)
        out_path.write_text(text)
        corpus.add(out_path, candidate, covered, params_line=params_line,
                   source=label, path_features=path_features)
        print(f"[{label}] +{len(new_blocks):3d} blocks -> "
              f"{len(corpus.covered):5d} total | wrote gen_{gen_idx}.input | "
              f"{_sample_blocks(new_blocks)}")
        debug.event("candidate", "candidate_kept",
                    label=label, reason="new_coverage",
                    output=str(out_path),
                    new_blocks=sorted(new_blocks),
                    blocks_total=len(corpus.covered))
        return "coverage", gen_idx + 1, trace_idx

    if new_paths:
        path_idx = _next_path_idx(out_dir)
        out_path = out_dir / f"path_{path_idx}.input"
        text = text or write_input_file(candidate, params_line)
        out_path.write_text(text)
        if corpus.add(out_path, candidate, covered, params_line=params_line,
                      source=label, allow_trace=True,
                      path_features=path_features):
            print(f"[{label}:path] +{len(new_paths):3d} "
                  f"{corpus.coverage_metric} paths | "
                  f"wrote path_{path_idx}.input")
            debug.event("candidate", "candidate_kept",
                        label=label, reason=f"new_{corpus.coverage_metric}",
                        output=str(out_path), new_paths=len(new_paths),
                        new_edges=len(new_edges),
                        new_context_edges=len(new_context_edges))
            return "path", gen_idx, trace_idx

    if allow_trace and corpus.should_add_trace(
        candidate, covered, path_features):
        out_path = out_dir / f"trace_{trace_idx}.input"
        text = text or write_input_file(candidate, params_line)
        out_path.write_text(text)
        if corpus.add(out_path, candidate, covered, params_line=params_line,
                      source=label, allow_trace=True,
                      path_features=path_features):
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


def _new_path_diagnostics() -> dict:
    return {
        "candidate_kind_summary": {},
        "candidate_target_summary": {},
        "candidate_objective_summary": {},
        "replay_status_by_target": {},
        "replay_status_by_objective": {},
        "new_blocks_by_target": {},
        "new_blocks_by_objective": {},
        "pruned_targets": {},
        "pruned_objectives": {},
        "pruned_candidates": 0,
        "first_new_block_depth": None,
        "first_new_block_target": None,
        "target_distance_best": None,
        "rare_branch_inputs": 0,
    }


def _bump(mapping: dict, key, amount: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + amount


def _record_path_diagnostic(diagnostics: dict | None, candidate,
                            result: dict, new_blocks: set) -> None:
    if diagnostics is None:
        return
    meta = getattr(candidate, "_swoosh_candidate_meta", {}) or {}
    kind = str(meta.get("kind") or "unknown")
    target = str(meta.get("target_block") or "unknown")
    objective = _candidate_objective_key(meta)
    status = str(result.get("status") or "ok")

    _bump(diagnostics["candidate_kind_summary"], kind)
    _bump(diagnostics["candidate_target_summary"], target)
    _bump(diagnostics["candidate_objective_summary"], objective)
    distance = meta.get("distance_to_uncovered")
    if distance is not None:
        try:
            distance = int(distance)
            if distance >= 0 and (
                diagnostics.get("target_distance_best") is None
                or distance < int(diagnostics["target_distance_best"])
            ):
                diagnostics["target_distance_best"] = distance
        except (TypeError, ValueError):
            pass
    by_status = diagnostics["replay_status_by_target"].setdefault(target, {})
    _bump(by_status, status)
    by_objective_status = diagnostics["replay_status_by_objective"].setdefault(
        objective, {})
    _bump(by_objective_status, status)
    if new_blocks:
        _bump(diagnostics["new_blocks_by_target"], target, len(new_blocks))
        _bump(
            diagnostics["new_blocks_by_objective"],
            objective,
            len(new_blocks),
        )
        if diagnostics.get("first_new_block_depth") is None:
            diagnostics["first_new_block_depth"] = meta.get(
                "distance_to_uncovered")
            diagnostics["first_new_block_target"] = target


def _candidate_objective_key(meta: dict) -> str:
    source = str(meta.get("source_block") or "unknown")
    target = str(meta.get("target_block") or "unknown")
    return f"{source}->{target}"


def _should_prune_replay(diagnostics: dict | None, candidate) -> bool:
    if diagnostics is None:
        return False
    meta = getattr(candidate, "_swoosh_candidate_meta", {}) or {}
    if meta.get("kind") not in {
        "value_profile",
        "compound_value_profile",
        "distance_guided",
    }:
        return False
    objective = _candidate_objective_key(meta)
    statuses = diagnostics["replay_status_by_objective"].get(objective, {})
    assertion_only = int(statuses.get("assert_violation", 0)) >= 64
    has_new_blocks = (
        int(diagnostics["new_blocks_by_objective"].get(objective, 0)) > 0
    )
    return assertion_only and not has_new_blocks


def _record_pruned_replay(diagnostics: dict | None, candidate) -> None:
    if diagnostics is None:
        return
    meta = getattr(candidate, "_swoosh_candidate_meta", {}) or {}
    target = str(meta.get("target_block") or "unknown")
    objective = _candidate_objective_key(meta)
    diagnostics["pruned_candidates"] = (
        int(diagnostics.get("pruned_candidates", 0)) + 1
    )
    _bump(diagnostics["pruned_targets"], target)
    _bump(diagnostics["pruned_objectives"], objective)
    _record_path_diagnostic(
        diagnostics, candidate, {"status": "pruned"}, set())


def _candidate_meta(candidate) -> dict:
    return getattr(candidate, "_swoosh_candidate_meta", {}) or {}


def _candidate_updates(candidate) -> dict:
    return dict(_candidate_meta(candidate).get("updates") or {})


def _materialize_replay_candidate(candidate):
    materialize = getattr(candidate, "materialize", None)
    if callable(materialize):
        return materialize()
    return candidate


def _candidate_sort_key(candidate) -> tuple:
    return _coverage_candidate_sort_key(candidate)


def _coverage_candidate_sort_key(
    candidate,
    *,
    corpus=None,
    score_policy: str = "new-blocks",
    rare_branch_schedule: str = "auto",
    target_uncovered: str = "auto",
) -> tuple:
    meta = _candidate_meta(candidate)
    kind_rank = {
        "branch_target": 0,
        "requires_repair": 1,
        "assert_repair": 2,
        "distance_guided": 3,
        "compound_value_profile": 4,
        "value_profile": 5,
    }.get(str(meta.get("kind") or ""), 9)
    if score_policy == "coverage-max":
        priority = int(meta.get("priority") or 0)
        distance = int(meta.get("distance_to_uncovered") or 999999)
        reachable = int(meta.get("reachable_uncovered") or 0)
        score = priority + reachable * 8_000
        if distance >= 0:
            score += max(0, 250_000 - distance * 2_500)
        if target_uncovered != "off" and distance == 0:
            score += 100_000
        if rare_branch_schedule != "off" and corpus is not None:
            target = str(meta.get("target_block") or "")
            block_counts = corpus.feature_frequencies("block")
            if target:
                score += int(50_000 / max(1, block_counts.get(("block", target), 1)))
        return (
            -score,
            kind_rank,
            distance,
            -reachable,
            str(meta.get("target_block") or ""),
            int(meta.get("raw_index") or 0),
        )
    return (
        kind_rank,
        -int(meta.get("priority") or 0),
        int(meta.get("distance_to_uncovered") or 999999),
        str(meta.get("target_block") or ""),
        int(meta.get("raw_index") or 0),
    )


def _candidate_group_key(candidate) -> tuple:
    meta = _candidate_meta(candidate)
    return (
        str(meta.get("target_block") or "unknown"),
        str(meta.get("source_block") or "unknown"),
    )


def _make_compound_candidate(base_inputs, candidates: list, *,
                             max_updates: int = 256):
    if len(candidates) < 2:
        return None

    from interpreter.coverage_gen.symbolic_state import apply_symbolic_updates

    bindings = getattr(candidates[0], "_swoosh_symbolic_bindings", {}) or {}
    havoc_bound = int(getattr(candidates[0], "_swoosh_havoc_bound", 8) or 8)
    merged_updates = {}
    raw_indices = []
    for candidate in candidates:
        updates = _candidate_updates(candidate)
        if not updates:
            continue
        conflict = any(
            name in merged_updates and merged_updates[name] != value
            for name, value in updates.items()
        )
        if conflict:
            continue
        if len(merged_updates) + len(updates) > max_updates:
            break
        merged_updates.update(updates)
        raw_indices.append(_candidate_meta(candidate).get("raw_index"))

    if len(merged_updates) < 2:
        return None

    compound = apply_symbolic_updates(
        base_inputs,
        bindings,
        merged_updates,
        havoc_bound=havoc_bound,
    )
    first_meta = _candidate_meta(candidates[0])
    numeric_indices = [
        int(idx) for idx in raw_indices if idx is not None
    ]
    compound._swoosh_symbolic_bindings = bindings
    compound._swoosh_havoc_bound = havoc_bound
    compound._swoosh_candidate_meta = {
        "raw_index": min(numeric_indices) if numeric_indices else 0,
        "source_block": first_meta.get("source_block"),
        "target_block": first_meta.get("target_block"),
        "branch_index": first_meta.get("branch_index"),
        "kind": "compound_value_profile",
        "priority": int(first_meta.get("priority") or 0) + len(merged_updates),
        "distance_to_uncovered": first_meta.get("distance_to_uncovered", -1),
        "reachable_uncovered": first_meta.get("reachable_uncovered", 0),
        "updates": merged_updates,
        "compound_size": len(merged_updates),
        "compound_raw_indices": raw_indices[:32],
    }
    return compound


def _compound_value_profile_candidates(base_inputs, candidates: list, *,
                                       children_per_trace: int,
                                       corpus=None,
                                       score_policy: str = "new-blocks",
                                       rare_branch_schedule: str = "auto",
                                       target_uncovered: str = "auto") -> list:
    groups = {}
    for candidate in candidates:
        if _candidate_meta(candidate).get("kind") != "value_profile":
            continue
        groups.setdefault(_candidate_group_key(candidate), []).append(candidate)

    compounds = []
    for group in groups.values():
        group = sorted(
            group,
            key=lambda candidate: _coverage_candidate_sort_key(
                candidate,
                corpus=corpus,
                score_policy=score_policy,
                rare_branch_schedule=rare_branch_schedule,
                target_uncovered=target_uncovered,
            ),
        )
        chunk = []
        used = {}
        for candidate in group:
            updates = _candidate_updates(candidate)
            if not updates:
                continue
            conflict = any(
                name in used and used[name] != value
                for name, value in updates.items()
            )
            if conflict and chunk:
                compound = _make_compound_candidate(base_inputs, chunk)
                if compound is not None:
                    compounds.append(compound)
                chunk = []
                used = {}
            if any(
                name in used and used[name] != value
                for name, value in updates.items()
            ):
                continue
            chunk.append(candidate)
            used.update(updates)
            if len(used) >= 256:
                compound = _make_compound_candidate(base_inputs, chunk)
                if compound is not None:
                    compounds.append(compound)
                chunk = []
                used = {}
        if chunk:
            compound = _make_compound_candidate(base_inputs, chunk)
            if compound is not None:
                compounds.append(compound)

    compounds.sort(
        key=lambda candidate: _coverage_candidate_sort_key(
            candidate,
            corpus=corpus,
            score_policy=score_policy,
            rare_branch_schedule=rare_branch_schedule,
            target_uncovered=target_uncovered,
        )
    )
    return compounds[:children_per_trace]


def _build_generational_children(base_inputs, candidates: list, *,
                                 children_per_trace: int,
                                 objective_policy: str,
                                 corpus=None,
                                 score_policy: str = "new-blocks",
                                 state_pruning: str = "off",
                                 rare_branch_schedule: str = "auto",
                                 target_uncovered: str = "auto") -> tuple[list, dict]:
    enabled = {
        item.strip()
        for item in str(objective_policy or "").split(",")
        if item.strip()
    }
    sort_key = lambda candidate: _coverage_candidate_sort_key(
        candidate,
        corpus=corpus,
        score_policy=score_policy,
        rare_branch_schedule=rare_branch_schedule,
        target_uncovered=target_uncovered,
    )
    singles = sorted(candidates, key=sort_key)
    children = []
    stats = {
        "children_symbolic_candidates": len(candidates),
        "children_compound_candidates": 0,
        "children_single_candidates": 0,
        "compatible_states_pruned": 0,
    }
    seen_compatible = set()

    for candidate in singles:
        kind = _candidate_meta(candidate).get("kind")
        if kind == "value_profile":
            continue
        if kind == "branch_target" and "branch" not in enabled:
            continue
        if kind in {"requires_repair", "assert_repair"} and "concrete-blocker" not in enabled:
            continue
        if kind == "distance_guided" and "branch-distance" not in enabled:
            continue
        if state_pruning == "compatible-branch":
            meta = _candidate_meta(candidate)
            compatible_key = (
                meta.get("source_block"),
                meta.get("target_block"),
                meta.get("kind"),
                tuple(sorted(_candidate_updates(candidate).items())),
            )
            if compatible_key in seen_compatible:
                stats["compatible_states_pruned"] += 1
                continue
            seen_compatible.add(compatible_key)
        children.append(candidate)
        stats["children_single_candidates"] += 1
        if len(children) >= children_per_trace:
            return children, stats

    if "compound-value-profile" in enabled:
        compounds = _compound_value_profile_candidates(
            base_inputs,
            singles,
            children_per_trace=children_per_trace - len(children),
            corpus=corpus,
            score_policy=score_policy,
            rare_branch_schedule=rare_branch_schedule,
            target_uncovered=target_uncovered,
        )
        children.extend(compounds)
        stats["children_compound_candidates"] = len(compounds)
        if len(children) >= children_per_trace:
            return children[:children_per_trace], stats

    if "value-profile" in enabled:
        for candidate in singles:
            if _candidate_meta(candidate).get("kind") != "value_profile":
                continue
            children.append(candidate)
            stats["children_single_candidates"] += 1
            if len(children) >= children_per_trace:
                break

    return children[:children_per_trace], stats


def _score_replayed_child(result: dict, new_blocks: set, kind: str,
                          score_policy: str, *, corpus=None,
                          path_features: dict | None = None,
                          new_feature_count: int | None = None) -> int:
    if result.get("status") != "ok":
        return 0
    base = len(new_blocks) * 1000
    if score_policy == "coverage-max" and corpus is not None:
        if new_feature_count is None:
            features = coverage_features(
                set(result.get("covered") or ()),
                path_features,
                metric=corpus.coverage_metric,
            )
            known = corpus.coverage_objectives(corpus.coverage_metric)
            new_feature_count = len(features - known)
        base += int(new_feature_count) * 250
        if not new_blocks and new_feature_count:
            base += 400
    if kind == "coverage":
        base += 500
    elif kind == "path":
        base += 300
    elif kind == "trace":
        base += 100
    return base


def run_coverage_generational(corpus, executor: CoverageExecutor, out_dir: Path,
                              *, loop_bound: int = 8,
                              max_path_depth: int = 512,
                              max_solver_queries: int = 10000,
                              solver_timeout_ms: int = 100,
                              havoc_bound: int = 8,
                              max_states: int = 256,
                              path_priority: str = "coverage-directed",
                              value_profile_policy: str = "late",
                              branch_distance_policy: str = "auto",
                              profile: str = "max-coverage",
                              seed_limit: int | None = None,
                              jobs: int = 1,
                              max_generations: int = 3,
                              children_per_trace: int = 256,
                              objective_policy: str = (
                                  "branch,compound-value-profile,value-profile"
                              ),
                              score_policy: str = "new-blocks",
                              state_pruning: str = "off",
                              rare_branch_schedule: str = "auto",
                              target_uncovered: str = "auto",
                              debug_candidates: bool = False,
                              debug_logger=None) -> tuple[int, dict]:
    """Run SAGE-style replay-scored symbolic generations."""
    debug = debug_logger or DebugLogger.disabled()
    gen_idx = _next_gen_idx(out_dir)
    trace_idx = _next_trace_idx(out_dir)
    new_count = 0
    stats_total = {
        "max_generations": int(max_generations),
        "children_per_trace": int(children_per_trace),
        "objective_policy": objective_policy,
        "score_policy": score_policy,
        "state_pruning": state_pruning,
        "rare_branch_schedule": rare_branch_schedule,
        "target_uncovered": target_uncovered,
    }
    replay_diagnostics = _new_path_diagnostics()
    entries = list(corpus.entries) + list(corpus.assertion_feedback_entries)
    selected, effective_seed_limit = _select_path_entries(
        corpus, entries, profile=profile, seed_limit=seed_limit, jobs=jobs,
        score_policy=score_policy,
        rare_branch_schedule=rare_branch_schedule,
        target_uncovered=target_uncovered)
    stats_total["available_seed_entries"] = len(entries)
    stats_total["selected_seed_entries"] = len(selected)
    stats_total["path_seed_limit"] = effective_seed_limit
    if not selected:
        return 0, stats_total

    print(f"[coverage-generational] scheduling {len(selected)}/{len(entries)} "
          f"seeds (limit={effective_seed_limit}, jobs={jobs})")
    worklist = [
        {
            "entry": entry,
            "entry_idx": entry_idx,
            "score": int(score),
            "generation": 0,
        }
        for entry_idx, entry, score in selected
    ]
    remaining_queries = max_solver_queries
    remaining_states = max_states
    generations = []

    for generation in range(int(max_generations)):
        if not worklist or remaining_queries <= 0 or remaining_states <= 0:
            break
        worklist.sort(key=lambda item: (-int(item.get("score", 0)), item["entry_idx"]))
        generation_items = worklist[:max(1, min(len(worklist), effective_seed_limit or len(worklist)))]
        worklist = worklist[len(generation_items):]
        generation_report = {
            "generation": generation + 1,
            "expanded_traces": len(generation_items),
            "children_generated": 0,
            "children_evaluated": 0,
            "children_pruned": 0,
            "replay_batches": 0,
            "max_replay_batch": 0,
            "new_inputs": 0,
            "blocks_added": 0,
            "best_score": 0,
        }
        before_blocks = len(corpus.covered)

        for item_idx, item in enumerate(generation_items):
            entry = item["entry"]
            if remaining_queries <= 0 or remaining_states <= 0:
                break
            traces_left = max(1, len(generation_items) - item_idx)
            query_budget = max(1, remaining_queries // traces_left)
            state_budget = max(1, remaining_states // traces_left)
            symbolic_explore = getattr(
                executor.evaluator, "symbolic_explore_lazy", None)
            if symbolic_explore is None:
                symbolic_explore = executor.evaluator.symbolic_explore
            candidates, stats = symbolic_explore(
                entry.inputs,
                f"coverage_gen_seed_{item['entry_idx']}_g{generation}",
                corpus.covered,
                loop_bound=loop_bound,
                max_path_depth=max_path_depth,
                max_solver_queries=query_budget,
                solver_timeout_ms=solver_timeout_ms,
                havoc_bound=havoc_bound,
                max_states=state_budget,
                path_priority=path_priority,
                value_profile_policy=value_profile_policy,
                branch_distance_policy=branch_distance_policy,
            )
            _merge_stats(stats_total, stats)
            remaining_queries = max(
                0, remaining_queries - int(stats.get("solver_queries", 0) or 0))
            remaining_states = max(
                0, remaining_states - int(stats.get("states_explored", 0) or 0))

            children, child_stats = _build_generational_children(
                entry.inputs,
                candidates,
                children_per_trace=int(children_per_trace),
                objective_policy=objective_policy,
                corpus=corpus,
                score_policy=score_policy,
                state_pruning=state_pruning,
                rare_branch_schedule=rare_branch_schedule,
                target_uncovered=target_uncovered,
            )
            _merge_stats(stats_total, child_stats)
            generation_report["children_generated"] += len(children)

            replay_tasks = []
            replay_meta = []
            label = f"coverage_gen_g{generation + 1}"
            for child_idx, candidate in enumerate(children):
                if _should_prune_replay(replay_diagnostics, candidate):
                    _record_pruned_replay(replay_diagnostics, candidate)
                    generation_report["children_pruned"] += 1
                    continue
                materialized = _materialize_replay_candidate(candidate)
                replay_tasks.append((
                    len(replay_tasks),
                    materialized,
                    f"{label}_{item['entry_idx']}_{child_idx}",
                ))
                replay_meta.append(materialized)

            if replay_tasks:
                generation_report["replay_batches"] += 1
                generation_report["max_replay_batch"] = max(
                    generation_report["max_replay_batch"],
                    len(replay_tasks),
                )
            for candidate, result in zip(
                replay_meta,
                executor.evaluate_many(replay_tasks),
            ):
                covered = set(result.get("covered") or [])
                path_features = _result_path_features(result, covered)
                new_blocks = covered - corpus.covered
                new_feature_count = len(
                    coverage_features(
                        covered,
                        path_features,
                        metric=corpus.coverage_metric,
                    )
                    - corpus.coverage_objectives(corpus.coverage_metric)
                )
                _record_path_diagnostic(
                    replay_diagnostics, candidate, result, new_blocks)
                kind, gen_idx, trace_idx = _record_candidate_result(
                    corpus, out_dir, candidate, covered,
                    entry.params_line or corpus.params_line,
                    gen_idx, trace_idx, "coverage-generational",
                    status=result.get("status", "ok"),
                    violation_pc=result.get("violation_pc"),
                    violation_block=result.get("violation_block"),
                    path_features=path_features,
                    debug_candidates=debug_candidates,
                    debug_logger=debug,
                )
                generation_report["children_evaluated"] += 1
                score = _score_replayed_child(
                    result, new_blocks, str(kind or ""), score_policy,
                    corpus=corpus, path_features=path_features,
                    new_feature_count=new_feature_count)
                generation_report["best_score"] = max(
                    generation_report["best_score"], score)
                if kind == "coverage":
                    new_count += 1
                    generation_report["new_inputs"] += 1
                if kind in {"coverage", "path", "trace"} and corpus.entries:
                    worklist.append({
                        "entry": corpus.entries[-1],
                        "entry_idx": len(corpus.entries) - 1,
                        "score": score,
                        "generation": generation + 1,
                    })

        generation_report["blocks_added"] = len(corpus.covered) - before_blocks
        generations.append(generation_report)
        print(f"[coverage-generational] generation {generation + 1} | "
              f"expanded={generation_report['expanded_traces']} "
              f"children={generation_report['children_generated']} "
              f"evals={generation_report['children_evaluated']} "
              f"new_inputs={generation_report['new_inputs']} "
              f"blocks_added={generation_report['blocks_added']} "
              f"best_score={generation_report['best_score']}")
        if generation_report["blocks_added"] == 0 and not worklist:
            break

    _merge_stats(stats_total, replay_diagnostics)
    stats_total["generations"] = generations
    stats_total["generations_completed"] = len(generations)
    stats_total["path_priority"] = path_priority
    stats_total["value_profile_policy"] = value_profile_policy
    stats_total["branch_distance_policy"] = branch_distance_policy
    stats_total["remaining_solver_queries"] = remaining_queries
    stats_total["remaining_symbolic_states"] = remaining_states
    stats_total["replay_batches"] = sum(
        int(item.get("replay_batches", 0)) for item in generations)
    stats_total["max_replay_batch"] = max(
        (int(item.get("max_replay_batch", 0)) for item in generations),
        default=0,
    )
    stats_total["coverage_objectives_reached"] = len(
        corpus.coverage_objectives(corpus.coverage_metric))
    stats_total.setdefault("dominated_states_pruned", 0)
    return new_count, stats_total


def _record_seed_assertion_feedback(corpus, path: Path, inputs, covered: set,
                                    params_line: str, result: dict,
                                    *, debug_candidates: bool = False,
                                    debug_logger=None):
    debug = debug_logger or DebugLogger.disabled()
    path_features = _result_path_features(result, covered)
    pc = result.get("violation_pc")
    pc_key = str(pc) if pc is not None else "unknown"
    assertion_sig = (
        pc_key,
        corpus.trace_signature(inputs, covered, path_features),
    )
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
                                  params_line=params_line, source="seed",
                                  path_features=path_features)
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
        path_features=_result_path_features(result, result["covered"]),
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
                              debug_logger=None,
                              diagnostics: dict | None = None):
    debug = debug_logger or DebugLogger.disabled()
    debug.event("candidate", "candidate_batch_start",
                count=len(candidates))
    coverage_new = 0
    trace_new = 0

    def handle_result(idx: int, candidate, params_line: str,
                      label: str, result: dict):
        nonlocal gen_idx, trace_idx, coverage_new, trace_new
        covered = set(result.get("covered") or [])
        new_blocks = covered - corpus.covered
        path_features = _result_path_features(result, covered)
        _record_path_diagnostic(diagnostics, candidate, result, new_blocks)
        kind, gen_idx, trace_idx = _record_candidate_result(
            corpus, out_dir, candidate, covered,
            params_line, gen_idx, trace_idx, label,
            debug_candidates=debug_candidates,
            status=result.get("status", "ok"),
            violation_pc=result.get("violation_pc"),
            violation_block=result.get("violation_block"),
            path_features=path_features,
            debug_logger=debug,
        )
        if kind == "coverage":
            coverage_new += 1
        elif kind in {"path", "trace"}:
            trace_new += 1

    if not executor.parallel_enabled:
        for idx, (candidate, params_line, label) in enumerate(candidates):
            if _should_prune_replay(diagnostics, candidate):
                _record_pruned_replay(diagnostics, candidate)
                continue
            result = executor.evaluate_many([
                (idx, candidate, f"{label}_{idx}")
            ])
            handle_result(
                idx, candidate, params_line, label,
                result[0] if result else {"covered": set()},
            )
    else:
        tasks = []
        task_meta = {}
        for idx, (candidate, params_line, label) in enumerate(candidates):
            if _should_prune_replay(diagnostics, candidate):
                _record_pruned_replay(diagnostics, candidate)
                continue
            task_meta[idx] = (candidate, params_line, label)
            tasks.append((idx, candidate, f"{label}_{idx}"))
        results = executor.evaluate_many(tasks)
        by_index = {result["index"]: result for result in results}
        for idx, (candidate, params_line, label) in task_meta.items():
            result = by_index.get(idx, {"covered": set()})
            handle_result(idx, candidate, params_line, label, result)

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
        path_features = _result_path_features(result, covered)
        status = result.get("status", "ok")
        if status == "assert_violation":
            _record_seed_assertion_feedback(
                corpus, path, inputs, covered, params_line, result,
                debug_candidates=debug_candidates,
                debug_logger=debug)
            continue
        if status == "assume_violation":
            corpus.invalid_inputs += 1
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
                        invalid_input=True,
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
            path_features=path_features,
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
                rare_branch_schedule: str = "auto",
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
            entry = (
                corpus.pick_rare()
                if rare_branch_schedule != "off"
                else corpus.pick()
            )
            if entry is None:
                break

            # 10% splice, 90% mutation
            if len(corpus.entries) > 1 and random.random() < 0.1:
                other = (
                    corpus.pick_rare()
                    if rare_branch_schedule != "off"
                    else corpus.pick()
                )
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
        elif key == "first_new_block_depth" and value is not None:
            if total.get(key) is None:
                total[key] = value
        elif isinstance(value, int):
            total[key] = int(total.get(key, 0)) + value
        elif isinstance(value, dict):
            if key not in total or not isinstance(total.get(key), dict):
                total[key] = {}
            _merge_stats(total[key], value)
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
    if executor.engine == "native":
        return "native-rust"
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


def _rank_path_entries(corpus, entries: list, *,
                       score_policy: str = "new-blocks",
                       rare_branch_schedule: str = "auto",
                       target_uncovered: str = "auto") -> list[tuple[int, object, float]]:
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
        if score_policy == "coverage-max":
            score += len(coverage_features(
                entry.covered_blocks,
                entry.path_features,
                metric=corpus.coverage_metric,
            )) * 0.01
        if rare_branch_schedule != "off":
            score += corpus.entry_rarity(entry, corpus.coverage_metric)
        if target_uncovered != "off":
            score += len(entry.path_features.get("context_edges", ())) * 0.001
        ranked.append((idx, entry, score))

    ranked.sort(key=lambda item: (
        -item[2],
        str(item[1].path or ""),
        item[0],
    ))
    return ranked


def _select_path_entries(corpus, entries: list, *, profile: str,
                         seed_limit: int | None, jobs: int,
                         score_policy: str = "new-blocks",
                         rare_branch_schedule: str = "auto",
                         target_uncovered: str = "auto") -> tuple[list, int]:
    ranked = _rank_path_entries(
        corpus,
        entries,
        score_policy=score_policy,
        rare_branch_schedule=rare_branch_schedule,
        target_uncovered=target_uncovered,
    )
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
                 path_priority: str = "coverage-directed",
                 value_profile_policy: str = "late",
                 branch_distance_policy: str = "auto",
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
    replay_diagnostics = _new_path_diagnostics()
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
                    "path_priority": path_priority,
                    "value_profile_policy": value_profile_policy,
                    "branch_distance_policy": branch_distance_policy,
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
            debug_logger=debug_logger, diagnostics=replay_diagnostics)
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
                path_priority=path_priority,
                value_profile_policy=value_profile_policy,
                branch_distance_policy=branch_distance_policy,
            )
            _merge_stats(stats_total, stats)
            for candidate in candidates:
                result = _run_eval(evaluator, candidate, f"gen_{gen_idx}")
                covered = set(result.get("covered") or [])
                _record_path_diagnostic(
                    replay_diagnostics, candidate, result,
                    covered - corpus.covered,
                )
                kind, gen_idx, trace_idx = _record_candidate_result(
                    corpus, out_dir, candidate, covered,
                    entry.params_line or corpus.params_line,
                    gen_idx, trace_idx, "symbolic",
                    status=result.get("status", "ok"),
                    violation_pc=result.get("violation_pc"),
                    violation_block=result.get("violation_block"),
                    debug_logger=debug_logger,
                )
                if kind == "coverage":
                    new_count += 1

    _merge_stats(stats_total, replay_diagnostics)
    stats_total["path_priority"] = path_priority
    stats_total["value_profile_policy"] = value_profile_policy
    stats_total["branch_distance_policy"] = branch_distance_policy
    return new_count, stats_total


def run_concolic(corpus, evaluator, out_dir: Path, *,
                 loop_bound: int = 8,
                 max_path_depth: int = 512,
                 max_solver_queries: int = 10000,
                 solver_timeout_ms: int = 100,
                 havoc_bound: int = 8,
                 branch_distance_policy: str = "auto",
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
    replay_diagnostics = _new_path_diagnostics()
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
                    "branch_distance_policy": branch_distance_policy,
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
            debug_logger=debug_logger, diagnostics=replay_diagnostics)
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
                branch_distance_policy=branch_distance_policy,
            )
            _merge_stats(stats_total, stats)
            for candidate in candidates:
                result = _run_eval(evaluator, candidate, f"gen_{gen_idx}")
                covered = set(result.get("covered") or [])
                _record_path_diagnostic(
                    replay_diagnostics, candidate, result,
                    covered - corpus.covered,
                )
                kind, gen_idx, trace_idx = _record_candidate_result(
                    corpus, out_dir, candidate, covered,
                    entry.params_line or corpus.params_line,
                    gen_idx, trace_idx, "concolic",
                    status=result.get("status", "ok"),
                    violation_pc=result.get("violation_pc"),
                    violation_block=result.get("violation_block"),
                    debug_logger=debug_logger,
                )
                if kind == "coverage":
                    new_count += 1

    _merge_stats(stats_total, replay_diagnostics)
    stats_total["branch_distance_policy"] = branch_distance_policy
    return new_count, stats_total


def _coverage_dict(program, covered_blocks: set) -> dict:
    try:
        from interpreter.runner import compute_coverage
        cov = compute_coverage(program, set(covered_blocks)) or {}
        return {k: v for k, v in cov.items() if k != "explored_blocks"}
    except Exception:
        return {}


def _rate(numerator: int | float, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(float(numerator) / seconds, 3)


def _add_phase_rates(phase_timings: list[dict]) -> None:
    for phase in phase_timings:
        seconds = float(phase.get("seconds") or 0.0)
        if "blocks_added" in phase:
            phase["blocks_per_sec"] = _rate(phase.get("blocks_added", 0), seconds)
        if "new_inputs" in phase:
            phase["new_inputs_per_sec"] = _rate(phase.get("new_inputs", 0), seconds)
        if "trace_added" in phase:
            phase["trace_inputs_per_sec"] = _rate(phase.get("trace_added", 0), seconds)


def _coverage_rate_dict(initial: dict, final: dict, total_seconds: float, *,
                        candidates_evaluated: int, total_new_inputs: int,
                        useful_inputs: int) -> dict:
    initial_blocks = int(initial.get("blocks_covered", 0) or 0)
    final_blocks = int(final.get("blocks_covered", 0) or 0)
    initial_reachable = int(initial.get("reachable_blocks_covered", initial_blocks) or 0)
    final_reachable = int(final.get("reachable_blocks_covered", final_blocks) or 0)
    blocks_added = max(0, final_blocks - initial_blocks)
    reachable_added = max(0, final_reachable - initial_reachable)
    return {
        "total_seconds": round(total_seconds, 6),
        "initial_blocks": initial_blocks,
        "final_blocks": final_blocks,
        "blocks_added": blocks_added,
        "initial_reachable_blocks": initial_reachable,
        "final_reachable_blocks": final_reachable,
        "reachable_blocks_added": reachable_added,
        "blocks_per_sec": _rate(blocks_added, total_seconds),
        "reachable_blocks_per_sec": _rate(reachable_added, total_seconds),
        "candidates_per_sec": _rate(candidates_evaluated, total_seconds),
        "new_inputs_per_sec": _rate(total_new_inputs, total_seconds),
        "useful_inputs_per_sec": _rate(useful_inputs, total_seconds),
    }


def _write_report(out_dir: Path, report: dict):
    out_path = out_dir / "coverage_gen.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out_path


def _reduce_corpus_coverage(corpus, out_dir: Path, *,
                            policy: str = "none") -> dict:
    """Greedily select a small concrete set preserving covered objectives."""
    if policy == "none":
        return {
            "corpus_reduce": "none",
            "minimized_input_count": 0,
            "coverage_objectives_total": len(
                corpus.coverage_objectives(corpus.coverage_metric)),
            "coverage_objectives_reached": 0,
            "coverage_marginal_by_input": [],
        }
    if policy != "coverage-setcover":
        raise ValueError(f"unknown corpus reduction policy: {policy}")

    from interpreter.coverage_gen.writer import write_input_file

    entries = list(corpus.entries)
    objective_by_idx = [
        coverage_features(
            entry.covered_blocks,
            entry.path_features,
            metric=corpus.coverage_metric,
        )
        for entry in entries
    ]
    uncovered = set().union(*objective_by_idx) if objective_by_idx else set()
    selected = []
    covered = set()
    while uncovered:
        best_idx = None
        best_gain = set()
        for idx, objectives in enumerate(objective_by_idx):
            if idx in selected:
                continue
            gain = objectives - covered
            if (
                best_idx is None
                or len(gain) > len(best_gain)
                or (
                    len(gain) == len(best_gain)
                    and str(entries[idx].path or "") < str(entries[best_idx].path or "")
                )
            ):
                best_idx = idx
                best_gain = gain
        if best_idx is None or not best_gain:
            break
        selected.append(best_idx)
        covered |= objective_by_idx[best_idx]
        uncovered -= best_gain

    min_dir = out_dir / "minimized"
    min_dir.mkdir(parents=True, exist_ok=True)
    marginal = []
    running = set()
    for out_idx, entry_idx in enumerate(selected):
        entry = entries[entry_idx]
        objectives = objective_by_idx[entry_idx]
        gain = objectives - running
        running |= objectives
        out_path = min_dir / f"min_{out_idx}.input"
        out_path.write_text(write_input_file(entry.inputs, entry.params_line))
        marginal.append({
            "input": str(out_path),
            "source": entry.source,
            "interesting_reason": entry.interesting_reason,
            "marginal_objectives": len(gain),
            "total_objectives_after": len(running),
        })

    return {
        "corpus_reduce": policy,
        "minimized_input_count": len(selected),
        "coverage_objectives_total": len(set().union(*objective_by_idx))
        if objective_by_idx else 0,
        "coverage_objectives_reached": len(running),
        "coverage_marginal_by_input": marginal,
    }


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
                             path_priority: str | None = None,
                             value_profile_policy: str | None = None,
                             branch_distance_policy: str | None = None,
                             max_generations: int | None = None,
                             children_per_trace: int | None = None,
                             objective_policy: str | None = None,
                             score_policy: str | None = None,
                             state_pruning: str | None = None,
                             rare_branch_schedule: str | None = None,
                             target_uncovered: str | None = None,
                             corpus_reduce: str | None = None,
                             coverage_metric: str = "block",
                             solver_timeout_ms: int | None = None,
                             max_steps_per_input: int | None = (
                                 DEFAULT_MAX_STEPS_PER_INPUT
                             ),
                             debug_logger=None) -> dict:
    """Run the unified deterministic/concolic/greybox input pipeline."""
    from interpreter.coverage_gen.corpus import Corpus

    mode, profile = _resolve_mode_profile(mode, profile)
    coverage_metric = str(coverage_metric or "block")
    if coverage_metric not in {"block", "edge", "context-edge", "branch"}:
        raise ValueError(f"unknown coverage metric: {coverage_metric}")
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
        "path_priority": path_priority,
        "value_profile_policy": value_profile_policy,
        "branch_distance_policy": branch_distance_policy,
        "max_generations": max_generations,
        "children_per_trace": children_per_trace,
        "objective_policy": objective_policy,
        "score_policy": score_policy,
        "state_pruning": state_pruning,
        "rare_branch_schedule": rare_branch_schedule,
        "target_uncovered": target_uncovered,
        "corpus_reduce": corpus_reduce,
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
    path_priority = str(cfg.get("path_priority", "coverage-directed"))
    value_profile_policy = str(cfg.get("value_profile_policy", "late"))
    branch_distance_policy = str(cfg.get("branch_distance_policy", "auto"))
    max_generations = int(cfg.get("max_generations", 1))
    children_per_trace = int(cfg.get("children_per_trace", 64))
    objective_policy = str(cfg.get(
        "objective_policy",
        "branch,compound-value-profile,value-profile",
    ))
    score_policy = str(cfg.get("score_policy", "new-blocks"))
    state_pruning = str(cfg.get("state_pruning", "off"))
    rare_branch_schedule = str(cfg.get("rare_branch_schedule", "auto"))
    target_uncovered = str(cfg.get("target_uncovered", "auto"))
    corpus_reduce = str(cfg.get("corpus_reduce", "none"))
    solver_timeout_ms = int(cfg["solver_timeout_ms"])
    max_steps_per_input = max(0, int(
        DEFAULT_MAX_STEPS_PER_INPUT
        if max_steps_per_input is None else max_steps_per_input
    ))

    debug = (debug_logger or DebugLogger.disabled()).bind(
        component="coverage-driver", test_name=test_name)

    random.seed(rng_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = CoverageExecutor(program, test_name, timeout=timeout,
                                engine=engine, jobs=jobs,
                                max_steps_per_input=max_steps_per_input,
                                debug_logger=debug)
    corpus = Corpus(coverage_metric=coverage_metric)
    phase_timings = []
    deterministic_new = 0
    concolic_new = 0
    concolic_stats = {}
    symbolic_new = 0
    symbolic_stats = {}
    coverage_generational_new = 0
    coverage_generational_stats = {}
    greybox_new = 0
    round_reports = []
    phase_decisions = []
    stall_reason = None
    n_seeds = 0
    seed_covered: set[str] = set()
    t_total = time.perf_counter()

    print("[driver] gen-input pipeline")
    print(f"[driver] target={test_name} mode={mode} profile={profile} "
          f"path_engine={path_engine} engine={engine} "
          f"coverage_metric={coverage_metric} rng_seed={rng_seed}")
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
          f"path_priority={path_priority}, "
          f"value_profile_policy={value_profile_policy}, "
          f"branch_distance_policy={branch_distance_policy}, "
          f"max_generations={max_generations}, "
          f"children_per_trace={children_per_trace}, "
          f"objective_policy={objective_policy}, "
          f"score_policy={score_policy}, "
          f"state_pruning={state_pruning}, "
          f"rare_branch_schedule={rare_branch_schedule}, "
          f"target_uncovered={target_uncovered}, "
          f"corpus_reduce={corpus_reduce}, "
          f"solver_timeout_ms={solver_timeout_ms}, "
          f"max_steps_per_input={max_steps_per_input}")
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
        coverage_metric=coverage_metric,
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
            "path_priority": path_priority,
            "value_profile_policy": value_profile_policy,
            "branch_distance_policy": branch_distance_policy,
            "max_generations": max_generations,
            "children_per_trace": children_per_trace,
            "objective_policy": objective_policy,
            "score_policy": score_policy,
            "state_pruning": state_pruning,
            "rare_branch_schedule": rare_branch_schedule,
            "target_uncovered": target_uncovered,
            "corpus_reduce": corpus_reduce,
            "solver_timeout_ms": solver_timeout_ms,
            "max_steps_per_input": max_steps_per_input,
            "coverage_metric": coverage_metric,
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
        seed_covered = set(corpus.covered)

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
            remaining_generational_queries = max_solver_queries
            remaining_generational_states = max_symbolic_states
            symbolic_disabled_reason = None
            concolic_disabled_reason = None
            concolic_no_yield_rounds = 0
            for round_idx in range(rounds):
                round_symbolic = 0
                round_concolic = 0
                round_coverage_generational = 0
                round_greybox = 0
                before_blocks = len(corpus.covered)
                before_trace = corpus.trace_interesting_inputs
                before_path = corpus.generated_path_inputs
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
                        path_priority=path_priority,
                        value_profile_policy=value_profile_policy,
                        branch_distance_policy=branch_distance_policy,
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
                        "branch_distance_objectives": stats.get(
                            "branch_distance_objectives", 0),
                        "branch_distance_candidates": stats.get(
                            "branch_distance_candidates", 0),
                        "value_profile_deferred": stats.get(
                            "value_profile_deferred", 0),
                        "candidate_kind_summary": stats.get(
                            "candidate_kind_summary", {}),
                    })
                    print(f"[driver] symbolic round {round_idx + 1} done in "
                          f"{elapsed:.2f}s | evals={evaluated} "
                          f"new_inputs={round_symbolic} "
                          f"states={stats.get('states_explored', 0)} "
                          f"unknown={stats.get('paths_unknown', 0)} "
                          f"solver_queries={stats.get('solver_queries', 0)} "
                          f"cache_hits={stats.get('solver_cache_hits', 0)} "
                          f"frontier={stats.get('frontier_ranked_objectives', 0)} "
                          f"value_profile={stats.get('value_profile_candidates', 0)} "
                          f"branch_distance={stats.get('branch_distance_candidates', 0)} "
                          f"deferred={stats.get('value_profile_deferred', 0)} "
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
                    path_engine == "coverage-generational"
                    and remaining_generational_queries > 0
                    and remaining_generational_states > 0
                ):
                    print(f"[driver] Round {round_idx + 1}/{rounds}: "
                          f"coverage-generational "
                          f"(generations={max_generations}, "
                          f"children={children_per_trace}, "
                          f"depth={max_path_depth}, "
                          f"states={remaining_generational_states}, "
                          f"solver_queries={remaining_generational_queries}) ...")
                    t0 = time.perf_counter()
                    before_runs = executor.runs
                    round_coverage_generational, stats = run_coverage_generational(
                        corpus, executor, out_dir,
                        loop_bound=loop_bound,
                        max_path_depth=max_path_depth,
                        max_solver_queries=remaining_generational_queries,
                        solver_timeout_ms=solver_timeout_ms,
                        havoc_bound=havoc_bound,
                        max_states=remaining_generational_states,
                        path_priority=path_priority,
                        value_profile_policy=value_profile_policy,
                        branch_distance_policy=branch_distance_policy,
                        profile=profile,
                        seed_limit=path_seed_limit,
                        jobs=executor.jobs,
                        max_generations=max_generations,
                        children_per_trace=children_per_trace,
                        objective_policy=objective_policy,
                        score_policy=score_policy,
                        state_pruning=state_pruning,
                        rare_branch_schedule=rare_branch_schedule,
                        target_uncovered=target_uncovered,
                        debug_candidates=debug_candidates,
                        debug_logger=debug,
                    )
                    elapsed = time.perf_counter() - t0
                    evaluated = executor.runs - before_runs
                    _merge_stats(coverage_generational_stats, stats)
                    coverage_generational_new += round_coverage_generational
                    remaining_generational_queries = max(
                        0,
                        max_solver_queries
                        - int(coverage_generational_stats.get("solver_queries", 0)),
                    )
                    remaining_generational_states = max(
                        0,
                        max_symbolic_states
                        - int(coverage_generational_stats.get("states_explored", 0)),
                    )
                    phase_timings.append({
                        "phase": "coverage-generational",
                        "round": round_idx + 1,
                        "seconds": round(elapsed, 6),
                        "evaluated": evaluated,
                        "evals_per_sec": round(evaluated / elapsed, 3) if elapsed else evaluated,
                        "new_inputs": round_coverage_generational,
                        "states_explored": stats.get("states_explored", 0),
                        "solver_queries": stats.get("solver_queries", 0),
                        "generations_completed": stats.get(
                            "generations_completed", 0),
                        "children_symbolic_candidates": stats.get(
                            "children_symbolic_candidates", 0),
                        "children_compound_candidates": stats.get(
                            "children_compound_candidates", 0),
                        "branch_distance_objectives": stats.get(
                            "branch_distance_objectives", 0),
                        "branch_distance_candidates": stats.get(
                            "branch_distance_candidates", 0),
                        "candidate_kind_summary": stats.get(
                            "candidate_kind_summary", {}),
                    })
                    print(f"[driver] coverage-generational round "
                          f"{round_idx + 1} done in {elapsed:.2f}s | "
                          f"evals={evaluated} "
                          f"new_inputs={round_coverage_generational} "
                          f"states={stats.get('states_explored', 0)} "
                          f"solver_queries={stats.get('solver_queries', 0)} "
                          f"branch_distance={stats.get('branch_distance_candidates', 0)} "
                          f"compounds={stats.get('children_compound_candidates', 0)} "
                          f"pruned={stats.get('pruned_candidates', 0)}")

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
                        branch_distance_policy=branch_distance_policy,
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
                        "branch_distance_objectives": stats.get(
                            "branch_distance_objectives", 0),
                        "branch_distance_candidates": stats.get(
                            "branch_distance_candidates", 0),
                        "candidate_kind_summary": stats.get(
                            "candidate_kind_summary", {}),
                    })
                    print(f"[driver] concolic round {round_idx + 1} done in "
                          f"{elapsed:.2f}s | evals={evaluated} "
                          f"new_inputs={round_concolic} "
                          f"solver_queries={stats.get('solver_queries', 0)} "
                          f"cache_hits={stats.get('solver_cache_hits', 0)} "
                          f"frontier={stats.get('frontier_ranked_objectives', 0)} "
                          f"value_profile={stats.get('value_profile_candidates', 0)} "
                          f"branch_distance={stats.get('branch_distance_candidates', 0)} "
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
                        rare_branch_schedule=rare_branch_schedule,
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
                path_added = corpus.generated_path_inputs - before_path
                round_elapsed = time.perf_counter() - round_t0
                round_evals = executor.runs - round_start_runs
                round_reports.append({
                    "round": round_idx + 1,
                    "seconds": round(round_elapsed, 6),
                    "evaluated": round_evals,
                    "symbolic_new_inputs": round_symbolic,
                    "coverage_generational_new_inputs": round_coverage_generational,
                    "concolic_new_inputs": round_concolic,
                    "greybox_new_inputs": round_greybox,
                    "generated_path_inputs": corpus.generated_path_inputs,
                    "path_interesting_added": path_added,
                    "blocks_added": blocks_added,
                    "trace_interesting_added": trace_added,
                    "blocks_covered": len(corpus.covered),
                    "corpus_entries": len(corpus.entries),
                })
                print(f"[driver] round {round_idx + 1} summary | "
                      f"blocks_added={blocks_added} "
                      f"path_added={path_added} "
                      f"trace_added={trace_added} "
                      f"corpus={len(corpus.entries)} "
                      f"covered={len(corpus.covered)} "
                      f"timeouts={executor.timeouts} "
                      f"step_limits={executor.step_limits} "
                      f"errors={executor.errors}")

                if blocks_added == 0 and path_added == 0:
                    stall_reason = "no-new-coverage-round"
                    break

            if stall_reason is None:
                if remaining_iters <= 0 and path_engine in ("hybrid", "fuzz"):
                    stall_reason = "iteration-budget-exhausted"
                elif remaining_symbolic_states <= 0 and path_engine in ("hybrid", "symbolic"):
                    stall_reason = "symbolic-state-budget-exhausted"
                elif remaining_generational_states <= 0 and path_engine == "coverage-generational":
                    stall_reason = "coverage-generational-state-budget-exhausted"
                elif remaining_solver_queries <= 0 and path_engine in ("hybrid", "symbolic", "concolic"):
                    stall_reason = "solver-query-budget-exhausted"
                elif remaining_generational_queries <= 0 and path_engine == "coverage-generational":
                    stall_reason = "coverage-generational-solver-budget-exhausted"
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
        or coverage_generational_stats.get("complete_within_bounds", False)
    )
    complete_within_bounds = (
        path_engine in ("symbolic", "coverage-generational")
        and bounded_symbolic_complete
        and executor.errors == 0
        and executor.timeouts == 0
        and executor.native_fallback_reason is None
    )
    total_new_inputs = (
        deterministic_new
        + symbolic_new
        + coverage_generational_new
        + concolic_new
        + greybox_new
    )
    useful_inputs = (
        total_new_inputs
        + corpus.generated_path_inputs
        + corpus.generated_trace_inputs
    )
    _add_phase_rates(phase_timings)
    coverage_initial = _coverage_dict(program, seed_covered)
    coverage_final = _coverage_dict(program, corpus.covered)
    coverage_per_time = _coverage_rate_dict(
        coverage_initial,
        coverage_final,
        total_elapsed,
        candidates_evaluated=executor.runs,
        total_new_inputs=total_new_inputs,
        useful_inputs=useful_inputs,
    )
    recommendations = _build_recommendations(
        executor=executor,
        stall_reason=stall_reason,
        total_new_inputs=total_new_inputs,
        symbolic_stats=symbolic_stats,
        phase_decisions=phase_decisions,
    )
    corpus_reduction = _reduce_corpus_coverage(
        corpus,
        out_dir,
        policy=corpus_reduce,
    )

    report = {
        "strategy": "unified-gen-input",
        "mode": mode,
        "pipeline_profile": profile,
        "phase_order": [
            "existing",
            "seed",
            "deterministic",
            "symbolic",
            "coverage-generational",
            "concolic",
            "fuzz",
        ],
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
            "path_priority": path_priority,
            "value_profile_policy": value_profile_policy,
            "branch_distance_policy": branch_distance_policy,
            "max_generations": max_generations,
            "children_per_trace": children_per_trace,
            "objective_policy": objective_policy,
            "score_policy": score_policy,
            "state_pruning": state_pruning,
            "rare_branch_schedule": rare_branch_schedule,
            "target_uncovered": target_uncovered,
            "corpus_reduce": corpus_reduce,
            "solver_timeout_ms": solver_timeout_ms,
            "max_steps_per_input": max_steps_per_input,
            "coverage_metric": coverage_metric,
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
        "coverage_metric": coverage_metric,
        "seed_inputs": n_seeds,
        "existing_inputs": n_seeds,
        "invalid_inputs": corpus.invalid_inputs,
        "duplicate_inputs": corpus.duplicate_inputs,
        "deterministic_new_inputs": deterministic_new,
        "symbolic_new_inputs": symbolic_new,
        "coverage_generational_new_inputs": coverage_generational_new,
        "concolic_new_inputs": concolic_new,
        "greybox_new_inputs": greybox_new,
        "total_new_inputs": total_new_inputs,
        "total_new_path_inputs": corpus.generated_path_inputs,
        "path_interesting_inputs": corpus.path_inputs,
        "generated_path_inputs": corpus.generated_path_inputs,
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
        "step_limits": executor.step_limits,
        "symbolic": symbolic_stats,
        "coverage_generational": coverage_generational_stats,
        "concolic": concolic_stats,
        "rounds_completed": len(round_reports),
        "rounds": round_reports,
        "stall_reason": stall_reason,
        "recommendations": recommendations,
        "blocks_covered": len(corpus.covered),
        "seed_blocks_covered": len(seed_covered),
        "coverage_initial": coverage_initial,
        "coverage": coverage_final,
        "coverage_per_time": coverage_per_time,
        "corpus_reduction": corpus_reduction,
        "minimized_input_count": corpus_reduction.get(
            "minimized_input_count", 0),
        "coverage_objectives_total": corpus_reduction.get(
            "coverage_objectives_total", 0),
        "coverage_objectives_reached": corpus_reduction.get(
            "coverage_objectives_reached", 0),
        "coverage_marginal_by_input": corpus_reduction.get(
            "coverage_marginal_by_input", []),
        "compatible_states_pruned": int(
            coverage_generational_stats.get("compatible_states_pruned", 0)
            or symbolic_stats.get("compatible_states_pruned", 0)
            or 0),
        "dominated_states_pruned": int(
            coverage_generational_stats.get("dominated_states_pruned", 0)
            or symbolic_stats.get("dominated_states_pruned", 0)
            or 0),
        "target_distance_best": (
            coverage_generational_stats.get("target_distance_best")
            if coverage_generational_stats.get("target_distance_best") is not None
            else symbolic_stats.get("target_distance_best")
        ),
        "rare_branch_inputs": int(
            coverage_generational_stats.get("rare_branch_inputs", 0)
            or symbolic_stats.get("rare_branch_inputs", 0)
            or 0),
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
    if corpus_reduce != "none":
        print(f"[driver] minimized corpus: "
              f"{corpus_reduction.get('minimized_input_count', 0)} inputs | "
              f"objectives={corpus_reduction.get('coverage_objectives_reached', 0)}/"
              f"{corpus_reduction.get('coverage_objectives_total', 0)}")
    print(f"[driver] final: covered={len(corpus.covered)} "
          f"new_inputs={total_new_inputs} "
          f"path_inputs={corpus.generated_path_inputs} "
          f"trace_inputs={corpus.generated_trace_inputs} "
          f"assert_feedback={len(corpus.assertion_feedback_entries)} "
          f"step_limits={executor.step_limits} "
          f"stall={stall_reason} engine={_engine_detail(executor)} "
          f"elapsed={total_elapsed:.2f}s")
    return report


_BOUND_LADDERS = {
    "auth-deep": [
        {
            "loop_bound": 4,
            "havoc_bound": 4,
            "max_path_depth": 1024,
            "max_symbolic_states": 8192,
            "max_solver_queries": 32768,
            "children_per_trace": 512,
            "max_generations": 4,
            "value_profile_policy": "late",
            "branch_distance_policy": "on",
        },
        {
            "loop_bound": 8,
            "havoc_bound": 8,
            "max_path_depth": 2048,
            "max_symbolic_states": 512,
            "max_solver_queries": 65536,
            "children_per_trace": 512,
            "max_generations": 4,
            "path_seed_limit": 3,
            "value_profile_policy": "enqueue",
            "branch_distance_policy": "on",
        },
        {
            "loop_bound": 16,
            "havoc_bound": 16,
            "max_path_depth": 4096,
            "max_symbolic_states": 2048,
            "max_solver_queries": 262144,
            "children_per_trace": 1024,
            "max_generations": 6,
            "path_seed_limit": 3,
            "value_profile_policy": "enqueue",
            "branch_distance_policy": "on",
        },
    ],
}


def _parse_bound_ladder(name: str | None) -> list[dict]:
    key = name or "auth-deep"
    if key in _BOUND_LADDERS:
        return [dict(item) for item in _BOUND_LADDERS[key]]
    raise ValueError(f"unknown bound ladder: {key}")


def generate_until_new_path(program, test_name: str, seed_dir: Path,
                            out_dir: Path, *, bound_ladder: str = "auth-deep",
                            max_total_seconds: int = 0,
                            **kwargs) -> dict:
    """Run increasingly larger bounded searches until concrete path gain."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder = _parse_bound_ladder(bound_ladder)
    reports = []
    success_report = None
    success_out_dir = None
    print(f"[until-new-path] ladder={bound_ladder} rungs={len(ladder)} "
          f"max_total_seconds={max_total_seconds or 'unbounded'}")
    for idx, rung in enumerate(ladder, start=1):
        if max_total_seconds and time.perf_counter() - started >= max_total_seconds:
            print("[until-new-path] stopping before next rung: time budget exhausted")
            break
        rung_out = out_dir / f"rung_{idx:02d}"
        rung_kwargs = dict(kwargs)
        rung_kwargs.update(rung)
        print(f"[until-new-path] rung {idx}/{len(ladder)} bounds={rung}")
        report = generate_coverage_inputs(
            program,
            test_name,
            seed_dir,
            rung_out,
            **rung_kwargs,
        )
        cg = report.get("coverage_generational", {})
        reports.append({
            "rung": idx,
            "out_dir": str(rung_out),
            "bounds": report.get("bounds", {}),
            "total_seconds": report.get("total_seconds", 0),
            "total_new_inputs": report.get("total_new_inputs", 0),
            "generated_path_inputs": report.get("generated_path_inputs", 0),
            "blocks_covered": report.get("blocks_covered", 0),
            "stall_reason": report.get("stall_reason"),
            "coverage_generational": {
                key: cg.get(key)
                for key in (
                    "states_explored",
                    "states_bounded",
                    "worklist_remaining",
                    "complete_within_bounds",
                    "paths_unknown",
                    "solver_queries",
                    "solver_sat",
                    "solver_unsat",
                    "solver_unknown",
                    "value_profile_candidates",
                    "branch_distance_objectives",
                    "branch_distance_candidates",
                    "branch_distance_best_delta",
                    "branch_distance_overflowed_objectives",
                    "children_symbolic_candidates",
                    "children_compound_candidates",
                    "pruned_candidates",
                )
            },
        })
        if (
            int(report.get("total_new_inputs", 0) or 0) > 0
            or int(report.get("generated_path_inputs", 0) or 0) > 0
        ):
            success_report = report
            success_out_dir = rung_out
            print(f"[until-new-path] success at rung {idx}: "
                  f"new_blocks={report.get('total_new_inputs', 0)} "
                  f"new_paths={report.get('generated_path_inputs', 0)}")
            break

    summary = {
        "strategy": "until-new-path",
        "bound_ladder": bound_ladder,
        "success": success_report is not None,
        "rungs_completed": len(reports),
        "total_seconds": round(time.perf_counter() - started, 6),
        "reports": reports,
        "success_out_dir": str(success_out_dir) if success_out_dir else None,
    }
    summary_path = out_dir / "until_new_path.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[until-new-path] wrote summary: {summary_path}")
    return summary


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
    parser.add_argument("--engine", choices=["native"],
                        default="native",
                        help="Execution engine for coverage candidates (default: native)")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for reproducible greybox mutation "
                             "(default: 0)")
    parser.add_argument("--path-engine", choices=[
                        "hybrid", "symbolic", "coverage-generational",
                        "concolic", "fuzz"],
                        default="fuzz",
                        help="Path search pipeline after deterministic probes "
                             "(default: fuzz; symbolic/concolic use the Rust engine)")
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
    parser.add_argument("--path-priority",
                        choices=["coverage-directed", "fifo"],
                        default=None,
                        help="Symbolic worklist priority policy "
                             "(profile default)")
    parser.add_argument("--value-profile-policy",
                        choices=["late", "enqueue", "off"],
                        default=None,
                        help="How comparison value-profile candidates feed "
                             "symbolic search (profile default)")
    parser.add_argument("--branch-distance",
                        dest="branch_distance_policy",
                        choices=["auto", "on", "off"],
                        default=None,
                        help="Use branch-distance/provenance guided input "
                             "candidates (profile default)")
    parser.add_argument("--max-generations", type=int, default=None,
                        help="Maximum coverage-generational expansion depth")
    parser.add_argument("--children-per-trace", type=int, default=None,
                        help="Maximum coverage-generational children per trace")
    parser.add_argument("--objective-policy", default=None,
                        help="Comma-separated coverage-generational objective "
                             "classes")
    parser.add_argument("--score-policy", choices=["new-blocks", "coverage-max"],
                        default=None,
                        help="Coverage-generational replay scoring policy")
    parser.add_argument("--state-pruning",
                        choices=["off", "compatible-branch"],
                        default=None,
                        help="Conservative coverage-compatible replay pruning")
    parser.add_argument("--rare-branch-schedule",
                        choices=["auto", "on", "off"],
                        default=None,
                        help="Bias seed selection toward low-frequency paths")
    parser.add_argument("--target-uncovered",
                        choices=["auto", "on", "off"],
                        default=None,
                        help="Bias symbolic candidates toward uncovered CFG targets")
    parser.add_argument("--corpus-reduce",
                        choices=["coverage-setcover", "none"],
                        default=None,
                        help="Write a minimized concrete coverage corpus")
    parser.add_argument("--coverage-metric",
                        choices=["block", "edge", "context-edge", "branch"],
                        default="block",
                        help="Concrete-input retention metric "
                             "(default: block)")
    parser.add_argument("--until-new-path", action="store_true",
                        help="Run a bound ladder until a new concrete block or "
                             "path input is written")
    parser.add_argument("--bound-ladder", default="auth-deep",
                        help="Named bound ladder for --until-new-path "
                             "(default: auth-deep)")
    parser.add_argument("--max-total-seconds", type=int, default=0,
                        help="Overall time budget for --until-new-path "
                             "(0 means no wrapper budget)")
    parser.add_argument("--solver-timeout-ms", type=int, default=None,
                        help="Override per-query SMT timeout in milliseconds")
    parser.add_argument("--max-steps-per-input", type=int,
                        default=DEFAULT_MAX_STEPS_PER_INPUT,
                        help="Maximum Rust VM steps per candidate execution "
                             f"(default: {DEFAULT_MAX_STEPS_PER_INPUT}; "
                             "0 disables the step budget)")
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

    run_kwargs = {
        "mode": args.mode,
        "profile": args.profile,
        "iters": args.iters,
        "timeout": args.timeout,
        "engine": args.engine,
        "rng_seed": args.rng_seed,
        "path_engine": args.path_engine,
        "jobs": args.jobs,
        "progress_interval": args.progress_interval,
        "debug_candidates": args.debug_candidates,
        "rounds": args.rounds,
        "deterministic_budget": args.deterministic_budget,
        "loop_bound": args.loop_bound,
        "havoc_bound": args.havoc_bound,
        "max_path_depth": args.max_path_depth,
        "max_solver_queries": args.max_solver_queries,
        "max_symbolic_states": args.max_symbolic_states,
        "path_seed_limit": args.path_seed_limit,
        "path_priority": args.path_priority,
        "value_profile_policy": args.value_profile_policy,
        "branch_distance_policy": args.branch_distance_policy,
        "max_generations": args.max_generations,
        "children_per_trace": args.children_per_trace,
        "objective_policy": args.objective_policy,
        "score_policy": args.score_policy,
        "state_pruning": args.state_pruning,
        "rare_branch_schedule": args.rare_branch_schedule,
        "target_uncovered": args.target_uncovered,
        "corpus_reduce": args.corpus_reduce,
        "coverage_metric": args.coverage_metric,
        "solver_timeout_ms": args.solver_timeout_ms,
        "max_steps_per_input": args.max_steps_per_input,
        "debug_logger": debug_logger,
    }
    if args.until_new_path:
        if run_kwargs["coverage_metric"] == "block":
            run_kwargs["coverage_metric"] = "context-edge"
        report = generate_until_new_path(
            program, test_name, seed_dir, out_dir,
            bound_ladder=args.bound_ladder,
            max_total_seconds=args.max_total_seconds,
            **run_kwargs,
        )
    else:
        report = generate_coverage_inputs(
            program, test_name, seed_dir, out_dir, **run_kwargs,
        )
    if report.get("seed_inputs", 1) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
