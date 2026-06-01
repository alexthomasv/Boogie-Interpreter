#!/usr/bin/env python3
"""Refresh the relative performance baseline for Rust symbolic execution."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


INTERPRETER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = INTERPRETER_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(INTERPRETER_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERPRETER_ROOT))

from interpreter.coverage_gen.evaluator import Evaluator  # noqa: E402
from interpreter.tests.helpers.boogie_cases import make_program, scalar_inputs  # noqa: E402


SOURCE = """
procedure main($x: int);
implementation {:entrypoint} main($x: int) {
entry:
  goto zero, high;
zero:
  assume $x == 0;
  return;
high:
  assume $x >= 10;
  return;
}
"""


def main() -> int:
    output = INTERPRETER_ROOT / "tests" / "fixtures" / "rust_symbolic_perf.json"
    cases = {
        "scalar_concolic": _measure("concolic"),
        "scalar_symbolic": _measure("symbolic"),
    }
    payload = {
        "schema": 1,
        "min_ratio": 0.5,
        "state_ms_max_ratio": 5.0,
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    return 0


def _measure(kind: str) -> dict:
    samples = [_run_sample(kind, idx) for idx in range(7)]
    return {
        "candidates_per_sec": statistics.median(
            sample["candidates_per_sec"] for sample in samples
        ),
        "symbolic_state_ms": statistics.median(
            sample["symbolic_state_ms"] for sample in samples
        ),
    }


def _run_sample(kind: str, idx: int) -> dict:
    program = make_program(SOURCE)
    inputs = scalar_inputs({"$x": 0})
    evaluator = Evaluator(program, f"bless_symbolic_perf_{kind}_{idx}", timeout=5)
    covered = evaluator.run(inputs, "seed")
    if kind == "concolic":
        candidates, stats = evaluator.concolic_suggest(
            inputs,
            "seed",
            covered,
            max_solver_queries=32,
        )
    else:
        candidates, stats = evaluator.symbolic_explore(
            inputs,
            "seed",
            covered,
            max_solver_queries=32,
            max_states=16,
        )
    if not candidates:
        raise RuntimeError(f"{kind} produced no candidates: {stats}")
    return {
        "candidates_per_sec": float(stats.get("candidates_per_sec", 0.0) or 0.0),
        "symbolic_state_ms": float(stats.get("symbolic_state_ms", 0.0) or 0.0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
