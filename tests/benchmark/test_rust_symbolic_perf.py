import json
import statistics
from pathlib import Path

import pytest

from interpreter.coverage_gen.evaluator import Evaluator
from interpreter.tests.helpers.boogie_cases import make_program, scalar_inputs


pytestmark = [pytest.mark.benchmark, pytest.mark.native, pytest.mark.symbolic]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "rust_symbolic_perf.json"

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


@pytest.mark.parametrize("case_name,kind", [
    ("scalar_concolic", "concolic"),
    ("scalar_symbolic", "symbolic"),
])
def test_rust_symbolic_engine_relative_perf(case_name, kind):
    pytest.importorskip("swoosh_interp")
    baseline = json.loads(FIXTURE.read_text())
    case = baseline["cases"][case_name]
    min_ratio = float(baseline.get("min_ratio", 0.5))
    state_ratio = float(baseline.get("state_ms_max_ratio", 5.0))

    samples = [_run_sample(kind, idx) for idx in range(3)]
    candidates_per_sec = statistics.median(
        sample["candidates_per_sec"] for sample in samples
    )
    state_ms = statistics.median(sample["symbolic_state_ms"] for sample in samples)

    assert candidates_per_sec >= case["candidates_per_sec"] * min_ratio
    assert state_ms <= max(float(case["symbolic_state_ms"]), 1.0) * state_ratio


def _run_sample(kind: str, idx: int) -> dict:
    program = make_program(SOURCE)
    inputs = scalar_inputs({"$x": 0})
    evaluator = Evaluator(program, f"rust_symbolic_perf_{kind}_{idx}", timeout=5)
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
    assert candidates, stats
    return {
        "candidates_per_sec": float(stats.get("candidates_per_sec", 0.0) or 0.0),
        "symbolic_state_ms": float(stats.get("symbolic_state_ms", 0.0) or 0.0),
    }
