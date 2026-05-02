import pytest

from interpreter.tests.helpers.boogie_cases import (
    concolic_candidates,
    run_python_case,
    scalar_inputs,
)

pytestmark = [pytest.mark.symbolic, pytest.mark.native]


BRANCH_PROGRAM = """
procedure main($x: int);
implementation {:entrypoint} main($x: int) {
entry:
  goto hit, miss;
hit:
  assume $x == 7;
  return;
miss:
  assume $x != 7;
  return;
}
"""


def test_concolic_candidate_replays_into_uncovered_branch(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="concolic_replay_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "miss"}

    candidates, stats = concolic_candidates(
        BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="concolic_replay",
    )

    assert stats["solver_sat"] >= 1
    assert candidates
    replayed = [
        run_python_case(
            BRANCH_PROGRAM,
            candidate,
            tmp_path=tmp_path,
            test_name=f"concolic_replay_{idx}",
        )["explored_blocks"]
        for idx, candidate in enumerate(candidates)
    ]
    assert any("hit" in blocks for blocks in replayed)


def test_concolic_query_budget_is_reported(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="concolic_budget_seed",
    )

    _candidates, stats = concolic_candidates(
        BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="concolic_budget",
        max_solver_queries=0,
    )

    assert stats["solver_query_bounded"] is True
    assert stats["solver_queries"] == 0
