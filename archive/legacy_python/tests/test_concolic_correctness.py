import pytest

from interpreter.coverage_gen.corpus import Corpus
from interpreter.coverage_gen.driver import generate_coverage_inputs, run_symbolic
from interpreter.coverage_gen.evaluator import Evaluator
from interpreter.coverage_gen.writer import write_input_file
from interpreter.tests.helpers.boogie_cases import (
    assert_replay_adds_coverage,
    assert_replay_strictly_increases_coverage,
    concolic_candidates,
    make_program,
    replay_candidates,
    run_python_case,
    scalar_inputs,
    symbolic_candidates,
)
from interpreter.utils.input_parser import parse_input_file

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


MULTI_BRANCH_PROGRAM = """
procedure main($x: int);
implementation {:entrypoint} main($x: int) {
entry:
  goto neg, nonneg;
neg:
  assume $x < 0;
  return;
nonneg:
  assume $x >= 0;
  goto seven, other;
seven:
  assume $x == 7;
  return;
other:
  assume $x != 7;
  return;
}
"""


INFEASIBLE_BRANCH_PROGRAM = """
procedure main($x: int);
implementation {:entrypoint} main($x: int) {
entry:
  goto feasible, contradictory;
feasible:
  assume $x != 1;
  return;
contradictory:
  assume $x == 1;
  assume $x == 2;
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


def test_concolic_candidates_expand_multi_branch_coverage(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="concolic_multi_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "nonneg", "other"}

    candidates, stats = concolic_candidates(
        MULTI_BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="concolic_multi",
        max_solver_queries=8,
    )

    assert "error" not in stats
    assert stats["solver_sat"] >= 2
    replayed = replay_candidates(
        MULTI_BRANCH_PROGRAM,
        candidates,
        tmp_path=tmp_path,
        test_name="concolic_multi_replay",
    )
    coverage = assert_replay_adds_coverage(
        seed_result["explored_blocks"],
        replayed,
        {"neg", "seven"},
    )
    assert coverage == {"entry", "neg", "nonneg", "seven", "other"}


def test_symbolic_candidate_replay_strictly_uncovers_new_branch(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_strict_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "miss"}

    candidates, stats = symbolic_candidates(
        BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="symbolic_strict_branch",
        max_solver_queries=4,
        max_states=2,
    )

    assert "error" not in stats
    assert stats["solver_sat"] >= 1
    assert candidates
    replayed = replay_candidates(
        BRANCH_PROGRAM,
        candidates,
        tmp_path=tmp_path,
        test_name="symbolic_strict_branch_replay",
    )
    coverage = assert_replay_strictly_increases_coverage(
        seed_result["explored_blocks"],
        replayed,
        {"hit"},
    )
    assert coverage == {"entry", "hit", "miss"}


def test_symbolic_worklist_strictly_expands_multi_branch_coverage(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_multi_strict_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "nonneg", "other"}

    candidates, stats = symbolic_candidates(
        MULTI_BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="symbolic_multi_strict",
        max_solver_queries=12,
        max_states=8,
    )

    assert "error" not in stats
    assert stats["solver_sat"] >= 2
    replayed = replay_candidates(
        MULTI_BRANCH_PROGRAM,
        candidates,
        tmp_path=tmp_path,
        test_name="symbolic_multi_strict_replay",
    )
    coverage = assert_replay_strictly_increases_coverage(
        seed_result["explored_blocks"],
        replayed,
        {"neg", "seven"},
    )
    assert coverage == {"entry", "neg", "nonneg", "seven", "other"}


@pytest.mark.exhaustive
def test_symbolic_worklist_covers_all_feasible_multi_branch_leaves(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_multi_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "nonneg", "other"}

    candidates, stats = symbolic_candidates(
        MULTI_BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="symbolic_multi",
        max_solver_queries=12,
        max_states=8,
    )

    assert "error" not in stats
    assert stats["solver_sat"] >= 2
    assert stats["states_explored"] >= 1
    replayed = replay_candidates(
        MULTI_BRANCH_PROGRAM,
        candidates,
        tmp_path=tmp_path,
        test_name="symbolic_multi_replay",
    )
    coverage = assert_replay_adds_coverage(
        seed_result["explored_blocks"],
        replayed,
        {"neg", "seven"},
    )
    assert coverage == {"entry", "neg", "nonneg", "seven", "other"}


@pytest.mark.exhaustive
def test_symbolic_driver_keeps_candidates_that_expand_corpus_coverage(tmp_path):
    pytest.importorskip("swoosh_interp")
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_driver_seed",
    )

    corpus = Corpus()
    assert corpus.add(
        tmp_path / "seed.input",
        seed,
        seed_result["explored_blocks"],
        params_line="// @params x:$x",
        source="seed",
    )
    before = set(corpus.covered)
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    evaluator = Evaluator(
        make_program(MULTI_BRANCH_PROGRAM),
        "symbolic_driver_multi",
        timeout=5,
        engine="native",
    )

    new_count, stats = run_symbolic(
        corpus,
        evaluator,
        out_dir,
        max_solver_queries=12,
        max_states=8,
        seed_limit=1,
    )

    assert "skipped" not in stats
    assert stats["solver_sat"] >= 2
    assert new_count >= 2
    assert new_count > 0
    assert before < corpus.covered
    assert corpus.covered - before == {"neg", "seven"}
    assert {"neg", "seven"} <= corpus.covered
    assert len(list(out_dir.glob("gen_*.input"))) >= 2


def test_symbolic_driver_strictly_increases_corpus_coverage(tmp_path):
    pytest.importorskip("swoosh_interp")
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_driver_strict_seed",
    )

    corpus = Corpus()
    assert corpus.add(
        tmp_path / "seed.input",
        seed,
        seed_result["explored_blocks"],
        params_line="// @params x:$x",
        source="seed",
    )
    before = set(corpus.covered)
    out_dir = tmp_path / "driver_generated"
    out_dir.mkdir()
    evaluator = Evaluator(
        make_program(MULTI_BRANCH_PROGRAM),
        "symbolic_driver_strict",
        timeout=5,
        engine="native",
    )

    new_count, stats = run_symbolic(
        corpus,
        evaluator,
        out_dir,
        max_solver_queries=12,
        max_states=8,
        seed_limit=1,
    )

    assert "skipped" not in stats
    assert stats["solver_sat"] >= 2
    assert new_count > 0
    assert before < corpus.covered
    assert corpus.covered - before == {"neg", "seven"}
    assert len(list(out_dir.glob("gen_*.input"))) >= 2


def test_symbolic_only_pipeline_strictly_increases_coverage(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(MULTI_BRANCH_PROGRAM)
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        MULTI_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_pipeline_seed",
    )
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    out_dir = tmp_path / "pipeline_generated"
    (seed_dir / "seed.input").write_text(
        write_input_file(seed, params_line="// @params x:$x")
    )

    report = generate_coverage_inputs(
        program,
        "symbolic_pipeline_strict",
        seed_dir,
        out_dir,
        engine="native",
        path_engine="symbolic",
        profile="fast-traces",
        deterministic_budget=0,
        iters=0,
        rounds=1,
        loop_bound=4,
        havoc_bound=4,
        max_path_depth=128,
        max_solver_queries=12,
        max_symbolic_states=8,
        path_seed_limit=1,
        timeout=5,
        jobs=1,
        rng_seed=0,
    )

    rates = report["coverage_per_time"]
    assert report["symbolic_new_inputs"] > 0
    assert report["total_new_inputs"] == report["symbolic_new_inputs"]
    assert rates["reachable_blocks_added"] > 0
    assert rates["initial_reachable_blocks"] < rates["final_reachable_blocks"]
    assert (
        report["coverage"]["reachable_blocks_covered"]
        > report["coverage_initial"]["reachable_blocks_covered"]
    )
    generated = sorted(out_dir.glob("gen_*.input"))
    assert generated
    replayed = [
        run_python_case(
            MULTI_BRANCH_PROGRAM,
            parse_input_file(path),
            tmp_path=tmp_path,
            test_name=f"symbolic_pipeline_replay_{idx}",
        )
        for idx, path in enumerate(generated)
    ]
    assert_replay_strictly_increases_coverage(
        seed_result["explored_blocks"],
        replayed,
        {"neg", "seven"},
    )


@pytest.mark.exhaustive
def test_symbolic_replay_does_not_claim_infeasible_branch_coverage(tmp_path):
    seed = scalar_inputs({"$x": 0})
    seed_result = run_python_case(
        INFEASIBLE_BRANCH_PROGRAM,
        seed,
        tmp_path=tmp_path,
        test_name="symbolic_infeasible_seed",
    )
    assert seed_result["explored_blocks"] == {"entry", "feasible"}

    candidates, stats = symbolic_candidates(
        INFEASIBLE_BRANCH_PROGRAM,
        seed,
        seed_result["explored_blocks"],
        tmp_path=tmp_path,
        test_name="symbolic_infeasible",
        max_solver_queries=12,
        max_states=8,
    )

    assert "error" not in stats
    assert stats["solver_queries"] > 0
    replayed = replay_candidates(
        INFEASIBLE_BRANCH_PROGRAM,
        candidates,
        tmp_path=tmp_path,
        test_name="symbolic_infeasible_replay",
    )
    for result in replayed:
        assert result["status"] in {"ok", "assume_violation"}
    ok_coverage = set().union(
        seed_result["explored_blocks"],
        *(
            result["explored_blocks"]
            for result in replayed
            if result["status"] == "ok"
        ),
    )
    assert "contradictory" not in ok_coverage
    assert all(
        result["status"] == "assume_violation"
        for result in replayed
        if "contradictory" in result["explored_blocks"]
    )


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
