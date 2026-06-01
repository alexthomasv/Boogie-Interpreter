import pytest

from interpreter.coverage_gen.corpus import Corpus
from interpreter.coverage_gen.driver import _record_candidate_result
from interpreter.tests.helpers.boogie_cases import scalar_inputs


pytestmark = [pytest.mark.integration]


def test_assume_violating_candidate_is_invalid_not_coverage_or_assertion(tmp_path):
    corpus = Corpus()
    candidate = scalar_inputs({"$x": 0})

    kind, gen_idx, trace_idx = _record_candidate_result(
        corpus,
        tmp_path,
        candidate,
        {"entry", "invalid_branch"},
        "// @params x:$x",
        gen_idx=0,
        trace_idx=0,
        label="symbolic",
        status="assume_violation",
        violation_pc=5,
        violation_block="invalid_branch",
    )

    assert kind is None
    assert gen_idx == 0
    assert trace_idx == 0
    assert corpus.invalid_inputs == 1
    assert corpus.covered == set()
    assert corpus.entries == []
    assert corpus.assertion_feedback_entries == []
    assert not list(tmp_path.glob("*.input"))


def test_step_limited_candidate_is_not_written_as_concrete_input(tmp_path):
    corpus = Corpus()
    candidate = scalar_inputs({"$x": 0})

    kind, gen_idx, trace_idx = _record_candidate_result(
        corpus,
        tmp_path,
        candidate,
        {"entry", "loop"},
        "// @params x:$x",
        gen_idx=0,
        trace_idx=0,
        label="greybox",
        status="step_limit",
        violation_pc=9,
        violation_block="loop",
    )

    assert kind is None
    assert gen_idx == 0
    assert trace_idx == 0
    assert corpus.invalid_inputs == 0
    assert corpus.covered == set()
    assert corpus.entries == []
    assert corpus.assertion_feedback_entries == []
    assert not list(tmp_path.glob("*.input"))


def test_ok_candidate_with_new_coverage_is_written(tmp_path):
    corpus = Corpus()
    corpus.covered = {"entry"}
    candidate = scalar_inputs({"$x": 10})

    kind, gen_idx, trace_idx = _record_candidate_result(
        corpus,
        tmp_path,
        candidate,
        {"entry", "new_block"},
        "// @params x:$x",
        gen_idx=0,
        trace_idx=0,
        label="deterministic",
        status="ok",
    )

    assert kind == "coverage"
    assert gen_idx == 1
    assert trace_idx == 0
    assert corpus.covered == {"entry", "new_block"}
    assert len(corpus.entries) == 1
    assert (tmp_path / "gen_0.input").exists()


def test_assert_violating_candidate_is_recorded_for_feedback(tmp_path):
    corpus = Corpus()
    candidate = scalar_inputs({"$x": 1})

    kind, gen_idx, trace_idx = _record_candidate_result(
        corpus,
        tmp_path,
        candidate,
        {"entry", "bad"},
        "// @params x:$x",
        gen_idx=0,
        trace_idx=0,
        label="symbolic",
        status="assert_violation",
        violation_pc=7,
        violation_block="bad",
    )

    assert kind == "assertion"
    assert gen_idx == 0
    assert trace_idx == 0
    assert corpus.invalid_inputs == 0
    assert corpus.covered == set()
    assert corpus.entries == []
    assert len(corpus.assertion_feedback_entries) == 1
    assert (tmp_path / "assert_0.input").exists()
