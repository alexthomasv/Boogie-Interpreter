import pytest

from interpreter.coverage_gen.driver import (
    _build_generational_children,
    _new_path_diagnostics,
    _compound_value_profile_candidates,
    _materialize_replay_candidate,
    _reduce_corpus_coverage,
    _record_path_diagnostic,
    _should_prune_replay,
    generate_coverage_inputs,
    generate_until_new_path,
    run_coverage_generational,
)
from interpreter.coverage_gen.corpus import (
    Corpus,
    CorpusEntry,
    path_features_from_sequence,
)
from interpreter.coverage_gen.writer import write_input_file
from interpreter.coverage_gen.symbolic_state import LazySymbolicCandidate
from interpreter.runner import prepare_native, run_native
from interpreter.tests.helpers.boogie_cases import make_program, scalar_inputs
from interpreter.tests.helpers.raw_trace import assert_raw_trace_file
from interpreter.utils.inputs import Input, ProgramInputs
from interpreter.utils.input_parser import parse_input_file


pytestmark = [pytest.mark.integration, pytest.mark.native]


FAST_BRANCH_SOURCE = """
procedure main($x: int);
implementation {:entrypoint} main($x: int) {
  var $y: int;
entry:
  $y := $x;
  goto zero, high;
zero:
  assume $x == 0;
  $y := $y + 1;
  return;
high:
  assume $x >= 10;
  $y := $y + 2;
  return;
}
"""


def test_gen_input_fast_search_then_replay_kept_inputs_with_raw_traces(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(FAST_BRANCH_SOURCE)
    seed_dir = tmp_path / "seeds"
    out_dir = tmp_path / "generated"
    seed_dir.mkdir()
    seed_path = seed_dir / "seed_0.input"
    seed_path.write_text(
        write_input_file(scalar_inputs({"$x": 0}), "// @params x:$x")
    )

    report = generate_coverage_inputs(
        program,
        "fast_gen_case",
        seed_dir,
        out_dir,
        profile="fast-traces",
        path_engine="fuzz",
        jobs=1,
        timeout=5,
        rounds=1,
        deterministic_budget=16,
        iters=0,
        loop_bound=4,
        havoc_bound=4,
        max_path_depth=64,
        max_solver_queries=8,
        max_symbolic_states=4,
        max_steps_per_input=10_000,
        progress_interval=1000,
    )

    assert report["engine_used"] == "native"
    assert report["native_fallback"] is False
    assert report["step_limits"] == 0
    assert report["candidates_evaluated"] >= 2
    assert report["total_new_inputs"] >= 1
    assert report["coverage_per_time"]["candidates_per_sec"] > 0

    kept_inputs = [seed_path]
    kept_inputs.extend(sorted(out_dir.glob("gen_*.input")))
    kept_inputs.extend(sorted(out_dir.glob("trace_*.input")))
    assert any(path.name.startswith("gen_") for path in kept_inputs)

    prepared = prepare_native(program)
    trace_dir = tmp_path / "raw_traces"
    trace_dir.mkdir()
    for input_path in kept_inputs[:3]:
        program_inputs = parse_input_file(input_path)
        raw_log = trace_dir / f"{input_path.stem}.trace.raw.zst"
        result = run_native(
            program,
            program_inputs,
            "fast_gen_case",
            input_path.stem,
            raw_log_path=raw_log,
            extra_data=program_inputs.extra_data,
            prepared=prepared,
            no_trace=False,
            log_read=True,
            return_status=True,
            return_memory_summary=False,
            validate_handoff=False,
            quiet=True,
            max_steps=10_000,
        )
        assert result["status"] == "ok"
        assert result["trace_records"] > 0
        assert_raw_trace_file(raw_log, expected_records=result["trace_records"])


def test_gen_input_symbolic_reports_guided_path_diagnostics(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(FAST_BRANCH_SOURCE)
    seed_dir = tmp_path / "seeds"
    out_dir = tmp_path / "generated"
    seed_dir.mkdir()
    (seed_dir / "seed_0.input").write_text(
        write_input_file(scalar_inputs({"$x": 0}), "// @params x:$x")
    )

    report = generate_coverage_inputs(
        program,
        "guided_symbolic_case",
        seed_dir,
        out_dir,
        profile="fast-traces",
        path_engine="symbolic",
        jobs=1,
        timeout=5,
        rounds=1,
        deterministic_budget=0,
        iters=0,
        loop_bound=4,
        havoc_bound=4,
        max_path_depth=64,
        max_solver_queries=16,
        max_symbolic_states=4,
        path_priority="coverage-directed",
        value_profile_policy="late",
        branch_distance_policy="on",
        max_steps_per_input=10_000,
        progress_interval=1000,
    )

    symbolic = report["symbolic"]
    assert report["total_new_inputs"] >= 1
    assert symbolic["path_priority"] == "coverage-directed"
    assert symbolic["value_profile_policy"] == "late"
    assert symbolic["branch_distance_policy"] == "on"
    assert symbolic["branch_distance_candidates"] >= 1
    assert symbolic["candidate_kind_summary"]["branch_target"] >= 1
    assert symbolic["candidate_kind_summary"]["distance_guided"] >= 1
    assert symbolic["replay_status_by_target"]["high"]["ok"] >= 1
    assert symbolic["first_new_block_target"] == "high"


def test_compound_value_profile_candidates_merge_disjoint_byte_updates():
    base = ProgramInputs({
        "$p0": Input(
            name="$p0",
            private=False,
            buffers=[{"contents": "0x0000", "size": 2}],
        ),
    })
    bindings = {
        "s0": {
            "kind": "buffer_byte",
            "input_var": "$p0",
            "buffer_index": 0,
            "byte_index": 0,
        },
        "s1": {
            "kind": "buffer_byte",
            "input_var": "$p0",
            "buffer_index": 0,
            "byte_index": 1,
        },
    }

    def candidate(raw_index, updates):
        item = ProgramInputs({})
        item._swoosh_symbolic_bindings = bindings
        item._swoosh_havoc_bound = 4
        item._swoosh_candidate_meta = {
            "raw_index": raw_index,
            "source_block": "entry",
            "target_block": "cmp@1",
            "branch_index": 0,
            "kind": "value_profile",
            "priority": 100 - raw_index,
            "distance_to_uncovered": 1,
            "reachable_uncovered": 1,
            "updates": updates,
        }
        return item

    compounds = _compound_value_profile_candidates(
        base,
        [
            candidate(0, {"s0": 0x41}),
            candidate(1, {"s1": 0x42}),
        ],
        children_per_trace=8,
    )

    assert len(compounds) == 1
    compound = compounds[0]
    meta = compound._swoosh_candidate_meta
    assert meta["kind"] == "compound_value_profile"
    assert meta["compound_size"] == 2
    assert meta["compound_raw_indices"] == [0, 1]
    assert compound.variables["$p0"].buffers[0]["contents"] == "0x4142"


def test_compound_value_profile_replay_prunes_assertion_only_targets():
    candidate = ProgramInputs({})
    candidate._swoosh_candidate_meta = {
        "kind": "compound_value_profile",
        "source_block": "entry",
        "target_block": "cmp@1",
    }
    diagnostics = _new_path_diagnostics()

    for _ in range(64):
        _record_path_diagnostic(
            diagnostics,
            candidate,
            {"status": "assert_violation"},
            set(),
        )

    assert _should_prune_replay(diagnostics, candidate)
    diagnostics["new_blocks_by_target"]["cmp@1"] = 1
    diagnostics["new_blocks_by_objective"]["entry->cmp@1"] = 1
    assert not _should_prune_replay(diagnostics, candidate)


def test_value_profile_pruning_is_scoped_to_source_block():
    hot_candidate = ProgramInputs({})
    hot_candidate._swoosh_candidate_meta = {
        "kind": "compound_value_profile",
        "source_block": "entry_a",
        "target_block": "cmp@1",
    }
    fresh_candidate = ProgramInputs({})
    fresh_candidate._swoosh_candidate_meta = {
        "kind": "compound_value_profile",
        "source_block": "entry_b",
        "target_block": "cmp@1",
    }
    diagnostics = _new_path_diagnostics()

    for _ in range(64):
        _record_path_diagnostic(
            diagnostics,
            hot_candidate,
            {"status": "assert_violation"},
            set(),
        )

    assert _should_prune_replay(diagnostics, hot_candidate)
    assert not _should_prune_replay(diagnostics, fresh_candidate)


def test_lazy_symbolic_candidate_materializes_only_for_replay():
    base = scalar_inputs({"$x": 0})
    lazy = LazySymbolicCandidate(
        base,
        {"s0": {"kind": "scalar", "var": "$x"}},
        4,
        {
            "raw_index": 0,
            "source_block": "entry",
            "target_block": "high",
            "branch_index": 0,
            "kind": "branch_target",
            "priority": 10,
            "distance_to_uncovered": 0,
            "reachable_uncovered": 1,
            "updates": {"s0": 42},
        },
    )

    children, stats = _build_generational_children(
        base,
        [lazy],
        children_per_trace=1,
        objective_policy="branch",
    )

    assert stats["children_single_candidates"] == 1
    assert children[0] is lazy
    assert base.variables["$x"].value == 0
    materialized = _materialize_replay_candidate(children[0])
    assert materialized.variables["$x"].value == 42
    assert _materialize_replay_candidate(children[0]) is materialized


def test_coverage_generational_replays_children_in_one_batch(tmp_path):
    base = scalar_inputs({"$x": 0})
    corpus = Corpus()
    corpus.add(
        tmp_path / "seed.input",
        base,
        {"entry"},
        source="existing",
        path_features=path_features_from_sequence(("entry",)),
    )

    class FakeEvaluator:
        def __init__(self):
            self.candidates = [
                LazySymbolicCandidate(
                    base,
                    {f"s{idx}": {"kind": "scalar", "var": "$x"}},
                    4,
                    {
                        "raw_index": idx,
                        "source_block": "entry",
                        "target_block": f"target_{idx}",
                        "branch_index": idx,
                        "kind": "branch_target",
                        "priority": 100 - idx,
                        "distance_to_uncovered": idx,
                        "reachable_uncovered": 1,
                        "updates": {f"s{idx}": idx + 1},
                    },
                )
                for idx in range(3)
            ]

        def symbolic_explore_lazy(self, *args, **kwargs):
            return self.candidates, {
                "solver_queries": 0,
                "states_explored": 1,
            }

    class FakeExecutor:
        def __init__(self):
            self.evaluator = FakeEvaluator()
            self.batch_sizes = []

        def evaluate_many(self, tasks):
            self.batch_sizes.append(len(tasks))
            assert all(isinstance(task[1], ProgramInputs) for task in tasks)
            return [
                {
                    "index": task[0],
                    "covered": {"entry"},
                    "status": "ok",
                    "block_sequence": ("entry",),
                }
                for task in tasks
            ]

    executor = FakeExecutor()
    new_count, stats = run_coverage_generational(
        corpus,
        executor,
        tmp_path,
        max_solver_queries=8,
        max_states=8,
        seed_limit=1,
        max_generations=1,
        children_per_trace=3,
        objective_policy="branch",
    )

    assert new_count == 0
    assert executor.batch_sizes == [3]
    assert stats["max_replay_batch"] == 3
    assert stats["replay_batches"] == 1


def test_context_edge_metric_keeps_same_block_new_path(tmp_path):
    corpus = Corpus(coverage_metric="context-edge")
    inputs = scalar_inputs({"$x": 0})
    covered = {"entry", "loop", "exit"}

    assert corpus.add(
        tmp_path / "seed.input",
        inputs,
        covered,
        source="existing",
        path_features=path_features_from_sequence(
            ("entry", "loop", "exit")
        ),
    )
    assert corpus.add(
        tmp_path / "path.input",
        scalar_inputs({"$x": 1}),
        covered,
        source="symbolic",
        path_features=path_features_from_sequence(
            ("entry", "loop", "loop", "exit")
        ),
    )

    assert corpus.generated_path_inputs == 1
    assert corpus.entries[-1].interesting_reason == "context-edge"


def test_coverage_max_prioritizes_near_uncovered_candidate():
    corpus = Corpus(coverage_metric="block")
    base = ProgramInputs({})

    def candidate(raw_index, target, priority, distance, reachable):
        item = ProgramInputs({})
        item._swoosh_candidate_meta = {
            "raw_index": raw_index,
            "source_block": "entry",
            "target_block": target,
            "branch_index": raw_index,
            "kind": "branch_target",
            "priority": priority,
            "distance_to_uncovered": distance,
            "reachable_uncovered": reachable,
            "updates": {f"s{raw_index}": raw_index},
        }
        return item

    children, stats = _build_generational_children(
        base,
        [
            candidate(0, "far", 4_000_000, 20, 0),
            candidate(1, "near", 3_900_000, 1, 10),
        ],
        children_per_trace=1,
        objective_policy="branch",
        corpus=corpus,
        score_policy="coverage-max",
        target_uncovered="on",
    )

    assert stats["children_single_candidates"] == 1
    assert children[0]._swoosh_candidate_meta["target_block"] == "near"


def test_corpus_reducer_writes_coverage_setcover(tmp_path):
    corpus = Corpus(coverage_metric="block")
    entries = [
        CorpusEntry(
            path=tmp_path / "a.input",
            inputs=scalar_inputs({"$x": 0}),
            covered_blocks={"A", "B"},
            path_features=path_features_from_sequence(("A", "B")),
            params_line="// @params x:$x",
            source="existing",
        ),
        CorpusEntry(
            path=tmp_path / "b.input",
            inputs=scalar_inputs({"$x": 1}),
            covered_blocks={"B", "C"},
            path_features=path_features_from_sequence(("B", "C")),
            params_line="// @params x:$x",
            source="existing",
        ),
        CorpusEntry(
            path=tmp_path / "abc.input",
            inputs=scalar_inputs({"$x": 2}),
            covered_blocks={"A", "B", "C"},
            path_features=path_features_from_sequence(("A", "B", "C")),
            params_line="// @params x:$x",
            source="coverage-generational",
        ),
    ]
    corpus.entries.extend(entries)

    report = _reduce_corpus_coverage(
        corpus,
        tmp_path / "out",
        policy="coverage-setcover",
    )

    assert report["minimized_input_count"] == 1
    assert report["coverage_objectives_total"] == 3
    assert report["coverage_objectives_reached"] == 3
    assert list((tmp_path / "out" / "minimized").glob("min_*.input"))


def test_gen_input_coverage_generational_discovers_untaken_branch(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(FAST_BRANCH_SOURCE)
    seed_dir = tmp_path / "seeds"
    out_dir = tmp_path / "generated"
    seed_dir.mkdir()
    (seed_dir / "seed_0.input").write_text(
        write_input_file(scalar_inputs({"$x": 0}), "// @params x:$x")
    )

    report = generate_coverage_inputs(
        program,
        "coverage_generational_case",
        seed_dir,
        out_dir,
        profile="fast-traces",
        path_engine="coverage-generational",
        jobs=1,
        timeout=5,
        rounds=1,
        deterministic_budget=0,
        iters=0,
        loop_bound=4,
        havoc_bound=4,
        max_path_depth=64,
        max_solver_queries=32,
        max_symbolic_states=16,
        path_priority="coverage-directed",
        value_profile_policy="late",
        branch_distance_policy="on",
        max_generations=2,
        children_per_trace=16,
        objective_policy="branch,branch-distance,compound-value-profile,value-profile",
        max_steps_per_input=10_000,
        progress_interval=1000,
    )

    coverage_generational = report["coverage_generational"]
    assert report["path_engine"] == "coverage-generational"
    assert report["coverage_generational_new_inputs"] >= 1
    assert report["total_new_inputs"] >= 1
    assert coverage_generational["branch_distance_policy"] == "on"
    assert coverage_generational["branch_distance_candidates"] >= 1
    assert coverage_generational["generations_completed"] >= 1
    assert coverage_generational["candidate_kind_summary"]["branch_target"] >= 1
    assert coverage_generational["replay_status_by_target"]["high"]["ok"] >= 1
    assert coverage_generational["first_new_block_target"] == "high"
    assert any(out_dir.glob("gen_*.input"))


def test_until_new_path_ladder_stops_after_concrete_path_gain(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(FAST_BRANCH_SOURCE)
    seed_dir = tmp_path / "seeds"
    out_dir = tmp_path / "until"
    seed_dir.mkdir()
    (seed_dir / "seed_0.input").write_text(
        write_input_file(scalar_inputs({"$x": 0}), "// @params x:$x")
    )

    report = generate_until_new_path(
        program,
        "until_new_path_case",
        seed_dir,
        out_dir,
        bound_ladder="auth-deep",
        profile="fast-traces",
        path_engine="coverage-generational",
        jobs=1,
        timeout=5,
        rounds=1,
        deterministic_budget=0,
        iters=0,
        path_seed_limit=1,
        path_priority="coverage-directed",
        coverage_metric="context-edge",
        max_total_seconds=30,
        max_steps_per_input=10_000,
        progress_interval=1000,
    )

    assert report["success"] is True
    assert report["rungs_completed"] == 1
    assert list((out_dir / "rung_01").glob("gen_*.input"))
    assert (out_dir / "until_new_path.json").exists()


def test_native_replay_reports_block_sequence(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program(FAST_BRANCH_SOURCE)
    prepared = prepare_native(program)
    result = run_native(
        program,
        scalar_inputs({"$x": 0}),
        "block_sequence_case",
        "seed",
        raw_log_path=tmp_path / "seed.trace.raw.zst",
        prepared=prepared,
        no_trace=True,
        log_read=False,
        return_status=True,
        return_memory_summary=False,
        validate_handoff=False,
        quiet=True,
        max_steps=10_000,
    )

    assert result["status"] == "ok"
    assert result["block_sequence"] == ["entry", "zero"]
    assert result["block_sequence_len"] == 2


def test_gen_input_reports_step_limited_seed_without_writing_candidate(tmp_path):
    pytest.importorskip("swoosh_interp")
    program = make_program("""
    procedure main();
    implementation {:entrypoint} main() {
    entry:
      goto loop;
    loop:
      goto loop;
    }
    """)
    seed_dir = tmp_path / "seeds"
    out_dir = tmp_path / "generated"
    seed_dir.mkdir()
    (seed_dir / "loop.input").write_text("\n")

    report = generate_coverage_inputs(
        program,
        "step_limited_case",
        seed_dir,
        out_dir,
        profile="fast-traces",
        path_engine="fuzz",
        jobs=1,
        timeout=5,
        rounds=1,
        deterministic_budget=0,
        iters=0,
        loop_bound=1,
        havoc_bound=1,
        max_path_depth=8,
        max_solver_queries=0,
        max_symbolic_states=0,
        max_steps_per_input=8,
        progress_interval=1000,
    )

    assert report["seed_inputs"] == 1
    assert report["step_limits"] == 1
    assert report["total_new_inputs"] == 0
    assert report["corpus_entries"] == 0
    assert not list(out_dir.glob("gen_*.input"))
    assert not list(out_dir.glob("trace_*.input"))
