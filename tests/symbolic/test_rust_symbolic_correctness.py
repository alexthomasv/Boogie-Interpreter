import pytest

from interpreter.tests.helpers.boogie_cases import (
    assert_replay_strictly_increases_coverage,
    concolic_candidates,
    havoc_inputs,
    replay_candidates,
    run_native_case,
    scalar_inputs,
    symbolic_candidates,
)
from interpreter.utils.inputs import Input, ProgramInputs


pytestmark = [pytest.mark.symbolic, pytest.mark.native]


def _buffer_inputs(contents: str, size: int) -> ProgramInputs:
    return ProgramInputs(
        {
            "$p0": Input(
                name="$p0",
                private=False,
                buffers=[{"contents": contents, "size": size}],
            )
        }
    )


def _struct_inputs(value: str) -> ProgramInputs:
    return ProgramInputs(
        {
            "$s": Input(
                name="$s",
                private=False,
                struct=[{"name": "tag", "size": 4, "value": value}],
            )
        }
    )


def _extra_inputs(data: bytes) -> ProgramInputs:
    return ProgramInputs({}, extra_data=data)


def _candidate_func(kind: str):
    return concolic_candidates if kind == "concolic" else symbolic_candidates


def _assert_symbolic_reaches(source: str, inputs: ProgramInputs, expected: set[str],
                             tmp_path, kind: str, test_name: str):
    pytest.importorskip("swoosh_interp")
    seed = run_native_case(
        source,
        inputs,
        tmp_path=tmp_path,
        test_name=f"{test_name}_seed",
    )
    assert seed["status"] == "ok", seed

    candidates, stats = _candidate_func(kind)(
        source,
        inputs,
        set(seed["explored_blocks"]),
        tmp_path=tmp_path,
        test_name=f"{test_name}_{kind}",
        max_solver_queries=32,
    )
    assert candidates, stats

    replayed = replay_candidates(
        source,
        candidates,
        tmp_path=tmp_path,
        test_name=f"{test_name}_{kind}_replay",
    )
    assert_replay_strictly_increases_coverage(
        set(seed["explored_blocks"]),
        replayed,
        expected_new_blocks=expected,
    )


@pytest.mark.parametrize("kind", ["concolic", "symbolic"])
def test_scalar_branch_candidate_replays(kind, tmp_path):
    source = """
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
    _assert_symbolic_reaches(
        source,
        scalar_inputs({"$x": 0}),
        {"high"},
        tmp_path,
        kind,
        "scalar_branch",
    )


def test_symbolic_prioritizes_branch_targets_before_value_profiles(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto zero, magic;
    zero:
      assume $x == 0;
      return;
    magic:
      assume $x == 7;
      return;
    }
    """
    seed = run_native_case(
        source,
        scalar_inputs({"$x": 0}),
        tmp_path=tmp_path,
        test_name="priority_seed",
    )
    assert seed["status"] == "ok", seed

    candidates, stats = symbolic_candidates(
        source,
        scalar_inputs({"$x": 0}),
        set(seed["explored_blocks"]),
        tmp_path=tmp_path,
        test_name="priority_symbolic",
        max_solver_queries=16,
        max_states=4,
        branch_distance_policy="off",
    )

    metas = [
        getattr(candidate, "_swoosh_candidate_meta", {})
        for candidate in candidates
    ]
    assert metas, stats
    assert metas[0]["kind"] == "branch_target"
    assert metas[0]["target_block"] == "magic"
    value_profile_indices = [
        idx for idx, meta in enumerate(metas)
        if meta.get("kind") == "value_profile"
    ]
    branch_indices = [
        idx for idx, meta in enumerate(metas)
        if meta.get("kind") == "branch_target"
    ]
    assert value_profile_indices, metas
    assert max(branch_indices) < min(value_profile_indices)
    assert stats["path_priority"] == "coverage-directed"
    assert stats["value_profile_policy"] == "late"
    assert stats["solver_backend"] == "z3-smtlib-persistent"
    assert stats["value_profile_deferred"] >= 1


@pytest.mark.parametrize("kind", ["concolic", "symbolic"])
def test_buffer_byte_branch_candidate_replays(kind, tmp_path):
    source = """
    procedure __SMACK_values(p: ref, n: int) returns (r: ref);
    procedure main($p0: ref);
    implementation {:entrypoint} main($p0: ref) {
      var $M.0: [ref]i8;
      var $M.0.shadow: [ref]i8;
      var $p0.shadow: ref;
      var $tmp: ref;
    entry:
      call {:name $p0} {:array "$load.i8", $M.0, $p0, 1, 2} $tmp := __SMACK_values($p0, 2);
      goto zero, magic;
    zero:
      assume $load.i8($M.0, $p0) == 0;
      return;
    magic:
      assume $load.i8($M.0, $p0) == 66;
      return;
    }
    """
    _assert_symbolic_reaches(
        source,
        _buffer_inputs("0x0000", 2),
        {"magic"},
        tmp_path,
        kind,
        "buffer_branch",
    )


def _assert_distance_guided_reaches(source: str, inputs: ProgramInputs,
                                    expected: set[str], tmp_path,
                                    test_name: str):
    pytest.importorskip("swoosh_interp")
    seed = run_native_case(
        source,
        inputs,
        tmp_path=tmp_path,
        test_name=f"{test_name}_seed",
    )
    assert seed["status"] == "ok", seed

    candidates, stats = concolic_candidates(
        source,
        inputs,
        set(seed["explored_blocks"]),
        tmp_path=tmp_path,
        test_name=f"{test_name}_distance",
        max_solver_queries=0,
        branch_distance_policy="on",
    )
    distance_candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, "_swoosh_candidate_meta", {}).get("kind")
        == "distance_guided"
    ]
    assert stats["solver_queries"] == 0
    assert stats["branch_distance_candidates"] >= 1
    assert distance_candidates, [
        getattr(candidate, "_swoosh_candidate_meta", {})
        for candidate in candidates
    ]

    replayed = replay_candidates(
        source,
        distance_candidates,
        tmp_path=tmp_path,
        test_name=f"{test_name}_distance_replay",
    )
    assert_replay_strictly_increases_coverage(
        set(seed["explored_blocks"]),
        replayed,
        expected_new_blocks=expected,
    )


def test_branch_distance_guides_transformed_byte_compare(tmp_path):
    source = """
    procedure __SMACK_values(p: ref, n: int) returns (r: ref);
    procedure main($p0: ref);
    implementation {:entrypoint} main($p0: ref) {
      var $M.0: [ref]i8;
      var $M.0.shadow: [ref]i8;
      var $p0.shadow: ref;
      var $tmp: ref;
    entry:
      call {:name $p0} {:array "$load.i8", $M.0, $p0, 1, 1} $tmp := __SMACK_values($p0, 1);
      goto zero, magic;
    zero:
      assume $load.i8($M.0, $p0) == 0;
      return;
    magic:
      assume $load.i8($M.0, $p0) + 3 == 69;
      return;
    }
    """
    _assert_distance_guided_reaches(
        source,
        _buffer_inputs("0x00", 1),
        {"magic"},
        tmp_path,
        "distance_transformed_byte",
    )


def test_branch_distance_guides_multibyte_load_compare(tmp_path):
    source = """
    procedure __SMACK_values(p: ref, n: int) returns (r: ref);
    procedure main($p0: ref);
    implementation {:entrypoint} main($p0: ref) {
      var $M.0: [ref]i8;
      var $M.0.shadow: [ref]i8;
      var $p0.shadow: ref;
      var $tmp: ref;
    entry:
      call {:name $p0} {:array "$load.i8", $M.0, $p0, 1, 4} $tmp := __SMACK_values($p0, 4);
      goto zero, magic;
    zero:
      assume $load.i32($M.0, $p0) == 0;
      return;
    magic:
      assume $load.i32($M.0, $p0) == 67305985;
      return;
    }
    """
    _assert_distance_guided_reaches(
        source,
        _buffer_inputs("0x00000000", 4),
        {"magic"},
        tmp_path,
        "distance_multibyte_load",
    )


@pytest.mark.parametrize("kind", ["concolic", "symbolic"])
def test_struct_scalar_byte_branch_candidate_replays(kind, tmp_path):
    source = """
    procedure __SMACK_value.i32(x: int) returns (r: int);
    procedure main($s: ref);
    implementation {:entrypoint} main($s: ref) {
      var $M.1: [ref]i8;
      var $M.1.shadow: [ref]i8;
      var $s.shadow: ref;
      var $tmp: int;
    entry:
      call {:name $s} {:field "$load.i32", $M.1, $s, 4} $tmp := __SMACK_value.i32(0);
      goto zero, magic;
    zero:
      assume $load.i8($M.1, $s) == 0;
      return;
    magic:
      assume $load.i8($M.1, $s) == 7;
      return;
    }
    """
    _assert_symbolic_reaches(
        source,
        _struct_inputs("0x00000000"),
        {"magic"},
        tmp_path,
        kind,
        "struct_scalar_branch",
    )


@pytest.mark.parametrize("kind", ["concolic", "symbolic"])
def test_havoc_sequence_branch_candidate_replays(kind, tmp_path):
    source = """
    procedure __SMACK_nondet_int() returns (r: int);
    procedure main();
    implementation {:entrypoint} main() {
      var $x: int;
    entry:
      call $x := __SMACK_nondet_int();
      goto zero, magic;
    zero:
      assume $x == 0;
      return;
    magic:
      assume $x == 9;
      return;
    }
    """
    _assert_symbolic_reaches(
        source,
        havoc_inputs({"$x": [0]}),
        {"magic"},
        tmp_path,
        kind,
        "havoc_branch",
    )


@pytest.mark.parametrize("kind", ["concolic", "symbolic"])
def test_extra_data_branch_candidate_replays(kind, tmp_path):
    source = """
    procedure read.cross_product(a: int, b: int, c: ref, d: ref, e: int, f: int);
    procedure main();
    implementation {:entrypoint} main() {
      var $M.0: [ref]i8;
      var $M.0.shadow: [ref]i8;
      var $p: ref;
      var $p_shadow: ref;
      var $n: int;
    entry:
      $p := 0;
      $p_shadow := 0;
      $n := 1;
      call read.cross_product(0, 0, $p, $p_shadow, $n, $n);
      goto zero, magic;
    zero:
      assume $load.i8($M.0, $p) == 0;
      return;
    magic:
      assume $load.i8($M.0, $p) == 5;
      return;
    }
    """
    _assert_symbolic_reaches(
        source,
        _extra_inputs(b"\x00"),
        {"magic"},
        tmp_path,
        kind,
        "extra_branch",
    )
