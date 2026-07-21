import pytest

from interpreter.coverage_gen.evaluator import Evaluator
from interpreter.tests.helpers.boogie_cases import (
    isolated_cwd,
    make_program,
    run_native_case,
    scalar_inputs,
)
from interpreter.runner import prepare_native, run_native
from interpreter.utils.inputs import Input, ProgramInputs

pytestmark = [pytest.mark.differential, pytest.mark.native]


def test_native_simple_branch(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto left, right;
    left:
      assume $x == 3;
      return;
    right:
      assume $x != 3;
      return;
    }
    """

    native = run_native_case(
        source,
        scalar_inputs({"$x": 3}),
        tmp_path=tmp_path,
        test_name="diff_branch",
    )

    assert native["status"] == "ok"
    assert native["explored_blocks"] == {"entry", "left"}
    assert native["explored_edges"] == {("entry", "left")}
    assert "block_sequence" not in native


def test_evaluator_carries_compact_native_edges(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto left, right;
    left:
      assume $x == 3;
      return;
    right:
      assume $x != 3;
      return;
    }
    """
    evaluator = Evaluator(make_program(source), "compact_edges", timeout=5)

    result = evaluator.run_result(scalar_inputs({"$x": 3}), "input_0")

    assert result.status == "ok"
    assert result.covered_edges == (("entry", "left"),)
    assert result.block_sequence == ()


def test_buffer_roots_in_different_memory_maps_get_distinct_addresses(tmp_path):
    source = """
    procedure __SMACK_values(p: ref, n: int) returns (r: ref);
    procedure main($bytes: ref, $words: ref);
    implementation {:entrypoint} main($bytes: ref, $words: ref) {
      var $M.bytes: [ref]i8;
      var $M.bytes.shadow: [ref]i8;
      var $M.words: [ref]i32;
      var $M.words.shadow: [ref]i32;
      var $bytes.shadow: ref;
      var $words.shadow: ref;
      var $tmp_bytes: ref;
      var $tmp_words: ref;
    entry:
      call {:name $bytes} {:array "$load.i8", $M.bytes, $bytes, 1, 4}
        $tmp_bytes := __SMACK_values($bytes, 4);
      call {:name $words} {:array "$load.i32", $M.words, $words, 4, 1}
        $tmp_words := __SMACK_values($words, 4);
      return;
    }
    """

    def buffer_input(name, contents):
        return Input(
            name=name,
            private=False,
            buffers=[{"contents": contents, "size": 4}],
        )

    result = run_native_case(
        source,
        ProgramInputs(
            {
                "$bytes": buffer_input("$bytes", "0x01020304"),
                "$bytes.shadow": buffer_input("$bytes.shadow", "0x01020304"),
                "$words": buffer_input("$words", "0x05060708"),
                "$words.shadow": buffer_input("$words.shadow", "0x05060708"),
            }
        ),
        tmp_path=tmp_path,
        test_name="diff_cross_map_input_addresses",
        return_scalar_summary=True,
    )

    assert result["status"] == "ok"
    assert result["final_scalars"]["$bytes"] != result["final_scalars"]["$words"]
    assert result["final_scalars"]["$bytes.shadow"] == result["final_scalars"]["$bytes"]
    assert result["final_scalars"]["$words.shadow"] == result["final_scalars"]["$words"]


def test_native_reports_assert_violation(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto ok, bad;
    ok:
      assume $x == 0;
      return;
    bad:
      assume $x != 0;
      assert false;
      return;
    }
    """

    native = run_native_case(
        source,
        scalar_inputs({"$x": 1}),
        tmp_path=tmp_path,
        test_name="diff_assert",
    )

    assert native["status"] == "assert_violation"
    assert native["violation_block"] == "bad"


def test_native_reports_assume_violation_invalid_input(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      assume $x == 1;
      return;
    }
    """
    inputs = scalar_inputs({"$x": 0})

    native = run_native_case(
        source,
        inputs,
        tmp_path=tmp_path,
        test_name="diff_assume_native",
    )

    assert native["status"] == "assume_violation"
    assert native["invalid_input"] is True
    assert native["invalid_reason"] == "assume"
    assert native["violation_block"] == "entry"


def test_prepared_native_matches_automatic_native(tmp_path):
    pytest.importorskip("swoosh_interp")
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto left, right;
    left:
      assume $x < 10;
      return;
    right:
      assume $x >= 10;
      return;
    }
    """
    program = make_program(source)
    inputs = scalar_inputs({"$x": 12})
    prepared = prepare_native(program)

    automatic = run_native(
        program,
        inputs,
        "input_0",
        tmp_path / "auto.raw.zst",
        no_trace=True,
        log_read=False,
        return_status=True,
    )
    prepared_result = run_native(
        program,
        inputs,
        "input_0",
        tmp_path / "prepared.raw.zst",
        no_trace=True,
        log_read=False,
        return_status=True,
        prepared=prepared,
    )

    assert prepared_result["prepared"] is True
    assert prepared_result["rust_input_state"] is True
    assert prepared_result["init_ms"] == 0.0
    assert prepared_result["status"] == automatic["status"] == "ok"
    assert prepared_result["explored_blocks"] == automatic["explored_blocks"]
    assert prepared_result["memory_summary"] == automatic["memory_summary"]


def test_native_no_trace_fast_options_skip_files_and_memory_summary(tmp_path):
    pytest.importorskip("swoosh_interp")
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      assume $x == 1;
      return;
    }
    """
    program = make_program(source)
    prepared = prepare_native(program)

    with isolated_cwd(tmp_path):
        result = run_native(
            program,
            scalar_inputs({"$x": 1}),
            "input_0",
            tmp_path / "unused.raw.zst",
            no_trace=True,
            log_read=False,
            return_status=True,
            prepared=prepared,
            return_memory_summary=False,
            quiet=True,
        )

    assert result["status"] == "ok"
    assert result["rust_input_state"] is True
    assert result["init_ms"] == 0.0
    assert result["memory_summary"] == {}
    assert not (tmp_path / "mem_ops_traces").exists()
    assert not (tmp_path / "target" / "swoosh" / "traces").exists()
