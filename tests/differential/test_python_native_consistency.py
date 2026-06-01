import pytest

from interpreter.tests.helpers.boogie_cases import (
    isolated_cwd,
    make_program,
    run_native_case,
    scalar_inputs,
)
from interpreter.runner import prepare_native, run_native

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
        "prepared_auto",
        "input_0",
        tmp_path / "auto.raw.zst",
        no_trace=True,
        log_read=False,
        return_status=True,
    )
    prepared_result = run_native(
        program,
        inputs,
        "prepared_native",
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
            "no_trace_fast",
            "input_0",
            tmp_path / "unused.raw.zst",
            no_trace=True,
            log_read=False,
            return_status=True,
            prepared=prepared,
            return_memory_summary=False,
            validate_handoff=False,
            quiet=True,
        )

    assert result["status"] == "ok"
    assert result["rust_input_state"] is True
    assert result["init_ms"] == 0.0
    assert result["memory_summary"] == {}
    assert not (tmp_path / "mem_ops_traces").exists()
    assert not (tmp_path / "positive_examples").exists()
