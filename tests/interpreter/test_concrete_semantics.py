import pytest

from interpreter.tests.helpers.boogie_cases import (
    assert_ok,
    assert_violation,
    run_native_case,
    scalar_inputs,
)

pytestmark = [pytest.mark.interpreter, pytest.mark.native]


def test_branch_selection_records_expected_path(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto low, high;
    low:
      assume $x < 10;
      return;
    high:
      assume $x >= 10;
      return;
    }
    """

    result = run_native_case(
        source,
        scalar_inputs({"$x": 11}),
        tmp_path=tmp_path,
        test_name="branch_case",
    )

    assert_ok(result, blocks={"entry", "high"})


def test_assert_violation_has_structured_location(tmp_path):
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

    result = run_native_case(
        source,
        scalar_inputs({"$x": 5}),
        tmp_path=tmp_path,
        test_name="assert_case",
    )

    assert_violation(result, "assert_violation", "bad")
    assert result["invalid_input"] is False
    assert "bad" in result["explored_blocks"]


def test_requires_violation_is_reported_as_assume_violation(tmp_path):
    source = """
    procedure main($x: int);
      requires $x == 1;
    implementation {:entrypoint} main($x: int) {
    entry:
      return;
    }
    """

    result = run_native_case(
        source,
        scalar_inputs({"$x": 0}),
        tmp_path=tmp_path,
        test_name="requires_case",
    )

    assert_violation(result, "assume_violation", "entry")
    assert result["invalid_input"] is True
    assert result["invalid_reason"] == "requires"


def test_assume_violation_is_reported_as_invalid_input(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      assume $x == 1;
      return;
    }
    """

    result = run_native_case(
        source,
        scalar_inputs({"$x": 0}),
        tmp_path=tmp_path,
        test_name="assume_invalid_case",
    )

    assert_violation(result, "assume_violation", "entry")
    assert result["invalid_input"] is True
    assert result["invalid_reason"] == "assume"


def test_infeasible_goto_is_reported_as_invalid_input(tmp_path):
    source = """
    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      goto left, right;
    left:
      assume $x == 1;
      return;
    right:
      assume $x == 2;
      return;
    }
    """

    result = run_native_case(
        source,
        scalar_inputs({"$x": 0}),
        tmp_path=tmp_path,
        test_name="goto_invalid_case",
    )

    assert_violation(result, "assume_violation", "entry")
    assert result["invalid_input"] is True
    assert result["invalid_reason"] == "infeasible_goto"


def test_final_scalar_summary_reports_return_values(tmp_path):
    source = """
    procedure main($x: int) returns ($r: int);
    implementation {:entrypoint} main($x: int) returns ($r: int) {
    entry:
      $r := $x + 7;
      return;
    }
    """

    result = run_native_case(
        source,
        scalar_inputs({"$x": 5}),
        tmp_path=tmp_path,
        test_name="final_scalar_case",
        return_scalar_summary=True,
    )

    assert_ok(result, blocks={"entry"})
    assert result["final_scalars"]["$x"] == 5
    assert result["final_scalars"]["$r"] == 12
