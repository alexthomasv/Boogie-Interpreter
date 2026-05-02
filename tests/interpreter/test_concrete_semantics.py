import pytest

from interpreter.tests.helpers.boogie_cases import (
    assert_ok,
    assert_violation,
    run_python_case,
    scalar_inputs,
)

pytestmark = [pytest.mark.interpreter]


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

    result = run_python_case(
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

    result = run_python_case(
        source,
        scalar_inputs({"$x": 5}),
        tmp_path=tmp_path,
        test_name="assert_case",
    )

    assert_violation(result, "assert_violation", "bad")
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

    result = run_python_case(
        source,
        scalar_inputs({"$x": 0}),
        tmp_path=tmp_path,
        test_name="requires_case",
    )

    assert_violation(result, "assume_violation", "entry")
