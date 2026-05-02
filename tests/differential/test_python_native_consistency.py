import pytest

from interpreter.tests.helpers.boogie_cases import run_both_case, scalar_inputs

pytestmark = [pytest.mark.differential, pytest.mark.native]


def test_python_native_match_simple_branch(tmp_path):
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

    explored, native = run_both_case(
        source,
        scalar_inputs({"$x": 3}),
        tmp_path=tmp_path,
        test_name="diff_branch",
    )

    assert explored == {"entry", "left"}
    assert native["status"] == "ok"


def test_python_native_match_assert_violation(tmp_path):
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

    with pytest.raises(AssertionError) as exc:
        run_both_case(
            source,
            scalar_inputs({"$x": 1}),
            tmp_path=tmp_path,
            test_name="diff_assert",
        )

    assert "block='bad'" in str(exc.value) or "block=bad" in str(exc.value)
