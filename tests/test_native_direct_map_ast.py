"""Native execution coverage for direct Boogie map select/update AST nodes."""

import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.runner import prepare_native
from interpreter.tests.helpers.boogie_cases import (
    concolic_candidates,
    replay_candidates,
    run_native_case,
    scalar_inputs,
)


@pytest.mark.native
@pytest.mark.parametrize(
    ("declaration", "mode"),
    [
        ("type i32 = int;\nvar $M: [int]int;", "int"),
        ("var $M: [bv64]bv8;", "bv"),
    ],
)
def test_direct_map_update_then_select_executes_one_declared_element(
    declaration, mode, tmp_path
):
    source = f"""
    {declaration}

    procedure main();
    implementation {{:entrypoint}} main() {{
    entry:
      $M := $M[7 := 171];
      assert $M[7] == 171;
      assert $M[8] == 0;
      return;
    }}
    """

    assert prepare_native(parse_boogie(source)).compiled.mode == mode
    result = run_native_case(source, tmp_path=tmp_path)

    assert result["status"] == "ok"


@pytest.mark.native
@pytest.mark.parametrize(
    ("declaration", "mode"),
    [
        ("type i32 = int;\nvar $M.P: [int]int;\nvar $M.Q: [int]int;", "int"),
        ("var $M.P: [bv64]bv8;\nvar $M.Q: [bv64]bv8;", "bv"),
    ],
)
def test_direct_map_equality_is_typed_and_extensional(declaration, mode, tmp_path):
    source = f"""
    {declaration}

    procedure main();
    implementation {{:entrypoint}} main() {{
    entry:
      $M.P := $M.P[7 := 171];
      $M.Q := $M.Q[7 := 171];
      $M.P := $M.P[9 := 0];
      assert $M.P == $M.Q;
      $M.Q := $M.Q[7 := 170];
      assert $M.P != $M.Q;
      return;
    }}
    """

    assert prepare_native(parse_boogie(source)).compiled.mode == mode
    result = run_native_case(source, tmp_path=tmp_path)

    assert result["status"] == "ok"


@pytest.mark.native
def test_concolic_map_equality_uses_concrete_typed_maps(tmp_path):
    source = """
    var $M.P: [bv64]bv8;
    var $M.Q: [bv64]bv8;

    procedure main();
    implementation {:entrypoint} main() {
    entry:
      $M.P := $M.P[7 := 171];
      $M.Q := $M.Q[7 := 171];
      goto equal, different;
    equal:
      assume $M.P == $M.Q;
      return;
    different:
      assume $M.P != $M.Q;
      return;
    }
    """
    inputs = scalar_inputs()
    seed = run_native_case(source, inputs, tmp_path=tmp_path)
    assert seed["status"] == "ok"

    candidates, stats = concolic_candidates(
        source,
        inputs,
        set(seed["explored_blocks"]),
        tmp_path=tmp_path,
        max_solver_queries=4,
    )

    assert isinstance(candidates, list)
    assert stats["unsupported"] == 0


@pytest.mark.native
@pytest.mark.parametrize("n", [0, 3])
def test_structured_while_true_break_has_an_unconditional_body_target(n, tmp_path):
    source = """
    procedure main(n: int);
    implementation {:entrypoint} main(n: int) {
      var i: int;
    entry:
      i := 0;
      while (true) {
        if (i >= n) {
          break;
        }
        i := i + 1;
      }
      assert i == n;
      return;
    }
    """
    inputs = scalar_inputs({"n": n})
    result = run_native_case(source, inputs, tmp_path=tmp_path)

    assert result["status"] == "ok"


@pytest.mark.native
@pytest.mark.parametrize("node", ["select", "update"])
def test_multi_index_direct_map_is_a_typed_lowering_error(node):
    expression = "$M[1, 2]" if node == "select" else "$M[1, 2 := 3]"
    source = f"""
    var $M: [int, int]int;
    var $x: int;

    procedure main();
    implementation {{:entrypoint}} main() {{
    entry:
      $x := {expression};
      return;
    }}
    """
    program = parse_boogie(source)

    with pytest.raises(ValueError, match="only single-index maps"):
        prepare_native(program)


@pytest.mark.native
def test_concolic_direct_map_update_and_select_preserve_symbolic_value(tmp_path):
    source = """
    var $M: [bv64]bv8;

    procedure main($x: int);
    implementation {:entrypoint} main($x: int) {
    entry:
      $M := $M[7 := $x];
      goto seed, target;
    seed:
      assume $M[7] == 0;
      return;
    target:
      assume $M[7] == 66;
      return;
    }
    """
    inputs = scalar_inputs({"$x": 0})
    seed = run_native_case(source, inputs, tmp_path=tmp_path)
    assert seed["status"] == "ok"

    candidates, stats = concolic_candidates(
        source,
        inputs,
        set(seed["explored_blocks"]),
        tmp_path=tmp_path,
        max_solver_queries=16,
    )
    assert candidates, stats
    replayed = replay_candidates(source, candidates, tmp_path=tmp_path)
    assert any("target" in result["explored_blocks"] for result in replayed)
