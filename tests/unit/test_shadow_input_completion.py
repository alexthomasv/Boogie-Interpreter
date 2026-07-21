"""Pipeline completion for compact cross-product input files."""

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.utils.inputs import (
    Input,
    ProgramInputs,
    complete_declared_shadow_inputs,
)


def _cross_product_program():
    return parse_boogie(
        """
        procedure main($x: int, $x.shadow: int, $p: ref, $p.shadow: ref);
        implementation {:entrypoint} main(
            $x: int, $x.shadow: int, $p: ref, $p.shadow: ref
        ) {
        entry:
          return;
        }
        """
    )


def test_completion_deep_clones_each_missing_declared_shadow():
    base_buffer = {"contents": "0x0102", "size": 2}
    inputs = ProgramInputs({
        "$x": Input(name="$x", private=False, value=7),
        "$p": Input(name="$p", private=True, buffers=[base_buffer]),
    })

    completed = complete_declared_shadow_inputs(_cross_product_program(), inputs)

    assert set(completed.variables) == {"$x", "$x.shadow", "$p", "$p.shadow"}
    assert completed.variables["$x.shadow"].value == 7
    assert completed.variables["$p.shadow"].buffers == [base_buffer]
    assert completed.variables["$p.shadow"] is not completed.variables["$p"]
    assert completed.variables["$p.shadow"].buffers is not completed.variables["$p"].buffers
    assert set(inputs.variables) == {"$x", "$p"}


def test_completion_preserves_explicit_shadow_as_authoritative():
    explicit = Input(name="$x.shadow", private=False, value=99)
    inputs = ProgramInputs({
        "$x": Input(name="$x", private=False, value=7),
        "$x.shadow": explicit,
    })

    completed = complete_declared_shadow_inputs(_cross_product_program(), inputs)

    assert completed.variables["$x.shadow"] is explicit
    assert completed.variables["$x.shadow"].value == 99
    assert "$p.shadow" not in completed.variables


def test_completion_is_noop_without_a_declared_shadow_lane():
    program = parse_boogie(
        """
        procedure main($x: int);
        implementation {:entrypoint} main($x: int) {
        entry:
          return;
        }
        """
    )
    inputs = ProgramInputs({"$x": Input(name="$x", private=False, value=7)})

    assert complete_declared_shadow_inputs(program, inputs) is inputs
