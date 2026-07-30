"""Safety bounds and precondition preservation for generated seed inputs."""

import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.parser.declaration import ProcedureDeclaration
from interpreter.utils.gen_input import (
    _apply_input_equalities,
    _apply_seed_variant,
)
from interpreter.utils.inputs import input_equalities_from_requires

def test_random_seed_scalars_are_deterministic_and_trace_bounded():
    entries = [
        {"var": "$i0", "private": False, "value": 0},
        {"var": "$i1", "private": False, "value": 0},
    ]
    repeated = [dict(entry) for entry in entries]

    _apply_seed_variant(entries, "random")
    _apply_seed_variant(repeated, "random")

    values = [entry["value"] for entry in entries]
    assert values == [entry["value"] for entry in repeated]
    assert all(0 <= value <= 256 for value in values)


def _procedure(source):
    program = parse_boogie(source)
    return next(
        declaration
        for declaration in program.declarations
        if type(declaration) is ProcedureDeclaration
    )


def test_input_equalities_use_only_typed_entry_parameters():
    proc = _procedure(
        """
        var g: int;
        procedure {:entrypoint} main(a: int, b: int, c: bool, d: bool)
          returns (r: int);
          requires {:swoosh_input_eq} a == b;
          requires c == d;
          requires g == a;
          ensures r == a;
        implementation {:entrypoint} main(a: int, b: int, c: bool, d: bool)
          returns (r: int) {
        entry: r := a; return;
        }
        """
    )

    assert input_equalities_from_requires(proc, ["a", "b", "c", "d"]) == [
        ("a", "b"),
        ("c", "d"),
    ]
    assert input_equalities_from_requires(proc, []) == []


def test_input_equalities_reject_type_mismatch_and_preserve_unrelated_fields():
    proc = _procedure(
        """
        procedure {:entrypoint} main(a: int, b: bool);
          requires a == b;
        implementation {:entrypoint} main(a: int, b: bool) {
        entry: return;
        }
        """
    )
    with pytest.raises(ValueError, match="matching parameter types"):
        input_equalities_from_requires(proc, ["a", "b"])

    entries = [
        {"var": "a", "private": False, "value": 7},
        {"var": "b", "private": False, "value": 9},
        {"var": "c", "private": False, "value": 11},
    ]
    _apply_input_equalities(entries, [("a", "b")])
    assert [entry["value"] for entry in entries] == [7, 7, 11]
