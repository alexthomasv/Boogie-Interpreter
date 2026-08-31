from __future__ import annotations

from types import SimpleNamespace

import cvc5
import pytest
from cvc5 import Kind

from interpreter.parser.expression import FunctionApplication
from interpreter.utils import utils_cvc5
from interpreter.utils.cvc5_serde import canonical_wire
from interpreter.utils.utils_cvc5 import cvc5_to_boogie_ast


_LIVE_CASES = (
    (Kind.BITVECTOR_AND, 1, "$and.i1"),
    (Kind.BITVECTOR_OR, 8, "$or.i8"),
    (Kind.BITVECTOR_XOR, 32, "$xor.i32"),
    (Kind.BITVECTOR_NOT, 16, "$not.i16"),
    (Kind.BITVECTOR_SHL, 8, "$shl.i8"),
    (Kind.BITVECTOR_LSHR, 32, "$lshr.i32"),
    (Kind.BITVECTOR_ASHR, 64, "$ashr.i64"),
)

_EVALUATION_CASES = (
    (Kind.BITVECTOR_AND, 8, (-1, 0x55)),
    (Kind.BITVECTOR_OR, 8, (-256, 0x81)),
    (Kind.BITVECTOR_XOR, 32, (0x80000000, -1)),
    (Kind.BITVECTOR_NOT, 16, (-2,)),
    (Kind.BITVECTOR_SHL, 8, (0x81, 2)),
    (Kind.BITVECTOR_LSHR, 32, (-1, 31)),
    (Kind.BITVECTOR_ASHR, 64, (1 << 63, 63)),
    (Kind.BITVECTOR_AND, 128, (-1, (1 << 127) | 0x55)),
    (Kind.BITVECTOR_OR, 128, (1 << 127, (1 << 64) | 1)),
    (Kind.BITVECTOR_XOR, 128, (-1, 1 << 127)),
    (Kind.BITVECTOR_SHL, 128, ((1 << 127) | 1, 1)),
    (Kind.BITVECTOR_LSHR, 128, (-1, 127)),
    (Kind.BITVECTOR_ASHR, 128, (1 << 127, 127)),
)


def _solver() -> cvc5.Solver:
    solver = cvc5.Solver()
    solver.setLogic("ALL")
    return solver


def _unsigned_result(solver, kind, width, operands):
    int_to_bv = solver.mkOp(Kind.INT_TO_BITVECTOR, width)
    bitvector_operands = [
        solver.mkTerm(int_to_bv, operand) for operand in operands
    ]
    bitvector_result = solver.mkTerm(kind, *bitvector_operands)
    return solver.mkTerm(Kind.BITVECTOR_UBV_TO_INT, bitvector_result)


@pytest.mark.parametrize("kind,width,intrinsic", _LIVE_CASES)
def test_unsigned_bitwise_live_term_uses_exact_width_aware_intrinsic(
    kind,
    width,
    intrinsic,
):
    solver = _solver()
    integer = solver.getIntegerSort()
    operands = [solver.mkConst(integer, "x")]
    if kind != Kind.BITVECTOR_NOT:
        operands.append(solver.mkConst(integer, "y"))

    ast = cvc5_to_boogie_ast(
        _unsigned_result(solver, kind, width, operands))

    assert isinstance(ast, FunctionApplication)
    assert ast.function.name == intrinsic
    assert " || " not in repr(ast)


@pytest.mark.parametrize(
    "width,intrinsic",
    ((1, "$or.i1"), (8, "$or.i8"), (32, "$or.i32"), (64, "$or.i64")),
)
def test_unsigned_or_serialized_wire_keeps_its_width(width, intrinsic):
    solver = _solver()
    integer = solver.getIntegerSort()
    term = _unsigned_result(
        solver,
        Kind.BITVECTOR_OR,
        width,
        (solver.mkConst(integer, "x"), solver.mkConst(integer, "y")),
    )

    live = cvc5_to_boogie_ast(term)
    wire = cvc5_to_boogie_ast(canonical_wire(term))

    assert repr(wire) == repr(live)
    assert isinstance(wire, FunctionApplication)
    assert wire.function.name == intrinsic


@pytest.mark.parametrize(
    "kind,intrinsic",
    (
        (Kind.BITVECTOR_AND, "$and.i128"),
        (Kind.BITVECTOR_OR, "$or.i128"),
        (Kind.BITVECTOR_XOR, "$xor.i128"),
        (Kind.BITVECTOR_SHL, "$shl.i128"),
        (Kind.BITVECTOR_LSHR, "$lshr.i128"),
        (Kind.BITVECTOR_ASHR, "$ashr.i128"),
    ),
)
def test_unsigned_i128_bitwise_live_and_wire_terms_are_executable(
    kind,
    intrinsic,
):
    solver = _solver()
    integer = solver.getIntegerSort()
    term = _unsigned_result(
        solver,
        kind,
        128,
        (solver.mkConst(integer, "x"), solver.mkConst(integer, "y")),
    )

    for ast in (cvc5_to_boogie_ast(term),
                cvc5_to_boogie_ast(canonical_wire(term))):
        assert isinstance(ast, FunctionApplication)
        assert ast.function.name == intrinsic
        assert " || " not in repr(ast)


def test_nested_unsigned_or_under_int_to_bv_extract_concat_lowers_exactly():
    """Nested freeze projections retain the exact inner unsigned OR."""
    solver = _solver()
    integer = solver.getIntegerSort()
    x = solver.mkConst(integer, "x")
    y = solver.mkConst(integer, "y")
    unsigned_or = _unsigned_result(
        solver, Kind.BITVECTOR_OR, 8, (x, y))
    int_to_bv = solver.mkOp(Kind.INT_TO_BITVECTOR, 8)
    roundtrip = solver.mkTerm(int_to_bv, unsigned_or)
    high_nibble = solver.mkTerm(
        solver.mkOp(Kind.BITVECTOR_EXTRACT, 7, 4), roundtrip)
    cleared_low_nibble = solver.mkTerm(
        Kind.BITVECTOR_CONCAT, high_nibble, solver.mkBitVector(4, 0))
    nested = solver.mkTerm(
        Kind.BITVECTOR_UBV_TO_INT, cleared_low_nibble)

    expected = (
        "((((($or.i8((x % 256), (y % 256)) % 256) / 16) % 16) * 16) + 0)"
    )
    assert repr(cvc5_to_boogie_ast(nested)) == expected
    assert repr(cvc5_to_boogie_ast(canonical_wire(nested))) == expected


@pytest.mark.parametrize("kind,width,values", _EVALUATION_CASES)
def test_unsigned_bitwise_ast_roundtrips_to_the_exact_cvc5_value(
    kind,
    width,
    values,
):
    solver = _solver()
    operands = [solver.mkInteger(str(value)) for value in values]
    original = _unsigned_result(solver, kind, width, operands)
    ast = cvc5_to_boogie_ast(original)

    previous_integer_encoding = utils_cvc5._INTEGER_ENCODING
    utils_cvc5.set_integer_encoding(True)
    try:
        converted = utils_cvc5.convert_expr_cvc5(
            utils_cvc5.generate_cvc5_function_map(solver),
            SimpleNamespace(cvc5_var=lambda _name: None),
            solver,
            ast,
            True,
        )
    finally:
        utils_cvc5.set_integer_encoding(previous_integer_encoding)

    assert solver.simplify(converted) == solver.simplify(original)
