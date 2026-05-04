"""Small AST evaluator for native input-state metadata.

This is intentionally not a program interpreter. It only evaluates the
constant/scalar expressions needed to build Rust-side input metadata.
"""

from interpreter.parser.declaration import AxiomDeclaration, ConstantDeclaration, StorageDeclaration
from interpreter.parser.expression import (
    ArithmeticNegation,
    BinaryExpression,
    BitvectorLiteral,
    BooleanLiteral,
    FunctionApplication,
    IfExpression,
    IntegerLiteral,
    LogicalNegation,
    ProcedureIdentifier,
    StorageIdentifier,
)
from interpreter.parser.type import MapType
from interpreter.utils.bitops import generate_function_map
from interpreter.utils.program import extract_boogie_variables


class StaticEvalError(ValueError):
    pass


_FUNCTIONS = generate_function_map()


def signed_i64(value):
    value = int(value)
    mask = (1 << 64) - 1
    max_i64 = (1 << 63) - 1
    if value > max_i64 or value < -(max_i64 + 1):
        value &= mask
        if value > max_i64:
            value -= (1 << 64)
    return value


def expr_base_ptr(expr):
    vars_ = extract_boogie_variables(expr)
    if len(vars_) != 1:
        raise StaticEvalError(
            f"Expected one base pointer variable in {expr}, got {vars_}"
        )
    return next(iter(vars_)).name


def eval_static(expr, values: dict[str, int | bool]):
    if isinstance(expr, BooleanLiteral):
        return bool(expr.value)
    if isinstance(expr, IntegerLiteral):
        return int(expr.value)
    if isinstance(expr, BitvectorLiteral):
        return int(expr.value)
    if isinstance(expr, (StorageIdentifier, ProcedureIdentifier)):
        if expr.name not in values:
            raise StaticEvalError(f"Unknown static identifier: {expr.name}")
        return values[expr.name]
    if isinstance(expr, LogicalNegation):
        return not bool(eval_static(expr.expression, values))
    if isinstance(expr, ArithmeticNegation):
        return -int(eval_static(expr.expression, values))
    if isinstance(expr, IfExpression):
        branch = expr.then if bool(eval_static(expr.condition, values)) else expr.else_
        return eval_static(branch, values)
    if isinstance(expr, BinaryExpression):
        lhs = eval_static(expr.lhs, values)
        rhs = eval_static(expr.rhs, values)
        return _eval_binary(expr.op, lhs, rhs)
    if isinstance(expr, FunctionApplication):
        name = expr.function.name
        args = [eval_static(arg, values) for arg in expr.arguments]
        if name == "$isExternal":
            return False
        spec = _FUNCTIONS.get(name)
        if spec is None or spec[0] is None:
            raise StaticEvalError(f"Unsupported static function: {name}")
        return spec[0](*args)
    raise StaticEvalError(f"Unsupported static expression: {expr!r}")


def _eval_binary(op, lhs, rhs):
    if op == "==>":
        return (not bool(lhs)) or bool(rhs)
    if op == "<==>":
        return bool(lhs) == bool(rhs)
    if op == "<:":
        return lhs == rhs

    spec = _FUNCTIONS.get(op)
    if spec is not None and spec[0] is not None:
        return spec[0](lhs, rhs)

    if op == "%":
        return int(lhs) % int(rhs)
    if op == "++":
        raise StaticEvalError("Bitvector concat is not supported in static metadata")
    raise StaticEvalError(f"Unsupported static binary operator: {op}")


def initial_static_values(program) -> dict[str, int | bool]:
    values: dict[str, int | bool] = {}
    for decl in program.declarations:
        if isinstance(decl, ConstantDeclaration):
            for name in decl.names:
                values.setdefault(name, 0)
        elif isinstance(decl, StorageDeclaration) and not isinstance(decl.type, MapType):
            for name in decl.names:
                values.setdefault(name, 0)
    return values


def compute_static_values(program) -> dict[str, int | bool]:
    values = initial_static_values(program)
    for decl in program.declarations:
        if not isinstance(decl, AxiomDeclaration):
            continue
        expr = decl.expression
        if not isinstance(expr, BinaryExpression) or expr.op != "==":
            continue
        if not isinstance(expr.lhs, StorageIdentifier):
            continue
        try:
            values[expr.lhs.name] = eval_static(expr.rhs, values)
        except StaticEvalError:
            continue
    return values


def compute_static_scalars(program) -> dict[str, int]:
    values = compute_static_values(program)
    return {
        name: signed_i64(value)
        for name, value in values.items()
        if isinstance(value, (int, bool)) and int(value) != 0
    }


def offset_delta(expr, base_ptr: str, static_values: dict[str, int | bool]) -> int:
    values = dict(static_values)
    for var in extract_boogie_variables(expr):
        values.setdefault(var.name, 0)
    values[base_ptr] = 0
    return signed_i64(eval_static(expr, values))
