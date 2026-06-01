from lark import Tree
import pytest

import interpreter.parser.boogie_parser as parser_mod
from interpreter.parser.boogie_parser import (
    BoogieParseError,
    bpl,
    parse_auto,
    parse_boogie,
    parse_decl,
    parse_expr,
    parse_kind,
    parse_stmt,
    parse_type,
)
from interpreter.parser.expression import (
    BinaryExpression,
    Expression,
    IfExpression,
    MapUpdate,
    OldExpression,
    RangeIndex,
    RealLiteral,
    StorageIdentifier,
    Trigger,
)
from interpreter.parser.node import Node
from interpreter.parser.specifications import LoopInvariant
from interpreter.parser.statement import (
    AssertStatement,
    AssignStatement,
    BreakStatement,
    CallStatement,
    HavocStatement,
    ParallelCallStatement,
    PopStatement,
    WhileStatement,
    YieldStatement,
)
from interpreter.parser.type import RealType


def assert_no_lark_trees(value):
    if isinstance(value, Tree):
        pytest.fail(f"raw Lark tree leaked into AST: {value!r}")
    if isinstance(value, Node):
        for child in value.each():
            assert not isinstance(child, Tree)
    elif isinstance(value, list):
        for item in value:
            assert_no_lark_trees(item)


@pytest.mark.parametrize(
    "source, expected_type",
    [
        ("old(x)", OldExpression),
        ("x[7:0]", RangeIndex),
        ("M[x := y]", MapUpdate),
        ("1.25", RealLiteral),
        ("1e-2", RealLiteral),
    ],
)
def test_expression_entrypoints_cover_supported_aliases(source, expected_type):
    parsed = parse_expr(source)
    assert isinstance(parsed, expected_type)
    assert_no_lark_trees(parsed)


@pytest.mark.parametrize(
    "source, op",
    [
        ("x <: y", "<:"),
        ("x ++ y", "++"),
    ],
)
def test_extended_binary_operators(source, op):
    parsed = parse_expr(source)
    assert isinstance(parsed, BinaryExpression)
    assert parsed.op == op
    assert_no_lark_trees(parsed)


def test_quantifier_triggers_are_ast_nodes():
    parsed = parse_expr("(forall x: int :: {x} x > 0)")
    assert parsed.triggers
    assert isinstance(parsed.triggers[0], Trigger)
    assert_no_lark_trees(parsed)


def test_parse_str_prefers_if_then_else_expression_over_statement():
    parsed = bpl("if x then 1 else 2")
    assert isinstance(parsed, IfExpression)


def test_parse_auto_is_explicit_name_for_parse_str_behavior():
    parsed = parse_auto("if x then 1 else 2")
    assert isinstance(parsed, IfExpression)


def test_bpl_known_kind_uses_requested_entrypoint(monkeypatch):
    monkeypatch.setattr(parser_mod, "parse_auto", lambda source: pytest.fail("parse_auto should not run"))
    parsed = bpl("assert true;", kind="stmt")
    assert isinstance(parsed, AssertStatement)


def test_parse_kind_rejects_unknown_kind():
    with pytest.raises(ValueError):
        parse_kind("x", "not-a-kind")


def test_parse_str_still_detects_if_statement():
    parsed = bpl("if (x) { }")
    assert parsed.__class__.__name__ == "IfStatement"


def test_real_type_parses_explicitly():
    parsed = parse_type("real")
    assert isinstance(parsed, RealType)


@pytest.mark.parametrize(
    "source, expected_type",
    [
        ("yield;", YieldStatement),
        ("pop;", PopStatement),
        ("break done;", BreakStatement),
        ("par a | r := b;", ParallelCallStatement),
    ],
)
def test_uncommon_statement_forms_return_ast_nodes(source, expected_type):
    parsed = parse_stmt(source)
    assert isinstance(parsed, expected_type)
    assert_no_lark_trees(parsed)


def test_while_statement_preserves_invariants_and_wildcard_guard():
    parsed = parse_stmt("while (*) invariant true; { x := 1; }")
    assert isinstance(parsed, WhileStatement)
    assert parsed.condition is Expression.Wildcard
    assert len(parsed.invariants) == 1
    assert isinstance(parsed.invariants[0], LoopInvariant)
    assert_no_lark_trees(parsed)


def test_while_statement_preserves_free_invariant():
    parsed = parse_stmt("while (*) free invariant true; { x := 1; }")
    assert isinstance(parsed, WhileStatement)
    assert len(parsed.invariants) == 1
    assert parsed.invariants[0].free is True
    assert_no_lark_trees(parsed)


def test_full_program_has_no_raw_lark_trees():
    parsed = parse_boogie(
        """
        var x: int;
        implementation main() {
          entry:
            x := 1;
            return;
        }
        """
    )
    assert_no_lark_trees(parsed)


@pytest.mark.parametrize("source", ["var x, y: int;", "const x, y: int;"])
def test_multi_name_declaration_parses_as_names_only(source):
    parsed = parse_decl(source)
    assert parsed.names == ["x", "y"]
    assert not hasattr(parsed, "ids")


def test_simplified_list_rules_keep_expected_ast_shape():
    typ = parse_decl("type T A B = int;")
    assert typ.arguments == ["A", "B"]

    const = parse_decl("const x: int <: unique y, z complete;")
    assert len(const.order_spec[0]) == 2
    assert const.order_spec[1] is True

    stmt = parse_stmt("M[x][y] := z;")
    assert isinstance(stmt, AssignStatement)
    assert str(stmt.lhs[0]) == "M[x][y]"

    call = parse_stmt("call forall foo(*, x, *);")
    assert isinstance(call, CallStatement)
    assert len(call.arguments) == 3

    attr_stmt = parse_stmt('assert {:tag "a", "b"} true;')
    assert attr_stmt.attributes[0].values == ['"a"', '"b"']


def test_node_clone_and_copy_are_structural_not_reparse(monkeypatch):
    parsed = parse_decl(
        """
        procedure p(x: int) returns (r: int)
        {
        entry:
          r := x;
          return;
        }
        """
    )
    monkeypatch.setattr(parser_mod, "bpl", lambda source, kind=None: pytest.fail("copy should not reparse"))

    cloned = parsed.clone()
    copied = parsed.copy()
    block_copy = parsed.body.blocks[0].copy()

    assert cloned is not parsed
    assert copied is not parsed
    assert block_copy is not parsed.body.blocks[0]
    assert str(cloned) == str(parsed)
    assert str(copied) == str(parsed)

    cloned.parameters[0].names[0] = "y"
    assert parsed.parameters[0].names == ["x"]


def test_mutable_statement_defaults_are_not_shared():
    first = HavocStatement()
    second = HavocStatement()
    first.identifiers.append(StorageIdentifier(name="x"))
    assert second.identifiers == []


def test_parse_errors_include_kind_and_source_context():
    with pytest.raises(BoogieParseError) as exc_info:
        parse_expr("x +")
    assert exc_info.value.kind == "expr"
    assert exc_info.value.source == "x +"
