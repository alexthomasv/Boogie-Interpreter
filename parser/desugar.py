"""Desugar structured Boogie ``while`` statements into goto-form blocks.

The native interpreter's lowering pass (``interpreter/native/src/lowering.rs``)
only understands goto-form blocks — there is no opcode for ``WhileStatement``.
Programs synthesized by ``deltarel.product_v2`` carry structured ``while``
loops because the verifier wants the inductive ``invariant`` annotations.
This module rewrites those loops into the equivalent goto cycle so the
native interpreter can execute the program for trace-driven cost evaluation.

The rewrite is local. For a block ``B`` ending its statements with::

    while (g) invariant I; { body_stmts }

it produces four blocks::

    B { leading_stmts; goto loop_head_<B>_<n>; }
    loop_head_<B>_<n> { goto loop_body_<B>_<n>, loop_exit_<B>_<n>; }
    loop_body_<B>_<n> { assume g; body_stmts; goto loop_head_<B>_<n>; }
    loop_exit_<B>_<n> { assume !g; <original transfer of B> }

Invariants are dropped — the interpreter does not check them; they are a
verifier-only annotation. Body statements may themselves contain
``WhileStatement`` and are recursively desugared.

The pass is idempotent and a no-op on programs without any
``WhileStatement`` (the SMACK-generated common case), so it is safe to
hook unconditionally into the native runner.
"""
from __future__ import annotations

from .expression import LabelIdentifier, LogicalNegation
from .statement import (
    AssumeStatement,
    BreakStatement,
    GotoStatement,
    IfStatement,
    ReturnStatement,
    Statement,
    WhileStatement,
    Block,
)


def _make_assume(expression) -> AssumeStatement:
    stmt = AssumeStatement()
    stmt.expression = expression
    return stmt


def _make_goto(*labels: str) -> GotoStatement:
    stmt = GotoStatement()
    stmt.identifiers = [LabelIdentifier(name=label) for label in labels]
    return stmt


def desugar_while_statements(program) -> None:
    """Mutate ``program`` in place, rewriting every structured ``while``
    statement reachable from any ``ImplementationDeclaration`` body into
    goto-form blocks.

    Returns ``None``. The same ``program`` object is reused; the parser
    keeps Python references on the AST (``label_to_pc``, etc.) so we
    avoid swapping out the body wholesale.
    """
    from .declaration import ImplementationDeclaration

    declarations = getattr(program, "declarations", None) or []
    for decl in declarations:
        if not isinstance(decl, ImplementationDeclaration):
            continue
        body = getattr(decl, "body", None)
        if body is None:
            continue
        body.blocks = _expand_blocks(list(body.blocks))


def _expand_blocks(blocks):
    """Expand every reachable structured loop into ordinary CFG blocks."""
    existing_labels: set[str] = set()
    for block in blocks:
        for name in getattr(block, "names", []) or [block.name]:
            if name:
                existing_labels.add(str(name))

    counter = [0]

    def fresh(prefix: str) -> str:
        while True:
            counter[0] += 1
            candidate = f"{prefix}_{counter[0]}"
            if candidate not in existing_labels:
                existing_labels.add(candidate)
                return candidate

    expanded: list[Block] = []
    for block in blocks:
        if not _contains_while(block.statements):
            expanded.append(block)
            continue
        _lower_sequence(
            expanded,
            fresh,
            str(block.name),
            list(block.names) if block.names else [str(block.name)],
            list(block.statements),
            fallthrough=None,
            break_targets=(),
        )
    return expanded


def _contains_while(statements) -> bool:
    for statement in statements:
        if isinstance(statement, WhileStatement):
            return True
        if isinstance(statement, IfStatement):
            if _contains_while(_structured_body(statement.blocks)):
                return True
            else_value = statement.else_
            if isinstance(else_value, IfStatement):
                if _contains_while([else_value]):
                    return True
            elif _contains_while(_structured_body(else_value or [])):
                return True
    return False


def _structured_body(items) -> list[Statement]:
    statements: list[Statement] = []
    for item in items or []:
        if isinstance(item, Block):
            statements.extend(item.statements)
        else:
            statements.append(item)
    return statements


def _emit(out, label: str, names, statements) -> None:
    block = Block(names=list(names), statements=list(statements))
    block.name = label
    out.append(block)


def _break_target(statement, break_targets) -> str:
    if not break_targets:
        raise ValueError("break statement is not enclosed by a structured loop")
    identifier = getattr(statement, "identifier", None)
    if identifier is None:
        return break_targets[-1][1]
    name = str(getattr(identifier, "name", identifier))
    for labels, target in reversed(break_targets):
        if name in labels:
            return target
    raise ValueError(f"break target {name!r} does not name an enclosing loop")


def _lower_sequence(
    out,
    fresh,
    label: str,
    names,
    statements,
    *,
    fallthrough: str | None,
    break_targets,
) -> None:
    """Lower one structured statement sequence using explicit continuations."""
    leading: list[Statement] = []
    for index, statement in enumerate(statements):
        trailing = list(statements[index + 1 :])

        if isinstance(statement, BreakStatement):
            leading.append(_make_goto(_break_target(statement, break_targets)))
            _emit(out, label, names, leading)
            return

        if isinstance(statement, (GotoStatement, ReturnStatement)):
            leading.append(statement)
            _emit(out, label, names, leading)
            return

        if isinstance(statement, IfStatement):
            then_label = fresh(f"if_then_{label}")
            else_label = fresh(f"if_else_{label}")
            continuation = fresh(f"if_cont_{label}")
            leading.append(_make_goto(then_label, else_label))
            _emit(out, label, names, leading)

            then_statements = [_make_assume(statement.condition)]
            then_statements.extend(_structured_body(statement.blocks))
            _lower_sequence(
                out,
                fresh,
                then_label,
                [then_label],
                then_statements,
                fallthrough=continuation,
                break_targets=break_targets,
            )

            negated = LogicalNegation()
            negated.expression = statement.condition
            else_statements = [_make_assume(negated)]
            if isinstance(statement.else_, IfStatement):
                else_statements.append(statement.else_)
            else:
                else_statements.extend(_structured_body(statement.else_ or []))
            _lower_sequence(
                out,
                fresh,
                else_label,
                [else_label],
                else_statements,
                fallthrough=continuation,
                break_targets=break_targets,
            )
            _lower_sequence(
                out,
                fresh,
                continuation,
                [continuation],
                trailing,
                fallthrough=fallthrough,
                break_targets=break_targets,
            )
            return

        if isinstance(statement, WhileStatement):
            head_label = fresh(f"loop_head_{label}")
            body_label = fresh(f"loop_body_{label}")
            exit_label = fresh(f"loop_exit_{label}")
            leading.append(_make_goto(head_label))
            _emit(out, label, names, leading)
            _emit(
                out,
                head_label,
                [head_label],
                [_make_goto(body_label, exit_label)],
            )

            loop_labels = {str(item) for item in names if str(item)}
            loop_labels.add(label)
            body_statements = [_make_assume(statement.condition)]
            body_statements.extend(_structured_body(statement.blocks))
            _lower_sequence(
                out,
                fresh,
                body_label,
                [body_label],
                body_statements,
                fallthrough=head_label,
                break_targets=(*break_targets, (loop_labels, exit_label)),
            )

            negated = LogicalNegation()
            negated.expression = statement.condition
            exit_statements = [_make_assume(negated), *trailing]
            _lower_sequence(
                out,
                fresh,
                exit_label,
                [exit_label],
                exit_statements,
                fallthrough=fallthrough,
                break_targets=break_targets,
            )
            return

        leading.append(statement)

    if fallthrough is not None:
        leading.append(_make_goto(fallthrough))
    _emit(out, label, names, leading)
