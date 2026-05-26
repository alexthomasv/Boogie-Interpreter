"""Desugar structured Boogie ``while`` statements into goto-form blocks.

The native interpreter's lowering pass (``interpreter/native/src/lowering.rs``)
only understands goto-form blocks — there is no opcode for ``WhileStatement``.
Programs synthesised by ``diffprod.corerel.reify`` carry structured ``while``
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
    GotoStatement,
    IfStatement,
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
    """Walk the block list once, expanding any ``while`` statements into
    additional blocks. Body block names are made unique against the input
    list so we never collide with an existing label."""
    existing_labels: set[str] = set()
    for block in blocks:
        for name in getattr(block, "names", []) or [block.name]:
            if name:
                existing_labels.add(name)

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
        expanded.extend(_expand_block(block, fresh))
    return expanded


def _expand_block(block, fresh) -> list:
    """Return a list of blocks equivalent to ``block`` but with
    ``WhileStatement`` rewritten into goto-form. Most blocks emit just
    themselves unchanged."""
    has_while = any(isinstance(s, WhileStatement) for s in block.statements)
    if not has_while:
        return [block]

    out: list = []
    current_stmts: list[Statement] = []
    current_label = block.name
    current_names = list(block.names) if block.names else [block.name]

    def emit(stmts: list[Statement], names: list[str], label: str) -> None:
        new_block = Block(names=list(names), statements=list(stmts))
        new_block.name = label
        out.append(new_block)

    pending_names = current_names
    for stmt in block.statements:
        if isinstance(stmt, WhileStatement):
            head_label = fresh(f"loop_head_{block.name}")
            body_label = fresh(f"loop_body_{block.name}")
            exit_label = fresh(f"loop_exit_{block.name}")

            # Close the current block by jumping to the loop head.
            current_stmts = list(current_stmts) + [_make_goto(head_label)]
            emit(current_stmts, pending_names, current_label)

            # Loop head: branch to body or exit.
            emit(
                [_make_goto(body_label, exit_label)],
                [head_label],
                head_label,
            )

            # Loop body: assume guard, run body, branch back to the head.
            # NOTE: nested while-statements inside ``stmt.blocks`` are NOT
            # recursively desugared — they stay as ``WhileStatement`` AST
            # nodes so the Rust VM can lower them into ``Stmt::While`` and
            # execute them natively. This keeps structured nested loops
            # working without having to flatten them into multiple gotos.
            body_blocks = list(stmt.blocks)
            body_stmts: list[Statement] = [_make_assume(stmt.condition)]
            for inner_block in body_blocks:
                body_stmts.extend(inner_block.statements)
            body_stmts.append(_make_goto(head_label))
            emit(body_stmts, [body_label], body_label)

            # The exit block holds the assume on the negated guard, and
            # any *trailing* statements from the original block follow it.
            negated = LogicalNegation()
            negated.expression = stmt.condition
            current_stmts = [_make_assume(negated)]
            current_label = exit_label
            pending_names = [exit_label]
            continue

        current_stmts.append(stmt)

    # Final tail block (with anything after the last while).
    emit(current_stmts, pending_names, current_label)
    return out
