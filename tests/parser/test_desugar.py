"""Tests for the structured-``while`` desugar pass.

The native interpreter (``interpreter/native``) only handles goto-form
blocks. ``desugar_while_statements`` rewrites Boogie programs that use
the structured ``while`` form into the equivalent goto cycle so they
can be executed by ``run_native``.
"""
import pytest

from interpreter.parser.boogie_parser import parse_boogie
from interpreter.parser.declaration import ImplementationDeclaration
from interpreter.parser.desugar import desugar_while_statements
from interpreter.parser.statement import (
    AssumeStatement,
    BreakStatement,
    GotoStatement,
    IfStatement,
    WhileStatement,
)


def _impl(program):
    for d in program.declarations:
        if isinstance(d, ImplementationDeclaration) and d.body is not None:
            return d
    raise AssertionError("no implementation found")


def _count_whiles(program) -> int:
    """Recursively count ``WhileStatement`` nodes anywhere in the body
    (including those nested inside another while's blocks)."""
    n = 0

    def walk_stmts(stmts):
        nonlocal n
        for s in stmts:
            if isinstance(s, WhileStatement):
                n += 1
                for inner_block in s.blocks:
                    walk_stmts(inner_block.statements)

    for d in program.declarations:
        if not isinstance(d, ImplementationDeclaration) or d.body is None:
            continue
        for blk in d.body.blocks:
            walk_stmts(blk.statements)
    return n


def _count_structured_control(program) -> tuple[int, int, int]:
    counts = [0, 0, 0]

    def walk(statements):
        for statement in statements:
            if isinstance(statement, WhileStatement):
                counts[0] += 1
                for block in statement.blocks:
                    walk(block.statements)
            elif isinstance(statement, IfStatement):
                counts[1] += 1
                for block in statement.blocks:
                    walk(block.statements)
                else_value = statement.else_
                if isinstance(else_value, IfStatement):
                    walk([else_value])
                else:
                    for block in else_value or []:
                        walk(block.statements)
            elif isinstance(statement, BreakStatement):
                counts[2] += 1

    for declaration in program.declarations:
        if isinstance(declaration, ImplementationDeclaration) and declaration.body:
            for block in declaration.body.blocks:
                walk(block.statements)
    return tuple(counts)


def _block_names(program) -> list[str]:
    impl = _impl(program)
    return [b.name for b in impl.body.blocks]


SIMPLE_WHILE = """
procedure foo() returns (r: int);
implementation foo() returns (r: int)
{
  var i: int;
  entry:
    i := 0;
    while (i < 5) {
      i := i + 1;
    }
    r := i;
    return;
}
"""


COUNT_WHILE = """
procedure foo(n: int) returns (r: int);
implementation foo(n: int) returns (r: int)
{
  var i: int;
  entry:
    i := 0;
    while (i < n) {
      i := i + 1;
    }
    r := i;
    return;
}
"""


def test_desugar_replaces_simple_while_with_goto_cycle():
    """One ``while`` becomes one entry stub + head + body + exit (4 blocks)
    where the original entry contributed one. The total grows by 3."""
    program = parse_boogie(SIMPLE_WHILE)
    assert _count_whiles(program) == 1
    before = len(_block_names(program))

    desugar_while_statements(program)

    assert _count_whiles(program) == 0
    after = len(_block_names(program))
    assert after == before + 3


def test_desugar_preserves_existing_blocks():
    """Blocks without any ``while`` are left untouched (same name, same
    statement list, same identity-equal references)."""
    program = parse_boogie("""
procedure trivial() returns ();
implementation trivial() returns ()
{
  entry:
    assume true;
    return;
}
""")
    impl = _impl(program)
    original_block = impl.body.blocks[0]
    original_name = original_block.name
    original_stmts = list(original_block.statements)
    desugar_while_statements(program)
    impl = _impl(program)
    assert len(impl.body.blocks) == 1
    assert impl.body.blocks[0].name == original_name
    # Statements unchanged (no rewrite happened).
    assert list(impl.body.blocks[0].statements) == original_stmts


def test_desugar_idempotent_on_program_without_while():
    """Running the pass twice on a desugared program is a no-op."""
    program = parse_boogie(COUNT_WHILE)
    desugar_while_statements(program)
    snapshot = _block_names(program)
    desugar_while_statements(program)
    assert _block_names(program) == snapshot


def test_desugar_emits_loop_head_body_exit_labels():
    """Naming convention: ``loop_head_<orig>_*`` + ``loop_body_<orig>_*`` +
    ``loop_exit_<orig>_*``. trace_guidance.py:519 keys off the ``loop_head_``
    prefix, so this is load-bearing."""
    program = parse_boogie(SIMPLE_WHILE)
    desugar_while_statements(program)
    names = _block_names(program)
    assert any(name.startswith("loop_head_entry_") for name in names)
    assert any(name.startswith("loop_body_entry_") for name in names)
    assert any(name.startswith("loop_exit_entry_") for name in names)


def test_desugar_body_block_starts_with_assume_guard():
    """The loop-body block must begin with ``assume guard`` so the goto
    branch resolution picks it iff the guard holds (``assume_cond`` in
    ``opcodes.rs``). Without this the interpreter would always fall
    through to the exit even when the guard is true."""
    program = parse_boogie(SIMPLE_WHILE)
    desugar_while_statements(program)
    impl = _impl(program)
    body_blocks = [b for b in impl.body.blocks if b.name.startswith("loop_body_")]
    assert len(body_blocks) == 1
    body = body_blocks[0]
    first = body.statements[0]
    assert isinstance(first, AssumeStatement)


def test_desugar_exit_block_starts_with_assume_negated_guard():
    """Symmetric: exit block needs ``assume !guard`` for branch resolution."""
    program = parse_boogie(SIMPLE_WHILE)
    desugar_while_statements(program)
    impl = _impl(program)
    exit_blocks = [b for b in impl.body.blocks if b.name.startswith("loop_exit_")]
    assert len(exit_blocks) == 1
    exit_block = exit_blocks[0]
    first = exit_block.statements[0]
    assert isinstance(first, AssumeStatement)
    # The condition is the negation of the original guard.
    from interpreter.parser.expression import LogicalNegation
    assert isinstance(first.expression, LogicalNegation)


def test_desugar_lowers_break_nested_in_if_to_loop_exit():
    program = parse_boogie("""
procedure foo(n: int) returns (r: int);
implementation foo(n: int) returns (r: int)
{
  var i: int;
  entry:
    i := 0;
    while (true) {
      if (i >= n) {
        break;
      }
      i := i + 1;
    }
    r := i;
    return;
}
""")
    assert _count_structured_control(program) == (1, 1, 1)

    desugar_while_statements(program)

    assert _count_structured_control(program) == (0, 0, 0)
    impl = _impl(program)
    exit_labels = {
        block.name
        for block in impl.body.blocks
        if block.name.startswith("loop_exit_")
    }
    assert len(exit_labels) == 1
    assert any(
        isinstance(statement, GotoStatement)
        and {identifier.name for identifier in statement.identifiers} == exit_labels
        for block in impl.body.blocks
        for statement in block.statements
    )


def test_desugar_lowers_nested_loops_and_their_breaks():
    program = parse_boogie("""
procedure foo() returns ();
implementation foo() returns ()
{
  entry:
    while (true) {
      while (true) {
        break;
      }
      break;
    }
    return;
}
""")

    desugar_while_statements(program)

    assert _count_structured_control(program) == (0, 0, 0)
    names = _block_names(program)
    assert len([name for name in names if name.startswith("loop_head_")]) == 2
    assert len([name for name in names if name.startswith("loop_exit_")]) == 2


def test_desugar_rejects_break_that_does_not_name_an_enclosing_loop():
    program = parse_boogie("""
procedure foo() returns ();
implementation foo() returns ()
{
  entry:
    while (true) {
      break missing;
    }
    return;
}
""")

    with pytest.raises(ValueError, match="does not name an enclosing loop"):
        desugar_while_statements(program)


def test_desugar_is_no_op_on_smack_style_goto_form():
    """Programs already in goto-form (the SMACK-generated common case) are
    untouched — block count and labels are unchanged."""
    program = parse_boogie("""
procedure foo() returns ();
implementation foo() returns ()
{
  entry:
    goto bb0;
  bb0:
    assume true;
    goto bb1;
  bb1:
    return;
}
""")
    before = _block_names(program)
    desugar_while_statements(program)
    after = _block_names(program)
    assert before == after


def test_desugar_followed_by_run_native_executes_loop(tmp_path):
    """End-to-end sanity: a program with a structured ``while`` runs to
    completion under ``run_native`` after the desugar pass (which the
    runner applies automatically)."""
    pytest.importorskip("swoosh_interp")
    from interpreter.runner import run_native
    from interpreter.utils.inputs import Input, ProgramInputs

    program = parse_boogie(
        COUNT_WHILE.replace(
            "implementation foo(",
            "implementation {:entrypoint} foo(",
            1,
        )
    )
    inputs = ProgramInputs({"n": Input(name="n", private=False, value=4)})
    raw = run_native(
        program, inputs,
        input_name="t",
        raw_log_path=tmp_path / "t.raw.zst", no_trace=False, log_read=False,
        return_status=True, return_scalar_summary=True,
        max_steps=20000,
    )
    assert raw.get("status") == "ok"
    # run_native never returns the per-entry block sequence
    # (return_block_sequence=False on its execute_inputs call), so assert
    # loop execution via explored blocks + the final counter values: the
    # goto-form body block was entered, and it ran exactly n times.
    explored = set(raw.get("explored_blocks") or ())
    assert any(b.startswith("loop_body_") for b in explored), explored
    scalars = raw.get("final_scalars") or {}
    assert scalars.get("i") == 4
    assert scalars.get("r") == 4
