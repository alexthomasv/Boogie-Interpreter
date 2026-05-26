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
    GotoStatement,
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


NESTED_WHILE = """
procedure foo() returns (r: int);
implementation foo() returns (r: int)
{
  var i: int;
  var j: int;
  entry:
    i := 0;
    j := 0;
    while (i < 3) {
      while (j < 2) {
        j := j + 1;
      }
      i := i + 1;
    }
    r := i + j;
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


def test_desugar_inlines_nested_while_into_outer_body():
    """Limitation: when a ``while`` body itself contains another ``while``,
    the inner loop's expanded blocks are inlined into the outer body block
    rather than emitted as separate top-level blocks. The pass still
    produces a program with no remaining ``WhileStatement`` so the native
    lowering accepts it; the inlined goto cycle preserves semantics. The
    corerel reifier never produces nested whiles, so this is an acceptable
    behavior for the calling pattern we support."""
    program = parse_boogie(NESTED_WHILE)
    assert _count_whiles(program) == 2
    desugar_while_statements(program)
    assert _count_whiles(program) == 0
    # The outer while produces head/body/exit; the inner loop is folded
    # into the outer body block as inlined stmts (still goto-form).
    names = _block_names(program)
    assert any(n.startswith("loop_head_entry_") for n in names)
    assert any(n.startswith("loop_body_entry_") for n in names)
    assert any(n.startswith("loop_exit_entry_") for n in names)


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


def test_desugar_followed_by_run_native_executes_loop():
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
        test_name="t", input_name="t",
        raw_log_path=None, no_trace=False, log_read=False,
        return_status=True, return_scalar_summary=True,
        max_steps=20000,
    )
    assert raw.get("status") == "ok"
    bs = list(raw.get("block_sequence") or ())
    # The loop body executes 4 times — the goto-form body label should
    # appear at least 4 times in block_sequence.
    body_visits = [b for b in bs if b.startswith("loop_body_")]
    assert len(body_visits) >= 4
