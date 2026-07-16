"""Regression gate: the native Rust inliner (`swoosh_interp.inline_lower`) must
stay behaviorally equivalent to the proof-facing AST inliner.

Fast tier (default): hand-written multi-procedure programs whose in-body asserts
encode correctness, run through BOTH inliners — exercises local-name collision,
param/return binding, multi-instance, nesting, in-callee branches, residual
calls, and cyclic-inline rejection.

Slow tier (`-m slow`): per-benchmark structural + behavioral differential on real
SMACK output (FC + CT crypto).

See tools/test_inline_behavioral.py and tools/test_inline_equiv.py for the harness.
"""
import pytest

pytestmark = [pytest.mark.differential, pytest.mark.native]


@pytest.fixture(scope="module")
def behavioral():
    pytest.importorskip("swoosh_interp")
    from tools import test_inline_behavioral as mod
    return mod


def test_inline_lower_behavioral_cases(behavioral, tmp_path):
    """Each hand-written case: Rust and AST inliners must pass the assertions."""
    for name, source, inputs in behavioral.CASES:
        assert behavioral.run_case(name, source, inputs, tmp_path / name), name


def test_inline_lower_bounds_recursion(behavioral):
    """A cyclic call graph must be BOUNDED like Boogie {:inline 1} — the recursive
    call becomes a residual havoc — so inline_lower terminates + succeeds (BearSSL's
    T0 handshake state machine is genuinely recursive)."""
    assert behavioral.check_recursion_guard()


@pytest.mark.parametrize("seed", range(8))
def test_inline_lower_fuzz(behavioral, seed, tmp_path):
    """Random nested affine chains with $t collisions at every level — Rust must
    match the AST path and satisfy the composed-result asserts."""
    source, inputs = behavioral.gen_random_chain(seed)
    assert behavioral.run_case(f"fuzz{seed}", source, inputs, tmp_path), f"fuzz seed {seed}"


@pytest.mark.slow
@pytest.mark.parametrize("name,ct", [
    ("c2i_002", False),
    ("sha256", False),
    ("aes_cbc_ct", True),
    ("aes_gcm_ct", True),
])
def test_inline_lower_benchmark_differential(name, ct):
    """Real-SMACK differential: globals preserved + final memory matches AST."""
    pytest.importorskip("swoosh_interp")
    import os
    from tools.test_inline_equiv import build_programs, structural_compare, behavioral_compare

    bpl = f"target/swoosh/bpl/{name}.bpl"
    if not os.path.exists(bpl):
        pytest.skip(f"{bpl} not built")
    pre, ast_inlined = build_programs(bpl, ct)
    assert structural_compare(name, pre, ast_inlined), f"{name}: structural"
    assert behavioral_compare(name, pre, ast_inlined), f"{name}: behavioral"
