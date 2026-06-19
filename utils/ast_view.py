"""Structured AST view of a proof obligation, derived from its cvc5 ``Term``.

The anvil LLM agents (proposer / refiner / synthesizer) reason about an
obligation from its Boogie *surface string* (``$sge.i32($i0,$i1) == 1``), which
hides the real structure the verifier and the DSL matcher operate on — the cvc5
AST ``(= (ite (>= $i0 $i1) 1 0) 1)``. This module walks the live cvc5 ``Term``
into JSON-friendly, LLM-ready structures so the agents reason on the *actual*
tree instead of re-deriving it (lossily) from a string.

It is intentionally **core-only**: it imports nothing from ``anvil`` and only the
cvc5 ``Kind`` enum, so the verifier (``src/loop/freeze.py``) can derive the view
once, at freeze time, from the exact Term it proved — the single source of truth
every agent then reads off the freeze payload (no re-parse, no sort/width drift).

- :func:`term_to_pattern` — cvc5 ``Term`` → structural ``pattern`` dict using the
  cvc5 ``Kind`` name directly (``EQUAL``/``ITE``/``GEQ``/``BITVECTOR_SGE``); the
  DSL matcher accepts uppercase Kind names, so no alias table is needed. Variable
  leaves → ``{capture: name}``; integer/bv literals → ``{const: N}``.
- :func:`term_to_emit` — same walk in *emit* form (one-key op mappings,
  ``{gte: [...]}``); raises :class:`UnsupportedShape` for a Kind with no emit op.
- :func:`derive_structural_template` — full ``structural_template`` (pattern +
  emit) with the materialized-comparison decomposition (``eq(ite(GUARD,1,0),1)``
  → emit ``GUARD``); returns ``{"unsupported": reason}`` (never silent ``None``)
  for shapes it can't handle, so callers know they're on their own and we get
  telemetry on the next coverage gap.
- :func:`build_obligation_ast_view` — the bounded, JSON-serializable view shipped
  on the freeze payload and rendered into agent prompts.
"""

from __future__ import annotations

from typing import Any

from cvc5 import Kind


class UnsupportedShape(Exception):
    """Raised when a cvc5 Kind has no emit-op mapping (typed, not silent)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# cvc5 ``Kind`` name → *emit* op key (the one-key op-mapping form the DSL emit
# surface uses). Patterns use the raw Kind name; emit uses these lowercase ops.
KIND_TO_EMIT_OP: dict[str, str] = {
    "EQUAL": "eq", "DISTINCT": "neq",
    "GEQ": "gte", "LEQ": "lte", "GT": "gt", "LT": "lt",
    "ADD": "add", "MULT": "mul", "SUB": "sub",
    "ITE": "ite", "NOT": "not", "AND": "and", "OR": "or", "IMPLIES": "implies",
    "SELECT": "select", "STORE": "store",
    "BITVECTOR_SGE": "bvsge", "BITVECTOR_SLE": "bvsle",
    "BITVECTOR_SGT": "bvsgt", "BITVECTOR_SLT": "bvslt",
    "BITVECTOR_UGE": "bvuge", "BITVECTOR_ULE": "bvule",
    "BITVECTOR_UGT": "bvugt", "BITVECTOR_ULT": "bvult",
    "BITVECTOR_ADD": "add", "BITVECTOR_MULT": "mul", "BITVECTOR_SUB": "sub",
    "BITVECTOR_AND": "band", "BITVECTOR_OR": "bor", "BITVECTOR_XOR": "bxor",
    "BITVECTOR_SHL": "shl", "BITVECTOR_LSHR": "lshr", "BITVECTOR_ASHR": "ashr",
    "BITVECTOR_UDIV": "udiv", "BITVECTOR_SDIV": "sdiv",
    "BITVECTOR_UREM": "urem", "BITVECTOR_SREM": "srem",
}


def _const_value(term) -> int | None:
    """Integer value of a literal leaf (int or bit-vector), else ``None``."""
    try:
        if term.isIntegerValue():
            return int(term.getIntegerValue())
    except Exception:
        pass
    try:
        if term.isBitVectorValue():
            return int(term.getBitVectorValue(10))
    except Exception:
        pass
    return None


def _capture_name(term, captures: dict[str, str]) -> str:
    """Positional capture name (``c0, c1, …`` by first-visit order) for a leaf
    symbol — one per distinct term.

    Positional, NOT SSA-derived: structurally-identical obligations over
    different variables (e.g. ``i3 - i4 >= 0`` and ``i7 - i8 >= 0``) produce
    byte-identical templates. That makes the family **operand-general** and lets
    the committer's duplicate-signature dedup collapse same-shape obligations to a
    single rule instead of one narrow rule per operand set. The same ``captures``
    dict is shared by the obligation-pattern and candidate-emit walks, so the
    candidate's operands bind to the obligation pattern's captures by position."""
    key = str(term)
    if key in captures:
        return captures[key]
    name = f"c{len(captures)}"
    captures[key] = name
    return name


def term_to_pattern(term, captures: dict[str, str], *,
                    max_depth: int = 64, _depth: int = 0) -> dict[str, Any]:
    """cvc5 ``Term`` → structural ``pattern`` dict (records leaf captures).

    Interior nodes use the raw cvc5 ``Kind`` name (uppercase), which the matcher
    accepts directly. ``max_depth`` elides deep subtrees for the bounded prompt
    view; pass a large value for the exact deterministic template.
    """
    c = _const_value(term)
    if c is not None:
        return {"const": c}
    if term.getNumChildren() == 0:
        return {"capture": _capture_name(term, captures)}
    if _depth >= max_depth:
        return {"elided": term.getKind().name,
                "nchildren": term.getNumChildren()}
    return {"kind": term.getKind().name,
            "args": [term_to_pattern(term[i], captures,
                                     max_depth=max_depth, _depth=_depth + 1)
                     for i in range(term.getNumChildren())]}


def term_to_emit(term, captures: dict[str, str]) -> dict[str, Any]:
    """cvc5 ``Term`` → *emit* expression, reusing the capture names assigned by
    :func:`term_to_pattern`. Raises :class:`UnsupportedShape` for a Kind with no
    emit op (caller decides the fallback)."""
    c = _const_value(term)
    if c is not None:
        return {"const": c}
    if term.getNumChildren() == 0:
        return {"capture": _capture_name(term, captures)}
    # Unary arithmetic negation has no dedicated emit op; express it as ``0 - x``.
    if term.getKind() == Kind.NEG and term.getNumChildren() == 1:
        return {"sub": [{"const": 0}, term_to_emit(term[0], captures)]}
    op = KIND_TO_EMIT_OP.get(term.getKind().name)
    if op is None:
        raise UnsupportedShape(f"no_emit_op:{term.getKind().name}")
    return {op: [term_to_emit(term[i], captures)
                 for i in range(term.getNumChildren())]}


def _materialized_guard(term):
    """If ``term`` is ``eq(ite(GUARD,1,0), 1|0)`` return ``(GUARD, negate)`` —
    the informative core of a SMACK materialized comparison. Else ``None``."""
    if term.getKind() != Kind.EQUAL or term.getNumChildren() != 2:
        return None
    a, b = term[0], term[1]
    rhs = _const_value(b)
    if rhs is None:
        a, b = term[1], term[0]
        rhs = _const_value(b)
    if rhs not in (0, 1):
        return None
    if a.getKind() != Kind.ITE or a.getNumChildren() != 3:
        return None
    then_v, else_v = _const_value(a[1]), _const_value(a[2])
    if then_v == 1 and else_v == 0:
        return (a[0], rhs == 0)        # ite(G,1,0)==1 → G ; ==0 → ¬G
    if then_v == 0 and else_v == 1:
        return (a[0], rhs == 1)        # ite(G,0,1)==1 → ¬G ; ==0 → G
    return None


def derive_structural_template(term, *, rule_id: str, scope: str = "root",
                               candidate_term=None) -> dict[str, Any]:
    """Build a ``structural_template`` rule dict — ``pattern`` from the obligation
    ``term`` (so it matches real obligations), ``emit`` the **proved candidate**
    (the invariant the verifier actually found closes it). This is uniform across
    obligation shapes — a materialized comparison `eq(ite(a>=b,1,0),1)` and an
    affine `i3+i4*-1>=0` are NOT special-cased: each emits its proved candidate
    (`a>=b` / `i3>=i4`), captured from the obligation pattern. There is no
    per-shape branch.

    When no ``candidate_term`` is available yet (e.g. freeze time, before
    synthesis), ``emit`` falls back to reconstructing the obligation — a hint,
    not the final template; the proposer re-derives with the candidate.

    Returns the rule dict, or ``{"unsupported": "<reason>"}`` (typed, never silent
    ``None``) when the shape can't be emitted.
    """
    captures: dict[str, str] = {}
    try:
        pattern = term_to_pattern(term, captures, max_depth=64)
    except Exception as exc:  # pragma: no cover - defensive
        return {"unsupported": f"pattern_error:{type(exc).__name__}"}
    # Emit the proved candidate, reusing the obligation pattern's captures (the
    # candidate's operands are the same symbols, so they bind to the same
    # captures). No candidate yet → reconstruct the obligation as a fallback hint.
    emit_src = candidate_term if candidate_term is not None else term
    note = ("emit the proved candidate" if candidate_term is not None
            else "reconstruct the obligation (no candidate yet)")
    try:
        emit = term_to_emit(emit_src, captures)
    except UnsupportedShape as exc:
        return {"unsupported": exc.reason}
    return {
        "id": rule_id,
        "kind": "structural_template",
        "source": "p_target_transformed",
        "scope": scope,
        "pattern": pattern,
        "emit": [emit],
        "rationale": f"auto-derived from obligation AST ({note})",
    }


def _bounded_tree(term, *, max_depth: int = 6,
                  max_nodes: int = 80) -> dict[str, Any]:
    """Depth- and node-bounded ``term_to_pattern`` for the prompt view: keeps the
    structure readable, elides overflow rather than blowing the token budget."""
    captures: dict[str, str] = {}
    count = [0]

    def walk(t, depth):
        count[0] += 1
        c = _const_value(t)
        if c is not None:
            return {"const": c}
        if t.getNumChildren() == 0:
            return {"capture": _capture_name(t, captures)}
        if depth >= max_depth or count[0] >= max_nodes:
            return {"elided": t.getKind().name, "nchildren": t.getNumChildren()}
        return {"kind": t.getKind().name,
                "args": [walk(t[i], depth + 1)
                         for i in range(t.getNumChildren())]}

    return walk(term, 0)


def build_obligation_ast_view(term, *, surface: str | None = None,
                              rule_id: str = "auto", candidate_term=None,
                              max_sexpr: int = 400) -> dict[str, Any]:
    """The bounded, JSON-serializable obligation AST view shipped on the freeze
    payload and rendered into agent prompts. Pure-structural (no sorts/var-types —
    callers enrich with those). Pass ``candidate_term`` (when known, e.g. at the
    proposer) so the auto-template emits the proved candidate; omit it at freeze
    time. Best-effort: never raises."""
    view: dict[str, Any] = {
        "surface": str(surface) if surface is not None else None,
        "sexpr": None, "ast_tree": None,
        "materialized_guard": None, "structural_template": None,
    }
    try:
        view["sexpr"] = str(term)[:max_sexpr]
    except Exception:
        pass
    try:
        view["ast_tree"] = _bounded_tree(term)
    except Exception:
        pass
    try:
        g = _materialized_guard(term)
        if g is not None:
            g_term, negate = g
            view["materialized_guard"] = {
                "is_materialized": True,
                "guard_sexpr": str(g_term)[:max_sexpr],
                "negated": bool(negate),
                "explanation": (
                    "SMACK materialized comparison: `$sge.i32(a,b)` IS "
                    "`ite(>=(a,b),1,0)`; the obligation `== 1` means the guard "
                    "`a >= b` holds. Emit the guard relation, not `== 1`."),
            }
    except Exception:
        pass
    try:
        view["structural_template"] = derive_structural_template(
            term, rule_id=rule_id, candidate_term=candidate_term)
    except Exception as exc:  # pragma: no cover - defensive
        view["structural_template"] = {
            "unsupported": f"view_error:{type(exc).__name__}"}
    return view
