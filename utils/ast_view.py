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
  DSL matcher accepts uppercase Kind names, so no alias table is needed. Free
  program-symbol leaves → ``{capture: name, sort: var}``; other symbolic leaves
  explicitly retain the old open-subterm domain as ``sort: subterm``; integer/bv
  literals → ``{const: N}``.  The explicit domain is executable AST, not prose:
  it prevents a capture observed on a scalar leaf from silently binding a
  compound expression in a learned rule.
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

import base64
import copy
import hashlib
import json
import pickle
from typing import Any

from cvc5 import Kind

from interpreter.utils.cvc5_serde import term_op_indices
from src.rule_surface import STRUCTURAL_KIND_TO_EMIT_OP


class UnsupportedShape(Exception):
    """Raised when a cvc5 Kind has no emit-op mapping (typed, not silent)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Compatibility name for callers.  The dependency-neutral rule-surface
# registry is the sole owner of these spellings.
KIND_TO_EMIT_OP = STRUCTURAL_KIND_TO_EMIT_OP


def _const_value(term) -> int | None:
    """Integer value of a literal leaf (int or bit-vector), else ``None``.
    INT-ONLY on purpose — `term_to_emit` / `_materialized_guard` do integer
    arithmetic on it. Pattern building uses :func:`_const_literal`."""
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


def _const_literal(term):
    """JSON-able literal value for ANY constant leaf — int/bv (raw int),
    bool/real/string (tagged string spellings: ``bool:``/``real:``/``str:`` —
    tagged so a real/string/bool literal can never collide with an int const
    or with each other under the pattern matcher's ``==``; Python would
    otherwise equate ``True == 1``). ``None`` for a non-literal.

    Before this, real/string/bool literals fell through to the CAPTURE branch
    of :func:`term_to_pattern` — a constant silently became a variable, so
    e.g. two predicates differing only in a real constant were one
    over-general family."""
    c = _const_value(term)
    if c is not None:
        return c
    try:
        if term.isBooleanValue():
            return "bool:true" if term.getBooleanValue() else "bool:false"
    except Exception:
        pass
    try:
        if term.isRealValue():
            return "real:" + str(term.getRealValue())
    except Exception:
        pass
    try:
        if term.isStringValue():
            return "str:" + term.getStringValue()
    except Exception:
        pass
    return None


def _capture_name(term, captures: dict) -> str:
    """Positional capture name (``c0, c1, …`` by first-visit order) for a leaf
    symbol — one per distinct term.

    Positional, NOT SSA-derived: structurally-identical obligations over
    different variables (e.g. ``i3 - i4 >= 0`` and ``i7 - i8 >= 0``) produce
    byte-identical templates. That makes the family **operand-general** and lets
    the committer's duplicate-signature dedup collapse same-shape obligations to a
    single rule instead of one narrow rule per operand set. The same ``captures``
    dict is shared by the obligation-pattern and candidate-emit walks, so the
    candidate's operands bind to the obligation pattern's captures by position.

    Keyed on ``(spelling, Kind)``, not the spelling alone: a QUANTIFIER bound
    variable and a free symbol can share one spelling (``i``), and a bare
    string key merged them into ONE capture — a false variable-sharing claim
    (the pattern then demanded the free and the bound occurrence be the same
    term). The Kind splits bound (VARIABLE) from free (CONSTANT) while still
    letting the obligation and candidate walks bind the same SYMBOL by name
    (they may come from separately deserialized terms). Positional
    α-invariance is unaffected (keys are walk-local; emitted names stay
    ``c<n>`` by first visit)."""
    try:
        key = (str(term), term.getKind().name)
    except Exception:
        key = (str(term), "")
    if key in captures:
        return captures[key]
    name = f"c{len(captures)}"
    captures[key] = name
    return name


def _match_type_from_native_sort(sort) -> dict[str, Any] | None:
    """Translate one cvc5 sort to the executable match-sort vocabulary.

    The result contains only the semantic type axis.  Callers add the leaf's
    syntactic class independently, so a native program symbol can become the
    product sort ``{class: var, type: ...}`` without changing capture identity.
    Unsupported theories remain untyped instead of being guessed from text.
    """
    try:
        if sort.isInteger():
            return {"type": "int"}
        if sort.isReal():
            return {"type": "real"}
        if sort.isString():
            return {"type": "string"}
        if sort.isBoolean():
            return {"type": "bool"}
        if sort.isBitVector():
            return {
                "type": "bv",
                "width": int(sort.getBitVectorSize()),
            }
        if sort.isArray():
            domain = _match_type_from_native_sort(sort.getArrayIndexSort())
            element = _match_type_from_native_sort(
                sort.getArrayElementSort())
            out: dict[str, Any] = {"type": "mem"}
            if domain is not None:
                out["domain"] = domain
            if element is not None:
                out["element"] = element
            return out
    except Exception:
        return None
    return None


def _symbol_capture_pattern(
        term, captures: dict, *, include_native_type: bool = False,
        zero_one_flag_vars: frozenset[str] = frozenset(),
        ) -> dict[str, Any]:
    """Canonical executable pattern for one non-literal symbolic leaf.

    cvc5 represents free program symbols with ``Kind.CONSTANT``.  Preserve that
    fact in the matcher AST with ``sort: var``.  Quantifier-bound symbols use
    ``Kind.VARIABLE``; the current DSL has no bound-variable leaf class, so make
    their historically open domain explicit as ``sort: subterm`` rather than
    smuggling it through an omitted field.
    """
    try:
        domain = "var" if term.getKind() == Kind.CONSTANT else "subterm"
    except Exception:
        domain = "subterm"
    out: dict[str, Any] = {
        "capture": _capture_name(term, captures),
        "sort": domain,
    }
    if include_native_type:
        try:
            native_type = _match_type_from_native_sort(term.getSort())
        except Exception:
            native_type = None
        if native_type is not None:
            typed_sort = dict(native_type)
            if domain == "var":
                typed_sort["class"] = "var"
            try:
                symbol = str(term.getSymbol() or "")
            except Exception:
                symbol = ""
            if (domain == "var" and native_type.get("type") == "int"
                    and symbol in zero_one_flag_vars):
                typed_sort["value_domain"] = "zero_one"
            out["sort"] = typed_sort
    return out


def exact_nullary_literal_pattern(term) -> dict[str, Any]:
    """Lossless executable pattern for a non-symbolic nullary cvc5 term.

    Floating-point special values, rounding modes, finite-field elements, and
    future theory literals must not fall through to a capture (which would turn
    one exact constant into a wildcard).  Kind + sort + cvc5's canonical text
    distinguish the value; indexed parameters are retained defensively.
    """
    literal: dict[str, Any] = {
        "kind": term.getKind().name,
        "sort": str(term.getSort()),
        "text": str(term),
    }
    indices = term_op_indices(term)
    if indices:
        literal["indices"] = list(indices)
    return {"literal": literal}


def term_to_pattern(term, captures: dict[str, str], *,
                    max_depth: int = 64, _depth: int = 0,
                    include_native_types: bool = False,
                    zero_one_flag_vars: frozenset[str] = frozenset(),
                    ) -> dict[str, Any]:
    """cvc5 ``Term`` → structural ``pattern`` dict (records leaf captures).

    Interior nodes use the raw cvc5 ``Kind`` name (uppercase), which the matcher
    accepts directly. ``max_depth`` elides deep subtrees for the bounded prompt
    view; pass a large value for the exact deterministic template.
    """
    c = _const_literal(term)
    if c is not None:
        return {"const": c}
    if term.getNumChildren() == 0:
        if term.getKind() in (Kind.CONSTANT, Kind.VARIABLE):
            return _symbol_capture_pattern(
                term, captures, include_native_type=include_native_types,
                zero_one_flag_vars=zero_one_flag_vars)
        return exact_nullary_literal_pattern(term)
    if _depth >= max_depth:
        return {"elided": term.getKind().name,
                "nchildren": term.getNumChildren()}
    pattern = {
        "kind": term.getKind().name,
        "args": [term_to_pattern(term[i], captures,
                                 max_depth=max_depth, _depth=_depth + 1,
                                 include_native_types=include_native_types,
                                 zero_one_flag_vars=zero_one_flag_vars)
                 for i in range(term.getNumChildren())],
    }
    indices = term_op_indices(term)
    if indices:
        pattern["indices"] = list(indices)
    return pattern


def erase_capture_type_refinements(pattern: Any) -> Any:
    """Project a typed authoring pattern to its class-only target skeleton.

    Native type refinements are authoring facts, while the existing canonical
    target pattern records structural identity.  This projection lets a phase
    boundary verify that a typed seed is exactly the same tree and captures as
    its target without putting authoring metadata into proof-obligation identity.
    """
    if isinstance(pattern, list):
        return [erase_capture_type_refinements(value) for value in pattern]
    if not isinstance(pattern, dict):
        return pattern
    if "capture" not in pattern and pattern.get("wildcard") is not True:
        return {
            key: erase_capture_type_refinements(value)
            for key, value in pattern.items()
        }
    out = {
        key: erase_capture_type_refinements(value)
        for key, value in pattern.items()
        if key != "sort"
    }
    if "sort" not in pattern:
        return out
    sort = pattern.get("sort")
    if isinstance(sort, dict):
        cls = sort.get("class")
        out["sort"] = cls if cls in {"var", "const"} else "subterm"
    else:
        out["sort"] = erase_capture_type_refinements(sort)
    return out


def _pattern_contains_elision(value: Any) -> bool:
    """Whether a target pattern contains any non-executable placeholder."""
    if isinstance(value, dict):
        if "elided" in value or value.get("wildcard") is True:
            return True
        return any(_pattern_contains_elision(item) for item in value.values())
    if isinstance(value, list):
        return any(_pattern_contains_elision(item) for item in value)
    return False


def canonical_target_pattern(term) -> dict[str, Any]:
    """Return the executable structural pattern for one native target.

    This deliberately uses the same ``term_to_pattern`` vocabulary consumed by
    the DSL matcher, but unlike :func:`_bounded_tree` it is NOT a prompt-budget
    view.  The generous depth guard is only a fail-closed safety bound: an
    ``elided`` node is rejected instead of being fingerprinted as if it were an
    executable pattern.  Every proposer consumer must use this helper rather
    than independently re-deriving a pattern from surface text.
    """
    pattern = term_to_pattern(term, {}, max_depth=64)
    if _pattern_contains_elision(pattern):
        raise UnsupportedShape("canonical_target_pattern_exceeds_depth_limit")
    return pattern


def target_pattern_fingerprint(pattern: dict[str, Any]) -> str:
    """SHA-256 of canonical compact JSON for an executable target pattern."""
    if not isinstance(pattern, dict) or not pattern:
        raise ValueError("canonical target pattern must be a non-empty object")
    if _pattern_contains_elision(pattern):
        raise ValueError(
            "canonical target pattern must not contain wildcard/elision")
    encoded = json.dumps(
        pattern, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _match_type_from_serialized_sort(sort: Any) -> dict[str, Any] | None:
    """Serialized-sort twin of :func:`_match_type_from_native_sort`."""
    kind = str(getattr(sort, "kind", "") or "")
    args = tuple(getattr(sort, "args", ()) or ())
    if kind == "INTEGER_SORT":
        return {"type": "int"}
    if kind == "REAL_SORT":
        return {"type": "real"}
    if kind == "STRING_SORT":
        return {"type": "string"}
    if kind == "BOOLEAN_SORT":
        return {"type": "bool"}
    if kind == "BITVECTOR_SORT" and args:
        return {"type": "bv", "width": int(args[0])}
    if kind == "ARRAY_SORT" and len(args) == 2:
        domain = _match_type_from_serialized_sort(args[0])
        element = _match_type_from_serialized_sort(args[1])
        out: dict[str, Any] = {"type": "mem"}
        if domain is not None:
            out["domain"] = domain
        if element is not None:
            out["element"] = element
        return out
    return None


def _serialized_target_pattern(
        root: Any, *, include_native_types: bool = False,
        zero_one_flag_vars: frozenset[str] = frozenset(),
        ) -> dict[str, Any]:
    """Derive the executable pattern directly from a serialized cvc5 term.

    ``Predicate.__getstate__`` replaces its live ``Term`` with a
    ``SerializedCvc5TermV2``. Walking that representation lets every publication and
    intake boundary bind the advertised pattern fingerprint to the native bytes
    without needing a program-specific solver/state cache.  Keep this traversal
    deliberately isomorphic to :func:`term_to_pattern`.
    """
    from interpreter.utils.cvc5_serde import SerializedCvc5TermV2

    if not isinstance(root, SerializedCvc5TermV2):
        raise ValueError("native predicate pickle has no serialized cvc5 term")
    captures: dict[tuple[Any, str], str] = {}

    def capture_name(node: Any, kind_name: str) -> str:
        symbol = str(node.node.symbol or "")
        key: tuple[Any, str] = (
            symbol if symbol else node,
            kind_name,
        )
        if key not in captures:
            captures[key] = f"c{len(captures)}"
        return captures[key]

    def walk(node: Any, depth: int) -> dict[str, Any]:
        if not isinstance(node, SerializedCvc5TermV2):
            raise ValueError("native predicate pickle has a malformed cvc5 term")
        try:
            kind = Kind[node.node.kind]
        except Exception as exc:
            raise ValueError(
                "native predicate pickle has unsupported cvc5 kind "
                f"{node.node.kind!r}"
            ) from exc
        if kind in (Kind.CONST_INTEGER, Kind.CONST_BITVECTOR):
            return {"const": int(node.value)}
        if kind == Kind.CONST_BOOLEAN:
            return {"const": "bool:true" if node.value else "bool:false"}
        if kind == Kind.CONST_RATIONAL:
            return {"const": "real:" + str(node.value)}
        if kind == Kind.CONST_STRING:
            return {"const": "str:" + str(node.value)}
        children = list(node.children or ())
        if not children and kind in (Kind.CONSTANT, Kind.VARIABLE):
            domain = "var" if kind == Kind.CONSTANT else "subterm"
            symbol = str(node.node.symbol or "")
            out: dict[str, Any] = {
                "capture": capture_name(node, kind.name),
                "sort": domain,
            }
            if include_native_types:
                native_type = _match_type_from_serialized_sort(node.node.sort)
                if native_type is not None:
                    typed_sort = dict(native_type)
                    if domain == "var":
                        typed_sort["class"] = "var"
                    if (domain == "var" and native_type.get("type") == "int"
                            and symbol in zero_one_flag_vars):
                        typed_sort["value_domain"] = "zero_one"
                    out["sort"] = typed_sort
            return out
        if not children:
            if node.value is None:
                raise ValueError(
                    "serialized cvc5 term omitted an exact nullary literal")
            literal = {
                "kind": kind.name,
                "sort": repr(node.node.sort.to_obj()),
                "text": str(node.value),
            }
            indices = tuple(int(value) for value in node.node.op_indices)
            if indices:
                literal["indices"] = list(indices)
            return {"literal": literal}
        if depth >= 64:
            raise ValueError(
                "native predicate pattern exceeds canonical depth limit")
        pattern = {
            "kind": kind.name,
            "args": [walk(child, depth + 1) for child in children],
        }
        indices = tuple(int(value) for value in node.node.op_indices)
        if indices:
            pattern["indices"] = list(indices)
        return pattern

    return walk(root, 0)


def native_predicate_pattern(
        value: Any, *, include_native_types: bool = False,
        zero_one_flag_vars: frozenset[str] = frozenset(),
        ) -> dict[str, Any]:
    """Return the exact structural pattern of a native predicate carrier.

    Accepts either a live cvc5 term/``Predicate`` or the serialized predicate
    retained inside a canonical ``ProofObligation``.  This is the common
    post-checker pattern boundary: consumers derive structure from the native
    carrier and never render/reparse its diagnostic surface.
    """
    term = getattr(value, "predicate", value)
    from interpreter.utils.cvc5_serde import SerializedCvc5TermV2

    if isinstance(term, SerializedCvc5TermV2):
        return _serialized_target_pattern(
            term, include_native_types=include_native_types,
            zero_one_flag_vars=zero_one_flag_vars)
    pattern = term_to_pattern(
        term, {}, max_depth=64,
        include_native_types=include_native_types,
        zero_one_flag_vars=zero_one_flag_vars)
    if _pattern_contains_elision(pattern):
        raise UnsupportedShape("canonical_target_pattern_exceeds_depth_limit")
    return pattern


def validate_proposer_native_target_b64(
        value: Any, *, expected_pattern_sha256: str = "") -> str:
    """Return a native payload whose bytes agree with its advertised pattern."""
    serialized = str(value or "").strip()
    if not serialized:
        raise ValueError(
            "proposer target frame needs serialized_transformed_b64")
    try:
        decoded = base64.b64decode(serialized, validate=True)
    except Exception as exc:
        raise ValueError(
            "proposer target frame serialized_transformed_b64 is malformed") \
            from exc
    if not decoded:
        raise ValueError(
            "proposer target frame serialized_transformed_b64 is empty")
    try:
        predicate = pickle.loads(decoded)
    except Exception as exc:
        raise ValueError(
            "proposer target frame serialized_transformed_b64 is not a "
            "native predicate pickle") from exc
    native_term = getattr(predicate, "predicate", None)
    if native_term is None:
        raise ValueError(
            "proposer target frame serialized_transformed_b64 is not a "
            "native predicate pickle")
    try:
        native_pattern = _serialized_target_pattern(native_term)
        native_fingerprint = target_pattern_fingerprint(native_pattern)
    except Exception as exc:
        raise ValueError(
            "proposer target frame serialized_transformed_b64 is not a "
            "native predicate pickle") from exc
    expected = str(expected_pattern_sha256 or "").strip().lower()
    if expected and native_fingerprint != expected:
        raise ValueError(
            "proposer target frame native predicate pattern fingerprint "
            f"mismatch ({native_fingerprint} != {expected})")
    return serialized


def build_proposer_target_frame_from_native_b64(
        serialized_transformed_b64: str, *, proof_obligation_id: str,
        run_id: str, pc: int | str, surface: str = "",
        source: str = "native") -> dict[str, Any]:
    """Build the canonical target frame directly from its native carrier.

    This is the publication boundary for code that retained the exact
    serialized transformed predicate but not a live solver term. The match
    pattern and optional display surface come from that same native AST; no
    predicate text is parsed or independently reconstructed.
    """
    serialized = validate_proposer_native_target_b64(
        serialized_transformed_b64)
    predicate = pickle.loads(base64.b64decode(serialized, validate=True))
    native_term = getattr(predicate, "predicate", None)
    if native_term is None:
        raise ValueError(
            "proposer target frame native carrier has no predicate term")
    pattern = _serialized_target_pattern(native_term)
    surface_text = str(surface or "").strip()
    if not surface_text:
        try:
            from interpreter.utils.cvc5_serde import hollow_to_str

            surface_text = str(
                hollow_to_str(native_term, max_depth=256) or "").strip()
        except Exception as exc:
            raise ValueError(
                "proposer target frame native surface is unavailable") from exc
    return build_proposer_target_frame_from_pattern(
        pattern,
        proof_obligation_id=proof_obligation_id,
        run_id=run_id,
        pc=pc,
        surface=surface_text,
        serialized_transformed_b64=serialized,
        source=source,
    )


def build_proposer_target_frame_from_pattern(
        pattern: dict[str, Any], *, proof_obligation_id: str, run_id: str,
        pc: int | str, surface: str, serialized_transformed_b64: str,
        source: str = "native") -> dict[str, Any]:
    """Build a target frame from a pattern already derived at the native edge."""
    poid = str(proof_obligation_id or "").strip()
    rid = str(run_id or "").strip()
    if not poid:
        raise ValueError("proposer target frame needs proof_obligation_id")
    if not rid:
        raise ValueError("proposer target frame needs run_id")
    surface_text = str(surface or "").strip()
    if not surface_text:
        raise ValueError("proposer target frame needs surface")
    source_text = str(source or "native").strip()
    if not source_text:
        raise ValueError("proposer target frame needs source")
    try:
        pc_value = int(pc)
    except (TypeError, ValueError) as exc:
        raise ValueError("proposer target frame needs an integer pc") from exc
    fingerprint = target_pattern_fingerprint(pattern)
    serialized = validate_proposer_native_target_b64(
        serialized_transformed_b64,
        expected_pattern_sha256=fingerprint,
    )
    return {
        "schema": "anvil.proposer-target-frame/v1",
        "proof_obligation_id": poid,
        "run_id": rid,
        "pc": pc_value,
        "surface": surface_text,
        "pattern": copy.deepcopy(pattern),
        "pattern_sha256": fingerprint,
        "serialized_transformed_b64": serialized,
        "source": source_text,
    }


def build_proposer_target_frame(
        term, *, proof_obligation_id: str, run_id: str, pc: int | str,
        surface: str, serialized_transformed_b64: str,
        source: str = "native") -> dict[str, Any]:
    """Build the one proposer target contract from a live native term.

    ``surface`` is display-only.  ``pattern`` and its fingerprint come from the
    native cvc5 term, while ``serialized_transformed_b64`` lets downstream
    verification rehydrate that exact target without parsing the display.
    """
    return build_proposer_target_frame_from_pattern(
        canonical_target_pattern(term),
        proof_obligation_id=proof_obligation_id,
        run_id=run_id,
        pc=pc,
        surface=surface,
        serialized_transformed_b64=serialized_transformed_b64,
        source=source,
    )


def term_to_emit(term, captures: dict[str, str]) -> dict[str, Any]:
    """cvc5 ``Term`` → *emit* expression, reusing the capture names assigned by
    :func:`term_to_pattern`. Raises :class:`UnsupportedShape` for a Kind with no
    emit op."""
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


def derive_structural_template(term, *, rule_id: str,
                               candidate_term,
                               include_native_types: bool = False,
                               ) -> dict[str, Any]:
    """Build a ``structural_template`` rule dict — ``pattern`` from the obligation
    ``term`` (so it matches real obligations), ``emit`` the **proved candidate**
    (the invariant the verifier actually found closes it). This is uniform across
    obligation shapes — a materialized comparison `eq(ite(a>=b,1,0),1)` and an
    affine `i3+i4*-1>=0` are NOT special-cased: each emits its proved candidate
    (`a>=b` / `i3>=i4`), captured from the obligation pattern. There is no
    per-shape branch.

    Returns the rule dict, or ``{"unsupported": "<reason>"}`` (typed, never silent
    ``None``) when the shape can't be emitted.
    """
    if candidate_term is None:
        raise ValueError(
            "derive_structural_template requires the proved candidate term")
    captures: dict[str, str] = {}
    try:
        pattern = term_to_pattern(
            term, captures, max_depth=64,
            include_native_types=include_native_types)
    except Exception as exc:  # pragma: no cover - defensive
        return {"unsupported": f"pattern_error:{type(exc).__name__}"}
    # Emit the proved candidate, reusing the obligation pattern's captures (the
    # candidate's operands are the same symbols, so they bind to the same
    # captures).
    try:
        emit = term_to_emit(candidate_term, captures)
    except UnsupportedShape as exc:
        return {"unsupported": exc.reason}
    return {
        "id": rule_id,
        "kind": "structural_template",
        "match": {
            "source": "p_target_transformed",
            "pattern": pattern,
        },
        "action": {"emit": [emit]},
        "rationale": {
            "summary": (
                "Auto-derived from the obligation AST and emits the proved "
                "candidate."
            ),
        },
    }


def _bounded_tree(term, *, max_depth: int = 6,
                  max_nodes: int = 80) -> dict[str, Any]:
    """Depth- and node-bounded ``term_to_pattern`` for the prompt view: keeps the
    structure readable, elides overflow rather than blowing the token budget."""
    captures: dict[str, str] = {}
    count = [0]

    def walk(t, depth):
        count[0] += 1
        c = _const_literal(t)
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
        "canonical_pattern": None, "canonical_pattern_sha256": None,
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
        # Authoritative executable pattern.  ``ast_tree`` above remains the
        # bounded diagnostic rendering; consumers must never mistake it for the
        # pattern/fingerprint used by lookahead and verification.
        pattern = canonical_target_pattern(term)
        view["canonical_pattern"] = pattern
        view["canonical_pattern_sha256"] = target_pattern_fingerprint(pattern)
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
    if candidate_term is not None:
        try:
            view["structural_template"] = derive_structural_template(
                term, rule_id=rule_id, candidate_term=candidate_term)
        except Exception as exc:  # pragma: no cover - defensive
            view["structural_template"] = {
                "unsupported": f"view_error:{type(exc).__name__}"}
    return view
