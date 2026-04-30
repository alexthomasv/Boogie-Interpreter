"""Versioned cvc5 term serialization.

The cvc5 Python bindings wrap native objects, so Terms, Sorts and Ops are
not pickleable.  This module stores the inspectable term DAG in a small,
immutable Python representation that can be pickled, hashed locally, and
rehydrated into another solver instance.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cvc5
import msgpack
from cachetools import LRUCache
from cvc5 import Kind, SortKind, Term


class Cvc5SerdeError(RuntimeError):
    """Raised when a serialized cvc5 term cannot be rebuilt."""


_SERDE_STATS = {
    "serialize_calls": 0,
    "serialize_hits": 0,
    "serialize_misses": 0,
    "serialize_nodes": 0,
    "serialize_ms": 0.0,
    "deserialize_calls": 0,
    "deserialize_cache_hits": 0,
    "deserialize_cache_misses": 0,
    "deserialize_root_cache_hits": 0,
    "deserialize_subterm_cache_hits": 0,
    "deserialize_nodes_rebuilt": 0,
    "deserialize_nodes_skipped": 0,
    "deserialize_nodes": 0,
    "deserialize_ms": 0.0,
    "deserialize_cache_writes": 0,
    "sort_cache_hits": 0,
    "sort_cache_misses": 0,
}


def reset_serde_stats() -> None:
    """Reset process-local cvc5 serialization counters."""
    for key in _SERDE_STATS:
        _SERDE_STATS[key] = 0.0 if key.endswith("_ms") else 0


def get_serde_stats(*, reset: bool = False) -> dict[str, int | float | dict]:
    """Return process-local cvc5 serialization counters.

    The cache info is included as a snapshot so perf reports can tell the
    difference between high serde volume and low cache effectiveness.
    """
    stats = dict(_SERDE_STATS)
    info = _serialize_cvc5_term_cached.cache_info()
    stats["serialize_cache"] = {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }
    if reset:
        reset_serde_stats()
    return stats


_COMPAT_KIND_NUMS = {
    "EQUAL": 0,
    "DISTINCT": 1,
    "AND": 2,
    "OR": 3,
    "NOT": 4,
    "ADD": 5,
    "MULT": 6,
    "SUB": 7,
    "DIVISION": 59,
    "ITE": 9,
    "LEQ": 10,
    "GEQ": 11,
    "LT": 12,
    "GT": 13,
    "IMPLIES": 14,
    "SELECT": 15,
    "STORE": 16,
    "APPLY_UF": 17,
    "BITVECTOR_ADD": 18,
    "BITVECTOR_MULT": 19,
    "INT_TO_BITVECTOR": 20,
    "BITVECTOR_SIGN_EXTEND": 21,
    "BITVECTOR_ULT": 22,
    "BITVECTOR_EXTRACT": 23,
    "BITVECTOR_ZERO_EXTEND": 24,
    "BITVECTOR_AND": 25,
    "BITVECTOR_LSHR": 26,
    "BITVECTOR_OR": 27,
    "BITVECTOR_SHL": 28,
    "BITVECTOR_XOR": 29,
    "BITVECTOR_SUB": 30,
    "BITVECTOR_CONCAT": 31,
    "BITVECTOR_NOT": 32,
    "BITVECTOR_SLT": 33,
    "BITVECTOR_SREM": 34,
    "BITVECTOR_ULE": 35,
    "BITVECTOR_SGE": 36,
    "BITVECTOR_UGE": 37,
    "BITVECTOR_NEG": 38,
    "BITVECTOR_UDIV": 39,
    "BITVECTOR_UGT": 40,
    "BITVECTOR_SGT": 41,
    "BITVECTOR_ASHR": 42,
    "BITVECTOR_UREM": 43,
    "BITVECTOR_SLE": 44,
    "BITVECTOR_SDIV": 45,
    "XOR": 46,
    "CONST_BITVECTOR": 47,
    "CONST_BOOLEAN": 48,
    "CONSTANT": 49,
    "SET_MEMBER": 50,
    "SET_INSERT": 51,
    "SET_EMPTY": 52,
    "CONST_INTEGER": 53,
    "INTS_MODULUS": 54,
    "INTS_DIVISION": 55,
    "NEG": 56,
    "INTS_MODULUS_TOTAL": 57,
    "INTS_DIVISION_TOTAL": 58,
    "DIVISION_TOTAL": 60,
    "TO_INTEGER": 61,
    "IS_INTEGER": 62,
    "BITVECTOR_UBV_TO_INT": 63,
    "BITVECTOR_SBV_TO_INT": 64,
    "FORALL": 65,
    "VARIABLE_LIST": 66,
    "VARIABLE": 67,
}

_COMPAT_SORT_NUMS = {
    "BITVECTOR_SORT": 0,
    "BOOLEAN_SORT": 1,
    "INTEGER_SORT": 2,
    "ARRAY_SORT": 4,
    "INTERNAL_SORT_KIND": 5,
    "SET_SORT": 6,
    "UNINTERPRETED_SORT": 7,
    "FUNCTION_SORT": 8,
}

_COMMUTATIVE_KINDS = frozenset(
    {
        "EQUAL",
        "DISTINCT",
        "ADD",
        "MULT",
        "AND",
        "OR",
        "XOR",
        "BITVECTOR_ADD",
        "BITVECTOR_MULT",
        "BITVECTOR_AND",
        "BITVECTOR_OR",
        "BITVECTOR_XOR",
    }
)


@dataclass(frozen=True, slots=True)
class SerializedSort:
    kind: str
    args: tuple[Any, ...] = ()

    def to_obj(self):
        return _sort_to_obj(self)


@dataclass(frozen=True, slots=True)
class SerializedNode:
    kind: str
    sort: SerializedSort
    op_indices: tuple[int, ...] = ()
    value: int | bool | str | None = None
    symbol: str = ""
    children: tuple[int, ...] = ()

    def to_obj(self):
        return (
            self.kind,
            _sort_to_obj(self.sort),
            self.op_indices,
            self.value,
            self.symbol,
            self.children,
        )


def _sort_to_obj(sort: SerializedSort):
    return (sort.kind, tuple(_sort_arg_to_obj(a) for a in sort.args))


def _sort_arg_to_obj(arg):
    if isinstance(arg, SerializedSort):
        return _sort_to_obj(arg)
    return arg


def _sort_from_obj(obj) -> SerializedSort:
    kind, args = obj
    converted = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == 2 and isinstance(arg[0], str):
            converted.append(_sort_from_obj(arg))
        else:
            converted.append(arg)
    return SerializedSort(kind, tuple(converted))


@dataclass(frozen=True, slots=True)
class SerializedCvc5TermV2:
    version: int
    root: int
    nodes: tuple[SerializedNode, ...]

    def __post_init__(self):
        if self.version != 2:
            raise Cvc5SerdeError(f"unsupported cvc5 term serialization version: {self.version}")
        if self.root < 0 or self.root >= len(self.nodes):
            raise Cvc5SerdeError(f"root node out of range: {self.root}")

    @property
    def node(self) -> SerializedNode:
        return self.nodes[self.root]

    @property
    def op(self) -> int | None:
        return _COMPAT_KIND_NUMS.get(self.node.kind)

    @property
    def type(self) -> int | None:
        return _COMPAT_SORT_NUMS.get(self.node.sort.kind)

    @property
    def var_name(self) -> str:
        return self.node.symbol if self.node.kind in {"CONSTANT", "VARIABLE"} else ""

    @property
    def bitwidth(self) -> int:
        sort = self.node.sort
        if sort.kind == "BITVECTOR_SORT":
            return int(sort.args[0])
        if sort.kind == "SET_SORT" and sort.args and sort.args[0].kind == "BITVECTOR_SORT":
            return int(sort.args[0].args[0])
        return 0

    @property
    def value(self) -> int | bool | str | None:
        return self.node.value

    @property
    def bv_extract_start(self) -> int:
        return int(self.node.op_indices[1]) if self.node.kind == "BITVECTOR_EXTRACT" else 0

    @property
    def bv_extract_end(self) -> int:
        return int(self.node.op_indices[0]) if self.node.kind == "BITVECTOR_EXTRACT" else 0

    @property
    def bv_sign_extend_bitwidth(self) -> int:
        return int(self.node.op_indices[0]) if self.node.kind == "BITVECTOR_SIGN_EXTEND" else 0

    @property
    def bv_zero_extend_bitwidth(self) -> int:
        return int(self.node.op_indices[0]) if self.node.kind == "BITVECTOR_ZERO_EXTEND" else 0

    @property
    def array_element_width(self) -> int:
        sort = self.node.sort
        if sort.kind == "ARRAY_SORT" and sort.args[1].kind == "BITVECTOR_SORT":
            return int(sort.args[1].args[0])
        return 0

    @property
    def array_index_width(self) -> int:
        sort = self.node.sort
        if sort.kind == "ARRAY_SORT" and sort.args[0].kind == "BITVECTOR_SORT":
            return int(sort.args[0].args[0])
        return 0

    @property
    def children(self) -> tuple["SerializedCvc5TermV2", ...]:
        return tuple(SerializedCvc5TermV2(self.version, child, self.nodes) for child in self.node.children)

    def with_root(self, root: int) -> "SerializedCvc5TermV2":
        return SerializedCvc5TermV2(self.version, root, self.nodes)

    def to_obj(self):
        return (self.version, self.root, tuple(node.to_obj() for node in self.nodes))

    def to_bytes(self) -> bytes:
        return msgpack.packb(self.to_obj(), use_bin_type=True)

    def exact_fingerprint(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return msgpack.packb(_canonical_node_obj(self, self.root), use_bin_type=True)

    def canonical_fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"HollowCvc5TermV2(root={self.root}, op={self.op}, "
            f"children={list(self.node.children)}, var_name={self.var_name!r})"
        )


HollowCvc5Term = SerializedCvc5TermV2


def _sort_signature(sort) -> SerializedSort:
    if sort.isBoolean():
        return SerializedSort("BOOLEAN_SORT")
    if sort.isInteger():
        return SerializedSort("INTEGER_SORT")
    if sort.isBitVector():
        return SerializedSort("BITVECTOR_SORT", (int(sort.getBitVectorSize()),))
    if sort.isArray():
        return SerializedSort(
            "ARRAY_SORT",
            (_sort_signature(sort.getArrayIndexSort()), _sort_signature(sort.getArrayElementSort())),
        )
    if sort.isSet():
        return SerializedSort("SET_SORT", (_sort_signature(sort.getSetElementSort()),))
    if sort.isFunction():
        return SerializedSort(
            "FUNCTION_SORT",
            tuple(_sort_signature(s) for s in sort.getFunctionDomainSorts())
            + (_sort_signature(sort.getFunctionCodomainSort()),),
        )
    if sort.isUninterpretedSort():
        symbol = sort.getSymbol() if sort.hasSymbol() else str(sort)
        return SerializedSort("UNINTERPRETED_SORT", (symbol,))
    if sort.getKind() == SortKind.INTERNAL_SORT_KIND:
        return SerializedSort("INTERNAL_SORT_KIND")
    raise Cvc5SerdeError(f"unsupported cvc5 sort: {sort} ({sort.getKind()})")


def _sort_from_signature(solver, sig: SerializedSort):
    if sig.kind == "BOOLEAN_SORT":
        return solver.getBooleanSort()
    if sig.kind == "INTEGER_SORT":
        return solver.getIntegerSort()
    if sig.kind == "BITVECTOR_SORT":
        return solver.mkBitVectorSort(int(sig.args[0]))
    if sig.kind == "ARRAY_SORT":
        return solver.mkArraySort(
            _sort_from_signature(solver, sig.args[0]),
            _sort_from_signature(solver, sig.args[1]),
        )
    if sig.kind == "SET_SORT":
        return solver.mkSetSort(_sort_from_signature(solver, sig.args[0]))
    if sig.kind == "FUNCTION_SORT":
        domain = [_sort_from_signature(solver, s) for s in sig.args[:-1]]
        codomain = _sort_from_signature(solver, sig.args[-1])
        return solver.mkFunctionSort(domain, codomain)
    if sig.kind == "UNINTERPRETED_SORT":
        return solver.mkUninterpretedSort(str(sig.args[0]))
    if sig.kind == "INTERNAL_SORT_KIND":
        raise Cvc5SerdeError("internal cvc5 sort cannot be materialized directly")
    raise Cvc5SerdeError(f"unsupported serialized sort: {sig}")


def _term_op_indices(term: Term) -> tuple[int, ...]:
    if not term.hasOp():
        return ()
    op = term.getOp()
    if not op.isIndexed():
        return ()
    indices = []
    for i in range(op.getNumIndices()):
        indices.append(int(op[i].getIntegerValue()))
    return tuple(indices)


def _term_payload(term: Term, kind_name: str):
    if kind_name in {"CONSTANT", "VARIABLE"}:
        try:
            return None, term.getSymbol()
        except RuntimeError:
            return None, str(term)
    if kind_name == "CONST_BITVECTOR":
        return int(term.getBitVectorValue(10)), ""
    if kind_name == "CONST_INTEGER":
        return int(term.getIntegerValue()), ""
    if kind_name == "CONST_BOOLEAN":
        return bool(term.getBooleanValue()), ""
    if kind_name == "CONST_STRING":
        return term.getStringValue(), ""
    return None, ""


def _serialize_uncached(term: Term) -> SerializedCvc5TermV2:
    stack = [(term, 0)]
    done: dict[Term, int] = {}
    nodes: list[SerializedNode] = []

    while stack:
        parent, idx = stack[-1]
        if parent in done:
            stack.pop()
            continue

        n = parent.getNumChildren()
        while idx < n and parent[idx] in done:
            idx += 1
        if idx < n:
            stack[-1] = (parent, idx + 1)
            stack.append((parent[idx], 0))
            continue

        stack.pop()
        kind_name = parent.getKind().name
        if kind_name not in cvc5.Kind.__members__:
            raise Cvc5SerdeError(f"unsupported cvc5 kind: {parent.getKind()}")
        value, symbol = _term_payload(parent, kind_name)
        node = SerializedNode(
            kind=kind_name,
            sort=_sort_signature(parent.getSort()),
            op_indices=_term_op_indices(parent),
            value=value,
            symbol=symbol,
            children=tuple(done[parent[i]] for i in range(n)),
        )
        done[parent] = len(nodes)
        nodes.append(node)

    return SerializedCvc5TermV2(2, done[term], tuple(nodes))


@lru_cache(maxsize=20000)
def _serialize_cvc5_term_cached(term: Term) -> SerializedCvc5TermV2:
    return _serialize_uncached(term)


def serialize_cvc5_term(term: Term | SerializedCvc5TermV2) -> SerializedCvc5TermV2:
    if isinstance(term, SerializedCvc5TermV2):
        return term
    if not isinstance(term, Term):
        raise TypeError(f"expected cvc5.Term, got {type(term).__name__}")
    _SERDE_STATS["serialize_calls"] += 1
    before = _serialize_cvc5_term_cached.cache_info()
    t0 = time.perf_counter()
    serialized = _serialize_cvc5_term_cached(term)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    after = _serialize_cvc5_term_cached.cache_info()
    if after.hits > before.hits:
        _SERDE_STATS["serialize_hits"] += 1
    elif after.misses > before.misses:
        _SERDE_STATS["serialize_misses"] += 1
    _SERDE_STATS["serialize_nodes"] += len(serialized.nodes)
    _SERDE_STATS["serialize_ms"] += elapsed_ms
    return serialized


serialize_cvc5_term.cache_clear = _serialize_cvc5_term_cached.cache_clear  # type: ignore[attr-defined]
serialize_cvc5_term.cache_info = _serialize_cvc5_term_cached.cache_info  # type: ignore[attr-defined]


def _ensure_term_cache(state_cache):
    cache = getattr(state_cache, "cvc5_term_cache", None)
    if cache is None:
        cache = LRUCache(maxsize=50000)
        state_cache.cvc5_term_cache = cache
    return cache


def _subterm_cache_keys(root_term: SerializedCvc5TermV2):
    keys = {}
    digests = {}
    for idx, node in enumerate(root_term.nodes):
        payload = (
            node.kind,
            _sort_to_obj(node.sort),
            node.op_indices,
            node.value,
            node.symbol,
            tuple(digests[child] for child in node.children),
        )
        digest = hashlib.sha256(msgpack.packb(payload, use_bin_type=True)).hexdigest()
        digests[idx] = digest
        keys[idx] = ("hollow-v2-subterm", digest)
    return keys


def _ensure_sort_cache(state_cache):
    cache = getattr(state_cache, "cvc5_sort_cache", None)
    if cache is None:
        cache = {}
        state_cache.cvc5_sort_cache = cache
    return cache


def _sort_from_signature_cached(state_cache, sig: SerializedSort):
    cache = _ensure_sort_cache(state_cache)
    cached = cache.get(sig)
    if cached is not None:
        _SERDE_STATS["sort_cache_hits"] += 1
        return cached
    _SERDE_STATS["sort_cache_misses"] += 1
    sort = _sort_from_signature(state_cache.solver, sig)
    cache[sig] = sort
    return sort


def _lookup_free_const(state_cache, node: SerializedNode):
    cvc5_var = getattr(state_cache, "cvc5_var", None)
    if cvc5_var is None:
        return None
    return cvc5_var(node.symbol)


def _remember_free_const(state_cache, name: str, term: Term):
    cache = getattr(state_cache, "cached_id_to_cvc5", None)
    if isinstance(cache, MutableMapping):
        cache[name] = term
    redis = getattr(state_cache, "redis", None)
    if redis is not None:
        from src.state.redis_keys import put_cvc5_var

        put_cvc5_var(redis, name, term)


def deserialize_cvc5_term(state_cache, root_term: SerializedCvc5TermV2 | Term) -> Term:
    if isinstance(root_term, Term):
        return root_term
    if not isinstance(root_term, SerializedCvc5TermV2):
        raise TypeError(f"expected serialized cvc5 term, got {type(root_term).__name__}")

    _SERDE_STATS["deserialize_calls"] += 1
    _SERDE_STATS["deserialize_nodes"] += len(root_term.nodes)
    t0 = time.perf_counter()
    global_cache = _ensure_term_cache(state_cache)
    cache_keys = _subterm_cache_keys(root_term)
    root_cache_key = cache_keys[root_term.root]
    if root_cache_key in global_cache:
        _SERDE_STATS["deserialize_cache_hits"] += 1
        _SERDE_STATS["deserialize_root_cache_hits"] += 1
        _SERDE_STATS["deserialize_ms"] += (time.perf_counter() - t0) * 1000
        return global_cache[root_cache_key]
    _SERDE_STATS["deserialize_cache_misses"] += 1

    solver = state_cache.solver
    memo: dict[int, Term] = {}
    rebuilt: dict[int, Term] = {}
    for idx, node in enumerate(root_term.nodes):
        subterm = root_term.with_root(idx)
        cache_key = cache_keys[idx]
        cached = global_cache.get(cache_key)
        if cached is not None:
            memo[idx] = cached
            _SERDE_STATS["deserialize_cache_hits"] += 1
            _SERDE_STATS["deserialize_subterm_cache_hits"] += 1
            _SERDE_STATS["deserialize_nodes_skipped"] += 1
            continue
        try:
            child_terms = [memo[child] for child in node.children]
            kind = Kind[node.kind]

            if node.kind == "CONSTANT":
                res = _lookup_free_const(state_cache, node)
                if res is None:
                    res = solver.mkConst(_sort_from_signature_cached(state_cache, node.sort), node.symbol)
                    _remember_free_const(state_cache, node.symbol, res)
            elif node.kind == "VARIABLE":
                res = solver.mkVar(_sort_from_signature_cached(state_cache, node.sort), node.symbol)
            elif node.kind == "CONST_BITVECTOR":
                res = solver.mkBitVector(subterm.bitwidth, int(node.value))
            elif node.kind == "CONST_BOOLEAN":
                res = solver.mkBoolean(bool(node.value))
            elif node.kind == "CONST_INTEGER":
                res = solver.mkInteger(str(int(node.value)))
            elif node.kind == "SET_EMPTY":
                res = solver.mkEmptySet(_sort_from_signature_cached(state_cache, node.sort))
            elif node.kind in {"BITVECTOR_EXTRACT", "BITVECTOR_SIGN_EXTEND", "BITVECTOR_ZERO_EXTEND", "INT_TO_BITVECTOR"}:
                res = solver.mkTerm(solver.mkOp(kind, *node.op_indices), *child_terms)
            else:
                res = solver.mkTerm(kind, *child_terms)
            memo[idx] = res
            rebuilt[idx] = res
            _SERDE_STATS["deserialize_nodes_rebuilt"] += 1
        except Exception as exc:
            raise Cvc5SerdeError(
                "failed to deserialize cvc5 node "
                f"{idx}: kind={node.kind} sort={node.sort} children={node.children}"
            ) from exc

    for idx, term in rebuilt.items():
        global_cache[cache_keys[idx]] = term
        _SERDE_STATS["deserialize_cache_writes"] += 1
    _SERDE_STATS["deserialize_ms"] += (time.perf_counter() - t0) * 1000
    return memo[root_term.root]


def _const_node_obj(node: SerializedNode):
    if node.kind == "CONST_BITVECTOR":
        return ("bv", node.value, node.sort.args[0])
    if node.kind == "CONST_INTEGER":
        return ("int", node.value)
    if node.kind == "CONST_BOOLEAN":
        return ("bool", bool(node.value))
    if node.kind in {"CONSTANT", "VARIABLE"}:
        return (node.kind.lower(), node.symbol, _sort_to_obj(node.sort))
    return None


def _canonical_node_obj(term: SerializedCvc5TermV2, idx: int):
    node = term.nodes[idx]
    const_obj = _const_node_obj(node)
    if const_obj is not None:
        return const_obj

    children = tuple(_canonical_node_obj(term, child) for child in node.children)

    # cvc5 commonly simplifies x * 2 into concat(extract(x), #b0).  Treat
    # that artifact as multiplication by a power of two for fingerprints.
    if node.kind == "BITVECTOR_CONCAT" and len(node.children) == 2:
        hi = term.nodes[node.children[0]]
        lo = term.nodes[node.children[1]]
        if hi.kind == "BITVECTOR_EXTRACT" and lo.kind == "CONST_BITVECTOR" and lo.value == 0:
            shift = lo.sort.args[0] if lo.sort.kind == "BITVECTOR_SORT" else 0
            inner_idx = hi.children[0] if hi.children else None
            if inner_idx is not None:
                inner = _canonical_node_obj(term, inner_idx)
                width = node.sort.args[0] if node.sort.kind == "BITVECTOR_SORT" else shift
                children = tuple(sorted((inner, ("bv", 1 << int(shift), width)), key=repr))
                return ("BITVECTOR_MULT", _sort_to_obj(node.sort), (), children)

    if node.kind in _COMMUTATIVE_KINDS:
        children = tuple(sorted(children, key=repr))
    return (node.kind, _sort_to_obj(node.sort), node.op_indices, children)


def canonical_term_bytes(term: Term | SerializedCvc5TermV2) -> bytes:
    serialized = serialize_cvc5_term(term)
    return serialized.canonical_bytes()


def canonical_term_fingerprint(term: Term | SerializedCvc5TermV2) -> str:
    return hashlib.sha256(canonical_term_bytes(term)).hexdigest()


def hollow_to_str(term: SerializedCvc5TermV2 | None, max_depth: int = 8) -> str:
    if term is None:
        return "<None>"
    if max_depth <= 0:
        return "..."
    node = term.node
    if node.kind in {"CONSTANT", "VARIABLE"}:
        return node.symbol or f"<{node.kind}>"
    if node.kind == "CONST_BITVECTOR":
        return f"{node.value}bv{term.bitwidth}"
    if node.kind == "CONST_INTEGER":
        return str(node.value)
    if node.kind == "CONST_BOOLEAN":
        return "true" if node.value else "false"
    if node.kind == "SET_EMPTY":
        return "{}"

    kids = [hollow_to_str(child, max_depth - 1) for child in term.children]
    if node.kind == "SELECT" and len(kids) == 2:
        return f"{kids[0]}[{kids[1]}]"
    if node.kind == "STORE" and len(kids) == 3:
        return f"store({kids[0]}, {kids[1]}, {kids[2]})"
    if node.kind == "ITE" and len(kids) == 3:
        return f"ite({kids[0]}, {kids[1]}, {kids[2]})"
    if node.kind == "BITVECTOR_EXTRACT" and len(kids) == 1:
        return f"({kids[0]})[{term.bv_extract_end}:{term.bv_extract_start}]"
    if node.kind == "BITVECTOR_SIGN_EXTEND" and len(kids) == 1:
        return f"sext{term.bv_sign_extend_bitwidth}({kids[0]})"
    if node.kind == "BITVECTOR_ZERO_EXTEND" and len(kids) == 1:
        return f"zext{term.bv_zero_extend_bitwidth}({kids[0]})"
    if node.kind == "BITVECTOR_CONCAT":
        return "concat(" + ", ".join(kids) + ")"
    infix = {
        "EQUAL": "==",
        "DISTINCT": "!=",
        "ADD": "+",
        "MULT": "*",
        "SUB": "-",
        "BITVECTOR_ADD": "+",
        "BITVECTOR_MULT": "*",
        "BITVECTOR_SUB": "-",
        "BITVECTOR_AND": "&",
        "BITVECTOR_OR": "|",
        "BITVECTOR_XOR": "^",
        "AND": "&&",
        "OR": "||",
        "IMPLIES": "=>",
    }
    if node.kind in infix and len(kids) >= 2:
        return "(" + f" {infix[node.kind]} ".join(kids) + ")"
    if len(kids) == 1 and node.kind in {"NOT", "BITVECTOR_NOT", "NEG", "BITVECTOR_NEG"}:
        return f"{node.kind.lower()}({kids[0]})"
    return f"{node.kind.lower()}(" + ", ".join(kids) + ")"


def classify_hollow(predicate) -> str:
    origin = getattr(predicate, "origin", None)
    if origin in ("reach", "custom"):
        return origin
    hollow = getattr(predicate, "predicate", None)
    if hollow is None:
        return "non-mem"

    def has_kind(t: SerializedCvc5TermV2, kinds: set[str], depth: int = 12) -> bool:
        if depth <= 0:
            return False
        if t.node.kind in kinds:
            return True
        return any(has_kind(child, kinds, depth - 1) for child in t.children)

    if hollow.node.kind == "EQUAL" and len(hollow.children) == 2:
        l, r = hollow.children
        if l.node.kind == "SELECT" and r.node.kind == "SELECT":
            return "mem-partial"
        if l.node.sort.kind == "ARRAY_SORT" and r.node.sort.kind == "ARRAY_SORT":
            return "mem-full"
    if getattr(predicate, "eq_const_predicate", False):
        if has_kind(hollow, {"SELECT"}):
            return "mem-partial"
    if has_kind(hollow, {"SELECT", "STORE"}):
        return "mem-partial"
    return "non-mem"


def term_from_bytes(data: bytes) -> SerializedCvc5TermV2:
    version, root, raw_nodes = msgpack.unpackb(data, raw=False)
    nodes = []
    for kind, sort_obj, op_indices, value, symbol, children in raw_nodes:
        nodes.append(
            SerializedNode(
                kind,
                _sort_from_obj(sort_obj),
                tuple(op_indices),
                value,
                symbol,
                tuple(children),
            )
        )
    return SerializedCvc5TermV2(int(version), int(root), tuple(nodes))
