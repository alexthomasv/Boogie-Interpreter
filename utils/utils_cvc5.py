from cvc5 import Kind, Term, Sort, Solver, SortKind
import math
from interpreter.parser.expression import FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier, BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral, BitvectorLiteral, OldExpression, LogicalNegation, ArithmeticNegation, QuantifiedExpression, IfExpression, Identifier
from interpreter.parser.statement import AssertStatement, AssumeStatement, AssignStatement, Block, CallStatement, GotoStatement, HavocStatement
from interpreter.parser.declaration import StorageDeclaration, ImplementationDeclaration, ProcedureDeclaration
from interpreter.parser.type import BooleanType, IntegerType, CustomType, MapType
from interpreter.utils.cvc5_helper import pretty_print_term, sign_extend, zero_extend
from interpreter.utils.cvc5_serde import (
    SerializedCvc5TermV2 as _SerializedCvc5TermV2,
    deserialize_cvc5_term as _deserialize_cvc5_term,
    hollow_to_str as _hollow_to_str,
)

from interpreter.utils.program import boogie_type_bitwidth
from collections import deque
import pickle
from functools import lru_cache
from cachetools import LRUCache
from interpreter.utils.indent_log import IndentLogger, indent_log

# (cvc5_op, op, num_args, op_bit_width, op_out_width)
fn_to_cvc5_op = {
    # Multiplication
    "$mul.ref": (Kind.BITVECTOR_MULT, 2, 64, 64),
    "$mul.i64": (Kind.BITVECTOR_MULT, 2, 64, 64),
    "$mul.i32": (Kind.BITVECTOR_MULT, 2, 32, 32),
    "$mul.i8": (Kind.BITVECTOR_MULT, 2, 8, 8),

    # Arithmetic operations
    "$add.ref": (Kind.BITVECTOR_ADD, 2, 64, 64),
    "$add.i64": (Kind.BITVECTOR_ADD, 2, 64, 64),
    "$add.i32": (Kind.BITVECTOR_ADD, 2, 32, 32),
    "$add.i8": (Kind.BITVECTOR_ADD, 2, 8, 8),

    "$sub.ref": (Kind.BITVECTOR_SUB, 2, 64, 64),
    "$sub.i64": (Kind.BITVECTOR_SUB, 2, 64, 64),
    "$sub.i32": (Kind.BITVECTOR_SUB, 2, 32, 32),
    "$sub.i16": (Kind.BITVECTOR_SUB, 2, 16, 16),
    "$sub.i8": (Kind.BITVECTOR_SUB, 2, 8, 8),

    # Bitwise operations
    "$and.ref": (Kind.BITVECTOR_AND, 2, 64, 64),
    "$and.i64": (Kind.BITVECTOR_AND, 2, 64, 64),
    "$and.i32": (Kind.BITVECTOR_AND, 2, 32, 32),
    "$and.i8": (Kind.BITVECTOR_AND, 2, 8, 8),
    "$and.i1": (Kind.BITVECTOR_AND, 2, 1, 1),

    "$or.ref": (Kind.BITVECTOR_OR, 2, 64, 64),
    "$or.i64": (Kind.BITVECTOR_OR, 2, 64, 64),
    "$or.i32": (Kind.BITVECTOR_OR, 2, 32, 32),
    "$or.i8": (Kind.BITVECTOR_OR, 2, 8, 8),
    "$or.i1": (Kind.BITVECTOR_OR, 2, 1, 1),

    "$xor.ref": (Kind.BITVECTOR_XOR, 2, 64, 64),
    "$xor.i64": (Kind.BITVECTOR_XOR, 2, 64, 64),
    "$xor.i32": (Kind.BITVECTOR_XOR, 2, 32, 32),
    "$xor.i8": (Kind.BITVECTOR_XOR, 2, 8, 8),
    "$xor.i1": (Kind.BITVECTOR_XOR, 2, 1, 1),

    "$not.i1": (Kind.BITVECTOR_NOT, 1, 1, 1),
    "$not.i8": (Kind.BITVECTOR_NOT, 1, 8, 8),
    "$not.i32": (Kind.BITVECTOR_NOT, 1, 32, 32),
    "$not.i64": (Kind.BITVECTOR_NOT, 1, 64, 64),
    "$not.ref": (Kind.BITVECTOR_NOT, 1, 64, 64),
    "$not.i16": (Kind.BITVECTOR_NOT, 1, 16, 16),

    # SMACK comparison helpers return ``i1`` per the prelude
    # (``smack/lib/smack/Prelude.cpp::IntPred::getIntFuncs``).  The
    # operand width comes from the suffix; the result is always 1-bit
    # (0 or 1).  Encoding output_type=1 lets downstream sort-coercion
    # match assignments such as ``$i2: i1 := $slt.i32($i1, 1000);``
    # and lets logical NOT round-trip ``!($helper(...))`` cleanly.
    # The ``.bool`` variants are emitted by the same prelude code path
    # and return Bool directly.
    "$ne.ref": (Kind.DISTINCT, 2, 64, 1),
    "$ne.ref.bool": (Kind.DISTINCT, 2, 64, bool),
    "$ne.i64": (Kind.DISTINCT, 2, 64, 1),
    "$ne.i64.bool": (Kind.DISTINCT, 2, 64, bool),
    "$ne.i32": (Kind.DISTINCT, 2, 32, 1),
    "$ne.i32.bool": (Kind.DISTINCT, 2, 32, bool),
    "$ne.i8": (Kind.DISTINCT, 2, 8, 1),
    "$ne.i8.bool": (Kind.DISTINCT, 2, 8, bool),

    "$eq.ref": (Kind.EQUAL, 2, 64, 1),
    "$eq.ref.bool": (Kind.EQUAL, 2, 64, bool),
    "$eq.i64": (Kind.EQUAL, 2, 64, 1),
    "$eq.i64.bool": (Kind.EQUAL, 2, 64, bool),
    "$eq.i32": (Kind.EQUAL, 2, 32, 1),
    "$eq.i32.bool": (Kind.EQUAL, 2, 32, bool),
    "$eq.i8": (Kind.EQUAL, 2, 8, 1),
    "$eq.i8.bool": (Kind.EQUAL, 2, 8, bool),
    "$eq.i1": (Kind.EQUAL, 2, 1, 1),
    "$eq.i1.bool": (Kind.EQUAL, 2, 1, bool),

    "$udiv.ref": (Kind.BITVECTOR_UDIV, 2, 64, 64),
    "$udiv.i64": (Kind.BITVECTOR_UDIV, 2, 64, 64),
    "$udiv.i32": (Kind.BITVECTOR_UDIV, 2, 32, 32),
    "$udiv.i8": (Kind.BITVECTOR_UDIV, 2, 8, 8),

    # Signed division
    "$sdiv.i64": (Kind.BITVECTOR_SDIV, 2, 64, 64),
    "$sdiv.i32": (Kind.BITVECTOR_SDIV, 2, 32, 32),
    "$sdiv.i8": (Kind.BITVECTOR_SDIV, 2, 8, 8),

    # SMACK comparison helpers return ``i1``; the ``.bool`` variants
    # return Bool.  See note above $ne.ref and the prelude in
    # ``smack/lib/smack/Prelude.cpp``.
    "$ult.ref": (Kind.BITVECTOR_ULT, 2, 64, 1),
    "$ult.ref.bool": (Kind.BITVECTOR_ULT, 2, 64, bool),
    "$ult.i64": (Kind.BITVECTOR_ULT, 2, 64, 1),
    "$ult.i64.bool": (Kind.BITVECTOR_ULT, 2, 64, bool),
    "$ult.i32": (Kind.BITVECTOR_ULT, 2, 32, 1),
    "$ult.i32.bool": (Kind.BITVECTOR_ULT, 2, 32, bool),
    "$ult.i8": (Kind.BITVECTOR_ULT, 2, 8, 1),
    "$ult.i8.bool": (Kind.BITVECTOR_ULT, 2, 8, bool),
    "$ugt.i64": (Kind.BITVECTOR_UGT, 2, 64, 1),
    "$ugt.i64.bool": (Kind.BITVECTOR_UGT, 2, 64, bool),
    "$ugt.i32": (Kind.BITVECTOR_UGT, 2, 32, 1),
    "$ugt.i32.bool": (Kind.BITVECTOR_UGT, 2, 32, bool),
    "$ugt.i8": (Kind.BITVECTOR_UGT, 2, 8, 1),
    "$ugt.i8.bool": (Kind.BITVECTOR_UGT, 2, 8, bool),
    "$uge.i64": (Kind.BITVECTOR_UGE, 2, 64, 1),
    "$uge.i64.bool": (Kind.BITVECTOR_UGE, 2, 64, bool),
    "$uge.i32": (Kind.BITVECTOR_UGE, 2, 32, 1),
    "$uge.i32.bool": (Kind.BITVECTOR_UGE, 2, 32, bool),
    "$uge.i8": (Kind.BITVECTOR_UGE, 2, 8, 1),
    "$uge.i8.bool": (Kind.BITVECTOR_UGE, 2, 8, bool),

    # Signed greater than
    "$sgt.ref.bool": (Kind.BITVECTOR_SGT, 2, 64, bool),
    "$sgt.i64": (Kind.BITVECTOR_SGT, 2, 64, 1),
    "$sgt.i64.bool": (Kind.BITVECTOR_SGT, 2, 64, bool),
    "$sgt.i32": (Kind.BITVECTOR_SGT, 2, 32, 1),
    "$sgt.i32.bool": (Kind.BITVECTOR_SGT, 2, 32, bool),
    "$sgt.i8": (Kind.BITVECTOR_SGT, 2, 8, 1),
    "$sgt.i8.bool": (Kind.BITVECTOR_SGT, 2, 8, bool),

    # Signed greater than or equal to
    "$sge.ref.bool": (Kind.BITVECTOR_SGE, 2, 64, bool),
    "$sge.i64": (Kind.BITVECTOR_SGE, 2, 64, 1),
    "$sge.i64.bool": (Kind.BITVECTOR_SGE, 2, 64, bool),
    "$sge.i32": (Kind.BITVECTOR_SGE, 2, 32, 1),
    "$sge.i32.bool": (Kind.BITVECTOR_SGE, 2, 32, bool),
    "$sge.i8": (Kind.BITVECTOR_SGE, 2, 8, 1),
    "$sge.i8.bool": (Kind.BITVECTOR_SGE, 2, 8, bool),

    # Signed less than or equal to
    "$sle.i64": (Kind.BITVECTOR_SLE, 2, 64, 1),
    "$sle.i64.bool": (Kind.BITVECTOR_SLE, 2, 64, bool),
    "$sle.i32": (Kind.BITVECTOR_SLE, 2, 32, 1),
    "$sle.i32.bool": (Kind.BITVECTOR_SLE, 2, 32, bool),
    "$sle.i8": (Kind.BITVECTOR_SLE, 2, 8, 1),
    "$sle.i8.bool": (Kind.BITVECTOR_SLE, 2, 8, bool),
    "$sle.ref.bool": (Kind.BITVECTOR_SLE, 2, 64, bool),

    "$slt.ref.bool": (Kind.BITVECTOR_SLT, 2, 64, bool),
    "$slt.i64": (Kind.BITVECTOR_SLT, 2, 64, 1),
    "$slt.i64.bool": (Kind.BITVECTOR_SLT, 2, 64, bool),
    "$slt.i32": (Kind.BITVECTOR_SLT, 2, 32, 1),
    "$slt.i32.bool": (Kind.BITVECTOR_SLT, 2, 32, bool),
    "$slt.i8": (Kind.BITVECTOR_SLT, 2, 8, 1),
    "$slt.i8.bool": (Kind.BITVECTOR_SLT, 2, 8, bool),
    "$ule.i64": (Kind.BITVECTOR_ULE, 2, 64, 1),
    "$ule.i64.bool": (Kind.BITVECTOR_ULE, 2, 64, bool),
    "$ule.i32": (Kind.BITVECTOR_ULE, 2, 32, 1),
    "$ule.i32.bool": (Kind.BITVECTOR_ULE, 2, 32, bool),
    "$ule.i8": (Kind.BITVECTOR_ULE, 2, 8, 1),
    "$ule.i8.bool": (Kind.BITVECTOR_ULE, 2, 8, bool),
    "$urem.i64": (Kind.BITVECTOR_UREM, 2, 64, 64),
    "$urem.i32": (Kind.BITVECTOR_UREM, 2, 32, 32),
    "$urem.i8": (Kind.BITVECTOR_UREM, 2, 8, 8),
    "$srem.i64": (Kind.BITVECTOR_SREM, 2, 64, 64),
    "$srem.i32": (Kind.BITVECTOR_SREM, 2, 32, 32),
    "$srem.i8": (Kind.BITVECTOR_SREM, 2, 8, 8),
    "$shl.i64": (Kind.BITVECTOR_SHL, 2, 64, 64),
    "$shl.i32": (Kind.BITVECTOR_SHL, 2, 32, 32),
    "$shl.i8": (Kind.BITVECTOR_SHL, 2, 8, 8),
    "$lshr.i64": (Kind.BITVECTOR_LSHR, 2, 64, 64),
    "$lshr.i32": (Kind.BITVECTOR_LSHR, 2, 32, 32),
    "$lshr.i8": (Kind.BITVECTOR_LSHR, 2, 8, 8),
    "$ashr.i64": (Kind.BITVECTOR_ASHR, 2, 64, 64),
    "$ashr.i32": (Kind.BITVECTOR_ASHR, 2, 32, 32),
    "$ashr.i8": (Kind.BITVECTOR_ASHR, 2, 8, 8),

    # ——— casts / identity ————————————————————————————————————————————
    "$bitcast.ref.ref": (None, 1, 64, 64),
    "$p2i.ref.i64": (None, 1, 64, 64),
    "$i2p.i64.ref": (None, 1, 64, 64),

    # For binary expressions
    "==": (Kind.EQUAL, 2, None, None),
    "!=": (Kind.DISTINCT, 2, None, None),
    "==>": (Kind.IMPLIES, 2, None, None),
    "||": (Kind.OR, 2, bool, bool),
    "&&": (Kind.AND, 2, bool, bool),
    "<": (Kind.LT, 2, None, None),
    ">": (Kind.GT, 2, None, None),
    "<=": (Kind.LEQ, 2, None, None),
    ">=": (Kind.GEQ, 2, None, None),
    "+": (Kind.ADD, 2, None, None),
    "-": (Kind.SUB, 2, None, None),
    "*": (Kind.MULT, 2, None, None),
    "/": (Kind.DIVISION, 2, None, None),
    "%": (Kind.INTS_MODULUS, 2, None, None),
}


# Exact width-sensitive source intrinsics used by SMACK's integer encoding.
# ``_int_enc_special_term`` lowers these names to
# ``bv2nat(bvop(int2bv(width, ...)))``.  The reverse structural lowerer uses
# the same table below to recover an executable Boogie FunctionApplication;
# Boolean BinaryExpression operators are a different theory and cannot carry
# these semantics.
_INT_ENC_BITWISE_OPS = {
    "$and": Kind.BITVECTOR_AND,
    "$or": Kind.BITVECTOR_OR,
    "$xor": Kind.BITVECTOR_XOR,
    "$not": Kind.BITVECTOR_NOT,
    "$shl": Kind.BITVECTOR_SHL,
    "$lshr": Kind.BITVECTOR_LSHR,
    "$ashr": Kind.BITVECTOR_ASHR,
}
_INT_ENC_BITWISE_ARITY = {
    Kind.BITVECTOR_AND: 2,
    Kind.BITVECTOR_OR: 2,
    Kind.BITVECTOR_XOR: 2,
    Kind.BITVECTOR_NOT: 1,
    Kind.BITVECTOR_SHL: 2,
    Kind.BITVECTOR_LSHR: 2,
    Kind.BITVECTOR_ASHR: 2,
}


# Reverse mapping: cvc5 Kind → SMACK function name (prefer i32 for BV32)
_CVC5_KIND_TO_BOOGIE = {}
for _name, (_kind, _nargs, _width, _out) in fn_to_cvc5_op.items():
    if _name.startswith("$") and _width == 32 and _kind is not None:
        if _kind not in _CVC5_KIND_TO_BOOGIE:
            _CVC5_KIND_TO_BOOGIE[_kind] = _name

# Width-aware reverse map for the exact integer-encoding bitwise family.  Its
# support set comes from ``fn_to_cvc5_op``, the existing owner of executable
# SMACK intrinsic names and widths.  Prefer ``.i64`` over the equivalent
# ``.ref`` spelling so a bitvector sort has one deterministic projection.
_INT_ENC_BITWISE_BOOGIE_BY_KIND_WIDTH = {}
for _name, (_kind, _nargs, _width, _out) in fn_to_cvc5_op.items():
    _base = next(
        (base for base, kind in _INT_ENC_BITWISE_OPS.items()
         if kind == _kind),
        None,
    )
    if (
        _base is None
        or not isinstance(_width, int)
        or _nargs != _INT_ENC_BITWISE_ARITY[_kind]
        or _out != _width
    ):
        continue
    _key = (_kind, _width)
    _canonical_name = f"{_base}.i{_width}"
    if (
        _key not in _INT_ENC_BITWISE_BOOGIE_BY_KIND_WIDTH
        or _name == _canonical_name
    ):
        _INT_ENC_BITWISE_BOOGIE_BY_KIND_WIDTH[_key] = _name
del _base, _canonical_name, _key, _kind, _name, _nargs, _out, _width


class Cvc5ToBoogieLoweringUnavailable(ValueError):
    """The host has no semantics-preserving Boogie lowering for a cvc5 term.

    This is deliberately distinct from malformed caller input.  Native proof
    obligations can contain well-formed cvc5 operators that this converter has
    not implemented yet; concrete falsification must report that condition as
    non-semantic infrastructure unavailability instead of blaming the model's
    exact proof-obligation identifier.
    """


class _SerializedSortView:
    """The sort operations used by the structural Boogie lowerer.

    A serialized term is the cross-process authority.  Adapting its immutable
    sort record lets the one cvc5-to-Boogie lowering walk that authority
    directly, without rebuilding a process-local cvc5 term first.
    """

    __slots__ = ("_sort",)

    def __init__(self, sort):
        self._sort = sort

    def isBitVector(self) -> bool:
        return self._sort.kind == "BITVECTOR_SORT"

    def getBitVectorSize(self) -> int:
        if not self.isBitVector():
            raise RuntimeError("serialized sort is not a bitvector")
        return int(self._sort.args[0])


class _SerializedOpView:
    __slots__ = ("_indices",)

    def __init__(self, indices):
        self._indices = tuple(indices)

    def getNumIndices(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> int:
        return self._indices[index]


class _SerializedTermView:
    """Term-shaped read-only view over one canonical serialized AST."""

    __slots__ = ("_wire",)

    def __init__(self, wire: _SerializedCvc5TermV2):
        self._wire = wire

    def getNumChildren(self) -> int:
        return len(self._wire.node.children)

    def __getitem__(self, index: int):
        return _SerializedTermView(self._wire.children[index])

    def getKind(self):
        return Kind[self._wire.node.kind]

    def getSort(self):
        return _SerializedSortView(self._wire.node.sort)

    def getOp(self):
        return _SerializedOpView(self._wire.node.op_indices)

    def isBooleanValue(self) -> bool:
        return self._wire.node.kind == "CONST_BOOLEAN"

    def getBooleanValue(self) -> bool:
        if not self.isBooleanValue():
            raise RuntimeError("serialized term is not a Boolean literal")
        return bool(self._wire.node.value)

    def isIntegerValue(self) -> bool:
        return self._wire.node.kind == "CONST_INTEGER"

    def getIntegerValue(self) -> int:
        if not self.isIntegerValue():
            raise RuntimeError("serialized term is not an integer literal")
        return int(self._wire.node.value)

    def isBitVectorValue(self) -> bool:
        return self._wire.node.kind == "CONST_BITVECTOR"

    def getBitVectorValue(self, base: int = 2) -> str:
        if not self.isBitVectorValue():
            raise RuntimeError("serialized term is not a bitvector literal")
        value = int(self._wire.node.value)
        if base == 2:
            return format(value, f"0{self.getSort().getBitVectorSize()}b")
        if base == 10:
            return str(value)
        raise ValueError(f"unsupported bitvector display base: {base}")

    def getSymbol(self) -> str:
        if self._wire.node.kind not in {"CONSTANT", "VARIABLE"}:
            raise RuntimeError("serialized term has no symbol")
        return self._wire.node.symbol

    def __str__(self) -> str:
        return _hollow_to_str(self._wire, max_depth=10_000)


def cvc5_to_boogie(term) -> str:
    """Render a cvc5 term as Boogie/SMACK syntax the parser can read back
    (e.g. ``$mul.i32($i6, $sub.i32($i6, 1))``).

    SINGLE SOURCE OF TRUTH: this is just the string rendering of
    :func:`cvc5_to_boogie_ast` — the one structural cvc5->Boogie lowering. The
    AST ``repr`` is fully parenthesized, so a biconditional ``(A==>B)&&(B==>A)``
    keeps its grouping; the previous hand-written string emitter dropped those
    parens and the relation collapsed under ``&&`` precedence (the bug that made
    obligations like the pc2881 sign-guard un-parseable). Unsupported terms
    fail at the structural lowering boundary.
    """
    if term is None:
        raise ValueError("cvc5_to_boogie: term is None")
    return repr(cvc5_to_boogie_ast(term))


def cvc5_to_boogie_ast(
    term,
    depth: int = 0,
    *,
    preserve_ite_values: bool = False,
):
    """Structurally lower a cvc5 ``Term`` into a Boogie expression AST — the
    SAME ``interpreter.parser.expression`` nodes the Boogie parser builds —
    WITHOUT round-tripping through an infix display string.

    This is the term-faithful sibling of :func:`cvc5_to_boogie` (which renders
    text and, for boolean connectives, drops the grouping parentheses a
    biconditional ``(A ==> B) && (B ==> A)`` needs: under operator precedence
    the printed ``A ==> B && B ==> A`` re-parses as ``A ==> (B && B) ==> A`` and
    the relation collapses). Because an AST is a tree, grouping is exact by
    construction — an ``AND`` of two ``IMPLIES`` is two distinct subtrees, never
    a precedence-ambiguous flat string.

    Used by the obligation-falsification inject path (``try-violate``): the
    frozen obligation carries the live serialized cvc5 term, which is
    deserialized and lowered here straight into the ``assert`` AST the native
    interpreter evaluates — no printed-text parse in the loop.

    Raises :class:`Cvc5ToBoogieLoweringUnavailable` on a well-formed ``Kind``
    for which this host has no semantics-preserving lowering.
    """
    from interpreter.parser.expression import (
        BinaryExpression, LogicalNegation, ArithmeticNegation,
        FunctionApplication, FunctionIdentifier, StorageIdentifier,
        IntegerLiteral, BooleanLiteral, IfExpression, MapSelect, MapUpdate,
    )
    if term is None:
        raise ValueError("cvc5_to_boogie_ast: term is None")
    if isinstance(term, _SerializedCvc5TermV2):
        term = _SerializedTermView(term)
    if depth > 60:
        raise ValueError("cvc5_to_boogie_ast: term nesting exceeds 60")

    n = term.getNumChildren()

    # --- leaves ---
    if n == 0:
        if term.isBooleanValue():
            return BooleanLiteral(bool(term.getBooleanValue()))
        if term.isIntegerValue():
            return IntegerLiteral(int(term.getIntegerValue()))
        if term.isBitVectorValue():
            return IntegerLiteral(int(term.getBitVectorValue(), 2))
        try:
            sym = term.getSymbol()
        except Exception:
            sym = str(term)
        return StorageIdentifier(name=sym)

    kind = term.getKind()
    rec = lambda i: cvc5_to_boogie_ast(
        term[i], depth + 1, preserve_ite_values=preserve_ite_values)

    def _op_indices(value):
        op = value.getOp()
        return tuple(int(str(op[i])) for i in range(op.getNumIndices()))

    def _unsigned_bv(value, value_depth):
        """Lower a BV value to its exact unsigned mathematical integer.

        Integer-encoding simplification routinely turns source intrinsics into
        ``ubv_to_int(concat(extract(int_to_bv(x)), ...))``.  Reconstruct that
        value with integer modulus/division/arithmetic, whose operands are
        exact and non-negative at every extraction boundary.  Unsupported BV
        operators retain a typed failure instead of receiving an approximate
        source expression.
        """
        if value_depth > 60:
            raise Cvc5ToBoogieLoweringUnavailable(
                "cvc5_to_boogie_ast: bitvector nesting exceeds 60")
        value_kind = value.getKind()
        value_n = value.getNumChildren()
        if value_n == 0 and value.isBitVectorValue():
            return IntegerLiteral(int(value.getBitVectorValue(), 2))
        if value_kind == Kind.INT_TO_BITVECTOR and value_n == 1:
            width = value.getSort().getBitVectorSize()
            source = cvc5_to_boogie_ast(
                value[0], value_depth + 1,
                preserve_ite_values=preserve_ite_values)
            return BinaryExpression(
                lhs=source, op="%", rhs=IntegerLiteral(1 << width))
        if value_kind == Kind.BITVECTOR_EXTRACT and value_n == 1:
            high, low = _op_indices(value)
            source = _unsigned_bv(value[0], value_depth + 1)
            shifted = source if low == 0 else BinaryExpression(
                lhs=source, op="/", rhs=IntegerLiteral(1 << low))
            extracted_width = high - low + 1
            return BinaryExpression(
                lhs=shifted, op="%",
                rhs=IntegerLiteral(1 << extracted_width))
        if value_kind == Kind.BITVECTOR_CONCAT and value_n >= 2:
            result = _unsigned_bv(value[0], value_depth + 1)
            for index in range(1, value_n):
                child = value[index]
                child_width = child.getSort().getBitVectorSize()
                result = BinaryExpression(
                    lhs=BinaryExpression(
                        lhs=result, op="*",
                        rhs=IntegerLiteral(1 << child_width)),
                    op="+",
                    rhs=_unsigned_bv(child, value_depth + 1),
                )
            return result
        if value_kind == Kind.BITVECTOR_ZERO_EXTEND and value_n == 1:
            return _unsigned_bv(value[0], value_depth + 1)
        bitwise_arity = _INT_ENC_BITWISE_ARITY.get(value_kind)
        if bitwise_arity == value_n:
            width = value.getSort().getBitVectorSize()
            intrinsic = _INT_ENC_BITWISE_BOOGIE_BY_KIND_WIDTH.get(
                (value_kind, width))
            if intrinsic is not None:
                return FunctionApplication(
                    function=FunctionIdentifier(name=intrinsic),
                    arguments=[
                        _unsigned_bv(value[index], value_depth + 1)
                        for index in range(value_n)
                    ],
                )
        if value_kind in {
                Kind.BITVECTOR_ADD, Kind.BITVECTOR_SUB,
                Kind.BITVECTOR_MULT} and value_n == 2:
            op = {
                Kind.BITVECTOR_ADD: "+",
                Kind.BITVECTOR_SUB: "-",
                Kind.BITVECTOR_MULT: "*",
            }[value_kind]
            arithmetic = BinaryExpression(
                lhs=_unsigned_bv(value[0], value_depth + 1), op=op,
                rhs=_unsigned_bv(value[1], value_depth + 1))
            return BinaryExpression(
                lhs=arithmetic, op="%",
                rhs=IntegerLiteral(
                    1 << value.getSort().getBitVectorSize()))
        if value_kind == Kind.BITVECTOR_NEG and value_n == 1:
            arithmetic = ArithmeticNegation(
                expression=_unsigned_bv(value[0], value_depth + 1))
            return BinaryExpression(
                lhs=arithmetic, op="%",
                rhs=IntegerLiteral(
                    1 << value.getSort().getBitVectorSize()))
        raise Cvc5ToBoogieLoweringUnavailable(
            "cvc5_to_boogie_ast: unsigned bitvector lowering unavailable for "
            f"{value_kind} (nchildren={value_n})")

    # --- unary ---
    if kind == Kind.NOT:
        return LogicalNegation(expression=rec(0))
    if kind == Kind.NEG:                       # integer unary minus
        return ArithmeticNegation(expression=rec(0))
    if kind == Kind.BITVECTOR_NEG:             # SMACK has no $neg.i32
        w = term.getSort().getBitVectorSize()
        return FunctionApplication(
            function=FunctionIdentifier(name=f"$sub.i{w}"),
            arguments=[IntegerLiteral(0), rec(0)])
    if kind == Kind.INT_TO_BITVECTOR:
        return _unsigned_bv(term, depth)
    if kind == Kind.BITVECTOR_UBV_TO_INT and n == 1:
        return _unsigned_bv(term[0], depth + 1)
    if kind == Kind.BITVECTOR_SBV_TO_INT and n == 1:
        width = term[0].getSort().getBitVectorSize()
        unsigned = _unsigned_bv(term[0], depth + 1)
        return IfExpression(
            condition=BinaryExpression(
                lhs=unsigned.clone(), op=">=",
                rhs=IntegerLiteral(1 << (width - 1))),
            then=BinaryExpression(
                lhs=unsigned.clone(), op="-", rhs=IntegerLiteral(1 << width)),
            else_=unsigned,
        )

    # --- n-ary connectives / arithmetic folded left into binary nodes ---
    _FOLD = {Kind.AND: "&&", Kind.OR: "||", Kind.ADD: "+", Kind.MULT: "*"}
    if kind in _FOLD:
        node = rec(0)
        for i in range(1, n):
            node = BinaryExpression(lhs=node, op=_FOLD[kind], rhs=rec(i))
        return node

    # --- binary relational / connective / arithmetic ---
    _BIN = {Kind.IMPLIES: "==>", Kind.EQUAL: "==", Kind.DISTINCT: "!=",
            Kind.LT: "<", Kind.LEQ: "<=", Kind.GT: ">", Kind.GEQ: ">=",
            Kind.SUB: "-", Kind.INTS_DIVISION: "/", Kind.INTS_MODULUS: "%"}
    if kind in _BIN and n == 2:
        return BinaryExpression(lhs=rec(0), op=_BIN[kind], rhs=rec(1))

    # cvc5's explicitly total integer operators have fixed zero-divisor
    # values: div_total(x, 0) = 0 and mod_total(x, 0) = x.  Preserve those
    # values explicitly instead of delegating the zero case to Boogie's
    # backend-specific division/modulus interpretation.
    if kind in {Kind.INTS_DIVISION_TOTAL, Kind.INTS_MODULUS_TOTAL} and n == 2:
        numerator = rec(0)
        denominator = rec(1)
        condition = BinaryExpression(
            lhs=denominator.clone(), op="==", rhs=IntegerLiteral(0))
        arithmetic = BinaryExpression(
            lhs=numerator.clone(),
            op="/" if kind == Kind.INTS_DIVISION_TOTAL else "%",
            rhs=denominator,
        )
        zero_value = (
            IntegerLiteral(0)
            if kind == Kind.INTS_DIVISION_TOTAL else numerator
        )
        return IfExpression(
            condition=condition,
            then=zero_value,
            else_=arithmetic,
        )

    # n-ary DISTINCT: all-pairwise !=  (rare; cvc5 usually binarizes)
    if kind == Kind.DISTINCT and n > 2:
        parts = [BinaryExpression(lhs=rec(i), op="!=", rhs=rec(j))
                 for i in range(n) for j in range(i + 1, n)]
        node = parts[0]
        for p in parts[1:]:
            node = BinaryExpression(lhs=node, op="&&", rhs=p)
        return node

    # --- if-then-else ---
    if kind == Kind.ITE and n == 3:
        def _leaf_val(t, v):
            if t.getNumChildren():
                return False
            try:
                if t.isBitVectorValue():
                    return int(t.getBitVectorValue(), 2) == v
                if t.isIntegerValue():
                    return int(t.getIntegerValue()) == v
            except Exception:
                return False
            return False
        # bool->bv cast ITE(c, 1, 0) is just the boolean c (matches cvc5_to_boogie)
        if (not preserve_ite_values
                and _leaf_val(term[1], 1) and _leaf_val(term[2], 0)):
            return rec(0)
        return IfExpression(condition=rec(0), then=rec(1), else_=rec(2))

    # --- arrays ---
    if kind == Kind.SELECT and n == 2:
        return MapSelect(map=rec(0), indexes=[rec(1)])
    if kind == Kind.STORE and n == 3:
        return MapUpdate(map=rec(0), indexes=[rec(1)], value=rec(2))

    # --- cvc5 BV simplification artifacts -> readable SMACK form (semantics-
    # preserving). These are the display niceties cvc5_to_boogie advertises;
    # keeping them here means the LLM string AND the injected assert share one
    # lowering. ---
    def _mul32(a_ast, b_ast):
        return FunctionApplication(
            function=FunctionIdentifier(name="$mul.i32"),
            arguments=[a_ast, b_ast])

    if kind == Kind.BITVECTOR_CONCAT and n == 2:
        hi, lo = term[0], term[1]
        # CONCAT(EXTRACT(.,.,x), #b0..0) -> $mul.i32(2^K, x)  (left shift)
        if (lo.isBitVectorValue() and int(lo.getBitVectorValue(), 2) == 0
                and hi.getKind() == Kind.BITVECTOR_EXTRACT):
            n_zeros = lo.getSort().getBitVectorSize()
            return _mul32(IntegerLiteral(1 << n_zeros),
                          cvc5_to_boogie_ast(
                              hi[0], depth + 1,
                              preserve_ite_values=preserve_ite_values))
        # CONCAT(#b0..0, x) -> x  (zero-extension, transparent in Boogie)
        if hi.isBitVectorValue() and int(hi.getBitVectorValue(), 2) == 0:
            return cvc5_to_boogie_ast(
                lo, depth + 1,
                preserve_ite_values=preserve_ite_values)

    if kind == Kind.BITVECTOR_ADD and n == 2:
        a, b = term[0], term[1]
        # X^2 - X  ->  $mul.i32(X, $sub.iW(X, 1))
        if (a.getKind() == Kind.BITVECTOR_MULT and a.getNumChildren() == 2
                and b.getKind() == Kind.BITVECTOR_NEG
                and str(a[0]) == str(a[1]) == str(b[0])):
            w = term.getSort().getBitVectorSize()
            return _mul32(cvc5_to_boogie_ast(
                              a[0], depth + 1,
                              preserve_ite_values=preserve_ite_values),
                          FunctionApplication(
                              function=FunctionIdentifier(name=f"$sub.i{w}"),
                              arguments=[cvc5_to_boogie_ast(
                                  a[0], depth + 1,
                                  preserve_ite_values=preserve_ite_values),
                                         IntegerLiteral(1)]))
        # X^2 + X  ->  $mul.i32(X, $add.iW(X, 1))
        if (a.getKind() == Kind.BITVECTOR_MULT and a.getNumChildren() == 2
                and b.getKind() == Kind.CONSTANT and b.getSort().isBitVector()
                and str(a[0]) == str(a[1]) == str(b)):
            w = term.getSort().getBitVectorSize()
            return _mul32(cvc5_to_boogie_ast(
                              a[0], depth + 1,
                              preserve_ite_values=preserve_ite_values),
                          FunctionApplication(
                              function=FunctionIdentifier(name=f"$add.i{w}"),
                              arguments=[cvc5_to_boogie_ast(
                                  a[0], depth + 1,
                                  preserve_ite_values=preserve_ite_values),
                                         IntegerLiteral(1)]))

    # --- SMACK intrinsic / BV-op function calls ($add.i32, $sge.i32, ...) ---
    if kind in _CVC5_KIND_TO_BOOGIE:
        return FunctionApplication(
            function=FunctionIdentifier(name=_CVC5_KIND_TO_BOOGIE[kind]),
            arguments=[rec(i) for i in range(n)])

    raise Cvc5ToBoogieLoweringUnavailable(
        f"cvc5_to_boogie_ast: unhandled kind {kind} (nchildren={n})")


def generate_cvc5_function_map(solver: Solver):
    """
    Returns a lazily initialized more_func_map using the provided stateful solver.
    """
    if _INTEGER_ENCODING:
        # In integer encoding, sext/zext/trunc are identity (integers have no width)
        int_cast_map = {}
        for name in ["$sext.i32.i64", "$sext.i8.i32", "$sext.i16.i32",
                      "$zext.i32.i64", "$zext.i8.i32", "$zext.i8.i64",
                      "$zext.i1.i32", "$zext.i1.i64", "$zext.i16.i32", "$zext.i16.i64",
                      "$trunc.i32.i8", "$trunc.i32.i16", "$trunc.i64.i8",
                      "$trunc.i64.i16", "$trunc.i64.i32", "$trunc.i32.i1", "$trunc.i64.i1",
                      "$p2i.ref.i64", "$i2p.i64.ref", "$bitcast.ref.ref"]:
            int_cast_map[name] = (None, 1, None, None)
        return int_cast_map | get_fn_map()

    return {
        # Sign extend
        "$sext.i32.i64": (solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 64), 1, 32, 64),
        "$sext.i8.i32": (solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 32), 1, 8, 32),
        "$sext.i16.i32": (solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 32), 1, 16, 32),

        # Zero extend
        "$zext.i32.i64": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 64), 1, 32, 64),
        "$zext.i8.i32": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 32), 1, 8, 32),
        "$zext.i8.i64": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 64), 1, 8, 64),
        "$zext.i1.i32": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 32), 1, 1, 32),
        "$zext.i1.i64": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 64), 1, 1, 64),
        "$zext.i16.i32": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 32), 1, 16, 32),
        "$zext.i16.i64": (solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 64), 1, 16, 64),

        # Truncate
        "$trunc.i32.i8": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 8, 0), 1, 32, 8),
        "$trunc.i32.i16": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 16, 0), 1, 32, 16),
        "$trunc.i64.i8": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 8, 0), 1, 64, 8),
        "$trunc.i64.i16": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 16, 0), 1, 64, 16),
        "$trunc.i64.i32": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 32, 0), 1, 64, 32),
        "$trunc.i32.i1": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 1, 0), 1, 32, 1),
        "$trunc.i64.i1": (solver.mkOp(Kind.BITVECTOR_EXTRACT, 1, 0), 1, 64, 1),
    } | get_fn_map()

def extract_variable_terms(term) -> set[Term]:
    variables = set()
    stack = [term]

    while stack:
        current_term = stack.pop()
        if current_term.getKind() == Kind.CONSTANT:
            variables.add(current_term)
        else:
            stack.extend(current_term)
    return variables

def extract_variable_terms_keep_select(term) -> set[Term]:
    variables = set()
    stack = [term]

    while stack:
        current_term = stack.pop()
        if current_term.getKind() == Kind.CONSTANT:
            variables.add(current_term)
        elif current_term.getKind() == Kind.SELECT:
            variables.add(current_term)
        else:
            stack.extend(current_term)
    return variables

def extract_all_constants(term) -> set[Term]:
    variables = set()
    stack = [term]

    while stack:
        current_term = stack.pop()
        if current_term.isBitVectorValue():
            variables.add(current_term)
        else:
            stack.extend(current_term)
    return variables


def dump_solver_assertions(solver, logger=None):
    if logger:
        logger.debug("solver assertions:")
        for a in solver.getAssertions():
            logger.debug(f"  - {pretty_print_term(a)}")
    else:
        IndentLogger.debug("solver assertions:")
        for a in solver.getAssertions():
            IndentLogger.debug(f"  - {pretty_print_term(a)}")

def cvc5_cast_to_int(solver, expr):
    if expr.getSort() == solver.getIntegerSort():
        return expr
    if expr.getSort() == solver.getBooleanSort():
        zero = solver.mkInteger(0)
        one = solver.mkInteger(1)
        return solver.mkTerm(Kind.ITE, expr, one, zero)
    # BV→Int: not supported (causes solver hangs on ubv_to_int terms).
    # Integer encoding + BV memory arrays need SMACK-level fix.
    return expr

def cvc5_cast_to_bv(solver, expr, bitwidth, zext=False) -> Term:
    extend_fn = zero_extend if zext else sign_extend

    if expr.getSort() == solver.getBooleanSort():
        one_bv = solver.mkBitVector(bitwidth, 1)
        zero_bv = solver.mkBitVector(bitwidth, 0)
        return solver.mkTerm(Kind.ITE, expr, one_bv, zero_bv)
    elif expr.getSort() == solver.getIntegerSort():
        op = solver.mkOp(Kind.INT_TO_BITVECTOR, bitwidth)
        return solver.mkTerm(op, expr)
    else:
        return extend_fn(solver, expr, bitwidth)
    
def assign_fix_type(solver, lhs_cvc5, rhs_cvc5):
    def fix_helper(dst, src):
        if dst.getSort() == solver.getIntegerSort():
            fixed = cvc5_cast_to_int(solver, src)
        elif dst.getSort() == solver.getBooleanSort():
            fixed = cvc5_cast_to_bool(solver, src)
        elif dst.getSort().isBitVector():
            bit_width = dst.getSort().getBitVectorSize()
            fixed = cvc5_cast_to_bv(solver, src, bit_width)
        else:
            IndentLogger.debug("unknown sort", dst.getSort(), src.getSort()) 
            assert False 
        return fixed

    if lhs_cvc5.getSort() != rhs_cvc5.getSort():
        if lhs_cvc5.isBitVectorValue():
            lhs_cvc5 = fix_helper(rhs_cvc5, lhs_cvc5)
        elif rhs_cvc5.getKind() == Kind.SELECT and lhs_cvc5.isBitVectorValue():
            IndentLogger.debug(f"[assign_fix_type] SELECT: {lhs_cvc5} {rhs_cvc5}")
            lhs_cvc5 = fix_helper(rhs_cvc5, lhs_cvc5)
        else:
            rhs_cvc5 = fix_helper(lhs_cvc5, rhs_cvc5)
    return lhs_cvc5, rhs_cvc5

def cvc5_cast_to_bool(solver, expr):
    if expr.getSort() != solver.getBooleanSort():          
        return solver.mkTerm(Kind.EQUAL, expr, solver.mkBitVector(expr.getSort().getBitVectorSize(), 1))
    else:
        return expr
    
# When True, SMACK integer encoding is active (type i32 = int).
# All iN types map to integer sort instead of bitvector sort.
_INTEGER_ENCODING = False

# Integer encoding function map: arithmetic uses integer ops, comparisons
# use integer comparisons returning ITE(cond, 1, 0).  'int_cmp' tag means
# the convert_expr_cvc5 path should wrap the result in ITE → int.
_INT_ENC_FN_MAP = {
    # Arithmetic → integer ops
    "$mul.ref": (Kind.MULT, 2, None, None),
    "$mul.i64": (Kind.MULT, 2, None, None),
    "$mul.i32": (Kind.MULT, 2, None, None),
    "$mul.i8":  (Kind.MULT, 2, None, None),
    "$add.ref": (Kind.ADD, 2, None, None),
    "$add.i64": (Kind.ADD, 2, None, None),
    "$add.i32": (Kind.ADD, 2, None, None),
    "$add.i8":  (Kind.ADD, 2, None, None),
    "$sub.ref": (Kind.SUB, 2, None, None),
    "$sub.i64": (Kind.SUB, 2, None, None),
    "$sub.i32": (Kind.SUB, 2, None, None),
    "$sub.i16": (Kind.SUB, 2, None, None),
    "$sub.i8":  (Kind.SUB, 2, None, None),
    # Equality / inequality (already sort-polymorphic).  ``int_cmp``
    # returns the i1-style 0/1 integer that Boogie's prelude uses for
    # the non-``.bool`` variant; the ``.bool`` variant returns Bool.
    "$eq.ref": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.ref.bool": (Kind.EQUAL, 2, None, bool),
    "$eq.i64": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i64.bool": (Kind.EQUAL, 2, None, bool),
    "$eq.i32": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i32.bool": (Kind.EQUAL, 2, None, bool),
    "$eq.i8":  (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i8.bool": (Kind.EQUAL, 2, None, bool),
    "$eq.i1":  (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i1.bool": (Kind.EQUAL, 2, None, bool),
    "$ne.ref": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.ref.bool": (Kind.DISTINCT, 2, None, bool),
    "$ne.i64": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i64.bool": (Kind.DISTINCT, 2, None, bool),
    "$ne.i32": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i32.bool": (Kind.DISTINCT, 2, None, bool),
    "$ne.i8":  (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i8.bool": (Kind.DISTINCT, 2, None, bool),
    # Signed comparisons → integer comparisons
    "$slt.i32": (Kind.LT, 2, None, "int_cmp"),
    "$slt.i32.bool": (Kind.LT, 2, None, bool),
    "$slt.i64": (Kind.LT, 2, None, "int_cmp"),
    "$slt.i64.bool": (Kind.LT, 2, None, bool),
    "$slt.i8":  (Kind.LT, 2, None, "int_cmp"),
    "$slt.i8.bool": (Kind.LT, 2, None, bool),
    "$slt.ref.bool": (Kind.LT, 2, None, bool),
    "$sle.i32": (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.i32.bool": (Kind.LEQ, 2, None, bool),
    "$sle.i64": (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.i64.bool": (Kind.LEQ, 2, None, bool),
    "$sle.i8":  (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.i8.bool": (Kind.LEQ, 2, None, bool),
    "$sle.ref.bool": (Kind.LEQ, 2, None, bool),
    "$sgt.i32": (Kind.GT, 2, None, "int_cmp"),
    "$sgt.i32.bool": (Kind.GT, 2, None, bool),
    "$sgt.i64": (Kind.GT, 2, None, "int_cmp"),
    "$sgt.i64.bool": (Kind.GT, 2, None, bool),
    "$sgt.i8":  (Kind.GT, 2, None, "int_cmp"),
    "$sgt.i8.bool": (Kind.GT, 2, None, bool),
    "$sgt.ref.bool": (Kind.GT, 2, None, bool),
    "$sge.i32": (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.i32.bool": (Kind.GEQ, 2, None, bool),
    "$sge.i64": (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.i64.bool": (Kind.GEQ, 2, None, bool),
    "$sge.i8":  (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.i8.bool": (Kind.GEQ, 2, None, bool),
    "$sge.ref.bool": (Kind.GEQ, 2, None, bool),
    # Unsigned comparisons → same as signed in integer mode (no wraparound)
    "$ult.i32": (Kind.LT, 2, None, "int_cmp"),
    "$ult.i32.bool": (Kind.LT, 2, None, bool),
    "$ult.i64": (Kind.LT, 2, None, "int_cmp"),
    "$ult.i64.bool": (Kind.LT, 2, None, bool),
    "$ult.i8":  (Kind.LT, 2, None, "int_cmp"),
    "$ult.i8.bool": (Kind.LT, 2, None, bool),
    "$ult.ref":  (Kind.LT, 2, None, "int_cmp"),
    "$ult.ref.bool": (Kind.LT, 2, None, bool),
    "$ule.i32": (Kind.LEQ, 2, None, "int_cmp"),
    "$ule.i32.bool": (Kind.LEQ, 2, None, bool),
    "$ule.i64": (Kind.LEQ, 2, None, "int_cmp"),
    "$ule.i64.bool": (Kind.LEQ, 2, None, bool),
    "$ule.i8":  (Kind.LEQ, 2, None, "int_cmp"),
    "$ule.i8.bool": (Kind.LEQ, 2, None, bool),
    "$ugt.i32": (Kind.GT, 2, None, "int_cmp"),
    "$ugt.i32.bool": (Kind.GT, 2, None, bool),
    "$ugt.i64": (Kind.GT, 2, None, "int_cmp"),
    "$ugt.i64.bool": (Kind.GT, 2, None, bool),
    "$ugt.i8":  (Kind.GT, 2, None, "int_cmp"),
    "$ugt.i8.bool": (Kind.GT, 2, None, bool),
    "$uge.i32": (Kind.GEQ, 2, None, "int_cmp"),
    "$uge.i32.bool": (Kind.GEQ, 2, None, bool),
    "$uge.i64": (Kind.GEQ, 2, None, "int_cmp"),
    "$uge.i64.bool": (Kind.GEQ, 2, None, bool),
    "$uge.i8":  (Kind.GEQ, 2, None, "int_cmp"),
    "$uge.i8.bool": (Kind.GEQ, 2, None, bool),
    # Division / remainder. The prelude defines $sdiv = $udiv = $idiv
    # (SMT-LIB Euclidean div) and $urem = $smod (Euclidean mod); those map
    # directly. $srem is NOT plain mod — the prelude's C-remainder
    # correction formula is built by _int_enc_special_term (the old
    # INTS_MODULUS entry here diverged from the prelude on negative
    # dividends and was a verifier-model bug caught by the kernel diff).
    "$sdiv.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$sdiv.i64": (Kind.INTS_DIVISION, 2, None, None),
    "$udiv.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$udiv.i64": (Kind.INTS_DIVISION, 2, None, None),
    "$urem.i32": (Kind.INTS_MODULUS, 2, None, None),
    "$urem.i64": (Kind.INTS_MODULUS, 2, None, None),
    # Bitwise/shift ops ($and/$or/$xor/$not/$shl/$lshr/$ashr.iN) are handled
    # by the int→bv(width)→op→nat handler (_int_enc_special_term) BEFORE the
    # fn-map lookup — no entries here. The historical placeholder kinds
    # (MULT/ADD/SUB/DIVISION) silently computed garbage; the kernel diff
    # (interpreter/tests/differential/test_smt_kernel_diff.py) pinned them.
    # Casts — identity in integer mode
    "$bitcast.ref.ref": (None, 1, None, None),
    "$p2i.ref.i64": (None, 1, None, None),
    "$i2p.i64.ref": (None, 1, None, None),
    # Binary expressions
    "==": (Kind.EQUAL, 2, None, None),
    "!=": (Kind.DISTINCT, 2, None, None),
    "<":  (Kind.LT, 2, None, None),
    "<=": (Kind.LEQ, 2, None, None),
    ">":  (Kind.GT, 2, None, None),
    ">=": (Kind.GEQ, 2, None, None),
    "+":  (Kind.ADD, 2, None, None),
    "-":  (Kind.SUB, 2, None, None),
    "*":  (Kind.MULT, 2, None, None),
    "/":  (Kind.INTS_DIVISION, 2, None, None),
    "%":  (Kind.INTS_MODULUS, 2, None, None),
    "&&": (Kind.AND, 2, bool, bool),
    "||": (Kind.OR, 2, bool, bool),
    "==>": (Kind.IMPLIES, 2, None, None),
}

# Residual division intrinsics of the integer encoding: {:builtin "div"}
# $idiv.iN and {:builtin "mod"} $smod.iN — the ONLY div/rem functions left in
# statements once the FunctionInlinePass has expanded the {:inline} prelude
# wrappers ($sdiv/$udiv/$srem/$urem). Width-insensitive in ℤ.
for _w in ("i1", "i5", "i6", "i8", "i16", "i24", "i32", "i33", "i40", "i48",
           "i56", "i64", "i128", "ref"):
    _INT_ENC_FN_MAP[f"$idiv.{_w}"] = (Kind.INTS_DIVISION, 2, None, None)
    _INT_ENC_FN_MAP[f"$smod.{_w}"] = (Kind.INTS_MODULUS, 2, None, None)
del _w

# --- Int-encoding special handler: bitwise/shift roundtrip + $srem ---------
#
# The int→bv(width)→op→nat handler (the "_INT_ENC_BITWISE_OPS handler" that
# older comments referenced but was never built): bitwise and shift intrinsics
# are the only width-sensitive residual ops in the integer encoding. Each is
# modeled EXACTLY as SMT-LIB `bv2nat(bvop((_ int2bv w) a, (_ int2bv w) b))`,
# which is also what the native interpreter's `builtins::int` implements.
_INT_ENC_WIDTH_SUFFIX = {
    "i1": 1, "i5": 5, "i6": 6, "i8": 8, "i16": 16, "i24": 24, "i32": 32,
    "i33": 33, "i40": 40, "i48": 48, "i56": 56, "i64": 64, "i128": 128,
    "ref": 64,
}


def _int_enc_to_int(solver, term):
    """Coerce a converted int-encoding operand to Int sort (Bool → 0/1)."""
    if term.getSort().isBoolean():
        return solver.mkTerm(Kind.ITE, term, solver.mkInteger(1),
                             solver.mkInteger(0))
    return term


def _int_enc_special_term(solver, name, arg_terms):
    """Int-encoding special forms over ALREADY-CONVERTED operand terms.

    Returns a Term for the bitwise/shift intrinsics (int→bv→op→nat) and for
    $srem.iN (the prelude's C-remainder correction formula), or None when
    ``name`` is not special. Shared by convert_expr_cvc5 (AST path) and
    _parse_infix_expr._mk_function_call (string path).
    """
    if "." not in name:
        return None
    base, _, suffix = name.partition(".")
    if base == "$srem":
        if suffix not in _INT_ENC_WIDTH_SUFFIX or len(arg_terms) != 2:
            return None
        a = _int_enc_to_int(solver, arg_terms[0])
        b = _int_enc_to_int(solver, arg_terms[1])
        zero = solver.mkInteger(0)
        m = solver.mkTerm(Kind.INTS_MODULUS, a, b)
        abs_b = solver.mkTerm(
            Kind.ITE, solver.mkTerm(Kind.GEQ, b, zero), b,
            solver.mkTerm(Kind.NEG, b))
        cond = solver.mkTerm(
            Kind.AND,
            solver.mkTerm(Kind.DISTINCT, m, zero),
            solver.mkTerm(Kind.LT, a, zero))
        return solver.mkTerm(Kind.ITE, cond,
                             solver.mkTerm(Kind.SUB, m, abs_b), m)
    bv_kind = _INT_ENC_BITWISE_OPS.get(base)
    if bv_kind is None:
        return None
    width = _INT_ENC_WIDTH_SUFFIX.get(suffix)
    if width is None:
        return None
    expected = 1 if base == "$not" else 2
    if len(arg_terms) != expected:
        raise ValueError(
            f"{name} expects {expected} arg(s), got {len(arg_terms)}")
    int2bv = solver.mkOp(Kind.INT_TO_BITVECTOR, width)
    bv_args = []
    for t in arg_terms:
        t = _int_enc_to_int(solver, t)
        if t.getSort().isBitVector():
            bv_args.append(cvc5_cast_to_bv(solver, t, width))
        else:
            bv_args.append(solver.mkTerm(int2bv, t))
    bv_result = solver.mkTerm(bv_kind, *bv_args)
    return solver.mkTerm(Kind.BITVECTOR_UBV_TO_INT, bv_result)

def get_fn_map():
    """Return the appropriate function map for current encoding mode."""
    return _INT_ENC_FN_MAP if _INTEGER_ENCODING else fn_to_cvc5_op


def mk_const_like(solver, var_or_sort, value: int):
    """Create a constant matching the sort of var_or_sort.
    Works in both BV and integer encoding modes."""
    from cvc5 import Kind as _K
    sort = var_or_sort if hasattr(var_or_sort, 'isInteger') and not hasattr(var_or_sort, 'getKind') else var_or_sort.getSort()
    if sort.isInteger():
        return solver.mkInteger(str(value))
    elif sort.isBitVector():
        return solver.mkBitVector(sort.getBitVectorSize(), value)
    elif sort.isBoolean():
        return solver.mkBoolean(bool(value))
    else:
        # Fallback: integer if encoding is on, BV32 otherwise
        if _INTEGER_ENCODING:
            return solver.mkInteger(str(value))
        return solver.mkBitVector(32, value)

def set_integer_encoding(enabled: bool):
    """Set whether the program uses SMACK integer encoding (type i32 = int)."""
    global _INTEGER_ENCODING
    _INTEGER_ENCODING = enabled

# Canonical implementation lives in the cvc5-free module
# interpreter.utils.integer_encoding so the PyPy compile stage (which cannot
# import cvc5, and therefore not this module) can share the ONE detector.
# Re-exported here for the existing CPython consumers
# (tools/common.py, tools/drivers/driver.py, src/state/state_cache.py).
def convert_type_to_cvc5(solver, type_, mono_mem = True) -> Sort:
    if isinstance(type_, BooleanType):
        return solver.getBooleanSort()
    elif isinstance(type_, IntegerType):
        return solver.getIntegerSort()
    elif isinstance(type_, CustomType):
        # In SMACK integer encoding, iN and ref types are just int (no BV semantics)
        if _INTEGER_ENCODING and type_.name in ("i1", "i8", "i16", "i32", "i64", "i128", "ref"):
            return solver.getIntegerSort()
        if type_.name == "i1":
            return solver.mkBitVectorSort(1)
        elif type_.name == "i8":
            return solver.mkBitVectorSort(8)
        elif type_.name == "i16":
            return solver.mkBitVectorSort(16)
        elif type_.name == "i32":
            return solver.mkBitVectorSort(32)
        elif type_.name == "i64":
            return solver.mkBitVectorSort(64)
        elif type_.name == "i128":
            return solver.mkBitVectorSort(128)
        elif type_.name.startswith("bv") and type_.name[2:].isdigit():
            # Boogie native bitvector types: bv32, bv64, etc.
            return solver.mkBitVectorSort(int(type_.name[2:]))
        elif type_.name == "bool":
            return solver.getBooleanSort()
        elif type_.name == "ref":
            return solver.mkBitVectorSort(64)
        elif type_.name == "$mop":
            return solver.mkUninterpretedSort("mop")
    elif isinstance(type_, MapType):
        domain = [convert_type_to_cvc5(solver, t) for t in type_.domain]
        if mono_mem:
            elementSort = solver.getIntegerSort() if _INTEGER_ENCODING else solver.mkBitVectorSort(64)
        else:
            elementSort = convert_type_to_cvc5(solver, type_.range)
            

        if len(domain) == 1:
            return solver.mkArraySort(domain[0], elementSort)
        else:
            assert False
    assert False, f"unknown type {type_} {type(type_)}"

def _coerce_bool_int_literal_equality(solver, lhs, rhs):
    """Boolean-vs-int-literal (dis)equality from printed wp text.

    The canonical printer flattens ``ite(c, 1, 0) == 1`` to ``(c) == (1)``,
    so re-parsing freeze-payload/agent-boundary text meets a Boolean term
    compared to an integer literal — ill-sorted for cvc5 EQUAL (observed:
    every synth candidate for c2i_126 pc 2906 died with target_parse_error
    'not a bit-vector sort' and the obligation was abandoned every
    iteration). Coerce by the elided-ite semantics: ``b == 1 -> b``,
    ``b == 0 -> not b``, any other literal -> false. Returns None when the
    shape doesn't apply (including plain bool==bool).
    """
    for a, b in ((lhs, rhs), (rhs, lhs)):
        try:
            if not a.getSort().isBoolean() or b.getSort().isBoolean():
                continue
            value = None
            if b.isBitVectorValue():
                value = int(b.getBitVectorValue(10))
            elif b.isIntegerValue():
                value = int(b.getIntegerValue())
        except Exception:
            return None
        if value is None:
            return None
        if value == 1:
            return a
        if value == 0:
            return solver.mkTerm(Kind.NOT, a)
        return solver.mkBoolean(False)
    return None


def convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr, mono_mem: bool) -> Term:
    if isinstance(expr, StorageIdentifier) or isinstance(expr, ProcedureIdentifier):
        result = state_cache.cvc5_var(expr.name)
        if result is None:
            raise ValueError(f"cvc5_var returned None for '{expr.name}'")
        return result
    elif isinstance(expr, BinaryExpression):
        if expr.op in cvc5_fn_map:
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
            fn_entry = cvc5_fn_map[expr.op]
            cvc5_op = fn_entry[0]
            if cvc5_op in (Kind.EQUAL, Kind.DISTINCT):
                # Printed-wp round-trip: ``ite(c,1,0) == 1`` is rendered as
                # ``(c) == (1)`` — coerce bool-vs-int-literal before the
                # bv sign-extend below chokes on the Boolean side.
                coerced = _coerce_bool_int_literal_equality(solver, lhs, rhs)
                if coerced is not None:
                    return (coerced if cvc5_op == Kind.EQUAL
                            else solver.mkTerm(Kind.NOT, coerced))
            if not _INTEGER_ENCODING and rhs.isBitVectorValue():
                rhs = sign_extend(solver, rhs, lhs.getSort().getBitVectorSize())

            _, _, op_type, out_type = fn_entry[:4]
            if op_type == bool:
                lhs = cvc5_cast_to_bool(solver, lhs)
                rhs = cvc5_cast_to_bool(solver, rhs)

            lhs, rhs = assign_fix_type(solver, lhs, rhs)
            if not _INTEGER_ENCODING and lhs.getSort().isBitVector() and rhs.getSort().isBitVector():
                cvc5_op = {
                    Kind.ADD: Kind.BITVECTOR_ADD,
                    Kind.SUB: Kind.BITVECTOR_SUB,
                    Kind.MULT: Kind.BITVECTOR_MULT,
                    Kind.DIVISION: Kind.BITVECTOR_UDIV,
                    Kind.INTS_MODULUS: Kind.BITVECTOR_UREM,
                    Kind.LT: Kind.BITVECTOR_ULT,
                    Kind.LEQ: Kind.BITVECTOR_ULE,
                    Kind.GT: Kind.BITVECTOR_UGT,
                    Kind.GEQ: Kind.BITVECTOR_UGE,
                }.get(cvc5_op, cvc5_op)
            return solver.mkTerm(cvc5_op, lhs, rhs)
        elif expr.op == "==":
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
            coerced = _coerce_bool_int_literal_equality(solver, lhs, rhs)
            if coerced is not None:
                return coerced
            lhs, rhs = assign_fix_type(solver, lhs, rhs)
            return solver.mkTerm(Kind.EQUAL, lhs, rhs)
        elif expr.op == "!=":
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
            coerced = _coerce_bool_int_literal_equality(solver, lhs, rhs)
            if coerced is not None:
                return solver.mkTerm(Kind.NOT, coerced)
            lhs, rhs = assign_fix_type(solver, lhs, rhs)
            return solver.mkTerm(Kind.DISTINCT, lhs, rhs)
        else:
            assert False, f"unknown binary operator {expr.op}"
    elif isinstance(expr, MapSelect):
        map_term = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.map, mono_mem)
        cvc5_indexes = []
        for idx in expr.indexes:
            cvc5_indexes.append(convert_expr_cvc5(cvc5_fn_map, state_cache, solver, idx, mono_mem))
        assert len(cvc5_indexes) == 1
        index = cvc5_indexes[0]
        if index.getSort() != map_term.getSort().getArrayIndexSort():
            index_value = int(index.getBitVectorValue(10))
            index = solver.mkBitVector(map_term.getSort().getArrayIndexSort().getBitVectorSize(), index_value)
        return solver.mkTerm(Kind.SELECT, map_term, index)
    elif isinstance(expr, QuantifiedExpression):
        IndentLogger.debug(f"WARNING: quantified expression not supported: {expr}")
        quantified_vars = []
        # Cache the bound mkVar by name and reuse it on every subsequent
        # forall that binds the same name.  cvc5's FORALL creates a scope
        # per binder, so sharing one mkVar across sibling/nested foralls
        # is semantically fine — and it guarantees the VL entry and every
        # body reference resolve to the SAME cvc5 Term, which is what
        # ``assertFormula`` requires (mismatched mkVars with the same
        # name are rejected as "free variables" deep in cvc5's C++).
        #
        # Assumption: a name used as a forall binder in SMACK-generated
        # Boogie is never also used as a free constant with the same
        # symbol.  If a same-named CONSTANT is in the cache we promote
        # it to VARIABLE — this is the bound-var-wins design documented
        # in tests/test_qfall_cache_pollution.py.
        for v in expr.variables:
            assert len(v.names) == 1
            name = v.names[0]
            cached = state_cache.cached_id_to_cvc5.get(name)
            if cached is not None and cached.getKind() == Kind.VARIABLE:
                var_cvc5 = cached
            else:
                var_cvc5 = solver.mkVar(convert_type_to_cvc5(solver, v.type), name)
                state_cache.cached_id_to_cvc5[name] = var_cvc5
            quantified_vars.append(var_cvc5)
        quantifiedVariables = solver.mkTerm(Kind.VARIABLE_LIST, *quantified_vars)
        q_expr = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.expression, mono_mem)
        forall_expr = solver.mkTerm(Kind.FORALL, quantifiedVariables, q_expr)
        return forall_expr
    elif isinstance(expr, IfExpression):
        cond = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.condition, mono_mem)
        then_branch = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.then, mono_mem)
        else_branch = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.else_, mono_mem)
        then_branch, else_branch = assign_fix_type(solver, then_branch, else_branch)
        return solver.mkTerm(Kind.ITE, cond, then_branch, else_branch)
    elif isinstance(expr, FunctionApplication):
        if _INTEGER_ENCODING and "." in expr.function.name:
            _base = expr.function.name.partition(".")[0]
            if _base == "$srem" or _base in _INT_ENC_BITWISE_OPS:
                arg_terms = [
                    convert_expr_cvc5(cvc5_fn_map, state_cache, solver, arg,
                                      mono_mem)
                    for arg in expr.arguments
                ]
                special = _int_enc_special_term(
                    solver, expr.function.name, arg_terms)
                if special is not None:
                    return special
        if expr.function.name in cvc5_fn_map:
            fn_entry = cvc5_fn_map[expr.function.name]
            cvc5_op = fn_entry[0]
            arg_exprs = [convert_expr_cvc5(cvc5_fn_map, state_cache, solver, arg, mono_mem) for arg in expr.arguments]

            _, _, op_bit_width, output_type = fn_entry[:4]
            if _INTEGER_ENCODING and op_bit_width is None:
                # Integer encoding: no BV casts needed.
                # Convert boolean args to integer (0/1) for arithmetic ops.
                for idx, a in enumerate(arg_exprs):
                    if a.getSort().isBoolean():
                        arg_exprs[idx] = solver.mkTerm(
                            Kind.ITE, a, solver.mkInteger(1), solver.mkInteger(0))
                if output_type == bool:
                    ret = solver.mkTerm(cvc5_op, *arg_exprs)
                    # Ensure boolean sort
                    if not ret.getSort().isBoolean():
                        ret = cvc5_cast_to_bool(solver, ret)
                elif output_type == "int_cmp":
                    # Comparison returning integer 0/1: ITE(cond, 1, 0)
                    cond = solver.mkTerm(cvc5_op, *arg_exprs)
                    if not cond.getSort().isBoolean():
                        cond = cvc5_cast_to_bool(solver, cond)
                    ret = solver.mkTerm(Kind.ITE, cond,
                                        solver.mkInteger(1), solver.mkInteger(0))
                elif cvc5_op is None:
                    assert len(arg_exprs) == 1
                    ret = arg_exprs[0]
                else:
                    try:
                        ret = solver.mkTerm(cvc5_op, *arg_exprs)
                    except RuntimeError as e:
                        import logging as _log
                        _log.warning("[INT-ENC] mkTerm failed: op=%s func=%s args=%s sorts=%s err=%s",
                                     cvc5_op, expr.function.name, arg_exprs,
                                     [a.getSort() for a in arg_exprs], e)
                        raise
                return ret
            # BV encoding path (original)
            arg_exprs = [cvc5_cast_to_bv(solver, arg, op_bit_width) for arg in arg_exprs]
            if output_type == bool:
                ret = cvc5_cast_to_bool(solver, solver.mkTerm(cvc5_op, *arg_exprs))
            else:
                if cvc5_op is None:
                    # This is to handle bitcast, but is there a better way?
                    assert len(arg_exprs) == 1
                    ret = arg_exprs[0]
                else:
                    ret = cvc5_cast_to_bv(solver, solver.mkTerm(cvc5_op, *arg_exprs), output_type)
            return ret
        elif expr.function.name in ["$load.i8", "$load.i16", "$load.i32", "$load.i64", "$load.ref"]:
            load_bit_width = expr.function.name.split(".")[-1]
            load_bit_width = boogie_type_bitwidth[load_bit_width]
            memory_map = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.arguments[0], mono_mem)
            addr = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.arguments[1], mono_mem)

            if not mono_mem:
                element_width = memory_map.getSort().getArrayElementSort().getBitVectorSize()
                assert load_bit_width >= element_width and load_bit_width % element_width == 0, f"Bit width {load_bit_width} is not a multiple of element width {element_width}"
                num_elements = load_bit_width // element_width
                load_terms = []
                for i in range(num_elements):
                    addr_term = solver.mkTerm(Kind.BITVECTOR_ADD, addr, solver.mkBitVector(addr.getSort().getBitVectorSize(), i))
                    load_terms.append(solver.mkTerm(Kind.SELECT, memory_map, addr_term))

                if len(load_terms) == 1:
                    ret_load_term = load_terms[0]
                else:
                    ret_load_term = solver.mkTerm(Kind.BITVECTOR_CONCAT, *load_terms)
            else:
                ret_load_term = solver.mkTerm(Kind.SELECT, memory_map, addr)
                return ret_load_term
                # if ret_load_term.getSort() != solver.mkBitVectorSort(load_bit_width):
                #     ret_load_term = cvc5_cast_to_bv(solver, ret_load_term, load_bit_width)
            return ret_load_term
        elif expr.function.name in ["$store.i8", "$store.i16", "$store.i32", "$store.i64", "$store.ref"]:
            store_bit_width = expr.function.name.split(".")[-1]
            store_bit_width = boogie_type_bitwidth[store_bit_width]
            
            # Memory is global
            memory_map = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.arguments[0], mono_mem)
            addr = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.arguments[1], mono_mem)
            val = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.arguments[2], mono_mem)

            if not mono_mem:
                if val.getSort().getBitVectorSize() != store_bit_width:
                    val = cvc5_cast_to_bv(solver, val, store_bit_width)

                element_width = memory_map.getSort().getArrayElementSort().getBitVectorSize()
                assert store_bit_width >= element_width and store_bit_width % element_width == 0, f"Bit width {store_bit_width} is not a multiple of element width {element_width}"
                num_elements = store_bit_width // element_width
            
                iter_memory_map = memory_map
                for i in range(num_elements):
                    IndentLogger.debug(f"element_width * (i + 1) - 1, element_width * i: {element_width * (i + 1) - 1}, {element_width * i}")
                    extract_op = solver.mkOp(Kind.BITVECTOR_EXTRACT, element_width * (i + 1) - 1, element_width * i)
                    val_term = solver.mkTerm(extract_op, val)
                    addr_term = solver.mkTerm(Kind.BITVECTOR_ADD, addr, solver.mkBitVector(addr.getSort().getBitVectorSize(), i))
                    iter_memory_map = solver.mkTerm(Kind.STORE, iter_memory_map, addr_term, val_term)
                ret_memory_map = iter_memory_map
            else:
                if memory_map.getSort().getArrayElementSort() != val.getSort():
                    val = cvc5_cast_to_bv(solver, val, memory_map.getSort().getArrayElementSort().getBitVectorSize())
                ret_memory_map = solver.mkTerm(Kind.STORE, memory_map, addr, val)
            return ret_memory_map
        elif expr.function.name in ["$isExternal"]:
            return solver.mkBoolean(True)
        else:
            assert False, f"unknown function application: {expr} {expr.function.name}" 
    elif isinstance(expr, UnaryExpression):
        unary_expr = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.expression, mono_mem)
        if isinstance(expr, OldExpression):
            assert False
        elif isinstance(expr, LogicalNegation):
            # If inner is BV1, convert to Bool before NOT. SMACK
            # comparison helpers (e.g. ``$slt.i32``) return ``i1`` per
            # the Boogie prelude, so the inner term lands here as BV1.
            if unary_expr.getSort().isBitVector() and unary_expr.getSort().getBitVectorSize() == 1:
                unary_expr = solver.mkTerm(Kind.EQUAL, unary_expr, solver.mkBitVector(1, 1))
            return solver.mkTerm(Kind.NOT, unary_expr)
        elif isinstance(expr, ArithmeticNegation):
            if _INTEGER_ENCODING or unary_expr.getSort().isInteger():
                return solver.mkTerm(Kind.NEG, unary_expr)
            else:
                return solver.mkTerm(Kind.BITVECTOR_NEG, unary_expr)
        else:
            assert False, f"unsupported unary expression {type(expr).__name__}"
    elif isinstance(expr, IntegerLiteral):
        if _INTEGER_ENCODING:
            return solver.mkInteger(str(expr.value))
        assert expr.value >= 0
        if expr.value == 0:
            return solver.mkBitVector(32, 0)

        bits_needed = math.ceil(math.log2(expr.value))
        if bits_needed <= 32:
            bits_needed = 32
        elif bits_needed <= 64:
            bits_needed = 64
        else:
            assert False, f"Requires more than 64 bits: {expr.value}"

        return solver.mkBitVector(bits_needed, expr.value)
    elif isinstance(expr, BooleanLiteral):
        if expr.value:
            return solver.mkBoolean(True)
        else:
            return solver.mkBoolean(False)
    elif isinstance(expr, BitvectorLiteral):
        # ``Nbv<width>`` literal — produce a typed BV constant directly so
        # callers (agent invariants like ``$i5 == 1bv1``) don't depend on
        # the surrounding op to widen an Integer.
        if _INTEGER_ENCODING:
            return solver.mkInteger(str(expr.value))
        return solver.mkBitVector(int(expr.base), int(expr.value))
    elif isinstance(expr, Term):
        return expr
    else:
        assert False, f"unknown expression {expr} {type(expr)}"

# The legacy numeric kind/sort maps (KIND_TO_NUM/NUM_TO_SORT/...) were
# DELETED with the legacy hollow scheme; the v2 wire (cvc5_serde)
# carries kinds/sorts natively.


def deserialize_predicate_helper(state_cache, predicate):
    if isinstance(predicate.predicate, _SerializedCvc5TermV2):
        predicate.predicate = _deserialize_cvc5_term(
            state_cache, predicate.predicate)
        # Clear cached hash AND cached string so they recompute from
        # the new live cvc5 Term — otherwise ``str(predicate)`` keeps
        # returning the hollow form even after rehydration, which
        # breaks anything keying on the stringified predicate.
        predicate._cached_hash = None
        predicate._cached_str = None
        predicate._cached_variable_terms = None

# The old in-module term wire was deleted. ``cvc5_serde`` is the sole wire
# owner; this module only consumes its private hydration primitives.

def make_to_cvc5(fn_map, state_cache):
    """
    Return a function:  expr → cvc5.Term

    The heavy objects (maps, solver) are captured once, so every call
    needs only the expression and two optional flags.
    """
    def to_cvc5(expr, mono_mem: bool = True,
                quantifier_handling: bool = False):
        ret = convert_expr_cvc5(fn_map, state_cache, state_cache.solver, expr, mono_mem)
        if ret.getKind() == Kind.FORALL and not quantifier_handling:
            ret = state_cache.solver.mkTrue()
        return ret

    return to_cvc5

def deserialize_state_key(state_cache, state_key):
    from src.state.persistence import deserialize_obligation
    from src.state.proof_obligation import ProofObligation

    obligation = deserialize_obligation(state_key)
    predicate = obligation.predicate
    predicate.predicate = _deserialize_cvc5_term(
        state_cache, predicate.predicate)
    # ``state_key`` already authenticates one canonical predicate wire.
    # Materialization changes only its representation from solver-independent
    # serde to a live cvc5 Term; running the semantic canonicalizer again can
    # move a non-fixpoint legacy/current-run identity while leaving the native
    # bytes unchanged.  The ordinary constructor recognizes the retained
    # ``_canonical_serialized`` authority and makes an identity-only copy.
    return ProofObligation(obligation.pc, predicate)


def deserialize_predicate_pickle(state_cache, raw: bytes):
    """Rehydrate a pickled ``Predicate`` (serde form) against ``state_cache``.

    The native-term counterpart of parsing a printed predicate: the anvil gate
    replay loads the freeze payload's live obligation/transformed terms with
    this instead of reparsing text — text round-trips build different ASTs
    (left-nested binaries, flattened bool==int) and every divergence splits
    the gate from the live engine.
    """
    predicate = pickle.loads(raw)
    predicate.predicate = _deserialize_cvc5_term(
        state_cache, predicate.predicate)
    return predicate


def to_boogie(cvc5_term: Term, *, signed_context: bool = False):
    """Render a cvc5 Term as a Boogie AST node.

    ``signed_context`` is propagated down by the CALLER when the
    enclosing operator is a signed bitvector comparison / arithmetic
    (e.g. BITVECTOR_SGE, BITVECTOR_SLT, BITVECTOR_SGT, BITVECTOR_SLE).
    Under a signed context, BV constants whose high bit is set are
    printed as their signed two's-complement value so e.g. ``-1`` shows
    as ``-1`` rather than ``4294967295``.

    Under an unsigned context (or when the enclosing op doesn't care —
    arithmetic like BITVECTOR_ADD/BITVECTOR_SUB where signed/unsigned
    share the same bit pattern), BV constants print as unsigned
    integers.
    """
    if cvc5_term.getKind() == Kind.CONSTANT:
        return StorageIdentifier(name=cvc5_term.getSymbol())
    elif cvc5_term.getKind() == Kind.CONST_BOOLEAN:
        value = bool(cvc5_term.getBooleanValue())
        return BooleanLiteral(value)
    elif cvc5_term.getKind() == Kind.CONST_BITVECTOR:
        assert cvc5_term.isBitVectorValue(), f"Constant is not a bitvector: {cvc5_term}"
        raw = int(cvc5_term.getBitVectorValue(), 2)
        if signed_context:
            width = cvc5_term.getSort().getBitVectorSize()
            if width > 0 and raw >= (1 << (width - 1)):
                raw -= (1 << width)
        return IntegerLiteral(raw)
    elif cvc5_term.getKind() == Kind.CONST_INTEGER:
        return IntegerLiteral(int(cvc5_term.getIntegerValue()))
    # Integer arithmetic ops (from integer encoding mode)
    elif cvc5_term.getKind() == Kind.ADD:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="+", rhs=rhs)
    elif cvc5_term.getKind() == Kind.SUB:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="-", rhs=rhs)
    elif cvc5_term.getKind() == Kind.MULT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="*", rhs=rhs)
    elif cvc5_term.getKind() == Kind.GEQ:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.LEQ:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.LT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<", rhs=rhs)
    elif cvc5_term.getKind() == Kind.GT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">", rhs=rhs)
    elif cvc5_term.getKind() == Kind.NEG:
        child = to_boogie(cvc5_term[0])
        from interpreter.parser.expression import ArithmeticNegation
        return ArithmeticNegation(expression=child)
    elif cvc5_term.getKind() == Kind.SELECT:
        memory_map = to_boogie(cvc5_term[0])
        load_addr = to_boogie(cvc5_term[1])
        return MapSelect(map=memory_map, indexes=[load_addr])
    elif cvc5_term.getKind() == Kind.STORE:
        memory_map = to_boogie(cvc5_term[0])
        store_addr = to_boogie(cvc5_term[1])
        store_val = to_boogie(cvc5_term[2])
        # Boogie syntax: M[addr := val]
        from interpreter.parser.expression import MapUpdate
        return MapUpdate(map=memory_map, indexes=[store_addr], value=store_val)
    elif cvc5_term.getKind() == Kind.EQUAL:
        lhs_term = cvc5_term[0]
        rhs_term = cvc5_term[1]
        lhs = to_boogie(lhs_term)
        rhs = to_boogie(rhs_term)
        # In integer encoding, fix bool/int mismatches in EQUAL:
        # - bool == 1 → bool, bool == 0 → !bool
        # - int == bool → int == (if bool then 1 else 0)
        if _INTEGER_ENCODING:
            def _is_bool_expr(e):
                return isinstance(e, BinaryExpression) and e.op in ("<", ">", "<=", ">=", "==", "!=") or isinstance(e, LogicalNegation) or isinstance(e, BooleanLiteral)
            for (a, b) in [(lhs, rhs), (rhs, lhs)]:
                if isinstance(b, IntegerLiteral) and _is_bool_expr(a):
                    if b.value == 1:
                        return a
                    elif b.value == 0:
                        return LogicalNegation(expression=a)
            # If one side is bool-typed (comparison) and the other is int-typed (variable),
            # wrap the bool side in if-then-else to produce int
            if _is_bool_expr(lhs) and not _is_bool_expr(rhs):
                lhs = IfExpression(condition=lhs, then=IntegerLiteral(value=1), else_=IntegerLiteral(value=0))
            elif _is_bool_expr(rhs) and not _is_bool_expr(lhs):
                rhs = IfExpression(condition=rhs, then=IntegerLiteral(value=1), else_=IntegerLiteral(value=0))
        return BinaryExpression(lhs=lhs, op="==", rhs=rhs)
    elif cvc5_term.getKind() == Kind.AND:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="&&", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_ADD:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="+", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SUB:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="-", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_MULT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="*", rhs=rhs)
    elif cvc5_term.getKind() == Kind.OR:
        children = [to_boogie(cvc5_term[i]) for i in range(cvc5_term.getNumChildren())]
        curr_binary_expr = BinaryExpression(lhs=children[0], op="||", rhs=children[1])
        for child in children[2:]:
            curr_binary_expr = BinaryExpression(lhs=curr_binary_expr, op="||", rhs=child)
        return curr_binary_expr
    elif cvc5_term.getKind() == Kind.DISTINCT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="!=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_EXTRACT:
        extract_expr = to_boogie(cvc5_term[0])
        return extract_expr
    elif cvc5_term.getKind() == Kind.BITVECTOR_SIGN_EXTEND:
        sign_extend_expr = to_boogie(cvc5_term[0])
        return sign_extend_expr
    elif cvc5_term.getKind() == Kind.BITVECTOR_ZERO_EXTEND:
        zero_extend_expr = to_boogie(cvc5_term[0])
        return zero_extend_expr
    elif cvc5_term.getKind() == Kind.ITE:
        condition = to_boogie(cvc5_term[0])
        then_expr = to_boogie(cvc5_term[1])
        else_expr = to_boogie(cvc5_term[2])
        combined = {"condition": condition, "then": then_expr, "else_": else_expr}
        return IfExpression(**combined)
    elif cvc5_term.getKind() == Kind.NOT:
        child = to_boogie(cvc5_term[0])
        # In Boogie, ! only works on bool. If the child might return i1
        # (function calls like $slt.i32, or any non-bool expression),
        # use == 0 instead of !. BinaryExpression with <,>,<=,>= returns
        # bool in Boogie so ! is safe for those.
        if isinstance(child, BinaryExpression) and child.op in ("<", ">", "<=", ">=", "==", "!="):
            return LogicalNegation(expression=child)
        # For everything else (function calls, identifiers), use == 0
        return BinaryExpression(lhs=child, op="==", rhs=IntegerLiteral(value=0))
    elif cvc5_term.getKind() == Kind.BITVECTOR_SGT:
        lhs = to_boogie(cvc5_term[0], signed_context=True)
        rhs = to_boogie(cvc5_term[1], signed_context=True)
        return BinaryExpression(lhs=lhs, op=">", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SGE:
        lhs = to_boogie(cvc5_term[0], signed_context=True)
        rhs = to_boogie(cvc5_term[1], signed_context=True)
        return BinaryExpression(lhs=lhs, op=">=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_ULT:
        # Unsigned comparison — BV constants print as unsigned (default).
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SLT:
        lhs = to_boogie(cvc5_term[0], signed_context=True)
        rhs = to_boogie(cvc5_term[1], signed_context=True)
        return BinaryExpression(lhs=lhs, op="<", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SLE:
        lhs = to_boogie(cvc5_term[0], signed_context=True)
        rhs = to_boogie(cvc5_term[1], signed_context=True)
        return BinaryExpression(lhs=lhs, op="<=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_ULE:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_UGE:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_UGT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">", rhs=rhs)
    elif cvc5_term.getKind() == Kind.SET_MEMBER:
        var = to_boogie(cvc5_term[0])
        vals = to_boogie(cvc5_term[1])
        or_boogie = []
        for v in vals:
            or_boogie.append(BinaryExpression(lhs=var, op="==", rhs=v))
        or_expr = or_boogie[0]
        for boogie_expr in or_boogie[1:]:
            or_expr = BinaryExpression(lhs=or_expr, op="||", rhs=boogie_expr)
        return or_expr
    elif cvc5_term.getKind() == Kind.SET_INSERT:
        vals = [to_boogie(cvc5_term[i]) for i in range(cvc5_term.getNumChildren() - 1)]
        set_expr = to_boogie(cvc5_term[cvc5_term.getNumChildren() - 1])
        return set(vals)
    elif cvc5_term.getKind() == Kind.SET_EMPTY:
        return set()
    else:
        assert False, f"Unknown term: {cvc5_term} {cvc5_term.getKind()}"


def disjunct(solver, predicates):
    from src.solver.predicate import Predicate
    if len(predicates) == 0:
        return Predicate(solver.mkTrue())
    elif len(predicates) == 1:
        return predicates[0]
    else:
        return Predicate(solver.mkTerm(Kind.OR, *[predicate.predicate for predicate in predicates]))

def conjunct(solver, predicates):
    from src.solver.predicate import Predicate
    if len(predicates) == 0:
        return Predicate(solver.mkTrue())
    elif len(predicates) == 1:
        return predicates[0]
    else:
        return Predicate(solver.mkTerm(Kind.AND, *[predicate.predicate for predicate in predicates]))

import re

def _coerce_bv_sorts(solver, a, b):
    """Ensure two bitvector terms have matching widths via zero-extension."""
    from cvc5 import Kind
    try:
        as_, bs_ = a.getSort(), b.getSort()
        if not as_.isBitVector() or not bs_.isBitVector():
            return a, b
        aw, bw = as_.getBitVectorSize(), bs_.getBitVectorSize()
        if aw < bw:
            op = solver.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, bw - aw)
            a = solver.solver.mkTerm(op, a)
        elif bw < aw:
            op = solver.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, aw - bw)
            b = solver.solver.mkTerm(op, b)
    except Exception:
        pass
    return a, b


def parse_constraint_tuple(tuple_str):
    """
    Parses constraint tuples including assignments, shadow checks, disjunctions,
    array selects, array store operations, and array-constant comparisons.
    """
    
    # 1. Extract ID and the "Body"
    outer_pattern = r"^\s*\(\s*(\d+)\s*,\s*(.*)\)\s*$"
    match = re.match(outer_pattern, tuple_str.strip())
    
    if not match:
        raise ValueError(f"Input string format is invalid: {tuple_str}")
    
    node_id = int(match.group(1))
    body = match.group(2).strip()

    result = {
        "id": node_id,
        "type": None,
        "variable": None,
        "constant": None,
        "shadow_variable": None,
        "index": None,
        "allowed_values": None,
        "lhs_array": None,
        "rhs_array": None,
        "index_variable": None,
        "value_variable": None,
        "array": None
    }

    # --- TYPE 4: Disjunction of constants ---
    if r"\/" in body:
        parts = [p.strip() for p in body.split(r"\/")]
        disjuncts = []
        for part in parts:
            # This fast path is only for simple finite-domain predicates
            # such as ``($i0) == (1) \/ ($i0) == (2)``.  Mixed Boolean
            # formulas containing a top-level disjunction must fall through
            # to the general infix parser; otherwise a left conjunction can
            # be misread as the "variable" and later crash EQ_CONST.
            match_const = re.fullmatch(
                r"\(?\s*([\$\w.]+(?:\[[^\]]+\])?)\s*\)?"
                r"\s*==\s*"
                r"\(?\s*(-?\d+)\s*\)?",
                part,
            )
            if not match_const:
                disjuncts = []
                break
            disjuncts.append((match_const.group(1).strip(),
                              int(match_const.group(2))))

        if disjuncts and len({var for var, _c in disjuncts}) == 1:
            result["variable"] = disjuncts[0][0]
            result["type"] = "disjunction_of_constants"
            result["allowed_values"] = [c for _var, c in disjuncts]
            return result

    # --- EQUALITIES ---
    eq_pattern = r"^\((.*?)\)\s*==\s*\((.*?)\)$"
    eq_match = re.match(eq_pattern, body)
    
    if eq_match:
        lhs = eq_match.group(1).strip()
        rhs = eq_match.group(2).strip()

        # --- TYPE 5: Array Store Equality (FIXED) ---
        clean_rhs = rhs
        while clean_rhs.startswith("(") and clean_rhs.endswith(")"):
            clean_rhs = clean_rhs[1:-1].strip()

        store_pattern = r"^(.*?)\[(.*?)\]\s*=\s*(.*)$"
        store_match = re.match(store_pattern, clean_rhs)

        if store_match:
            result["type"] = "array_store_equality"
            result["lhs_array"] = lhs
            result["rhs_array"] = store_match.group(1).strip()
            result["index_variable"] = store_match.group(2).strip()
            result["value_variable"] = store_match.group(3).strip()
            return result

        # --- TYPE 6: Variable Shadow Array Equality (Select) ---
        select_pattern = r"^(.*?)\[(.*?)\]$"
        select_match = re.match(select_pattern, rhs)

        if select_match and "[" not in lhs:
            result["type"] = "variable_shadow_array_equality"
            result["variable"] = lhs
            result["array"] = select_match.group(1).strip()
            result["index_variable"] = select_match.group(2).strip()
            return result

        # --- TYPE 3: Shadow Array Check (Constant Index) ---
        array_pattern = r"^(.+)\[(\d+)\]$"
        lhs_arr = re.match(array_pattern, lhs)
        rhs_arr = re.match(array_pattern, rhs)

        if lhs_arr and rhs_arr and "shadow" in rhs:
            result["type"] = "shadow_array"
            result["variable"] = lhs_arr.group(1)
            result["shadow_variable"] = rhs_arr.group(1)
            result["index"] = int(lhs_arr.group(2))
            return result

        # --- TYPE 3b: Shadow Array Check (Variable Index) ---
        # e.g. ($M.18[$p0]) == ($M.18.shadow[$p0.shadow])
        var_select_pattern = r"^(.+)\[([^\]]+)\]$"
        lhs_var_arr = re.match(var_select_pattern, lhs)
        rhs_var_arr = re.match(var_select_pattern, rhs)

        if lhs_var_arr and rhs_var_arr and "shadow" in rhs:
            result["type"] = "shadow_array_var_index"
            result["variable"] = lhs_var_arr.group(1)
            result["shadow_variable"] = rhs_var_arr.group(1)
            result["index_variable"] = lhs_var_arr.group(2)
            return result

        # --- TYPE 7: Array Select Constant Equality ---
        # Intercepts cases like: ($M.18[1024]) == (525)
        # lhs = "$M.18[1024]", rhs = "525"
        array_const_match = re.match(r"^(.*?)\[(\d+)\]$", lhs)
        is_digit_rhs = rhs.isdigit() or (rhs.startswith('-') and rhs[1:].isdigit())

        if array_const_match and is_digit_rhs:
            result["type"] = "array_select_constant_equality"
            result["array"] = array_const_match.group(1).strip()
            result["index"] = int(array_const_match.group(2))
            result["constant"] = int(rhs)
            return result

        # --- TYPE 7b: Array Select Variable Index Constant Equality ---
        # e.g. ($M.18[$p0]) == (42)
        array_var_const_match = re.match(var_select_pattern, lhs)
        if array_var_const_match and is_digit_rhs:
            result["type"] = "array_select_var_constant_equality"
            result["array"] = array_var_const_match.group(1).strip()
            result["index_variable"] = array_var_const_match.group(2).strip()
            result["constant"] = int(rhs)
            return result

        # --- TYPE 2: Shadow Variable Check ---
        # RHS must be a simple variable name (e.g. $i4730.shadow), not a
        # compound expression like "$i4729.shadow < $i4727.shadow"
        if "shadow" in rhs and "[" not in lhs and re.match(r'^[\$\w.]+$', rhs):
            result["type"] = "shadow_variable"
            result["variable"] = lhs
            result["shadow_variable"] = rhs
            return result

        # --- TYPE 1: Constant Comparison ---
        # Generic fallback for standard variables (e.g. ($i6640) == (18))
        if is_digit_rhs and "[" not in lhs and re.match(r'^[\$\w.]+$', lhs):
            result["type"] = "constant_comparison"
            result["variable"] = lhs
            result["constant"] = int(rhs)
            return result

    result["type"] = "unknown"
    result["raw_body"] = body
    return result

def _parse_infix_expr(s, state_cache):
    """Parse an infix expression (from pretty_print_term output) into a cvc5 Term.

    Handles variables, integers, array select, STORE, and binary operators.
    All arithmetic is bitvector (64-bit default, or inferred from variable sort).
    """
    from cvc5 import Kind
    import re as _re

    # Reverse operator map: string op → cvc5 Kind (prefer bitvector)
    _BIN_OPS = {
        '+': Kind.BITVECTOR_ADD, '-': Kind.BITVECTOR_SUB,
        '*': Kind.BITVECTOR_MULT, '/': Kind.BITVECTOR_UDIV,
        '%': Kind.BITVECTOR_SREM,
        '>>': Kind.BITVECTOR_LSHR, '<<': Kind.BITVECTOR_SHL,
        '<': Kind.BITVECTOR_ULT, '>': Kind.BITVECTOR_UGT,
        '<=': Kind.BITVECTOR_ULE, '>=': Kind.BITVECTOR_UGE,
        '&': Kind.BITVECTOR_AND, '^': Kind.BITVECTOR_XOR,
        '||': Kind.BITVECTOR_OR,
        '==': Kind.EQUAL, '!=': Kind.DISTINCT,
    }

    # Logical (Bool-theory) operators.  Agent-authored invariants
    # commonly use ``||``, ``&&``, and Boogie's ``==>``; Swoosh's
    # term renderer emits ``\/``, ``/\``, and ``=>``.  Accept both
    # spellings so rendered proof targets round-trip through this
    # parser instead of treating ``/`` as BV division.
    _BOOL_OPS = {
        '||': Kind.OR,
        '\\/': Kind.OR,
        '&&': Kind.AND,
        '/\\': Kind.AND,
        '==>': Kind.IMPLIES,
        '=>': Kind.IMPLIES,
    }

    # Operator precedence (lower = binds tighter, but all >= 0 so the
    # ``min_prec=0`` parser entry catches every op).  Logical
    # connectives are the loosest so ``a < b || c <= d`` parses as
    # ``(a < b) || (c <= d)``; ``==>`` is looser than ``||`` / ``&&``.
    _PREC = {
        '==>': 0, '=>': 0,
        '||': 1, '\\/': 1, '&&': 1, '/\\': 1,
        '==': 2, '!=': 2,
        '<': 3, '>': 3, '<=': 3, '>=': 3,
        '+': 4, '-': 4,
        '*': 5, '/': 5, '%': 5,
        '>>': 6, '<<': 6,
        '&': 7, '^': 7,
    }

    solver = state_cache.solver
    tokens = []
    i = 0
    s = s.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] == '$' or s[i].isalpha() or s[i] == '_':
            # Variable or STORE keyword.  SMACK-emitted names embed
            # additional ``$`` segments (e.g.
            # ``$free_105_inline$__VERIFIER_nondet_int$0$$i0`` and
            # ``inline$foo$0$$i0``), so the tokenizer must accept ``$``
            # inside identifiers and also preserve non-$ prefixes.
            # Without this, the parser sees truncated names like
            # ``$foo$0$$i0`` and reports "Unknown variable".
            j = i
            if s[i] == '$':
                j += 1
                while j < len(s) and (s[j].isalnum() or s[j] in '._$'):
                    j += 1
            else:
                while j < len(s) and (s[j].isalnum() or s[j] in '._$'):
                    j += 1
            tokens.append(('ID', s[i:j]))
            i = j
        elif s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(('NUM', s[i:j]))
            i = j
        elif s[i:i+3] in ('==>',):
            tokens.append(('OP', s[i:i+3]))
            i += 3
        elif s[i:i+2] in (
            '>>', '<<', '<=', '>=', '==', '!=',
            '||', '\\/', '&&', '/\\', '=>',
        ):
            tokens.append(('OP', s[i:i+2]))
            i += 2
        elif s[i] in '+-*/%<>&^~!':
            tokens.append(('OP', s[i]))
            i += 1
        elif s[i] in '()[],:?':
            tokens.append((s[i], s[i]))
            i += 1
        else:
            i += 1  # skip unknown chars

    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else (None, None)

    def advance():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(typ):
        t = advance()
        assert t[0] == typ, f"Expected {typ}, got {t} at pos {pos[0]-1}"
        return t

    # Integer-sort fallback map: when both operands are Int (not BV) the
    # cvc5 BITVECTOR_* operators reject the term with "expecting a
    # bit-vector term".  Swap to the integer-theory equivalents.
    _INT_OPS = {
        '+': Kind.ADD, '-': Kind.SUB, '*': Kind.MULT,
        '/': Kind.INTS_DIVISION, '%': Kind.INTS_MODULUS,
        '<': Kind.LT, '>': Kind.GT, '<=': Kind.LEQ, '>=': Kind.GEQ,
        '==': Kind.EQUAL, '!=': Kind.DISTINCT,
    }

    def _both_int(a, b):
        try:
            return a.getSort().isInteger() and b.getSort().isInteger()
        except Exception:
            return False

    def _coerce_to_int(t):
        """If ``t`` is a BV literal, rebuild it as an Int preserving
        signed semantics.  Returns ``t`` unchanged when it's already
        Int or when no cheap coercion is possible.

        Signed interpretation matters: the unary-minus handler builds
        a negative BV literal as a two's-complement value (e.g. -1 at
        BV32 → 0xFFFFFFFF).  ``getBitVectorValue(10)`` returns the
        raw unsigned magnitude (4294967295), which is what cvc5 stores
        for BV arithmetic but the WRONG value when the surrounding
        expression is Int-theory — ``$u0 * -1`` must multiply by -1,
        not by 2**32 - 1.
        """
        def _fold_const_bv_expr(term):
            try:
                if not term.getSort().isBitVector():
                    return None
                width = term.getSort().getBitVectorSize()
                modulus = 1 << width
                if term.isBitVectorValue():
                    return int(term.getBitVectorValue(10)) % modulus
                kind = term.getKind()
                if kind in {
                        Kind.BITVECTOR_ADD,
                        Kind.BITVECTOR_SUB,
                        Kind.BITVECTOR_MULT}:
                    values = [
                        _fold_const_bv_expr(term[i])
                        for i in range(term.getNumChildren())
                    ]
                    if any(v is None for v in values):
                        return None
                    acc = int(values[0])
                    for value in values[1:]:
                        if kind == Kind.BITVECTOR_ADD:
                            acc += int(value)
                        elif kind == Kind.BITVECTOR_SUB:
                            acc -= int(value)
                        else:
                            acc *= int(value)
                    return acc % modulus
                if kind == Kind.BITVECTOR_NEG and term.getNumChildren() == 1:
                    value = _fold_const_bv_expr(term[0])
                    if value is None:
                        return None
                    return (-int(value)) % modulus
            except Exception:
                return None
            return None

        try:
            if t.getSort().isInteger():
                return t
            if t.getSort().isBoolean():
                return solver.mkTerm(
                    Kind.ITE, t, solver.mkInteger(1), solver.mkInteger(0))
            folded = _fold_const_bv_expr(t)
            if folded is not None:
                width = t.getSort().getBitVectorSize()
                if folded >= (1 << (width - 1)):
                    folded -= (1 << width)
                return solver.mkInteger(folded)
            if t.isBitVectorValue():
                width = t.getSort().getBitVectorSize()
                raw_val = int(t.getBitVectorValue(10))
                # Sign-extend: if the high bit is set, interpret as
                # negative two's-complement.
                if raw_val >= (1 << (width - 1)):
                    raw_val -= (1 << width)
                return solver.mkInteger(raw_val)
            if (t.getKind() == Kind.ITE
                    and t.getNumChildren() == 3
                    and t[0].getSort().isBoolean()
                    and t[1].isBitVectorValue()
                    and t[2].isBitVectorValue()):
                return solver.mkTerm(
                    Kind.ITE,
                    t[0],
                    _coerce_to_int(t[1]),
                    _coerce_to_int(t[2]))
        except Exception:
            pass
        return t

    def _either_int(a, b):
        try:
            return a.getSort().isInteger() or b.getSort().isInteger()
        except Exception:
            return False

    def _parse_call_args():
        args = []
        expect('(')
        if peek()[0] == ')':
            advance()
            return args
        while True:
            args.append(parse_expr(0))
            typ, _val = peek()
            if typ == ',':
                advance()
                continue
            expect(')')
            return args

    def _mk_function_call(name, args):
        # ``hollow_to_str`` is the canonical display owner for persisted
        # cvc5 terms and renders Boolean NOT as ``not(<term>)``.  Accept that
        # spelling at the explicitly authored/copy-paste text boundary.  Live
        # verifier terms never round-trip through this parser.
        if name == "not":
            if len(args) != 1:
                raise ValueError(f"not expects 1 arg, got {len(args)}")
            if not args[0].getSort().isBoolean():
                raise ValueError("not expects one Bool argument")
            return solver.mkTerm(Kind.NOT, args[0])
        if _INTEGER_ENCODING:
            special = _int_enc_special_term(solver, name, args)
            if special is not None:
                return special
        fn_map = get_fn_map()
        if name not in fn_map:
            raise ValueError(f"Unknown function: {name}")
        cvc5_op, expected_args, op_bit_width, output_type = fn_map[name][:4]
        if expected_args is not None and len(args) != expected_args:
            raise ValueError(
                f"{name} expects {expected_args} args, got {len(args)}")

        if _INTEGER_ENCODING and op_bit_width is None:
            fixed_args = []
            for arg in args:
                if arg.getSort().isBoolean():
                    fixed_args.append(solver.mkTerm(
                        Kind.ITE, arg, solver.mkInteger(1),
                        solver.mkInteger(0)))
                else:
                    fixed_args.append(arg)
            if cvc5_op is None:
                if len(fixed_args) != 1:
                    raise ValueError(f"{name} identity cast needs one arg")
                return fixed_args[0]
            if output_type == "int_cmp":
                cond = solver.mkTerm(cvc5_op, *fixed_args)
                if not cond.getSort().isBoolean():
                    cond = cvc5_cast_to_bool(solver, cond)
                return solver.mkTerm(
                    Kind.ITE, cond, solver.mkInteger(1),
                    solver.mkInteger(0))
            ret = solver.mkTerm(cvc5_op, *fixed_args)
            if output_type == bool and not ret.getSort().isBoolean():
                ret = cvc5_cast_to_bool(solver, ret)
            return ret

        fixed_args = [cvc5_cast_to_bv(solver, arg, op_bit_width)
                      for arg in args]
        if cvc5_op is None:
            if len(fixed_args) != 1:
                raise ValueError(f"{name} identity cast needs one arg")
            return fixed_args[0]
        ret = solver.mkTerm(cvc5_op, *fixed_args)
        if output_type == bool:
            return cvc5_cast_to_bool(solver, ret)
        return cvc5_cast_to_bv(solver, ret, output_type)

    def _coerce_ite_branches(then_term, else_term):
        """Make C-style ternary branches share a cvc5 sort."""
        try:
            if then_term.getSort() == else_term.getSort():
                return then_term, else_term
            if then_term.getSort().isBoolean() and else_term.getSort().isBitVector():
                return cvc5_cast_to_bv(
                    solver, then_term, else_term.getSort().getBitVectorSize()), else_term
            if else_term.getSort().isBoolean() and then_term.getSort().isBitVector():
                return then_term, cvc5_cast_to_bv(
                    solver, else_term, then_term.getSort().getBitVectorSize())
            if then_term.getSort().isBitVector() and else_term.getSort().isBitVector():
                return _match_bv_sorts(solver, then_term, else_term)
            if _either_int(then_term, else_term):
                return _coerce_to_int(then_term), _coerce_to_int(else_term)
        except Exception:
            pass
        return then_term, else_term

    def _mk_ite(cond_term, then_term, else_term):
        cond = (
            cond_term
            if cond_term.getSort().isBoolean()
            else cvc5_cast_to_bool(solver, cond_term)
        )
        then_term, else_term = _coerce_ite_branches(then_term, else_term)
        return solver.mkTerm(Kind.ITE, cond, then_term, else_term)

    def parse_expr(min_prec=0):
        left = parse_primary()
        while True:
            typ, val = peek()
            if typ != 'OP' or val not in _PREC or _PREC[val] < min_prec:
                break
            advance()
            right = parse_expr(_PREC[val] + 1)
            # Logical connectives operate only on Bool terms.
            if val in _BOOL_OPS:
                if not (left.getSort().isBoolean()
                        and right.getSort().isBoolean()):
                    raise RuntimeError(
                        f"logical operator {val!r} requires Bool operands "
                        f"(got sorts {left.getSort()} and {right.getSort()})")
                left = solver.mkTerm(_BOOL_OPS[val], left, right)
                continue

            if _either_int(left, right) and val in _INT_OPS:
                # Integer-theory path — skip BV matching, use the
                # integer-theory operators.  When one side is an Int
                # variable (from a StateCache that stored scalars as
                # Int) and the other is a BV literal (our default for
                # numbers), coerce the literal to Int so the
                # mkTerm(LT, Int, BV) doesn't crash.
                left_i = _coerce_to_int(left)
                right_i = _coerce_to_int(right)
                kind = _INT_OPS[val]
                result = solver.mkTerm(kind, left_i, right_i)
            else:
                kind = _BIN_OPS[val]
                left, right = _match_bv_sorts(solver, left, right)
                result = solver.mkTerm(kind, left, right)
            left = result
        if min_prec <= 0 and peek()[0] == '?':
            advance()
            then_term = parse_expr(0)
            expect(':')
            else_term = parse_expr(0)
            left = _mk_ite(left, then_term, else_term)
        return left

    def parse_primary():
        typ, val = peek()
        if typ == '(':
            advance()
            expr = parse_expr(0)
            expect(')')
            # Check for array select: expr[index]
            if peek()[0] == '[':
                advance()
                idx = parse_expr(0)
                expect(']')
                return solver.mkTerm(Kind.SELECT, expr, idx)
            return expr
        elif typ == 'ID' and val in ('true', 'True', 'false', 'False'):
            # Accept Python-stringified booleans ("True"/"False") as well as the
            # cvc5/Boogie lowercase spelling. A re-parsed capital "True" (e.g. a
            # vacuously-true WP printed via str(bool)) would otherwise fall to
            # the ID->variable branch below and become an unresolved FREE
            # VARIABLE named "True", defeating the is_true discharge gate.
            advance()
            return solver.mkBoolean(val.lower() == 'true')
        elif typ == 'ID' and val == 'STORE':
            advance()
            expect('(')
            arr = parse_expr(0)
            expect(',')
            idx = parse_expr(0)
            expect(',')
            v = parse_expr(0)
            expect(')')
            return solver.mkTerm(Kind.STORE, arr, idx, v)
        elif typ == 'ID':
            advance()
            if peek()[0] == '(':
                return _mk_function_call(val, _parse_call_args())
            term = state_cache.cvc5_var(val)
            if term is None:
                raise ValueError(f"Unknown variable: {val}")
            # Check for array select: var[index]
            if peek()[0] == '[':
                advance()
                idx = parse_expr(0)
                expect(']')
                return solver.mkTerm(Kind.SELECT, term, idx)
            return term
        elif typ == 'NUM':
            advance()
            # In SMACK integer encoding (type iN = int) a numeric literal is an
            # Int — emit it directly so it composes with arithmetic helpers in
            # any position (e.g. as a function arg `$sge.i32($i0, 1)`, where no
            # surrounding infix operator is present to coerce a BV literal). The
            # infix-binary coercion only fixes operand-position BV literals.
            if _INTEGER_ENCODING:
                return solver.mkInteger(int(val))
            # Default 32-bit bitvector — matches the SMACK-emitted
            # scalar width.  _match_bv_sorts will resize on sort
            # mismatch when the other operand is BV64 etc.
            return solver.mkBitVector(32, int(val))
        elif typ == 'OP' and val in ('~', '!'):
            # Boogie-style negation: ``~(expr)`` or ``!(expr)``. The
            # operand's sort decides whether to emit Bool NOT or BV NOT.
            advance()
            operand = parse_primary()
            if operand.getSort().isBoolean():
                return solver.mkTerm(Kind.NOT, operand)
            return solver.mkTerm(Kind.BITVECTOR_NOT, operand)
        elif typ == 'OP' and val == '-':
            # Unary minus.  ``-1`` tokenizes as OP=- followed by
            # NUM=1 because the tokenizer has no concept of sign
            # on a number literal.  Handle it here so expressions
            # like ``$i0 + $u0 * -1 < 0`` parse.
            #
            # When the operand is a plain BV literal we rebuild it
            # in place as a negative BV value (two's-complement).
            # This keeps the result in the same theory (BV) as
            # sibling literals, which ``_match_bv_sorts`` already
            # handles when widths differ.  For non-literal operands
            # we emit ``BITVECTOR_NEG``, which works for BV vars
            # and produces an Int-theory negate later via the
            # ``_either_int`` bridge if the surrounding expression
            # is Int-sorted.
            advance()
            operand = parse_primary()
            try:
                if operand.isBitVectorValue():
                    width = operand.getSort().getBitVectorSize()
                    v = int(operand.getBitVectorValue(10))
                    modular = ((-v) % (1 << width))
                    return solver.mkBitVector(width, modular)
            except Exception:
                pass
            # Variable operand or non-BV literal: defer to the BV
            # negate op.  Integer-sorted operands get handled by
            # the top-level arithmetic path which will route via
            # Kind.NEG through _either_int above.
            try:
                if operand.getSort().isInteger():
                    return solver.mkTerm(Kind.NEG, operand)
            except Exception:
                pass
            return solver.mkTerm(Kind.BITVECTOR_NEG, operand)
        else:
            raise ValueError(f"Unexpected token: {typ}={val} at pos {pos[0]}")

    def _match_bv_sorts(solver, a, b):
        """Ensure two bitvector terms share a width.

        Policy:
          1. Non-BV sides (Int, Bool) pass through untouched — caller
             will hit a real cvc5 error, which we *want* to surface
             rather than mask behind ``except: pass``.
          2. When widths differ and one side is a literal, rebuild the
             literal at the other side's width. This is the common
             inject-engine case (``$i1 >= 0`` → BV32 ``$i1`` vs. BV64
             default literal; rebuild ``0`` at BV32).
          3. When both sides are non-literal BV terms of different
             widths, zero-extend the narrower.

        The previous implementation silently swallowed every
        exception, turning genuine mismatches into a ``mkTerm``
        crash downstream with only the opaque message "expecting a
        bit-vector term". This revision surfaces the failure.
        """
        as_ = a.getSort()
        bs_ = b.getSort()
        if not as_.isBitVector() or not bs_.isBitVector():
            return a, b
        aw = as_.getBitVectorSize()
        bw = bs_.getBitVectorSize()
        if aw == bw:
            return a, b
        # Literal → rebuild at the other side's width.
        if a.isBitVectorValue():
            return (solver.mkBitVector(bw, int(a.getBitVectorValue(10))),
                    b)
        if b.isBitVectorValue():
            return (a,
                    solver.mkBitVector(aw, int(b.getBitVectorValue(10))))
        # Non-literal BV terms of different widths: zero-extend the
        # narrower.  Access the raw cvc5 solver for ``mkOp`` which
        # takes an Op argument rather than going through
        # SolverWrapper's ``__getattr__``.
        raw = getattr(solver, "solver", solver)
        if aw < bw:
            op = raw.mkOp(Kind.BITVECTOR_ZERO_EXTEND, bw - aw)
            return raw.mkTerm(op, a), b
        op = raw.mkOp(Kind.BITVECTOR_ZERO_EXTEND, aw - bw)
        return a, raw.mkTerm(op, b)

    return parse_expr(0)


def _parse_smt_term(s, ae, state_cache):
    """Parse an smt-lib s-expression string into a cvc5 Term."""
    from cvc5 import Kind
    s = s.strip()

    # Bitvector literal: #b0101...
    if s.startswith("#b"):
        bits = s[2:]
        return state_cache.solver.mkBitVector(len(bits), int(bits, 2))

    # Bitvector hex literal: #x0A...
    if s.startswith("#x"):
        hex_digits = s[2:]
        return state_cache.solver.mkBitVector(len(hex_digits) * 4, int(hex_digits, 16))

    # S-expression: (op arg1 arg2 ...)
    if s.startswith("("):
        # Find matching close paren and tokenize
        inner = s[1:-1].strip()
        tokens = _tokenize_sexp(inner)
        op = tokens[0]
        args = [_parse_smt_term(t, ae, state_cache) for t in tokens[1:]]

        if op == "=":
            return state_cache.solver.mkTerm(Kind.EQUAL, *args)
        elif op == "select":
            array_term, index_term = args
            idx_sort = array_term.getSort().getArrayIndexSort()
            if index_term.getSort() != idx_sort:
                if index_term.isBitVectorValue():
                    index_val = int(index_term.getBitVectorValue(10))
                    index_term = state_cache.solver.mkBitVector(idx_sort.getBitVectorSize(), index_val)
                elif index_term.getSort().isBitVector() and idx_sort.isBitVector():
                    val_w = index_term.getSort().getBitVectorSize()
                    idx_w = idx_sort.getBitVectorSize()
                    if val_w > idx_w:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_EXTRACT, idx_w - 1, 0)
                        index_term = state_cache.solver.mkTerm(op_obj, index_term)
                    else:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, idx_w - val_w)
                        index_term = state_cache.solver.mkTerm(op_obj, index_term)
            return state_cache.solver.mkTerm(Kind.SELECT, array_term, index_term)
        elif op == "store":
            array_term, index_term, value_term = args
            idx_sort = array_term.getSort().getArrayIndexSort()
            elem_sort = array_term.getSort().getArrayElementSort()
            if index_term.getSort() != idx_sort:
                if index_term.isBitVectorValue():
                    index_val = int(index_term.getBitVectorValue(10))
                    index_term = state_cache.solver.mkBitVector(idx_sort.getBitVectorSize(), index_val)
                elif index_term.getSort().isBitVector() and idx_sort.isBitVector():
                    # Truncate or extend to match index sort
                    val_w = index_term.getSort().getBitVectorSize()
                    idx_w = idx_sort.getBitVectorSize()
                    if val_w > idx_w:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_EXTRACT, idx_w - 1, 0)
                        index_term = state_cache.solver.mkTerm(op_obj, index_term)
                    else:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, idx_w - val_w)
                        index_term = state_cache.solver.mkTerm(op_obj, index_term)
            if value_term.getSort() != elem_sort:
                if value_term.isBitVectorValue():
                    val = int(value_term.getBitVectorValue(10))
                    value_term = state_cache.solver.mkBitVector(elem_sort.getBitVectorSize(), val)
                elif value_term.getSort().isBitVector() and elem_sort.isBitVector():
                    # Truncate or extend to match element sort
                    val_w = value_term.getSort().getBitVectorSize()
                    elem_w = elem_sort.getBitVectorSize()
                    if val_w > elem_w:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_EXTRACT, elem_w - 1, 0)
                        value_term = state_cache.solver.mkTerm(op_obj, value_term)
                    else:
                        op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, elem_w - val_w)
                        value_term = state_cache.solver.mkTerm(op_obj, value_term)
            return state_cache.solver.mkTerm(Kind.STORE, array_term, index_term, value_term)
        elif op == "bvadd":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_ADD, *args)
        elif op == "bvmul":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_MUL, *args)
        elif op == "bvand":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_AND, *args)
        elif op == "bvor":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_OR, *args)
        elif op == "bvxor":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_XOR, *args)
        elif op == "bvshl":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_SHL, *args)
        elif op == "bvlshr":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_LSHR, *args)
        elif op == "bvashr":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_ASHR, *args)
        elif op == "bvnot":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_NOT, *args)
        elif op == "bvsub":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_SUB, *args)
        elif op == "concat":
            return state_cache.solver.mkTerm(Kind.BITVECTOR_CONCAT, *args)
        elif op == "and":
            return state_cache.solver.mkTerm(Kind.AND, *args)
        elif op == "or":
            return state_cache.solver.mkTerm(Kind.OR, *args)
        elif op == "not":
            return state_cache.solver.mkTerm(Kind.NOT, *args)
        elif op == "_" or op.startswith("(_"):
            # Indexed operator: (_ extract 7 0), (_ zero_extend 32), etc.
            # op="_" means tokens[1]="extract", tokens[2]="7", tokens[3]="0", then args from remaining
            # Or the whole (_ extract 7 0) was parsed as a single sub-sexp token
            if op == "_":
                idx_op = tokens[1]
                idx_params = tokens[2:]
                actual_args = args  # args were parsed from tokens after the (_ ...) group
            else:
                # (_ op params) was a single token, need to re-parse
                inner_tokens = op[2:-1].strip().split()  # strip "(_" and ")"
                idx_op = inner_tokens[0]
                idx_params = inner_tokens[1:]
                actual_args = args

            if idx_op == "extract":
                hi, lo = int(idx_params[0]), int(idx_params[1])
                op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_EXTRACT, hi, lo)
                return state_cache.solver.mkTerm(op_obj, actual_args[-1])
            elif idx_op == "zero_extend":
                width = int(idx_params[0])
                op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, width)
                return state_cache.solver.mkTerm(op_obj, actual_args[-1])
            elif idx_op == "sign_extend":
                width = int(idx_params[0])
                op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, width)
                return state_cache.solver.mkTerm(op_obj, actual_args[-1])
            elif idx_op == "repeat":
                count = int(idx_params[0])
                op_obj = state_cache.solver.mkOp(Kind.BITVECTOR_REPEAT, count)
                return state_cache.solver.mkTerm(op_obj, actual_args[-1])
            else:
                raise ValueError(f"Unknown indexed op '(_ {idx_op} ...)' in: {s}")
        else:
            raise ValueError(f"Unknown smt-lib op '{op}' in: {s}")

    # Variable name
    var = state_cache.cvc5_var(s)
    if var is not None:
        return var
    raise ValueError(f"Unknown smt-lib term: {s}")


def _tokenize_sexp(s):
    """Split an s-expression body into top-level tokens (respecting nesting)."""
    tokens = []
    depth = 0
    start = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            if depth == 0:
                # flush any preceding atom
                pre = s[start:i].strip()
                if pre:
                    tokens.append(pre)
                start = i
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                tokens.append(s[start:i+1])
                start = i + 1
        elif c in (' ', '\t', '\n') and depth == 0:
            atom = s[start:i].strip()
            if atom:
                tokens.append(atom)
            start = i + 1
        i += 1
    # trailing atom
    rest = s[start:].strip()
    if rest:
        tokens.append(rest)
    return tokens


def _is_smt_body(body):
    """Check if the body of a constraint tuple is an smt-lib s-expression."""
    body = body.strip()
    return body.startswith("(=") or body.startswith("(select") or body.startswith("(store")


def str_to_key(str_key, ae, state_cache):
    from src.solver.predicate import Predicate
    from src.state.proof_obligation import ProofObligation
    from cvc5 import Kind

    # Try to parse as (pc, smt-lib-expr)
    # Use paren-aware extraction: find pc, then take the rest as body
    outer_pattern = r"^\s*\(\s*(\d+)\s*,\s*"
    outer_match = re.match(outer_pattern, str_key.strip())
    if outer_match:
        pc_str = outer_match.group(1)
        rest = str_key.strip()[outer_match.end():]
        # Strip trailing closing paren(s) that belong to the outer tuple
        if rest.endswith(")"):
            body = rest[:-1].strip()
        else:
            body = rest.strip()
    else:
        body = None
    if body and _is_smt_body(body):
        # Validate balanced parens
        if body.count("(") != body.count(")"):
            raise ValueError(f"Unbalanced parens in smt body: opens={body.count('(')}, closes={body.count(')')}, body=...{body[-40:]}")
        pc = int(pc_str)
        cvc5_term = _parse_smt_term(body, ae, state_cache)

        # Determine predicate type from the term structure
        if cvc5_term.getKind() == Kind.EQUAL:
            lhs = cvc5_term[0]
            rhs = cvc5_term[1]
            lhs_vars = extract_variable_terms(lhs)
            rhs_vars = extract_variable_terms(rhs)
            all_vars = lhs_vars | rhs_vars
            has_shadow = any(v.getSymbol().endswith(".shadow") for v in all_vars)
            has_non_shadow = any(not v.getSymbol().endswith(".shadow") for v in all_vars)

            if has_shadow and has_non_shadow:
                # eq predicate (shadow equality)
                ret_term = Predicate(cvc5_term)
            elif rhs.isBitVectorValue() or lhs.isBitVectorValue():
                # eq_const predicate
                ret_term = Predicate(cvc5_term)
            else:
                ret_term = Predicate(cvc5_term)
        else:
            ret_term = Predicate(cvc5_term)
        return ProofObligation.from_predicate(
            pc, ret_term, solver=state_cache.solver)

    x = parse_constraint_tuple(str_key)
    pc = x["id"]
    
    if x["type"] == "shadow_variable":
        lhs_name = x["variable"]
        rhs_name = x["shadow_variable"]

        # Case 1: Standard Self-Shadow Check (Var == Var.shadow)
        if rhs_name == f"{lhs_name}.shadow":
            var = state_cache.cvc5_var(lhs_name)
            ret_term = ae.EQ(var)
        
        # Case 2: General Equality (VarA == VarB.shadow)
        else:
            lhs_var = state_cache.cvc5_var(lhs_name)
            rhs_var = state_cache.cvc5_var(rhs_name)

            # Sort mismatch (e.g., i64 == i32.shadow after zext removal)
            lhs_var, rhs_var = _coerce_bv_sorts(state_cache.solver, lhs_var, rhs_var)
            eq_term = state_cache.solver.mkTerm(Kind.EQUAL, lhs_var, rhs_var)
            ret_term = Predicate(eq_term)
        
    elif x["type"] == "constant_comparison":
        var = state_cache.cvc5_var(x["variable"])
        ret_term = ae.EQ_CONST(var, [x["constant"]])
        
    elif x["type"] == "shadow_array":
        # Case: select(array, constant_index)
        mem_map_var = state_cache.cvc5_var(x["variable"])
        index_term = state_cache.solver.mkBitVector(
            mem_map_var.getSort().getArrayIndexSort().getBitVectorSize(), 
            x["index"]
        )
        select_term = state_cache.solver.mkTerm(Kind.SELECT, mem_map_var, index_term)
        ret_term = ae.EQ(select_term)
        
    elif x["type"] == "variable_shadow_array_equality":
        # Case: variable == select(array, variable_index)
        lhs_var = state_cache.cvc5_var(x["variable"])
        mem_map_var = state_cache.cvc5_var(x["array"])
        index_var = state_cache.cvc5_var(x["index_variable"])
        
        select_term = state_cache.solver.mkTerm(Kind.SELECT, mem_map_var, index_var)
        lhs_var, select_term = assign_fix_type(state_cache.solver, lhs_var, select_term)
        eq_term = state_cache.solver.mkTerm(Kind.EQUAL, lhs_var, select_term)
        ret_term = ae.EQ(eq_term)

    elif x["type"] == "array_store_equality":
        # Case: LHS_Array == Store(RHS_Array, index, value)
        lhs_array_var = state_cache.cvc5_var(x["lhs_array"])
        rhs_array_var = state_cache.cvc5_var(x["rhs_array"])
        index_var = state_cache.cvc5_var(x["index_variable"])
        value_var = state_cache.cvc5_var(x["value_variable"])
        
        store_term = state_cache.solver.mkTerm(Kind.STORE, rhs_array_var, index_var, value_var)
        eq_term = state_cache.solver.mkTerm(Kind.EQUAL, lhs_array_var, store_term)
        
        ret_term = Predicate(eq_term)

    elif x["type"] == "disjunction_of_constants":
        var = state_cache.cvc5_var(x["variable"])
        ret_term = ae.EQ_CONST(var, x["allowed_values"])
        
    elif x["type"] == "array_select_constant_equality":
        # Case: select(Array, constant_index) == constant_value
        # Matches: ($M.18[1024]) == (525)
        mem_map_var = state_cache.cvc5_var(x["array"])

        # 1. Create the BitVector term for the constant array index
        index_sort = mem_map_var.getSort().getArrayIndexSort()
        index_term = state_cache.solver.mkBitVector(index_sort.getBitVectorSize(), x["index"])

        # 2. Create the SELECT term: Array[index]
        select_term = state_cache.solver.mkTerm(Kind.SELECT, mem_map_var, index_term)

        # 3. Create the EQUAL term: Array[index] == constant_value
        ret_term = ae.EQ_CONST(select_term,[x["constant"]])

    elif x["type"] == "shadow_array_var_index":
        # Case: select(Array, var_index) == select(Array.shadow, var_index.shadow)
        # Matches: ($M.18[$p0]) == ($M.18.shadow[$p0.shadow])
        mem_map_var = state_cache.cvc5_var(x["variable"])
        index_var = state_cache.cvc5_var(x["index_variable"])
        select_term = state_cache.solver.mkTerm(Kind.SELECT, mem_map_var, index_var)
        ret_term = ae.EQ(select_term)

    elif x["type"] == "array_select_var_constant_equality":
        # Case: select(Array, var_index) == constant_value
        # Matches: ($M.18[$p0]) == (42)
        mem_map_var = state_cache.cvc5_var(x["array"])
        index_var = state_cache.cvc5_var(x["index_variable"])
        select_term = state_cache.solver.mkTerm(Kind.SELECT, mem_map_var, index_var)
        ret_term = ae.EQ_CONST(select_term, [x["constant"]])

    elif x["type"] == "unknown":
        # Fallback: parse the full infix body with the general parser
        body = x.get("raw_body", "")
        cvc5_term = _parse_infix_expr(body, state_cache)
        ret_term = Predicate(cvc5_term)

    else:
        raise ValueError(f"Unsupported constraint type '{x['type']}' for: {str_key}")

    return ProofObligation.from_predicate(
        pc, ret_term, solver=state_cache.solver)
