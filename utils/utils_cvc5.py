from cvc5 import Kind, Term, Sort, Solver, SortKind
import math
from interpreter.parser.expression import FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier, BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral, OldExpression, LogicalNegation, ArithmeticNegation, QuantifiedExpression, IfExpression, Identifier
from interpreter.parser.statement import AssertStatement, AssumeStatement, AssignStatement, Block, CallStatement, GotoStatement, HavocStatement
from interpreter.parser.declaration import StorageDeclaration, ImplementationDeclaration, ProcedureDeclaration
from interpreter.parser.type import BooleanType, IntegerType, CustomType, MapType
from interpreter.utils.cvc5_helper import term_to_string, sign_extend, zero_extend

from interpreter.utils.utils import boogie_type_bitwidth
from collections import deque
import pickle
from functools import lru_cache
from cachetools import LRUCache
from interpreter.utils.utils import IndentLogger, indent_log

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

    "$ne.ref": (Kind.DISTINCT, 2, 64, 64),
    "$ne.i64": (Kind.DISTINCT, 2, 64, 64),
    "$ne.i32": (Kind.DISTINCT, 2, 32, 32),
    "$ne.i8": (Kind.DISTINCT, 2, 8, 8),

    "$eq.ref": (Kind.EQUAL, 2, 64, 64),
    "$eq.i64": (Kind.EQUAL, 2, 64, 64),
    "$eq.i32": (Kind.EQUAL, 2, 32, 32),
    "$eq.i8": (Kind.EQUAL, 2, 8, 8),
    "$eq.i1": (Kind.EQUAL, 2, 1, 1),

    "$udiv.ref": (Kind.BITVECTOR_UDIV, 2, 64, 64),
    "$udiv.i64": (Kind.BITVECTOR_UDIV, 2, 64, 64),
    "$udiv.i32": (Kind.BITVECTOR_UDIV, 2, 32, 32),
    "$udiv.i8": (Kind.BITVECTOR_UDIV, 2, 8, 8),

    # Signed division
    "$sdiv.i64": (Kind.BITVECTOR_SDIV, 2, 64, 64),
    "$sdiv.i32": (Kind.BITVECTOR_SDIV, 2, 32, 32),
    "$sdiv.i8": (Kind.BITVECTOR_SDIV, 2, 8, 8),

    "$ult.ref": (Kind.BITVECTOR_ULT, 2, 64, 64),
    "$ult.i64": (Kind.BITVECTOR_ULT, 2, 64, 64),
    "$ult.i32": (Kind.BITVECTOR_ULT, 2, 32, 32),
    "$ult.i8": (Kind.BITVECTOR_ULT, 2, 8, 8),
    "$ugt.i64": (Kind.BITVECTOR_UGT, 2, 64, 64),
    "$ugt.i32": (Kind.BITVECTOR_UGT, 2, 32, 32),
    "$ugt.i8": (Kind.BITVECTOR_UGT, 2, 8, 8),
    "$uge.i64": (Kind.BITVECTOR_UGE, 2, 64, 64),
    "$uge.i32": (Kind.BITVECTOR_UGE, 2, 32, 32),
    "$uge.i8": (Kind.BITVECTOR_UGE, 2, 8, 8),

    # Signed greater than
    "$sgt.ref.bool": (Kind.BITVECTOR_SGT, 2, 64, bool),
    "$sgt.i64": (Kind.BITVECTOR_SGT, 2, 64, 64),
    "$sgt.i32": (Kind.BITVECTOR_SGT, 2, 32, 32),
    "$sgt.i8": (Kind.BITVECTOR_SGT, 2, 8, 8),

    # Signed greater than or equal to
    "$sge.ref.bool": (Kind.BITVECTOR_SGE, 2, 64, bool),
    "$sge.i64": (Kind.BITVECTOR_SGE, 2, 64, 64),
    "$sge.i32": (Kind.BITVECTOR_SGE, 2, 32, 32),
    "$sge.i8": (Kind.BITVECTOR_SGE, 2, 8, 8),

    # Signed less than or equal to
    "$sle.i64": (Kind.BITVECTOR_SLE, 2, 64, 64),
    "$sle.i32": (Kind.BITVECTOR_SLE, 2, 32, 32),
    "$sle.i8": (Kind.BITVECTOR_SLE, 2, 8, 8),
    "$sle.ref.bool": (Kind.BITVECTOR_SLE, 2, 64, bool),

    "$slt.ref.bool": (Kind.BITVECTOR_SLT, 2, 64, bool),
    "$slt.i64": (Kind.BITVECTOR_SLT, 2, 64, 64),
    "$slt.i32": (Kind.BITVECTOR_SLT, 2, 32, 32),
    "$slt.i8": (Kind.BITVECTOR_SLT, 2, 8, 8),
    "$ule.i64": (Kind.BITVECTOR_ULE, 2, 64, 64),
    "$ule.i32": (Kind.BITVECTOR_ULE, 2, 32, 32),
    "$ule.i8": (Kind.BITVECTOR_ULE, 2, 8, 8),
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
}

# Reverse mapping: cvc5 Kind → SMACK function name (prefer i32 for BV32)
_CVC5_KIND_TO_BOOGIE = {}
for _name, (_kind, _nargs, _width, _out) in fn_to_cvc5_op.items():
    if _name.startswith("$") and _width == 32 and _kind is not None:
        if _kind not in _CVC5_KIND_TO_BOOGIE:
            _CVC5_KIND_TO_BOOGIE[_kind] = _name


def cvc5_to_boogie(term, depth=0) -> str:
    """Convert a cvc5 term to Boogie/SMACK syntax.

    Unlike term_to_string (display-oriented), this produces syntax the LLM
    can parse back: $mul.i32($i6, $sub.i32($i6, 1)) instead of $i6 * ($i6 - 1).
    Handles cvc5 simplification artifacts (CONCAT→MUL, ITE→bool, etc.).
    """
    if depth > 30:
        return "..."
    if term is None:
        return "EMPTY"

    # Base case: leaf nodes
    if term.getNumChildren() == 0:
        if term.isBitVectorValue():
            return str(int(term.getBitVectorValue(), 2))
        if term.isBooleanValue():
            return "true" if term.getBooleanValue() else "false"
        try:
            return term.getSymbol()
        except Exception:
            return str(term)

    kind = term.getKind()

    # --- Handle cvc5 simplification artifacts ---

    # CONCAT patterns (cvc5 simplification artifacts)
    if kind == Kind.BITVECTOR_CONCAT and term.getNumChildren() == 2:
        hi, lo = term[0], term[1]
        # CONCAT(EXTRACT(N-1,0,x), #b0...0) → $mul.i32(2^K, x)  (left-shift)
        if lo.isBitVectorValue() and int(lo.getBitVectorValue(), 2) == 0:
            n_zeros = lo.getSort().getBitVectorSize()
            if hi.getKind() == Kind.BITVECTOR_EXTRACT:
                inner = cvc5_to_boogie(hi[0], depth + 1)
                return f"$mul.i32({1 << n_zeros}, {inner})"
        # CONCAT(#b0...0, x) → x  (zero-extension, transparent in Boogie)
        if hi.isBitVectorValue() and int(hi.getBitVectorValue(), 2) == 0:
            return cvc5_to_boogie(lo, depth + 1)

    # ITE(cond, bv(1), bv(0)) → cond  (Bool-to-BV cast)
    if kind == Kind.ITE and term.getNumChildren() == 3:
        t_val, f_val = term[1], term[2]
        if (t_val.isBitVectorValue() and int(t_val.getBitVectorValue(), 2) == 1
                and f_val.isBitVectorValue() and int(f_val.getBitVectorValue(), 2) == 0):
            return cvc5_to_boogie(term[0], depth + 1)

    # --- Standard mappings ---

    if kind == Kind.EQUAL:
        lhs = cvc5_to_boogie(term[0], depth + 1)
        rhs = cvc5_to_boogie(term[1], depth + 1)
        return f"{lhs} == {rhs}"

    if kind == Kind.DISTINCT:
        lhs = cvc5_to_boogie(term[0], depth + 1)
        rhs = cvc5_to_boogie(term[1], depth + 1)
        return f"{lhs} != {rhs}"

    if kind == Kind.NOT:
        inner = cvc5_to_boogie(term[0], depth + 1)
        return f"!({inner})"

    # bvneg($x) → $sub.i32(0, $x) (SMACK has no $neg.i32)
    if kind == Kind.BITVECTOR_NEG:
        inner = cvc5_to_boogie(term[0], depth + 1)
        width = term.getSort().getBitVectorSize()
        return f"$sub.i{width}(0, {inner})"

    if kind == Kind.AND:
        parts = [cvc5_to_boogie(term[i], depth + 1) for i in range(term.getNumChildren())]
        return " && ".join(parts)

    if kind == Kind.OR:
        parts = [cvc5_to_boogie(term[i], depth + 1) for i in range(term.getNumChildren())]
        return " || ".join(parts)

    if kind == Kind.SELECT:
        arr = cvc5_to_boogie(term[0], depth + 1)
        idx = cvc5_to_boogie(term[1], depth + 1)
        return f"{arr}[{idx}]"

    if kind == Kind.STORE:
        arr = cvc5_to_boogie(term[0], depth + 1)
        idx = cvc5_to_boogie(term[1], depth + 1)
        val = cvc5_to_boogie(term[2], depth + 1)
        return f"STORE({arr}, {idx}, {val})"

    if kind == Kind.IMPLIES:
        lhs = cvc5_to_boogie(term[0], depth + 1)
        rhs = cvc5_to_boogie(term[1], depth + 1)
        return f"({lhs}) ==> ({rhs})"

    # Refactor cvc5 simplification artifacts back to readable Boogie:
    # ADD(MULT(X, X), NEG(X)) → $mul.i32(X, $sub.i32(X, 1))  [x² - x = x*(x-1)]
    # ADD(MULT(X, X), X)      → $mul.i32(X, $add.i32(X, 1))  [x² + x = x*(x+1)]
    if kind == Kind.BITVECTOR_ADD and term.getNumChildren() == 2:
        a, b = term[0], term[1]
        # Pattern: X² - X → X*(X-1)
        if (a.getKind() == Kind.BITVECTOR_MULT and a.getNumChildren() == 2
                and b.getKind() == Kind.BITVECTOR_NEG):
            if str(a[0]) == str(a[1]) == str(b[0]):
                x = cvc5_to_boogie(a[0], depth + 1)
                w = term.getSort().getBitVectorSize()
                return f"$mul.i32({x}, $sub.i{w}({x}, 1))"
        # Pattern: X² + X → X*(X+1)
        if (a.getKind() == Kind.BITVECTOR_MULT and a.getNumChildren() == 2
                and b.getKind() == Kind.CONSTANT and b.getSort().isBitVector()):
            if str(a[0]) == str(a[1]) == str(b):
                x = cvc5_to_boogie(a[0], depth + 1)
                w = term.getSort().getBitVectorSize()
                return f"$mul.i32({x}, $add.i{w}({x}, 1))"

    # SMACK function call: Kind → $op.i32(args)
    if kind in _CVC5_KIND_TO_BOOGIE:
        fn = _CVC5_KIND_TO_BOOGIE[kind]
        args = ", ".join(cvc5_to_boogie(term[i], depth + 1)
                        for i in range(term.getNumChildren()))
        return f"{fn}({args})"

    # Fallback: use term_to_string for unknown kinds
    return term_to_string(term)


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
        "$sext.i32.i64": (solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, 64), 1, 32, 64),

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
            logger.debug(f"  - {term_to_string(a)}")
    else:
        IndentLogger.debug("solver assertions:")
        for a in solver.getAssertions():
            IndentLogger.debug(f"  - {term_to_string(a)}")

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
    # Equality / inequality (already sort-polymorphic)
    "$eq.ref": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i64": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i32": (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i8":  (Kind.EQUAL, 2, None, "int_cmp"),
    "$eq.i1":  (Kind.EQUAL, 2, None, "int_cmp"),
    "$ne.ref": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i64": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i32": (Kind.DISTINCT, 2, None, "int_cmp"),
    "$ne.i8":  (Kind.DISTINCT, 2, None, "int_cmp"),
    # Signed comparisons → integer comparisons
    "$slt.i32": (Kind.LT, 2, None, "int_cmp"),
    "$slt.i64": (Kind.LT, 2, None, "int_cmp"),
    "$slt.i8":  (Kind.LT, 2, None, "int_cmp"),
    "$slt.ref.bool": (Kind.LT, 2, None, bool),
    "$sle.i32": (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.i64": (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.i8":  (Kind.LEQ, 2, None, "int_cmp"),
    "$sle.ref.bool": (Kind.LEQ, 2, None, bool),
    "$sgt.i32": (Kind.GT, 2, None, "int_cmp"),
    "$sgt.i64": (Kind.GT, 2, None, "int_cmp"),
    "$sgt.i8":  (Kind.GT, 2, None, "int_cmp"),
    "$sgt.ref.bool": (Kind.GT, 2, None, bool),
    "$sge.i32": (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.i64": (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.i8":  (Kind.GEQ, 2, None, "int_cmp"),
    "$sge.ref.bool": (Kind.GEQ, 2, None, bool),
    # Unsigned comparisons → same as signed in integer mode (no wraparound)
    "$ult.i32": (Kind.LT, 2, None, "int_cmp"),
    "$ult.i64": (Kind.LT, 2, None, "int_cmp"),
    "$ult.i8":  (Kind.LT, 2, None, "int_cmp"),
    "$ult.ref":  (Kind.LT, 2, None, "int_cmp"),
    "$ule.i32": (Kind.LEQ, 2, None, "int_cmp"),
    "$ule.i64": (Kind.LEQ, 2, None, "int_cmp"),
    "$ule.i8":  (Kind.LEQ, 2, None, "int_cmp"),
    "$ugt.i32": (Kind.GT, 2, None, "int_cmp"),
    "$ugt.i64": (Kind.GT, 2, None, "int_cmp"),
    "$ugt.i8":  (Kind.GT, 2, None, "int_cmp"),
    "$uge.i32": (Kind.GEQ, 2, None, "int_cmp"),
    "$uge.i64": (Kind.GEQ, 2, None, "int_cmp"),
    "$uge.i8":  (Kind.GEQ, 2, None, "int_cmp"),
    # Division / remainder → integer ops
    "$sdiv.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$sdiv.i64": (Kind.INTS_DIVISION, 2, None, None),
    "$udiv.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$udiv.i64": (Kind.INTS_DIVISION, 2, None, None),
    "$srem.i32": (Kind.INTS_MODULUS, 2, None, None),
    "$srem.i64": (Kind.INTS_MODULUS, 2, None, None),
    "$urem.i32": (Kind.INTS_MODULUS, 2, None, None),
    "$urem.i64": (Kind.INTS_MODULUS, 2, None, None),
    # Bitwise ops — handled by _INT_ENC_BITWISE_OPS in convert_expr_cvc5
    # (int→bv→op→nat wrapping for exact semantics). These entries are
    # kept as placeholders so the fn_map lookup succeeds; the actual
    # cvc5_op value is ignored because the bitwise path runs first.
    "$and.i32": (Kind.MULT, 2, None, None),
    "$and.i64": (Kind.MULT, 2, None, None),
    "$and.i8":  (Kind.MULT, 2, None, None),
    "$and.i1":  (Kind.MULT, 2, None, None),
    "$and.ref":  (Kind.MULT, 2, None, None),
    "$or.i32":  (Kind.ADD, 2, None, None),
    "$or.i64":  (Kind.ADD, 2, None, None),
    "$or.i8":   (Kind.ADD, 2, None, None),
    "$or.i1":   (Kind.ADD, 2, None, None),
    "$or.ref":   (Kind.ADD, 2, None, None),
    "$xor.i32": (Kind.SUB, 2, None, None),
    "$xor.i64": (Kind.SUB, 2, None, None),
    "$xor.i8":  (Kind.SUB, 2, None, None),
    "$xor.i1":  (Kind.SUB, 2, None, None),
    "$xor.ref":  (Kind.SUB, 2, None, None),
    "$not.i1":  (Kind.SUB, 2, None, None),
    "$not.i8":  (Kind.SUB, 2, None, None),
    "$not.i32": (Kind.SUB, 2, None, None),
    "$not.i64": (Kind.SUB, 2, None, None),
    "$not.ref": (Kind.SUB, 2, None, None),
    "$shl.i32":  (Kind.MULT, 2, None, None),
    "$shl.i64":  (Kind.MULT, 2, None, None),
    "$lshr.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$lshr.i64": (Kind.INTS_DIVISION, 2, None, None),
    "$ashr.i32": (Kind.INTS_DIVISION, 2, None, None),
    "$ashr.i64": (Kind.INTS_DIVISION, 2, None, None),
    # Casts — identity in integer mode
    "$bitcast.ref.ref": (None, 1, None, None),
    "$p2i.ref.i64": (None, 1, None, None),
    "$i2p.i64.ref": (None, 1, None, None),
    # Mul.ref
    "$mul.ref": (Kind.MULT, 2, None, None),
    "$add.ref": (Kind.ADD, 2, None, None),
    "$sub.ref": (Kind.SUB, 2, None, None),
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
    "/":  (Kind.DIVISION, 2, None, None),
    "&&": (Kind.AND, 2, bool, bool),
    "||": (Kind.OR, 2, bool, bool),
    "==>": (Kind.IMPLIES, 2, None, None),
}

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

def detect_integer_encoding(program):
    """Detect if program uses SMACK unbounded-integer encoding.

    ``SMACK ≥ 2.8.0`` always emits ``type i32 = int`` as a Boogie alias
    regardless of the actual SMT encoding, but the program then uses
    BV-named function intrinsics (``$add.bv32``, ``$slt.bv32`` etc.)
    when SMACK is invoked with ``--integer-encoding bit-vector``. The
    truthful signal is the presence of ``$add.bv*`` or similar BV
    function declarations: BV mode → present; pure-int mode → absent.

    Returns True iff this is truly unbounded-integer mode (no BV
    function declarations found). Defaults to False (BV) on ambiguity
    so existing BV-tested verification proofs keep working.
    """
    from interpreter.parser.declaration import (
        FunctionDeclaration,
        TypeDeclaration,
    )
    from interpreter.parser.type import IntegerType

    has_int_alias = False
    for d in program.declarations:
        if isinstance(d, TypeDeclaration):
            if hasattr(d, 'names') and 'i32' in d.names:
                if hasattr(d, 'type') and isinstance(d.type, IntegerType):
                    has_int_alias = True
                    break
    if not has_int_alias:
        return False

    # Look for SMACK's BV intrinsic function names — present in BV mode,
    # absent in unbounded-integer mode.
    for d in program.declarations:
        if isinstance(d, FunctionDeclaration):
            name = getattr(d, 'name', '') or ''
            if name.startswith('$add.bv') or name.startswith('$slt.bv') or \
               name.startswith('$mul.bv') or name.startswith('$bv2int'):
                return False  # BV intrinsics present → BV mode

    return True

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
            if not _INTEGER_ENCODING and rhs.isBitVectorValue():
                rhs = sign_extend(solver, rhs, lhs.getSort().getBitVectorSize())
            fn_entry = cvc5_fn_map[expr.op]
            cvc5_op = fn_entry[0]

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
                    Kind.LT: Kind.BITVECTOR_ULT,
                    Kind.LEQ: Kind.BITVECTOR_ULE,
                    Kind.GT: Kind.BITVECTOR_UGT,
                    Kind.GEQ: Kind.BITVECTOR_UGE,
                }.get(cvc5_op, cvc5_op)
            return solver.mkTerm(cvc5_op, lhs, rhs)
        elif expr.op == "==":
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
            lhs, rhs = assign_fix_type(solver, lhs, rhs)
            return solver.mkTerm(Kind.EQUAL, lhs, rhs)
        elif expr.op == "!=":
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
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
            # If inner is BV1, convert to Bool before NOT
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
    elif isinstance(expr, Term):
        return expr
    else:
        assert False, f"unknown expression {expr} {type(expr)}"

KIND_TO_NUM = {
    Kind.EQUAL: 0,
    Kind.DISTINCT: 1,
    Kind.AND: 2,
    Kind.OR: 3,
    Kind.NOT: 4,
    Kind.ADD: 5,
    Kind.MULT: 6,
    Kind.SUB: 7,
    Kind.DIVISION: 8,
    Kind.ITE: 9,
    Kind.LEQ: 10,
    Kind.GEQ: 11,
    Kind.LT: 12,
    Kind.GT: 13,
    Kind.IMPLIES: 14,
    Kind.SELECT: 15,
    Kind.STORE: 16,
    Kind.APPLY_UF: 17,
    Kind.BITVECTOR_ADD: 18,
    Kind.BITVECTOR_MULT: 19,
    Kind.INT_TO_BITVECTOR: 20,
    Kind.BITVECTOR_SIGN_EXTEND: 21,
    Kind.BITVECTOR_ULT: 22,
    Kind.BITVECTOR_EXTRACT: 23,
    Kind.BITVECTOR_ZERO_EXTEND: 24,
    Kind.BITVECTOR_AND: 25,
    Kind.BITVECTOR_LSHR: 26,
    Kind.BITVECTOR_OR : 27,
    Kind.BITVECTOR_SHL: 28,
    Kind.BITVECTOR_XOR: 29,
    Kind.BITVECTOR_SUB: 30,
    Kind.BITVECTOR_CONCAT: 31,
    Kind.BITVECTOR_NOT: 32,
    Kind.BITVECTOR_SLT: 33,
    Kind.BITVECTOR_SREM: 34,
    Kind.BITVECTOR_ULE: 35,
    Kind.BITVECTOR_SGE: 36,
    Kind.BITVECTOR_UGE: 37,
    Kind.BITVECTOR_NEG: 38,
    Kind.BITVECTOR_UDIV: 39,
    Kind.BITVECTOR_UGT: 40,
    Kind.BITVECTOR_SGT: 41,
    Kind.BITVECTOR_ASHR: 42,
    Kind.BITVECTOR_UREM: 43,
    Kind.BITVECTOR_SLE: 44,
    Kind.BITVECTOR_SDIV: 45,
    Kind.XOR: 46,
    Kind.CONST_BITVECTOR: 47,
    Kind.CONST_BOOLEAN: 48,
    Kind.CONSTANT: 49,
    Kind.SET_MEMBER: 50,
    Kind.SET_INSERT: 51,
    Kind.SET_EMPTY: 52,
    Kind.CONST_INTEGER: 53,
    Kind.INTS_MODULUS: 54,
    Kind.INTS_DIVISION: 55,
    Kind.NEG: 56,
    Kind.INTS_MODULUS_TOTAL: 57,
    Kind.INTS_DIVISION_TOTAL: 58,
    Kind.DIVISION: 59,
    Kind.DIVISION_TOTAL: 60,
    Kind.TO_INTEGER: 61,
    Kind.IS_INTEGER: 62,
    Kind.BITVECTOR_UBV_TO_INT: 63,
    Kind.BITVECTOR_SBV_TO_INT: 64,
    Kind.FORALL: 65,
    Kind.VARIABLE_LIST: 66,
    Kind.VARIABLE: 67,
}

NUM_TO_KIND = {num: kind for kind, num in KIND_TO_NUM.items()}

sort_to_num = {
    SortKind.BITVECTOR_SORT: 0,
    SortKind.BOOLEAN_SORT: 1,
    SortKind.INTEGER_SORT: 2,
    SortKind.ARRAY_SORT: 4,
    SortKind.INTERNAL_SORT_KIND: 5,
    SortKind.SET_SORT: 6,
    SortKind.UNINTERPRETED_SORT: 7,
}

NUM_TO_SORT = {num: sort for sort, num in sort_to_num.items()}

def deserialize_predicate_helper(state_cache, predicate):
    if isinstance(predicate.predicate, HollowCvc5Term):
        predicate.predicate = deserialize_cvc5_term(state_cache, predicate.predicate)
        # Clear cached hash AND cached string so they recompute from
        # the new live cvc5 Term — otherwise ``str(predicate)`` keeps
        # returning the hollow form even after rehydration, which
        # breaks anything keying on the stringified predicate.
        predicate._cached_hash = None
        predicate._cached_str = None
        predicate._cached_variable_terms = None

def deserialize_cvc5_term(state_cache, root_term):
    # 1. FASTEST PATH: Check global cache
    if hasattr(state_cache, "cvc5_term_cache"):
        if root_term in state_cache.cvc5_term_cache:
            return state_cache.cvc5_term_cache[root_term]
    else:
        state_cache.cvc5_term_cache = LRUCache(maxsize=50000)

    # 2. METHOD HOISTING: Bind once to avoid dot-lookup in the loop
    solver = state_cache.solver
    mkTerm = solver.mkTerm
    mkOp = solver.mkOp
    mkBitVector = solver.mkBitVector
    mkBoolean = solver.mkBoolean
    mkConst = solver.mkConst
    mkEmptySet = solver.mkEmptySet
    mkSetSort = solver.mkSetSort
    bool_sort = solver.getBooleanSort()
    global_cache = state_cache.cvc5_term_cache
    # Use a local dict for traversal to avoid LRU eviction mid-deserialize.
    # Check global_cache for existing entries, but store new results locally
    # to prevent eviction of children before parents are built.
    memo = {}

    # 3. ITERATIVE post-order traversal (avoids recursion limit + function call overhead)
    stack = [(root_term, 0)]

    while stack:
        term, idx = stack[-1]

        # Already resolved locally
        if term in memo:
            stack.pop()
            continue

        # Check global cache (won't evict during our traversal)
        if term in global_cache:
            memo[term] = global_cache[term]
            stack.pop()
            continue

        op_kind = NUM_TO_KIND[term.op]

        # --- LEAVES: resolve immediately ---
        if op_kind == Kind.CONSTANT:
            res = state_cache.cvc5_var(term.var_name)
            if not res:
                sort_kind = NUM_TO_SORT[term.type]
                if sort_kind == SortKind.BOOLEAN_SORT:
                    res = mkConst(bool_sort, term.var_name)
                elif sort_kind == SortKind.BITVECTOR_SORT:
                    res = mkConst(solver.mkBitVectorSort(term.bitwidth), term.var_name)
                elif sort_kind == SortKind.INTEGER_SORT:
                    res = mkConst(solver.getIntegerSort(), term.var_name)
                elif sort_kind == SortKind.ARRAY_SORT:
                    idx_sort = solver.mkBitVectorSort(term.array_index_width) if term.array_index_width > 0 else solver.getIntegerSort()
                    elem_sort = solver.mkBitVectorSort(term.array_element_width) if term.array_element_width > 0 else solver.getIntegerSort()
                    res = mkConst(solver.mkArraySort(idx_sort, elem_sort), term.var_name)
                else:
                    raise ValueError(f"Unknown sort: {sort_kind}")
                from src.state.redis_keys import put_cvc5_var
                put_cvc5_var(state_cache.redis, term.var_name, res)
            memo[term] = res
            stack.pop()
            continue

        if op_kind == Kind.VARIABLE:
            # Bound variable (e.g. quantified x in forall).  Reuse a
            # previously-created mkVar for this name if one exists in
            # the per-process cache; otherwise create and cache it.
            # Sharing across every forall that binds this name is
            # semantically fine (each forall has its own cvc5 scope)
            # and guarantees VL / body consistency even when subterms
            # come from separate deserialize calls that get later
            # combined into one big term.
            existing = state_cache.cached_id_to_cvc5.get(term.var_name)
            if existing is not None and existing.getKind() == Kind.VARIABLE:
                res = existing
            else:
                sort_kind = NUM_TO_SORT[term.type]
                if sort_kind == SortKind.BOOLEAN_SORT:
                    res = solver.mkVar(bool_sort, term.var_name)
                elif sort_kind == SortKind.BITVECTOR_SORT:
                    res = solver.mkVar(solver.mkBitVectorSort(term.bitwidth), term.var_name)
                elif sort_kind == SortKind.INTEGER_SORT:
                    res = solver.mkVar(solver.getIntegerSort(), term.var_name)
                elif sort_kind == SortKind.ARRAY_SORT:
                    idx_s = solver.mkBitVectorSort(term.array_index_width) if term.array_index_width > 0 else solver.getIntegerSort()
                    elem_s = solver.mkBitVectorSort(term.array_element_width) if term.array_element_width > 0 else solver.getIntegerSort()
                    res = solver.mkVar(solver.mkArraySort(idx_s, elem_s), term.var_name)
                else:
                    raise ValueError(f"Unknown sort for VARIABLE: {sort_kind}")
                state_cache.cached_id_to_cvc5[term.var_name] = res
            memo[term] = res
            stack.pop()
            continue

        if op_kind == Kind.CONST_BITVECTOR:
            memo[term] = mkBitVector(term.bitwidth, term.value)
            stack.pop()
            continue

        if op_kind == Kind.CONST_BOOLEAN:
            memo[term] = mkBoolean(term.value)
            stack.pop()
            continue

        if op_kind == Kind.CONST_INTEGER:
            memo[term] = solver.mkInteger(str(term.value))
            stack.pop()
            continue

        if op_kind == Kind.SET_EMPTY:
            bv_sort = solver.mkBitVectorSort(term.bitwidth)
            memo[term] = mkEmptySet(mkSetSort(bv_sort))
            stack.pop()
            continue

        # --- INTERNAL NODES: ensure children are resolved first ---
        children = term.children
        n = len(children)

        # Advance idx past already-resolved children
        while idx < n and children[idx] in memo:
            idx += 1

        if idx < n:
            # Found unresolved child — save our place and push child
            stack[-1] = (term, idx)
            stack.append((children[idx], 0))
            continue

        # All children resolved — build the cvc5 term
        stack.pop()
        child_terms = [memo[c] for c in children]

        if op_kind == Kind.BITVECTOR_EXTRACT:
            op = mkOp(Kind.BITVECTOR_EXTRACT, term.bv_extract_end, term.bv_extract_start)
            res = mkTerm(op, *child_terms)
        elif op_kind == Kind.BITVECTOR_SIGN_EXTEND:
            op = mkOp(Kind.BITVECTOR_SIGN_EXTEND, term.bv_sign_extend_bitwidth)
            res = mkTerm(op, *child_terms)
        elif op_kind == Kind.BITVECTOR_ZERO_EXTEND:
            op = mkOp(Kind.BITVECTOR_ZERO_EXTEND, term.bv_zero_extend_bitwidth)
            res = mkTerm(op, *child_terms)
        elif op_kind == Kind.SET_MEMBER:
            res = mkTerm(Kind.SET_MEMBER, *child_terms)
        elif op_kind == Kind.INT_TO_BITVECTOR:
            # INT_TO_BITVECTOR needs bitwidth index. Use the stored bitwidth.
            bw = term.bitwidth if term.bitwidth > 0 else 64
            op = mkOp(Kind.INT_TO_BITVECTOR, bw)
            res = mkTerm(op, *child_terms)
        elif op_kind in (Kind.BITVECTOR_UBV_TO_INT, Kind.BITVECTOR_SBV_TO_INT):
            # Unary BV→Int conversion. If child is already Int (integer encoding
            # deserialized it), just return the child directly.
            c = child_terms[0]
            if c.getSort() == solver.getIntegerSort():
                res = c
            else:
                res = mkTerm(op_kind, c)
        elif op_kind == Kind.SET_INSERT:
            _set = child_terms[-1]
            elems = child_terms[:-1]
            res = mkTerm(Kind.SET_INSERT, *elems, _set)
        else:
            try:
                res = mkTerm(op_kind, *child_terms)
            except RuntimeError:
                # Sort mismatch (e.g. BV op on integer children) — cast children
                # to match. In integer encoding, BV ops become integer ops.
                if _INTEGER_ENCODING:
                    _int_kind_map = {
                        Kind.BITVECTOR_ADD: Kind.ADD, Kind.BITVECTOR_SUB: Kind.SUB,
                        Kind.BITVECTOR_MULT: Kind.MULT,
                        Kind.BITVECTOR_SGE: Kind.GEQ, Kind.BITVECTOR_SGT: Kind.GT,
                        Kind.BITVECTOR_SLE: Kind.LEQ, Kind.BITVECTOR_SLT: Kind.LT,
                        Kind.BITVECTOR_UGE: Kind.GEQ, Kind.BITVECTOR_UGT: Kind.GT,
                        Kind.BITVECTOR_ULE: Kind.LEQ, Kind.BITVECTOR_ULT: Kind.LT,
                        Kind.BITVECTOR_SDIV: Kind.INTS_DIVISION,
                        Kind.BITVECTOR_UDIV: Kind.INTS_DIVISION,
                        Kind.BITVECTOR_SREM: Kind.INTS_MODULUS,
                        Kind.BITVECTOR_UREM: Kind.INTS_MODULUS,
                        Kind.BITVECTOR_AND: Kind.MULT,
                        Kind.BITVECTOR_OR: Kind.ADD,
                        Kind.BITVECTOR_XOR: Kind.SUB,
                    }
                    int_kind = _int_kind_map.get(op_kind)
                    if int_kind:
                        # Cast non-integer children to integer
                        fixed = []
                        for c in child_terms:
                            if c.getSort().isBoolean():
                                c = solver.mkTerm(Kind.ITE, c,
                                                  solver.mkInteger(1), solver.mkInteger(0))
                            elif c.getSort().isBitVector():
                                # BV constant → integer
                                if c.isBitVectorValue():
                                    c = solver.mkInteger(str(int(c.getBitVectorValue(), 2)))
                                else:
                                    # BV variable — recreate as integer
                                    try:
                                        name = str(c)
                                        c = solver.mkConst(solver.getIntegerSort(), name)
                                    except Exception:
                                        pass
                            fixed.append(c)
                        try:
                            res = mkTerm(int_kind, *fixed)
                        except RuntimeError:
                            import logging as _log2
                            _log2.warning("[DESER] int_kind=%s fixed_sorts=%s",
                                          int_kind, [f.getSort() for f in fixed])
                            raise
                    else:
                        raise
                else:
                    raise

        memo[term] = res

    # Write all newly resolved terms back to the global LRU cache
    for k, v in memo.items():
        try:
            global_cache[k] = v
        except (RuntimeError, KeyError) as e:
            import logging as _log
            _log.error("[DESER-CACHE] LRU corrupt: %s term=%r — replacing cache", type(e).__name__, k)
            # The cache's internal OrderedDict is inconsistent.
            # .clear() itself can crash (iterates the corrupt dict),
            # so replace the entire cache object instead.
            # (LRUCache is imported at module top; no local import
            # here — that would shadow the module-level name and
            # make ALL references in this function local, which
            # breaks the ``state_cache.cvc5_term_cache = LRUCache(...)``
            # initialization path earlier in the function.)
            new_cache = LRUCache(maxsize=global_cache.maxsize)
            state_cache.cvc5_term_cache = new_cache
            global_cache = new_cache
            global_cache[k] = v

    return memo[root_term]

@lru_cache(maxsize=20000)
def serialize_cvc5_term(term: Term):
    # Stack stores (term, next_child_index_to_visit)
    stack = [(term, 0)]
    done = {}
    
    # Cache methods for inner-loop speed
    stack_pop = stack.pop
    stack_append = stack.append
    
    while stack:
        # Peek at the current node and the index of the next child to visit
        parent, idx = stack[-1]

        # 1. DAG OPTIMIZATION: If we visited this node via another path, skip it.
        if parent in done:
            stack_pop()
            continue

        kind = parent.getKind()

        # 2. LEAF HANDLING: Memoize and pop immediately
        if kind == Kind.CONSTANT or kind == Kind.VARIABLE or kind == Kind.CONST_BITVECTOR or kind == Kind.CONST_BOOLEAN or kind == Kind.CONST_INTEGER:
            done[parent] = HollowCvc5Term(parent)
            stack_pop()
            continue

        n = parent.getNumChildren()

        # 3. CHILD TRAVERSAL: Advance idx until we find a child that isn't done
        while idx < n:
            child = parent[idx]
            if child not in done:
                # Found a missing child. Save our place (idx) and push the child.
                stack[-1] = (parent, idx)
                stack_append((child, 0))
                break
            idx += 1
        else:
            # 4. BUILD: The "else" block runs only if the loop finished (idx == n),
            # meaning all children are processed and in `done`.
            stack_pop()
            
            h_term = HollowCvc5Term(parent)
            # Fast list comprehension to link children
            h_term.children = [done[parent[i]] for i in range(n)]
            done[parent] = h_term

    return done[term]

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

class HollowCvc5Term:
    __slots__ = ('op', 'type', 'var_name', 'bitwidth', 'value',
                 'bv_extract_start', 'bv_extract_end',
                 'bv_sign_extend_bitwidth', 'bv_zero_extend_bitwidth',
                 'array_element_width', 'array_index_width',
                 'children', '_hash')

    def __init__(self, term: Term):
        self.var_name = ""
        self.bitwidth = 0
        self.bv_extract_start = 0
        self.bv_extract_end = 0
        self.bv_sign_extend_bitwidth = 0
        self.bv_zero_extend_bitwidth = 0
        self.value = 0
        self.array_element_width = 0
        self.array_index_width = 0
        self._hash = None

        kind = term.getKind()
        if kind == Kind.CONSTANT or kind == Kind.VARIABLE:
            self.var_name = f"{term}"
        elif kind == Kind.CONST_BITVECTOR:
            self.value = int(term.getBitVectorValue(10))
        elif kind == Kind.CONST_INTEGER:
            self.value = int(term.getIntegerValue())
        elif kind == Kind.CONST_BOOLEAN:
            self.value = bool(term.getBooleanValue())
        elif kind == Kind.BITVECTOR_EXTRACT:
            bitvector_op = term.getOp()
            self.bv_extract_start = bitvector_op[1].getIntegerValue()
            self.bv_extract_end = bitvector_op[0].getIntegerValue()
        elif kind == Kind.BITVECTOR_SIGN_EXTEND:
            bitvector_op = term.getOp()
            self.bv_sign_extend_bitwidth = bitvector_op[0].getIntegerValue()
        elif kind == Kind.BITVECTOR_ZERO_EXTEND:
            bitvector_op = term.getOp()
            self.bv_zero_extend_bitwidth = bitvector_op[0].getIntegerValue()
        elif kind == Kind.SET_EMPTY:
            sort = term.getSort()
            self.bitwidth = sort.getSetElementSort().getBitVectorSize()

        sort = term.getSort()
        sort_kind = sort.getKind()
        assert sort_kind in sort_to_num, f"Unknown sort: {term} {sort} {sort_kind}"
        self.type = sort_to_num[sort_kind]
        if sort.isBitVector():
            self.bitwidth = sort.getBitVectorSize()
        if sort.isArray():
            elem_sort = sort.getArrayElementSort()
            idx_sort = sort.getArrayIndexSort()
            self.array_element_width = elem_sort.getBitVectorSize() if elem_sort.isBitVector() else 0
            self.array_index_width = idx_sort.getBitVectorSize() if idx_sort.isBitVector() else 0

        self.op = KIND_TO_NUM[kind]
        self.children = []

    def __eq__(self, other):
        if not isinstance(other, HollowCvc5Term):
            return NotImplemented
        if self.op != other.op:
            return False
        return (
            self.var_name                    == other.var_name                    and
            self.bitwidth                    == other.bitwidth                    and
            self.value                       == other.value                       and
            self.bv_extract_start            == other.bv_extract_start            and
            self.bv_extract_end              == other.bv_extract_end              and
            self.bv_sign_extend_bitwidth     == other.bv_sign_extend_bitwidth     and
            self.bv_zero_extend_bitwidth     == other.bv_zero_extend_bitwidth     and
            self.array_element_width         == other.array_element_width         and
            self.array_index_width           == other.array_index_width           and
            self.children                    == other.children
        )

    def __hash__(self):
        h = self._hash
        if h is None:
            h = hash((
                self.op,
                self.var_name,
                self.bitwidth,
                self.bv_extract_start,
                self.bv_extract_end,
                self.bv_sign_extend_bitwidth,
                self.bv_zero_extend_bitwidth,
                self.value,
                self.array_element_width,
                self.array_index_width,
                tuple(self.children),
            ))
            self._hash = h
        return h

    def __repr__(self):
        return f"HollowCvc5Term(op={self.op}, children={self.children} var_name={self.var_name})"

    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__ if slot != '_hash'}

    def __setstate__(self, state):
        for slot, val in state.items():
            object.__setattr__(self, slot, val)
        object.__setattr__(self, '_hash', None)

def deserialize_state_key(state_cache, state_key):
    from src.state.proof_obligation import ProofObligation
    from src.abduction.wp_depth import restore_or_record_wp_depth, wp_depth_of
    pc, predicate = pickle.loads(state_key)
    predicate.predicate = deserialize_cvc5_term(state_cache, predicate.predicate)
    restore_or_record_wp_depth(
        state_cache,
        pc,
        predicate,
        default=wp_depth_of(predicate),
        source="deserialize_state_key",
    )
    return ProofObligation(pc, predicate)


_HOLLOW_INFIX = {
    0: "==", 1: "!=",
    5: "+", 6: "*", 7: "-", 8: "/",
    10: "<=", 11: ">=", 12: "<", 13: ">",
    14: "=>",
    18: "+", 19: "*", 22: "<u", 25: "&", 26: ">>u", 27: "|",
    28: "<<", 29: "^", 30: "-",
    33: "<s", 34: "srem", 35: "<=u", 36: ">=s", 37: ">=u",
    39: "/u", 40: ">u", 41: ">s", 42: ">>a", 43: "%u",
    44: "<=s", 45: "/s",
    54: "%i", 55: "/i", 57: "%it", 58: "/it", 60: "/t",
}
_HOLLOW_PREFIX = {4: "!", 32: "~", 38: "-", 56: "-"}


def hollow_to_str(term, max_depth=8):
    """Pretty-print a HollowCvc5Term tree without cvc5 context."""
    if term is None:
        return "<None>"
    if max_depth <= 0:
        return "..."
    op = term.op
    # Leaves
    if op in (49, 67):  # CONSTANT, VARIABLE
        return term.var_name or f"<var_op={op}>"
    if op == 47:  # CONST_BITVECTOR
        return f"{term.value}bv{term.bitwidth}"
    if op == 53:  # CONST_INTEGER
        return str(term.value)
    if op == 48:  # CONST_BOOLEAN
        return "true" if term.value else "false"
    if op == 52:  # SET_EMPTY
        return "{}"

    nd = max_depth - 1
    kids = [hollow_to_str(c, nd) for c in term.children]

    if op == 15 and len(kids) == 2:  # SELECT
        return f"{kids[0]}[{kids[1]}]"
    if op == 16 and len(kids) == 3:  # STORE
        return f"store({kids[0]}, {kids[1]}, {kids[2]})"
    if op == 9 and len(kids) == 3:  # ITE
        return f"ite({kids[0]}, {kids[1]}, {kids[2]})"
    if op == 23 and len(kids) == 1:  # BITVECTOR_EXTRACT
        return f"({kids[0]})[{term.bv_extract_end}:{term.bv_extract_start}]"
    if op == 21 and len(kids) == 1:  # SIGN_EXTEND
        return f"sext{term.bv_sign_extend_bitwidth}({kids[0]})"
    if op == 24 and len(kids) == 1:  # ZERO_EXTEND
        return f"zext{term.bv_zero_extend_bitwidth}({kids[0]})"
    if op == 20 and len(kids) == 1:  # INT_TO_BITVECTOR
        return f"int2bv({kids[0]})"
    if op == 61 and len(kids) == 1:  # TO_INTEGER
        return f"to_int({kids[0]})"
    if op == 62 and len(kids) == 1:  # IS_INTEGER
        return f"is_int({kids[0]})"
    if op == 63 and len(kids) == 1:  # BITVECTOR_UBV_TO_INT
        return f"bv2nat({kids[0]})"
    if op == 64 and len(kids) == 1:  # BITVECTOR_SBV_TO_INT
        return f"sbv2int({kids[0]})"
    if op == 31:  # BITVECTOR_CONCAT
        return "concat(" + ", ".join(kids) + ")"
    if op == 17 and kids:  # APPLY_UF
        return f"{kids[0]}({', '.join(kids[1:])})"
    if op == 2:  # AND
        return "(" + " && ".join(kids) + ")" if len(kids) > 1 else (kids[0] if kids else "true")
    if op == 3:  # OR
        return "(" + " || ".join(kids) + ")" if len(kids) > 1 else (kids[0] if kids else "false")
    if op == 46:  # XOR
        return "(" + " xor ".join(kids) + ")"
    if op == 50 and len(kids) == 2:  # SET_MEMBER
        return f"({kids[0]} in {kids[1]})"
    if op == 51:  # SET_INSERT
        return "insert(" + ", ".join(kids) + ")"
    if op == 65 and len(kids) >= 2:  # FORALL
        return f"forall({kids[0]}. {kids[1]})"
    if op == 66:  # VARIABLE_LIST
        return "[" + ", ".join(kids) + "]"

    if op in _HOLLOW_PREFIX and len(kids) == 1:
        return f"{_HOLLOW_PREFIX[op]}({kids[0]})"
    if op in _HOLLOW_INFIX and len(kids) == 2:
        return f"({kids[0]} {_HOLLOW_INFIX[op]} {kids[1]})"

    return f"<op={op}>({', '.join(kids)})"


def classify_hollow(predicate):
    """Classify a pickle-loaded Predicate by origin / structural type.

    Mirrors StateIterator._classify_predicate but operates on the HollowCvc5Term
    form so it works without a live cvc5 solver context.
    """
    origin = getattr(predicate, "origin", None)
    if origin in ("reach", "custom"):
        return origin

    hollow = predicate.predicate
    if hollow is None:
        return "non-mem"

    if getattr(predicate, "eq_predicate", False) and hollow.op == 0 and len(hollow.children) == 2:
        l, r = hollow.children
        if l.op == 15 and r.op == 15:
            return "mem-partial"
        if l.type == 4 and r.type == 4:
            return "mem-full"

    if getattr(predicate, "eq_const_predicate", False):
        def _has_select(t, d=8):
            if t is None or d == 0:
                return False
            if t.op == 15:
                return True
            return any(_has_select(c, d - 1) for c in t.children)
        if _has_select(hollow):
            return "mem-partial"

    return "non-mem"


# HollowCvc5Term v2: keep the legacy public names while delegating the
# persistence representation to the versioned flat DAG serializer.
from interpreter.utils.cvc5_serde import (  # noqa: E402
    Cvc5SerdeError,
    HollowCvc5Term,
    SerializedCvc5TermV2,
    canonical_term_bytes,
    canonical_term_fingerprint,
    classify_hollow,
    deserialize_cvc5_term,
    get_serde_stats,
    hollow_to_str,
    reset_serde_stats,
    serialize_cvc5_term,
)


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
        # In integer encoding, ITE(cond, 1, 0) is how comparisons return int.
        # For Boogie output, simplify back to the condition (bool in Boogie).
        t, f = cvc5_term[1], cvc5_term[2]
        if _INTEGER_ENCODING:
            t_is_1 = (t.getKind() == Kind.CONST_INTEGER and int(t.getIntegerValue()) == 1)
            f_is_0 = (f.getKind() == Kind.CONST_INTEGER and int(f.getIntegerValue()) == 0)
            t_is_0 = (t.getKind() == Kind.CONST_INTEGER and int(t.getIntegerValue()) == 0)
            f_is_1 = (f.getKind() == Kind.CONST_INTEGER and int(f.getIntegerValue()) == 1)
            if t_is_1 and f_is_0:
                return to_boogie(cvc5_term[0])  # ITE(cond, 1, 0) → cond
            if t_is_0 and f_is_1:
                return LogicalNegation(expression=to_boogie(cvc5_term[0]))  # ITE(cond, 0, 1) → !cond
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

    # --- TYPE 4: Disjunction ---
    if r"\/" in body:
        parts = body.split(r"\/")
        first_match = re.search(r"\((.*?)\)\s*==\s*\((.*?)\)", parts[0])
        
        if first_match:
            result["variable"] = first_match.group(1).strip()
            result["type"] = "disjunction_of_constants"
            constants = []
            for p in parts:
                c_match = re.search(r"==\s*\(\s*(\d+)\s*\)", p)
                if c_match:
                    constants.append(int(c_match.group(1)))
            result["allowed_values"] = constants
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
        if is_digit_rhs and "[" not in lhs:
            result["type"] = "constant_comparison"
            result["variable"] = lhs
            result["constant"] = int(rhs)
            return result

    result["type"] = "unknown"
    result["raw_body"] = body
    return result

def _parse_infix_expr(s, state_cache):
    """Parse an infix expression (from term_to_string output) into a cvc5 Term.

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

    # Logical (Bool-theory) operators — agent-authored invariants
    # commonly use ``||``, ``&&``, and Boogie's ``==>`` to combine
    # comparison predicates.  When both operands are Bool we route
    # through the Bool-theory Kind instead of the BV one.
    _BOOL_OPS = {
        '||': Kind.OR,
        '&&': Kind.AND,
        '==>': Kind.IMPLIES,
    }

    # Operator precedence (lower = binds tighter, but all >= 0 so the
    # ``min_prec=0`` parser entry catches every op).  Logical
    # connectives are the loosest so ``a < b || c <= d`` parses as
    # ``(a < b) || (c <= d)``; ``==>`` is looser than ``||`` / ``&&``.
    _PREC = {
        '==>': 0,
        '||': 1, '&&': 1,
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
        elif s[i] == '$' or (s[i].isalpha() and s[i:i+5] == 'STORE'):
            # Variable or STORE keyword.  SMACK-emitted names embed
            # additional ``$`` segments (e.g.
            # ``$free_105_inline$__VERIFIER_nondet_int$0$$i0`` for a
            # loop-snapshot of an inlined nondet-int call), so the
            # tokenizer must accept ``$`` inside an identifier after
            # the leading ``$``.  Without this, the second ``$`` ends
            # the token early and the parser sees an "Unknown variable"
            # for the truncated prefix.
            j = i
            if s[i] == '$':
                j += 1
                while j < len(s) and (s[j].isalnum() or s[j] in '._$'):
                    j += 1
            else:
                j = i + 5
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
        elif s[i:i+2] in ('>>', '<<', '<=', '>=', '==', '!=', '||', '&&'):
            tokens.append(('OP', s[i:i+2]))
            i += 2
        elif s[i] in '+-*/%<>&^~':
            tokens.append(('OP', s[i]))
            i += 1
        elif s[i] in '()[],':
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

    def _bool_to_bv1(term):
        """Convert Bool term to bv1: ITE(term, 1, 0)."""
        return solver.mkTerm(Kind.ITE, term,
                             solver.mkBitVector(1, 1),
                             solver.mkBitVector(1, 0))

    # Integer-sort fallback map: when both operands are Int (not BV) the
    # cvc5 BITVECTOR_* operators reject the term with "expecting a
    # bit-vector term".  Swap to the integer-theory equivalents.
    _INT_OPS = {
        '+': Kind.ADD, '-': Kind.SUB, '*': Kind.MULT,
        '<': Kind.LT, '>': Kind.GT, '<=': Kind.LEQ, '>=': Kind.GEQ,
        '==': Kind.EQUAL, '!=': Kind.DISTINCT,
    }

    def _unwrap_bool_or_bv1(t):
        """If ``t`` is Bool, return it.  If it's the ``_bool_to_bv1``
        ITE wrapper (``(ite cmp #b1 #b0)``), pull the Bool cmp out.
        Otherwise return None — caller decides how to handle it.

        The bv1 ITE wrapping was introduced so comparison results
        could participate in BV bitwise expressions.  For logical
        connectives (``||``, ``&&``, ``==>``) we need the Bool back.
        """
        try:
            if t.getSort().isBoolean():
                return t
            if (t.getKind() == Kind.ITE
                    and t.getNumChildren() == 3
                    and t[1].isBitVectorValue()
                    and int(t[1].getBitVectorValue(10)) == 1
                    and t[2].isBitVectorValue()
                    and int(t[2].getBitVectorValue(10)) == 0
                    and t[0].getSort().isBoolean()):
                return t[0]
        except Exception:
            pass
        return None

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
        try:
            if t.getSort().isInteger():
                return t
            if t.isBitVectorValue():
                width = t.getSort().getBitVectorSize()
                raw_val = int(t.getBitVectorValue(10))
                # Sign-extend: if the high bit is set, interpret as
                # negative two's-complement.
                if raw_val >= (1 << (width - 1)):
                    raw_val -= (1 << width)
                return solver.mkInteger(raw_val)
        except Exception:
            pass
        return t

    def _either_int(a, b):
        try:
            return a.getSort().isInteger() or b.getSort().isInteger()
        except Exception:
            return False

    def parse_expr(min_prec=0):
        left = parse_primary()
        while True:
            typ, val = peek()
            if typ != 'OP' or val not in _PREC or _PREC[val] < min_prec:
                break
            advance()
            right = parse_expr(_PREC[val] + 1)
            # Logical connective — dispatch to Kind.OR / Kind.AND /
            # Kind.IMPLIES instead of the BV bitwise operators.  Agent-
            # authored disjunctive invariants (``$i0 < $u0 || $u0 <= 0``)
            # land here.  Both operands are already Bool when they came
            # from the int-sort path, but bv-sort comparisons were
            # wrapped as bv1 ITE for backward compat — unwrap those
            # before applying the logical op so the result is a proper
            # Bool predicate.
            if val in _BOOL_OPS:
                try:
                    l_b = _unwrap_bool_or_bv1(left)
                    r_b = _unwrap_bool_or_bv1(right)
                    if (l_b is not None and r_b is not None
                            and l_b.getSort().isBoolean()
                            and r_b.getSort().isBoolean()):
                        result = solver.mkTerm(_BOOL_OPS[val], l_b, r_b)
                        left = result
                        continue
                except Exception:
                    pass
                # Can't coerce to Bool — raise a specific error
                # rather than falling through to the BV bitwise path
                # (which would silently produce a wrong term).
                raise RuntimeError(
                    f"logical operator {val!r} requires Bool operands "
                    f"(got sorts {left.getSort()} and {right.getSort()})")

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
                # Comparisons return Bool; wrap as bv1 for use in BV
                # expressions.
                if val in ('<', '>', '<=', '>=') and result.getSort().isBoolean():
                    result = _bool_to_bv1(result)
            left = result
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
            # Default 32-bit bitvector — matches the SMACK-emitted
            # scalar width.  _match_bv_sorts will resize on sort
            # mismatch when the other operand is BV64 etc.
            return solver.mkBitVector(32, int(val))
        elif typ == 'OP' and val == '~':
            # Boogie-style negation: ``~(expr)``.  The operand's sort
            # decides whether to emit a Bool NOT or a BV NOT.
            # Predicates like ``~($i1 >= 0)`` land here with a Bool
            # operand (after fix #3 unwrapped the top-level
            # comparison); integer terms would use Kind.NEG but we
            # don't expect those in Swoosh-emitted invariants.
            advance()
            operand = parse_primary()
            try:
                if operand.getSort().isBoolean():
                    return solver.mkTerm(Kind.NOT, operand)
            except Exception:
                pass
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

    result = parse_expr(0)
    # Top-level unwrap: if the whole expression is a comparison, it's
    # been wrapped as ``(ite cmp #b1 #b0)`` by ``_bool_to_bv1`` for
    # use in nested BV arithmetic.  At the top level we want the
    # Bool predicate itself so ``Predicate(cvc5_term)`` builds a
    # valid assertable term.  Detect the shape and unwrap.
    try:
        if (result.getKind() == Kind.ITE
                and result.getNumChildren() == 3
                and result[1].isBitVectorValue()
                and int(result[1].getBitVectorValue(10)) == 1
                and result[2].isBitVectorValue()
                and int(result[2].getBitVectorValue(10)) == 0
                and result[0].getSort().isBoolean()):
            result = result[0]
    except Exception:
        pass
    return result


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
                ret_term.eq_predicate = True
            elif rhs.isBitVectorValue() or lhs.isBitVectorValue():
                # eq_const predicate
                ret_term = Predicate(cvc5_term)
                ret_term.eq_const_predicate = True
            else:
                ret_term = Predicate(cvc5_term)
                ret_term.eq_predicate = True
        else:
            ret_term = Predicate(cvc5_term)
        return ProofObligation(pc, ret_term)

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
            ret_term.eq_predicate = True
        
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
        ret_term.eq_predicate = True

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
        if cvc5_term.getKind() == Kind.EQUAL:
            ret_term = Predicate(cvc5_term)
            ret_term.eq_predicate = True
        else:
            ret_term = Predicate(cvc5_term)

    else:
        raise ValueError(f"Unsupported constraint type '{x['type']}' for: {str_key}")

    return ProofObligation(pc, ret_term)
