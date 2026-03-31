from cvc5 import Kind, Term, Sort, Solver, SortKind
import math
from interpreter.parser.expression import FunctionApplication, MapSelect, StorageIdentifier, ProcedureIdentifier, BinaryExpression, UnaryExpression, BooleanLiteral, IntegerLiteral, OldExpression, LogicalNegation, QuantifiedExpression, IfExpression, Identifier
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

def generate_cvc5_function_map(solver: Solver):
    """
    Returns a lazily initialized more_func_map using the provided stateful solver.
    """
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
    } | fn_to_cvc5_op

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
    if expr.getSort() == solver.getBooleanSort():
        # Create terms for 0 and 1
        zero = solver.mkInteger(0)
        one = solver.mkInteger(1)
        # Use an ITE (if-then-else) term to cast the boolean to an integer
        return solver.mkTerm(Kind.ITE, expr, one, zero)
    else:
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
    
def convert_type_to_cvc5(solver, type_, mono_mem = True) -> Sort:
    if isinstance(type_, BooleanType):
        return solver.getBooleanSort()
    elif isinstance(type_, IntegerType):
        return solver.getIntegerSort()
    elif isinstance(type_, CustomType):
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
            elementSort = solver.mkBitVectorSort(64)
        else:
            # For all memory maps, we assume 64-bit elements
            elementSort = convert_type_to_cvc5(solver, type_.range)
            

        if len(domain) == 1:
            return solver.mkArraySort(domain[0], elementSort)
        else:
            assert False
    assert False, f"unknown type {type_} {type(type_)}"

def convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr, mono_mem: bool) -> Term:
    if isinstance(expr, StorageIdentifier) or isinstance(expr, ProcedureIdentifier):
        return state_cache.cvc5_var(expr.name)
    elif isinstance(expr, BinaryExpression):
        if expr.op in cvc5_fn_map:
            lhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.lhs, mono_mem)
            rhs = convert_expr_cvc5(cvc5_fn_map, state_cache, solver, expr.rhs, mono_mem)
            if rhs.isBitVectorValue():
                rhs = sign_extend(solver, rhs, lhs.getSort().getBitVectorSize())
            cvc5_op, _, op_type, out_type = cvc5_fn_map[expr.op]
            if op_type == bool:
                lhs = cvc5_cast_to_bool(solver, lhs)
                rhs = cvc5_cast_to_bool(solver, rhs)

            lhs, rhs = assign_fix_type(solver, lhs, rhs)
            return solver.mkTerm(cvc5_op, lhs, rhs)
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
        for v in expr.variables:
            assert len(v.names) == 1
            var_cvc5 = solver.mkVar(convert_type_to_cvc5(solver, v.type), v.names[0])
            state_cache.cached_id_to_cvc5[v.names[0]] = var_cvc5
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
            cvc5_op, _, op_bit_width, output_type = cvc5_fn_map[expr.function.name]
            arg_exprs = [convert_expr_cvc5(cvc5_fn_map, state_cache, solver, arg, mono_mem) for arg in expr.arguments]
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
            return solver.mkTerm(Kind.NOT, unary_expr)
        else:
            assert False, f"unsupported unary operator {expr.op}"
    elif isinstance(expr, IntegerLiteral):
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
        # Clear cached hash so it's recomputed from the new cvc5 Term
        predicate._cached_hash = None

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
                elif sort_kind == SortKind.ARRAY_SORT:
                    idx_sort = solver.mkBitVectorSort(term.array_index_width)
                    elem_sort = solver.mkBitVectorSort(term.array_element_width)
                    res = mkConst(solver.mkArraySort(idx_sort, elem_sort), term.var_name)
                else:
                    raise ValueError(f"Unknown sort: {sort_kind}")
                from AbductUtils import put_cvc5_var
                put_cvc5_var(state_cache.redis, term.var_name, res)
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
        elif op_kind == Kind.SET_INSERT:
            _set = child_terms[-1]
            elems = child_terms[:-1]
            res = mkTerm(Kind.SET_INSERT, *elems, _set)
        else:
            res = mkTerm(op_kind, *child_terms)

        memo[term] = res

    # Write all newly resolved terms back to the global LRU cache
    for k, v in memo.items():
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
        if kind == Kind.CONSTANT or kind == Kind.CONST_BITVECTOR or kind == Kind.CONST_BOOLEAN:
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
        if kind == Kind.CONSTANT:
            self.var_name = f"{term}"
        elif kind == Kind.CONST_BITVECTOR:
            self.value = int(term.getBitVectorValue(10))
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
            self.array_element_width = sort.getArrayElementSort().getBitVectorSize()
            self.array_index_width = sort.getArrayIndexSort().getBitVectorSize()

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
    inter_key = pickle.loads(state_key)
    inter_key[1].predicate = deserialize_cvc5_term(state_cache, inter_key[1].predicate)
    return inter_key

def to_boogie(cvc5_term: Term):
    if cvc5_term.getKind() == Kind.CONSTANT:
        return StorageIdentifier(name=cvc5_term.getSymbol())
    elif cvc5_term.getKind() == Kind.CONST_BOOLEAN:
        value = bool(cvc5_term.getBooleanValue())
        return BooleanLiteral(value)
    elif cvc5_term.getKind() == Kind.CONST_BITVECTOR:
        assert cvc5_term.isBitVectorValue(), f"Constant is not a bitvector: {cvc5_term}"
        value = int(cvc5_term.getBitVectorValue(), 2)
        return IntegerLiteral(value)
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
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
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
        return LogicalNegation(expression=child)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SGT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SGE:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op=">=", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_ULT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<", rhs=rhs)
    elif cvc5_term.getKind() == Kind.BITVECTOR_SLT:
        lhs = to_boogie(cvc5_term[0])
        rhs = to_boogie(cvc5_term[1])
        return BinaryExpression(lhs=lhs, op="<", rhs=rhs)
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
    from predicate import Predicate
    if len(predicates) == 0:
        return Predicate(solver.mkTrue())
    elif len(predicates) == 1:
        return predicates[0]
    else:
        return Predicate(solver.mkTerm(Kind.OR, *[predicate.predicate for predicate in predicates]))

def conjunct(solver, predicates):
    from predicate import Predicate
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

    # Operator precedence (lower = binds tighter)
    _PREC = {
        '==': 1, '!=': 1,
        '<': 2, '>': 2, '<=': 2, '>=': 2,
        '+': 3, '-': 3,
        '*': 4, '/': 4, '%': 4,
        '>>': 5, '<<': 5,
        '&': 6, '^': 6, '||': 6,
    }

    solver = state_cache.solver
    tokens = []
    i = 0
    s = s.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] == '$' or (s[i].isalpha() and s[i:i+5] == 'STORE'):
            # Variable or STORE keyword
            j = i
            if s[i] == '$':
                j += 1
                while j < len(s) and (s[j].isalnum() or s[j] in '._'):
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
        elif s[i:i+2] in ('>>', '<<', '<=', '>=', '==', '!=', '||'):
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

    def parse_expr(min_prec=0):
        left = parse_primary()
        while True:
            typ, val = peek()
            if typ != 'OP' or val not in _PREC or _PREC[val] < min_prec:
                break
            advance()
            right = parse_expr(_PREC[val] + 1)
            kind = _BIN_OPS[val]
            # Ensure operands match sort (including for equality)
            left, right = _match_bv_sorts(solver, left, right)
            result = solver.mkTerm(kind, left, right)
            # Comparisons return Bool; wrap as bv1 for use in BV expressions
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
            # Default 64-bit bitvector; caller may need to adjust
            return solver.mkBitVector(64, int(val))
        elif typ == 'OP' and val == '~':
            advance()
            operand = parse_primary()
            return solver.mkTerm(Kind.BITVECTOR_NOT, operand)
        else:
            raise ValueError(f"Unexpected token: {typ}={val} at pos {pos[0]}")

    def _match_bv_sorts(solver, a, b):
        """Ensure two bitvector terms have matching widths via zero-extension."""
        try:
            as_ = a.getSort()
            bs_ = b.getSort()
            if not as_.isBitVector() or not bs_.isBitVector():
                return a, b
            aw = as_.getBitVectorSize()
            bw = bs_.getBitVectorSize()
            if aw != bw:
                if a.isBitVectorValue():
                    a = solver.mkBitVector(bw, int(a.getBitVectorValue(10)))
                elif b.isBitVectorValue():
                    b = solver.mkBitVector(aw, int(b.getBitVectorValue(10)))
                elif aw < bw:
                    op = solver.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, bw - aw)
                    a = solver.solver.mkTerm(op, a)
                elif bw < aw:
                    op = solver.solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, aw - bw)
                    b = solver.solver.mkTerm(op, b)
        except Exception:
            pass
        return a, b

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
    from predicate import Predicate
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
        return (pc, ret_term)

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

    state_key = (pc, ret_term)
    return state_key