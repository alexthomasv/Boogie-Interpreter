from cvc5 import Kind, Term
from itertools import product

ADDR_DISTINCT_PREDICATE = "Addr Distinct"
VAL_EQ_PREDICATE = "Val EQ"

# Map cvc5 Kinds to their human-readable Python-like equivalents
KIND_TRANSLATIONS = {
    Kind.EQUAL: "==",
    Kind.DISTINCT: "!=",
    Kind.AND: "/\\",
    Kind.OR: "\\/",
    Kind.NOT: "not",
    Kind.ADD: "+",
    Kind.MULT: "*",
    Kind.SUB: "-",
    Kind.DIVISION: "/",
    Kind.ITE: "if",
    Kind.LEQ: "<=",
    Kind.GEQ: ">=",
    Kind.LT: "<",
    Kind.GT: ">",
    Kind.IMPLIES: "=>",  # Implication
    Kind.SELECT: "select",  # Array/map access (a[i])
    Kind.STORE: "store",  # Array/map update (a[i] = value)
    Kind.APPLY_UF: "apply_uf",  # Apply function (f(x))
    Kind.BITVECTOR_ADD: "+",
    Kind.BITVECTOR_MULT: "*",
    Kind.INT_TO_BITVECTOR: "int_to_bitvector",
    Kind.BITVECTOR_SIGN_EXTEND: "sign_extend",
    Kind.BITVECTOR_ULT: "<",
    Kind.BITVECTOR_EXTRACT: "extract",
    Kind.BITVECTOR_ZERO_EXTEND: "zero_extend",
    Kind.BITVECTOR_AND: "&",
    Kind.BITVECTOR_LSHR: ">>",
    Kind.BITVECTOR_OR: "||",
    Kind.BITVECTOR_SHL: "<<",
    Kind.BITVECTOR_XOR: "^",
    Kind.BITVECTOR_SUB: "-",
    Kind.BITVECTOR_CONCAT: "++",  # Bit concatenation (NOT bitwise OR)
    Kind.BITVECTOR_NOT: "~",
    Kind.BITVECTOR_SLT: "<",
    Kind.BITVECTOR_SREM: "%",
    Kind.BITVECTOR_ULE: "<=",
    Kind.BITVECTOR_SGE: ">=",
    Kind.BITVECTOR_UGE: ">=",
    Kind.BITVECTOR_NEG: "-",
    Kind.BITVECTOR_UDIV: "/",
    Kind.BITVECTOR_UGT: ">",
    Kind.BITVECTOR_SGT: ">",
    Kind.BITVECTOR_ASHR: ">>",
    Kind.BITVECTOR_UREM: "%",
    Kind.BITVECTOR_SLE: "<=",
    Kind.BITVECTOR_SDIV: "/",
    Kind.XOR: "^",
}

def get_canonical_name(expr):
    # Retrieves the actual variable name (as seen in the Boogie code) from the cvc5 term
    assert isinstance(expr, Term)
    assert expr.getKind() == Kind.CONSTANT or expr.getKind() == Kind.VARIABLE, f"expr is not a constant/variable: {expr} {expr.getKind()}"
    if expr.getSymbol().startswith("$M."):
        return expr.getSymbol()
    elif expr.getSymbol().endswith(".shadow"):
        return f"{expr.getSymbol().split('.')[-2]}.shadow"
    else:
        name_split = expr.getSymbol().split('.')
        if "cross_product" in name_split:
            cross_product_index = name_split.index("cross_product")
            return ".".join(name_split[cross_product_index + 1:])
        else:
            return expr.getSymbol() 

def sign_extend(solver, term, target_bit_width):
        bit_width = term.getSort().getBitVectorSize()
        if target_bit_width == bit_width:
            return term
        elif target_bit_width > bit_width:
            sign_extend_op = solver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, target_bit_width - bit_width)
            return solver.mkTerm(sign_extend_op, term)
        else:
            extract_op = solver.mkOp(Kind.BITVECTOR_EXTRACT, target_bit_width - 1, 0)
            return solver.mkTerm(extract_op, term)

def zero_extend(solver, term, target_bit_width):
    bit_width = term.getSort().getBitVectorSize()
    if target_bit_width == bit_width:
        return term
    elif target_bit_width > bit_width:
        sign_extend_op = solver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, target_bit_width - bit_width)
        return solver.mkTerm(sign_extend_op, term)
    else:
        assert False

def term_to_string(term, id_to_cexpr=None, depth=0, indent=0):
    """
    Converts a cvc5 term into a Python-like string representation, minimizing parentheses,
    but always adding parentheses for ternary (ITE) operators.
    
    Args:
    - term (cvc5.Term): The cvc5 term to convert.
    - parent_kind (cvc5.Kind): The kind of the parent term to control when to add parentheses.

    Returns:
    - str: A Python-like string representation of the term.
    """
    prefix = "\t" * indent
    if term is None:
        return f"{prefix}EMPTY"

    if depth > 25:
        return f"{prefix}..."
    # Base case: if the term is a constant or variable, return it as a string
    if term.getNumChildren() == 0:
        if id_to_cexpr and term.__str__() in id_to_cexpr:
            return f"{prefix}{id_to_cexpr[term.__str__()]}"
        else:
            if term.isBitVectorValue():
                return f"{prefix}{term.getBitVectorValue(10)}"
            elif term.isBooleanValue():
                return f"{prefix}{term.getBooleanValue()}"
            elif term.isIntegerValue():
                return f"{prefix}{term.getIntegerValue()}"
            else:
                try:
                    return f"{prefix}{term.getSymbol()}"
                except Exception:
                    return f"{prefix}{term}"

    # Get the kind of the current term
    current_kind = term.getKind()

    # Handle specific cases for known operators
    if current_kind in KIND_TRANSLATIONS:
        operator = KIND_TRANSLATIONS[current_kind]

        if current_kind == Kind.AND:
            children = [term_to_string(term[i], id_to_cexpr, depth + 1, indent) for i in range(term.getNumChildren())]
            expression = f" {operator} ".join(children)
            return f"{expression}"

        if current_kind == Kind.NOT:
            expr = term_to_string(term[0], id_to_cexpr, depth + 1)
            return f"{prefix}~({expr})"

        if current_kind == Kind.BITVECTOR_NEG:
            expr = term_to_string(term[0], id_to_cexpr, depth + 1)
            return f"{prefix}(-{expr})"

        if current_kind == Kind.IMPLIES:
            lhs = term_to_string(term[0], id_to_cexpr, depth + 1, indent)
            rhs = term_to_string(term[1], id_to_cexpr, depth + 1, indent + 1)
            return f"{lhs} {operator} {rhs}"

        # Handle Ternary operator (ITE) - always use parentheses
        if current_kind == Kind.ITE:
            condition = term_to_string(term[0], id_to_cexpr, depth + 1)
            true_expr = term_to_string(term[1], id_to_cexpr, depth + 1)
            false_expr = term_to_string(term[2], id_to_cexpr, depth + 1)
            if true_expr == "1" and false_expr == "0":
                return f"{prefix}{condition}"
            else:
                return f"{prefix}({condition} ? {true_expr} : {false_expr})"

        # Handle Kind.SELECT (array or map access)
        if current_kind == Kind.SELECT:
            if term[0].getKind() == Kind.STORE:
                store_expr = term_to_string(term[0], id_to_cexpr, depth + 1)
                index = term_to_string(term[1], id_to_cexpr, depth + 1)
                return f"{prefix}{store_expr}[{index}]"
            else:
                array = term_to_string(term[0], id_to_cexpr, depth + 1)
                index = term_to_string(term[1], id_to_cexpr, depth + 1)
                return f"{prefix}{array}[{index}]"

        # Handle Kind.STORE (array or map update)
        if current_kind == Kind.STORE:
            array = term_to_string(term[0], id_to_cexpr, depth + 1)
            index = term_to_string(term[1], id_to_cexpr, depth + 1)
            value = term_to_string(term[2], id_to_cexpr, depth + 1)
            return f"{prefix}STORE({array}, {index}, {value})"
        
        if current_kind == Kind.EQUAL:
            lhs = term_to_string(term[0], id_to_cexpr, depth + 1)
            rhs = term_to_string(term[1], id_to_cexpr, depth + 1)
            return f"{prefix}({lhs}) == ({rhs})"

        # Special case: concat(extract(N-1,0,x), #b0...0) is x << K (i.e. x * 2^K)
        if current_kind == Kind.BITVECTOR_CONCAT:
            if term.getNumChildren() == 2:
                hi = term[0]
                lo = term[1]
                if lo.isBitVectorValue() and int(lo.getBitVectorValue(), 2) == 0:
                    n_zeros = lo.getSort().getBitVectorSize()
                    if hi.getKind() == Kind.BITVECTOR_EXTRACT:
                        inner = term_to_string(hi[0], id_to_cexpr, depth + 1, indent)
                        multiplier = 1 << n_zeros
                        return f"{prefix}{multiplier} * {inner}"

        # Binary/n-ary operators (ADD, MULT, AND, OR, etc.)
        children = [term_to_string(term[i], id_to_cexpr, depth + 1, indent) for i in range(term.getNumChildren())]
        expression = f" {operator} ".join(children)
        return expression
    else:
        if current_kind == Kind.SET_INSERT:
            ret_str = ""
            for x in term:
                if x.getKind() == Kind.SET_EMPTY:
                    break
                else:
                    ret_str += f"{term_to_string(x, id_to_cexpr, depth + 1, indent)} "
            ret_str = ret_str[:-1]
            return f"{prefix}({ret_str})"

        if current_kind == Kind.SET_MEMBER:
            var = term_to_string(term[0], id_to_cexpr, depth + 1)
            set_term = term_to_string(term[1], id_to_cexpr, depth + 1)
            return f"{prefix}({var} in {set_term})"
            
        if current_kind == Kind.LAMBDA:
            return term.__str__()
        if current_kind == Kind.BITVECTOR_TO_NAT:
            inner = term_to_string(term[0], id_to_cexpr, depth + 1)
            return f"nat({inner})"
        if current_kind == Kind.INT_TO_BITVECTOR:
            inner = term_to_string(term[0], id_to_cexpr, depth + 1)
            return f"bv({inner})"
        # Unknown kinds: use cvc5's built-in string representation
        return str(term)

