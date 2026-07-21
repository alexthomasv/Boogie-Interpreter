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
    Kind.NEG: "-",
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

_LOW_PRECEDENCE_ARITH = {
    Kind.ADD,
    Kind.SUB,
    Kind.BITVECTOR_ADD,
    Kind.BITVECTOR_SUB,
}

_MULTIPLICATIVE_KINDS = {
    Kind.MULT,
    Kind.DIVISION,
    Kind.BITVECTOR_MULT,
    Kind.BITVECTOR_UDIV,
    Kind.BITVECTOR_SDIV,
}


def _pretty_print_term_child(term, parent_kind, id_to_cexpr=None, depth=0, indent=0):
    rendered = pretty_print_term(term, id_to_cexpr, depth, indent)
    if term is None or term.getNumChildren() == 0:
        return rendered
    child_kind = term.getKind()
    if parent_kind in _MULTIPLICATIVE_KINDS and child_kind in _LOW_PRECEDENCE_ARITH:
        return f"({rendered})"
    if parent_kind in (Kind.SUB, Kind.BITVECTOR_SUB) and child_kind in _LOW_PRECEDENCE_ARITH:
        return f"({rendered})"
    if parent_kind in (Kind.NEG, Kind.BITVECTOR_NEG) and child_kind in _LOW_PRECEDENCE_ARITH:
        return f"({rendered})"
    return rendered


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

def pretty_print_term(term, id_to_cexpr=None, depth=0, indent=0):
    """DISPLAY ONLY — a human/LLM-facing pretty-print of a cvc5 term.

    **NEVER use this for serialization, identity, hashing, canonical ordering,
    or any value a machine re-consumes.** It is lossy and ambiguous on purpose
    (minimizes parens, renders `=>`/`/\\`, may not round-trip). For a value that
    crosses a process/wire boundary or keys state, use the cvc5 serialization
    (`serialize_cvc5_term` / `serialized_*_b64` + the `deserialize_*` helpers);
    for a parser-faithful Boogie rendering use `cvc5_to_boogie` /
    `cvc5_to_boogie_ast`; for a term-canonical identity use
    `canonical_term_fingerprint`.

    Args:
    - term (cvc5.Term): The cvc5 term to convert.

    Returns:
    - str: a pretty, display-only string (not a serialization).
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
            children = [pretty_print_term(term[i], id_to_cexpr, depth + 1, indent) for i in range(term.getNumChildren())]
            expression = f" {operator} ".join(children)
            return f"{expression}"

        if current_kind == Kind.NOT:
            expr = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            # ``~`` is BITWISE NOT in Boogie / SMACK; logical NOT is
            # ``!``.  Emitting ``~`` here produced strings like
            # ``~($i5 == 0)`` that cvc5's parser later rejected with
            # ``unknown function application: ~(($i5 == 0))`` —
            # ``$i5 == 0`` is Bool, ``~`` only applies to bit-vectors.
            # The bug surfaced as a ``target_parse_error`` blocking
            # check-llm-invariants on every transformed assertion.
            return f"{prefix}!({expr})"

        if current_kind in (Kind.NEG, Kind.BITVECTOR_NEG):
            expr = _pretty_print_term_child(
                term[0], current_kind, id_to_cexpr, depth + 1)
            return f"{prefix}(-{expr})"

        if current_kind == Kind.IMPLIES:
            lhs = pretty_print_term(term[0], id_to_cexpr, depth + 1, indent)
            rhs = pretty_print_term(term[1], id_to_cexpr, depth + 1, indent + 1)
            return f"{lhs} {operator} {rhs}"

        # Handle Ternary operator (ITE) - always use parentheses
        if current_kind == Kind.ITE:
            condition = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            true_expr = pretty_print_term(term[1], id_to_cexpr, depth + 1)
            false_expr = pretty_print_term(term[2], id_to_cexpr, depth + 1)
            if true_expr == "1" and false_expr == "0":
                return f"{prefix}{condition}"
            else:
                return f"{prefix}({condition} ? {true_expr} : {false_expr})"

        # Handle Kind.SELECT (array or map access)
        if current_kind == Kind.SELECT:
            if term[0].getKind() == Kind.STORE:
                store_expr = pretty_print_term(term[0], id_to_cexpr, depth + 1)
                index = pretty_print_term(term[1], id_to_cexpr, depth + 1)
                return f"{prefix}{store_expr}[{index}]"
            else:
                array = pretty_print_term(term[0], id_to_cexpr, depth + 1)
                index = pretty_print_term(term[1], id_to_cexpr, depth + 1)
                return f"{prefix}{array}[{index}]"

        # Handle Kind.STORE (array or map update)
        if current_kind == Kind.STORE:
            array = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            index = pretty_print_term(term[1], id_to_cexpr, depth + 1)
            value = pretty_print_term(term[2], id_to_cexpr, depth + 1)
            return f"{prefix}STORE({array}, {index}, {value})"
        
        if current_kind == Kind.EQUAL:
            lhs = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            rhs = pretty_print_term(term[1], id_to_cexpr, depth + 1)
            return f"{prefix}({lhs}) == ({rhs})"

        # Special case: concat(extract(N-1,0,x), #b0...0) is x << K (i.e. x * 2^K)
        if current_kind == Kind.BITVECTOR_CONCAT:
            if term.getNumChildren() == 2:
                hi = term[0]
                lo = term[1]
                if lo.isBitVectorValue() and int(lo.getBitVectorValue(), 2) == 0:
                    n_zeros = lo.getSort().getBitVectorSize()
                    if hi.getKind() == Kind.BITVECTOR_EXTRACT:
                        inner = _pretty_print_term_child(
                            hi[0],
                            Kind.BITVECTOR_MULT,
                            id_to_cexpr,
                            depth + 1,
                            indent,
                        )
                        multiplier = 1 << n_zeros
                        return f"{prefix}{multiplier} * {inner}"

        # Binary/n-ary operators (ADD, MULT, AND, OR, etc.)
        children = [
            _pretty_print_term_child(
                term[i], current_kind, id_to_cexpr, depth + 1, indent)
            for i in range(term.getNumChildren())
        ]
        expression = f" {operator} ".join(children)
        return expression
    else:
        if current_kind == Kind.SET_INSERT:
            ret_str = ""
            for x in term:
                if x.getKind() == Kind.SET_EMPTY:
                    break
                else:
                    ret_str += f"{pretty_print_term(x, id_to_cexpr, depth + 1, indent)} "
            ret_str = ret_str[:-1]
            return f"{prefix}({ret_str})"

        if current_kind == Kind.SET_MEMBER:
            var = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            set_term = pretty_print_term(term[1], id_to_cexpr, depth + 1)
            return f"{prefix}({var} in {set_term})"
            
        if current_kind == Kind.LAMBDA:
            return term.__str__()
        if current_kind == Kind.BITVECTOR_TO_NAT:
            inner = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            return f"nat({inner})"
        if current_kind == Kind.INT_TO_BITVECTOR:
            inner = pretty_print_term(term[0], id_to_cexpr, depth + 1)
            return f"bv({inner})"
        # Unknown kinds: use cvc5's built-in string representation
        return str(term)


def _collect_flat(term, target_kind):
    """Collect children of an n-ary node (AND/OR), flattening nested same-kind nodes."""
    result = []
    for i in range(term.getNumChildren()):
        child = term[i]
        if child.getNumChildren() > 0 and child.getKind() == target_kind:
            result.extend(_collect_flat(child, target_kind))
        else:
            result.append(child)
    return result


def _pretty_atom(term, id_to_cexpr, inner_pad):
    """Pretty-print an atom inside an IMPLIES chain — break AND/OR across lines."""
    if term.getNumChildren() > 0:
        kind = term.getKind()
        if kind == Kind.AND:
            children = _collect_flat(term, Kind.AND)
            lines = [pretty_print_term(children[0], id_to_cexpr)]
            for child in children[1:]:
                lines.append(inner_pad + "/\\ " + pretty_print_term(child, id_to_cexpr))
            return "\n".join(lines)
        if kind == Kind.OR:
            children = _collect_flat(term, Kind.OR)
            lines = [pretty_print_term(children[0], id_to_cexpr)]
            for child in children[1:]:
                lines.append(inner_pad + "\\/ " + pretty_print_term(child, id_to_cexpr))
            return "\n".join(lines)
    return pretty_print_term(term, id_to_cexpr)


def pretty_term(term, id_to_cexpr=None, indent=0, indent_str="    "):
    """Multi-line pretty-print of a cvc5 term. Breaks on IMPLIES chains and AND/OR."""
    if term is None:
        return "EMPTY"

    kind = term.getKind() if term.getNumChildren() > 0 else None
    pad = indent_str * indent

    # Flatten right-associative IMPLIES chains: A => (B => (C => D)) -> [A, B, C, D]
    if kind == Kind.IMPLIES:
        chain = []
        cur = term
        while cur.getNumChildren() > 0 and cur.getKind() == Kind.IMPLIES:
            chain.append(cur[0])
            cur = cur[1]
        chain.append(cur)

        lines = [pad + _pretty_atom(chain[0], id_to_cexpr, pad)]
        for i, link in enumerate(chain[1:], 1):
            inner_pad = indent_str * (indent + i)
            lines.append(inner_pad + "=> " + _pretty_atom(link, id_to_cexpr, inner_pad + "   "))
        return "\n".join(lines)

    # Break AND conjuncts
    if kind == Kind.AND:
        children = _collect_flat(term, Kind.AND)
        lines = [pad + pretty_print_term(children[0], id_to_cexpr)]
        for child in children[1:]:
            lines.append(pad + "/\\ " + pretty_print_term(child, id_to_cexpr))
        return "\n".join(lines)

    # Break OR disjuncts
    if kind == Kind.OR:
        children = _collect_flat(term, Kind.OR)
        lines = [pad + pretty_print_term(children[0], id_to_cexpr)]
        for child in children[1:]:
            lines.append(pad + "\\/ " + pretty_print_term(child, id_to_cexpr))
        return "\n".join(lines)

    # Atomic — single line
    return pad + pretty_print_term(term, id_to_cexpr)
