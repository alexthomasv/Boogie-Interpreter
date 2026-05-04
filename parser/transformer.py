from functools import reduce

from lark import Transformer

from .expression import *
from .statement import *
from .declaration import *
from .type import *
from .specifications import *
from .program import *
from .node import Attribute


class BoogieToObjectTransformer(Transformer):
    def __init__(self, visit_tokens, solver=None):
        super().__init__(visit_tokens)
        self.solver = solver

    def start(self, items):
        declarations = []
        for item in items:
            if isinstance(item, list):
                declarations.extend(item)
            else:
                declarations.append(item)
        return Program(declarations=declarations)

    def ens(self, items):
        return items[0]

    def enss_opt(self, items):
        return items[0]

    def enss(self, items):
        return list(items)

    def decl(self, items):
        return items[0]

    def uses_block(self, items):
        return list(items)

    def type_decl(self, items):
        attrs = items[0]
        finite_opt = items[1]
        name = items[2]
        tc_params = items[3]
        type_syn = items[4]
        combined = {"attributes": attrs, "name": name, "arguments": tc_params, "finite": finite_opt, "type": type_syn}
        return TypeDeclaration(**combined)

    def type_syn(self, items):
        type_t = items[0]
        return type_t

    def empty_type_syn(self, items):
        return None

    def tc_params(self, items):
        return list(items)

    def finite_opt(self, items):
        return True

    def empty_finite(self, items):
        return False

    def pspec(self, items):
        if len(items) == 1:
            specs = items[0]
            return {"specs": specs, "body": None}
        elif len(items) == 2:
            specs = items[0]
            body = items[1]
            return {"specs": specs, "body": body}
        else:
            assert(False)

    def specs(self, items):
        return list(items)

    def spec(self, items):
        return items[0]

    def spec_requires(self, items):
        free_opt = items[0]
        attrs = items[1]
        expr = items[2]
        return RequiresClause(attributes=attrs, expression=expr, free=free_opt)

    def spec_modifies(self, items):
        free_opt = items[0]
        attrs = items[1]
        idents = items[2]
        idents = [StorageIdentifier(**i) for i in idents]
        return ModifiesClause(attributes=attrs, free=free_opt, identifiers=idents)

    def spec_ensures(self, items):
        free_opt = items[0]
        attrs = items[1]
        expr = items[2]
        return EnsuresClause(attributes=attrs, expression=expr, free=free_opt)

    def empty_enss(self, items):
        return []

    def empty_ants(self, items):
        return []

    def ants(self, items):
        return list(items)

    def ants_opt(self, items):
        return items[0]

    def ant(self, items):
        return items[0]


    def out_params(self, items):
        return items[0]

    def empty_out_params(self, items):
        return []

    def name(self, items):
        return items[0]

    def names(self, items):
        return list(items)

    def empty_type_args(self, items):
        return []

    def type_args(self, items):
        return items[0]

    def empty_param_decls(self, items):
        return []

    def param_decls(self, items):
        return list(items)

    def param_decls_opt(self, items):
        return items[0]

    def param_decl(self, items):
        typed_idents_where = items[0]
        return StorageDeclaration(**typed_idents_where)

    def proc_decl(self, items):
        attrs = items[0]
        name = items[1]
        type_args = items[2]
        param_decls_opt = items[3]
        out_params = items[4]
        pspec = items[5]
        return ProcedureDeclaration(attributes=attrs, name=name, type_arguments=type_args, parameters=param_decls_opt, returns=out_params, specifications=pspec["specs"], body=pspec["body"])

    def block_stmt(self, items):
        stmt = items[0]
        return {"names": [], "statements": [stmt]}

    def block_label(self, items):
        name = items[0]
        return {"names": [name], "statements": []}

    def empty_blocks(self, items):
        return {}

    def blocks(self, items):
        blocks = items
        new_blocks = []
        curr_statements = []
        curr_names = []
        for b in blocks:
            if "names" in b and len(b["names"]) > 0:
                if curr_names or curr_statements:
                    if not curr_names:
                        # This can happen if there are no labels in the block
                        pass
                    new_block = Block(curr_names[:], curr_statements[:])
                    new_blocks.append(new_block)

                curr_names.clear()
                curr_statements.clear()
                curr_names.append(b["names"][0])
            elif "statements" in b and len(b["statements"]) > 0:
                if not (curr_names or curr_statements):
                    curr_names = []
                    curr_statements = []
                curr_statements.append(b["statements"][0])
            else:
                # Empty block
                curr_names = []
                curr_statements = []

        if curr_names or curr_statements:
            if not curr_names:
                # This can happen if there are no labels in the block
                pass
            new_blocks.append(Block(curr_names[:], curr_statements[:]))
        return new_blocks

    def impl_decl(self, items):
        attrs = items[0]
        name = items[1]
        type_args = items[2]
        param_decls_opt = items[3]
        out_params = items[4]
        body = items[5]
        return ImplementationDeclaration(attributes=attrs, name=name, type_arguments=type_args, parameters=param_decls_opt, returns=out_params, specifications=[], body=body)

    def break_cmd(self, items):
        identifier = LabelIdentifier(**items[0]) if items else None
        return BreakStatement(identifier=identifier)

    def assert_cmd(self, items):
        # Return the proposition (the expression being asserted)
        attrs = items[0]
        expr = items[1]
        return AssertStatement(attributes=attrs, expression=expr)

    def assume_cmd(self, items):
        # Return the proposition (the expression being asserted)
        attrs = items[0]
        expr = items[1]
        return AssumeStatement(attributes=attrs, expression=expr)

    def index(self, items):
        map = items[0]
        indexes = items[1]
        return MapSelect(map=map, indexes=indexes)

    def assign_index(self, items):
        map = items[0]
        indexes = items[1]
        value = items[2]
        return MapUpdate(map=map, indexes=indexes, value=value)

    def range_index(self, items):
        return RangeIndex(expression=items[0], high=int(items[1]), low=int(items[2]))

    def map_type(self, items):
        type_args = items[0]
        types = items[1]
        type = items[2]
        return MapType(arguments=type_args, domain=types, range=type)

    def types(self, items):
        return items

    def type(self, items):
        return items[0]

    def exprs(self, items):
        return items

    def return_cmd(self, items):
        return ReturnStatement()

    def return_expr_cmd(self, items):
        expr = items[0]
        return ReturnStatement(expression=expr)

    def literal(self, items):
        return items[0]

    def axiom_decl(self, items):
        attr = items[0]
        expr = items[1]
        return AxiomDeclaration(attributes=attr, expression=expr)

    def const_decl(self, items):
        attr = items[0]
        unique_opt = items[1]
        typed_idents = items[2]
        order_spec = items[3]

        combined = typed_idents | {"attributes": attr, "unique": unique_opt, "order_spec": order_spec}
        return ConstantDeclaration(**combined)

    def unique_opt(self, items):
        return True

    def empty_unique(self, items):
        return False

    def order_spec(self, items):
        parent_info = items[0]
        complete_opt = items[1]
        return [parent_info, complete_opt]

    def parent_info(self, items):
        return items[0]

    def empty_parent_info(self, items):
        return None

    def parent_edge(self, items):
        unique_opt = items[0]
        ident = items[1]
        return [unique_opt, StorageIdentifier(**ident)]

    def parent_edges(self, items):
        return list(items)

    def parent_edges_opt(self, items):
        return items[0]

    def empty_parent_edges(self, items):
        return []

    def complete_opt(self, items):
        return True

    def empty_complete(self, items):
        return False

    def empty_fbody(self, items):
        return None

    def fbody(self, items):
        expr = items[0]
        return expr

    def body(self, items):
        var_decls = items[0]
        blocks = items[1]
        return Body(locals=var_decls, blocks=blocks)

    def expr_ident(self, items):
        return StorageIdentifier(**items[0])

    def custom_type(self, items):
        id = items[0]
        tc_args = items[1]
        return CustomType(name=id, arguments=tc_args)

    def fargs_type_comma(self, items):
        type_t = items[0]
        fargs = items[1]

        if "names" not in fargs[0]:
            return [{"names": [], "type": type_t}] + fargs
        else:
            fargs[0]["names"].insert(0, type_t)
            return fargs

    def fargs_type_only(self, items):
        return [{"names": [], "type": items[0]}]

    def farg_type_only(self, items):
        return StorageDeclaration(**{"names": [], "type": items[0]})

    def fargs(self, items):
        type_t = items[0]
        fargs = items[1]
        if len(fargs[0]["names"]) == 0:
            fargs[0]["names"].insert(0, type_t)
            return fargs
        else:
            head = fargs[0]["names"].pop(0)
            return [{"names": [items[0]], "type":head}] + fargs

    def fargs_opt(self, items):
        fargs = items[0]
        fargs_transformed = []
        for arg in fargs:
            arg["names"] = [a.name for a in arg["names"]]
            fargs_transformed.append(StorageDeclaration(**arg))
        return fargs_transformed

    def empty_fargs(self, items):
        return []

    def farg(self, items):
        return StorageDeclaration(**items[0])

    def tc_args(self, items):
        type_atom = items[0]
        tc_args = items[1]
        return [type_atom] + tc_args

    def empty_tc_args(self, items):
        return []

    def func_decl(self, items):
        attrs = items[0]
        name = items[1]
        type_args = items[2]
        fargs_opt = items[3]
        returns = items[4]
        fbody = items[5]
        return FunctionDeclaration(attributes=attrs, name=name, type_arguments=type_args, arguments=fargs_opt, return_type=returns, body=fbody)

    def guard(self, items):
        return items[0]

    def wc_expr(self, items):
        return items[0]

    def wildcard_expr(self, items):
        return Expression.Wildcard

    def wc_exprs(self, items):
        return list(items)

    def wc_exprs_opt(self, items):
        return items[0]

    def empty_wc_exprs(self, items):
        return []

    def if_stmt(self, items):
        wc_expr = items[0]
        if_block = items[1]
        else_stmt = items[2]
        return IfStatement(condition=wc_expr, blocks=if_block, else_=else_stmt)

    def empty_else_stmt(self, items):
        return None

    def else_stmt_blocks(self, items):
        else_blocks = items[0]
        return else_blocks

    def else_if_stmt(self, items):
        if_stmt = items[0]
        return if_stmt

    def free_opt(self, items):
        return True

    def empty_free(self, items):
        return False

    def while_cmd(self, items):
        guard = items[0]
        invariants = items[1:-1]
        blocks = items[-1]
        return WhileStatement(condition=guard, invariants=invariants, blocks=blocks)

    def _binary(self, items, op):
        return BinaryExpression(lhs=items[0], op=op, rhs=items[1])

    def binary_expr_iff(self, items):     return self._binary(items, "<==>")
    def binary_expr_implies(self, items): return self._binary(items, "==>")
    def binary_expr_or(self, items):      return self._binary(items, "||")
    def binary_expr_and(self, items):     return self._binary(items, "&&")
    def binary_expr_eq(self, items):      return self._binary(items, "==")
    def binary_expr_neq(self, items):     return self._binary(items, "!=")
    def binary_expr_lt(self, items):      return self._binary(items, "<")
    def binary_expr_gt(self, items):      return self._binary(items, ">")
    def binary_expr_le(self, items):      return self._binary(items, "<=")
    def binary_expr_ge(self, items):      return self._binary(items, ">=")
    def binary_expr_lt_colon(self, items): return self._binary(items, "<:")
    def binary_expr_plus_plus(self, items): return self._binary(items, "++")
    def binary_expr_plus(self, items):    return self._binary(items, "+")
    def binary_expr_minus(self, items):   return self._binary(items, "-")
    def binary_expr_times(self, items):   return self._binary(items, "*")
    def binary_expr_div(self, items):     return self._binary(items, "/")
    def binary_expr_mod(self, items):     return self._binary(items, "%")

    def func_expr(self, items):
        id = FunctionIdentifier(**items[0])
        args = items[1]
        return FunctionApplication(function=id, arguments=args)

    def logical_negation(self, items):
        expr = items[0]
        return LogicalNegation(expression=expr)

    def arithmetic_negation(self, items):
        expr = items[0]
        return ArithmeticNegation(expression=expr)

    def typed_ids_where(self, items):
        typed_ids = items[0]
        where_clause = items[1]
        return typed_ids | where_clause

    def typed_ids(self, items):
        idents = items[0]
        type_t = items[1]
        return {"names": idents, "type": type_t}

    def where_clause(self, items):
        expr = items[0]
        return {"where": expr}

    def empty_where_clause(self, items):
        return {}

    def empty_var_decls(self, items):
        return []

    def var_decls(self, items):
        return list(items)

    def var_decl(self, items):
        attrs = items[0]
        typed_idents_wheres = items[1]
        combined = typed_idents_wheres | {"attributes": attrs}
        return VariableDeclaration(**combined)

    def type_atom(self, items):
        type = items[0]
        if not isinstance(type, str):
            return type
        if type == "bool":
            return BooleanType()
        elif type == "int":
            return IntegerType()
        elif type == "real":
            return RealType()
        else:
            assert(False)

    def bool_lit(self, items):
        if items[0].value == "true":
            return BooleanLiteral(True)
        elif items[0].value == "false":
            return BooleanLiteral(False)
        else:
            assert(False)

    def select(self, items):
        return items[0]

    def selects(self, items):
        return list(items)

    def empty_exprs(self, items):
        return []

    def lhss(self, items):
        return list(items)

    def lhs(self, items):
        ident = items[0]
        selects = items[1]

        id = StorageIdentifier(**ident)
        ret = reduce(lambda m, x: MapSelect(map=m, indexes=x), selects, id)
        return ret

    def quantifier(self, items):
        value = items[0]
        if value == "∀":
            return "forall"
        if value == "∃":
            return "exists"
        return value

    def quantifier_expr(self, items):
        quantifier = items[0]
        type_args = items[1]
        param_decls = items[2]
        ants = items[3]
        expr = items[4]
        return QuantifiedExpression(quantifier=quantifier, type_arguments=type_args, variables=param_decls, attributes=ants, triggers=ants, expression=expr)

    def call_lhs(self, items):
        if(len(items) == 1):
            # This is a single identifier
            ident = items[0]
            return {"name": ProcedureIdentifier(**ident), "rets":[]}
        else:
            lhs = items[0]
            rets = []
            for l in lhs:
                rets.append(StorageIdentifier(**l))
            rhs = items[1]
            return {"name": ProcedureIdentifier(**rhs), "rets":rets}


    def call_cmd(self, items):
        attrs = items[0]
        call_lhs = items[1]
        exprs_opt = items[2]
        return CallStatement(attributes=attrs, procedure=call_lhs["name"], arguments=exprs_opt, assignments=call_lhs["rets"])

    def call_forall_cmd(self, items):
        ident = items[0]
        exprs_opt = items[1]
        return CallStatement(
            attributes=[Attribute(key="forall", values=[])],
            procedure=ProcedureIdentifier(**ident),
            arguments=exprs_opt,
            assignments=[],
        )

    def par_call_cmd(self, items):
        attrs = items[0]
        calls = items[1:]
        return ParallelCallStatement(attributes=attrs, calls=calls)

    def yield_cmd(self, items):
        return YieldStatement()

    def pop_cmd(self, items):
        return PopStatement()

    def stmt(self, items):
        return items[0]

    def ident(self, token):
        assert(len(token) == 1)
        return {"name": token[0]}

    def IDENT(self, token):
        return str(token)

    def nat(self, token):
        assert(len(token) == 1)
        return IntegerLiteral(int(token[0]))

    def dec(self, items):
        return RealLiteral(f"{items[0]}{items[1]}")

    def float_lit(self, items):
        return RealLiteral(f"{items[0]}.{items[1]}")

    def bv_lit(self, items):
        return BitvectorLiteral(int(items[0]), int(items[1]))

    def DIGITS(self, token):
        return int(token)

    def string(self, token):
        return str(token[0])

    def idents(self, token):
        return list(token)

    def goto_cmd(self, items):
        idents = items[0]
        ids = [LabelIdentifier(**id) for id in idents]
        return GotoStatement(identifiers=ids)

    def assign_cmd(self, items):
        lhs = items[0]
        rhs = items[1]
        return AssignStatement(lhs=lhs, rhs=rhs)

    def if_then_else_expr(self, items):
        cond = items[0]
        then_expr = items[1]
        else_expr = items[2]
        combined = {"condition": cond, "then": then_expr, "else_": else_expr}
        return IfExpression(**combined)

    def old_expr(self, items):
        return OldExpression(expression=items[0])

    def havoc_cmd(self, items):
        idents = items[0]
        ids = []
        for id in idents:
            ids.append(StorageIdentifier(**id))
        return HavocStatement(identifiers=ids)

    def attrs_opt(self, items):
        attr = items[0]
        attrs = items[1]
        return [attr] + attrs

    def attr(self, items):
        name = items[0]
        val = items[1]
        return Attribute(key=name, values=val)

    def trigger(self, items):
        return Trigger(expressions=items[0])

    def empty_attr(self, items):
        return []

    def invariant(self, items):
        attrs = items[0]
        expr = items[1]
        invariant = LoopInvariant(expression=expr, free=False)
        invariant.attributes = attrs
        return invariant

    def free_invariant(self, items):
        attrs = items[0]
        expr = items[1]
        invariant = LoopInvariant(expression=expr, free=True)
        invariant.attributes = attrs
        return invariant
