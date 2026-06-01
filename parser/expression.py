from .node import Node

# Expressions

class Expression(Node):
    Wildcard = None

    def __repr__(self):
        if self is Expression.Wildcard:
            return "*"
        return super().__repr__()

    def show(self):
        return "*"

Expression.Wildcard = Expression()

class Literal(Expression):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __eq__(self, other):
        if isinstance(other, Literal):
            return self.value == other.value
        return False

    def __hash__(self):
        return hash((self.value))
        
class BooleanLiteral(Literal):
    def __eq__(self, other):
        if not isinstance(other, BooleanLiteral):
            return NotImplemented
        return other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"{'true' if self.value else 'false'}"

class IntegerLiteral(Literal):
    def __eq__(self, other):
        if not isinstance(other, IntegerLiteral):
            return NotImplemented
        return other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)

class BitvectorLiteral(Literal):
    def __init__(self, value, base):
        super().__init__(value)
        self.base = base

    def __eq__(self, other):
        if not isinstance(other, BitvectorLiteral):
            return NotImplemented
        return self.base == other.base and self.value == other.value
    
    def __hash__(self):
        return hash((self.base, self.value))

    def __repr__(self):
        return f"{self.value}bv{self.base}"

    def show(self):
        return f"{self.value}bv{self.base}"

class RealLiteral(Literal):
    def __eq__(self, other):
        if not isinstance(other, RealLiteral):
            return NotImplemented
        return other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)

# Identifier Classes
class Identifier(Expression):
    name = ""
    
    def __init__(self, **opts):
        super().__init__(**opts)
        self.prefix = None

    def __eq__(self, other):
        if not isinstance(other, Identifier):
            return NotImplemented
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent

    def __repr__(self):
        return str(self.name)

class StorageIdentifier(Identifier):
    pass

class ProcedureIdentifier(Identifier):
    pass

class FunctionIdentifier(Identifier):
    def __repr__(self):
        return f"{self.name}"

class LabelIdentifier(Identifier):
    pass

# FunctionApplication Class

class FunctionApplication(Expression):
    function = None
    arguments = None
    children = ["function", "arguments"]

    def __eq__(self, other):
        if not isinstance(other, FunctionApplication):
            return NotImplemented
        return self.function == other.function and self.arguments == other.arguments
    
    def __hash__(self):
        return hash((self.function, tuple(self.arguments)))

    def __repr__(self):
        args_str = ', '.join([arg.__repr__() for arg in self.arguments])
        return f"{self.function}({args_str})"

    def hilite(self):
        args_str = ', '.join([arg.hilite() for arg in self.arguments])
        return f"{self.function.hilite()}({args_str})"

# Unary Expression

class UnaryExpression(Expression):
    expression = None
    children = ["expression"]

    def __eq__(self, other):
        if not isinstance(other, UnaryExpression):
            return NotImplemented
        return self.expression == other.expression

    def __hash__(self):
        return hash(self.expression)

class OldExpression(UnaryExpression):
    def __repr__(self):
        return f"old({self.expression})"

    def show(self):
        return f"old({self.expression.show()})"

class LogicalNegation(UnaryExpression):
    def __repr__(self):
        return f"!{self.expression}"

class ArithmeticNegation(UnaryExpression):
    def __repr__(self):
        return f"-{self.expression}"

# Binary Expression

class BinaryExpression(Expression):
    lhs = None
    op = None
    rhs = None
    children = ["lhs", "op", "rhs"]

    def __eq__(self, other):
        if not isinstance(other, BinaryExpression):
            return NotImplemented
        return self.lhs == other.lhs and self.op == other.op and self.rhs == other.rhs
    
    def __hash__(self):
        return hash((self.lhs, self.op, self.rhs))

    def __repr__(self):
        return f"({self.lhs} {self.op} {self.rhs})"

# Other Expression Classes

class IfExpression(Expression):
    condition = None
    then = None
    else_ = None
    children = ["condition", "then", "else_"]

    def __eq__(self, other):
        if not isinstance(other, IfExpression):
            return NotImplemented
        return self.condition == other.condition and self.then == other.then and self.else_ == other.else_

    def __hash__(self):
        return hash((self.condition, self.then, self.else_))

    def __repr__(self):
        return f"(if {str(self.condition)} then {str(self.then)} else {str(self.else_)})"

class MapSelect(Expression):
    map = None
    indexes = None
    children = ["map", "indexes"]

    def __init__(self, **opts):
        super().__init__(**opts)
   
    def __eq__(self, other):
        if not isinstance(other, MapSelect):
            return NotImplemented
        return self.map == other.map and self.indexes == other.indexes
    
    def __hash__(self):
        return hash((self.map, tuple(self.indexes)))

    def __repr__(self):
        return f"{self.map}[{', '.join([str(idx) for idx in self.indexes])}]"

class MapUpdate(Expression):
    map = None
    indexes = None
    value = None
    children = ["map", "indexes", "value"]

    def __eq__(self, other):
        if not isinstance(other, MapUpdate):
            return NotImplemented
        return self.map == other.map and self.indexes == other.indexes and self.value == other.value

    def __hash__(self):
        return hash((self.map, tuple(self.indexes), self.value))

    def eql(self, mu):
        return isinstance(mu, MapUpdate) and self.map.eql(mu.map) and self.indexes == mu.indexes and self.value.eql(mu.value)

    def __repr__(self):
        return f"{self.map}[{', '.join([str(idx) for idx in self.indexes])} := {self.value}]"

class RangeIndex(Expression):
    expression = None
    high = None
    low = None
    children = ["expression", "high", "low"]

    def __eq__(self, other):
        if not isinstance(other, RangeIndex):
            return NotImplemented
        return self.expression == other.expression and self.high == other.high and self.low == other.low

    def __hash__(self):
        return hash((self.expression, self.high, self.low))

    def __repr__(self):
        return f"{self.expression}[{self.high}:{self.low}]"

# Quantified Expression

class QuantifiedExpression(Expression):
    quantifier = None
    type_arguments = None
    variables = None
    expression = None
    triggers = None
    children = ["quantifier", "type_arguments", "variables", "expression", "triggers"]

    def __eq__(self, other):
        return (isinstance(other, QuantifiedExpression) and self.quantifier == other.quantifier and
                self.type_arguments == other.type_arguments and self.variables == other.variables and
                self.expression == other.expression and self.triggers == other.triggers)
    
    def __hash__(self):
        return hash((self.quantifier, tuple(self.type_arguments), tuple(self.variables), self.expression, tuple(self.triggers)))

    def __repr__(self):
        type_args_str = f"<{', '.join([str(arg) for arg in self.type_arguments])}>" if self.type_arguments else ""
        vars_str = ', '.join([str(var) for var in self.variables])
        triggers_str = ' '.join([str(trig) for trig in self.triggers])
        return f"({self.quantifier} {type_args_str} {vars_str} :: {triggers_str} {self.expression})"

class Trigger(Expression):
    expressions = None
    children = ["expressions"]

    def __eq__(self, t):
        return isinstance(t, Trigger) and t.expressions == self.expressions

    def __hash__(self):
        return hash(tuple(self.expressions)) if self.expressions else hash(())

    def __repr__(self):
        return f"{{{', '.join([str(e) for e in self.expressions])}}}"

    def show(self):
        return f"{{{', '.join([str(e) for e in self.expressions])}}}"
