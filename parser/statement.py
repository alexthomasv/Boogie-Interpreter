from .node import Node
from .declaration import Declaration
from .scope import Scope
from .expression import LabelIdentifier, Identifier


class Statement(Node):
    pass

class AssertStatement(Statement):
    expression = None
    children = ["expression"]

    def __repr__(self):
        return f"{'assert'} {self.show_attrs()} {self.expression};"

class AssumeStatement(Statement):
    expression = None
    children = ["expression"]

    def __repr__(self):
        return f"{'assume'} {self.show_attrs()} {self.expression};"

class HavocStatement(Statement):
    identifiers = []

    children = ["identifiers"]

    def __eq__(self, other):
        if not isinstance(other, HavocStatement):
            return NotImplemented
        return tuple(self.identifiers) == tuple(other.identifiers)
    
    def __hash__(self):
        return hash(tuple(self.identifiers))

    def __repr__(self):
        return f"{str('havoc')} {', '.join(str(a) for a in self.identifiers)};"

class AssignStatement(Statement):
    lhs = []
    rhs = []

    children = ["lhs", "rhs"]

    def __eq__(self, other):
        if not isinstance(other, AssignStatement):
            return NotImplemented
        return tuple(self.lhs) == tuple(other.lhs) and tuple(self.rhs) == tuple(other.rhs)
    
    def __hash__(self):
        return hash((tuple(self.lhs), tuple(self.rhs)))

    def __repr__(self):
        lhs = ', '.join(str(item) for item in self.lhs)
        rhs = ', '.join(str(item) for item in self.rhs)
        return f"{lhs} := {rhs};"
    
    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent
        for l in self.lhs:
            l.set_prefix(prefix, indent)
        for r in self.rhs:
            if isinstance(r, Identifier):
                r.set_prefix(prefix, indent)

class GotoStatement(Statement):
    identifiers = []
    children = ["identifiers"]

    def __eq__(self, other):
        if not isinstance(other, GotoStatement):
            return NotImplemented
        return tuple(self.identifiers) == tuple(other.identifiers)
    
    def __hash__(self):
        return hash(tuple(self.identifiers))

    def __repr__(self):
        return f"{'goto'} {', '.join(str(a) for a in self.identifiers)};"

class ReturnStatement(Statement):
    expression = None
    children = ["expression"]

    def __eq__(self, other):
        if not isinstance(other, ReturnStatement):
            return NotImplemented
        return self.expression == other.expression
    
    def __hash__(self):
        return hash(self.expression)

    def __repr__(self):
        return f"{str('return')} {str(self.expression) if self.expression else ''};"

class CallStatement(Statement):

    procedure = None
    arguments = []
    assignments = []

    children = ["procedure", "arguments", "assignments"]

    def forall(self):
        return len(self.assignments) == 0

    def target(self):
        return self.procedure.declaration

    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent
        for a in self.arguments:
            if isinstance(a, Identifier):
                a.set_prefix(prefix, indent)
        for a in self.assignments:
            if isinstance(a, Identifier):
                a.set_prefix(prefix, indent)
        
    def __repr__(self):
        rets = (', '.join(str(a) for a in self.assignments) + ' := ') if len(self.assignments) else ''
        proc = str(self.procedure)
        args = ', '.join(str(a) for a in self.arguments)
        return f"{str('call')} {self.show_attrs()} {rets} {proc}({args});"

class IfStatement(Statement):
    condition = None
    blocks = []
    else_ = None

    children = ["condition", "blocks", "else_"]

    def declarations(self):
        return self.blocks + (self.else_.declarations() if isinstance(self.else_, IfStatement) else self.else_ or [])
    
    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent
        for b in self.blocks:
            b.set_prefix(prefix, indent+2)

    def __repr__(self):
        # --------- build the “then” body -------------------------------------------------
        then_body  = "\n".join(str(b) for b in self.blocks)
        body       = "{\n" + then_body + "\n" + self.indent + "}"

        # --------- build the “else” part -------------------------------------------------
        if isinstance(self.else_, IfStatement):
            else_part = f" else {self.else_}"
        elif self.else_:                                   # list of blocks
            else_body  = "\n".join(str(b) for b in self.else_)
            else_part  = " else {\n" + else_body + "\n" + self.indent + "}"
        else:                                               # no else-clause
            else_part = ""

        # --------- final string ----------------------------------------------------------
        return f"if ({self.condition}) {body}{else_part}"

class WhileStatement(Statement):

    condition = None
    invariants = []
    blocks = []

    children = ["condition", "invariants", "blocks"]

    def declarations(self):
        return self.blocks

    def show(self, blk):
        body = Printing.braces('\n'.join(blk(b) for b in self.blocks))
        invs = '\n'.join(blk(a) for a in self.invariants)
        invs = f"\n{invs}\n" if invs else ""
        return f"{blk('while')} ({blk(self.condition)}){invs} {body}"

class BreakStatement(Statement):
    def __init__(self, identifier=None):
        self.identifier = identifier

    def show(self, blk):
        return f"{blk('break')}{' ' if self.identifier else ''}{blk(self.identifier)};"

class Block(Declaration):

    indent = ""
    indent_num = 0
    name = ""
    names = []
    statements = []
    prefix = None

    def __init__(self, names, statements):
        self.names = names
        if len(self.names) == 0:
            self.name = "Default"
        else:
            self.name = self.names[0]
        self.statements = statements

    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent
        self.indent_num = indent
        for s in self.statements:
            if isinstance(s, IfStatement):
                s.set_prefix(prefix, indent+2)
            elif isinstance(s, AssignStatement):
                s.set_prefix(prefix, indent)
            elif isinstance(s, CallStatement):
                s.set_prefix(prefix, indent)

    def id(self):
        return LabelIdentifier(name=self.name, declaration=self)

    def copy(self):
        return Block(names=self.names[:], statements=[s.copy() for s in self.statements])

    def __repr__(self):
        indent_prefix = (self.indent_num - 2) * "  "
        label = [indent_prefix + f"{n}:" for n in self.names if n]
        stmts = [self.indent + f"{s}" for s in self.statements]
        return '\n'.join(label + stmts)

class Body(Node):

    children = ["locals", "blocks"]
    locals = []
    blocks = []

    def declarations(self):
        return self.locals + self.blocks
    
    def set_prefix(self, prefix, indent=1):
        self.prefix = prefix
        self.indent = "  " * indent
        for b in self.blocks:
            b.set_prefix(prefix, indent+2)

    def __repr__(self):
        body_locals = [f"{self.indent}{b}" for b in self.locals]
        body_blocks = [f"{b}" for b in self.blocks]
        return "{\n" + '\n'.join(body_locals + body_blocks) + "\n}"

    def fresh_var(self, type_, prefix, taken=None):
        if taken is None:
            taken = []
        taken += [d.names for d in self.locals]
        name = self.fresh_from(prefix or "$var", taken)
        self.locals.append(bpl(f"var {name}: {type_};"))
        return bpl(name)

    def fresh_label(self, prefix):
        name = self.fresh_from(prefix or "$bb", [b.names for b in self.blocks])
        decl = Label(name=name)
        return LabelIdentifier(name=name, declaration=decl)

    def fresh_from(self, prefix, taken):
        if prefix not in taken and prefix:
            return prefix
        i = 0
        while True:
            candidate = f"{prefix}_{i}"
            if candidate not in taken:
                return candidate
            i += 1
