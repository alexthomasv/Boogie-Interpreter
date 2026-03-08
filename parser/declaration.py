from parser.node import Node
from parser.binding import Binding
from parser.scope import Scope


class Declaration(Node):
    @property
    def names(self):
        if hasattr(self, 'name'):
            return [self.name]
        return []


# Type Declarations
class TypeDeclaration(Declaration): 
    name = None
    arguments = []
    finite = None
    type = None
    children = ["name", "arguments", "finite", "type"]

    def signature(self):
        return f"type {self.name}"

    def __repr__(self):
        args = " ".join([str(a) for a in self.arguments])
        finite = f"{str(self.finite)} " if self.finite else ""
        typ_str = f" = {str(self.type)}" if self.type else ""
        return f"type {self.show_attrs()} {finite}{str(self.name)} {args}{typ_str};"


class AxiomDeclaration(Declaration):
    expression = None
    children = ["expression"]

    def __repr__(self):
        return f"axiom {self.show_attrs()} {str(self.expression)};"


# Storage Declaration and Variable Declaration
class StorageDeclaration(Declaration):
    names = []
    ids = []
    type = None
    where = None
    children = ["names", "type", "where"]
    prefix = None

    def __init__(self, **opts):
        super().__init__(**opts)
        self.ids = self.idents()

    def set_prefix(self, prefix, indent):
        self.prefix = prefix
        self.indent = "  " * indent
        for id in self.ids:
            id.set_prefix(prefix, indent)

    def signature(self):
        return f"{', '.join(self.names)}: {self.type_t}"

    def __repr__(self):
        names_str = ", ".join([str(name) for name in self.names]) + ": " if self.names else ""
        where_str = f" where {str(self.where)}" if self.where else ""
        return f"{names_str} {str(self.type)}{where_str}"

    def flatten(self):
        if not self.names:
            return self
        else:
            return [self.__class__(names=[name], typ=self.typ, where=self.where) for name in self.names]

    def idents(self):
        from parser.expression import StorageIdentifier
        ids = [StorageIdentifier(name=name, declaration=self) for name in self.names]
        return ids


class VariableDeclaration(StorageDeclaration):
    def signature(self):
        return f"var {', '.join(self.names)}: {self.typ}"

    def __repr__(self,):
        return f"var {super().__repr__()};"


class ConstantDeclaration(StorageDeclaration):
    unique = None
    order_spec = None
    children = ["unique", "order_spec"]

    def signature(self):
        return f"const {', '.join(self.names)}: {self.typ}"

    def __repr__(self):
        names_str = ", ".join([str(name) for name in self.names]) + ": " if self.names else ""
        ord_str = ""
        if self.order_spec:
            if self.order_spec[0]:
                ord_str += " <: " + ", ".join([(f"unique {p}" if c else p) for c, p in self.order_spec[0]])
            if self.order_spec[1]:
                ord_str += " complete"
        return f"const {names_str} {str(self.type)}{ord_str};"

# Procedure and Implementation Declaration
class ProcedureDeclaration(Declaration, Scope):
    name = None
    type_arguments = []
    parameters = []
    returns = []
    specifications = []
    body = None
    prefix = None
    children = ["name", "type_arguments", "parameters", "returns", "specifications", "body"]

    def __init__(self, **opts):
        super().__init__(**opts)
        self.set_initial_prefix(self.name)

    def set_initial_prefix(self, prefix, indent=0):
        self.prefix = prefix
        self.indent = "  " * indent

    def set_prefix(self, indent=0):
        assert(self.prefix)

        if self.body:
            self.body.set_prefix(self.prefix, indent)

        for a in self.parameters:
            a.set_prefix(self.prefix, indent)
            
        for a in self.returns:
            a.set_prefix(self.prefix, indent)

    @property
    def declarations(self):
        return self.parameters + self.returns

    def modifies(self):
        return [s.identifiers for s in self.specifications if isinstance(s, ModifiesClause)]

    def fresh_var(self, typ, prefix=""):
        taken = [param.names for param in self.parameters] + [ret.names for ret in self.returns]
        return self.body.fresh_var(typ, prefix, taken=taken) if self.body else None

    def fresh_label(self, prefix="$label"):
        return self.body.fresh_label(prefix) if self.body else None

    def sig(self):
        params = ", ".join([str(param) for param in self.parameters])
        rets = "" if not self.returns else f"returns ({', '.join([str(ret) for ret in self.returns])})"
        return f"{self.show_attrs()} {str(self.name)}({params}) {rets}"

    def signature(self):
        params = ", ".join([str(param.type) for param in self.parameters])
        rets = "" if not self.returns else f": {', '.join([str(ret.type) for ret in self.returns])}"
        return f"{self.name}({params}){rets}"

    def __repr__(self):
        specs = "\n".join([str(spec) for spec in self.specifications])
        specs_str = f"\n{specs}" if specs else ""
        if self.body:
            return f"procedure {self.sig()}{specs_str}\n{str(self.body)}"
        else:
            return f"procedure {self.sig()};{specs_str}"


class ImplementationDeclaration(ProcedureDeclaration):
    def __repr__(self):
        return f"implementation {self.sig()}\n{str(self.body)}"

class FunctionDeclaration(Declaration, Scope):
    name = None
    type_arguments = []
    arguments = []
    return_type = None
    body = None
    children = ["name", "type_arguments", "arguments", "return_type", "body"]

    @property
    def declarations(self):
        return self.arguments

    def signature(self):
        args = ", ".join([str(arg.type) for arg in self.arguments])
        return f"{self.name}({args}): {self.return_type.type}".replace(" ", "")

    def __repr__(self):
        args = ", ".join([str(arg) for arg in self.arguments])
        ret = self.return_type
        body = f" {{ {self.body} }}" if self.body else ";"
        return f"function {self.show_attrs()} {self.name}({args}) returns ({ret}){body}"