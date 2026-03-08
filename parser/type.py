from parser.node import Node

class Type(Node):
    Boolean = None
    Integer = None

    def __init__(self, **opts):
        super().__init__(**opts)

    def expand(self):
        return self

# Initialize the static instances
class BooleanType(Type):
    def __init__(self, **opts):
        super().__init__(**opts)

    def __repr__(self):
        return "bool"

class IntegerType(Type):
    def __init__(self, **opts):
        super().__init__(**opts)
    
    def __repr__(self):
        return "int"

class BitvectorType(Type):
    width = None

    def __init__(self, **opts):
        super().__init__(**opts)

    def __repr__(self):
        return f"bv{self.width}"

    def show(self, yield_func):
        return f"{yield_func(f'bv{self.width}')}"

class CustomType(Type):
    name = None
    arguments = None
    children = ["name", "arguments"]
    
    def __init__(self, **opts):
        super().__init__(**opts)

    def expand(self):
        if self.declaration and self.declaration.type:
            return self.declaration.type.expand()
        return self

    def __repr__(self):
        if self.arguments:
            args = " ".join(str(a) for a in self.arguments)
            return f"{self.name} {args}"
        return f"{self.name}"

    def hilite(self, yield_func):
        color = 'blue' if self.declaration else 'red'
        args = " ".join(yield_func(a) for a in self.arguments)
        return f"{self.name} {args}".fmt()

class MapType(Type):
    arguments = None
    domain = None
    range = None
    children = ["arguments", "domain", "range"]
    
    def __init__(self, **opts):
        super().__init__(**opts)

    def expand(self):
        return MapType(
            arguments=self.arguments,
            domain=[d.expand() for d in self.domain],
            range=self.range.expand() if isinstance(self.range, Type) else [r.expand() for r in self.range]
        )

    def __repr__(self):
        args = f"<{', '.join(str(a) for a in self.arguments)}> " if self.arguments else ""
        domain = ", ".join(str(d) for d in self.domain)
        range_ = str(self.range)
        return f"{args}[{domain}] {range_}"

