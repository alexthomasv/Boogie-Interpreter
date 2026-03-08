from parser.node import Node

class Specification(Node):
    free = False
    def __init__(self, **opts):
        super().__init__(**opts)
        if "free" in opts:
            self.free = opts["free"]

class LoopInvariant(Specification):
    expression = None
    children = ["expression"]

    def __init__(self, expression, free=False):
        super().__init__(free=free)
        self.expression = expression

    def __repr__(self):
        free_part = 'free ' if self.free else ''
        return f"{free_part}invariant {self.expression};"

class RequiresClause(Specification):
    expression = None
    children = ["expression"]
    def __repr__(self):
        free_part = 'free ' if self.free else ''
        return f"{free_part}requires {self.show_attrs()} {self.expression};"

class ModifiesClause(Specification):
    identifiers = []
    children = ["identifiers"]
    def __repr__(self):
        free_part = 'free ' if self.free else ''
        identifiers_part = ", ".join(str(a) for a in self.identifiers)
        return f"{free_part}modifies {self.show_attrs()} {identifiers_part};"

class EnsuresClause(Specification):
    expression = None
    children = ["expression"]
    def __repr__(self):
        free_part = 'free ' if self.free else ''
        return f"{free_part}ensures {self.show_attrs()} {self.expression};"
