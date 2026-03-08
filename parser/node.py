class Node:
    indent = ""

    @classmethod
    def get_all_children(cls):
        all_children = []
        seen = set()
        for base in cls.__mro__:
            if hasattr(base, 'children'):
                if 'children' in base.__dict__:
                    for child in base.children:
                        if child not in seen:
                            seen.add(child)
                            all_children.append(child)
        return all_children

    children = ["attributes"]
    attributes = []

    def __init__(self, **opts):
        self.attributes = []
        for k, v in opts.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def has_attribute(self, key):
        return any(a.key == key for a in self.attributes)

    def get_attribute(self, key):
        a = next((_a for _a in self.attributes if _a.key == key), None)
        if a:
            return a.values

    def add_attribute(self, key, *values):
        self.attributes.append(Attribute(key=key, values=list(values)))

    def remove_attribute(self, key):
        self.attributes = [a for a in self.attributes if a.key != key]

    def show_attrs(self):
        return " ".join(str(a) for a in self.attributes)

    def hilite(self):
        return self.show(lambda a: a.hilite())

    def copy(self):
        from parser.boogie_parser import bpl
        other = bpl(str(self))
        for mine, theirs in zip(self.each(), other.each()):
            if mine.__class__ != theirs.__class__:
                raise Exception(f"{mine.__class__} != {theirs.__class__} !?!?")
            if hasattr(theirs, 'bind') and getattr(mine, 'declaration', None):
                theirs.bind(mine.declaration)
        return other

    def each(self, preorder=True):
        return self.enumerate(preorder)

    def each_child(self):
        return self.enumerate_children()

    def enumerate(self, preorder):
        yielder = []
        self._enumerate(yielder, preorder)
        for item in yielder:
            yield item

    def _enumerate(self, yielder, preorder):
        if preorder:
            yielder.append(self)

        for sym in self.__class__.get_all_children():
            node = getattr(self, f"{sym}", None)
            if isinstance(node, Node):
                node._enumerate(yielder, preorder)
            elif isinstance(node, list):
                for n in node.copy():
                    if isinstance(n, Node):
                        n._enumerate(yielder, preorder)

        if not preorder:
            yielder.append(self)

    def enumerate_children(self):
        for sym in self.__class__.get_all_children():
            node = getattr(self, f"{sym}", None)
            if isinstance(node, Node):
                yield node
            elif isinstance(node, list):
                for n in node.copy():
                    if isinstance(n, Node):
                        yield n

    def show(self, func=None):
        return self.show_attrs()


class Attribute(Node):
    key = None
    values = []
    children = ["key", "values"]

    def __repr__(self):
        vs = [str(e) for e in self.values]
        return f"{{:{self.key}{' ' + ', '.join(vs) if vs else ''}}}"
