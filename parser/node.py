import copy as _copy


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
        self._init_mutable_defaults()
        for k, v in opts.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _init_mutable_defaults(self):
        for base in reversed(self.__class__.__mro__):
            for name, value in base.__dict__.items():
                if name == "children" or name.startswith("__"):
                    continue
                if isinstance(value, list):
                    setattr(self, name, value.copy())
                elif isinstance(value, dict):
                    setattr(self, name, value.copy())
                elif isinstance(value, set):
                    setattr(self, name, value.copy())

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

    def clone(self):
        return _copy.deepcopy(self)

    def copy(self):
        return self.clone()

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
