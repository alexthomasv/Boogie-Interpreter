class Binding:

    @property
    def declaration(self):
        if not hasattr(self, '_declaration'):
            self._declaration = None
        return self._declaration

    @declaration.setter
    def declaration(self, value):
        self._declaration = value

    def bind(self, decl):
        if self.declaration != decl:
            self.unbind()
        self._declaration = decl
        return decl

    def unbind(self):
        if not self.declaration:
            return
        self._declaration = None
