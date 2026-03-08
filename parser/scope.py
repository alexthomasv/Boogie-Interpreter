class Scope:
    def _get_kind_map(self):
        from parser.declaration import StorageDeclaration, ConstantDeclaration, VariableDeclaration, TypeDeclaration, FunctionDeclaration, ProcedureDeclaration, ImplementationDeclaration
        from parser.expression import LabelIdentifier, FunctionIdentifier, StorageIdentifier, ProcedureIdentifier
        from parser.type import CustomType
        from parser.statement import Block

        return {
            CustomType: 'type',
            TypeDeclaration: 'type',
            LabelIdentifier: 'label',
            Block: 'label',
            StorageIdentifier: 'storage',
            StorageDeclaration: 'storage',
            ConstantDeclaration: 'storage',
            VariableDeclaration: 'storage',
            FunctionIdentifier: 'function',
            FunctionDeclaration: 'function',
            ProcedureIdentifier: 'procedure',
            ProcedureDeclaration: 'procedure',
            ImplementationDeclaration: 'procedure'
        }

    def lookup_table(self, kind, name):
        if not hasattr(self, '_lookup_table'):
            self._lookup_table = {}
        if kind not in self._lookup_table:
            self._lookup_table[kind] = {}
        if name not in self._lookup_table[kind]:
            self._lookup_table[kind][name] = set()
        return self._lookup_table[kind][name]

    def resolve(self, id):
        kind_map = self._get_kind_map()
        return next(iter(self.lookup_table(kind_map[id.__class__], id.name)), None)
