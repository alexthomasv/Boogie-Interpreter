# Assuming necessary imports and base classes are defined elsewhere
from parser.declaration import VariableDeclaration, ConstantDeclaration
from parser.node import Node
from parser.scope import Scope
class Program(Node, Scope):
    declarations = []
    children = ["declarations"]

    def __repr__(self):
        return "\n".join(str(d) for d in self.declarations) + "\n"

    def global_variables(self):
        return [d for d in self.declarations if isinstance(d, VariableDeclaration)]
    
    def const_variables(self):
        return [d for d in self.declarations if isinstance(d, ConstantDeclaration)]
 
