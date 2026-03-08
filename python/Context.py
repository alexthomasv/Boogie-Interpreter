from collections import defaultdict
from utils.utils import extract_boogie_variables

class Context:
    def __init__(self):
        self.var_store = {}
        self.constraints = defaultdict(set)

    def get(self, key):
        return self.var_store[key]
    
    def set(self, key, value):
        self.var_store[key] = value

    def add_constraint(self, constraint):
        boogie_vars = extract_boogie_variables(constraint)
        for var in boogie_vars:
            # print(f"Adding constraint: {var.name} {constraint}")
            self.constraints[var.name].add(constraint)

    def clear(self):
        self.var_store = {}
        self.constraints = defaultdict(set)
