import cvc5
from utils_cvc5 import *
from AbductUtils import *

class AssignTuple:
    def __init__(self, rhs: cvc5.Term):
        self.rhs = rhs

    def __eq__(self, other):
        return self.rhs == other.rhs
    
    def __hash__(self):
        return hash((self.rhs))
    
    def __repr__(self):
        if isinstance(self.rhs, cvc5.Term):
            return f"{term_to_string(self.rhs)}"
        else:
            return f"{self.rhs}"
        
    def __getstate__(self):
        state = self.__dict__.copy()
        rhs = state.pop("rhs")
        state["rhs"] = serialize_cvc5_term(rhs)
        return state

    def to_cvc5(self, solver, lhs):
        lhs_equals_rhs = solver.mkTerm(Kind.EQUAL, lhs, self.rhs)
        return lhs_equals_rhs
        
    def deserialize(self, state_cache):
        self.rhs = deserialize_cvc5_term(state_cache, self.rhs)

class Assignment:
    def __init__(self, lhs):
        assert isinstance(lhs, str)
        self.lhs = lhs
        self.rhs = []

        self.additional_proofs = set()
        self.real_rhs = None
        self.just_nested_loop = False
        self.replace_var = None
        self.related_vars = set()

    def __eq__(self, other):
        """Check if two Assignment objects are equal."""
        if not isinstance(other, Assignment):
            return False

        return (
            self.lhs == other.lhs and
            self.rhs == other.rhs and
            self.additional_proofs == other.additional_proofs and
            self.real_rhs == other.real_rhs
        )
    
    def copy(self):
        """Returns a deep copy of the current Assignment object."""
        assignment = Assignment(self.lhs)
        assignment.rhs = self.rhs[:]
        assignment.real_rhs = self.real_rhs.copy()
        assignment.additional_proofs.update(self.additional_proofs)
        return assignment

    def add(self, assign_tuple: AssignTuple):
        self.rhs.append(assign_tuple)
        self.real_rhs = None
    
    def to_cvc5(self, solver, state_cache) -> cvc5.Term:
        if self.real_rhs is None:
            lhs_cvc5 = state_cache.cvc5_var(self.lhs)
            terms = []
            for rhs in self.rhs:
                terms.append(rhs.to_cvc5(solver, lhs_cvc5))
                self.related_vars.update(extract_variable_terms_keep_select(rhs.rhs))

            if len(terms) == 0:
                simplified_term = None
                # assert(len(self.additional_proofs) > 0)
            else:
                default_value = solver.mkConst(lhs_cvc5.getSort(), "dummy")
                lhs_default_value = solver.mkTerm(Kind.EQUAL, lhs_cvc5, default_value)
                self.replace_var = lhs_default_value
                terms.append(lhs_default_value)
                simplified_term = solver.mkTerm(Kind.AND, *terms)
  
            if simplified_term is None:
                self.real_rhs = None
            else:
                self.real_rhs = solver.simplify(simplified_term)
        return self.real_rhs

    def __repr__(self):
        repr_str = f"{self.lhs} <- "
        if self.just_nested_loop:
            repr_str += f""
        else:
            for rhs in self.rhs:
                repr_str += f"{rhs} && "
            repr_str = repr_str[:-4]
        for related_var in self.related_vars:
            repr_str += f" {related_var} "
        for additional_proof in self.additional_proofs:
            repr_str += f" {additional_proof}"
        return repr_str
    
    def deserialize(self, state_cache):
        for rhs in self.rhs:
            rhs.deserialize(state_cache)
        self.to_cvc5(state_cache.solver, state_cache)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't serialize these because they can be reconstructed from the other fields
        state.pop("related_vars")
        state.pop("replace_var")
        state.pop("real_rhs")
        state["replace_var"] = None
        state["real_rhs"] = None
        state["related_vars"] = set()

        return state