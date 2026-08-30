from __future__ import annotations

import cvc5
from cvc5 import Kind

from interpreter.utils.cvc5_serde import canonical_wire
from interpreter.utils.utils_cvc5 import cvc5_to_boogie


def test_canonical_wire_lowers_to_boogie_without_solver_rehydration():
    solver = cvc5.Solver()
    solver.setLogic("ALL")
    integer = solver.getIntegerSort()
    i0 = solver.mkConst(integer, "$i0")
    i6 = solver.mkConst(integer, "$i6")
    term = solver.mkTerm(
        Kind.IMPLIES,
        solver.mkTerm(Kind.GEQ, i0, solver.mkInteger(1)),
        solver.mkTerm(
            Kind.GEQ,
            solver.mkTerm(
                Kind.ADD,
                solver.mkTerm(Kind.MULT, i6, solver.mkInteger(-1)),
                i0,
                solver.mkInteger(-1),
            ),
            solver.mkInteger(0),
        ),
    )

    wire = canonical_wire(term)

    assert cvc5_to_boogie(wire) == (
        "(($i0 >= 1) ==> (((($i6 * -1) + $i0) + -1) >= 0))"
    )
