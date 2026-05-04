"""Human-readable Boogie feature support matrix.

This is intentionally compact: it is meant to be included in debug sidecars
and reports so unsupported behavior is visible without scanning source code.
"""

from __future__ import annotations

from copy import deepcopy


BOOGIE_SUPPORT_MATRIX = {
    "statements": {
        "supported": [
            "assign",
            "assert",
            "assume",
            "goto",
            "havoc",
            "return",
            "selected SMACK/native calls",
        ],
        "partial": [
            "quantified memset/memcpy assumes",
            "nondet calls via scalar or havoc_seq input",
            "requires clauses as concrete preconditions",
        ],
        "unsupported": [
            "general procedure calls",
            "ensures checking",
            "arbitrary quantified assumptions",
        ],
    },
    "expressions": {
        "supported": [
            "booleans",
            "integers/bitvectors",
            "map load/store",
            "if-then-else",
            "common SMACK bitvector builtins",
        ],
        "partial": [
            "symbolic map load/store only for concrete indices",
            "symbolic branch objectives for SMT-LIB-renderable booleans",
        ],
        "unsupported": [
            "general quantifiers in symbolic formulas",
            "symbolic memory addresses",
            "unmodeled function applications",
        ],
    },
    "execution_engines": {
        "supported": [
            "Rust native concrete interpreter",
            "Rust ProgramInputs concretization",
        ],
        "partial": [
            "Rust memory summary checks",
        ],
        "unsupported": [
            "archived legacy concrete runtime",
            "native bounded concolic branch negation",
            "native bounded symbolic worklist exploration",
            "unbounded all-path symbolic execution",
            "interprocedural symbolic execution",
        ],
    },
}


def support_matrix() -> dict:
    return deepcopy(BOOGIE_SUPPORT_MATRIX)


def support_matrix_summary() -> dict:
    summary = {}
    for area, groups in BOOGIE_SUPPORT_MATRIX.items():
        summary[area] = {
            key: len(values)
            for key, values in groups.items()
        }
    return summary
