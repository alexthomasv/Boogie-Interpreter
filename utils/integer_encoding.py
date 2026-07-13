"""SMACK integer-encoding detection and the interpreter semantics-mode tag.

Canonical, dependency-light home for :func:`detect_integer_encoding` — it is
imported by BOTH the PyPy compile stage (``tools/compile.py``, which cannot
import cvc5) and the CPython verify/trace stages (via
``interpreter.utils.utils_cvc5``, which re-exports it for its existing
consumers). Keep this module stdlib + parser-AST only.

The semantics mode is content-derived from the program (never guessed from
flags): SMACK >= 2.8.0 always emits ``type i32 = int``; the truthful signal
for bit-vector mode is the presence of ``$add.bv*``-style arithmetic
intrinsics.

Mode strings are the wire values used across the manifest
(``integer_encoding`` key), ``swoosh_interp.lower(mode=...)`` and the
``CompiledProgram.mode`` tag inside ``.swcp`` packages.
"""

MODE_INT = "int"
MODE_BV = "bv"


def detect_integer_encoding(program):
    """Detect if program uses SMACK unbounded-integer encoding.

    ``SMACK ≥ 2.8.0`` always emits ``type i32 = int`` as a Boogie alias
    regardless of the actual SMT encoding, but the program then uses
    BV-named function intrinsics (``$add.bv32``, ``$slt.bv32`` etc.)
    when SMACK is invoked with ``--integer-encoding bit-vector``. The
    truthful signal is the presence of ``$add.bv*`` or similar BV
    function declarations: BV mode → present; pure-int mode → absent.

    Returns True iff this is truly unbounded-integer mode (no BV
    function declarations found). Defaults to False (BV) on ambiguity
    so existing BV-tested verification proofs keep working.
    """
    from interpreter.parser.declaration import (
        FunctionDeclaration,
        TypeDeclaration,
    )
    from interpreter.parser.type import IntegerType

    has_int_alias = False
    for d in program.declarations:
        if isinstance(d, TypeDeclaration):
            if hasattr(d, 'names') and 'i32' in d.names:
                if hasattr(d, 'type') and isinstance(d.type, IntegerType):
                    has_int_alias = True
                    break
    if not has_int_alias:
        return False

    # Look for SMACK's BV-mode integer arithmetic intrinsics — present
    # only when SMACK was invoked with --integer-encoding bit-vector for
    # the iN program types. Pointer/ref conversion helpers like
    # ``$bv2int.64`` and ``$int2bv.64`` are emitted in both modes (SMACK
    # always uses bv64 for ref) and must NOT be treated as BV-mode
    # markers; restrict the check to the iN-arith intrinsics.
    bv_arith_prefixes = (
        '$add.bv', '$sub.bv', '$mul.bv', '$div.bv', '$rem.bv',
        '$slt.bv', '$sle.bv', '$sgt.bv', '$sge.bv',
        '$ult.bv', '$ule.bv', '$ugt.bv', '$uge.bv',
        '$shl.bv', '$lshr.bv', '$ashr.bv',
        '$and.bv', '$or.bv', '$xor.bv', '$not.bv',
    )
    for d in program.declarations:
        if isinstance(d, FunctionDeclaration):
            name = getattr(d, 'name', '') or ''
            if name.startswith(bv_arith_prefixes):
                return False  # BV-mode arithmetic intrinsics present

    return True


def detect_semantics_mode(program) -> str:
    """Content-derived semantics mode of ``program``: ``"int"`` or ``"bv"``."""
    return MODE_INT if detect_integer_encoding(program) else MODE_BV


def mode_from_integer_encoding(flag) -> str:
    """Map the manifest's boolean ``integer_encoding`` value to a mode string."""
    return MODE_INT if flag else MODE_BV
