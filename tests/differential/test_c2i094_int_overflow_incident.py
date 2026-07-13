"""Incident fixture: c2i_094 false refutation via 32-bit wrap in int mode.

The c2i_094 package is SMACK unbounded-integer encoding (``type i32 = int``,
prelude bodies are plain ``(i1 + i2)``), so its arithmetic is mathematical-
integer arithmetic. The native interpreter used to hardcode wrapping
bit-vector semantics (``$add.i32`` -> masked add, ``$sgt.i32`` ->
``to_signed`` comparison), which falsely "refuted" the root theorem:

    C source:  while (i <= n) { i++; j += i; }  assert(i + j + k > 2*n);
    inputs:    n = 65535 ($i1), k = 0 ($i0)
    math-int:  i = 65536, j = 65536*65537/2 = 2147516416
               i + j + k = 2147581952 > 131070  -> assert HOLDS
    the bug:   $sgt.i32($i12, $i13) reinterpreted $i12 = 2147581952 (>= 2^31)
               as NEGATIVE via to_signed -> comparison false -> the root
               assert ($i15 != $0) at block $bb5 "failed".

``test_c2i094_math_int_root_assert_holds`` pins the CORRECT exact-Z behavior
permanently: under ``SemanticsMode::Int`` the interpreter must evaluate this
package with mathematical-integer semantics (matching the verifier's cvc5
model). Its former strict-xfail marker and the wrap-pinning twin test
(``test_c2i094_incident_wrap_reproduces_today``) were removed when the
exact-Z core landed, exactly as their docstrings instructed.

Read-only with respect to the package: the run uses no_trace and writes its
raw-log path under tmp_path only.
"""

import pickle
from pathlib import Path

import pytest

from interpreter.runner import prepare_native, run_native
from interpreter.utils.inputs import Input, ProgramInputs

pytestmark = [
    pytest.mark.differential,
    pytest.mark.native,
    pytest.mark.requires_compiled_package,
]

_PKG_CANDIDATES = [
    Path("/home/ubuntu/boogie-parser/target/swoosh/package/c2i_094_pkg"),
    Path(__file__).resolve().parents[3] / "target/swoosh/package/c2i_094_pkg",
    Path(__file__).resolve().parents[3] / "test_packages/c2i_094_pkg",
]

# Root-assert dataflow in the compiled package ($bb5):
#   $i11 := $add.i32($i7, $i0);   j + k
#   $i12 := $add.i32($i6, $i11);  i + (j + k)
#   $i13 := $mul.i32(2, $i1);     2 * n
#   $i14 := $sgt.i32($i12, $i13);
#   $i15 := $zext.i1.i32($i14);
#   assert ($i15 != $0);
N = 65535
K = 0
I_FINAL = N + 1                       # 65536
J_FINAL = (N + 1) * (N + 2) // 2      # 2147516416 — exceeds 2^31 - 1
SUM_FINAL = I_FINAL + J_FINAL + K     # 2147581952
TWO_N = 2 * N                         # 131070


def _pkg_dir():
    for cand in _PKG_CANDIDATES:
        if (cand / "c2i_094.pkl").exists():
            return cand
    return None


def _run_incident(tmp_path):
    pkg = _pkg_dir()
    if pkg is None:
        pytest.skip("c2i_094 compiled package not available")
    program = pickle.load(open(pkg / "c2i_094.pkl", "rb"))
    prepared = prepare_native(program, test_path=pkg / "c2i_094.pkl")
    assert prepared.semantics_mode == "int", (
        "c2i_094 must detect as SMACK integer encoding — if this fails the "
        "mode detector regressed and the rest of this fixture is meaningless"
    )
    inputs = ProgramInputs({
        "$i0": Input(name="$i0", private=False, havoc_seq=[K]),
        "$i1": Input(name="$i1", private=False, havoc_seq=[N]),
    })
    return run_native(
        program,
        inputs,
        "c2i_094_incident",
        "n65535_k0",
        tmp_path / "incident.raw.zst",
        no_trace=True,
        log_read=False,
        return_status=True,
        prepared=prepared,
        return_scalar_summary=True,
        return_memory_summary=False,
    )


def test_c2i094_math_int_root_assert_holds(tmp_path):
    """Exact-Z semantics: the root assert holds on n=65535, k=0."""
    result = _run_incident(tmp_path)
    scalars = result.get("final_scalars", {})

    # The accumulator values around the overflow site, as mathematical ints.
    assert scalars.get("$i6") == I_FINAL
    assert scalars.get("$i7") == J_FINAL
    assert scalars.get("$i12") == SUM_FINAL
    assert scalars.get("$i13") == TWO_N
    # The comparison the wrap used to corrupt: 2147581952 > 131070 in Z.
    assert scalars.get("$i14") == 1
    assert scalars.get("$i15") == 1
    assert result["status"] == "ok"
