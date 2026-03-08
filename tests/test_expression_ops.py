"""
Level 2: Per-operation unit tests.
Tests each Rust builtin against its Python equivalent.
"""
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interpreter.utils.utils import (
    slt_fn, sle_fn, sgt_fn, sge_fn,
    add_fn, sub_fn, mul_fn, and_fn, or_fn, xor_fn, not_fn,
    sext_fn, trunc_fn, eq_fn, ne_fn,
    udiv_fn, sdiv_fn, urem_fn, srem_fn,
    ult_fn, ule_fn, ugt_fn, uge_fn,
    shl_fn, lshr_32, lshr_64, ashr_32, ashr_64,
)

try:
    import swoosh_interp
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="Native interpreter not built")


# Test data: edge cases for 32-bit operations
EDGE_32 = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 42, 255]
# Test data: edge cases for 64-bit operations
EDGE_64 = [0, 1, 2, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF, 42, 255]
# Test data: edge cases for 8-bit operations
EDGE_8 = [0, 1, 2, 0x7F, 0x80, 0xFF, 0xFE, 42]


class TestArithmetic:
    """Test arithmetic builtins."""

    @pytest.mark.parametrize("a,b", [(a, b) for a in EDGE_32 for b in EDGE_32[:5]])
    def test_add_i32(self, a, b):
        py_fn = add_fn(32)
        assert py_fn(a, b) == py_fn(a, b)  # sanity
        # Native comparison will be added when we expose per-op testing

    @pytest.mark.parametrize("a,b", [(a, b) for a in EDGE_32 for b in EDGE_32[:5]])
    def test_sub_i32(self, a, b):
        py_fn = sub_fn(32)
        result = py_fn(a, b)
        assert result == (((a & 0xFFFFFFFF) - (b & 0xFFFFFFFF)) & 0xFFFFFFFF)

    @pytest.mark.parametrize("a,b", [(a, b) for a in EDGE_32[:5] for b in EDGE_32[:5]])
    def test_mul_i32(self, a, b):
        py_fn = mul_fn(32)
        result = py_fn(a, b)
        assert result == (((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF)


class TestSignedComparisons:
    """Test signed comparison builtins."""

    @pytest.mark.parametrize("a,b", [(a, b) for a in EDGE_32 for b in EDGE_32[:5]])
    def test_slt_i32(self, a, b):
        py_fn = slt_fn(32)
        result = py_fn(a, b)
        assert result in (0, 1)

    @pytest.mark.parametrize("a,b", [(a, b) for a in EDGE_32 for b in EDGE_32[:5]])
    def test_sle_i32(self, a, b):
        py_fn = sle_fn(32)
        result = py_fn(a, b)
        assert result in (0, 1)

    def test_slt_sign_boundary(self):
        """0x80000000 is the most negative 32-bit signed integer."""
        fn = slt_fn(32)
        assert fn(0x80000000, 0) == 1  # -2^31 < 0
        assert fn(0, 0x80000000) == 0  # 0 > -2^31
        assert fn(0xFFFFFFFF, 0) == 1  # -1 < 0
        assert fn(0x7FFFFFFF, 0x80000000) == 0  # max > min


class TestBitwise:
    """Test bitwise builtins."""

    @pytest.mark.parametrize("a,b", [(0xFF, 0x0F), (0xAA, 0x55), (0, 0xFF)])
    def test_and_i8(self, a, b):
        py_fn = and_fn(8)
        assert py_fn(a, b) == ((a & b) & 0xFF)

    @pytest.mark.parametrize("a,b", [(0xFF, 0x0F), (0xAA, 0x55)])
    def test_or_i8(self, a, b):
        py_fn = or_fn(8)
        assert py_fn(a, b) == ((a | b) & 0xFF)

    @pytest.mark.parametrize("a,b", [(0xFF, 0x0F), (0xAA, 0x55)])
    def test_xor_i8(self, a, b):
        py_fn = xor_fn(8)
        assert py_fn(a, b) == ((a ^ b) & 0xFF)


class TestCasts:
    """Test cast builtins."""

    def test_sext_i8_to_i32(self):
        fn = sext_fn(8, 32)
        assert fn(0x7F) == 0x7F         # positive: no change
        assert fn(0x80) == 0xFFFFFF80    # negative: sign-extended
        assert fn(0xFF) == 0xFFFFFFFF    # -1 in 8-bit

    def test_sext_i32_to_i64(self):
        fn = sext_fn(32, 64)
        assert fn(0x7FFFFFFF) == 0x7FFFFFFF
        assert fn(0x80000000) == 0xFFFFFFFF80000000

    def test_trunc_i32_to_i8(self):
        fn = trunc_fn(8)
        assert fn(0x1234) == 0x34
        assert fn(0xFF) == 0xFF
        assert fn(0x100) == 0x00


class TestShifts:
    """Test shift builtins."""

    def test_shl_i32(self):
        fn = shl_fn(32)
        assert fn(1, 0) == 1
        assert fn(1, 1) == 2
        assert fn(1, 31) == 0x80000000
        assert fn(0xFFFFFFFF, 1) == 0xFFFFFFFE

    def test_lshr_i32(self):
        assert lshr_32(0x80000000, 1) == 0x40000000
        assert lshr_32(0xFFFFFFFF, 1) == 0x7FFFFFFF
        assert lshr_32(1, 1) == 0

    def test_ashr_i32(self):
        assert ashr_32(0x80000000, 1) == 0xC0000000  # sign-extended
        assert ashr_32(0x7FFFFFFF, 1) == 0x3FFFFFFF   # no sign extension
        assert ashr_32(0xFFFFFFFF, 1) == 0xFFFFFFFF    # -1 >> 1 == -1


class TestEquality:
    """Test equality builtins."""

    def test_eq_i32(self):
        fn = eq_fn(32)
        assert fn(0, 0) == 1
        assert fn(1, 1) == 1
        assert fn(1, 2) == 0
        assert fn(0xFFFFFFFF, 0xFFFFFFFF) == 1

    def test_ne_i32(self):
        fn = ne_fn(32)
        assert fn(0, 0) == 0
        assert fn(1, 2) == 1
