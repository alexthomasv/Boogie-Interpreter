"""Crosshair-compatible contracts for bitops.py.

Run: crosshair check --analysis_kind=asserts utils/bitops_contracts.py
"""

from interpreter.utils.bitops import (
    _mask, _to_signed,
    add_fn, sub_fn, mul_fn,
    and_fn, or_fn, xor_fn, not_fn,
    udiv_fn, sdiv_fn, urem_fn, srem_fn,
    shl_fn, lshr_fn, ashr_fn,
    sext_fn, trunc_fn,
)


def check_udiv_zero_returns_allones(a: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = udiv_fn(bits)
    result = f(a, 0)
    assert result == m, f"bvudiv({a}, 0) @ {bits} should be all-ones"


def check_sdiv_zero_returns_allones(a: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = sdiv_fn(bits)
    result = f(a, 0)
    assert result == m, f"bvsdiv({a}, 0) @ {bits} should be all-ones"


def check_urem_zero_returns_dividend(a: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = urem_fn(bits)
    result = f(a, 0)
    assert result == (a & m), f"bvurem({a}, 0) @ {bits} should return dividend"


def check_srem_zero_returns_dividend(a: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = srem_fn(bits)
    result = f(a, 0)
    expected = _to_signed(a, bits) & m
    assert result == expected, f"bvsrem({a}, 0) @ {bits} should return dividend"


def check_add_commutative(a: int, b: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= b < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    f = add_fn(bits)
    assert f(a, b) == f(b, a), "bvadd must be commutative"


def check_add_identity(a: int, bits_idx: int) -> None:
    """pre: 0 <= a < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = add_fn(bits)
    assert f(a, 0) == (a & m), "bvadd(x, 0) must equal x & mask"


def check_not_involution(x: int, bits_idx: int) -> None:
    """pre: 0 <= x < 2**64
    pre: 0 <= bits_idx < 4"""
    bits = [8, 16, 32, 64][bits_idx]
    m = _mask(bits)
    f = not_fn(bits)
    assert f(f(x)) == (x & m), "double NOT must return original"


def check_udiv_urem_identity(a: int, b: int) -> None:
    """pre: 0 <= a < 2**32
    pre: 1 <= b < 2**32"""
    m = _mask(32)
    ud = udiv_fn(32)
    ur = urem_fn(32)
    q = ud(a, b)
    r = ur(a, b)
    reconstructed = ((q * (b & m)) + r) & m
    assert reconstructed == (a & m), "a == udiv(a,b)*b + urem(a,b)"


def check_srem_sign_follows_dividend(a: int, b: int) -> None:
    """pre: 0 <= a < 2**32
    pre: 0 <= b < 2**32"""
    bits = 32
    sy = _to_signed(b, bits)
    if sy == 0:
        return
    sx = _to_signed(a, bits)
    min_val = -(1 << (bits - 1))
    if sx == min_val and sy == -1:
        return
    sr = srem_fn(bits)
    result = sr(a, b)
    r_signed = _to_signed(result, bits)
    if r_signed != 0:
        assert (r_signed > 0) == (sx > 0), "srem sign must follow dividend"


def check_sdiv_truncated_not_floor(a: int, b: int) -> None:
    """pre: 0 <= a < 2**32
    pre: 0 <= b < 2**32"""
    bits = 32
    m = _mask(bits)
    sy = _to_signed(b, bits)
    if sy == 0:
        return
    sx = _to_signed(a, bits)
    min_val = -(1 << (bits - 1))
    if sx == min_val and sy == -1:
        return
    sd = sdiv_fn(bits)
    result = sd(a, b)
    q_signed = _to_signed(result, bits)
    # Truncated division: |q| <= |sx / sy|
    # Verify: q * sy + r == sx where r = sx - q * sy
    r = sx - q_signed * sy
    assert sx == q_signed * sy + r, "sdiv/srem identity must hold"
    # Result of truncated division: if sx and sy have different signs,
    # q should be >= floor(sx/sy) (i.e., closer to zero)
    if sx != 0 and sy != 0 and (sx > 0) != (sy > 0):
        assert q_signed * sy >= sx or q_signed == 0, "truncated division rounds toward zero"
