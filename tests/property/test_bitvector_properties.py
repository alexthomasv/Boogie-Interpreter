"""Property-style checks for Boogie bitvector helper semantics."""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st

from utils.bitops import (
    _mask,
    _to_signed,
    add_fn,
    and_fn,
    ashr_fn,
    lshr_fn,
    mul_fn,
    not_fn,
    or_fn,
    sdiv_fn,
    sext_fn,
    shl_fn,
    srem_fn,
    sub_fn,
    trunc_fn,
    udiv_fn,
    urem_fn,
    xor_fn,
)


pytestmark = [pytest.mark.property]

BIT_WIDTHS = st.sampled_from([4, 8, 16, 32, 64])
VALUES = st.integers(min_value=-(2**80), max_value=2**80)


def _signed_result(value: int, bits: int) -> int:
    return _to_signed(value, bits)


@settings(max_examples=200, deadline=None)
@given(bits=BIT_WIDTHS, x=VALUES, y=VALUES)
def test_arithmetic_is_modular(bits: int, x: int, y: int) -> None:
    mask = _mask(bits)

    assert add_fn(bits)(x, y) == ((x & mask) + (y & mask)) & mask
    assert sub_fn(bits)(x, y) == ((x & mask) - (y & mask)) & mask
    assert mul_fn(bits)(x, y) == ((x & mask) * (y & mask)) & mask
    assert add_fn(bits)(x, y) == add_fn(bits)(y, x)
    assert mul_fn(bits)(x, y) == mul_fn(bits)(y, x)


@settings(max_examples=200, deadline=None)
@given(bits=BIT_WIDTHS, x=VALUES, y=VALUES)
def test_bitwise_ops_are_masked_and_algebraic(bits: int, x: int, y: int) -> None:
    mask = _mask(bits)

    assert and_fn(bits)(x, y) == ((x & mask) & (y & mask))
    assert or_fn(bits)(x, y) == ((x & mask) | (y & mask))
    assert xor_fn(bits)(x, y) == ((x & mask) ^ (y & mask))
    assert and_fn(bits)(x, y) == and_fn(bits)(y, x)
    assert or_fn(bits)(x, y) == or_fn(bits)(y, x)
    assert xor_fn(bits)(x, y) == xor_fn(bits)(y, x)
    assert not_fn(bits)(not_fn(bits)(x)) == (x & mask)


@settings(max_examples=200, deadline=None)
@given(bits=BIT_WIDTHS, x=VALUES, y=VALUES)
def test_unsigned_division_remainder_reconstructs_dividend(bits: int, x: int, y: int) -> None:
    mask = _mask(bits)
    dividend = x & mask
    divisor = y & mask

    if divisor == 0:
        assert udiv_fn(bits)(x, y) == mask
        assert urem_fn(bits)(x, y) == dividend
        return

    quotient = udiv_fn(bits)(x, y)
    remainder = urem_fn(bits)(x, y)

    assert quotient * divisor + remainder == dividend
    assert 0 <= remainder < divisor


@settings(max_examples=200, deadline=None)
@given(bits=BIT_WIDTHS, x=VALUES, y=VALUES)
def test_signed_division_remainder_reconstructs_dividend(bits: int, x: int, y: int) -> None:
    divisor = _to_signed(y, bits)

    if divisor == 0:
        assert sdiv_fn(bits)(x, y) == _mask(bits)
        assert srem_fn(bits)(x, y) == (x & _mask(bits))
        return

    dividend = _to_signed(x, bits)
    min_int = -(1 << (bits - 1))
    if dividend == min_int and divisor == -1:
        assert sdiv_fn(bits)(x, y) == (dividend & _mask(bits))
        assert srem_fn(bits)(x, y) == 0
        return

    quotient = _signed_result(sdiv_fn(bits)(x, y), bits)
    remainder = _signed_result(srem_fn(bits)(x, y), bits)

    assert quotient * divisor + remainder == dividend
    assert abs(remainder) < abs(divisor)
    if remainder:
        assert (remainder < 0) == (dividend < 0)


@settings(max_examples=200, deadline=None)
@given(bits=BIT_WIDTHS, x=VALUES, shift=st.integers(min_value=0, max_value=255))
def test_shift_operations_match_declared_semantics(bits: int, x: int, shift: int) -> None:
    mask = _mask(bits)
    normalized_shift = shift % bits
    bounded_shift = shift & (bits - 1)

    assert lshr_fn(bits)(x, shift) == ((x & mask) >> normalized_shift) & mask
    assert shl_fn(bits)(x, shift) == ((x & mask) << bounded_shift) & mask

    signed = _to_signed(x, bits)
    assert ashr_fn(bits)(x, shift) == (signed >> normalized_shift) & mask


@settings(max_examples=200, deadline=None)
@given(
    src_bits=st.sampled_from([4, 8, 16, 32]),
    extra_bits=st.sampled_from([0, 1, 8, 16, 32]),
    x=VALUES,
)
def test_truncation_and_sign_extension_preserve_expected_bits(
    src_bits: int,
    extra_bits: int,
    x: int,
) -> None:
    dst_bits = src_bits + extra_bits
    truncated = trunc_fn(src_bits)(x)
    extended = sext_fn(src_bits, dst_bits)(x)

    assert truncated == x & _mask(src_bits)
    assert extended & _mask(src_bits) == truncated
    assert _to_signed(extended, dst_bits) == _to_signed(x, src_bits)


@pytest.mark.exhaustive
def test_exhaustive_four_bit_unsigned_division_remainder_contract() -> None:
    bits = 4
    mask = _mask(bits)

    for dividend in range(mask + 1):
        for divisor in range(mask + 1):
            quotient = udiv_fn(bits)(dividend, divisor)
            remainder = urem_fn(bits)(dividend, divisor)
            if divisor == 0:
                assert quotient == mask
                assert remainder == dividend
            else:
                assert quotient * divisor + remainder == dividend
                assert 0 <= remainder < divisor
