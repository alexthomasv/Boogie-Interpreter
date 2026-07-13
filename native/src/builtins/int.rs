//! Int-mode (exact mathematical ℤ) builtin semantics.
//!
//! One source of truth: the SMACK **integer-encoding prelude** bodies (see any
//! int-mode `.bpl`, e.g. `$add.i32(i1,i2) = (i1 + i2)`), which are also what
//! the verifier's cvc5 model implements (`utils_cvc5._INT_ENC_FN_MAP` + the
//! int-encoding bitwise handler). Every function here mirrors the prelude
//! definition of the SAME intrinsic name:
//!
//! * `$add/$sub/$mul.iN`      — plain exact arithmetic (no wrap, no mask);
//! * `$eq/$ne/$slt/…/$uge.iN` — exact comparisons (signed == unsigned in ℤ);
//! * `$sext/$zext/$trunc/$p2i/$i2p/$bitcast` — identity (ℤ has no width);
//! * `$idiv.iN` / `$smod.iN`  — SMT-LIB Euclidean `div` / `mod`
//!   ({:builtin "div"/"mod"} in the prelude);
//! * `$sdiv = $udiv = $idiv`, `$urem = $smod`, and `$srem` is the prelude's
//!   C-remainder correction formula
//!   `if smod(a,b) != 0 && a < 0 then smod(a,b) - |b| else smod(a,b)`;
//! * bitwise/shift (`$and/$or/$xor/$not/$shl/$lshr/$ashr.iN`) — the
//!   int→bv(width)→op→nat roundtrip (matching `(_ int2bv w)` … `bv2nat`),
//!   the only width-sensitive residual ops in the encoding.
//!
//! Division/modulus **by zero is a loud panic**: SMT-LIB leaves `div/mod _ 0`
//! uninterpreted, so no concrete value would be faithful — the interpreter
//! refuses rather than invent one (traces refute only; a fabricated value
//! could manufacture a false refutation).

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::opcodes::BuiltinFn;

/// Exact integer value: `S` for the i64 fast path, `B` for out-of-i64.
/// INVARIANT: `B` is only used for values strictly outside i64 range —
/// every constructor normalizes, so comparisons can use the sign shortcut.
#[derive(Debug, Clone)]
pub enum Z {
    S(i64),
    B(Box<BigInt>),
}

/// Result of an int-mode builtin: a numeric value or a Boogie bool.
#[derive(Debug, Clone)]
pub enum ZResult {
    Num(Z),
    Bool(bool),
}

impl Z {
    #[inline]
    pub fn from_big(b: BigInt) -> Z {
        match b.to_i64() {
            Some(v) => Z::S(v),
            None => Z::B(Box::new(b)),
        }
    }

    #[inline]
    pub fn to_bigint(&self) -> BigInt {
        match self {
            Z::S(v) => BigInt::from(*v),
            Z::B(b) => (**b).clone(),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            Z::S(v) => *v == 0,
            Z::B(_) => false, // invariant: Big is outside i64, never 0
        }
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        match self {
            Z::S(v) => *v < 0,
            Z::B(b) => b.is_negative(),
        }
    }
}

/// Exact three-way comparison honoring the "Big is outside i64" invariant.
#[inline]
pub fn cmp(a: &Z, b: &Z) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    match (a, b) {
        (Z::S(x), Z::S(y)) => x.cmp(y),
        (Z::S(_), Z::B(y)) => {
            // y is outside i64: positive ⇒ x < y, negative ⇒ x > y.
            if y.is_negative() {
                Greater
            } else {
                Less
            }
        }
        (Z::B(x), Z::S(_)) => {
            if x.is_negative() {
                Less
            } else {
                Greater
            }
        }
        (Z::B(x), Z::B(y)) => x.cmp(y),
    }
}

#[inline]
pub fn add(a: &Z, b: &Z) -> Z {
    if let (Z::S(x), Z::S(y)) = (a, b) {
        if let Some(v) = x.checked_add(*y) {
            return Z::S(v);
        }
    }
    Z::from_big(a.to_bigint() + b.to_bigint())
}

#[inline]
pub fn sub(a: &Z, b: &Z) -> Z {
    if let (Z::S(x), Z::S(y)) = (a, b) {
        if let Some(v) = x.checked_sub(*y) {
            return Z::S(v);
        }
    }
    Z::from_big(a.to_bigint() - b.to_bigint())
}

#[inline]
pub fn mul(a: &Z, b: &Z) -> Z {
    if let (Z::S(x), Z::S(y)) = (a, b) {
        if let Some(v) = x.checked_mul(*y) {
            return Z::S(v);
        }
    }
    Z::from_big(a.to_bigint() * b.to_bigint())
}

/// SMT-LIB Euclidean division: `a = b*q + r` with `0 <= r < |b|`.
/// Panics loudly on b == 0 (uninterpreted in the model — see module docs).
pub fn euclid_div(a: &Z, b: &Z) -> Z {
    if b.is_zero() {
        panic!(
            "int-mode division by zero: SMT-LIB `div` is uninterpreted at 0 \
             — no faithful concrete value exists (exact-ℤ semantics)"
        );
    }
    if let (Z::S(x), Z::S(y)) = (a, b) {
        if let Some(q) = x.checked_div_euclid(*y) {
            return Z::S(q);
        }
    }
    let (x, y) = (a.to_bigint(), b.to_bigint());
    let (mut q, r) = (&x / &y, &x % &y);
    if r < BigInt::zero() {
        if y > BigInt::zero() {
            q -= BigInt::one();
        } else {
            q += BigInt::one();
        }
    }
    Z::from_big(q)
}

/// SMT-LIB Euclidean modulus: result always in `[0, |b|)`.
/// Panics loudly on b == 0 (uninterpreted in the model — see module docs).
pub fn euclid_mod(a: &Z, b: &Z) -> Z {
    if b.is_zero() {
        panic!(
            "int-mode modulus by zero: SMT-LIB `mod` is uninterpreted at 0 \
             — no faithful concrete value exists (exact-ℤ semantics)"
        );
    }
    if let (Z::S(x), Z::S(y)) = (a, b) {
        if let Some(r) = x.checked_rem_euclid(*y) {
            return Z::S(r);
        }
    }
    let (x, y) = (a.to_bigint(), b.to_bigint());
    let mut r = &x % &y;
    if r < BigInt::zero() {
        r += y.abs();
    }
    Z::from_big(r)
}

/// Prelude `$srem.iN`: C-style remainder derived from Euclidean mod —
/// `if smod(a,b) != 0 && a < 0 then smod(a,b) - smax(b, -b) else smod(a,b)`.
pub fn srem(a: &Z, b: &Z) -> Z {
    let m = euclid_mod(a, b);
    if !m.is_zero() && a.is_negative() {
        let abs_b = Z::from_big(b.to_bigint().abs());
        sub(&m, &abs_b)
    } else {
        m
    }
}

/// `x mod 2^bits` as a nonnegative BigInt — the `(_ int2bv bits)` image
/// read back through `bv2nat`.
fn to_unsigned_bits(x: &Z, bits: u8) -> BigInt {
    let modulus = BigInt::one() << (bits as usize);
    let mut v = x.to_bigint() % &modulus;
    if v.is_negative() {
        v += &modulus;
    }
    v
}

/// `x mod 2^bits` as u64 for widths <= 64 and in-i64 operands — the
/// allocation-free image of `to_unsigned_bits`.
#[inline]
fn small_unsigned_bits(x: i64, bits: u8) -> u64 {
    if bits >= 64 {
        x as u64
    } else {
        (x as u64) & ((1u64 << bits) - 1)
    }
}

/// Fold a `[0, 2^bits)` u64 result back into the exact-integer domain.
/// Only 64-bit results can exceed i64 (bv2nat is unsigned).
#[inline]
fn unsigned_result(v: u64) -> Z {
    if v <= i64::MAX as u64 {
        Z::S(v as i64)
    } else {
        Z::B(Box::new(BigInt::from(v)))
    }
}

/// Allocation-free bitwise/shift roundtrip for widths <= 64 with in-i64
/// operands. Semantics identical to the BigInt path below (SMT-LIB
/// `bv2nat(bvop(int2bv a, int2bv b))`), including shift saturation.
#[inline]
fn small_roundtrip_binary(fn_id: BuiltinFn, bits: u8, a: i64, b: i64) -> Z {
    let x = small_unsigned_bits(a, bits);
    match fn_id {
        BuiltinFn::And { .. } => unsigned_result(x & small_unsigned_bits(b, bits)),
        BuiltinFn::Or { .. } => unsigned_result(x | small_unsigned_bits(b, bits)),
        BuiltinFn::Xor { .. } => unsigned_result(x ^ small_unsigned_bits(b, bits)),
        BuiltinFn::Shl { .. } | BuiltinFn::Lshr { .. } | BuiltinFn::Ashr { .. } => {
            let s = small_unsigned_bits(b, bits);
            let saturated = s >= bits as u64;
            let v = match fn_id {
                BuiltinFn::Shl { .. } => {
                    if saturated {
                        0
                    } else {
                        small_unsigned_bits((x << s) as i64, bits)
                    }
                }
                BuiltinFn::Lshr { .. } => {
                    if saturated {
                        0
                    } else {
                        x >> s
                    }
                }
                BuiltinFn::Ashr { .. } => {
                    let msb_set = bits > 0 && (x >> (bits - 1)) & 1 == 1;
                    if saturated {
                        if msb_set {
                            small_unsigned_bits(-1, bits)
                        } else {
                            0
                        }
                    } else if msb_set {
                        // Sign-extend x to 64 bits, arithmetic shift, re-mask.
                        let sign_extended = if bits >= 64 {
                            x as i64
                        } else {
                            (x | (u64::MAX << bits)) as i64
                        };
                        small_unsigned_bits(sign_extended >> s, bits)
                    } else {
                        x >> s
                    }
                }
                _ => unreachable!(),
            };
            unsigned_result(v)
        }
        _ => unreachable!("small_roundtrip_binary: {:?}", fn_id),
    }
}

/// Bitwise/shift ops as int→bv(bits)→op→nat, matching the SMT-LIB semantics
/// of the corresponding BV operator applied to `(_ int2bv bits)` operands,
/// with the result read back via `bv2nat` (always in `[0, 2^bits)`).
fn bv_roundtrip_binary(fn_id: BuiltinFn, a: &Z, b: &Z) -> Z {
    let (op_bits, is_shift): (u8, bool) = match fn_id {
        BuiltinFn::And { bits } | BuiltinFn::Or { bits } | BuiltinFn::Xor { bits } => (bits, false),
        BuiltinFn::Shl { bits } | BuiltinFn::Lshr { bits } | BuiltinFn::Ashr { bits } => {
            (bits, true)
        }
        _ => unreachable!("bv_roundtrip_binary: {:?}", fn_id),
    };
    if op_bits <= 64 {
        if let (Z::S(x), Z::S(y)) = (a, b) {
            return small_roundtrip_binary(fn_id, op_bits, *x, *y);
        }
    }
    let x = to_unsigned_bits(a, op_bits);
    let modulus = BigInt::one() << (op_bits as usize);
    if !is_shift {
        let y = to_unsigned_bits(b, op_bits);
        let r = match fn_id {
            BuiltinFn::And { .. } => x & y,
            BuiltinFn::Or { .. } => x | y,
            BuiltinFn::Xor { .. } => x ^ y,
            _ => unreachable!(),
        };
        return Z::from_big(r);
    }
    // Shift amount is ALSO a bv(bits) operand: reduce mod 2^bits, then apply
    // SMT-LIB saturation (shift >= width ⇒ 0 / sign-fill).
    let s = to_unsigned_bits(b, op_bits);
    let saturated = s >= BigInt::from(op_bits as u64);
    let r = match fn_id {
        BuiltinFn::Shl { .. } => {
            if saturated {
                BigInt::zero()
            } else {
                (x << s.to_usize().unwrap()) % &modulus
            }
        }
        BuiltinFn::Lshr { .. } => {
            if saturated {
                BigInt::zero()
            } else {
                x >> s.to_usize().unwrap()
            }
        }
        BuiltinFn::Ashr { .. } => {
            let msb_set = &x >= &(&modulus >> 1);
            if saturated {
                if msb_set {
                    &modulus - BigInt::one()
                } else {
                    BigInt::zero()
                }
            } else {
                let sh = s.to_usize().unwrap();
                if msb_set {
                    // Sign-extend: shift the (negative) signed reading, re-mask.
                    let signed = &x - &modulus; // in [-2^(bits-1), 0)
                    let shifted = signed >> sh; // arithmetic shift (floor)
                    let mut v = shifted % &modulus;
                    if v.is_negative() {
                        v += &modulus;
                    }
                    v
                } else {
                    x >> sh
                }
            }
        }
        _ => unreachable!(),
    };
    Z::from_big(r)
}

/// Execute a unary builtin under exact-ℤ semantics.
#[inline]
pub fn exec_unary(fn_id: BuiltinFn, x: &Z) -> Z {
    match fn_id {
        // Prelude int-mode casts are identity: ℤ has no representation width.
        BuiltinFn::Sext { .. }
        | BuiltinFn::Zext { .. }
        | BuiltinFn::Trunc { .. }
        | BuiltinFn::Bitcast
        | BuiltinFn::P2i
        | BuiltinFn::I2p => x.clone(),
        // $not.iN — int→bv→bvnot→nat: 2^bits - 1 - (x mod 2^bits).
        BuiltinFn::Not { bits } => {
            if bits <= 64 {
                if let Z::S(v) = x {
                    let m = small_unsigned_bits(*v, bits);
                    return unsigned_result(small_unsigned_bits(!(m as i64), bits));
                }
            }
            let modulus = BigInt::one() << (bits as usize);
            Z::from_big(&modulus - BigInt::one() - to_unsigned_bits(x, bits))
        }
        _ => unreachable!("int::exec_unary called with binary fn: {:?}", fn_id),
    }
}

/// Execute a binary builtin under exact-ℤ semantics.
#[inline]
pub fn exec_binary(fn_id: BuiltinFn, a: &Z, b: &Z) -> ZResult {
    use std::cmp::Ordering::*;
    let ord = || cmp(a, b);
    match fn_id {
        // Exact arithmetic — the prelude bodies are plain (i1 ± i2), (i1 * i2).
        BuiltinFn::Add { .. } => ZResult::Num(add(a, b)),
        BuiltinFn::Sub { .. } => ZResult::Num(sub(a, b)),
        BuiltinFn::Mul { .. } => ZResult::Num(mul(a, b)),

        // Exact comparisons. In ℤ the prelude defines signed AND unsigned
        // variants identically as (i1 < i2) etc.
        BuiltinFn::Slt { .. } | BuiltinFn::Ult { .. } => {
            ZResult::Num(Z::S((ord() == Less) as i64))
        }
        BuiltinFn::Sle { .. } | BuiltinFn::Ule { .. } => {
            ZResult::Num(Z::S((ord() != Greater) as i64))
        }
        BuiltinFn::Sgt { .. } | BuiltinFn::Ugt { .. } => {
            ZResult::Num(Z::S((ord() == Greater) as i64))
        }
        BuiltinFn::Sge { .. } | BuiltinFn::Uge { .. } => {
            ZResult::Num(Z::S((ord() != Less) as i64))
        }
        BuiltinFn::SltBool { .. } => ZResult::Bool(ord() == Less),
        BuiltinFn::SleBool { .. } => ZResult::Bool(ord() != Greater),
        BuiltinFn::SgtBool { .. } => ZResult::Bool(ord() == Greater),
        BuiltinFn::SgeBool { .. } => ZResult::Bool(ord() != Less),
        BuiltinFn::BvEq { .. } => ZResult::Num(Z::S((ord() == Equal) as i64)),
        BuiltinFn::BvNe { .. } => ZResult::Num(Z::S((ord() != Equal) as i64)),

        // Division / remainder per the prelude formulas.
        BuiltinFn::Idiv { .. } => ZResult::Num(euclid_div(a, b)),
        BuiltinFn::Smod { .. } => ZResult::Num(euclid_mod(a, b)),
        BuiltinFn::Sdiv { .. } | BuiltinFn::Udiv { .. } => ZResult::Num(euclid_div(a, b)),
        BuiltinFn::Urem { .. } => ZResult::Num(euclid_mod(a, b)),
        BuiltinFn::Srem { .. } => ZResult::Num(srem(a, b)),

        // Bitwise / shifts via the int→bv(width)→op→nat roundtrip.
        BuiltinFn::And { .. }
        | BuiltinFn::Or { .. }
        | BuiltinFn::Xor { .. }
        | BuiltinFn::Shl { .. }
        | BuiltinFn::Lshr { .. }
        | BuiltinFn::Ashr { .. } => ZResult::Num(bv_roundtrip_binary(fn_id, a, b)),

        _ => unreachable!("int::exec_binary called with unary fn: {:?}", fn_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes::BuiltinFn as F;

    fn s(v: i64) -> Z {
        Z::S(v)
    }

    fn num(r: ZResult) -> i64 {
        match r {
            ZResult::Num(Z::S(v)) => v,
            other => panic!("expected small num, got {:?}", other),
        }
    }

    #[test]
    fn euclidean_div_mod_match_smtlib() {
        // (div -7 3) = -3, (mod -7 3) = 2 ; (div 7 -3) = -2, (mod 7 -3) = 1
        assert_eq!(num(exec_binary(F::Idiv { bits: 32 }, &s(-7), &s(3))), -3);
        assert_eq!(num(exec_binary(F::Smod { bits: 32 }, &s(-7), &s(3))), 2);
        assert_eq!(num(exec_binary(F::Idiv { bits: 32 }, &s(7), &s(-3))), -2);
        assert_eq!(num(exec_binary(F::Smod { bits: 32 }, &s(7), &s(-3))), 1);
    }

    #[test]
    fn srem_matches_prelude_formula() {
        // C-style: -7 % 3 = -1, 7 % -3 = 1, -7 % -3 = -1, 7 % 3 = 1
        assert_eq!(num(exec_binary(F::Srem { bits: 32 }, &s(-7), &s(3))), -1);
        assert_eq!(num(exec_binary(F::Srem { bits: 32 }, &s(7), &s(-3))), 1);
        assert_eq!(num(exec_binary(F::Srem { bits: 32 }, &s(-7), &s(-3))), -1);
        assert_eq!(num(exec_binary(F::Srem { bits: 32 }, &s(7), &s(3))), 1);
    }

    #[test]
    fn bitwise_roundtrip_masks_to_width() {
        // $and.i32(-1, 15) : int2bv32(-1)=0xFFFFFFFF & 15 → 15
        assert_eq!(num(exec_binary(F::And { bits: 32 }, &s(-1), &s(15))), 15);
        // $not.i32(12) = 2^32 - 1 - 12
        match exec_unary(F::Not { bits: 32 }, &s(12)) {
            Z::S(v) => assert_eq!(v, 4294967283),
            other => panic!("unexpected {:?}", other),
        }
        // $lshr.i32(-8, 2): (2^32-8)>>2 = 0x3FFFFFFE
        assert_eq!(
            num(exec_binary(F::Lshr { bits: 32 }, &s(-8), &s(2))),
            0x3FFF_FFFE
        );
        // $ashr.i32(-8, 2): sign-fill → 2^32 - 2
        assert_eq!(
            num(exec_binary(F::Ashr { bits: 32 }, &s(-8), &s(2))),
            4294967294
        );
    }

    #[test]
    fn arithmetic_promotes_past_i64() {
        let r = exec_binary(F::Add { bits: 64 }, &s(i64::MAX), &s(1));
        match r {
            ZResult::Num(Z::B(b)) => {
                assert_eq!(*b, BigInt::from(i64::MAX) + 1);
            }
            other => panic!("expected Big, got {:?}", other),
        }
        // ... and normalizes back down when the result fits.
        let big = Z::from_big(BigInt::from(i64::MAX) + 1);
        match sub(&big, &s(1)) {
            Z::S(v) => assert_eq!(v, i64::MAX),
            other => panic!("expected Small, got {:?}", other),
        }
    }
}
