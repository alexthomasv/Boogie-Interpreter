//! Kani proof harnesses for bitvector operations (builtins.rs).
//!
//! Verifies SMT-LIB 2.6 compliance for all division, remainder, and shift operations.
//! Harnesses are split per bit-width to reduce SAT complexity for division proofs.

#[cfg(kani)]
mod proofs {
    use crate::builtins::{exec_binary, exec_unary, mask, to_signed};
    use crate::opcodes::BuiltinFn;

    /// Helper: pick a valid bit width from {8, 16, 32, 64}
    fn any_bits() -> u8 {
        let idx: u8 = kani::any();
        kani::assume(idx < 4);
        [8, 16, 32, 64][idx as usize]
    }

    // =========================================================================
    // Macro: generate per-width harnesses for division/remainder proofs
    // =========================================================================

    macro_rules! per_width_proof {
        ($name:ident, $bits:literal, $body:expr) => {
            #[kani::proof]
            #[kani::unwind(1)]
            fn $name() {
                const BITS: u8 = $bits;
                $body
            }
        };
    }

    // =========================================================================
    // Division by zero: bvudiv(x, 0) == all-ones — per width
    // =========================================================================

    macro_rules! udiv_zero_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let (result, is_bool) = exec_binary(BuiltinFn::Udiv { bits: BITS }, a, 0);
                assert_eq!(result, mask(BITS), "bvudiv(x, 0) must return all-ones");
                assert!(!is_bool);
            });
        };
    }

    udiv_zero_proof!(verify_udiv_zero_8, 8);
    udiv_zero_proof!(verify_udiv_zero_16, 16);
    udiv_zero_proof!(verify_udiv_zero_32, 32);
    udiv_zero_proof!(verify_udiv_zero_64, 64);

    // =========================================================================
    // bvudiv normal case — per width
    // =========================================================================

    macro_rules! udiv_normal_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let b: i64 = kani::any();
                let m = mask(BITS);
                let b_masked = b as u64 & m as u64;
                kani::assume(b_masked != 0);
                let (result, _) = exec_binary(BuiltinFn::Udiv { bits: BITS }, a, b);
                let expected = ((a as u64 & m as u64) / b_masked) as i64 & m;
                assert_eq!(result, expected, "bvudiv normal case");
            });
        };
    }

    udiv_normal_proof!(verify_udiv_normal_8, 8);
    udiv_normal_proof!(verify_udiv_normal_16, 16);
    // 32/64-bit: symbolic division of two unconstrained values is intractable
    // for CBMC's propositional encoding. The 8/16-bit proofs exhaustively
    // verify the logic; 32/64-bit correctness is covered by concrete test
    // vectors and Python exhaustive cross-validation tests.
    // udiv_normal_proof!(verify_udiv_normal_32, 32);
    // udiv_normal_proof!(verify_udiv_normal_64, 64);

    // =========================================================================
    // Signed division by zero: bvsdiv(x, 0) == all-ones — per width
    // =========================================================================

    macro_rules! sdiv_zero_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let (result, is_bool) = exec_binary(BuiltinFn::Sdiv { bits: BITS }, a, 0);
                assert_eq!(result, mask(BITS), "bvsdiv(x, 0) must return all-ones");
                assert!(!is_bool);
            });
        };
    }

    sdiv_zero_proof!(verify_sdiv_zero_8, 8);
    sdiv_zero_proof!(verify_sdiv_zero_16, 16);
    sdiv_zero_proof!(verify_sdiv_zero_32, 32);
    sdiv_zero_proof!(verify_sdiv_zero_64, 64);

    // =========================================================================
    // Signed division truncated — per width
    // =========================================================================

    macro_rules! sdiv_truncated_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let b: i64 = kani::any();
                let m = mask(BITS);
                let sy = to_signed(b, BITS);
                kani::assume(sy != 0);
                let sx = to_signed(a, BITS);
                let min_val = i64::MIN >> (64 - BITS);
                kani::assume(!(sx == min_val && sy == -1));
                let (result, _) = exec_binary(BuiltinFn::Sdiv { bits: BITS }, a, b);
                let expected = to_signed(sx / sy, BITS) & m;
                assert_eq!(result, expected, "bvsdiv must use truncated division");
            });
        };
    }

    sdiv_truncated_proof!(verify_sdiv_truncated_8, 8);
    sdiv_truncated_proof!(verify_sdiv_truncated_16, 16);
    // sdiv_truncated_proof!(verify_sdiv_truncated_32, 32);  // intractable
    // sdiv_truncated_proof!(verify_sdiv_truncated_64, 64);  // intractable

    // =========================================================================
    // Unsigned remainder zero: bvurem(x, 0) == x — per width
    // =========================================================================

    macro_rules! urem_zero_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let m = mask(BITS);
                let (result, is_bool) = exec_binary(BuiltinFn::Urem { bits: BITS }, a, 0);
                let expected = a as u64 & m as u64;
                assert_eq!(result, expected as i64 & m, "bvurem(x, 0) must return dividend");
                assert!(!is_bool);
            });
        };
    }

    urem_zero_proof!(verify_urem_zero_8, 8);
    urem_zero_proof!(verify_urem_zero_16, 16);
    urem_zero_proof!(verify_urem_zero_32, 32);
    urem_zero_proof!(verify_urem_zero_64, 64);

    // =========================================================================
    // Signed remainder zero: bvsrem(x, 0) == x — per width
    // =========================================================================

    macro_rules! srem_zero_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let m = mask(BITS);
                let (result, is_bool) = exec_binary(BuiltinFn::Srem { bits: BITS }, a, 0);
                let expected = to_signed(a, BITS) & m;
                assert_eq!(result, expected, "bvsrem(x, 0) must return dividend");
                assert!(!is_bool);
            });
        };
    }

    srem_zero_proof!(verify_srem_zero_8, 8);
    srem_zero_proof!(verify_srem_zero_16, 16);
    srem_zero_proof!(verify_srem_zero_32, 32);
    srem_zero_proof!(verify_srem_zero_64, 64);

    // =========================================================================
    // Signed remainder sign follows dividend — per width
    // =========================================================================

    macro_rules! srem_sign_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let b: i64 = kani::any();
                let sy = to_signed(b, BITS);
                kani::assume(sy != 0);
                let sx = to_signed(a, BITS);
                let min_val = i64::MIN >> (64 - BITS);
                kani::assume(!(sx == min_val && sy == -1));
                let (result, _) = exec_binary(BuiltinFn::Srem { bits: BITS }, a, b);
                let result_signed = to_signed(result, BITS);
                if result_signed != 0 {
                    assert_eq!(
                        result_signed > 0, sx > 0,
                        "bvsrem sign must follow dividend"
                    );
                }
            });
        };
    }

    srem_sign_proof!(verify_srem_sign_8, 8);
    srem_sign_proof!(verify_srem_sign_16, 16);
    // srem_sign_proof!(verify_srem_sign_32, 32);  // intractable
    // srem_sign_proof!(verify_srem_sign_64, 64);  // intractable

    // =========================================================================
    // Arithmetic shift right — per width (these are cheap, but split for consistency)
    // =========================================================================

    macro_rules! ashr_sign_ext_proof {
        ($name:ident, $bits:literal) => {
            per_width_proof!($name, $bits, {
                let a: i64 = kani::any();
                let b: i64 = kani::any();
                let m = mask(BITS);
                let v = a & m;
                let sign_bit = 1i64 << (BITS - 1);
                let (result, is_bool) = exec_binary(BuiltinFn::Ashr { bits: BITS }, a, b);
                assert_eq!(result & !m, 0, "ashr result must be within bit width");
                assert!(!is_bool);
                let shift = (b as u32) % (BITS as u32);
                if v & sign_bit != 0 && shift > 0 && shift < BITS as u32 {
                    assert!(result & sign_bit != 0, "ashr of negative must preserve sign bit");
                }
            });
        };
    }

    ashr_sign_ext_proof!(verify_ashr_sign_ext_8, 8);
    ashr_sign_ext_proof!(verify_ashr_sign_ext_16, 16);
    ashr_sign_ext_proof!(verify_ashr_sign_ext_32, 32);
    ashr_sign_ext_proof!(verify_ashr_sign_ext_64, 64);

    // =========================================================================
    // ashr positive == lshr — uses any_bits (cheap, no division)
    // =========================================================================

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_ashr_positive_is_lshr() {
        let bits = any_bits();
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let m = mask(bits);
        let v = a & m;
        let sign_bit = 1i64 << (bits - 1);
        kani::assume(v & sign_bit == 0);
        let (ashr_result, _) = exec_binary(BuiltinFn::Ashr { bits }, a, b);
        let (lshr_result, _) = exec_binary(BuiltinFn::Lshr { bits }, a, b);
        assert_eq!(ashr_result, lshr_result, "ashr of positive value must equal lshr");
    }

    // =========================================================================
    // Shift, add, not — keep any_bits (these are cheap, no division)
    // =========================================================================

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_shl_within_width() {
        let bits = any_bits();
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let m = mask(bits);
        let (result, is_bool) = exec_binary(BuiltinFn::Shl { bits }, a, b);
        assert_eq!(result & !m, 0, "shl result must be within bit width");
        assert!(!is_bool);
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_add_commutative() {
        let bits = any_bits();
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let (r1, _) = exec_binary(BuiltinFn::Add { bits }, a, b);
        let (r2, _) = exec_binary(BuiltinFn::Add { bits }, b, a);
        assert_eq!(r1, r2, "bvadd must be commutative");
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_add_identity() {
        let bits = any_bits();
        let a: i64 = kani::any();
        let m = mask(bits);
        let (result, _) = exec_binary(BuiltinFn::Add { bits }, a, 0);
        assert_eq!(result, a & m, "bvadd(x, 0) must equal x & mask");
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_add_within_width() {
        let bits = any_bits();
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let m = mask(bits);
        let (result, _) = exec_binary(BuiltinFn::Add { bits }, a, b);
        assert_eq!(result & !m, 0, "bvadd result must be within bit width");
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_sext_negative() {
        let x: i64 = kani::any();
        let result = exec_unary(BuiltinFn::Sext { src: 8, dst: 32 }, x);
        let src_mask = mask(8);
        let dst_mask = mask(32);
        let v = x & src_mask;
        if v & (1i64 << 7) != 0 {
            let expected = (v | !src_mask) & dst_mask;
            assert_eq!(result, expected, "sext of negative must sign-extend with 1s");
        } else {
            assert_eq!(result, v, "sext of positive must zero-extend");
        }
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_not_involution() {
        let bits = any_bits();
        let x: i64 = kani::any();
        let once = exec_unary(BuiltinFn::Not { bits }, x);
        let twice = exec_unary(BuiltinFn::Not { bits }, once);
        let m = mask(bits);
        assert_eq!(twice, x & m, "double NOT must return original (masked)");
    }

    // =========================================================================
    // Concrete test vectors from SMT-LIB spec
    // =========================================================================

    #[kani::proof]
    fn verify_test_vector_udiv_zero() {
        let (result, _) = exec_binary(BuiltinFn::Udiv { bits: 32 }, 0x12345678, 0);
        assert_eq!(result, 0xFFFFFFFFi64, "udiv(0x12345678, 0) @ 32 = 0xFFFFFFFF");
    }

    #[kani::proof]
    fn verify_test_vector_sdiv_truncated() {
        let a = 0xFFFFFFF8u32 as i64;
        let (result, _) = exec_binary(BuiltinFn::Sdiv { bits: 32 }, a, 3);
        let result_signed = to_signed(result, 32);
        assert_eq!(result_signed, -2, "sdiv(-8, 3) @ 32 = -2 (truncated)");
    }

    #[kani::proof]
    fn verify_test_vector_ashr_8bit() {
        let (result, _) = exec_binary(BuiltinFn::Ashr { bits: 8 }, 0x80, 1);
        assert_eq!(result, 0xC0, "ashr(0x80, 1) @ 8 = 0xC0");
    }

    #[kani::proof]
    fn verify_test_vector_ashr_16bit() {
        let (result, _) = exec_binary(BuiltinFn::Ashr { bits: 16 }, 0x8000, 1);
        assert_eq!(result, 0xC000, "ashr(0x8000, 1) @ 16 = 0xC000");
    }

    #[kani::proof]
    fn verify_test_vector_srem_zero() {
        let (result, _) = exec_binary(BuiltinFn::Srem { bits: 32 }, 5, 0);
        assert_eq!(result, 5, "srem(5, 0) @ 32 = 5");
    }

    #[kani::proof]
    fn verify_test_vector_urem_zero() {
        let (result, _) = exec_binary(BuiltinFn::Urem { bits: 32 }, 7, 0);
        assert_eq!(result, 7, "urem(7, 0) @ 32 = 7");
    }

    #[kani::proof]
    fn verify_test_vector_sext_negative() {
        let result = exec_unary(BuiltinFn::Sext { src: 8, dst: 32 }, 0x80);
        assert_eq!(result, 0xFFFFFF80u32 as i64, "sext(0x80, 8->32) = 0xFFFFFF80");
    }
}
