use crate::opcodes::BuiltinFn;

#[inline]
pub fn mask(bits: u8) -> i64 {
    if bits >= 64 {
        -1i64 // all bits set = u64::MAX as i64
    } else {
        (1i64 << bits) - 1
    }
}

/// Interpret a value as signed in the given bit width.
#[inline]
pub fn to_signed(val: i64, bits: u8) -> i64 {
    let m = mask(bits);
    let v = val & m;
    if bits < 64 && v & (1i64 << (bits - 1)) != 0 {
        v - (1i64 << bits)
    } else {
        v
    }
}

/// Execute a unary builtin function.
#[inline]
pub fn exec_unary(fn_id: BuiltinFn, x: i64) -> i64 {
    match fn_id {
        BuiltinFn::Not { bits } => {
            let m = mask(bits);
            (!x) & m
        }
        BuiltinFn::Sext { src, dst } => {
            let src_mask = mask(src);
            let dst_mask = mask(dst);
            let sign_bit = 1i64 << (src - 1);
            let mut v = x & src_mask;
            if v & sign_bit != 0 {
                v |= !src_mask;
            }
            v & dst_mask
        }
        BuiltinFn::Zext { .. } | BuiltinFn::Bitcast | BuiltinFn::P2i | BuiltinFn::I2p => x,
        BuiltinFn::Trunc { dst } => x & mask(dst),
        _ => unreachable!("exec_unary called with binary fn: {:?}", fn_id),
    }
}

/// Execute a binary builtin function.
/// Returns (result, is_bool) where is_bool indicates the result should be treated as boolean.
#[inline]
pub fn exec_binary(fn_id: BuiltinFn, a: i64, b: i64) -> (i64, bool) {
    match fn_id {
        // Arithmetic
        BuiltinFn::Add { bits } => {
            let m = mask(bits);
            (((a & m).wrapping_add(b & m)) & m, false)
        }
        BuiltinFn::Sub { bits } => {
            let m = mask(bits);
            (((a & m).wrapping_sub(b & m)) & m, false)
        }
        BuiltinFn::Mul { bits } => {
            let m = mask(bits);
            (((a & m).wrapping_mul(b & m)) & m, false)
        }

        // Bitwise
        BuiltinFn::And { bits } => {
            let m = mask(bits);
            ((a & m) & (b & m) & m, false)
        }
        BuiltinFn::Or { bits } => {
            let m = mask(bits);
            (((a & m) | (b & m)) & m, false)
        }
        BuiltinFn::Xor { bits } => {
            let m = mask(bits);
            (((a & m) ^ (b & m)) & m, false)
        }

        // Shifts
        BuiltinFn::Shl { bits } => {
            let m = mask(bits);
            let shift = (b as u32) & (bits as u32 - 1);
            (((a & m) << shift) & m, false)
        }
        BuiltinFn::Lshr { bits } => {
            let m = mask(bits);
            let v = (a & m) as u64 & (m as u64);
            let shift = (b as u32) & (bits as u32 - 1);
            ((v >> shift) as i64 & m, false)
        }
        // pf:ensures:ashr.sign_extension - high bits filled with sign bit copies
        // pf:ensures:ashr.generic_width - works for all bit widths (8, 16, 32, 64)
        BuiltinFn::Ashr { bits } => {
            let m = mask(bits);
            let v = a & m;
            // Shift amount is reduced modulo bit width per SMT-LIB
            let shift = (b as u32) % (bits as u32);
            let sign_bit = 1i64 << (bits - 1);
            if v & sign_bit != 0 {
                // Negative value - sign extend with 1s
                let fill = if shift == 0 {
                    0
                } else {
                    !0i64 << (bits as i64 - shift as i64)
                };
                (((v >> shift) | fill) & m, false)
            } else {
                // Positive value - zero extend
                ((v >> shift) & m, false)
            }
        }

        // Signed comparisons
        BuiltinFn::Slt { bits } => {
            let result = to_signed(a, bits) < to_signed(b, bits);
            (result as i64, false)
        }
        BuiltinFn::Sle { bits } => {
            let result = to_signed(a, bits) <= to_signed(b, bits);
            (result as i64, false)
        }
        BuiltinFn::Sgt { bits } => {
            let result = to_signed(a, bits) > to_signed(b, bits);
            (result as i64, false)
        }
        BuiltinFn::Sge { bits } => {
            let result = to_signed(a, bits) >= to_signed(b, bits);
            (result as i64, false)
        }

        // Boolean result variants
        BuiltinFn::SltBool { bits } => {
            let result = to_signed(a, bits) < to_signed(b, bits);
            (result as i64, true)
        }
        BuiltinFn::SleBool { bits } => {
            let result = to_signed(a, bits) <= to_signed(b, bits);
            (result as i64, true)
        }
        BuiltinFn::SgtBool { bits } => {
            let result = to_signed(a, bits) > to_signed(b, bits);
            (result as i64, true)
        }
        BuiltinFn::SgeBool { bits } => {
            let result = to_signed(a, bits) >= to_signed(b, bits);
            (result as i64, true)
        }

        // Unsigned comparisons
        BuiltinFn::Ult { bits } => {
            let m = mask(bits) as u64;
            let result = (a as u64 & m) < (b as u64 & m);
            (result as i64, false)
        }
        BuiltinFn::Ule { bits } => {
            let m = mask(bits) as u64;
            let result = (a as u64 & m) <= (b as u64 & m);
            (result as i64, false)
        }
        BuiltinFn::Ugt { bits } => {
            let m = mask(bits) as u64;
            let result = (a as u64 & m) > (b as u64 & m);
            (result as i64, false)
        }
        BuiltinFn::Uge { bits } => {
            let m = mask(bits) as u64;
            let result = (a as u64 & m) >= (b as u64 & m);
            (result as i64, false)
        }

        // Equality
        BuiltinFn::BvEq { bits } => {
            let m = mask(bits);
            let result = (a & m) == (b & m);
            (result as i64, false)
        }
        BuiltinFn::BvNe { bits } => {
            let m = mask(bits);
            let result = (a & m) != (b & m);
            (result as i64, false)
        }

        // Division
        // pf:ensures:udiv.zero_divisor - returns all-ones when divisor is zero (SMT-LIB bvudiv)
        BuiltinFn::Udiv { bits } => {
            let m = mask(bits);
            let m_u = m as u64;
            let a_masked = a as u64 & m_u;
            let b_masked = b as u64 & m_u;
            if b_masked == 0 {
                return (m, false); // all-ones per SMT-LIB
            }
            let result = a_masked / b_masked;
            (result as i64 & m, false)
        }
        // pf:ensures:sdiv.zero_divisor - returns all-ones when divisor is zero (SMT-LIB bvsdiv)
        // pf:ensures:sdiv.truncated_division - Rust's / is truncated division (toward zero)
        BuiltinFn::Sdiv { bits } => {
            let sx = to_signed(a, bits);
            let sy = to_signed(b, bits);
            let m = mask(bits);
            if sy == 0 {
                return (m, false); // all-ones per SMT-LIB
            }
            (to_signed(sx / sy, bits) & m, false)
        }

        // Remainder
        // pf:ensures:urem.zero_divisor - returns dividend when divisor is zero (SMT-LIB bvurem)
        BuiltinFn::Urem { bits } => {
            let m = mask(bits);
            let m_u = m as u64;
            let a_masked = a as u64 & m_u;
            let b_masked = b as u64 & m_u;
            if b_masked == 0 {
                return (a_masked as i64 & m, false); // return dividend per SMT-LIB
            }
            let result = a_masked % b_masked;
            (result as i64 & m, false)
        }
        // pf:ensures:srem.zero_divisor - returns dividend when divisor is zero (SMT-LIB bvsrem)
        BuiltinFn::Srem { bits } => {
            let sx = to_signed(a, bits);
            let sy = to_signed(b, bits);
            let m = mask(bits);
            if sy == 0 {
                return (sx & m, false); // return dividend per SMT-LIB
            }
            let q = sx / sy;
            let r = sx - q * sy;
            (r & m, false)
        }

        _ => unreachable!("exec_binary called with unary/unknown fn: {:?}", fn_id),
    }
}

/// Return the number of arguments for a builtin function.
pub fn num_args(fn_id: BuiltinFn) -> usize {
    match fn_id {
        BuiltinFn::Not { .. }
        | BuiltinFn::Sext { .. }
        | BuiltinFn::Zext { .. }
        | BuiltinFn::Trunc { .. }
        | BuiltinFn::Bitcast
        | BuiltinFn::P2i
        | BuiltinFn::I2p => 1,
        _ => 2,
    }
}

/// Is the output of this builtin a Python bool (vs int)?
pub fn output_is_bool(fn_id: BuiltinFn) -> bool {
    matches!(
        fn_id,
        BuiltinFn::SltBool { .. }
            | BuiltinFn::SleBool { .. }
            | BuiltinFn::SgtBool { .. }
            | BuiltinFn::SgeBool { .. }
    )
}
