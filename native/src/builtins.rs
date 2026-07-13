//! Builtin intrinsic semantics, split per `SemanticsMode`.
//!
//! * [`bv`] — the wrapping/masked bit-vector algebra (today's historical
//!   semantics, byte-for-byte). Correct model for `SemanticsMode::Bv`.
//! * [`int`] — exact mathematical-integer (ℤ) semantics matching the SMACK
//!   integer-encoding prelude and the verifier's cvc5 model
//!   (`interpreter/utils/utils_cvc5.py` under `set_integer_encoding(True)`).
//!   Correct model for `SemanticsMode::Int`.
//!
//! `vm::eval` and `lowering::fold` dispatch on `CompiledProgram.mode`.
//! Arity and bool-output classification are mode-independent and live here.

pub mod bv;
pub mod int;

// Legacy surface: unqualified `builtins::exec_*` / `mask` / `to_signed` are the
// BV semantics (pre-mode default). Concolic (BV-gated) and the kani proofs use
// these re-exports (`mask`/`to_signed` only under `cfg(kani)`).
#[allow(unused_imports)]
pub use bv::{exec_binary, exec_unary, mask, to_signed};

use crate::opcodes::BuiltinFn;

/// Return the number of arguments for a builtin function (mode-independent).
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

/// Is the output of this builtin a Python bool (vs int)? (mode-independent)
#[allow(dead_code)]
pub fn output_is_bool(fn_id: BuiltinFn) -> bool {
    matches!(
        fn_id,
        BuiltinFn::SltBool { .. }
            | BuiltinFn::SleBool { .. }
            | BuiltinFn::SgtBool { .. }
            | BuiltinFn::SgeBool { .. }
    )
}
