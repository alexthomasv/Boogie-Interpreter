//! Kani bounded model checking harnesses for bitvector operations.
//!
//! These harnesses verify that exec_binary / exec_unary match SMT-LIB 2.6
//! FixedSizeBitVectors semantics for all possible inputs within bounded bit widths.
//!
//! Run: `cargo kani -Z function-contracts`

mod bitvector_ops;
