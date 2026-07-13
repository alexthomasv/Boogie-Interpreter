//! Constant folding over the lowered IR — mode-parametric.
//!
//! Semantics match `vm::eval` exactly PER MODE (`SemanticsMode::Bv`: wrapping
//! arithmetic; `SemanticsMode::Int`: exact-ℤ with BigInt promotion), so folding
//! can never change an execution result. Only var-free subtrees fold, and
//! constants are never trace-recorded, so traces are byte-identical with
//! folding on.
//!
//! Int-mode division/modulus with a CONSTANT ZERO divisor is deliberately left
//! unfolded: `vm::eval` panics loudly there (SMT-LIB `div/mod _ 0` is
//! uninterpreted), and a dead branch containing `x div 0` must not detonate at
//! lower time.
//!
//! Shared by the inline lowering path (interleaved per-block via
//! `flush_block`) and the direct `lower_program_full` path (post-hoc pass).

use crate::builtins::int::{Z, ZResult};
use crate::opcodes::{BinOp, BuiltinFn, Expr, SemanticsMode, Stmt};

pub(crate) fn fold_expr(e: Expr, mode: SemanticsMode) -> Expr {
    match e {
        Expr::BinOp { op, lhs, rhs } => {
            let lhs = fold_expr(*lhs, mode);
            let rhs = fold_expr(*rhs, mode);
            match mode {
                SemanticsMode::Bv => match (as_i64(&lhs), as_i64(&rhs)) {
                    (Some(l), Some(r)) => match op {
                        BinOp::Eq => Expr::Bool(l == r),
                        BinOp::Ne => Expr::Bool(l != r),
                        BinOp::Lt => Expr::Bool(l < r),
                        BinOp::Gt => Expr::Bool(l > r),
                        BinOp::Le => Expr::Bool(l <= r),
                        BinOp::Ge => Expr::Bool(l >= r),
                        BinOp::And => Expr::Bool(l != 0 && r != 0),
                        BinOp::Or => Expr::Bool(l != 0 || r != 0),
                        BinOp::Implies => Expr::Bool(l == 0 || r != 0),
                        BinOp::Iff => Expr::Bool((l != 0) == (r != 0)),
                        BinOp::Sub => Expr::Const(l.wrapping_sub(r)),
                        BinOp::Mul => Expr::Const(l.wrapping_mul(r)),
                        BinOp::Add => Expr::Const(l.wrapping_add(r)),
                    },
                    _ => Expr::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                },
                SemanticsMode::Int => match (as_z(&lhs), as_z(&rhs)) {
                    (Some(l), Some(r)) => {
                        use std::cmp::Ordering::*;
                        let ord = || crate::builtins::int::cmp(&l, &r);
                        match op {
                            BinOp::Eq => Expr::Bool(ord() == Equal),
                            BinOp::Ne => Expr::Bool(ord() != Equal),
                            BinOp::Lt => Expr::Bool(ord() == Less),
                            BinOp::Gt => Expr::Bool(ord() == Greater),
                            BinOp::Le => Expr::Bool(ord() != Greater),
                            BinOp::Ge => Expr::Bool(ord() != Less),
                            BinOp::And => Expr::Bool(!l.is_zero() && !r.is_zero()),
                            BinOp::Or => Expr::Bool(!l.is_zero() || !r.is_zero()),
                            BinOp::Implies => Expr::Bool(l.is_zero() || !r.is_zero()),
                            BinOp::Iff => Expr::Bool(l.is_zero() == r.is_zero()),
                            BinOp::Sub => z_to_expr(crate::builtins::int::sub(&l, &r)),
                            BinOp::Mul => z_to_expr(crate::builtins::int::mul(&l, &r)),
                            BinOp::Add => z_to_expr(crate::builtins::int::add(&l, &r)),
                        }
                    }
                    _ => Expr::BinOp {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                },
            }
        }
        Expr::Builtin { fn_id, args } => {
            let args: Vec<Expr> = args.into_iter().map(|a| fold_expr(a, mode)).collect();
            match mode {
                SemanticsMode::Bv => {
                    let consts: Option<Vec<i64>> = args.iter().map(as_i64).collect();
                    match consts {
                        Some(c) if crate::builtins::num_args(fn_id) == 1 => {
                            Expr::Const(crate::builtins::exec_unary(fn_id, c[0]))
                        }
                        Some(c) => {
                            let (r, is_bool) = crate::builtins::exec_binary(fn_id, c[0], c[1]);
                            if is_bool {
                                Expr::Bool(r != 0)
                            } else {
                                Expr::Const(r)
                            }
                        }
                        None => Expr::Builtin { fn_id, args },
                    }
                }
                SemanticsMode::Int => {
                    let consts: Option<Vec<Z>> = args.iter().map(as_z).collect();
                    match consts {
                        Some(c) if crate::builtins::num_args(fn_id) == 1 => {
                            z_to_expr(crate::builtins::int::exec_unary(fn_id, &c[0]))
                        }
                        // vm::eval panics on a zero divisor; leave the expr
                        // for the (possibly dead) branch to hit at runtime.
                        Some(c) if is_div_like(fn_id) && c[1].is_zero() => {
                            Expr::Builtin { fn_id, args }
                        }
                        Some(c) => match crate::builtins::int::exec_binary(fn_id, &c[0], &c[1]) {
                            ZResult::Num(z) => z_to_expr(z),
                            ZResult::Bool(b) => Expr::Bool(b),
                        },
                        None => Expr::Builtin { fn_id, args },
                    }
                }
            }
        }
        Expr::Not(inner) => {
            let inner = fold_expr(*inner, mode);
            match as_bool(&inner) {
                Some(b) => Expr::Bool(!b),
                None => Expr::Not(Box::new(inner)),
            }
        }
        Expr::IfThenElse { cond, then_, else_ } => {
            let cond = fold_expr(*cond, mode);
            match as_bool(&cond) {
                Some(true) => fold_expr(*then_, mode),
                Some(false) => fold_expr(*else_, mode),
                None => Expr::IfThenElse {
                    cond: Box::new(cond),
                    then_: Box::new(fold_expr(*then_, mode)),
                    else_: Box::new(fold_expr(*else_, mode)),
                },
            }
        }
        Expr::Store {
            bit_width,
            map,
            index,
            value,
        } => Expr::Store {
            bit_width,
            map: Box::new(fold_expr(*map, mode)),
            index: Box::new(fold_expr(*index, mode)),
            value: Box::new(fold_expr(*value, mode)),
        },
        Expr::Load {
            bit_width,
            map,
            index,
        } => Expr::Load {
            bit_width,
            map: Box::new(fold_expr(*map, mode)),
            index: Box::new(fold_expr(*index, mode)),
        },
        leaf => leaf,
    }
}

fn is_div_like(fn_id: BuiltinFn) -> bool {
    matches!(
        fn_id,
        BuiltinFn::Idiv { .. }
            | BuiltinFn::Smod { .. }
            | BuiltinFn::Sdiv { .. }
            | BuiltinFn::Udiv { .. }
            | BuiltinFn::Srem { .. }
            | BuiltinFn::Urem { .. }
    )
}

fn z_to_expr(z: Z) -> Expr {
    match z {
        Z::S(v) => Expr::Const(v),
        Z::B(b) => Expr::ConstBig(b),
    }
}

/// Constant value as i64, matching `vm::eval_i64` (Bool → 1/0). None if not
/// const. BV mode only — `ConstBig` cannot occur there (loud lowering error).
fn as_i64(e: &Expr) -> Option<i64> {
    match e {
        Expr::Const(v) => Some(*v),
        Expr::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// Constant value as exact ℤ (Int mode), matching `vm::eval_z`.
fn as_z(e: &Expr) -> Option<Z> {
    match e {
        Expr::Const(v) => Some(Z::S(*v)),
        Expr::ConstBig(b) => Some(Z::B(b.clone())),
        Expr::Bool(b) => Some(Z::S(*b as i64)),
        _ => None,
    }
}

/// Constant boolean — conservative: only a literal Bool (don't guess the
/// Scalar→bool rule), so a non-Bool leaves its enclosing expr intact.
fn as_bool(e: &Expr) -> Option<bool> {
    match e {
        Expr::Bool(b) => Some(*b),
        _ => None,
    }
}

/// Const-fold the expressions inside a statement, in place.
pub(crate) fn fold_stmt(s: &mut Stmt, mode: SemanticsMode) {
    match s {
        Stmt::Assign1 { rhs, .. } => fold_in_place(rhs, mode),
        Stmt::AssignN { rhs, .. } => {
            for e in rhs.iter_mut() {
                fold_in_place(e, mode);
            }
        }
        Stmt::Assert { expr } | Stmt::Assume { expr } => fold_in_place(expr, mode),
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            fold_in_place(cond, mode);
            for s in then_body.iter_mut() {
                fold_stmt(s, mode);
            }
            for s in else_body.iter_mut() {
                fold_stmt(s, mode);
            }
        }
        Stmt::While { cond, body } => {
            fold_in_place(cond, mode);
            for s in body.iter_mut() {
                fold_stmt(s, mode);
            }
        }
        Stmt::CallPrintf { args }
        | Stmt::CallRead { args }
        | Stmt::CallMemmove { args }
        | Stmt::CallTime { args, .. }
        | Stmt::CallWrite { args, .. } => {
            for e in args.iter_mut() {
                fold_in_place(e, mode);
            }
        }
        _ => {}
    }
}

pub(crate) fn fold_in_place(e: &mut Expr, mode: SemanticsMode) {
    let taken = std::mem::replace(e, Expr::IsExternal);
    *e = fold_expr(taken, mode);
}
