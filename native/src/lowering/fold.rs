//! Constant folding over the lowered IR.
//!
//! Semantics match `vm::eval` exactly (wrapping arithmetic, Bool→1/0 in
//! arithmetic positions, conservative bool detection), so folding can never
//! change an execution result. Only var-free subtrees fold, and constants are
//! never trace-recorded, so traces are byte-identical with folding on.
//!
//! Shared by the inline lowering path (interleaved per-block via
//! `flush_block`) and the direct `lower_program_full` path (post-hoc pass).

use crate::opcodes::{BinOp, Expr, Stmt};

pub(crate) fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::BinOp { op, lhs, rhs } => {
            let lhs = fold_expr(*lhs);
            let rhs = fold_expr(*rhs);
            match (as_i64(&lhs), as_i64(&rhs)) {
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
            }
        }
        Expr::Builtin { fn_id, args } => {
            let args: Vec<Expr> = args.into_iter().map(fold_expr).collect();
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
        Expr::Not(inner) => {
            let inner = fold_expr(*inner);
            match as_bool(&inner) {
                Some(b) => Expr::Bool(!b),
                None => Expr::Not(Box::new(inner)),
            }
        }
        Expr::IfThenElse { cond, then_, else_ } => {
            let cond = fold_expr(*cond);
            match as_bool(&cond) {
                Some(true) => fold_expr(*then_),
                Some(false) => fold_expr(*else_),
                None => Expr::IfThenElse {
                    cond: Box::new(cond),
                    then_: Box::new(fold_expr(*then_)),
                    else_: Box::new(fold_expr(*else_)),
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
            map: Box::new(fold_expr(*map)),
            index: Box::new(fold_expr(*index)),
            value: Box::new(fold_expr(*value)),
        },
        Expr::Load {
            bit_width,
            map,
            index,
        } => Expr::Load {
            bit_width,
            map: Box::new(fold_expr(*map)),
            index: Box::new(fold_expr(*index)),
        },
        leaf => leaf,
    }
}

/// Constant value as i64, matching `vm::eval_i64` (Bool → 1/0). None if not const.
fn as_i64(e: &Expr) -> Option<i64> {
    match e {
        Expr::Const(v) => Some(*v),
        Expr::Bool(b) => Some(*b as i64),
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
pub(crate) fn fold_stmt(s: &mut Stmt) {
    match s {
        Stmt::Assign1 { rhs, .. } => fold_in_place(rhs),
        Stmt::AssignN { rhs, .. } => {
            for e in rhs.iter_mut() {
                fold_in_place(e);
            }
        }
        Stmt::Assert { expr } | Stmt::Assume { expr } => fold_in_place(expr),
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            fold_in_place(cond);
            for s in then_body.iter_mut() {
                fold_stmt(s);
            }
            for s in else_body.iter_mut() {
                fold_stmt(s);
            }
        }
        Stmt::While { cond, body } => {
            fold_in_place(cond);
            for s in body.iter_mut() {
                fold_stmt(s);
            }
        }
        Stmt::CallPrintf { args }
        | Stmt::CallRead { args }
        | Stmt::CallMemmove { args }
        | Stmt::CallTime { args, .. }
        | Stmt::CallWrite { args, .. } => {
            for e in args.iter_mut() {
                fold_in_place(e);
            }
        }
        _ => {}
    }
}

pub(crate) fn fold_in_place(e: &mut Expr) {
    let taken = std::mem::replace(e, Expr::IsExternal);
    *e = fold_expr(taken);
}
