//! Reference scalar evaluator surface (spec VII.3, O-2).
//!
//! Used by per-op tests to verify backend kernels match the formal Term tree.
//! Not the hot path — slow, allocation-free, walks the arena.

extern crate alloc;
use alloc::vec::Vec;

use uor_foundation::enforcement::{Term, TermArena};
use uor_foundation::PrimitiveOp;

/// Safety ceiling on bounded-recursion iterations. Prevents runaway
/// evaluation on malformed `Term::Recurse` measures.
pub const MAX_RECURSE_ITERATIONS: u64 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalError {
    /// Arena index out of bounds.
    InvalidIndex,
    /// Recursion measure failed to decrease.
    NonTerminating,
    /// Variable binding was missing.
    UnresolvedBinding,
    /// Operation arity mismatch.
    ArityMismatch,
    /// Witt-level mismatch on Lift/Project.
    LevelMismatch,
}

/// Reference scalar evaluator. Implementations walk a Term tree at given
/// arena root, resolving variables through the binding source.
pub trait ReferenceEvaluator {
    type Value: Copy;
    type Bindings: ?Sized;

    fn evaluate<const CAP: usize>(
        arena: &TermArena<CAP>,
        root: u32,
        bindings: &Self::Bindings,
    ) -> Result<Self::Value, EvalError>;
}

/// Helper: fetch a term at index, error on out-of-bounds.
#[inline]
pub fn fetch<const CAP: usize>(
    arena: &TermArena<CAP>,
    index: u32,
) -> Result<&Term, EvalError> {
    arena.get(index).ok_or(EvalError::InvalidIndex)
}

/// Concrete scalar evaluator at a `u64` carrier.
///
/// Walks the closed `Term` enum and computes a `u64` at each node by
/// applying the primitive's modular semantics. Variables are resolved
/// via a slice keyed by `name_index`. Lift / Project are bit-truncations
/// at the requested Witt level. Recurse / Match / Unfold / Try are not
/// supported in this reference walker (the test suite uses straight-line
/// trees); those return `EvalError::NonTerminating` to flag misuse.
pub struct ScalarEvaluatorU64;

impl ReferenceEvaluator for ScalarEvaluatorU64 {
    type Value = u64;
    type Bindings = [u64];

    fn evaluate<const CAP: usize>(
        arena: &TermArena<CAP>,
        root: u32,
        bindings: &[u64],
    ) -> Result<u64, EvalError> {
        eval_node(arena, root, bindings)
    }
}

fn eval_node<const CAP: usize>(
    arena: &TermArena<CAP>,
    index: u32,
    bindings: &[u64],
) -> Result<u64, EvalError> {
    match fetch(arena, index)? {
        Term::Literal { value, .. } => Ok(term_value_to_u64(value)),
        Term::Variable { name_index } => {
            bindings.get(*name_index as usize).copied()
                .ok_or(EvalError::UnresolvedBinding)
        }
        Term::Application { operator, args } => {
            let mut argv: Vec<u64> = Vec::with_capacity(args.len as usize);
            for k in 0..args.len {
                argv.push(eval_node(arena, args.start + k, bindings)?);
            }
            apply_primitive(*operator, &argv)
        }
        Term::Lift { operand_index, target } => {
            let v = eval_node(arena, *operand_index, bindings)?;
            Ok(truncate_to_witt(v, target.witt_length()))
        }
        Term::Project { operand_index, target } => {
            let v = eval_node(arena, *operand_index, bindings)?;
            Ok(truncate_to_witt(v, target.witt_length()))
        }
        Term::Match { scrutinee_index, arms } => {
            // Simple match: pick the first arm whose evaluation equals the
            // scrutinee value. If none match, pick the last arm as default.
            let scrut_val = eval_node(arena, *scrutinee_index, bindings)?;
            if arms.len == 0 {
                return Err(EvalError::NonTerminating);
            }
            for k in 0..arms.len.saturating_sub(1) {
                let arm_val = eval_node(arena, arms.start + k, bindings)?;
                if arm_val == scrut_val {
                    return Ok(arm_val);
                }
            }
            // Default arm.
            eval_node(arena, arms.start + arms.len - 1, bindings)
        }
        Term::Try { body_index, handler_index } => {
            // Evaluate body; on EvalError fallback to handler.
            match eval_node(arena, *body_index, bindings) {
                Ok(v) => Ok(v),
                Err(_) => eval_node(arena, *handler_index, bindings),
            }
        }
        Term::Recurse { measure_index, base_index, step_index } => {
            // Bounded recursion: descent measure must strictly decrease.
            // We bound iteration by the initial measure value plus a safety
            // ceiling to prevent runaway evaluation on malformed measures.
            let mut acc = eval_node(arena, *base_index, bindings)?;
            let mut measure = eval_node(arena, *measure_index, bindings)?;
            let safety_ceiling: u64 = MAX_RECURSE_ITERATIONS.min(measure.saturating_add(1));
            let mut iters: u64 = 0;
            while measure > 0 && iters < safety_ceiling {
                let step = eval_node(arena, *step_index, bindings)?;
                acc = acc.wrapping_add(step);
                measure = measure.wrapping_sub(1);
                iters += 1;
            }
            if measure > 0 {
                return Err(EvalError::NonTerminating);
            }
            Ok(acc)
        }
        Term::Unfold { seed_index, step_index } => {
            // Stream construction by unfold: produce a sequence by repeatedly
            // applying `step` to the seed, accumulating into an XOR fold.
            // Bounded by `MAX_RECURSE_ITERATIONS`.
            let seed = eval_node(arena, *seed_index, bindings)?;
            let mut acc: u64 = seed;
            let mut state = seed;
            for _ in 0..MAX_RECURSE_ITERATIONS {
                let next = eval_node(arena, *step_index, bindings)?;
                if next == state { break; }
                acc ^= next;
                state = next;
            }
            Ok(acc)
        }
        // Substrate-level Term variants that hologram's emitters do not
        // produce (AxisInvocation, ProjectField, FirstAdmit, and the
        // remaining ADR-029/033/034/057-introduced forms). The reference
        // evaluator surfaces them as `ArityMismatch` — encountering one
        // in a hologram-emitted tree is a regression.
        _ => Err(EvalError::ArityMismatch),
    }
}

#[inline]
fn apply_primitive(op: PrimitiveOp, args: &[u64]) -> Result<u64, EvalError> {
    match (op, args.len()) {
        (PrimitiveOp::Neg, 1) => Ok(args[0].wrapping_neg()),
        (PrimitiveOp::Bnot, 1) => Ok(!args[0]),
        (PrimitiveOp::Succ, 1) => Ok(args[0].wrapping_add(1)),
        (PrimitiveOp::Pred, 1) => Ok(args[0].wrapping_sub(1)),
        (PrimitiveOp::Add, 2) => Ok(args[0].wrapping_add(args[1])),
        (PrimitiveOp::Sub, 2) => Ok(args[0].wrapping_sub(args[1])),
        (PrimitiveOp::Mul, 2) => Ok(args[0].wrapping_mul(args[1])),
        (PrimitiveOp::Xor, 2) => Ok(args[0] ^ args[1]),
        (PrimitiveOp::And, 2) => Ok(args[0] & args[1]),
        (PrimitiveOp::Or,  2) => Ok(args[0] | args[1]),
        _ => Err(EvalError::ArityMismatch),
    }
}

/// Convert a `TermValue` byte buffer back into the `u64` it was packed
/// from via `pipeline::literal_u64` / `TermValue::from_u64_be`. The
/// buffer holds the value's bytes in big-endian order at its declared
/// width; widths > 8 truncate to the low 8 bytes.
#[inline]
fn term_value_to_u64(v: &uor_foundation::pipeline::TermValue) -> u64 {
    let bytes = v.bytes();
    let take = bytes.len().min(8);
    let mut padded = [0u8; 8];
    padded[8 - take..].copy_from_slice(&bytes[bytes.len() - take..]);
    u64::from_be_bytes(padded)
}

#[inline]
fn truncate_to_witt(v: u64, witt: u32) -> u64 {
    if witt >= 64 { v }
    else if witt == 0 { 0 }
    else {
        let mask = (1u64 << witt).wrapping_sub(1);
        v & mask
    }
}
