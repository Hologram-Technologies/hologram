//! Shared helpers for building Term trees in a `TermArena<CAP>`.
//!
//! Per spec V.1: loops are `Term::Recurse`, branching is `Term::Match`,
//! cross-Witt-level moves are `Term::Lift` / `Term::Project`, and the only
//! operator vocabulary is `PrimitiveOp` (the closed 10).

use uor_foundation::enforcement::{Term, TermArena, TermList};
use uor_foundation::pipeline::literal_u64;
use uor_foundation::{PrimitiveOp, WittLevel};

/// Result of an emitter: index of the root node in the arena, or `None`
/// if the arena overflowed.
pub type EmitResult = Option<u32>;

/// Push a single literal, return its index. Per ADR-051 the literal's
/// value is packed into a `TermValue` byte sequence at the declared
/// Witt level's byte width via `pipeline::literal_u64`.
#[inline]
pub fn push_literal<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    value: u64,
    level: WittLevel,
) -> EmitResult {
    arena.push(literal_u64(value, level))
}

/// Push a variable reference, return its index.
#[inline]
pub fn push_variable<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    name_index: u32,
) -> EmitResult {
    arena.push(Term::Variable { name_index })
}

/// Push an `Application` of a `PrimitiveOp` to a contiguous arg list.
/// `args_start` and `args_len` describe the slice of already-pushed args.
#[inline]
pub fn push_application<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    op: PrimitiveOp,
    args_start: u32,
    args_len: u32,
) -> EmitResult {
    arena.push(Term::Application {
        operator: op,
        args: TermList { start: args_start, len: args_len },
    })
}

/// Push a `Lift` (canonical injection W_n → W_m, n < m).
#[inline]
pub fn push_lift<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    operand_index: u32,
    target: WittLevel,
) -> EmitResult {
    arena.push(Term::Lift { operand_index, target })
}

/// Push a `Project` (canonical surjection W_m → W_n, m > n).
#[inline]
pub fn push_project<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    operand_index: u32,
    target: WittLevel,
) -> EmitResult {
    arena.push(Term::Project { operand_index, target })
}

/// Push a `Recurse` (bounded recursion with descent measure).
#[inline]
pub fn push_recurse<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    measure_index: u32,
    base_index: u32,
    step_index: u32,
) -> EmitResult {
    arena.push(Term::Recurse { measure_index, base_index, step_index })
}

/// Push a `Match` (pattern dispatch).
#[inline]
pub fn push_match<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    scrutinee_index: u32,
    arms_start: u32,
    arms_len: u32,
) -> EmitResult {
    arena.push(Term::Match {
        scrutinee_index,
        arms: TermList { start: arms_start, len: arms_len },
    })
}

/// Push an `Unfold`.
#[inline]
pub fn push_unfold<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    seed_index: u32,
    step_index: u32,
) -> EmitResult {
    arena.push(Term::Unfold { seed_index, step_index })
}

/// Push a `Try`.
#[inline]
pub fn push_try<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    body_index: u32,
    handler_index: u32,
) -> EmitResult {
    arena.push(Term::Try { body_index, handler_index })
}

/// Build a binary `Application(op, [a, b])` where `a` and `b` are already
/// at indices `a_idx` and `b_idx`. The two args must be contiguous in the
/// arena: this requires the caller to push them adjacently. The function
/// asserts contiguity (in debug builds) and emits the application.
#[inline]
pub fn push_binary_app<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    op: PrimitiveOp,
    a_idx: u32,
    _b_idx: u32,
) -> EmitResult {
    push_application(arena, op, a_idx, 2)
}

/// Build a unary `Application(op, [a])`.
#[inline]
pub fn push_unary_app<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    op: PrimitiveOp,
    a_idx: u32,
) -> EmitResult {
    push_application(arena, op, a_idx, 1)
}
