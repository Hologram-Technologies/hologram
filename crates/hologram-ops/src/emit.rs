//! Shared helpers for building Term trees in a `HoloArena<CAP>`.
//!
//! Per spec V.1: loops are `Term::Recurse`, branching is `Term::Match`,
//! cross-Witt-level moves are `Term::Lift` / `Term::Project`, and the only
//! operator vocabulary is `PrimitiveOp` (the closed 10).

use uor_foundation::enforcement::{Term, TermList};
use uor_foundation::pipeline::literal_u64;
use uor_foundation::{PrimitiveOp, WittLevel};

/// ADR-060 inline-carrier width for hologram. `carrier_inline_bytes`
/// reduces to `max(WITT_LEVEL_MAX_BITS / 8, FINGERPRINT_MAX_BYTES, κ)`; the
/// κ-label term (`HASHER_IDENTIFIER_BYTES + 1 + 2·32 = 97`) dominates for
/// every hologram backend (witt ≤ 64 bytes, fingerprint = 32), so the inline
/// width is invariant across the CPU/AVX2/AVX-512/NEON/Metal/wgpu
/// monomorphizations and can be pinned as a single workspace constant.
pub const HOLOGRAM_INLINE_BYTES: usize =
    uor_foundation::pipeline::HASHER_IDENTIFIER_BYTES + 1 + 2 * 32;

/// Hologram's monomorphic `TermArena`. Every emitted term is `'static`
/// (literals, variables and applications carry no borrowed payload), so the
/// arena lifetime is fixed and only the capacity `CAP` varies per op.
pub type HoloArena<const CAP: usize> =
    uor_foundation::enforcement::TermArena<'static, HOLOGRAM_INLINE_BYTES, CAP>;

/// Hologram's monomorphic `Term` — the element type of [`HoloArena`].
pub type HoloTerm = Term<'static, HOLOGRAM_INLINE_BYTES>;

/// Result of an emitter: index of the root node in the arena, or `None`
/// if the arena overflowed.
pub type EmitResult = Option<u32>;

/// Push a single literal, return its index. Per ADR-051 the literal's
/// value is packed into a `TermValue` byte sequence at the declared
/// Witt level's byte width via `pipeline::literal_u64`.
#[inline]
pub fn push_literal<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    value: u64,
    level: WittLevel,
) -> EmitResult {
    arena.push(literal_u64(value, level))
}

/// Push a variable reference, return its index.
#[inline]
pub fn push_variable<const CAP: usize>(arena: &mut HoloArena<CAP>, name_index: u32) -> EmitResult {
    arena.push(Term::Variable { name_index })
}

/// Push an `Application` of a `PrimitiveOp` to a contiguous arg list.
/// `args_start` and `args_len` describe the slice of already-pushed args.
#[inline]
pub fn push_application<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    op: PrimitiveOp,
    args_start: u32,
    args_len: u32,
) -> EmitResult {
    arena.push(Term::Application {
        operator: op,
        args: TermList {
            start: args_start,
            len: args_len,
        },
    })
}

/// Push a `Lift` (canonical injection W_n → W_m, n < m).
#[inline]
pub fn push_lift<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    operand_index: u32,
    target: WittLevel,
) -> EmitResult {
    arena.push(Term::Lift {
        operand_index,
        target,
    })
}

/// Push a `Project` (canonical surjection W_m → W_n, m > n).
#[inline]
pub fn push_project<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    operand_index: u32,
    target: WittLevel,
) -> EmitResult {
    arena.push(Term::Project {
        operand_index,
        target,
    })
}

/// Push a `Recurse` (bounded recursion with descent measure).
#[inline]
pub fn push_recurse<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    measure_index: u32,
    base_index: u32,
    step_index: u32,
) -> EmitResult {
    arena.push(Term::Recurse {
        measure_index,
        base_index,
        step_index,
    })
}

/// Push a `Match` (pattern dispatch).
#[inline]
pub fn push_match<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    scrutinee_index: u32,
    arms_start: u32,
    arms_len: u32,
) -> EmitResult {
    arena.push(Term::Match {
        scrutinee_index,
        arms: TermList {
            start: arms_start,
            len: arms_len,
        },
    })
}
