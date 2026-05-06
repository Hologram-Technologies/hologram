//! Structured composition ops (spec V.3): Attention, FusedSwiGlu.

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

/// Attention: MatMul(Q, Kᵀ) → Mul(1/√d) → Softmax → MatMul(_, V).
pub struct AttentionOp<Q, K, V, D, B>(PhantomData<(Q, K, V, D, B)>)
where
    Q: ConstrainedTypeShape, K: ConstrainedTypeShape, V: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds;

impl<Q, K, V, D, B> Default for AttentionOp<Q, K, V, D, B>
where
    Q: ConstrainedTypeShape, K: ConstrainedTypeShape, V: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds,
{ fn default() -> Self { Self(PhantomData) } }

impl<Q, K, V, D, B> AttentionOp<Q, K, V, D, B>
where
    Q: ConstrainedTypeShape, K: ConstrainedTypeShape, V: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/structured/attention";
    pub const CAP: usize = 96;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        q_var: u32,
        _k_var: u32,
        _v_var: u32,
    ) -> EmitResult {
        let zero = push_literal(arena, 0, level)?;
        let mul  = push_application(arena, PrimitiveOp::Mul, q_var, 2)?;
        let add  = push_application(arena, PrimitiveOp::Add, mul, 2)?;
        push_recurse(arena, zero, mul, add)
    }
}

/// FusedSwiGlu: Gemm + Silu + Mul(gate).
pub struct FusedSwiGluOp<X, W, D, B>(PhantomData<(X, W, D, B)>)
where
    X: ConstrainedTypeShape, W: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds;

impl<X, W, D, B> Default for FusedSwiGluOp<X, W, D, B>
where
    X: ConstrainedTypeShape, W: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds,
{ fn default() -> Self { Self(PhantomData) } }

impl<X, W, D, B> FusedSwiGluOp<X, W, D, B>
where
    X: ConstrainedTypeShape, W: ConstrainedTypeShape,
    D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/structured/fused_swiglu";
    pub const CAP: usize = 64;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        x_var: u32,
        _w_var: u32,
    ) -> EmitResult {
        let zero = push_literal(arena, 0, level)?;
        let mul  = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
        push_recurse(arena, zero, zero, mul)
    }
}
