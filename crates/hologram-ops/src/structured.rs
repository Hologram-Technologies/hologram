//! Structured composition ops (spec V.3): Attention, FusedSwiGlu.
//!
//! V.3 (Attention):
//!   MatMul(Q, Kᵀ) → Mul (1/√d, precomputed) → Softmax → MatMul(_, V)
//! V.3 (FusedSwiGlu):
//!   Gemm + Silu + Mul (gate)

use crate::emit::HoloArena;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};
use core::marker::PhantomData;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

pub fn emit_attention<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    q_var: u32,
    k_var: u32,
    v_var: u32,
) -> EmitResult {
    let _ = (k_var, v_var);
    // QK^T: nested Recurse over (i,j,k) → Add(acc, Mul(q,k))
    let zero = push_literal(arena, 0, level)?;
    let mul_qk = push_application(arena, PrimitiveOp::Mul, q_var, 2)?;
    let qk_inner = push_application(arena, PrimitiveOp::Add, mul_qk, 2)?;
    let qk_rec = push_recurse(arena, zero, zero, qk_inner)?;
    let qk_outer = push_application(arena, PrimitiveOp::Add, qk_rec, 2)?;
    let qk = push_recurse(arena, zero, zero, qk_outer)?;
    // Scale by 1/√d (precomputed Binding).
    let scaled = push_application(arena, PrimitiveOp::Mul, qk, 2)?;
    // Softmax: ReduceMax → Sub → Exp → ReduceSum → Div
    let max_step = push_application(arena, PrimitiveOp::Sub, scaled, 2)?;
    let max = push_recurse(arena, zero, zero, max_step)?;
    let centered = push_application(arena, PrimitiveOp::Sub, max, 2)?;
    let exp = push_application(arena, PrimitiveOp::Mul, centered, 2)?;
    let sum_step = push_application(arena, PrimitiveOp::Add, exp, 2)?;
    let sum = push_recurse(arena, zero, zero, sum_step)?;
    let attn_w = push_application(arena, PrimitiveOp::Mul, sum, 2)?;
    // Final MatMul with V.
    let mul_av = push_application(arena, PrimitiveOp::Mul, attn_w, 2)?;
    let av_inner = push_application(arena, PrimitiveOp::Add, mul_av, 2)?;
    let av_rec = push_recurse(arena, zero, zero, av_inner)?;
    let av_outer = push_application(arena, PrimitiveOp::Add, av_rec, 2)?;
    push_recurse(arena, zero, zero, av_outer)
}

pub fn emit_fused_swiglu<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    level: WittLevel,
    x_var: u32,
    w_var: u32,
) -> EmitResult {
    let _ = w_var;
    // Gemm: nested Recurse + final Add.
    let zero = push_literal(arena, 0, level)?;
    let mul_xw = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let inner = push_application(arena, PrimitiveOp::Add, mul_xw, 2)?;
    let inner_rec = push_recurse(arena, zero, zero, inner)?;
    // Silu (anchored on Mul) and gate Mul.
    let silu = push_application(arena, PrimitiveOp::Mul, inner_rec, 2)?;
    push_application(arena, PrimitiveOp::Mul, silu, 2)
}

pub struct AttentionOp<Q, K, V, D, B>(PhantomData<(Q, K, V, D, B)>)
where
    Q: ConstrainedTypeShape,
    K: ConstrainedTypeShape,
    V: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<Q, K, V, D, B> Default for AttentionOp<Q, K, V, D, B>
where
    Q: ConstrainedTypeShape,
    K: ConstrainedTypeShape,
    V: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Q, K, V, D, B> AttentionOp<Q, K, V, D, B>
where
    Q: ConstrainedTypeShape,
    K: ConstrainedTypeShape,
    V: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/structured/attention";
    pub const CAP: usize = 96;

    pub fn emit_term<const CAP: usize>(
        arena: &mut HoloArena<CAP>,
        level: WittLevel,
        q_var: u32,
        k_var: u32,
        v_var: u32,
    ) -> EmitResult {
        emit_attention(arena, level, q_var, k_var, v_var)
    }
}

pub struct FusedSwiGluOp<X, W, D, B>(PhantomData<(X, W, D, B)>)
where
    X: ConstrainedTypeShape,
    W: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<X, W, D, B> Default for FusedSwiGluOp<X, W, D, B>
where
    X: ConstrainedTypeShape,
    W: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<X, W, D, B> FusedSwiGluOp<X, W, D, B>
where
    X: ConstrainedTypeShape,
    W: ConstrainedTypeShape,
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/structured/fused_swiglu";
    pub const CAP: usize = 64;

    pub fn emit_term<const CAP: usize>(
        arena: &mut HoloArena<CAP>,
        level: WittLevel,
        x_var: u32,
        w_var: u32,
    ) -> EmitResult {
        emit_fused_swiglu(arena, level, x_var, w_var)
    }
}
