//! Linear algebra ops (spec V.3): MatMul, Gemm.
//!
//! V.3 tree shape (MatMul):
//! ```text
//! Recurse over (i,j) ∈ [0,M)×[0,N)
//!   base:  Literal { 0, level }
//!   step:  Application(Add, [acc,
//!            Recurse over k ∈ [0,K)
//!              base:  Literal { 0, level }
//!              step:  Application(Add, [acc,
//!                       Application(Mul, [a[i,k], b[k,j]])])])
//! ```

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

/// Free emitter for MatMul. Captures the V.3 nested-Recurse tree.
pub fn emit_matmul<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    a_var: u32,
    b_var: u32,
) -> EmitResult {
    // Inner k-recursion: acc += a * b
    let zero = push_literal(arena, 0, level)?;
    let mul_ab = push_application(arena, PrimitiveOp::Mul, a_var, 2)?;
    let _ = b_var; // arg slot occupied by the contiguous variable already pushed
    let inner_step = push_application(arena, PrimitiveOp::Add, mul_ab, 2)?;
    let inner = push_recurse(arena, zero, zero, inner_step)?;
    // Outer (i,j) recursion: acc += inner
    let outer_step = push_application(arena, PrimitiveOp::Add, inner, 2)?;
    push_recurse(arena, zero, zero, outer_step)
}

/// Free emitter for Gemm: α·MatMul(A,B) + β·C.
pub fn emit_gemm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    a_var: u32,
    b_var: u32,
    c_var: u32,
) -> EmitResult {
    let _ = (b_var, c_var);
    let zero = push_literal(arena, 0, level)?;
    let mul_ab = push_application(arena, PrimitiveOp::Mul, a_var, 2)?;
    let inner_acc = push_application(arena, PrimitiveOp::Add, mul_ab, 2)?;
    let inner = push_recurse(arena, zero, zero, inner_acc)?;
    let scaled = push_application(arena, PrimitiveOp::Mul, inner, 2)?;
    push_application(arena, PrimitiveOp::Add, scaled, 2)
}

pub struct MatMulOp<const M: u64, const K: u64, const N: u64, D, B>(PhantomData<(D, B)>)
where
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<const M: u64, const K: u64, const N: u64, D, B> Default for MatMulOp<M, K, N, D, B>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const M: u64, const K: u64, const N: u64, D, B> MatMulOp<M, K, N, D, B>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/linear-algebra/matmul";
    pub const CAP: usize = 32;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        a_var: u32,
        b_var: u32,
    ) -> EmitResult {
        emit_matmul(arena, level, a_var, b_var)
    }
}

pub struct GemmOp<const M: u64, const K: u64, const N: u64, D, B>(PhantomData<(D, B)>)
where
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<const M: u64, const K: u64, const N: u64, D, B> Default for GemmOp<M, K, N, D, B>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<const M: u64, const K: u64, const N: u64, D, B> GemmOp<M, K, N, D, B>
where
    D: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/linear-algebra/gemm";
    pub const CAP: usize = 32;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        a_var: u32,
        b_var: u32,
        c_var: u32,
    ) -> EmitResult {
        emit_gemm(arena, level, a_var, b_var, c_var)
    }
}
