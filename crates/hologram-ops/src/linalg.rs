//! Linear algebra ops (spec V.3): MatMul, Gemm.

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_recurse, push_literal, EmitResult};

/// MatMul iso: (Tensor[M,K], Tensor[K,N]) → Tensor[M,N].
///
/// Term tree (spec V.3 sketch):
/// ```text
/// Recurse over (i,j) ∈ [0,M)×[0,N)
///   base:  Literal { 0, level }
///   step:  Application(Add, [acc,
///            Recurse over k ∈ [0,K)
///              base:  Literal { 0, level }
///              step:  Application(Add, [acc,
///                       Application(Mul, [a[i,k], b[k,j]])])])
/// ```
pub struct MatMulOp<const M: u64, const K: u64, const N: u64, D, B>(PhantomData<(D, B)>)
where
    D: ConstrainedTypeShape,
    B: HostBounds;

impl<const M: u64, const K: u64, const N: u64, D, B> Default for MatMulOp<M, K, N, D, B>
where D: ConstrainedTypeShape, B: HostBounds,
{
    fn default() -> Self { Self(PhantomData) }
}

impl<const M: u64, const K: u64, const N: u64, D, B> MatMulOp<M, K, N, D, B>
where D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/linear-algebra/matmul";
    pub const CAP: usize = 32;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        a_var_idx: u32,
        b_var_idx: u32,
    ) -> EmitResult {
        // Bottom-up: zero base, mul application, add accumulator, recurse.
        let zero = push_literal(arena, 0, level)?;
        let mul  = push_application(arena, PrimitiveOp::Mul, a_var_idx, 2)?;
        let _    = b_var_idx; // referenced positionally via args layout
        let add  = push_application(arena, PrimitiveOp::Add, zero, 2)?;
        // Inner k recursion
        let inner = push_recurse(arena, zero, zero, mul)?;
        // Outer (i,j) recursion
        push_recurse(arena, zero, inner, add)
    }
}

/// Gemm iso: α·MatMul(A,B) + β·C.
pub struct GemmOp<const M: u64, const K: u64, const N: u64, D, B>(PhantomData<(D, B)>)
where D: ConstrainedTypeShape, B: HostBounds;

impl<const M: u64, const K: u64, const N: u64, D, B> Default for GemmOp<M, K, N, D, B>
where D: ConstrainedTypeShape, B: HostBounds,
{
    fn default() -> Self { Self(PhantomData) }
}

impl<const M: u64, const K: u64, const N: u64, D, B> GemmOp<M, K, N, D, B>
where D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/linear-algebra/gemm";
    pub const CAP: usize = 32;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        a_var_idx: u32,
        _b_var_idx: u32,
        _c_var_idx: u32,
    ) -> EmitResult {
        let zero = push_literal(arena, 0, level)?;
        let mul  = push_application(arena, PrimitiveOp::Mul, a_var_idx, 2)?;
        let add  = push_application(arena, PrimitiveOp::Add, mul, 2)?;
        push_recurse(arena, zero, mul, add)
    }
}
