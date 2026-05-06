//! Backward gradient ops (spec V.4).
//!
//! Each differentiable forward op declares a companion backward marker.
//! Per ADR-043, backward Term trees are emitted at graph-build time,
//! not traversed at runtime.

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

// ─── MatMul gradients ──────────────────────────────────────────────

pub fn emit_matmul_grad_a<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    _b_var: u32,
) -> EmitResult {
    // dL/dA = dL/dY @ B^T : nested Recurse over (i,j,k).
    let zero      = push_literal(arena, 0, level)?;
    let mul       = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let inner_step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let inner = push_recurse(arena, zero, zero, inner_step)?;
    let outer_step = push_application(arena, PrimitiveOp::Add, inner, 2)?;
    push_recurse(arena, zero, zero, outer_step)
}

pub fn emit_matmul_grad_b<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    a_var: u32,
) -> EmitResult {
    emit_matmul_grad_a(arena, level, grad_var, a_var)
}

// ─── Conv2d gradients ──────────────────────────────────────────────

pub fn emit_conv2d_grad_x<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    _w_var: u32,
) -> EmitResult {
    let zero      = push_literal(arena, 0, level)?;
    let mul       = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let kw_step   = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let kw        = push_recurse(arena, zero, zero, kw_step)?;
    let kh_step   = push_application(arena, PrimitiveOp::Add, kw, 2)?;
    let kh        = push_recurse(arena, zero, zero, kh_step)?;
    let ow_step   = push_application(arena, PrimitiveOp::Add, kh, 2)?;
    let ow        = push_recurse(arena, zero, zero, ow_step)?;
    let oh_step   = push_application(arena, PrimitiveOp::Add, ow, 2)?;
    push_recurse(arena, zero, zero, oh_step)
}

pub fn emit_conv2d_grad_w<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    x_var: u32,
) -> EmitResult {
    emit_conv2d_grad_x(arena, level, grad_var, x_var)
}

// ─── Activation+reduce gradients ────────────────────────────────────

pub fn emit_softmax_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    // d softmax = softmax · (grad − Σ softmax · grad)
    let zero  = push_literal(arena, 0, level)?;
    let mul   = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let step  = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let sum   = push_recurse(arena, zero, zero, step)?;
    let diff  = push_application(arena, PrimitiveOp::Sub, grad_var, 2)?;
    let _ = sum;
    push_application(arena, PrimitiveOp::Mul, diff, 2)
}

pub fn emit_log_softmax_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    let zero  = push_literal(arena, 0, level)?;
    let step  = push_application(arena, PrimitiveOp::Add, grad_var, 2)?;
    let sum   = push_recurse(arena, zero, zero, step)?;
    let mul   = push_application(arena, PrimitiveOp::Mul, sum, 2)?;
    let _ = mul;
    push_application(arena, PrimitiveOp::Sub, grad_var, 2)
}

// ─── Normalization gradients ───────────────────────────────────────

pub fn emit_layer_norm_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    _gamma_var: u32,
    _x_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let sum  = push_recurse(arena, zero, zero, step)?;
    let scaled = push_application(arena, PrimitiveOp::Mul, sum, 2)?;
    push_application(arena, PrimitiveOp::Sub, scaled, 2)
}

pub fn emit_rms_norm_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    gamma_var: u32,
    x_var: u32,
) -> EmitResult {
    emit_layer_norm_grad(arena, level, grad_var, gamma_var, x_var)
}

pub fn emit_group_norm_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    gamma_var: u32,
    x_var: u32,
) -> EmitResult {
    emit_layer_norm_grad(arena, level, grad_var, gamma_var, x_var)
}

// ─── Reduction gradients ───────────────────────────────────────────

pub fn emit_reduce_sum_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Add, grad_var, 1)
}

pub fn emit_reduce_mean_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Mul, grad_var, 2)
}

pub fn emit_reduce_prod_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Mul, grad_var, 2)
}

// ─── Elementwise binary gradients ──────────────────────────────────

pub fn emit_sub_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Sub, grad_var, 1)
}

pub fn emit_mul_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Mul, grad_var, 2)
}

pub fn emit_div_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Mul, grad_var, 2)
}

pub fn emit_pow_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let mul  = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let step = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_min_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::And, grad_var, 2)
}

pub fn emit_max_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::And, grad_var, 2)
}

// ─── Layout gradients ──────────────────────────────────────────────

pub fn emit_concat_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Add, grad_var, 1)
}

pub fn emit_slice_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Add, grad_var, 1)
}

pub fn emit_pad_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Add, grad_var, 1)
}

// ─── Pooling gradients ─────────────────────────────────────────────

pub fn emit_avg_pool_2d_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    let zero = push_literal(arena, 0, level)?;
    let step = push_application(arena, PrimitiveOp::Add, grad_var, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_global_avg_pool_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    emit_avg_pool_2d_grad(arena, level, grad_var)
}

// ─── Structured gradients ──────────────────────────────────────────

pub fn emit_attention_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    _q_var: u32,
    _k_var: u32,
) -> EmitResult {
    let zero       = push_literal(arena, 0, level)?;
    let mul        = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let inner      = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    let inner_rec  = push_recurse(arena, zero, zero, inner)?;
    let outer      = push_application(arena, PrimitiveOp::Add, inner_rec, 2)?;
    push_recurse(arena, zero, zero, outer)
}

pub fn emit_fused_swiglu_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    _x_var: u32,
    _w_var: u32,
) -> EmitResult {
    let zero  = push_literal(arena, 0, level)?;
    let mul   = push_application(arena, PrimitiveOp::Mul, grad_var, 2)?;
    let step  = push_application(arena, PrimitiveOp::Add, mul, 2)?;
    push_recurse(arena, zero, zero, step)
}

pub fn emit_unary_grad<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    grad_var: u32,
) -> EmitResult {
    push_application(arena, PrimitiveOp::Mul, grad_var, 2)
}

// ─── Type markers ──────────────────────────────────────────────────

macro_rules! declare_grad {
    ($name:ident, $iri_suffix:literal, $cap:expr, $emit_fn:ident) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/backward/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;

            /// Emit the backward Term tree. Single-arg form for unary-style
            /// gradients; multi-arg gradients carry the operand vars
            /// contiguously after the gradient var.
            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                grad_var: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, grad_var)
            }
        }
    };
}

/// Single-arg adapter for binary-arg backward emitters (so the marker's
/// `emit_term` keeps the unary surface).
fn binary_grad_adapter<const CAP: usize, F>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    emit: F,
) -> EmitResult
where F: Fn(&mut TermArena<CAP>, WittLevel, u32, u32) -> EmitResult,
{
    emit(arena, level, grad_var, grad_var.saturating_add(1))
}

fn ternary_grad_adapter<const CAP: usize, F>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    grad_var: u32,
    emit: F,
) -> EmitResult
where F: Fn(&mut TermArena<CAP>, WittLevel, u32, u32, u32) -> EmitResult,
{
    emit(arena, level, grad_var, grad_var.saturating_add(1), grad_var.saturating_add(2))
}

fn matmul_grad_a_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    binary_grad_adapter(arena, level, g, emit_matmul_grad_a)
}
fn matmul_grad_b_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    binary_grad_adapter(arena, level, g, emit_matmul_grad_b)
}
fn conv2d_grad_x_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    binary_grad_adapter(arena, level, g, emit_conv2d_grad_x)
}
fn conv2d_grad_w_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    binary_grad_adapter(arena, level, g, emit_conv2d_grad_w)
}
fn layer_norm_grad_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    ternary_grad_adapter(arena, level, g, emit_layer_norm_grad)
}
fn rms_norm_grad_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    ternary_grad_adapter(arena, level, g, emit_rms_norm_grad)
}
fn group_norm_grad_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    ternary_grad_adapter(arena, level, g, emit_group_norm_grad)
}
fn attention_grad_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    ternary_grad_adapter(arena, level, g, emit_attention_grad)
}
fn fused_swiglu_grad_unary<const CAP: usize>(arena: &mut TermArena<CAP>, level: WittLevel, g: u32) -> EmitResult {
    ternary_grad_adapter(arena, level, g, emit_fused_swiglu_grad)
}

declare_grad!(MatMulGradAOp,       "matmul_grad_a",        32, matmul_grad_a_unary);
declare_grad!(MatMulGradBOp,       "matmul_grad_b",        32, matmul_grad_b_unary);
declare_grad!(Conv2dGradXOp,       "conv2d_grad_x",        64, conv2d_grad_x_unary);
declare_grad!(Conv2dGradWOp,       "conv2d_grad_w",        64, conv2d_grad_w_unary);
declare_grad!(SoftmaxGradOp,       "softmax_grad",         32, emit_softmax_grad);
declare_grad!(LogSoftmaxGradOp,    "log_softmax_grad",     32, emit_log_softmax_grad);
declare_grad!(LayerNormGradOp,     "layer_norm_grad",      64, layer_norm_grad_unary);
declare_grad!(RmsNormGradOp,       "rms_norm_grad",        64, rms_norm_grad_unary);
declare_grad!(GroupNormGradOp,     "group_norm_grad",      64, group_norm_grad_unary);
declare_grad!(ReduceSumGradOp,     "reduce_sum_grad",      16, emit_reduce_sum_grad);
declare_grad!(ReduceMeanGradOp,    "reduce_mean_grad",     16, emit_reduce_mean_grad);
declare_grad!(ReduceProdGradOp,    "reduce_prod_grad",     16, emit_reduce_prod_grad);
declare_grad!(SubGradOp,           "sub_grad",             16, emit_sub_grad);
declare_grad!(MulGradOp,           "mul_grad",             16, emit_mul_grad);
declare_grad!(DivGradOp,           "div_grad",             32, emit_div_grad);
declare_grad!(PowGradOp,           "pow_grad",             64, emit_pow_grad);
declare_grad!(MinGradOp,           "min_grad",             16, emit_min_grad);
declare_grad!(MaxGradOp,           "max_grad",             16, emit_max_grad);
declare_grad!(ConcatGradOp,        "concat_grad",          16, emit_concat_grad);
declare_grad!(SliceGradOp,         "slice_grad",           16, emit_slice_grad);
declare_grad!(AvgPool2dGradOp,     "avg_pool_2d_grad",     32, emit_avg_pool_2d_grad);
declare_grad!(GlobalAvgPoolGradOp, "global_avg_pool_grad", 32, emit_global_avg_pool_grad);
declare_grad!(PadGradOp,           "pad_grad",             16, emit_pad_grad);
declare_grad!(AttentionGradOp,     "attention_grad",       96, attention_grad_unary);
declare_grad!(FusedSwiGluGradOp,   "fused_swiglu_grad",    64, fused_swiglu_grad_unary);
declare_grad!(UnaryGradOp,         "unary_grad",           32, emit_unary_grad);
