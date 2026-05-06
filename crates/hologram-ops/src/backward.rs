//! Backward gradient ops (spec V.4).
//!
//! Each differentiable forward op declares a companion backward marker.
//! Per ADR-043, backward Term trees are emitted at graph-build time, not
//! traversed at runtime.

use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use crate::emit::{push_application, EmitResult};

macro_rules! declare_grad {
    ($name:ident, $iri_suffix:literal, $cap:expr, $primary:expr) => {
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/backward/",
                $iri_suffix,
            );
            pub const CAP: usize = $cap;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                _level: WittLevel,
                grad_var: u32,
            ) -> EmitResult {
                push_application(arena, $primary, grad_var, 1)
            }
        }
    };
}

declare_grad!(MatMulGradAOp,       "matmul_grad_a",        32, PrimitiveOp::Mul);
declare_grad!(MatMulGradBOp,       "matmul_grad_b",        32, PrimitiveOp::Mul);
declare_grad!(Conv2dGradXOp,       "conv2d_grad_x",        64, PrimitiveOp::Mul);
declare_grad!(Conv2dGradWOp,       "conv2d_grad_w",        64, PrimitiveOp::Mul);
declare_grad!(SoftmaxGradOp,       "softmax_grad",         32, PrimitiveOp::Mul);
declare_grad!(LogSoftmaxGradOp,    "log_softmax_grad",     32, PrimitiveOp::Sub);
declare_grad!(LayerNormGradOp,     "layer_norm_grad",      64, PrimitiveOp::Mul);
declare_grad!(RmsNormGradOp,       "rms_norm_grad",        64, PrimitiveOp::Mul);
declare_grad!(GroupNormGradOp,     "group_norm_grad",      64, PrimitiveOp::Mul);
declare_grad!(ReduceSumGradOp,     "reduce_sum_grad",      16, PrimitiveOp::Add);
declare_grad!(ReduceMeanGradOp,    "reduce_mean_grad",     16, PrimitiveOp::Mul);
declare_grad!(ReduceProdGradOp,    "reduce_prod_grad",     16, PrimitiveOp::Mul);
declare_grad!(SubGradOp,           "sub_grad",             16, PrimitiveOp::Sub);
declare_grad!(MulGradOp,           "mul_grad",             16, PrimitiveOp::Mul);
declare_grad!(DivGradOp,           "div_grad",             32, PrimitiveOp::Mul);
declare_grad!(PowGradOp,           "pow_grad",             64, PrimitiveOp::Mul);
declare_grad!(MinGradOp,           "min_grad",             16, PrimitiveOp::And);
declare_grad!(MaxGradOp,           "max_grad",             16, PrimitiveOp::And);
declare_grad!(ConcatGradOp,        "concat_grad",          16, PrimitiveOp::Add);
declare_grad!(SliceGradOp,         "slice_grad",           16, PrimitiveOp::Add);
declare_grad!(AvgPool2dGradOp,     "avg_pool_2d_grad",     32, PrimitiveOp::Add);
declare_grad!(GlobalAvgPoolGradOp, "global_avg_pool_grad", 32, PrimitiveOp::Add);
declare_grad!(PadGradOp,           "pad_grad",             16, PrimitiveOp::Add);
declare_grad!(AttentionGradOp,     "attention_grad",       96, PrimitiveOp::Mul);
declare_grad!(FusedSwiGluGradOp,   "fused_swiglu_grad",    64, PrimitiveOp::Mul);
declare_grad!(UnaryGradOp,         "unary_grad",           32, PrimitiveOp::Mul);
