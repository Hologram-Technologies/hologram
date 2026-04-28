//! Canonical semantic operation model for Hologram.
//!
//! `hologram-ops` is the single home for *all* op-related artefacts:
//!
//! - **Semantic identity** — the `Op` trait + per-op marker structs +
//!   the closed `SemanticOp` enum.
//! - **Executable form** — the `KernelCall` enum + per-op `Call`
//!   structs.
//! - **Reference CPU kernels** — forward (and where applicable
//!   backward) implementations.
//! - **(Future)** per-op LUT generators, landing alongside their kernel
//!   when Plan 074 (`uor-foundation` 0.3.0) exposes the address API.
//!
//! Each canonical op lives in its own file under [`kernels`]: the
//! marker struct, the `Op` trait impl, the `Call` struct(s), and the
//! kernel function(s) are co-located. To add a new op you create one
//! new kernel file plus three small touch points (the `SemanticOp`
//! variant + macro arm in [`semantic`], the `KernelCall` variant +
//! `dispatch` arm in [`kernels::mod`]) — the bulk of the op's
//! definition is in one file.
//!
//! See [ADR-044](../../specs/adrs/044-op-trait-canonical-semantics.md)
//! and [ADR-045](../../specs/adrs/045-ops-as-single-source-of-truth.md).

#![deny(missing_docs)]

pub mod attrs;
pub mod kernels;
pub mod semantic;
pub mod span;
pub mod trait_def;

pub use attrs::{
    AttentionAttrs, ClipAttrs, ConcatAttrs, Conv2dAttrs, ConvTransposeAttrs, CumSumAttrs,
    ExpandAttrs, GemmAttrs, GlobalAvgPoolAttrs, GroupNormAttrs, LrnAttrs, MatMulAttrs, NormAttrs,
    PadAttrs, Pool2dAttrs, ReduceAttrs, ResizeAttrs, RotaryEmbeddingAttrs, SliceAttrs,
    SoftmaxAttrs, TransposeAttrs,
};
pub use kernels::{
    add::Add,
    attention::Attention,
    binary::{
        And, Div, Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Max, Min, Mod, Mul, Or, Pow,
        Sub, Xor,
    },
    clip::Clip,
    conv::Conv2d,
    conv_transpose::ConvTranspose2d,
    cumsum::CumSum,
    dispatch,
    expand::Expand,
    fused::FusedSwiGlu,
    gemm::Gemm,
    lrn::Lrn,
    matmul::MatMul,
    norm::{AddRmsNorm, GroupNorm, InstanceNorm, LayerNorm, RmsNorm},
    pad::Pad,
    pool::{AvgPool2d, GlobalAvgPool, MaxPool2d},
    reduce::{ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum},
    reshape::Reshape,
    resize::Resize,
    rotary::RotaryEmbedding,
    select::Where,
    shape::{Concat, Slice, Transpose},
    softmax::{LogSoftmax, Softmax},
    unary::{
        Abs, Ceil, Cos, Erf, Exp, Floor, Gelu, IsNaN, Log, Neg, Not, Reciprocal, Relu, Round,
        Sigmoid, Sign, Silu, Sin, Sqrt, Tanh,
    },
    AddCall, AddGradCall, AddRmsNormCall, AddRmsNormGradCall, AttentionCall, AttentionGradCall,
    BinaryCall, ClipCall, ConcatCall, ConcatGradCall, Conv2dCall, Conv2dGradCall,
    ConvTranspose2dGradCall, ConvTransposeCall, CumSumCall, DivGradCall, ExpandCall,
    FusedSwiGluGradCall, GemmCall, GlobalAvgPoolCall, GlobalAvgPoolGradCall, GroupNormCall,
    GroupNormGradCall, InstanceNormGradCall, KernelCall, LayerNormGradCall, LrnCall, MatMulCall,
    MatMulGradACall, MatMulGradBCall, MinMaxGradCall, MinMaxGradKind, MulGradCall, NegGradCall,
    NormFullCall, NormScaleCall, PadCall, Pool2dCall, Pool2dGradCall, Pool2dKind, PowGradCall,
    ReduceArgGradCall, ReduceArgGradKind, ReduceCall, ReduceGradCall, ReduceGradKind, ReduceKind,
    ReduceProdGradCall, ReshapeCall, ResizeCall, RmsNormGradCall, RotaryEmbeddingCall, SliceCall,
    SliceGradCall, SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind, SubGradCall, TransposeCall,
    TransposeGradCall, UnaryCall, UnaryGradCall, UnaryGradKind, UnaryKind, WhereCall,
};
pub use semantic::SemanticOp;
pub use span::SlotSpan;
pub use trait_def::{BackwardRule, Op, OpCategory, OpSignature};
