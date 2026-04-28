//! Executable kernel form.
//!
//! `KernelCall` is the closed enum of all op invocations the executor can
//! dispatch. Each variant pairs an op identity with its pre-resolved
//! `SlotSpan`s and op-specific parameters. The single [`dispatch`]
//! function is the executor's hot loop body — every backend that wants
//! to reuse the same plan implements its own variant of this match.
//!
//! Per-op modules (`add`, `matmul`, `unary`, …) own both the `Call`
//! struct and the kernel function, so adding a new op means editing one
//! file plus three small touch points (this enum, the dispatch arm, and
//! the planner). See ADR-044.

pub mod add;
pub mod attention;
pub mod binary;
pub mod clip;
pub mod conv;
pub mod conv_transpose;
pub mod cumsum;
pub mod expand;
pub mod fused;
pub mod gemm;
pub mod lrn;
pub mod matmul;
pub mod norm;
pub mod pad;
pub mod pool;
pub mod reduce;
pub mod reshape;
pub mod resize;
pub mod rotary;
pub mod select;
pub mod shape;
pub mod softmax;
pub mod unary;

pub use add::{AddCall, AddGradCall};
pub use attention::{AttentionCall, AttentionGradCall};
pub use binary::{
    BinaryCall, DivGradCall, MinMaxGradCall, MinMaxGradKind, MulGradCall, PowGradCall, SubGradCall,
};
pub use clip::ClipCall;
pub use conv::{Conv2dCall, Conv2dGradCall};
pub use conv_transpose::{ConvTranspose2dGradCall, ConvTransposeCall};
pub use cumsum::CumSumCall;
pub use expand::ExpandCall;
pub use fused::FusedSwiGluGradCall;
pub use gemm::GemmCall;
pub use lrn::LrnCall;
pub use matmul::{MatMulCall, MatMulGradACall, MatMulGradBCall};
pub use norm::{
    AddRmsNormCall, AddRmsNormGradCall, GroupNormCall, GroupNormGradCall, InstanceNormGradCall,
    LayerNormGradCall, NormFullCall, NormScaleCall, RmsNormGradCall,
};
pub use pad::PadCall;
pub use pool::{GlobalAvgPoolCall, GlobalAvgPoolGradCall, Pool2dCall, Pool2dGradCall, Pool2dKind};
pub use reduce::{
    ReduceArgGradCall, ReduceArgGradKind, ReduceCall, ReduceGradCall, ReduceGradKind, ReduceKind,
    ReduceProdGradCall,
};
pub use reshape::ReshapeCall;
pub use resize::ResizeCall;
pub use rotary::RotaryEmbeddingCall;
pub use select::WhereCall;
pub use shape::{
    ConcatCall, ConcatGradCall, SliceCall, SliceGradCall, TransposeCall, TransposeGradCall,
};
pub use softmax::{SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind};
pub use unary::{NegGradCall, UnaryCall, UnaryGradCall, UnaryGradKind, UnaryKind};

/// A single executable kernel call.
///
/// Variants are dispatched via `match` in [`dispatch`] — no virtual
/// dispatch, no function pointers. Adding a new kernel is a new variant
/// here plus a new arm in `dispatch`, and the kernel logic lives in the
/// corresponding `kernels/<op>.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCall {
    /// Forward elementwise add: `c = a + b`.
    Add(AddCall),
    /// Backward of add: `da += dc`, `db += dc`.
    AddGrad(AddGradCall),
    /// Forward elementwise sub: `c = a - b`.
    Sub(BinaryCall),
    /// Backward of sub: `da += dc`, `db += -dc`.
    SubGrad(SubGradCall),
    /// Forward elementwise mul: `c = a * b`.
    Mul(BinaryCall),
    /// Backward of mul: `da += dc * b`, `db += dc * a`.
    MulGrad(MulGradCall),
    /// Backward of div: `da += dc / b`, `db += -dc * a / b²`.
    DivGrad(DivGradCall),
    /// Backward of neg: `da += -dc`.
    NegGrad(NegGradCall),
    /// Backward of an elementwise unary op (Relu, Sigmoid, Tanh, Exp,
    /// Log, Sqrt, Abs, Reciprocal).
    UnaryGrad(UnaryGradCall, UnaryGradKind),
    /// Backward of `Min` / `Max` (gradient routes to whichever input wins).
    MinMaxGrad(MinMaxGradCall, MinMaxGradKind),
    /// Backward of `ReduceSum` / `ReduceMean` (broadcast back).
    ReduceGrad(ReduceGradCall, ReduceGradKind),
    /// Backward of `Concat` (split `dC` into the two input grads).
    ConcatGrad(ConcatGradCall),
    /// Backward of `Slice` (scatter `dC` into the slice region).
    SliceGrad(SliceGradCall),
    /// Backward of `Transpose` (apply inverse permutation).
    TransposeGrad(TransposeGradCall),
    /// Backward of `Pow`.
    PowGrad(PowGradCall),
    /// Backward of `Softmax` / `LogSoftmax`.
    SoftmaxGrad(SoftmaxGradCall, SoftmaxGradKind),
    /// Backward of `ReduceMax` / `ReduceMin` (argmax/argmin routing).
    ReduceArgGrad(ReduceArgGradCall, ReduceArgGradKind),
    /// Backward of `ReduceProd` (zero-aware).
    ReduceProdGrad(ReduceProdGradCall),
    /// Backward of `RmsNorm` (`dx`, `dw`).
    RmsNormGrad(RmsNormGradCall),
    /// Backward of `LayerNorm` (`dx`, `dw`, `db`).
    LayerNormGrad(LayerNormGradCall),
    /// Backward of `InstanceNorm` (`dx`, `dw`).
    InstanceNormGrad(InstanceNormGradCall),
    /// Backward of `AddRmsNorm` (`d_residual`, `d_input`, `dw`).
    AddRmsNormGrad(AddRmsNormGradCall),
    /// Backward of `MaxPool2d` / `AvgPool2d`.
    Pool2dGrad(Pool2dGradCall, Pool2dKind),
    /// Backward of `GlobalAvgPool`.
    GlobalAvgPoolGrad(GlobalAvgPoolGradCall),
    /// Backward of `GroupNorm` (`dx`, `dw`, `db`).
    GroupNormGrad(GroupNormGradCall),
    /// Backward of `FusedSwiGlu` (`d_gate`, `d_up`).
    FusedSwiGluGrad(FusedSwiGluGradCall),
    /// Backward of `Conv2d` (`dx`, `dw`, `db`).
    Conv2dGrad(Conv2dGradCall),
    /// Backward of `ConvTranspose2d` (`dx`, `dw`, `db`).
    ConvTranspose2dGrad(ConvTranspose2dGradCall),
    /// Backward of canonical scaled-dot-product `Attention`
    /// (`dQ`, `dK`, `dV`).
    AttentionGrad(AttentionGradCall),
    /// Forward elementwise div: `c = a / b`.
    Div(BinaryCall),
    /// Forward elementwise pow: `c = a^b`.
    Pow(BinaryCall),
    /// Forward elementwise mod: `c = a mod b`.
    Mod(BinaryCall),
    /// Forward elementwise min.
    Min(BinaryCall),
    /// Forward elementwise max.
    Max(BinaryCall),
    /// Forward elementwise equality (1.0 / 0.0).
    Equal(BinaryCall),
    /// Forward elementwise less-than (1.0 / 0.0).
    Less(BinaryCall),
    /// Forward elementwise less-or-equal.
    LessOrEqual(BinaryCall),
    /// Forward elementwise greater-than.
    Greater(BinaryCall),
    /// Forward elementwise greater-or-equal.
    GreaterOrEqual(BinaryCall),
    /// Forward elementwise logical AND on f32 truthiness.
    And(BinaryCall),
    /// Forward elementwise logical OR on f32 truthiness.
    Or(BinaryCall),
    /// Forward elementwise logical XOR on f32 truthiness.
    Xor(BinaryCall),
    /// Forward elementwise unary, dispatched by [`UnaryKind`].
    Unary(UnaryCall, UnaryKind),
    /// Forward softmax along the last axis.
    Softmax(SoftmaxCall),
    /// Forward log-softmax along the last axis.
    LogSoftmax(SoftmaxCall),
    /// Forward reshape: contiguous copy from input to output.
    Reshape(ReshapeCall),
    /// Forward physical transpose (up to 4-D).
    Transpose(TransposeCall),
    /// Forward last-axis contiguous slice.
    Slice(SliceCall),
    /// Forward last-axis concatenation of two operands.
    Concat(ConcatCall),
    /// Forward RMSNorm.
    RmsNorm(NormScaleCall),
    /// Forward LayerNorm.
    LayerNorm(NormFullCall),
    /// Forward InstanceNorm (mean/var per row, scale by weight, no bias).
    InstanceNorm(NormScaleCall),
    /// Forward GroupNorm.
    GroupNorm(GroupNormCall),
    /// Forward residual add followed by RMSNorm.
    AddRmsNorm(AddRmsNormCall),
    /// Forward fused SwiGLU: `out = silu(gate) * up`.
    FusedSwiGlu(BinaryCall),
    /// Forward 2-D convolution (direct, reference correctness only).
    Conv2d(Conv2dCall),
    /// Forward 2-D pool (max or avg, dispatched by [`Pool2dKind`]).
    Pool2d(Pool2dCall, Pool2dKind),
    /// Forward global average pool (spatial → 1×1).
    GlobalAvgPool(GlobalAvgPoolCall),
    /// Forward row-wise reduction (sum/mean/max/min/prod).
    Reduce(ReduceCall, ReduceKind),
    /// Forward ternary select: `out = c ? x : y`.
    Where(WhereCall),
    /// Forward elementwise clamp.
    Clip(ClipCall),
    /// Forward cumulative sum along the last axis.
    CumSum(CumSumCall),
    /// Forward constant-mode pad (NCHW symmetric).
    Pad(PadCall),
    /// Forward nearest-neighbor resize (NCHW).
    ResizeNearest(ResizeCall),
    /// Forward bilinear resize (NCHW).
    ResizeLinear(ResizeCall),
    /// Forward Local Response Normalization (cross-channel).
    Lrn(LrnCall),
    /// Forward 2-D transposed convolution.
    ConvTranspose2d(ConvTransposeCall),
    /// Forward general matmul (`Y = α·A@B + β·C`).
    Gemm(GemmCall),
    /// Forward broadcast expand to target shape.
    Expand(ExpandCall),
    /// Forward half-rotation rotary embedding.
    RotaryEmbedding(RotaryEmbeddingCall),
    /// Forward scaled-dot-product attention (ADR-049).
    Attention(AttentionCall),
    /// Forward matmul: `c = a @ b` (row-major, no transpose).
    MatMul(MatMulCall),
    /// Backward of matmul w.r.t. A: `da += dc @ bᵀ`.
    MatMulGradA(MatMulGradACall),
    /// Backward of matmul w.r.t. B: `db += aᵀ @ dc`.
    MatMulGradB(MatMulGradBCall),
}

/// Execute a single kernel call against `storage` (the executor's flat
/// f32 workspace).
///
/// This is the canonical CPU dispatcher — alternative backends (Metal,
/// WebGPU, Atlas) implement their own match but consume the same
/// `KernelCall` form. The dispatch is exhaustive: a new variant produces
/// a compile error here.
#[inline]
pub fn dispatch(storage: &mut [f32], call: &KernelCall) {
    match call {
        KernelCall::Add(c) => add::add(storage, c),
        KernelCall::AddGrad(c) => add::add_grad(storage, c),
        KernelCall::Sub(c) => binary::sub(storage, c),
        KernelCall::SubGrad(c) => binary::sub_grad(storage, c),
        KernelCall::Mul(c) => binary::mul(storage, c),
        KernelCall::MulGrad(c) => binary::mul_grad(storage, c),
        KernelCall::DivGrad(c) => binary::div_grad(storage, c),
        KernelCall::NegGrad(c) => unary::neg_grad(storage, c),
        KernelCall::UnaryGrad(c, kind) => unary::dispatch_grad(storage, c, *kind),
        KernelCall::MinMaxGrad(c, kind) => binary::min_max_grad(storage, c, *kind),
        KernelCall::ReduceGrad(c, kind) => reduce::dispatch_grad(storage, c, *kind),
        KernelCall::ConcatGrad(c) => shape::concat_grad(storage, c),
        KernelCall::SliceGrad(c) => shape::slice_grad(storage, c),
        KernelCall::TransposeGrad(c) => shape::transpose_grad(storage, c),
        KernelCall::PowGrad(c) => binary::pow_grad(storage, c),
        KernelCall::SoftmaxGrad(c, kind) => softmax::dispatch_grad(storage, c, *kind),
        KernelCall::ReduceArgGrad(c, kind) => reduce::dispatch_arg_grad(storage, c, *kind),
        KernelCall::ReduceProdGrad(c) => reduce::reduce_prod_grad(storage, c),
        KernelCall::RmsNormGrad(c) => norm::rms_norm_grad(storage, c),
        KernelCall::LayerNormGrad(c) => norm::layer_norm_grad(storage, c),
        KernelCall::InstanceNormGrad(c) => norm::instance_norm_grad(storage, c),
        KernelCall::AddRmsNormGrad(c) => norm::add_rms_norm_grad(storage, c),
        KernelCall::Pool2dGrad(c, kind) => pool::dispatch_pool2d_grad(storage, c, *kind),
        KernelCall::GlobalAvgPoolGrad(c) => pool::global_avg_pool_grad(storage, c),
        KernelCall::GroupNormGrad(c) => norm::group_norm_grad(storage, c),
        KernelCall::FusedSwiGluGrad(c) => fused::fused_swiglu_grad(storage, c),
        KernelCall::Conv2dGrad(c) => conv::conv2d_grad(storage, c),
        KernelCall::ConvTranspose2dGrad(c) => conv_transpose::conv_transpose_2d_grad(storage, c),
        KernelCall::AttentionGrad(c) => attention::attention_grad(storage, c),
        KernelCall::Div(c) => binary::div(storage, c),
        KernelCall::Pow(c) => binary::pow(storage, c),
        KernelCall::Mod(c) => binary::modulo(storage, c),
        KernelCall::Min(c) => binary::min(storage, c),
        KernelCall::Max(c) => binary::max(storage, c),
        KernelCall::Equal(c) => binary::equal(storage, c),
        KernelCall::Less(c) => binary::less(storage, c),
        KernelCall::LessOrEqual(c) => binary::less_or_equal(storage, c),
        KernelCall::Greater(c) => binary::greater(storage, c),
        KernelCall::GreaterOrEqual(c) => binary::greater_or_equal(storage, c),
        KernelCall::And(c) => binary::and(storage, c),
        KernelCall::Or(c) => binary::or(storage, c),
        KernelCall::Xor(c) => binary::xor(storage, c),
        KernelCall::Unary(c, kind) => unary::dispatch(storage, c, *kind),
        KernelCall::Softmax(c) => softmax::softmax(storage, c),
        KernelCall::LogSoftmax(c) => softmax::log_softmax(storage, c),
        KernelCall::Reshape(c) => reshape::reshape(storage, c),
        KernelCall::Transpose(c) => shape::transpose(storage, c),
        KernelCall::Slice(c) => shape::slice(storage, c),
        KernelCall::Concat(c) => shape::concat(storage, c),
        KernelCall::RmsNorm(c) => norm::rms_norm(storage, c),
        KernelCall::LayerNorm(c) => norm::layer_norm(storage, c),
        KernelCall::InstanceNorm(c) => norm::instance_norm(storage, c),
        KernelCall::GroupNorm(c) => norm::group_norm(storage, c),
        KernelCall::AddRmsNorm(c) => norm::add_rms_norm(storage, c),
        KernelCall::FusedSwiGlu(c) => fused::fused_swiglu(storage, c),
        KernelCall::Conv2d(c) => conv::conv2d(storage, c),
        KernelCall::Pool2d(c, kind) => pool::dispatch_pool2d(storage, c, *kind),
        KernelCall::GlobalAvgPool(c) => pool::global_avg_pool(storage, c),
        KernelCall::Reduce(c, kind) => reduce::dispatch(storage, c, *kind),
        KernelCall::Where(c) => select::r#where(storage, c),
        KernelCall::Clip(c) => clip::clip(storage, c),
        KernelCall::CumSum(c) => cumsum::cumsum(storage, c),
        KernelCall::Pad(c) => pad::pad(storage, c),
        KernelCall::ResizeNearest(c) => resize::resize_nearest(storage, c),
        KernelCall::ResizeLinear(c) => resize::resize_linear(storage, c),
        KernelCall::Lrn(c) => lrn::lrn(storage, c),
        KernelCall::ConvTranspose2d(c) => conv_transpose::conv_transpose_2d(storage, c),
        KernelCall::Gemm(c) => gemm::gemm(storage, c),
        KernelCall::Expand(c) => expand::expand(storage, c),
        KernelCall::RotaryEmbedding(c) => rotary::rotary_embedding(storage, c),
        KernelCall::Attention(c) => attention::attention(storage, c),
        KernelCall::MatMul(c) => matmul::matmul(storage, c),
        KernelCall::MatMulGradA(c) => matmul::matmul_grad_a(storage, c),
        KernelCall::MatMulGradB(c) => matmul::matmul_grad_b(storage, c),
    }
}
