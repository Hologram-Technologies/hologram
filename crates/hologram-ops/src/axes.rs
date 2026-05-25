//! Hologram extended-compute axis declarations (wiki ADR-030 + ADR-031).
//!
//! prism-tensor's standard-library Layer-3 sub-crate carries the
//! `TensorAxis` (matmul) and `ActivationAxis` (relu, sigmoid_q)
//! surfaces; hologram-backend implements both on its f32 markers in
//! `hologram_backend::prism_axes`. The operations hologram performs
//! that fall *outside* prism-tensor's catalog — GEMM with bias,
//! Conv2d, LayerNorm/RmsNorm, Softmax, Attention — are declared here
//! as hologram-introduced axes, each emitted through foundation-sdk's
//! `axis!` macro so callers reach them through the same `AxisExtension`
//! interface as the prism-canonical axes.
//!
//! Each axis follows the wiki convention:
//!
//! - Every kernel method has signature
//!   `fn name(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>`.
//! - `AXIS_ADDRESS` is a stable IRI under `https://hologram.uor.foundation/axis/`.
//! - `MAX_OUTPUT_BYTES` is a conservative ceiling consistent with
//!   hologram-host's `HostBounds::AXIS_OUTPUT_BYTES_MAX`.
//!
//! Concrete impls of these axes live in `hologram-backend` (the
//! kernel-realization layer). Per ADR-055 each impl receives the
//! foundation-sdk's empty `body_arena()` (the primitive-fast-path
//! equivalent realization).

use uor_foundation::enforcement::ShapeViolation;
use uor_foundation_sdk::axis;

axis! {
    /// Hologram's general-matrix-multiply axis: `out = α·A·B + β·C`.
    ///
    /// Beyond prism-tensor's pure `matmul` (`out = A·B`), GEMM admits
    /// an additive bias `C` and scalar coefficients `α`/`β`. The input
    /// byte layout per impl is `[A_bytes || B_bytes || C_bytes ||
    /// α_bytes || β_bytes]`; output is row-major `out_bytes`.
    pub trait HologramGemmAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramGemmAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        /// Compute `out = α·A·B + β·C`.
        fn gemm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}

axis! {
    /// Hologram's 2D convolution axis. Carries `conv2d` (forward) and
    /// `conv_transpose_2d` (transposed / fractional-stride).
    ///
    /// Input layout: `[X_bytes || W_bytes]`. Per-impl const generics
    /// pin the shape (batch, channels, kernel, stride, padding).
    pub trait HologramConvAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramConvAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        /// Standard 2D convolution.
        fn conv2d(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        /// Transposed 2D convolution (a.k.a. fractional-stride / deconvolution).
        fn conv_transpose_2d(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}

axis! {
    /// Hologram's normalization axis. Carries layer / RMS / group /
    /// instance norms plus the AddRmsNorm fused-residual variant.
    ///
    /// Input layout: `[X_bytes || gamma_bytes || beta_bytes (|| residual_bytes)]`.
    pub trait HologramNormAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramNormAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        fn layer_norm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn rms_norm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn group_norm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn instance_norm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn add_rms_norm(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}

axis! {
    /// Hologram's reduction + softmax axis. The byte-domain compose
    /// of every reduction operator hologram emits in `hologram-ops`.
    pub trait HologramReduceAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramReduceAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        fn reduce_sum(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn reduce_mean(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn reduce_prod(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn reduce_min(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn reduce_max(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn softmax(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn log_softmax(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}

axis! {
    /// Hologram's structured-composition axis. Currently carries
    /// scaled-dot-product `attention` and `fused_swiglu`; future
    /// fused transformer-block compositions land here.
    ///
    /// Input layout (attention): `[Q_bytes || K_bytes || V_bytes]`;
    /// (fused_swiglu): `[X_bytes || W_bytes]`.
    pub trait HologramStructuredAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramStructuredAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        fn attention(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
        fn fused_swiglu(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}

axis! {
    /// Hologram's quantization axis. Per spec X-5 + the prism
    /// "tensor saturation" pattern (wiki ADR-056): byte-comparator-
    /// using kernels live in axis bodies, not in route bodies.
    ///
    /// Input layout: `[Q_bytes || scale_bits (4) || zero_point (4)]`
    /// where the scale is `f32::to_bits` and zero-point is `i32`.
    pub trait HologramQuantAxis: AxisExtension {
        const AXIS_ADDRESS: &'static str = "https://hologram.uor.foundation/axis/HologramQuantAxis";
        const MAX_OUTPUT_BYTES: usize = 8_192;
        /// Dequantize packed INT8 / INT4 weights into a float buffer.
        fn dequantize(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation>;
    }
}
