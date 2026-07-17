//! Hologram's implementations of prism-tensor's canonical compute axes
//! (wiki ADR-031). Every hologram-emitted kernel is reachable through
//! the prism axis surface — external callers using `TensorAxis::matmul`
//! / `ActivationAxis::{relu, sigmoid_q}` invoke hologram's f32 CPU
//! kernels by selecting the markers declared here.
//!
//! Per ADR-055 the substrate-Term verb body discipline is satisfied by
//! the foundation-sdk-emitted default empty `body_arena()` — these
//! impls are operationally hand-written for-loop kernels, matching the
//! pattern prism-tensor uses for `CpuI8MatmulSquare`.
//!
//! Two axes are implemented:
//!
//! - `prism::tensor::TensorAxis` on `HologramF32MatmulSquare<DIM>` —
//!   square `DIM × DIM` f32 matmul. Input layout: `A || B` (each
//!   `DIM*DIM*4` bytes, row-major); output layout: `C` (`DIM*DIM*4`
//!   bytes, row-major).
//!
//! - `prism::tensor::ActivationAxis` on `HologramF32VectorActivation<N>`
//!   — element-wise nonlinearities over an `N`-element f32 vector
//!   (`N*4` bytes input / output). Both `relu` and `sigmoid_q` (the
//!   canonical sigmoid surface) are provided.

use prism::pipeline::ShapeViolation;
use prism::tensor::{ActivationAxis, TensorAxis};
use prism_tensor::{axis_extension_impl_for_activation_axis, axis_extension_impl_for_tensor_axis};

/// Maximum square dimension supported by `HologramF32MatmulSquare`.
/// Mirrors prism-tensor's `MAX_TENSOR_DIM = 16` so external callers
/// can interchange the i8 and f32 impls at the same DIM.
pub const HOLOGRAM_MAX_TENSOR_DIM: usize = 16;

/// Maximum vector length supported by `HologramF32VectorActivation`.
pub const HOLOGRAM_MAX_ACTIVATION_LEN: usize = 256;

fn shape_violation(constraint_iri: &'static str) -> ShapeViolation {
    ShapeViolation {
        shape_iri: "https://hologram.uor.foundation/axis/HologramTensorShape",
        constraint_iri,
        property_iri: "https://hologram.uor.foundation/axis/inputBytes",
        expected_range: "https://hologram.uor.foundation/axis/InputArity",
        min_count: 0,
        max_count: 0,
        kind: prism::pipeline::ViolationKind::ValueCheck,
    }
}

#[inline]
fn read_f32(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([
        bytes[4 * i],
        bytes[4 * i + 1],
        bytes[4 * i + 2],
        bytes[4 * i + 3],
    ])
}

#[inline]
fn write_f32(bytes: &mut [u8], i: usize, v: f32) {
    bytes[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}

// ─── TensorAxis: f32 square matmul ─────────────────────────────────

/// Hologram f32 square-matmul kernel marker.
///
/// `DIM × DIM` row-major f32 matrices; input is `A || B`
/// (`8 * DIM * DIM` bytes total), output is `C = A·B`
/// (`4 * DIM * DIM` bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct HologramF32MatmulSquare<const DIM: usize>;

impl<const DIM: usize> TensorAxis for HologramF32MatmulSquare<DIM> {
    const AXIS_ADDRESS: &'static str =
        "https://hologram.uor.foundation/axis/TensorAxis/HologramF32MatmulSquare";
    const MAX_OUTPUT_BYTES: usize = 4 * DIM * DIM;

    fn matmul(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation> {
        if DIM == 0 || DIM > HOLOGRAM_MAX_TENSOR_DIM {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/dimInRange",
            ));
        }
        let mat_bytes = 4 * DIM * DIM;
        let input_bytes = 2 * mat_bytes;
        let output_bytes = mat_bytes;
        if input.len() != input_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/inputByteLength",
            ));
        }
        if out.len() < output_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/outputByteLength",
            ));
        }
        let (a_bytes, b_bytes) = input.split_at(mat_bytes);
        for row in 0..DIM {
            for col in 0..DIM {
                let mut acc: f32 = 0.0;
                for k in 0..DIM {
                    let a = read_f32(a_bytes, row * DIM + k);
                    let b = read_f32(b_bytes, k * DIM + col);
                    acc += a * b;
                }
                write_f32(out, row * DIM + col, acc);
            }
        }
        Ok(output_bytes)
    }
}

axis_extension_impl_for_tensor_axis!(@generic HologramF32MatmulSquare<DIM>, [const DIM: usize]);

/// 4×4 f32 matmul — the canonical small-tensor reference for hologram.
pub type HologramF32Tensor4x4Matmul = HologramF32MatmulSquare<4>;
/// 8×8 f32 matmul.
pub type HologramF32Tensor8x8Matmul = HologramF32MatmulSquare<8>;
/// 16×16 f32 matmul (the `HOLOGRAM_MAX_TENSOR_DIM` ceiling).
pub type HologramF32Tensor16x16Matmul = HologramF32MatmulSquare<16>;

// ─── Runtime-dimensioned TensorAxis: the execution dispatch surface ───

/// Header width of the runtime matmul axis input: `[m, k, n : u32 LE]`.
#[cfg(feature = "cpu")]
pub const HOLOGRAM_MATMUL_HEADER_BYTES: usize = 12;

/// Runtime-dimensioned f32 matmul exposed as prism-tensor's [`TensorAxis`]
/// (wiki ADR-031) — the canonical prism axis surface hologram's CPU executor
/// dispatches `MatMul` through (rather than calling the kernel ad hoc).
///
/// Input layout: `[m: u32 LE][k: u32 LE][n: u32 LE] || A(m·k·4) || B(k·n·4)`
/// (row-major f32); output: `C = A·B` row-major (`m·n·4` bytes). The body
/// delegates to the register-blocked FMA micro-kernel
/// ([`crate::cpu::simd::matmul_f32_blocked`]), so dispatching through the
/// axis costs only operand marshalling — `Θ(m·k + k·n)` vs the kernel's
/// `Θ(m·k·n)`, i.e. negligible at model scale.
#[cfg(feature = "cpu")]
#[derive(Debug, Clone, Copy, Default)]
pub struct HologramTensorMatmulF32;

#[cfg(feature = "cpu")]
impl TensorAxis for HologramTensorMatmulF32 {
    const AXIS_ADDRESS: &'static str =
        "https://hologram.uor.foundation/axis/TensorAxis/HologramMatmulF32";
    // Runtime-sized: the caller supplies the workspace output slot as `out`,
    // so there is no substrate-arbitrary byte-width cap (ADR-060).
    const MAX_OUTPUT_BYTES: usize = 0;

    fn matmul(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation> {
        if input.len() < HOLOGRAM_MATMUL_HEADER_BYTES {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/matmulHeader",
            ));
        }
        let rd = |o: usize| {
            u32::from_le_bytes([input[o], input[o + 1], input[o + 2], input[o + 3]]) as usize
        };
        let (m, k, n) = (rd(0), rd(4), rd(8));
        let a_bytes = m * k * 4;
        let b_bytes = k * n * 4;
        let o_bytes = m * n * 4;
        let ab = &input[HOLOGRAM_MATMUL_HEADER_BYTES..];
        if ab.len() != a_bytes + b_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/inputByteLength",
            ));
        }
        if out.len() < o_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramTensorShape/outputByteLength",
            ));
        }
        if m == 0 || k == 0 || n == 0 {
            return Ok(o_bytes);
        }
        let out32 = bytemuck::cast_slice_mut::<u8, f32>(&mut out[..o_bytes]);
        let mut scratch = alloc::vec::Vec::new();
        // Zero-copy operand views when 4-aligned (the marshalling buffer is);
        // otherwise a one-shot aligned copy (always correct).
        match (
            bytemuck::try_cast_slice::<u8, f32>(&ab[..a_bytes]),
            bytemuck::try_cast_slice::<u8, f32>(&ab[a_bytes..]),
        ) {
            (Ok(a32), Ok(b32)) => {
                crate::cpu::simd::matmul_f32_blocked(a32, b32, out32, m, k, n, &mut scratch);
            }
            _ => {
                let mut af = alloc::vec![0f32; m * k];
                let mut bf = alloc::vec![0f32; k * n];
                for (i, v) in af.iter_mut().enumerate() {
                    *v = read_f32(ab, i);
                }
                for (i, v) in bf.iter_mut().enumerate() {
                    *v = read_f32(&ab[a_bytes..], i);
                }
                crate::cpu::simd::matmul_f32_blocked(&af, &bf, out32, m, k, n, &mut scratch);
            }
        }
        Ok(o_bytes)
    }
}

#[cfg(feature = "cpu")]
axis_extension_impl_for_tensor_axis!(HologramTensorMatmulF32);

// ─── ActivationAxis: f32 element-wise nonlinearities ──────────────

/// Hologram f32 element-wise activation kernel marker.
///
/// `N`-element row-major f32 vector. Input/output layout: `N*4` bytes
/// little-endian f32 little-endian. Provides `relu` (max(0, x)) and
/// `sigmoid_q` (canonical 1/(1+exp(-x))).
#[derive(Debug, Clone, Copy, Default)]
pub struct HologramF32VectorActivation<const N: usize>;

impl<const N: usize> ActivationAxis for HologramF32VectorActivation<N> {
    const AXIS_ADDRESS: &'static str =
        "https://hologram.uor.foundation/axis/ActivationAxis/HologramF32Vector";
    const MAX_OUTPUT_BYTES: usize = 4 * N;

    fn relu(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation> {
        if N == 0 || N > HOLOGRAM_MAX_ACTIVATION_LEN {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/nInRange",
            ));
        }
        let n_bytes = 4 * N;
        if input.len() != n_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/inputByteLength",
            ));
        }
        if out.len() < n_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/outputByteLength",
            ));
        }
        for i in 0..N {
            let v = read_f32(input, i);
            write_f32(out, i, if v > 0.0 { v } else { 0.0 });
        }
        Ok(n_bytes)
    }

    fn sigmoid_q(input: &[u8], out: &mut [u8]) -> Result<usize, ShapeViolation> {
        if N == 0 || N > HOLOGRAM_MAX_ACTIVATION_LEN {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/nInRange",
            ));
        }
        let n_bytes = 4 * N;
        if input.len() != n_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/inputByteLength",
            ));
        }
        if out.len() < n_bytes {
            return Err(shape_violation(
                "https://hologram.uor.foundation/axis/HologramActivationShape/outputByteLength",
            ));
        }
        for i in 0..N {
            let x = read_f32(input, i);
            let y = 1.0 / (1.0 + libm::expf(-x));
            write_f32(out, i, y);
        }
        Ok(n_bytes)
    }
}

axis_extension_impl_for_activation_axis!(@generic HologramF32VectorActivation<N>, [const N: usize]);

/// 16-element f32 activation — canonical small-vector reference.
pub type HologramF32VectorActivation16 = HologramF32VectorActivation<16>;
/// 64-element f32 activation.
pub type HologramF32VectorActivation64 = HologramF32VectorActivation<64>;
/// 256-element f32 activation (the `HOLOGRAM_MAX_ACTIVATION_LEN` ceiling).
pub type HologramF32VectorActivation256 = HologramF32VectorActivation<256>;
