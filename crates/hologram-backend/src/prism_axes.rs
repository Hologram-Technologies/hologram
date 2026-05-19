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
use prism_tensor::{
    axis_extension_impl_for_activation_axis,
    axis_extension_impl_for_tensor_axis,
};

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
    f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]])
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
