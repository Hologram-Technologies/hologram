//! Post-fusion shape projection: compute output shape from input shapes + op params.
//!
//! Used by `emit_stage()` after fusion to build a complete shape map covering
//! 100% of tape nodes. Unlike the pre-fusion `ShapeContextGraph` in hologram-ai,
//! this operates on the post-fusion `GraphOp` variants directly — no projection
//! chains, no missing nodes from fusion.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use super::float_op::FloatOp;

/// Compute the output shape of a `FloatOp` given its input shapes.
///
/// Returns `None` for ops whose output shape can't be determined
/// statically (data-dependent ops like NonZero, TopK).
#[must_use]
pub fn float_output_shape(op: &FloatOp, input_shapes: &[&[usize]]) -> Option<Vec<usize>> {
    match op {
        // ── Unary element-preserving ───────────────────────────────────────
        FloatOp::Neg
        | FloatOp::Relu
        | FloatOp::Gelu
        | FloatOp::Silu
        | FloatOp::Tanh
        | FloatOp::Sigmoid
        | FloatOp::Exp
        | FloatOp::Log
        | FloatOp::Sqrt
        | FloatOp::Abs
        | FloatOp::Reciprocal
        | FloatOp::Cos
        | FloatOp::Sin
        | FloatOp::Sign
        | FloatOp::Floor
        | FloatOp::Ceil
        | FloatOp::Round
        | FloatOp::Erf
        | FloatOp::Clip { .. }
        | FloatOp::IsNaN
        | FloatOp::Not
        | FloatOp::Dequantize
        | FloatOp::Cast { .. } => input_shapes.first().map(|s| s.to_vec()),

        // ── Binary broadcast ──────────────────────────────────────────────
        FloatOp::Add
        | FloatOp::Sub
        | FloatOp::Mul
        | FloatOp::Div
        | FloatOp::Pow
        | FloatOp::Mod
        | FloatOp::Min
        | FloatOp::Max
        | FloatOp::And
        | FloatOp::Or
        | FloatOp::Xor
        | FloatOp::Equal
        | FloatOp::Less
        | FloatOp::LessOrEqual
        | FloatOp::Greater
        | FloatOp::GreaterOrEqual
        | FloatOp::Where => broadcast_shapes(input_shapes.first()?, input_shapes.get(1)?),

        // ── Norms, softmax, activations (shape-preserving) ────────────────
        FloatOp::Softmax { .. }
        | FloatOp::LogSoftmax { .. }
        | FloatOp::RmsNorm { .. }
        | FloatOp::AddRmsNorm { .. }
        | FloatOp::LayerNorm { .. }
        | FloatOp::InstanceNorm { .. }
        | FloatOp::GroupNorm { .. }
        | FloatOp::FusedSwiGLU
        | FloatOp::RotaryEmbedding { .. } => input_shapes.first().map(|s| s.to_vec()),

        // ── MatMul / Gemm ─────────────────────────────────────────────────
        FloatOp::MatMul { m, k: _, n } | FloatOp::Gemm { m, k: _, n, .. } => {
            Some(vec![*m as usize, *n as usize])
        }

        // ── Reshape (metadata-only) ───────────────────────────────────────
        // Output shape comes from graph metadata, not from op params.
        // The shape is seeded by node_shapes in the compiled graph.
        FloatOp::Reshape => None,

        // ── Transpose ─────────────────────────────────────────────────────
        FloatOp::Transpose { perm, ndim } => {
            let input = input_shapes.first()?;
            let n = *ndim as usize;
            if input.len() < n {
                return None;
            }
            Some((0..n).map(|i| input[perm[i] as usize]).collect())
        }

        // ── Reductions ────────────────────────────────────────────────────
        FloatOp::ReduceSum { .. }
        | FloatOp::ReduceMean { .. }
        | FloatOp::ReduceMax { .. }
        | FloatOp::ReduceMin { .. }
        | FloatOp::ReduceProd { .. } => {
            let input = input_shapes.first()?;
            if input.len() <= 1 {
                return Some(vec![1]);
            }
            Some(input[..input.len() - 1].to_vec())
        }

        // ── Gather ────────────────────────────────────────────────────────
        // Output shape depends on indices shape — can't determine without it.
        FloatOp::Gather { .. } | FloatOp::GatherND => None,

        // ── Concat ────────────────────────────────────────────────────────
        FloatOp::Concat { size_a, size_b, .. } => {
            let a = input_shapes.first()?;
            if a.is_empty() {
                return None;
            }
            let mut out = a.to_vec();
            if let Some(last) = out.last_mut() {
                *last = *size_a as usize + *size_b as usize;
            }
            Some(out)
        }

        // ── Slice ─────────────────────────────────────────────────────────
        FloatOp::Slice { start, end, .. } => {
            let input = input_shapes.first()?;
            let mut out = input.to_vec();
            let slice_len = (*end as usize).saturating_sub(*start as usize);
            if let Some(last) = out.last_mut() {
                *last = slice_len;
            }
            Some(out)
        }

        // ── Embed ─────────────────────────────────────────────────────────
        FloatOp::Embed { dim, .. } => {
            let indices = input_shapes.first()?;
            let len: usize = indices.iter().product();
            Some(vec![len, *dim as usize])
        }

        // ── Expand ────────────────────────────────────────────────────────
        FloatOp::Expand { ndim, target_shape } => {
            let n = *ndim as usize;
            Some(target_shape[..n].iter().map(|&d| d as usize).collect())
        }

        // ── Attention ─────────────────────────────────────────────────────
        // Output shape = Q input shape (same seq, same head structure).
        FloatOp::Attention { .. } => input_shapes.first().map(|s| s.to_vec()),

        // ── KV cache ops ──────────────────────────────────────────────────
        FloatOp::KvWrite { .. } | FloatOp::KvRead { .. } => {
            input_shapes.first().map(|s| s.to_vec())
        }

        // ── Fused norm+projection ops ─────────────────────────────────────
        FloatOp::NormProjectionGemv { n_total, .. }
        | FloatOp::AddNormProjectionGemv { n_total, .. } => {
            // Output shape: [batch*seq, n_total] (projected)
            let input = input_shapes.first()?;
            let m: usize = input.iter().take(input.len().saturating_sub(1)).product();
            Some(vec![m, *n_total as usize])
        }
        FloatOp::SwiGluProjectionGemv { k: _, n } => {
            let input = input_shapes.first()?;
            let m: usize = input.iter().take(input.len().saturating_sub(1)).product();
            Some(vec![m, *n as usize])
        }

        // ── Conv2d ────────────────────────────────────────────────────────
        FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            input_h,
            input_w,
            ..
        } => {
            let input = input_shapes.first()?;
            let weight = input_shapes.get(1)?;
            if input.len() < 4 || weight.is_empty() {
                return None;
            }
            let n = input[0];
            let c_out = weight[0];
            let ih = *input_h as usize;
            let iw = *input_w as usize;
            let kh = *kernel_h as usize;
            let kw = *kernel_w as usize;
            let sh = *stride_h as usize;
            let sw = *stride_w as usize;
            let ph = *pad_h as usize;
            let pw = *pad_w as usize;
            let dh = *dilation_h as usize;
            let dw = *dilation_w as usize;
            let h_out = (ih + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
            let w_out = (iw + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
            Some(vec![n, c_out, h_out, w_out])
        }

        // ── ConvTranspose ─────────────────────────────────────────────────
        FloatOp::ConvTranspose {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            input_h,
            input_w,
            ..
        } => {
            let input = input_shapes.first()?;
            let weight = input_shapes.get(1)?;
            if input.len() < 4 || weight.is_empty() {
                return None;
            }
            let n = input[0];
            let c_out = weight[1]; // ConvTranspose: weight shape [C_in, C_out, kH, kW]
            let ih = *input_h as usize;
            let iw = *input_w as usize;
            let sh = *stride_h as usize;
            let sw = *stride_w as usize;
            let ph = *pad_h as usize;
            let pw = *pad_w as usize;
            let oph = *output_pad_h as usize;
            let opw = *output_pad_w as usize;
            let kh = *kernel_h as usize;
            let kw = *kernel_w as usize;
            let h_out = (ih - 1) * sh - 2 * ph + kh + oph;
            let w_out = (iw - 1) * sw - 2 * pw + kw + opw;
            Some(vec![n, c_out, h_out, w_out])
        }

        // ── Pooling ───────────────────────────────────────────────────────
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            ..
        }
        | FloatOp::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            ..
        } => {
            let input = input_shapes.first()?;
            if input.len() < 4 {
                return None;
            }
            let n = input[0];
            let c = input[1];
            let ih = input[2];
            let iw = input[3];
            let h_out = (ih + 2 * *pad_h as usize - *kernel_h as usize) / *stride_h as usize + 1;
            let w_out = (iw + 2 * *pad_w as usize - *kernel_w as usize) / *stride_w as usize + 1;
            Some(vec![n, c, h_out, w_out])
        }

        // ── Global pooling ────────────────────────────────────────────────
        FloatOp::GlobalAvgPool { .. } => {
            let input = input_shapes.first()?;
            if input.len() < 2 {
                return None;
            }
            Some(vec![input[0], input[1], 1, 1])
        }

        // ── Shape, Range ──────────────────────────────────────────────────
        FloatOp::Shape { start, end, .. } => {
            let input = input_shapes.first()?;
            let s = *start as usize;
            let e = if *end == 0 {
                input.len()
            } else {
                *end as usize
            };
            Some(vec![e.saturating_sub(s)])
        }
        FloatOp::Range => None, // data-dependent

        // ── Resize, Pad ───────────────────────────────────────────────────
        FloatOp::Resize { .. } | FloatOp::PadOp { .. } => None,

        // ── ArgMax ────────────────────────────────────────────────────────
        FloatOp::ArgMax { keepdims, .. } => {
            let input = input_shapes.first()?;
            if *keepdims {
                let mut out = input.to_vec();
                if let Some(last) = out.last_mut() {
                    *last = 1;
                }
                Some(out)
            } else {
                if input.len() <= 1 {
                    return Some(vec![1]);
                }
                Some(input[..input.len() - 1].to_vec())
            }
        }

        // ── LRN ───────────────────────────────────────────────────────────
        FloatOp::LRN { .. } => input_shapes.first().map(|s| s.to_vec()),

        // ── Data-dependent / complex ops ──────────────────────────────────
        FloatOp::TopK { .. }
        | FloatOp::ScatterND
        | FloatOp::CumSum { .. }
        | FloatOp::NonZero
        | FloatOp::Compress { .. }
        | FloatOp::ReverseSequence { .. } => None,
    }
}

/// Broadcast two shapes following NumPy/ONNX rules.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < max_ndim - a.len() {
            1
        } else {
            a[i - (max_ndim - a.len())]
        };
        let db = if i < max_ndim - b.len() {
            1
        } else {
            b[i - (max_ndim - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return None; // incompatible
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_preserves_shape() {
        let shape = [1, 24, 2048];
        let out = float_output_shape(&FloatOp::Relu, &[&shape]);
        assert_eq!(out, Some(vec![1, 24, 2048]));
    }

    #[test]
    fn binary_broadcast() {
        let a = [1, 24, 2048];
        let b = [2048];
        let out = float_output_shape(&FloatOp::Add, &[&a, &b]);
        assert_eq!(out, Some(vec![1, 24, 2048]));
    }

    #[test]
    fn matmul_shape() {
        let out = float_output_shape(
            &FloatOp::MatMul {
                m: 24,
                k: 2048,
                n: 2048,
            },
            &[&[24, 2048], &[2048, 2048]],
        );
        assert_eq!(out, Some(vec![24, 2048]));
    }

    #[test]
    fn reduce_drops_last_dim() {
        let out = float_output_shape(&FloatOp::ReduceSum { size: 2048 }, &[&[1, 24, 2048]]);
        assert_eq!(out, Some(vec![1, 24]));
    }

    #[test]
    fn transpose_permutes() {
        let out = float_output_shape(
            &FloatOp::Transpose {
                perm: [0, 2, 1, 3, 0, 0, 0, 0],
                ndim: 4,
            },
            &[&[1, 24, 32, 64]],
        );
        assert_eq!(out, Some(vec![1, 32, 24, 64]));
    }

    #[test]
    fn softmax_preserves() {
        let out = float_output_shape(&FloatOp::Softmax { size: 24 }, &[&[32, 24]]);
        assert_eq!(out, Some(vec![32, 24]));
    }

    #[test]
    fn broadcast_scalar() {
        let out = broadcast_shapes(&[1], &[3, 4]);
        assert_eq!(out, Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_incompatible() {
        let out = broadcast_shapes(&[3], &[4]);
        assert_eq!(out, None);
    }
}
