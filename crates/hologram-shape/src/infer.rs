//! Shape inference: given a `FloatOp` and input shapes, compute the output shape.

use crate::validate::broadcast_shapes;
use crate::TensorShape;
use hologram_core::op::{FloatDType, FloatOp};
use smallvec::SmallVec;

/// Errors arising from shape inference.
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    /// The op requires a target shape that cannot be inferred from inputs alone.
    #[error("shape inference requires target shape for {op}")]
    NeedsTargetShape { op: &'static str },

    /// Input shapes are incompatible with each other or the op parameters.
    #[error("incompatible shapes for {op}: {detail}")]
    Incompatible { op: &'static str, detail: String },

    /// Shape inference is not implemented for this op.
    #[error("shape inference not supported for {op}")]
    Unsupported { op: &'static str },

    /// Not enough input shapes were provided.
    #[error("not enough inputs for {op}: need {need}, got {got}")]
    NotEnoughInputs {
        op: &'static str,
        need: usize,
        got: usize,
    },
}

/// Infer the output `TensorShape` for `op` given input shapes.
///
/// Returns `Err` if the op requires information not available from input
/// shapes alone (e.g. `Reshape` needs a target shape) or if the op is
/// not yet supported.
pub fn infer_output_shape(
    op: &FloatOp,
    inputs: &[&TensorShape],
) -> Result<TensorShape, ShapeError> {
    // Helper to check arity.
    let need = |n: usize| -> Result<(), ShapeError> {
        if inputs.len() < n {
            Err(ShapeError::NotEnoughInputs {
                op: op.name(),
                need: n,
                got: inputs.len(),
            })
        } else {
            Ok(())
        }
    };

    match op {
        // ── Unary elementwise: output = input[0] ──────────────────────
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
        | FloatOp::Clip { .. } => {
            need(1)?;
            Ok(inputs[0].clone())
        }

        // ── Boolean/comparison unary: preserve shape, dtype = U8 ──────
        FloatOp::IsNaN | FloatOp::Not => {
            need(1)?;
            Ok(TensorShape::new(FloatDType::U8, &inputs[0].dims))
        }

        // ── Binary elementwise: broadcast ─────────────────────────────
        FloatOp::Add
        | FloatOp::Sub
        | FloatOp::Mul
        | FloatOp::Div
        | FloatOp::Pow
        | FloatOp::Mod
        | FloatOp::Min
        | FloatOp::Max => {
            need(2)?;
            let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[0].dtype,
            })
        }

        // ── Binary boolean ops: broadcast, dtype = U8 ─────────────────
        FloatOp::And | FloatOp::Or | FloatOp::Xor => {
            need(2)?;
            let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
            Ok(TensorShape {
                dims: out_dims,
                dtype: FloatDType::U8,
            })
        }

        // ── Binary comparisons: broadcast, dtype = U8 ─────────────────
        FloatOp::Equal
        | FloatOp::Less
        | FloatOp::LessOrEqual
        | FloatOp::Greater
        | FloatOp::GreaterOrEqual => {
            need(2)?;
            let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
            Ok(TensorShape {
                dims: out_dims,
                dtype: FloatDType::U8,
            })
        }

        // ── MatMul ────────────────────────────────────────────────────
        FloatOp::MatMul { m, k: _, n } => {
            need(2)?;
            let a = inputs[0];
            let b = inputs[1];
            // Resolve M: if baked m=0 (variable-length), use input[0]'s
            // second-to-last dim.
            let actual_m = if *m == 0 {
                if a.ndim() >= 2 {
                    a.dims[a.ndim() - 2]
                } else if a.ndim() == 1 {
                    // [K] x [K, N] -> [N] (vector-matrix)
                    1
                } else {
                    return Err(ShapeError::Incompatible {
                        op: op.name(),
                        detail: "cannot resolve M from scalar input".into(),
                    });
                }
            } else {
                *m as usize
            };
            let actual_n = *n as usize;

            // Batch dims: all dims except last 2 from the larger input.
            let a_batch = if a.ndim() > 2 {
                &a.dims[..a.ndim() - 2]
            } else {
                &[]
            };
            let b_batch = if b.ndim() > 2 {
                &b.dims[..b.ndim() - 2]
            } else {
                &[]
            };
            let batch = if a_batch.len() >= b_batch.len() {
                a_batch
            } else {
                b_batch
            };

            let mut out_dims: SmallVec<[usize; 4]> = SmallVec::from_slice(batch);
            out_dims.push(actual_m);
            out_dims.push(actual_n);
            Ok(TensorShape {
                dims: out_dims,
                dtype: FloatDType::F32,
            })
        }

        // ── Gemm ─────────────────────────────────────────────────────
        FloatOp::Gemm { m, n, .. } => {
            need(2)?;
            let a = inputs[0];
            let actual_m = if *m == 0 {
                if a.ndim() >= 2 {
                    a.dims[a.ndim() - 2]
                } else {
                    1
                }
            } else {
                *m as usize
            };
            let actual_n = *n as usize;

            // Batch dims from input[0]
            let batch = if a.ndim() > 2 {
                &a.dims[..a.ndim() - 2]
            } else {
                &[]
            };
            let mut out_dims: SmallVec<[usize; 4]> = SmallVec::from_slice(batch);
            out_dims.push(actual_m);
            out_dims.push(actual_n);
            Ok(TensorShape {
                dims: out_dims,
                dtype: FloatDType::F32,
            })
        }

        // ── Softmax / LogSoftmax: preserve shape ─────────────────────
        FloatOp::Softmax { .. } | FloatOp::LogSoftmax { .. } => {
            need(1)?;
            Ok(inputs[0].clone())
        }

        // ── Normalization: preserve input[0] shape ───────────────────
        FloatOp::RmsNorm { .. }
        | FloatOp::LayerNorm { .. }
        | FloatOp::AddRmsNorm { .. }
        | FloatOp::GroupNorm { .. }
        | FloatOp::InstanceNorm { .. } => {
            need(1)?;
            Ok(inputs[0].clone())
        }

        // ── Attention: output = Q shape ──────────────────────────────
        FloatOp::Attention { .. } => {
            need(3)?;
            Ok(inputs[0].clone())
        }

        // ── RotaryEmbedding: preserve shape ──────────────────────────
        FloatOp::RotaryEmbedding { .. } => {
            need(1)?;
            Ok(inputs[0].clone())
        }

        // ── Transpose ────────────────────────────────────────────────
        FloatOp::Transpose { perm, ndim } => {
            need(1)?;
            let nd = *ndim as usize;
            let inp = &inputs[0].dims;
            if inp.len() < nd {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: format!("input has {} dims but transpose expects {nd}", inp.len()),
                });
            }
            let mut out_dims: SmallVec<[usize; 4]> = SmallVec::with_capacity(nd);
            for (i, &p) in perm.iter().enumerate().take(nd) {
                let src = p as usize;
                if src >= inp.len() {
                    return Err(ShapeError::Incompatible {
                        op: op.name(),
                        detail: format!("perm[{i}]={src} out of range for {}-D input", inp.len()),
                    });
                }
                out_dims.push(inp[src]);
            }
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[0].dtype,
            })
        }

        // ── Reshape: cannot infer without target ─────────────────────
        FloatOp::Reshape => Err(ShapeError::NeedsTargetShape { op: op.name() }),

        // ── Slice ────────────────────────────────────────────────────
        FloatOp::Slice {
            axis_from_end,
            start,
            end,
            ..
        } => {
            need(1)?;
            let inp = &inputs[0].dims;
            let nd = inp.len();
            let afe = *axis_from_end as usize;
            if afe == 0 || afe > nd {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: format!("axis_from_end={afe} invalid for {nd}-D input"),
                });
            }
            let axis = nd - afe;
            let slice_len = (*end as usize).saturating_sub(*start as usize);
            let mut out_dims = inp.clone();
            out_dims[axis] = slice_len;
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[0].dtype,
            })
        }

        // ── Concat: concatenate along last dim ───────────────────────
        FloatOp::Concat {
            size_a,
            size_b,
            dtype,
        } => {
            need(2)?;
            let mut out_dims = inputs[0].dims.clone();
            if out_dims.is_empty() {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: "cannot concatenate scalars".into(),
                });
            }
            let last = out_dims.len() - 1;
            out_dims[last] = *size_a as usize + *size_b as usize;
            Ok(TensorShape {
                dims: out_dims,
                dtype: *dtype,
            })
        }

        // ── Gather: output = indices shape ++ possibly [dim] ─────────
        FloatOp::Gather { dim, dtype } => {
            need(2)?;
            let indices = inputs[1];
            // For 1-D indices gathering from a 2-D table, output is [indices..., dim].
            // For general case, output shape = indices shape (simple gather).
            if *dim > 0 {
                let mut out_dims = indices.dims.clone();
                out_dims.push(*dim as usize);
                Ok(TensorShape {
                    dims: out_dims,
                    dtype: *dtype,
                })
            } else {
                Ok(TensorShape::new(*dtype, &indices.dims))
            }
        }

        // ── Embed: [len] -> [len, dim] ──────────────────────────────
        FloatOp::Embed { dim, .. } => {
            need(1)?;
            let ids = inputs[0];
            let mut out_dims: SmallVec<[usize; 4]> = ids.dims.clone();
            out_dims.push(*dim as usize);
            Ok(TensorShape {
                dims: out_dims,
                dtype: FloatDType::F32,
            })
        }

        // ── Where: broadcast(input[1], input[2]) ────────────────────
        FloatOp::Where => {
            need(3)?;
            let out_dims = broadcast_shapes(&inputs[1].dims, &inputs[2].dims)?;
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[1].dtype,
            })
        }

        // ── Range: 1-D [ceil((limit - start) / delta)] ──────────────
        FloatOp::Range => {
            // Cannot compute without runtime values. Return a placeholder error.
            // The caller should provide the actual range length.
            Err(ShapeError::NeedsTargetShape { op: op.name() })
        }

        // ── Shape: output is 1-D [ndim] ─────────────────────────────
        FloatOp::Shape { dtype, start, end } => {
            need(1)?;
            let nd = inputs[0].ndim() as i64;
            let s = if *start >= 0 {
                (*start).min(nd)
            } else {
                (nd + *start).max(0)
            };
            let e = if *end == i64::MAX {
                nd
            } else if *end >= 0 {
                (*end).min(nd)
            } else {
                (nd + *end).max(0)
            };
            let out_len = (e - s).max(0) as usize;
            Ok(TensorShape::vector(*dtype, out_len))
        }

        // ── Cast: same shape, different dtype ────────────────────────
        FloatOp::Cast { to, .. } => {
            need(1)?;
            Ok(TensorShape::new(*to, &inputs[0].dims))
        }

        // ── Conv2d ───────────────────────────────────────────────────
        FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group: _,
            input_h: _,
            input_w: _,
        } => {
            need(2)?;
            let data = inputs[0];
            let weight = inputs[1];
            if data.ndim() < 4 {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: format!("data needs >= 4 dims, got {}", data.ndim()),
                });
            }
            let n = data.dims[0];
            let h_in = data.dims[2];
            let w_in = data.dims[3];
            let c_out = weight.dims[0]; // weight is [C_out, C_in/group, kH, kW]

            let h_out = (h_in + 2 * (*pad_h as usize)
                - (*dilation_h as usize) * (*kernel_h as usize - 1)
                - 1)
                / (*stride_h as usize)
                + 1;
            let w_out = (w_in + 2 * (*pad_w as usize)
                - (*dilation_w as usize) * (*kernel_w as usize - 1)
                - 1)
                / (*stride_w as usize)
                + 1;

            Ok(TensorShape::new(data.dtype, &[n, c_out, h_out, w_out]))
        }

        // ── ConvTranspose ────────────────────────────────────────────
        FloatOp::ConvTranspose {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group: _,
            output_pad_h,
            output_pad_w,
            input_h: _,
            input_w: _,
        } => {
            need(2)?;
            let data = inputs[0];
            let weight = inputs[1];
            if data.ndim() < 4 {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: format!("data needs >= 4 dims, got {}", data.ndim()),
                });
            }
            let n = data.dims[0];
            let h_in = data.dims[2];
            let w_in = data.dims[3];
            let c_out = weight.dims[1]; // weight is [C_in, C_out/group, kH, kW]

            let h_out = (h_in - 1) * (*stride_h as usize) - 2 * (*pad_h as usize)
                + (*dilation_h as usize) * (*kernel_h as usize - 1)
                + *output_pad_h as usize
                + 1;
            let w_out = (w_in - 1) * (*stride_w as usize) - 2 * (*pad_w as usize)
                + (*dilation_w as usize) * (*kernel_w as usize - 1)
                + *output_pad_w as usize
                + 1;

            Ok(TensorShape::new(data.dtype, &[n, c_out, h_out, w_out]))
        }

        // ── MaxPool2d / AvgPool2d ────────────────────────────────────
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        }
        | FloatOp::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            need(1)?;
            let data = inputs[0];
            if data.ndim() < 4 {
                return Err(ShapeError::Incompatible {
                    op: op.name(),
                    detail: format!("data needs >= 4 dims, got {}", data.ndim()),
                });
            }
            let n = data.dims[0];
            let c = data.dims[1];
            let h_in = data.dims[2];
            let w_in = data.dims[3];

            let h_out =
                (h_in + 2 * (*pad_h as usize) - *kernel_h as usize) / (*stride_h as usize) + 1;
            let w_out =
                (w_in + 2 * (*pad_w as usize) - *kernel_w as usize) / (*stride_w as usize) + 1;

            Ok(TensorShape::new(data.dtype, &[n, c, h_out, w_out]))
        }

        // ── GlobalAvgPool ────────────────────────────────────────────
        FloatOp::GlobalAvgPool { channels, .. } => {
            need(1)?;
            let n = if inputs[0].ndim() >= 1 {
                inputs[0].dims[0]
            } else {
                1
            };
            Ok(TensorShape::new(
                inputs[0].dtype,
                &[n, *channels as usize, 1, 1],
            ))
        }

        // ── Resize: cannot infer ─────────────────────────────────────
        FloatOp::Resize { .. } => Err(ShapeError::NeedsTargetShape { op: op.name() }),

        // ── Reductions: drop last dim ────────────────────────────────
        FloatOp::ReduceSum { .. }
        | FloatOp::ReduceMean { .. }
        | FloatOp::ReduceMax { .. }
        | FloatOp::ReduceMin { .. }
        | FloatOp::ReduceProd { .. } => {
            need(1)?;
            let inp = &inputs[0].dims;
            if inp.is_empty() {
                return Ok(inputs[0].clone());
            }
            let out_dims: SmallVec<[usize; 4]> = inp[..inp.len() - 1].into();
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[0].dtype,
            })
        }

        // ── Expand ───────────────────────────────────────────────────
        FloatOp::Expand {
            ndim, target_shape, ..
        } => {
            need(1)?;
            let nd = *ndim as usize;
            let out_dims: SmallVec<[usize; 4]> =
                target_shape[..nd].iter().map(|&d| d as usize).collect();
            Ok(TensorShape {
                dims: out_dims,
                dtype: inputs[0].dtype,
            })
        }

        // ── FusedSwiGLU: output = input[0] (gate) shape ─────────────
        FloatOp::FusedSwiGLU => {
            need(2)?;
            Ok(inputs[0].clone())
        }

        // ── ArgMax: drop reduced axis ────────────────────────────────
        FloatOp::ArgMax { keepdims, .. } => {
            need(1)?;
            let inp = &inputs[0].dims;
            if inp.is_empty() {
                return Ok(TensorShape::scalar(FloatDType::I64));
            }
            if *keepdims {
                let mut out_dims = inp.clone();
                let last = out_dims.len() - 1;
                out_dims[last] = 1;
                Ok(TensorShape {
                    dims: out_dims,
                    dtype: FloatDType::I64,
                })
            } else {
                let out_dims: SmallVec<[usize; 4]> = inp[..inp.len() - 1].into();
                Ok(TensorShape {
                    dims: out_dims,
                    dtype: FloatDType::I64,
                })
            }
        }

        // ── KV cache ops: tape-level, unsupported ────────────────────
        FloatOp::KvWrite { .. } | FloatOp::KvRead { .. } => {
            Err(ShapeError::Unsupported { op: op.name() })
        }

        // ── Deep decode fusions: unsupported ─────────────────────────
        FloatOp::NormProjectionGemv { .. }
        | FloatOp::AddNormProjectionGemv { .. }
        | FloatOp::SwiGluProjectionGemv { .. } => Err(ShapeError::Unsupported { op: op.name() }),

        // ── Dequantize: unsupported ──────────────────────────────────
        FloatOp::Dequantize => Err(ShapeError::Unsupported { op: op.name() }),

        // ── Other unsupported ops ────────────────────────────────────
        FloatOp::GatherND
        | FloatOp::ScatterND
        | FloatOp::TopK { .. }
        | FloatOp::CumSum { .. }
        | FloatOp::NonZero
        | FloatOp::Compress { .. }
        | FloatOp::ReverseSequence { .. }
        | FloatOp::PadOp { .. }
        | FloatOp::LRN { .. } => Err(ShapeError::Unsupported { op: op.name() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_shape(dims: &[usize]) -> TensorShape {
        TensorShape::new(FloatDType::F32, dims)
    }

    // ── Unary tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_preserves_shape() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(&FloatOp::Relu, &[&input]).expect("relu should succeed");
        assert_eq!(result, input);
    }

    #[test]
    fn test_neg_preserves_shape() {
        let input = f32_shape(&[1, 13, 2048]);
        let result = infer_output_shape(&FloatOp::Neg, &[&input]).expect("neg should succeed");
        assert_eq!(result, input);
    }

    #[test]
    fn test_sigmoid_preserves_shape() {
        let input = f32_shape(&[2, 3]);
        let result =
            infer_output_shape(&FloatOp::Sigmoid, &[&input]).expect("sigmoid should succeed");
        assert_eq!(result, input);
    }

    #[test]
    fn test_isnan_returns_u8() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(&FloatOp::IsNaN, &[&input]).expect("isnan should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
        assert_eq!(result.dtype, FloatDType::U8);
    }

    // ── Binary tests ─────────────────────────────────────────────────

    #[test]
    fn test_add_same_shape() {
        let a = f32_shape(&[4, 8]);
        let b = f32_shape(&[4, 8]);
        let result =
            infer_output_shape(&FloatOp::Add, &[&a, &b]).expect("add same shape should succeed");
        assert_eq!(result, a);
    }

    #[test]
    fn test_add_broadcast() {
        let a = f32_shape(&[4, 8]);
        let b = f32_shape(&[1, 8]);
        let result =
            infer_output_shape(&FloatOp::Add, &[&a, &b]).expect("add broadcast should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
    }

    #[test]
    fn test_add_scalar_broadcast() {
        let a = f32_shape(&[4, 8]);
        let b = f32_shape(&[]);
        let result = infer_output_shape(&FloatOp::Add, &[&a, &b])
            .expect("add scalar broadcast should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
    }

    #[test]
    fn test_mul_broadcast_different_ranks() {
        let a = f32_shape(&[2, 3, 4]);
        let b = f32_shape(&[4]);
        let result = infer_output_shape(&FloatOp::Mul, &[&a, &b])
            .expect("mul broadcast diff ranks should succeed");
        assert_eq!(result.dims.as_slice(), &[2, 3, 4]);
    }

    // ── Comparison tests ─────────────────────────────────────────────

    #[test]
    fn test_equal_returns_u8() {
        let a = f32_shape(&[4, 8]);
        let b = f32_shape(&[4, 8]);
        let result = infer_output_shape(&FloatOp::Equal, &[&a, &b]).expect("equal should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
        assert_eq!(result.dtype, FloatDType::U8);
    }

    #[test]
    fn test_less_broadcast_u8() {
        let a = f32_shape(&[4, 8]);
        let b = f32_shape(&[1, 8]);
        let result = infer_output_shape(&FloatOp::Less, &[&a, &b]).expect("less should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
        assert_eq!(result.dtype, FloatDType::U8);
    }

    // ── MatMul tests ─────────────────────────────────────────────────

    #[test]
    fn test_matmul_basic() {
        let a = f32_shape(&[4, 64]);
        let b = f32_shape(&[64, 128]);
        let result = infer_output_shape(
            &FloatOp::MatMul {
                m: 4,
                k: 64,
                n: 128,
            },
            &[&a, &b],
        )
        .expect("matmul should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 128]);
        assert_eq!(result.dtype, FloatDType::F32);
    }

    #[test]
    fn test_matmul_variable_m() {
        let a = f32_shape(&[13, 2048]);
        let b = f32_shape(&[2048, 4096]);
        let result = infer_output_shape(
            &FloatOp::MatMul {
                m: 0,
                k: 2048,
                n: 4096,
            },
            &[&a, &b],
        )
        .expect("matmul variable M should succeed");
        assert_eq!(result.dims.as_slice(), &[13, 4096]);
    }

    #[test]
    fn test_matmul_batched() {
        let a = f32_shape(&[2, 4, 64]);
        let b = f32_shape(&[2, 64, 128]);
        let result = infer_output_shape(
            &FloatOp::MatMul {
                m: 4,
                k: 64,
                n: 128,
            },
            &[&a, &b],
        )
        .expect("batched matmul should succeed");
        assert_eq!(result.dims.as_slice(), &[2, 4, 128]);
    }

    // ── Softmax tests ────────────────────────────────────────────────

    #[test]
    fn test_softmax_preserves_shape() {
        let input = f32_shape(&[4, 1024]);
        let result = infer_output_shape(&FloatOp::Softmax { size: 1024 }, &[&input])
            .expect("softmax should succeed");
        assert_eq!(result, input);
    }

    // ── Normalization tests ──────────────────────────────────────────

    #[test]
    fn test_rmsnorm_preserves_shape() {
        let input = f32_shape(&[1, 13, 2048]);
        let weight = f32_shape(&[2048]);
        let result = infer_output_shape(
            &FloatOp::RmsNorm {
                size: 2048,
                epsilon: 1e-5_f32.to_bits(),
            },
            &[&input, &weight],
        )
        .expect("rmsnorm should succeed");
        assert_eq!(result, input);
    }

    #[test]
    fn test_layernorm_preserves_shape() {
        let input = f32_shape(&[1, 13, 768]);
        let weight = f32_shape(&[768]);
        let bias = f32_shape(&[768]);
        let result = infer_output_shape(
            &FloatOp::LayerNorm {
                size: 768,
                epsilon: 1e-5_f32.to_bits(),
            },
            &[&input, &weight, &bias],
        )
        .expect("layernorm should succeed");
        assert_eq!(result, input);
    }

    // ── Attention tests ──────────────────────────────────────────────

    #[test]
    fn test_attention_output_is_q_shape() {
        let q = f32_shape(&[32, 13, 64]);
        let k = f32_shape(&[4, 13, 64]);
        let v = f32_shape(&[4, 13, 64]);
        let result = infer_output_shape(
            &FloatOp::Attention {
                head_dim: 64,
                num_q_heads: 32,
                num_kv_heads: 4,
                scale: (1.0f32 / 8.0).to_bits(),
                causal: true,
                heads_first: true,
                qk_norm: false,
                rope: false,
                rope_base: 0,
                sparse_v: true,
            },
            &[&q, &k, &v],
        )
        .expect("attention should succeed");
        assert_eq!(result, q);
    }

    // ── Transpose tests ──────────────────────────────────────────────

    #[test]
    fn test_transpose_4d() {
        let input = f32_shape(&[1, 32, 13, 64]);
        let result = infer_output_shape(
            &FloatOp::Transpose {
                perm: [0, 2, 1, 3, 0, 0, 0, 0],
                ndim: 4,
            },
            &[&input],
        )
        .expect("transpose should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 13, 32, 64]);
    }

    // ── Slice tests ──────────────────────────────────────────────────

    #[test]
    fn test_slice_axis_1() {
        let input = f32_shape(&[1, 13, 2048]);
        // axis_from_end=2 means second-to-last dim (axis=1 for 3-D)
        let result = infer_output_shape(
            &FloatOp::Slice {
                axis_from_end: 2,
                start: 5,
                end: 10,
                axis_size: 13,
            },
            &[&input],
        )
        .expect("slice should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 5, 2048]);
    }

    #[test]
    fn test_slice_last_axis() {
        let input = f32_shape(&[4, 2048]);
        let result = infer_output_shape(
            &FloatOp::Slice {
                axis_from_end: 1,
                start: 0,
                end: 512,
                axis_size: 2048,
            },
            &[&input],
        )
        .expect("slice last axis should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 512]);
    }

    // ── Concat tests ─────────────────────────────────────────────────

    #[test]
    fn test_concat_last_dim() {
        let a = f32_shape(&[4, 3]);
        let b = f32_shape(&[4, 5]);
        let result = infer_output_shape(
            &FloatOp::Concat {
                size_a: 3,
                size_b: 5,
                dtype: FloatDType::F32,
            },
            &[&a, &b],
        )
        .expect("concat should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
    }

    // ── Gather tests ─────────────────────────────────────────────────

    #[test]
    fn test_gather_with_dim() {
        let table = f32_shape(&[50257, 768]);
        let indices = TensorShape::new(FloatDType::I64, &[5]);
        let result = infer_output_shape(
            &FloatOp::Gather {
                dim: 768,
                dtype: FloatDType::F32,
            },
            &[&table, &indices],
        )
        .expect("gather should succeed");
        assert_eq!(result.dims.as_slice(), &[5, 768]);
        assert_eq!(result.dtype, FloatDType::F32);
    }

    // ── Embed tests ──────────────────────────────────────────────────

    #[test]
    fn test_embed() {
        let ids = TensorShape::new(FloatDType::I32, &[13]);
        let table = f32_shape(&[50257, 2048]);
        let result = infer_output_shape(
            &FloatOp::Embed {
                dim: 2048,
                quant: 0,
            },
            &[&ids, &table],
        )
        .expect("embed should succeed");
        assert_eq!(result.dims.as_slice(), &[13, 2048]);
        assert_eq!(result.dtype, FloatDType::F32);
    }

    // ── Cast tests ───────────────────────────────────────────────────

    #[test]
    fn test_cast_preserves_dims_changes_dtype() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(
            &FloatOp::Cast {
                from: FloatDType::F32,
                to: FloatDType::F16,
            },
            &[&input],
        )
        .expect("cast should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
        assert_eq!(result.dtype, FloatDType::F16);
    }

    // ── Reduce tests ─────────────────────────────────────────────────

    #[test]
    fn test_reduce_sum_drops_last() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(&FloatOp::ReduceSum { size: 8 }, &[&input])
            .expect("reduce_sum should succeed");
        assert_eq!(result.dims.as_slice(), &[4]);
    }

    #[test]
    fn test_reduce_mean_3d() {
        let input = f32_shape(&[2, 3, 4]);
        let result = infer_output_shape(&FloatOp::ReduceMean { size: 4 }, &[&input])
            .expect("reduce_mean should succeed");
        assert_eq!(result.dims.as_slice(), &[2, 3]);
    }

    // ── Expand tests ─────────────────────────────────────────────────

    #[test]
    fn test_expand() {
        let input = f32_shape(&[1, 1, 13, 64]);
        let result = infer_output_shape(
            &FloatOp::Expand {
                ndim: 4,
                target_shape: [1, 32, 13, 64, 0, 0, 0, 0],
            },
            &[&input],
        )
        .expect("expand should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 32, 13, 64]);
    }

    // ── Conv2d tests ─────────────────────────────────────────────────

    #[test]
    fn test_conv2d_with_padding() {
        let data = f32_shape(&[1, 3, 64, 64]);
        let weight = f32_shape(&[64, 3, 3, 3]);
        let bias = f32_shape(&[64]);
        let result = infer_output_shape(
            &FloatOp::Conv2d {
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
                dilation_h: 1,
                dilation_w: 1,
                group: 1,
                input_h: 64,
                input_w: 64,
            },
            &[&data, &weight, &bias],
        )
        .expect("conv2d should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 64, 64, 64]);
    }

    #[test]
    fn test_conv2d_stride_2() {
        let data = f32_shape(&[1, 64, 32, 32]);
        let weight = f32_shape(&[128, 64, 3, 3]);
        let bias = f32_shape(&[128]);
        let result = infer_output_shape(
            &FloatOp::Conv2d {
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 2,
                stride_w: 2,
                pad_h: 1,
                pad_w: 1,
                dilation_h: 1,
                dilation_w: 1,
                group: 1,
                input_h: 32,
                input_w: 32,
            },
            &[&data, &weight, &bias],
        )
        .expect("conv2d stride 2 should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 128, 16, 16]);
    }

    // ── GlobalAvgPool tests ──────────────────────────────────────────

    #[test]
    fn test_global_avg_pool() {
        let data = f32_shape(&[1, 512, 7, 7]);
        let result = infer_output_shape(
            &FloatOp::GlobalAvgPool {
                channels: 512,
                spatial_h: 7,
                spatial_w: 7,
            },
            &[&data],
        )
        .expect("global avg pool should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 512, 1, 1]);
    }

    // ── Where tests ──────────────────────────────────────────────────

    #[test]
    fn test_where_broadcast() {
        let cond = TensorShape::new(FloatDType::U8, &[4, 1]);
        let x = f32_shape(&[4, 8]);
        let y = f32_shape(&[1, 8]);
        let result =
            infer_output_shape(&FloatOp::Where, &[&cond, &x, &y]).expect("where should succeed");
        assert_eq!(result.dims.as_slice(), &[4, 8]);
        assert_eq!(result.dtype, FloatDType::F32);
    }

    // ── Shape tests ──────────────────────────────────────────────────

    #[test]
    fn test_shape_op() {
        let input = f32_shape(&[1, 13, 2048]);
        let result = infer_output_shape(
            &FloatOp::Shape {
                dtype: FloatDType::I64,
                start: 0,
                end: i64::MAX,
            },
            &[&input],
        )
        .expect("shape should succeed");
        assert_eq!(result.dims.as_slice(), &[3]);
        assert_eq!(result.dtype, FloatDType::I64);
    }

    #[test]
    fn test_shape_op_sliced() {
        let input = f32_shape(&[1, 13, 2048]);
        let result = infer_output_shape(
            &FloatOp::Shape {
                dtype: FloatDType::I64,
                start: 1,
                end: 3,
            },
            &[&input],
        )
        .expect("shape sliced should succeed");
        assert_eq!(result.dims.as_slice(), &[2]);
    }

    // ── FusedSwiGLU tests ────────────────────────────────────────────

    #[test]
    fn test_fused_swiglu() {
        let gate = f32_shape(&[1, 13, 5632]);
        let up = f32_shape(&[1, 13, 5632]);
        let result = infer_output_shape(&FloatOp::FusedSwiGLU, &[&gate, &up])
            .expect("swiglu should succeed");
        assert_eq!(result, gate);
    }

    // ── Error tests ──────────────────────────────────────────────────

    #[test]
    fn test_reshape_needs_target() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(&FloatOp::Reshape, &[&input]);
        assert!(result.is_err());
        let err = result.expect_err("reshape should fail");
        assert!(
            matches!(err, ShapeError::NeedsTargetShape { .. }),
            "expected NeedsTargetShape, got {err:?}"
        );
    }

    #[test]
    fn test_not_enough_inputs() {
        let result = infer_output_shape(&FloatOp::Add, &[]);
        assert!(result.is_err());
        let err = result.expect_err("should fail with no inputs");
        assert!(
            matches!(
                err,
                ShapeError::NotEnoughInputs {
                    need: 2,
                    got: 0,
                    ..
                }
            ),
            "expected NotEnoughInputs, got {err:?}"
        );
    }

    #[test]
    fn test_kv_write_unsupported() {
        let input = f32_shape(&[4, 13, 64]);
        let result = infer_output_shape(
            &FloatOp::KvWrite {
                layer: 0,
                n_kv_heads: 4,
                head_dim: 64,
                is_key: true,
                heads_first: true,
            },
            &[&input],
        );
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("kv_write should be unsupported"),
            ShapeError::Unsupported { .. }
        ));
    }

    // ── RotaryEmbedding tests ────────────────────────────────────────

    #[test]
    fn test_rope_preserves_shape() {
        let input = f32_shape(&[32, 13, 64]);
        let cos_sin = f32_shape(&[13, 64]);
        let result = infer_output_shape(
            &FloatOp::RotaryEmbedding {
                dim: 64,
                base: 10000_f32.to_bits(),
                n_heads: 32,
            },
            &[&input, &cos_sin],
        )
        .expect("rope should succeed");
        assert_eq!(result, input);
    }

    // ── MaxPool2d tests ──────────────────────────────────────────────

    #[test]
    fn test_maxpool2d() {
        let data = f32_shape(&[1, 64, 32, 32]);
        let result = infer_output_shape(
            &FloatOp::MaxPool2d {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
                pad_h: 0,
                pad_w: 0,
            },
            &[&data],
        )
        .expect("maxpool should succeed");
        assert_eq!(result.dims.as_slice(), &[1, 64, 16, 16]);
    }

    // ── Incompatible broadcast test ──────────────────────────────────

    #[test]
    fn test_incompatible_broadcast() {
        let a = f32_shape(&[3, 4]);
        let b = f32_shape(&[5, 4]);
        let result = infer_output_shape(&FloatOp::Add, &[&a, &b]);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should be incompatible"),
            ShapeError::Incompatible { .. }
        ));
    }

    // ── ArgMax tests ─────────────────────────────────────────────────

    #[test]
    fn test_argmax_no_keepdims() {
        let input = f32_shape(&[4, 8]);
        let result = infer_output_shape(
            &FloatOp::ArgMax {
                axis: 1,
                keepdims: false,
            },
            &[&input],
        )
        .expect("argmax should succeed");
        assert_eq!(result.dims.as_slice(), &[4]);
        assert_eq!(result.dtype, FloatDType::I64);
    }
}
