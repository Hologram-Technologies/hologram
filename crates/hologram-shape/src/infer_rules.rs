//! Per-op shape inference rules.
//!
//! Each category of `FloatOp` has its own inference function. The main
//! dispatcher in `infer.rs` delegates here after arity checking.

use crate::infer::ShapeError;
use crate::validate::broadcast_shapes;
use crate::TensorShape;
use hologram_core::op::{FloatDType, FloatOp};
use smallvec::SmallVec;

/// Unary elementwise ops that preserve input shape and dtype.
pub(crate) fn infer_unary_elementwise(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    Ok(inputs[0].clone())
}

/// Boolean/comparison unary ops: preserve shape, dtype = U8.
pub(crate) fn infer_unary_boolean(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    Ok(TensorShape::new(FloatDType::U8, &inputs[0].dims))
}

/// Binary elementwise ops: broadcast, preserve dtype from input[0].
pub(crate) fn infer_binary_elementwise(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
    Ok(TensorShape {
        dims: out_dims,
        dtype: inputs[0].dtype,
    })
}

/// Binary boolean ops: broadcast, dtype = U8.
pub(crate) fn infer_binary_boolean(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
    Ok(TensorShape {
        dims: out_dims,
        dtype: FloatDType::U8,
    })
}

/// Binary comparisons: broadcast, dtype = U8.
pub(crate) fn infer_binary_comparison(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    let out_dims = broadcast_shapes(&inputs[0].dims, &inputs[1].dims)?;
    Ok(TensorShape {
        dims: out_dims,
        dtype: FloatDType::U8,
    })
}

/// MatMul shape inference.
pub(crate) fn infer_matmul(
    op: &FloatOp,
    inputs: &[&TensorShape],
    m: u32,
    _k: u32,
    n: u32,
) -> Result<TensorShape, ShapeError> {
    let a = inputs[0];
    let b = inputs[1];
    // Resolve M: if baked m=0 (variable-length), use input[0]'s
    // second-to-last dim.
    let actual_m = if m == 0 {
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
        m as usize
    };
    let actual_n = n as usize;

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

/// Gemm shape inference.
pub(crate) fn infer_gemm(
    inputs: &[&TensorShape],
    m: u32,
    n: u32,
) -> Result<TensorShape, ShapeError> {
    let a = inputs[0];
    let actual_m = if m == 0 {
        if a.ndim() >= 2 {
            a.dims[a.ndim() - 2]
        } else {
            1
        }
    } else {
        m as usize
    };
    let actual_n = n as usize;

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

/// Transpose shape inference.
pub(crate) fn infer_transpose(
    op: &FloatOp,
    inputs: &[&TensorShape],
    perm: &[u8; 8],
    ndim: u8,
) -> Result<TensorShape, ShapeError> {
    let nd = ndim as usize;
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

/// Slice shape inference.
pub(crate) fn infer_slice(
    op: &FloatOp,
    inputs: &[&TensorShape],
    axis_from_end: u8,
    start: u32,
    end: u32,
) -> Result<TensorShape, ShapeError> {
    let inp = &inputs[0].dims;
    let nd = inp.len();
    let afe = axis_from_end as usize;
    if afe == 0 || afe > nd {
        return Err(ShapeError::Incompatible {
            op: op.name(),
            detail: format!("axis_from_end={afe} invalid for {nd}-D input"),
        });
    }
    let axis = nd - afe;
    let slice_len = (end as usize).saturating_sub(start as usize);
    let mut out_dims = inp.clone();
    out_dims[axis] = slice_len;
    Ok(TensorShape {
        dims: out_dims,
        dtype: inputs[0].dtype,
    })
}

/// Concat shape inference.
pub(crate) fn infer_concat(
    op: &FloatOp,
    inputs: &[&TensorShape],
    size_a: u32,
    size_b: u32,
    dtype: FloatDType,
) -> Result<TensorShape, ShapeError> {
    let mut out_dims = inputs[0].dims.clone();
    if out_dims.is_empty() {
        return Err(ShapeError::Incompatible {
            op: op.name(),
            detail: "cannot concatenate scalars".into(),
        });
    }
    let last = out_dims.len() - 1;
    out_dims[last] = size_a as usize + size_b as usize;
    Ok(TensorShape {
        dims: out_dims,
        dtype,
    })
}

/// Gather shape inference.
pub(crate) fn infer_gather(
    inputs: &[&TensorShape],
    dim: u32,
    dtype: FloatDType,
) -> Result<TensorShape, ShapeError> {
    let indices = inputs[1];
    // For 1-D indices gathering from a 2-D table, output is [indices..., dim].
    // For general case, output shape = indices shape (simple gather).
    if dim > 0 {
        let mut out_dims = indices.dims.clone();
        out_dims.push(dim as usize);
        Ok(TensorShape {
            dims: out_dims,
            dtype,
        })
    } else {
        Ok(TensorShape::new(dtype, &indices.dims))
    }
}

/// Embed shape inference: [len] -> [len, dim].
pub(crate) fn infer_embed(inputs: &[&TensorShape], dim: u32) -> Result<TensorShape, ShapeError> {
    let ids = inputs[0];
    let mut out_dims: SmallVec<[usize; 4]> = ids.dims.clone();
    out_dims.push(dim as usize);
    Ok(TensorShape {
        dims: out_dims,
        dtype: FloatDType::F32,
    })
}

/// Where shape inference: broadcast(input[1], input[2]).
pub(crate) fn infer_where(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
    let out_dims = broadcast_shapes(&inputs[1].dims, &inputs[2].dims)?;
    Ok(TensorShape {
        dims: out_dims,
        dtype: inputs[1].dtype,
    })
}

/// Shape op inference: output is 1-D [ndim].
pub(crate) fn infer_shape_op(
    inputs: &[&TensorShape],
    dtype: FloatDType,
    start: i64,
    end: i64,
) -> Result<TensorShape, ShapeError> {
    let nd = inputs[0].ndim() as i64;
    let s = if start >= 0 {
        start.min(nd)
    } else {
        (nd + start).max(0)
    };
    let e = if end == i64::MAX {
        nd
    } else if end >= 0 {
        end.min(nd)
    } else {
        (nd + end).max(0)
    };
    let out_len = (e - s).max(0) as usize;
    Ok(TensorShape::vector(dtype, out_len))
}

/// Cast shape inference: same shape, different dtype.
pub(crate) fn infer_cast(
    inputs: &[&TensorShape],
    to: FloatDType,
) -> Result<TensorShape, ShapeError> {
    Ok(TensorShape::new(to, &inputs[0].dims))
}

/// Spatial parameters for Conv2d / ConvTranspose / Pool2d inference.
pub(crate) struct SpatialParams {
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub output_pad_h: u32,
    pub output_pad_w: u32,
}

/// Conv2d shape inference.
pub(crate) fn infer_conv2d(
    op: &FloatOp,
    inputs: &[&TensorShape],
    p: &SpatialParams,
) -> Result<TensorShape, ShapeError> {
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

    let h_out =
        (h_in + 2 * (p.pad_h as usize) - (p.dilation_h as usize) * (p.kernel_h as usize - 1) - 1)
            / (p.stride_h as usize)
            + 1;
    let w_out =
        (w_in + 2 * (p.pad_w as usize) - (p.dilation_w as usize) * (p.kernel_w as usize - 1) - 1)
            / (p.stride_w as usize)
            + 1;

    Ok(TensorShape::new(data.dtype, &[n, c_out, h_out, w_out]))
}

/// ConvTranspose shape inference.
pub(crate) fn infer_conv_transpose(
    op: &FloatOp,
    inputs: &[&TensorShape],
    p: &SpatialParams,
) -> Result<TensorShape, ShapeError> {
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

    let h_out = (h_in - 1) * (p.stride_h as usize) - 2 * (p.pad_h as usize)
        + (p.dilation_h as usize) * (p.kernel_h as usize - 1)
        + p.output_pad_h as usize
        + 1;
    let w_out = (w_in - 1) * (p.stride_w as usize) - 2 * (p.pad_w as usize)
        + (p.dilation_w as usize) * (p.kernel_w as usize - 1)
        + p.output_pad_w as usize
        + 1;

    Ok(TensorShape::new(data.dtype, &[n, c_out, h_out, w_out]))
}

/// MaxPool2d / AvgPool2d shape inference.
pub(crate) fn infer_pool2d(
    op: &FloatOp,
    inputs: &[&TensorShape],
    p: &SpatialParams,
) -> Result<TensorShape, ShapeError> {
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

    let h_out = (h_in + 2 * (p.pad_h as usize) - p.kernel_h as usize) / (p.stride_h as usize) + 1;
    let w_out = (w_in + 2 * (p.pad_w as usize) - p.kernel_w as usize) / (p.stride_w as usize) + 1;

    Ok(TensorShape::new(data.dtype, &[n, c, h_out, w_out]))
}

/// GlobalAvgPool shape inference.
pub(crate) fn infer_global_avg_pool(
    inputs: &[&TensorShape],
    channels: u32,
) -> Result<TensorShape, ShapeError> {
    let n = if inputs[0].ndim() >= 1 {
        inputs[0].dims[0]
    } else {
        1
    };
    Ok(TensorShape::new(
        inputs[0].dtype,
        &[n, channels as usize, 1, 1],
    ))
}

/// Reduction ops: drop last dim.
pub(crate) fn infer_reduce(inputs: &[&TensorShape]) -> Result<TensorShape, ShapeError> {
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

/// Expand shape inference.
pub(crate) fn infer_expand(
    inputs: &[&TensorShape],
    ndim: u8,
    target_shape: &[u32; 8],
) -> Result<TensorShape, ShapeError> {
    let nd = ndim as usize;
    let out_dims: SmallVec<[usize; 4]> = target_shape[..nd].iter().map(|&d| d as usize).collect();
    Ok(TensorShape {
        dims: out_dims,
        dtype: inputs[0].dtype,
    })
}

/// ArgMax shape inference: drop reduced axis.
pub(crate) fn infer_argmax(
    inputs: &[&TensorShape],
    keepdims: bool,
) -> Result<TensorShape, ShapeError> {
    let inp = &inputs[0].dims;
    if inp.is_empty() {
        return Ok(TensorShape::scalar(FloatDType::I64));
    }
    if keepdims {
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
