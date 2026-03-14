//! Float-domain kernel dispatch for `FloatOp` graph operations.
//!
//! All kernels operate on `&[u8]` inputs interpreted as `&[f32]` via bytemuck,
//! matching the pattern used by `MatMulLut4`/`MatMulLut8`.

use hologram_core::op::{bits_to_f32, FloatDType, FloatOp, OpCategory};

use crate::error::{ExecError, ExecResult};

/// Parameters for a GEMM (General Matrix Multiply) operation:
/// `C = alpha * op(A) * op(B) + beta * C`
#[derive(Debug, Clone, Copy)]
pub struct GemmParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

/// Dispatch a `FloatOp` with the given byte-buffer inputs.
///
/// Category-based dispatch: generic kernel patterns (unary, binary, compare,
/// byte-bool) are handled by `OpCategory`, while ops needing dedicated logic
/// are dispatched individually via `dispatch_custom`.
pub fn dispatch_float(op: &FloatOp, inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    match op.category() {
        OpCategory::UnaryElementwise => unary_map(inputs, |v| op.apply_unary(v)),
        OpCategory::BinaryElementwise => binary_elementwise(inputs, |a, b| op.apply_binary(a, b)),
        OpCategory::BinaryCompare => binary_compare(inputs, |a, b| op.apply_compare(a, b)),
        OpCategory::BinaryByteBool => binary_byte_bool(inputs, |a, b| op.apply_byte_bool(a, b)),
        OpCategory::UnaryByteBool => unary_byte_bool(inputs, |a| if a != 0 { 0 } else { 1 }),
        OpCategory::UnaryToU8 => dispatch_isnan(inputs),
        OpCategory::Custom => dispatch_custom(op, inputs),
    }
}

/// Dispatch ops that need dedicated kernel logic.
fn dispatch_custom(op: &FloatOp, inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    match op {
        FloatOp::MatMul { m, k, n } => {
            dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize)
        }
        FloatOp::Gemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => dispatch_gemm(
            inputs,
            GemmParams {
                m: *m as usize,
                n: *n as usize,
                k: *k as usize,
                alpha: bits_to_f32(*alpha),
                beta: bits_to_f32(*beta),
                trans_a: *trans_a,
                trans_b: *trans_b,
            },
            *quant_b,
        ),
        FloatOp::Softmax { size } => dispatch_softmax(inputs, *size as usize),
        FloatOp::LogSoftmax { size } => dispatch_log_softmax(inputs, *size as usize),
        FloatOp::RmsNorm { size, epsilon } => {
            dispatch_rms_norm(inputs, *size as usize, bits_to_f32(*epsilon))
        }
        FloatOp::LayerNorm { size, epsilon } => {
            dispatch_layer_norm(inputs, *size as usize, bits_to_f32(*epsilon))
        }
        FloatOp::ReduceSum { size } => dispatch_reduce(inputs, *size as usize, reduce_sum),
        FloatOp::ReduceMean { size } => dispatch_reduce(inputs, *size as usize, reduce_mean),
        FloatOp::ReduceMax { size } => dispatch_reduce(inputs, *size as usize, reduce_max),
        FloatOp::ReduceMin { size } => dispatch_reduce(inputs, *size as usize, reduce_min),
        FloatOp::Gather { dim, dtype } => dispatch_gather(inputs, *dim as usize, *dtype),
        FloatOp::Concat {
            size_a,
            size_b,
            dtype,
        } => dispatch_concat(inputs, *size_a as usize, *size_b as usize, *dtype),
        FloatOp::Reshape | FloatOp::Transpose { .. } | FloatOp::GatherND => Ok(inputs[0].to_vec()),
        FloatOp::Cast { from, to } => dispatch_cast(inputs, *from, *to),
        FloatOp::Embed { dim, quant } => dispatch_embed(inputs, *dim as usize, *quant),
        FloatOp::Where => dispatch_where(inputs),
        FloatOp::Range => dispatch_range(inputs),
        FloatOp::Shape { dtype, start, end } => dispatch_shape(inputs, *dtype, *start, *end),
        FloatOp::RotaryEmbedding { dim, base, n_heads } => {
            dispatch_rope(inputs, *dim as usize, bits_to_f32(*base), *n_heads as usize)
        }
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
        } => dispatch_attention(
            inputs,
            *head_dim as usize,
            *num_q_heads as usize,
            *num_kv_heads as usize,
            bits_to_f32(*scale),
            *causal,
        ),
        FloatOp::Dequantize => dispatch_dequantize(inputs),
        // ── Vision / spatial ops ──────────────────────────────────────────
        FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
        } => dispatch_conv2d(
            inputs,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
            *dilation_h as usize,
            *dilation_w as usize,
            *group as usize,
        ),
        FloatOp::ConvTranspose {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            output_pad_h,
            output_pad_w,
        } => dispatch_conv_transpose(
            inputs,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
            *dilation_h as usize,
            *dilation_w as usize,
            *group as usize,
            *output_pad_h as usize,
            *output_pad_w as usize,
        ),
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => dispatch_max_pool_2d(
            inputs,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
        ),
        FloatOp::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => dispatch_avg_pool_2d(
            inputs,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
        ),
        FloatOp::GlobalAvgPool => dispatch_global_avg_pool(inputs),
        FloatOp::Resize { mode } => dispatch_resize(inputs, *mode),
        FloatOp::PadOp { mode } => dispatch_pad(inputs, *mode),
        FloatOp::InstanceNorm { size, epsilon } => {
            dispatch_instance_norm(inputs, *size as usize, bits_to_f32(*epsilon))
        }
        FloatOp::LRN {
            size,
            alpha,
            beta,
            bias,
        } => dispatch_lrn(
            inputs,
            *size as usize,
            bits_to_f32(*alpha),
            bits_to_f32(*beta),
            bits_to_f32(*bias),
        ),
        // ── Utility ops ─────────────────────────────────────────────────
        FloatOp::ReduceProd { size } => dispatch_reduce(inputs, *size as usize, reduce_prod),
        FloatOp::TopK { axis, largest } => dispatch_top_k(inputs, *axis as usize, *largest),
        FloatOp::ScatterND => dispatch_scatter_nd(inputs),
        FloatOp::CumSum { axis } => dispatch_cumsum(inputs, *axis as usize),
        FloatOp::NonZero => dispatch_nonzero(inputs),
        FloatOp::Compress { axis } => dispatch_compress(inputs, *axis as usize),
        FloatOp::ReverseSequence {
            batch_axis,
            time_axis,
        } => dispatch_reverse_sequence(inputs, *batch_axis as usize, *time_axis as usize),
        _ => unreachable!("non-custom op {:?} routed to dispatch_custom", op),
    }
}

/// Dispatch a fused chain of unary element-wise f32 ops.
///
/// Applies each op in sequence to every element, avoiding intermediate buffers.
pub fn dispatch_fused_chain(chain: &[FloatOp], inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<f32> = x
        .iter()
        .map(|&v| {
            let mut val = v;
            for op in chain {
                val = op.apply_unary(val);
            }
            val
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn cast_f32(bytes: &[u8]) -> ExecResult<std::borrow::Cow<'_, [f32]>> {
    match bytemuck::try_cast_slice(bytes) {
        Ok(s) => Ok(std::borrow::Cow::Borrowed(s)),
        Err(_) => Ok(std::borrow::Cow::Owned(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )),
    }
}

/// Iterator over i64 values read from potentially-misaligned bytes.
fn iter_i64(bytes: &[u8]) -> impl Iterator<Item = i64> + '_ {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
}

/// Read a single i64 at element index `idx` from potentially-misaligned bytes.
fn read_i64_at(bytes: &[u8], idx: usize) -> Option<i64> {
    let off = idx * 8;
    bytes
        .get(off..off + 8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
}

/// Iterator over i32 values read from potentially-misaligned bytes.
fn iter_i32(bytes: &[u8]) -> impl Iterator<Item = i32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
}

/// Read a single i32 at element index `idx` from potentially-misaligned bytes.
fn read_i32_at(bytes: &[u8], idx: usize) -> Option<i32> {
    let off = idx * 4;
    bytes
        .get(off..off + 4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
}

/// Zero-copy conversion from `Vec<f32>` to `Vec<u8>`.
///
/// Takes ownership and reinterprets the backing allocation in-place —
/// no memcpy, no extra allocation.
pub(crate) fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let len = data.len() * 4;
    let cap = data.capacity() * 4;
    let ptr = data.as_ptr() as *mut u8;
    std::mem::forget(data);
    // SAFETY: f32 has alignment >= u8; len/cap are scaled correctly.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// ── Unary ────────────────────────────────────────────────────────────────────

fn unary_map(inputs: &[&[u8]], f: impl Fn(f32) -> f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<f32> = x.iter().map(|&v| f(v)).collect();
    Ok(f32_vec_to_bytes(out))
}

/// Dispatch a `FloatOp` with shape information for proper N-D broadcasting.
///
/// For binary elementwise ops, uses `input_shapes` to perform numpy-style
/// broadcasting instead of cycling. Falls back to `dispatch_float` for
/// non-binary ops or when shapes are unavailable.
pub fn dispatch_float_with_shapes(
    op: &FloatOp,
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
) -> ExecResult<Vec<u8>> {
    match op.category() {
        OpCategory::BinaryElementwise if input_shapes.len() >= 2 => {
            binary_elementwise_broadcast(inputs, input_shapes, |a, b| op.apply_binary(a, b))
        }
        OpCategory::BinaryCompare if input_shapes.len() >= 2 => {
            binary_compare_broadcast(inputs, input_shapes, |a, b| op.apply_compare(a, b))
        }
        _ => dispatch_float(op, inputs),
    }
}

// ── Binary ───────────────────────────────────────────────────────────────────

fn binary_elementwise(inputs: &[&[u8]], f: impl Fn(f32, f32) -> f32) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<f32> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(f32_vec_to_bytes(out))
}

/// Binary elementwise with proper N-D broadcasting using input shapes.
///
/// Follows numpy broadcasting rules: dimensions are compared right-to-left,
/// and each dimension must be either equal or 1. A dimension of 1 is broadcast
/// (repeated) to match the other operand's dimension.
fn binary_elementwise_broadcast(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f32, f32) -> f32,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let sa = &input_shapes[0];
    let sb = &input_shapes[1];

    // Fast path: same shape or one is scalar — cycling is correct.
    if sa == sb || a.len() == 1 || b.len() == 1 {
        let out_len = a.len().max(b.len());
        let out: Vec<f32> = (0..out_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Validate shapes match data sizes.
    let a_prod: usize = sa.iter().product();
    let b_prod: usize = sb.iter().product();
    if a_prod != a.len() || b_prod != b.len() {
        // Shape doesn't match data — fall back to cycling.
        let out_len = a.len().max(b.len());
        let out: Vec<f32> = (0..out_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Compute broadcast output shape. Fall back to cycling if shapes
    // are not broadcast-compatible (e.g. non-f32 dtype mismatch).
    let out_shape = match broadcast_shapes(sa, sb) {
        Some(s) => s,
        None => {
            let out_len = a.len().max(b.len());
            let out: Vec<f32> = (0..out_len)
                .map(|i| f(a[i % a.len()], b[i % b.len()]))
                .collect();
            return Ok(f32_vec_to_bytes(out));
        }
    };
    let out_len: usize = out_shape.iter().product();

    // If broadcast would inflate output beyond both input sizes, the compiled
    // input_shapes are stale (0-sentinels resolved to wrong values creating
    // orthogonal broadcast dimensions at runtime). Fall back to cycling.
    if out_len > a.len().max(b.len()) {
        let safe_len = a.len().max(b.len());
        let out: Vec<f32> = (0..safe_len)
            .map(|i| f(a[i % a.len()], b[i % b.len()]))
            .collect();
        return Ok(f32_vec_to_bytes(out));
    }

    // Compute strides for index mapping.
    let a_strides = compute_broadcast_strides(sa, &out_shape);
    let b_strides = compute_broadcast_strides(sb, &out_shape);
    let out_strides = compute_strides(&out_shape);

    let out: Vec<f32> = (0..out_len)
        .map(|flat_idx| {
            let a_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &a_strides);
            let b_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &b_strides);
            f(a[a_idx], b[b_idx])
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

/// Binary compare with proper N-D broadcasting.
fn binary_compare_broadcast(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f32, f32) -> bool,
) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let sa = &input_shapes[0];
    let sb = &input_shapes[1];

    if sa == sb || a.len() == 1 || b.len() == 1 {
        let out_len = a.len().max(b.len());
        let out: Vec<u8> = (0..out_len)
            .map(|i| {
                if f(a[i % a.len()], b[i % b.len()]) {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        return Ok(out);
    }

    let a_prod: usize = sa.iter().product();
    let b_prod: usize = sb.iter().product();
    if a_prod != a.len() || b_prod != b.len() {
        let out_len = a.len().max(b.len());
        let out: Vec<u8> = (0..out_len)
            .map(|i| {
                if f(a[i % a.len()], b[i % b.len()]) {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        return Ok(out);
    }

    let out_shape = match broadcast_shapes(sa, sb) {
        Some(s) => s,
        None => {
            let out_len = a.len().max(b.len());
            let out: Vec<u8> = (0..out_len)
                .map(|i| {
                    if f(a[i % a.len()], b[i % b.len()]) {
                        1u8
                    } else {
                        0u8
                    }
                })
                .collect();
            return Ok(out);
        }
    };
    let out_len: usize = out_shape.iter().product();

    // Same stale-shape guard as binary_elementwise_broadcast.
    if out_len > a.len().max(b.len()) {
        let safe_len = a.len().max(b.len());
        let out: Vec<u8> = (0..safe_len)
            .map(|i| {
                if f(a[i % a.len()], b[i % b.len()]) {
                    1u8
                } else {
                    0u8
                }
            })
            .collect();
        return Ok(out);
    }

    let a_strides = compute_broadcast_strides(sa, &out_shape);
    let b_strides = compute_broadcast_strides(sb, &out_shape);
    let out_strides = compute_strides(&out_shape);

    let out: Vec<u8> = (0..out_len)
        .map(|flat_idx| {
            let a_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &a_strides);
            let b_idx = broadcast_flat_index(flat_idx, &out_shape, &out_strides, &b_strides);
            if f(a[a_idx], b[b_idx]) {
                1u8
            } else {
                0u8
            }
        })
        .collect();
    Ok(out)
}

/// Compute numpy-style broadcast output shape.
/// Returns `None` if shapes are not broadcast-compatible (dimensions must be
/// equal or one of them must be 1).
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            return None; // Not broadcast-compatible
        }
        result.push(da.max(db));
    }
    Some(result)
}

/// Compute strides for a shape (row-major).
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute broadcast strides: for dimensions where `src` has size 1 (broadcast),
/// the stride is 0 (same element repeated). Otherwise, uses normal strides.
fn compute_broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let src_strides = compute_strides(src_shape);
    let offset = out_shape.len() - src_shape.len();
    let mut strides = vec![0usize; out_shape.len()];
    for i in 0..src_shape.len() {
        if src_shape[i] != 1 {
            strides[i + offset] = src_strides[i];
        }
        // else: stride stays 0 (broadcast dimension)
    }
    strides
}

/// Convert a flat output index to a flat source index using broadcast strides.
#[inline]
fn broadcast_flat_index(
    flat_idx: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    src_strides: &[usize],
) -> usize {
    let mut src_idx = 0;
    let mut remaining = flat_idx;
    for i in 0..out_shape.len() {
        let coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        src_idx += coord * src_strides[i];
    }
    src_idx
}

// ── MatMul ───────────────────────────────────────────────────────────────────

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod blas {
    use super::GemmParams;

    #[allow(non_camel_case_types)]
    type cblas_int = i32;

    const CBLAS_ROW_MAJOR: cblas_int = 101;
    const CBLAS_NO_TRANS: cblas_int = 111;
    const CBLAS_TRANS: cblas_int = 112;

    extern "C" {
        fn cblas_sgemm(
            order: cblas_int,
            trans_a: cblas_int,
            trans_b: cblas_int,
            m: cblas_int,
            n: cblas_int,
            k: cblas_int,
            alpha: f32,
            a: *const f32,
            lda: cblas_int,
            b: *const f32,
            ldb: cblas_int,
            beta: f32,
            c: *mut f32,
            ldc: cblas_int,
        );
    }

    /// BLAS sgemm: C = A * B (row-major, no transpose).
    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        sgemm_full(
            GemmParams {
                m,
                n,
                k,
                alpha: 1.0,
                beta: 0.0,
                trans_a: false,
                trans_b: false,
            },
            a,
            b,
            c,
        );
    }

    /// BLAS sgemm: C = alpha * op(A) * op(B) + beta * C (row-major).
    pub fn sgemm_full(p: GemmParams, a: &[f32], b: &[f32], c: &mut [f32]) {
        let ta = if p.trans_a {
            CBLAS_TRANS
        } else {
            CBLAS_NO_TRANS
        };
        let tb = if p.trans_b {
            CBLAS_TRANS
        } else {
            CBLAS_NO_TRANS
        };
        let lda = if p.trans_a {
            p.m as cblas_int
        } else {
            p.k as cblas_int
        };
        let ldb = if p.trans_b {
            p.k as cblas_int
        } else {
            p.n as cblas_int
        };
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                ta,
                tb,
                p.m as cblas_int,
                p.n as cblas_int,
                p.k as cblas_int,
                p.alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                p.beta,
                c.as_mut_ptr(),
                p.n as cblas_int,
            );
        }
    }
}

/// Dispatch a MatMul using runtime-aware shape inference.
///
/// The compiled `k` (inner dimension) is used as a hint. When it cannot
/// cleanly divide both inputs, we attempt to infer k from the compiled `n`
/// and `m` hints, or from common factors between the two buffer lengths.
pub fn dispatch_matmul(inputs: &[&[u8]], m: usize, k: usize, n: usize) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;

    let actual_k = infer_matmul_k(k, m, n, a.len(), b.len())?;
    let actual_m = a.len() / actual_k;
    let actual_n = b.len() / actual_k;

    // Cap output to prevent OOM from shape inference errors.
    let out_size = actual_m.saturating_mul(actual_n);
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("matmul output < 1GB (compiled k={k})"),
            actual: format!(
                "[{actual_m},{actual_k}]x[{actual_k},{actual_n}] = {} floats",
                out_size
            ),
        });
    }

    let mut out = vec![0.0f32; out_size];

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        blas::sgemm(actual_m, actual_n, actual_k, &a, &b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        // ikj loop order: inner j loop is stride-1 on both out[] and b[],
        // enabling auto-vectorization and cache-friendly access.
        for i in 0..actual_m {
            for p in 0..actual_k {
                let a_val = a[i * actual_k + p];
                let b_row = &b[p * actual_n..(p + 1) * actual_n];
                let o_row = &mut out[i * actual_n..(i + 1) * actual_n];
                for j in 0..actual_n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

/// Batched matmul: A[batch, M, K] × B[batch, K, N] → C[batch, M, N].
///
/// Each batch independently computes a 2D matrix multiply. This is required
/// for multi-head attention where Q@K^T operates per-head.
///
/// Returns `(output_bytes, output_shape)`.
pub fn dispatch_batched_matmul(
    inputs: &[&[u8]],
    a_shape: &[usize],
    b_shape: &[usize],
) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;

    // Last 2 dims are the matrix dims; everything before is batch.
    let mat_m = a_shape[a_shape.len() - 2];
    let mat_k = a_shape[a_shape.len() - 1];
    let mat_n = b_shape[b_shape.len() - 1];

    let batch: usize = a_shape[..a_shape.len() - 2]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);

    let a_stride = mat_m * mat_k;
    let b_stride = mat_k * mat_n;
    let c_stride = mat_m * mat_n;

    // Support broadcast: 2-D B (shared weight) reuses the same matrix for
    // every batch slice. b_batch_count=1 means b_off stays at 0 each iteration.
    let b_batch_count = if b_stride > 0 {
        (b.len() / b_stride).max(1)
    } else {
        1
    };

    // Validate sizes.
    if batch * a_stride > a.len() || b_stride > b.len() {
        return Err(ExecError::ShapeMismatch {
            expected: format!(
                "batched matmul: batch={batch} A=[{mat_m},{mat_k}] B=[{mat_k},{mat_n}]"
            ),
            actual: format!("a_len={}, b_len={}", a.len(), b.len()),
        });
    }

    let out_size = batch * c_stride;
    if out_size > 256 * 1024 * 1024 {
        return Err(ExecError::ShapeMismatch {
            expected: "batched matmul output < 1GB".to_string(),
            actual: format!("{out_size} floats"),
        });
    }

    let mut out = vec![0.0f32; out_size];

    for bat in 0..batch {
        let a_off = bat * a_stride;
        let b_off = (bat % b_batch_count) * b_stride;
        let c_off = bat * c_stride;
        let a_slice = &a[a_off..a_off + a_stride];
        let b_slice = &b[b_off..b_off + b_stride];
        let c_slice = &mut out[c_off..c_off + c_stride];

        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(mat_m, mat_n, mat_k, a_slice, b_slice, c_slice);
        }

        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            for i in 0..mat_m {
                for p in 0..mat_k {
                    let a_val = a_slice[i * mat_k + p];
                    let b_row = &b_slice[p * mat_n..(p + 1) * mat_n];
                    let o_row = &mut c_slice[i * mat_n..(i + 1) * mat_n];
                    for j in 0..mat_n {
                        o_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    }

    // Output shape: A's batch dims + [M, N]
    let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
    out_shape.push(mat_n);

    Ok((f32_vec_to_bytes(out), out_shape))
}

/// Infer the shared inner dimension `k` for MatMul A[M,K] × B[K,N].
///
/// Uses compiled k/m/n as hints. When compiled k is wrong (doesn't divide
/// both inputs), tries to infer k from compiled n (B's last dim, typically
/// concrete for weight matrices) or from common factors.
fn infer_matmul_k(
    compiled_k: usize,
    compiled_m: usize,
    compiled_n: usize,
    a_len: usize,
    b_len: usize,
) -> ExecResult<usize> {
    // Primary: compiled k divides both inputs cleanly.
    if compiled_k > 1 && a_len.is_multiple_of(compiled_k) && b_len.is_multiple_of(compiled_k) {
        return Ok(compiled_k);
    }
    // Fallback 1: if compiled n is known, infer k = b_len / n.
    if compiled_n > 1 && b_len.is_multiple_of(compiled_n) {
        let k = b_len / compiled_n;
        if k > 0 && a_len.is_multiple_of(k) {
            return Ok(k);
        }
    }
    // Fallback 2: if compiled m is known, infer k = a_len / m.
    if compiled_m > 1 && a_len.is_multiple_of(compiled_m) {
        let k = a_len / compiled_m;
        if k > 0 && b_len.is_multiple_of(k) {
            return Ok(k);
        }
    }
    // Fallback 3: when compiled_n is known, try k = b_len / compiled_n even
    // if a_len % k != 0 — this handles batched MatMul where A is [batch, m, k].
    // Also try using compiled_n as k directly (common for square weight matrices).
    if compiled_n > 1 {
        // Try compiled_n as k (e.g. weight is [2048, 2048], both k and n are 2048)
        if a_len.is_multiple_of(compiled_n) && b_len.is_multiple_of(compiled_n) {
            return Ok(compiled_n);
        }
    }

    // Fallback 4: use GCD of a_len and b_len as k (finds the largest shared dim).
    let g = gcd(a_len, b_len);
    if g > 1 {
        // Verify output won't be absurdly large (m * n < 256M floats)
        let m_cand = a_len / g;
        let n_cand = b_len / g;
        if m_cand.saturating_mul(n_cand) < 256 * 1024 * 1024 {
            return Ok(g);
        }
        // GCD is too large, try smaller common factors by dividing GCD
        // Use compiled_n hint if available
        if compiled_n > 1 && g.is_multiple_of(compiled_n) {
            let k = g / compiled_n * compiled_n; // round down to compiled_n multiple
            if k > 1 && a_len.is_multiple_of(k) && b_len.is_multiple_of(k) {
                return Ok(k);
            }
        }
    }

    // Last resort: compiled_k works if it divides both (even k=1 for scalar matmul).
    if compiled_k > 0 && a_len.is_multiple_of(compiled_k) && b_len.is_multiple_of(compiled_k) {
        // Guard against k=1 producing impossibly large outputs.
        let m_cand = a_len / compiled_k;
        let n_cand = b_len / compiled_k;
        if m_cand.saturating_mul(n_cand) < 256 * 1024 * 1024 {
            return Ok(compiled_k);
        }
    }
    Err(ExecError::ShapeMismatch {
        expected: format!("matmul k dividing both inputs (compiled k={compiled_k}, m={compiled_m}, n={compiled_n})"),
        actual: format!("a={a_len}, b={b_len}"),
    })
}

// ── Softmax ──────────────────────────────────────────────────────────────────

fn dispatch_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }

    let mut out = x.to_vec();
    let uniform = 1.0f32 / size as f32;
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max == f32::INFINITY {
            // Overflow: some scores are +inf (padding positions with overflowed Q@K).
            // Only +inf positions get non-zero weight; finite positions get 0.
            // This is the limit of softmax as the max diverges to infinity.
            let inf_count = row.iter().filter(|&&v| v == f32::INFINITY).count();
            let w = if inf_count > 0 {
                1.0f32 / inf_count as f32
            } else {
                uniform
            };
            for v in row.iter_mut() {
                *v = if *v == f32::INFINITY { w } else { 0.0 };
            }
            continue;
        }
        if !max.is_finite() {
            // All-masked (-inf or NaN): uniform output to prevent NaN propagation.
            for v in row.iter_mut() {
                *v = uniform;
            }
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        } else {
            for v in row.iter_mut() {
                *v = uniform;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── RmsNorm ──────────────────────────────────────────────────────────────────

fn dispatch_rms_norm(inputs: &[&[u8]], size: usize, epsilon: f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if weight.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight: [{size}]"),
            actual: format!("{} floats", weight.len()),
        });
    }
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
        let rms = (ms + epsilon).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = (*v / rms) * w;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

fn dispatch_layer_norm(inputs: &[&[u8]], size: usize, epsilon: f32) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;
    if weight.len() != size || bias.len() != size {
        return Err(ExecError::ShapeMismatch {
            expected: format!("weight/bias: [{size}]"),
            actual: format!("weight={}, bias={}", weight.len(), bias.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let mean: f32 = row.iter().sum::<f32>() / size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / size as f32;
        let std = (var + epsilon).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v - mean) / std) * weight[i] + bias[i];
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Reductions ───────────────────────────────────────────────────────────────

fn dispatch_reduce(
    inputs: &[&[u8]],
    size: usize,
    f: impl Fn(&[f32]) -> f32,
) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let out: Vec<f32> = x.chunks(size).map(f).collect();
    Ok(f32_vec_to_bytes(out))
}

fn reduce_sum(row: &[f32]) -> f32 {
    row.iter().sum()
}

fn reduce_mean(row: &[f32]) -> f32 {
    row.iter().sum::<f32>() / row.len() as f32
}

fn reduce_max(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn reduce_min(row: &[f32]) -> f32 {
    row.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn reduce_prod(row: &[f32]) -> f32 {
    row.iter().product()
}

// ── Gather ───────────────────────────────────────────────────────────────────

fn dispatch_gather(inputs: &[&[u8]], dim: usize, dtype: FloatDType) -> ExecResult<Vec<u8>> {
    let index_bytes = inputs[0];
    let n_indices = index_bytes.len() / 8;
    let table_bytes = inputs[1];

    match dtype {
        FloatDType::I64 => {
            // i64 gather (shape subgraph): indices select individual i64 values.
            let n_table = table_bytes.len() / 8;
            let mut out = Vec::with_capacity(n_indices * 8);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= n_table {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i64 index < {n_table}"),
                        actual: format!("index = {idx}"),
                    });
                }
                let val = read_i64_at(table_bytes, idx).unwrap();
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        FloatDType::I32 => {
            // i32 gather: indices select individual i32 values.
            let n_table = table_bytes.len() / 4;
            let mut out = Vec::with_capacity(n_indices * 4);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= n_table {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("i32 index < {n_table}"),
                        actual: format!("index = {idx}"),
                    });
                }
                let val = read_i32_at(table_bytes, idx).unwrap();
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            // f32 embedding gather (default for F32, F16, BF16, etc.).
            let table = cast_f32(table_bytes)?;
            let dim = if dim > 0 { dim } else { 1 };
            let vocab = table.len() / dim;
            let mut out = Vec::with_capacity(n_indices * dim * 4);
            for idx in iter_i64(index_bytes).map(|v| v as usize) {
                if idx >= vocab {
                    return Err(ExecError::ShapeMismatch {
                        expected: format!("index < {vocab}"),
                        actual: format!("index = {idx}"),
                    });
                }
                out.extend_from_slice(bytemuck::cast_slice(&table[idx * dim..(idx + 1) * dim]));
            }
            Ok(out)
        }
    }
}

// ── Concat ───────────────────────────────────────────────────────────────────

/// Check if raw bytes look like valid f32 data vs i64 byte reinterpretation.
///
/// On little-endian, an i64 value < 2^31 (all shape dimensions) has its high
/// 4 bytes as 0x00000000. When reinterpreted as two f32 values, the second
/// (high word) is always 0.0. Real f32 shape data (after Cast) contains
fn dispatch_concat(
    inputs: &[&[u8]],
    size_a: usize,
    size_b: usize,
    dtype: FloatDType,
) -> ExecResult<Vec<u8>> {
    let a_bytes = inputs[0];
    let b_bytes = inputs[1];

    let elem_size = dtype.byte_size();

    // For non-f32 types (I64, I32, etc.), use byte-level operations.
    // This prevents i64 data from being split at 4-byte f32 boundaries.
    if !matches!(dtype, FloatDType::F32) {
        if size_a <= 1 && size_b <= 1 {
            // axis=0 concat: simple byte append.
            let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
            out.extend_from_slice(a_bytes);
            out.extend_from_slice(b_bytes);
            return Ok(out);
        }
        // Interleave at element granularity (not f32 granularity).
        let row_bytes_a = size_a * elem_size;
        let row_bytes_b = size_b * elem_size;
        if row_bytes_a > 0 && row_bytes_b > 0 {
            let rows_a = a_bytes.len() / row_bytes_a;
            let rows_b = b_bytes.len() / row_bytes_b;
            if rows_a == rows_b && rows_a > 0 {
                let mut out = Vec::with_capacity(rows_a * (row_bytes_a + row_bytes_b));
                for i in 0..rows_a {
                    out.extend_from_slice(&a_bytes[i * row_bytes_a..(i + 1) * row_bytes_a]);
                    out.extend_from_slice(&b_bytes[i * row_bytes_b..(i + 1) * row_bytes_b]);
                }
                return Ok(out);
            }
        }
        // Fallback: simple append.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        return Ok(out);
    }

    // F32 path (original behavior).
    if size_a <= 1 && size_b <= 1 {
        // axis=0 concat: simple byte append.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        return Ok(out);
    }

    if size_a > 0 && a_bytes.len().is_multiple_of(4) && b_bytes.len().is_multiple_of(4) {
        let a = cast_f32(a_bytes)?;
        let b = cast_f32(b_bytes)?;
        let rows_a = a.len() / size_a;
        let rows_b = b.len() / size_b;
        if rows_a == rows_b && rows_a > 0 {
            // Last-axis concat: interleave rows.
            let mut out = Vec::with_capacity(rows_a * (size_a + size_b));
            for i in 0..rows_a {
                out.extend_from_slice(&a[i * size_a..(i + 1) * size_a]);
                out.extend_from_slice(&b[i * size_b..(i + 1) * size_b]);
            }
            Ok(f32_vec_to_bytes(out))
        } else {
            // Fallback: simple append (axis=0 or shape mismatch).
            let mut out = Vec::with_capacity(a.len() + b.len());
            out.extend_from_slice(&a);
            out.extend_from_slice(&b);
            Ok(f32_vec_to_bytes(out))
        }
    } else {
        // Data doesn't cleanly partition into f32 rows — raw byte concat.
        let mut out = Vec::with_capacity(a_bytes.len() + b_bytes.len());
        out.extend_from_slice(a_bytes);
        out.extend_from_slice(b_bytes);
        Ok(out)
    }
}

// ── Erf approximation (moved to FloatOp::apply_unary) ───────────────────

// ── Boolean / byte-wise ops ─────────────────────────────────────────────

/// Convert raw bytes to per-element booleans (0 or 1).
///
/// If the buffer is f32-aligned, each 4-byte f32 becomes one bool (nonzero → 1).
/// Otherwise, each byte is a boolean directly.
fn to_bools(data: &[u8]) -> Vec<u8> {
    if data.len().is_multiple_of(4) && data.len() >= 4 {
        // Try interpreting as f32 — common when upstream is a comparison or cast.
        if let Ok(floats) = bytemuck::try_cast_slice::<u8, f32>(data) {
            return floats.iter().map(|&v| (v != 0.0) as u8).collect();
        }
    }
    // Byte-level booleans.
    data.iter().map(|&v| (v != 0) as u8).collect()
}

fn binary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8, u8) -> u8) -> ExecResult<Vec<u8>> {
    let a = to_bools(inputs[0]);
    let b = to_bools(inputs[1]);
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]))
        .collect();
    Ok(out)
}

fn unary_byte_bool(inputs: &[&[u8]], f: impl Fn(u8) -> u8) -> ExecResult<Vec<u8>> {
    let bools = to_bools(inputs[0]);
    let out: Vec<u8> = bools.iter().map(|&x| f(x)).collect();
    Ok(out)
}

// ── Comparison ops (f32 → u8) ───────────────────────────────────────────

fn binary_compare(inputs: &[&[u8]], f: impl Fn(f32, f32) -> bool) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = cast_f32(inputs[1])?;
    let out_len = a.len().max(b.len());
    let out: Vec<u8> = (0..out_len)
        .map(|i| f(a[i % a.len()], b[i % b.len()]) as u8)
        .collect();
    Ok(out)
}

// ── IsNaN (f32 → u8) ───────────────────────────────────────────────────

fn dispatch_isnan(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    let out: Vec<u8> = x.iter().map(|v| v.is_nan() as u8).collect();
    Ok(out)
}

// ── Gemm ────────────────────────────────────────────────────────────────

fn dispatch_gemm(inputs: &[&[u8]], p: GemmParams, quant_b: u8) -> ExecResult<Vec<u8>> {
    let a = cast_f32(inputs[0])?;
    let b = decode_weights(inputs[1], quant_b)?;
    let c: std::borrow::Cow<'_, [f32]> = if inputs.len() > 2 {
        cast_f32(inputs[2])?
    } else {
        std::borrow::Cow::Owned(vec![])
    };
    // Derive m and n from actual inputs — compile-time values may be wrong.
    let k = p.k;
    let n = if k > 0 { b.len() / k } else { 0 };
    let m = if k > 0 { a.len() / k } else { 0 };
    let mut out = vec![0.0f32; m * n];

    // Copy bias (C) into output — BLAS computes C := alpha*A*B + beta*C in-place.
    if p.beta != 0.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                out[idx] = if idx < c.len() { c[idx] } else { 0.0 };
            }
        }
    }

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        blas::sgemm_full(GemmParams { m, n, k, ..p }, &a, &b, &mut out);
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for q in 0..k {
                    let a_val = if p.trans_a {
                        a[q * m + i]
                    } else {
                        a[i * k + q]
                    };
                    let b_val = if p.trans_b {
                        b[j * k + q]
                    } else {
                        b[q * n + j]
                    };
                    sum += a_val * b_val;
                }
                let c_val = if i * n + j < c.len() {
                    c[i * n + j]
                } else {
                    0.0
                };
                out[i * n + j] = p.alpha * sum + p.beta * c_val;
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── LogSoftmax ──────────────────────────────────────────────────────────

fn dispatch_log_softmax(inputs: &[&[u8]], size: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;
    if x.len() % size != 0 {
        return Err(ExecError::ShapeMismatch {
            expected: format!("multiple of {size}"),
            actual: format!("{} floats", x.len()),
        });
    }
    let mut out = x.to_vec();
    for row in out.chunks_mut(size) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
        for v in row.iter_mut() {
            *v -= log_sum_exp;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Embed ───────────────────────────────────────────────────────────────

/// Dequantize Q4_0 data: each 18-byte block produces 32 f32 values.
/// Format: 2-byte f16 scale + 16 bytes of nibble pairs (each nibble - 8).
fn dequantize_q4_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 32);
    for block in data.chunks(18) {
        if block.len() < 18 {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // ggml Q4_0 layout: low nibbles → positions 0..15, high nibbles → 16..31
        for byte_idx in 0..16 {
            let lo = (block[2 + byte_idx] & 0x0F) as i8 - 8;
            out.push(lo as f32 * scale);
        }
        for byte_idx in 0..16 {
            let hi = (block[2 + byte_idx] >> 4) as i8 - 8;
            out.push(hi as f32 * scale);
        }
    }
    out
}

/// Dequantize Q6_K data: 256 values per super-block (210 bytes each).
/// Layout: ql[128] + qh[64] + scales[16] + d(f16)[2] = 210 bytes.
/// Each value is a 6-bit signed integer (-32..31) scaled by (d * scale_i).
fn dequantize_q6_k(data: &[u8]) -> Vec<f32> {
    const QK: usize = 256;
    const BLOCK_SIZE: usize = QK / 2 + QK / 4 + QK / 16 + 2; // 128 + 64 + 16 + 2 = 210

    let n_blocks = data.len() / BLOCK_SIZE;
    let mut out = vec![0.0f32; n_blocks * QK];

    for (bi, block_data) in data.chunks(BLOCK_SIZE).enumerate() {
        if block_data.len() < BLOCK_SIZE {
            break;
        }
        let ql = &block_data[0..128];
        let qh = &block_data[128..192];
        let sc = &block_data[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block_data[208], block_data[209]]));
        let y = &mut out[bi * QK..];

        // Match ggml's dequantize_row_q6_K exactly:
        // Two passes of 128 values each, each pass processes 4 groups of 32.
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut y_off = 0usize;
        for n_pass in 0..2u8 {
            let is = (n_pass as usize) * 8; // scale index base
            for l in 0..32 {
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
                y[y_off + l] = d * sc[is] as i8 as f32 * q1 as f32;
                y[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            y_off += 128;
        }
    }
    out
}

/// Decode bytes as f32, applying dequantization if quant != 0.
/// quant: 0=f32, 1=Q4_0, 2=Q8_0, 3=Q6_K.
fn decode_weights(data: &[u8], quant: u8) -> ExecResult<std::borrow::Cow<'_, [f32]>> {
    match quant {
        1 => Ok(std::borrow::Cow::Owned(dequantize_q4_0(data))),
        3 => Ok(std::borrow::Cow::Owned(dequantize_q6_k(data))),
        // TODO: Q8_0 dequantization
        _ => cast_f32(data),
    }
}

fn dispatch_embed(inputs: &[&[u8]], dim: usize, quant: u8) -> ExecResult<Vec<u8>> {
    // inputs[0] = token_ids (i64 or u32), inputs[1] = table (f32 or quantized) [vocab, dim]
    let raw = inputs[0];
    let table_bytes = inputs[1];
    let table = decode_weights(table_bytes, quant)?;

    let vocab = table.len() / dim;

    // Detect token ID dtype: i64 (8 bytes each) or u32 (4 bytes each).
    let token_ids: Vec<usize> = if raw.len().is_multiple_of(8) {
        iter_i64(raw).map(|v| v as usize).collect()
    } else {
        iter_i32(raw).map(|v| v as usize).collect()
    };

    let mut out = Vec::with_capacity(token_ids.len() * dim);
    for idx in &token_ids {
        if *idx >= vocab {
            return Err(ExecError::ShapeMismatch {
                expected: format!("token id < {vocab}"),
                actual: format!("token id = {idx}"),
            });
        }
        out.extend_from_slice(&table[idx * dim..(idx + 1) * dim]);
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Where ───────────────────────────────────────────────────────────────

fn dispatch_where(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [cond (u8 or f32), x (f32), y (f32)]
    // Condition is normalized to per-element booleans via to_bools(),
    // which handles both u8 masks and f32-encoded booleans uniformly.
    let cond = to_bools(inputs[0]);
    let x = cast_f32(inputs[1])?;
    let y = cast_f32(inputs[2])?;

    let n = cond.len().max(x.len()).max(y.len());

    let out: Vec<f32> = (0..n)
        .map(|i| {
            if cond[i % cond.len()] != 0 {
                x[i % x.len()]
            } else {
                y[i % y.len()]
            }
        })
        .collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Range ───────────────────────────────────────────────────────────────

fn dispatch_range(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [start (f32), limit (f32), delta (f32)] — each is a scalar
    let start = cast_f32(inputs[0])?[0];
    let limit = cast_f32(inputs[1])?[0];
    let delta = cast_f32(inputs[2])?[0];
    let n = ((limit - start) / delta).ceil() as usize;
    let out: Vec<f32> = (0..n).map(|i| start + i as f32 * delta).collect();
    Ok(f32_vec_to_bytes(out))
}

// ── Cast ────────────────────────────────────────────────────────────────

fn dispatch_cast(inputs: &[&[u8]], from: FloatDType, to: FloatDType) -> ExecResult<Vec<u8>> {
    if from == to {
        return Ok(inputs[0].to_vec());
    }
    let data = inputs[0];
    let from_size = from.byte_size();

    // If the data doesn't divide evenly by the declared `from` dtype but
    // DOES divide evenly by the `to` dtype (or by 4 for f32), the upstream
    // already converted and this Cast is a no-op.  This handles chains of
    // Casts where dtype metadata wasn't fully propagated.
    if from_size > 0 && !data.len().is_multiple_of(from_size) {
        return Ok(data.to_vec());
    }

    match (from, to) {
        (FloatDType::I64, FloatDType::F32) => {
            let out: Vec<f32> = iter_i64(data).map(|v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::I32, FloatDType::F32) => {
            let out: Vec<f32> = iter_i32(data).map(|v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::I64) => {
            let src = cast_f32(data)?;
            let out: Vec<i64> = src.iter().map(|&v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::I32) => {
            let src = cast_f32(data)?;
            let out: Vec<i32> = src.iter().map(|&v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::Bool, FloatDType::F32) => {
            let out: Vec<f32> = data
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::Bool) => {
            let src = cast_f32(data)?;
            Ok(src
                .iter()
                .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
                .collect())
        }
        (FloatDType::I64, FloatDType::Bool) => Ok(iter_i64(data)
            .map(|v| if v != 0 { 1u8 } else { 0u8 })
            .collect()),
        (FloatDType::I64, FloatDType::I32) => {
            let out: Vec<i32> = iter_i64(data).map(|v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::I64) => {
            let out: Vec<i64> = iter_i32(data).map(|v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::Bool) => Ok(iter_i32(data)
            .map(|v| if v != 0 { 1u8 } else { 0u8 })
            .collect()),
        (FloatDType::Bool, FloatDType::I64) => {
            let out: Vec<i64> = data
                .iter()
                .map(|&v| if v != 0 { 1i64 } else { 0i64 })
                .collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::Bool, FloatDType::I32) => {
            let out: Vec<i32> = data
                .iter()
                .map(|&v| if v != 0 { 1i32 } else { 0i32 })
                .collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        // Fallback: pass-through (same bytes, different interpretation).
        _ => Ok(data.to_vec()),
    }
}

// ── Shape ───────────────────────────────────────────────────────────────

fn dispatch_shape(
    inputs: &[&[u8]],
    dtype: FloatDType,
    _start: i64,
    _end: i64,
) -> ExecResult<Vec<u8>> {
    // float_dispatch is a kernel-level path with no shape metadata, so it can
    // only infer total element count — not the per-axis dims. Return the element
    // count as a single i64. (The executor path has access to tracked shapes and
    // performs proper per-axis shape extraction with start/end slicing.)
    let elem_bytes = dtype.byte_size();
    let n_elements = if elem_bytes > 0 {
        inputs[0].len() as i64 / elem_bytes as i64
    } else {
        inputs[0].len() as i64
    };
    Ok(bytemuck::cast_slice(&[n_elements]).to_vec())
}

/// Slice a tensor's shape according to ONNX Shape opset-15 `start`/`end` attributes.
///
/// Returns an i64 buffer containing `in_shape[s..e]` where `s` and `e` are
/// clamped/normalised from `start`/`end` exactly as the ONNX spec requires:
/// - `start = i64::MAX` is treated as "end of dims" (only meaningful for end).
/// - Negative indices count from the rank end.
/// - Indices are clamped to `[0, rank]`.
///
/// Used by the executor's `FloatOp::Shape` handler. Exposed `pub` so that
/// unit tests can exercise start/end slicing without requiring a full compiled
/// graph (the AiGraph pipeline constant-folds Shape when input dims are concrete).
pub fn dispatch_shape_sliced(
    in_shape: &[usize],
    _dtype: FloatDType,
    start: i64,
    end: i64,
) -> ExecResult<Vec<u8>> {
    let rank = in_shape.len() as i64;
    let s = if start < 0 {
        (rank + start).max(0) as usize
    } else {
        (start as usize).min(in_shape.len())
    };
    let e = if end == i64::MAX {
        in_shape.len()
    } else if end < 0 {
        (rank + end).max(0) as usize
    } else {
        (end as usize).min(in_shape.len())
    };
    if s >= e {
        return Ok(vec![]);
    }
    let sliced: Vec<i64> = in_shape[s..e].iter().map(|&d| d as i64).collect();
    Ok(bytemuck::cast_slice(&sliced).to_vec())
}

// ── Attention ───────────────────────────────────────────────────────────

/// Transpose from [seq, n_heads, head_dim] to [n_heads, seq, head_dim].
fn transpose_heads(data: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for t in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim {
                out[h * seq * head_dim + t * head_dim + d] =
                    data[t * n_heads * head_dim + h * head_dim + d];
            }
        }
    }
    out
}

fn dispatch_attention(
    inputs: &[&[u8]],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    causal: bool,
) -> ExecResult<Vec<u8>> {
    // Input Q/K/V arrive as [seq, n_heads, head_dim] (interleaved heads per token).
    // Transpose to [n_heads, seq, head_dim] for per-head attention computation.
    let q_raw = cast_f32(inputs[0])?;
    let k_raw = cast_f32(inputs[1])?;
    let v_raw = cast_f32(inputs[2])?;

    let seq_q = q_raw.len() / (num_q_heads * head_dim);
    let seq_k = k_raw.len() / (num_kv_heads * head_dim);

    let q = transpose_heads(&q_raw, seq_q, num_q_heads, head_dim);
    let k = transpose_heads(&k_raw, seq_k, num_kv_heads, head_dim);
    let v = transpose_heads(&v_raw, seq_k, num_kv_heads, head_dim);
    let group_size = num_q_heads / num_kv_heads.max(1);

    let mut out = vec![0.0f32; num_q_heads * seq_q * head_dim];
    // Allocate scores buffer once, reuse across all heads.
    let mut scores = vec![0.0f32; seq_q * seq_k];

    // Per-head attention: iterate over Q heads, map to KV head via group_size.
    for qh in 0..num_q_heads {
        let kh = qh / group_size;
        let q_off = qh * seq_q * head_dim;
        let k_off = kh * seq_k * head_dim;
        let o_off = qh * seq_q * head_dim;

        let q_head = &q[q_off..q_off + seq_q * head_dim];
        let k_head = &k[k_off..k_off + seq_k * head_dim];
        let v_head = &v[k_off..k_off + seq_k * head_dim];

        // scores = Q_head × K_head^T * scale → [seq_q, seq_k]
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm_full(
                GemmParams {
                    m: seq_q,
                    n: seq_k,
                    k: head_dim,
                    alpha: scale,
                    beta: 0.0,
                    trans_a: false,
                    trans_b: true,
                },
                q_head,
                k_head,
                &mut scores,
            );
            // Apply causal mask after BLAS.
            if causal {
                for i in 0..seq_q {
                    for j in (i + 1)..seq_k {
                        scores[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            // Fused QK^T with causal mask: skip upper triangle entirely.
            for i in 0..seq_q {
                let limit = if causal { (i + 1).min(seq_k) } else { seq_k };
                for j in 0..limit {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                    }
                    scores[i * seq_k + j] = dot * scale;
                }
                // Fill masked positions with -inf.
                if causal {
                    for j in limit..seq_k {
                        scores[i * seq_k + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax each row (2-pass: max+exp+sum, then divide).
        for row in scores.chunks_mut(seq_k) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max).exp();
                sum += *val;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for val in row.iter_mut() {
                    *val *= inv;
                }
            }
        }

        // out_head = scores × V_head → [seq_q, head_dim]
        // ikj loop order for cache-friendly access to V.
        let out_head = &mut out[o_off..o_off + seq_q * head_dim];
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            blas::sgemm(seq_q, head_dim, seq_k, &scores, v_head, out_head);
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            out_head.fill(0.0);
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let s = scores[i * seq_k + j];
                    if s == 0.0 {
                        continue;
                    }
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    let o_row = &mut out_head[i * head_dim..(i + 1) * head_dim];
                    for d in 0..head_dim {
                        o_row[d] += s * v_row[d];
                    }
                }
            }
        }
    }

    // Transpose output back from [n_heads, seq, head_dim] to [seq, n_heads, head_dim]
    let mut final_out = vec![0.0f32; out.len()];
    for h in 0..num_q_heads {
        for t in 0..seq_q {
            for d in 0..head_dim {
                final_out[t * num_q_heads * head_dim + h * head_dim + d] =
                    out[h * seq_q * head_dim + t * head_dim + d];
            }
        }
    }

    Ok(f32_vec_to_bytes(final_out))
}

// ── Dequantize ──────────────────────────────────────────────────────────

fn dispatch_dequantize(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // Q4_0 dequantization: blocks of 18 bytes (2 byte scale + 16 nibbles = 32 values)
    let data = inputs[0];
    let block_size = 18;
    if !data.len().is_multiple_of(block_size) {
        // Not Q4_0 format — just pass through
        return Ok(data.to_vec());
    }
    let n_blocks = data.len() / block_size;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for block in data.chunks(block_size) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for byte_idx in 0..16 {
            let byte = block[2 + byte_idx];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    Ok(f32_vec_to_bytes(out))
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // subnormal
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

fn dispatch_rope(inputs: &[&[u8]], dim: usize, base: f32, n_heads: usize) -> ExecResult<Vec<u8>> {
    let x = cast_f32(inputs[0])?;

    // Position input: either a single u32 start offset, or absent (sequential from 0).
    let start_pos: usize = if inputs.len() >= 2 && inputs[1].len() == 4 {
        u32::from_le_bytes([inputs[1][0], inputs[1][1], inputs[1][2], inputs[1][3]]) as usize
    } else {
        0
    };

    let half = dim / 2;
    let n_heads = n_heads.max(1);
    let mut out = x.to_vec();
    // Apply RoPE to each chunk of `dim` elements. Multiple heads per token
    // share the same position: pos = chunk_index / n_heads.
    // Uses interleaved convention (ggml): pairs (0,1), (2,3), (4,5), ...
    for (chunk_idx, chunk) in out.chunks_mut(dim).enumerate() {
        let token_pos = chunk_idx / n_heads;
        let pos = (start_pos + token_pos) as f32;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x0 = chunk[2 * i];
            let x1 = chunk[2 * i + 1];
            chunk[2 * i] = x0 * cos_a - x1 * sin_a;
            chunk[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Shape-aware ops (called from executor with ShapeMap) ─────────────────────

/// Reshape: data passes through, shape is read from the shape tensor (inputs[1]).
/// Returns `(data_bytes, new_shape)`.
pub fn dispatch_reshape_with_shape(inputs: &[&[u8]]) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let data = inputs[0].to_vec();
    if inputs.len() >= 2 && !inputs[1].is_empty() {
        let n_elems = data.len() / 4; // assume f32

        let shape = crate::eval::shape_resolve::parse_shape_values(inputs[1], n_elems)
            .unwrap_or_else(|| vec![n_elems]);

        let shape_product: usize = shape.iter().product();
        if shape_product == n_elems {
            Ok((data, shape))
        } else if shape_product > n_elems && n_elems > 0 && shape_product <= n_elems * 1024 {
            // Broadcast expansion (e.g. GQA key repeat): replicate data.
            let src = cast_f32(&data)?;
            let expanded = broadcast_to(&src, n_elems, &shape);
            Ok((f32_vec_to_bytes(expanded), shape))
        } else {
            // Can't match — fall back to 1-D.
            Ok((data, vec![n_elems]))
        }
    } else {
        // No shape tensor — return 1D.
        let n = data.len() / 4;
        Ok((data, vec![n]))
    }
}

/// Transpose: physically reorder f32 data according to `perm`.
/// Returns `(permuted_bytes, output_shape)`.
pub fn dispatch_transpose(
    input: &[u8],
    perm: &[u8],
    input_shape: &[usize],
) -> ExecResult<(Vec<u8>, Vec<usize>)> {
    let src = cast_f32(input)?;
    let ndim = perm.len();

    if ndim == 0 || input_shape.is_empty() {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    // Guard: perm must not reference dims beyond the input shape.
    if perm.iter().any(|&p| (p as usize) >= input_shape.len()) {
        return Ok((input.to_vec(), input_shape.to_vec()));
    }

    let strides = compute_strides(input_shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p as usize]).collect();
    let out_strides = compute_strides(&out_shape);

    let total = src.len();
    // Verify shape matches data length.
    let shape_elems: usize = input_shape
        .iter()
        .copied()
        .fold(1usize, usize::saturating_mul);
    if shape_elems != total {
        // Shape doesn't match data — fall back to pass-through.
        return Ok((input.to_vec(), input_shape.to_vec()));
    }
    let mut dst = vec![0.0f32; total];
    for (flat_idx, dst_val) in dst.iter_mut().enumerate().take(total) {
        let mut remaining = flat_idx;
        let mut src_flat = 0usize;
        for (d, &p) in perm.iter().enumerate() {
            let coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            src_flat += coord * strides[p as usize];
        }
        // Guard against stride overflow from shape mismatches.
        if src_flat >= total {
            continue;
        }
        *dst_val = src[src_flat];
    }
    Ok((f32_vec_to_bytes(dst), out_shape))
}

/// Broadcast `src` to fill `target_shape` using numpy-style broadcasting.
///
/// Infers the source shape from `n_src` and `target_shape`: each source dim
/// is either 1 (broadcast) or equals the target dim.
fn broadcast_to(src: &[f32], n_src: usize, target_shape: &[usize]) -> Vec<f32> {
    let target_elems: usize = target_shape
        .iter()
        .copied()
        .fold(1usize, usize::saturating_mul);
    if src.is_empty() || target_elems == 0 {
        return vec![0.0f32; target_elems];
    }

    // Infer source shape: for each target dim, source dim is target dim
    // if it divides evenly into remaining elements, else 1 (broadcast).
    let ndim = target_shape.len();
    let mut src_shape = vec![1usize; ndim];
    let mut remaining = n_src;
    for i in (0..ndim).rev() {
        if remaining > 1 && target_shape[i] > 0 && remaining.is_multiple_of(target_shape[i]) {
            src_shape[i] = target_shape[i];
            remaining /= target_shape[i];
        }
    }

    let src_strides = compute_strides(&src_shape);
    let tgt_strides = compute_strides(target_shape);

    let mut out = Vec::with_capacity(target_elems);
    for flat_idx in 0..target_elems {
        let mut src_flat = 0usize;
        let mut rem = flat_idx;
        for d in 0..ndim {
            let coord = rem / tgt_strides[d];
            rem %= tgt_strides[d];
            // Wrap coordinate if source dim is 1 (broadcast).
            let src_coord = if src_shape[d] == 1 { 0 } else { coord };
            src_flat += src_coord * src_strides[d];
        }
        out.push(src[src_flat]);
    }
    out
}

// Shape inference functions (infer_output_shape, infer_custom_output_shape)
// have been consolidated into eval::shape_resolve::resolve_float_shape().

// ── Conv2d ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn dispatch_conv2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    // Infer shapes: data=[N,C,H,W], weight=[OC,IC/group,KH,KW]
    let oc = weight.len() / (kh * kw * (weight.len() / (kh * kw))).max(1);
    // More robust: total weight = OC * (IC/group) * KH * KW
    let ic_per_group = if oc > 0 {
        weight.len() / (oc * kh * kw)
    } else {
        1
    };
    let ic = ic_per_group * group;
    let spatial = data.len() / ic.max(1);
    // Infer H, W from spatial (assume square if ambiguous)
    let h_in = (spatial as f32).sqrt() as usize;
    let w_in = if h_in > 0 { spatial / h_in } else { 1 };

    // For N>1 batches, we need to figure out batch size
    let n = data.len() / (ic * h_in * w_in).max(1);

    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let mut out = vec![0.0f32; n * oc * h_out * w_out];

    let oc_per_group = oc / group.max(1);

    for batch in 0..n {
        for g in 0..group {
            for oc_idx in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc_idx;
                let bias_val = if has_bias {
                    let b = cast_f32(bias_bytes).unwrap_or_default();
                    if abs_oc < b.len() {
                        b[abs_oc]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = bias_val;
                        for ic_idx in 0..ic_per_group {
                            let abs_ic = g * ic_per_group + ic_idx;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = oh * sh + ky * dh;
                                    let ix = ow * sw + kx * dw;
                                    let iy = iy as isize - ph as isize;
                                    let ix = ix as isize - pw as isize;
                                    if iy >= 0
                                        && iy < h_in as isize
                                        && ix >= 0
                                        && ix < w_in as isize
                                    {
                                        let d_idx = ((batch * ic + abs_ic) * h_in + iy as usize)
                                            * w_in
                                            + ix as usize;
                                        let w_idx =
                                            ((abs_oc * ic_per_group + ic_idx) * kh + ky) * kw + kx;
                                        if d_idx < data.len() && w_idx < weight.len() {
                                            sum += data[d_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        let o_idx = ((batch * oc + abs_oc) * h_out + oh) * w_out + ow;
                        if o_idx < out.len() {
                            out[o_idx] = sum;
                        }
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── ConvTranspose ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn dispatch_conv_transpose(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    output_pad_h: usize,
    output_pad_w: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    // weight=[IC, OC/group, KH, KW]
    let weight_per_filter = kh * kw;
    let ic = if weight_per_filter > 0 {
        weight.len() / weight_per_filter
    } else {
        return Ok(vec![]);
    };
    // ic here = IC * (OC/group), need to separate
    // Actually weight = [IC, OC/group, KH, KW] so total = IC * (OC/group) * KH * KW
    // We need additional info to split IC and OC/group. Use group to help.
    // Heuristic: assume IC comes from data channel count
    let data_channels = if data.is_empty() {
        1
    } else {
        // data=[N,IC,H,W], try to infer
        let total_spatial = ic * weight_per_filter; // weight.len()
        let oc_per_group = ic / group.max(1); // This isn't quite right
                                              // Simpler: just treat as single batch for now
        let _ = total_spatial;
        let _ = oc_per_group;
        group // fallback
    };
    let _ = data_channels;

    // For transposed conv: H_out = (H_in - 1) * stride - 2*pad + dilation*(kernel-1) + output_pad + 1
    // Infer input spatial dims from data
    // This is complex without shape metadata. Do a simplified version.
    let total = data.len();
    let ic_actual = weight.len() / (kh * kw);
    // weight=[IC, OC/group, KH, KW], so ic_actual = IC * OC/group
    // For group=1: IC channels in, OC channels out
    let oc_per_group = if ic_actual > 0 {
        ic_actual / group.max(1)
    } else {
        1
    };
    // Heuristic: assume square spatial, batch=1
    let in_channels = group; // minimal assumption
    let spatial_per_channel = total / in_channels.max(1);
    let h_in = (spatial_per_channel as f32).sqrt() as usize;
    let w_in = if h_in > 0 {
        spatial_per_channel / h_in
    } else {
        1
    };

    let h_out = (h_in.saturating_sub(1)) * sh + dh * (kh - 1) + output_pad_h + 1 - 2 * ph;
    let w_out = (w_in.saturating_sub(1)) * sw + dw * (kw - 1) + output_pad_w + 1 - 2 * pw;
    let oc = oc_per_group * group;

    let mut out = vec![0.0f32; oc * h_out * w_out];

    // Add bias
    if has_bias {
        if let Ok(b) = cast_f32(bias_bytes) {
            for c in 0..oc {
                let bias_val = if c < b.len() { b[c] } else { 0.0 };
                for h in 0..h_out {
                    for w in 0..w_out {
                        out[(c * h_out + h) * w_out + w] = bias_val;
                    }
                }
            }
        }
    }

    // Transposed convolution: scatter input values through the kernel
    for g in 0..group {
        for ic_idx in 0..1 {
            // simplified: 1 input channel per group
            let abs_ic = g + ic_idx;
            for oc_idx in 0..oc_per_group {
                let abs_oc = g * oc_per_group + oc_idx;
                for ih in 0..h_in {
                    for iw in 0..w_in {
                        let d_idx = (abs_ic * h_in + ih) * w_in + iw;
                        let d_val = if d_idx < data.len() {
                            data[d_idx]
                        } else {
                            continue;
                        };
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let oh = ih * sh + ky * dh;
                                let ow = iw * sw + kx * dw;
                                if oh >= ph && ow >= pw {
                                    let oh = oh - ph;
                                    let ow = ow - pw;
                                    if oh < h_out && ow < w_out {
                                        let w_idx =
                                            ((abs_ic * oc_per_group + oc_idx) * kh + ky) * kw + kx;
                                        let o_idx = (abs_oc * h_out + oh) * w_out + ow;
                                        if w_idx < weight.len() && o_idx < out.len() {
                                            out[o_idx] += d_val * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── MaxPool2d ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn dispatch_max_pool_2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // Infer [N,C,H,W]. Pool doesn't change channels, only spatial dims.
    // Without shape metadata, we infer from the pool params and data length.
    // Use the pool stride/kernel to figure out reasonable H,W.
    let total = data.len();
    // Heuristic: try common spatial sizes
    let (channels, h_in, w_in) = infer_nchw(total, 1);
    let n = 1;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut out = vec![f32::NEG_INFINITY; n * channels * h_out * w_out];

    for batch in 0..n {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f32::NEG_INFINITY;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oh * sh + ky) as isize - ph as isize;
                            let ix = (ow * sw + kx) as isize - pw as isize;
                            if iy >= 0 && iy < h_in as isize && ix >= 0 && ix < w_in as isize {
                                let idx = ((batch * channels + c) * h_in + iy as usize) * w_in
                                    + ix as usize;
                                if idx < data.len() {
                                    max_val = max_val.max(data[idx]);
                                }
                            }
                        }
                    }
                    let o_idx = ((batch * channels + c) * h_out + oh) * w_out + ow;
                    if o_idx < out.len() {
                        out[o_idx] = max_val;
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── AvgPool2d ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn dispatch_avg_pool_2d(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let total = data.len();
    let (channels, h_in, w_in) = infer_nchw(total, 1);
    let n = 1;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut out = vec![0.0f32; n * channels * h_out * w_out];

    for batch in 0..n {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oh * sh + ky) as isize - ph as isize;
                            let ix = (ow * sw + kx) as isize - pw as isize;
                            if iy >= 0 && iy < h_in as isize && ix >= 0 && ix < w_in as isize {
                                let idx = ((batch * channels + c) * h_in + iy as usize) * w_in
                                    + ix as usize;
                                if idx < data.len() {
                                    sum += data[idx];
                                    count += 1;
                                }
                            }
                        }
                    }
                    let o_idx = ((batch * channels + c) * h_out + oh) * w_out + ow;
                    if o_idx < out.len() {
                        out[o_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}

/// Heuristic to infer (C, H, W) from total element count and batch size.
fn infer_nchw(total: usize, n: usize) -> (usize, usize, usize) {
    let per_batch = total / n.max(1);
    // Try common channel counts: 1, 3, then factors
    for &c in &[3, 1, 64, 128, 256, 512, 32, 16] {
        if per_batch.is_multiple_of(c) {
            let spatial = per_batch / c;
            let h = (spatial as f32).sqrt() as usize;
            if h > 0 && spatial.is_multiple_of(h) {
                return (c, h, spatial / h);
            }
        }
    }
    // Fallback: single channel, try square
    let h = (per_batch as f32).sqrt() as usize;
    if h > 0 && per_batch.is_multiple_of(h) {
        (1, h, per_batch / h)
    } else {
        (1, 1, per_batch)
    }
}

// ── GlobalAvgPool ───────────────────────────────────────────────────────────

fn dispatch_global_avg_pool(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // GlobalAvgPool: [N,C,H,W] → [N,C,1,1]
    // Without shape metadata, infer channels from data
    let total = data.len();
    let (channels, h, w) = infer_nchw(total, 1);
    let spatial = h * w;

    let mut out = Vec::with_capacity(channels);
    for c in 0..channels {
        let start = c * spatial;
        let end = (start + spatial).min(data.len());
        if start < data.len() {
            let sum: f32 = data[start..end].iter().sum();
            out.push(sum / spatial as f32);
        }
    }
    Ok(f32_vec_to_bytes(out))
}

// ── Resize ──────────────────────────────────────────────────────────────────

fn dispatch_resize(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // inputs[1] = scales or sizes (f32 or i64)
    let scales_bytes = inputs.get(1).copied().unwrap_or(&[][..]);

    if data.is_empty() {
        return Ok(vec![]);
    }

    // Parse scales as f32
    let scales: Vec<f32> = if !scales_bytes.is_empty() && scales_bytes.len() % 4 == 0 {
        cast_f32(scales_bytes)?.to_vec()
    } else {
        vec![1.0; 4]
    };

    // If all scales are 1.0, pass through
    if scales.iter().all(|&s| (s - 1.0).abs() < 1e-6) {
        return Ok(inputs[0].to_vec());
    }

    // Simple 1-D resize using the product of all scales
    let total_scale: f32 = scales.iter().product();
    let out_len = ((data.len() as f32) * total_scale) as usize;
    if out_len == 0 || out_len > data.len() * 64 {
        return Ok(inputs[0].to_vec());
    }

    let out: Vec<f32> = match mode {
        1 => {
            // Linear interpolation
            (0..out_len)
                .map(|i| {
                    let src_f = (i as f32) / total_scale;
                    let lo = src_f.floor() as usize;
                    let hi = (lo + 1).min(data.len() - 1);
                    let frac = src_f - lo as f32;
                    let lo = lo.min(data.len() - 1);
                    data[lo] * (1.0 - frac) + data[hi] * frac
                })
                .collect()
        }
        _ => {
            // Nearest neighbor (mode 0) or cubic/unknown fallback
            (0..out_len)
                .map(|i| {
                    let src = ((i as f32) / total_scale) as usize;
                    data[src.min(data.len() - 1)]
                })
                .collect()
        }
    };

    Ok(f32_vec_to_bytes(out))
}

// ── PadOp ───────────────────────────────────────────────────────────────────

fn dispatch_pad(inputs: &[&[u8]], mode: u8) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let pads_bytes = inputs.get(1).copied().unwrap_or(&[][..]);

    if pads_bytes.is_empty() {
        return Ok(inputs[0].to_vec());
    }

    // Pads are i64: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    let pads: Vec<i64> = if pads_bytes.len() % 8 == 0 {
        iter_i64(pads_bytes).collect()
    } else {
        // Try as f32 (some models pass pads as float)
        cast_f32(pads_bytes)?.iter().map(|&v| v as i64).collect()
    };

    if pads.iter().all(|&p| p == 0) {
        return Ok(inputs[0].to_vec());
    }

    // Simple 1-D padding: sum all begin pads and end pads
    let ndim = pads.len() / 2;
    let total_begin: usize = pads[..ndim].iter().map(|&p| p.max(0) as usize).sum();
    let total_end: usize = pads[ndim..].iter().map(|&p| p.max(0) as usize).sum();

    let pad_val = match mode {
        0 => 0.0f32, // constant
        _ => 0.0f32, // reflect/edge simplified to constant
    };

    let out_len = total_begin + data.len() + total_end;
    let mut out = vec![pad_val; out_len];
    out[total_begin..total_begin + data.len()].copy_from_slice(&data);

    if mode == 1 && data.len() > 1 {
        // Reflect: mirror edges
        for (i, v) in out[..total_begin].iter_mut().enumerate() {
            let src = total_begin - i;
            *v = if src < data.len() { data[src] } else { data[0] };
        }
        let tail_start = total_begin + data.len();
        for (i, v) in out[tail_start..tail_start + total_end]
            .iter_mut()
            .enumerate()
        {
            let src = data.len().saturating_sub(2).saturating_sub(i);
            *v = data[src];
        }
    } else if mode == 2 {
        // Edge: replicate border
        let first = data[0];
        let last = *data.last().unwrap_or(&0.0);
        out[..total_begin].fill(first);
        let tail_start = total_begin + data.len();
        out[tail_start..tail_start + total_end].fill(last);
    }

    Ok(f32_vec_to_bytes(out))
}

// ── InstanceNorm ────────────────────────────────────────────────────────────

fn dispatch_instance_norm(inputs: &[&[u8]], size: usize, epsilon: f32) -> ExecResult<Vec<u8>> {
    // inputs: [data, scale, bias]
    // InstanceNorm: normalize each (N,C) spatial slice independently
    // size = number of spatial elements per channel (H*W)
    let data = cast_f32(inputs[0])?;
    let scale = cast_f32(inputs[1])?;
    let bias = cast_f32(inputs[2])?;

    let n_channels = scale.len();
    let spatial = if n_channels > 0 {
        data.len() / n_channels
    } else {
        data.len()
    };
    let actual_size = if size > 0 { size } else { spatial };

    let mut out = data.to_vec();

    for c in 0..n_channels {
        let start = c * actual_size;
        let end = (start + actual_size).min(out.len());
        if start >= out.len() {
            break;
        }
        let slice = &out[start..end];

        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let var: f32 =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / slice.len() as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();

        let s = if c < scale.len() { scale[c] } else { 1.0 };
        let b = if c < bias.len() { bias[c] } else { 0.0 };

        for v in out[start..end].iter_mut() {
            *v = (*v - mean) * inv_std * s + b;
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── LRN ─────────────────────────────────────────────────────────────────────

fn dispatch_lrn(
    inputs: &[&[u8]],
    size: usize,
    alpha: f32,
    beta: f32,
    bias: f32,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // LRN: across channels. data=[N,C,H,W]
    // out[n,c,h,w] = data[n,c,h,w] / (bias + alpha/size * sum(data[n,c',h,w]^2))^beta
    // where c' ranges over [max(0,c-floor(size/2)), min(C-1,c+floor(size/2))]

    // Without shape info, treat as 1-D across-channel normalization
    let half = size / 2;
    let n = data.len();
    let mut out = vec![0.0f32; n];

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let sum_sq: f32 = data[lo..hi].iter().map(|v| v * v).sum();
        let denom = (bias + alpha / size as f32 * sum_sq).powf(beta);
        out[i] = data[i] / denom;
    }

    Ok(f32_vec_to_bytes(out))
}

// ── TopK ────────────────────────────────────────────────────────────────────

fn dispatch_top_k(inputs: &[&[u8]], _axis: usize, largest: bool) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // K from inputs[1] (i64 scalar)
    let k = if inputs.len() >= 2 && inputs[1].len() >= 8 {
        iter_i64(inputs[1]).next().unwrap_or(1) as usize
    } else if inputs.len() >= 2 && inputs[1].len() >= 4 {
        cast_f32(inputs[1])?.first().copied().unwrap_or(1.0) as usize
    } else {
        1
    };

    let k = k.min(data.len());
    // Simple: sort all elements, take top/bottom K
    // Returns values only (indices would need separate output handling)
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    if largest {
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    indexed.truncate(k);

    // Output: interleave values and indices as f32
    // Standard TopK produces two outputs, but since we have single output dispatch,
    // output values as f32 (the primary output)
    let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    Ok(f32_vec_to_bytes(values))
}

// ── ScatterND ───────────────────────────────────────────────────────────────

fn dispatch_scatter_nd(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // inputs: [data, indices, updates]
    let data = cast_f32(inputs[0])?;
    let indices_bytes = inputs[1];
    let updates = cast_f32(inputs[2])?;

    let mut out = data.to_vec();

    // Simple 1-D scatter: indices are i64, each indexing into the flat output
    let indices: Vec<usize> = iter_i64(indices_bytes).map(|v| v as usize).collect();

    for (i, &idx) in indices.iter().enumerate() {
        if idx < out.len() && i < updates.len() {
            out[idx] = updates[i];
        }
    }

    Ok(f32_vec_to_bytes(out))
}

// ── CumSum ──────────────────────────────────────────────────────────────────

fn dispatch_cumsum(inputs: &[&[u8]], _axis: usize) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let mut out = data.to_vec();

    // Simple 1-D cumulative sum
    for i in 1..out.len() {
        out[i] += out[i - 1];
    }

    Ok(f32_vec_to_bytes(out))
}

// ── NonZero ─────────────────────────────────────────────────────────────────

fn dispatch_nonzero(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // NonZero returns indices of non-zero elements as i64
    let indices: Vec<i64> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, _)| i as i64)
        .collect();

    Ok(bytemuck::cast_slice(&indices).to_vec())
}

// ── Compress ────────────────────────────────────────────────────────────────

fn dispatch_compress(inputs: &[&[u8]], _axis: usize) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    // condition from inputs[1]: boolean mask
    let cond = to_bools(inputs[1]);

    let out: Vec<f32> = data
        .iter()
        .zip(cond.iter().chain(std::iter::repeat(&0u8)))
        .filter(|(_, &c)| c != 0)
        .map(|(&v, _)| v)
        .collect();

    Ok(f32_vec_to_bytes(out))
}

// ── ReverseSequence ─────────────────────────────────────────────────────────

fn dispatch_reverse_sequence(
    inputs: &[&[u8]],
    _batch_axis: usize,
    _time_axis: usize,
) -> ExecResult<Vec<u8>> {
    // Simple: reverse the entire f32 sequence
    let data = cast_f32(inputs[0])?;
    let mut out = data.to_vec();
    out.reverse();
    Ok(f32_vec_to_bytes(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(data).to_vec()
    }

    #[test]
    fn test_float_add() {
        let a = f32_bytes(&[1.0, 2.0, 3.0]);
        let b = f32_bytes(&[4.0, 5.0, 6.0]);
        let result = dispatch_float(&FloatOp::Add, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_float_add_broadcast() {
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let b = f32_bytes(&[10.0]);
        let result = dispatch_float(&FloatOp::Add, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_float_relu() {
        let x = f32_bytes(&[-1.0, 0.0, 1.0, 2.0]);
        let result = dispatch_float(&FloatOp::Relu, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_float_sigmoid() {
        let x = f32_bytes(&[0.0]);
        let result = dispatch_float(&FloatOp::Sigmoid, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_float_matmul() {
        // [2,3] × [3,2] → [2,2]
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f32_bytes(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = dispatch_float(&FloatOp::MatMul { m: 2, k: 3, n: 2 }, &[&a, &b]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(out, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_float_softmax() {
        let x = f32_bytes(&[1.0, 2.0, 3.0]);
        let result = dispatch_float(&FloatOp::Softmax { size: 3 }, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn test_float_rms_norm() {
        use hologram_core::op::f32_to_bits;
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let w = f32_bytes(&[1.0, 1.0, 1.0, 1.0]);
        let result = dispatch_float(
            &FloatOp::RmsNorm {
                size: 4,
                epsilon: f32_to_bits(1e-5),
            },
            &[&x, &w],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // rms = sqrt((1+4+9+16)/4 + 1e-5) ≈ sqrt(7.5) ≈ 2.7386
        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);
        assert!((out[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_float_gather() {
        // vocab=3, dim=2
        let indices = bytemuck::cast_slice::<i64, u8>(&[0i64, 2]).to_vec();
        let table = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dispatch_float(
            &FloatOp::Gather {
                dim: 2,
                dtype: FloatDType::F32,
            },
            &[&indices, &table],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_float_fused_swiglu() {
        let gate = f32_bytes(&[0.0, 1.0]);
        let up = f32_bytes(&[2.0, 3.0]);
        let result = dispatch_float(&FloatOp::FusedSwiGLU, &[&gate, &up]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        // silu(0)*2 = 0, silu(1)*3 = 0.7310...*3 ≈ 2.1932
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - silu(1.0) * 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_float_reduce_sum() {
        let x = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dispatch_float(&FloatOp::ReduceSum { size: 3 }, &[&x]).unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[6.0, 15.0]);
    }

    #[test]
    fn test_float_concat() {
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0]); // 2 rows of 2
        let b = f32_bytes(&[5.0, 6.0]); // 2 rows of 1
        let result = dispatch_float(
            &FloatOp::Concat {
                size_a: 2,
                size_b: 1,
                dtype: FloatDType::F32,
            },
            &[&a, &b],
        )
        .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    // ── N-D broadcasting tests ──────────────────────────────────────────

    #[test]
    fn test_broadcast_shapes_compatible() {
        assert_eq!(broadcast_shapes(&[2, 1], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[1], &[5]), Some(vec![5]));
        assert_eq!(
            broadcast_shapes(&[4, 1, 3], &[1, 5, 1]),
            Some(vec![4, 5, 3])
        );
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // [2,32] vs [1,64]: dim 1 has 32 vs 64, neither is 1
        assert_eq!(broadcast_shapes(&[2, 32], &[1, 64]), None);
        assert_eq!(broadcast_shapes(&[3], &[4]), None);
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 4]), None);
    }

    #[test]
    fn test_broadcast_2d_row_vector() {
        // [2,3] + [1,3] => broadcast row: result should add row-wise
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // shape [2,3]
        let b = f32_bytes(&[10.0, 20.0, 30.0]); // shape [1,3]
        let result =
            dispatch_float_with_shapes(&FloatOp::Add, &[&a, &b], &[vec![2, 3], vec![1, 3]])
                .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_2d_column_vector() {
        // [2,3] / [2,1] => broadcast column (the LayerNorm pattern)
        let a = f32_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]); // shape [2,3]
        let b = f32_bytes(&[2.0, 5.0]); // shape [2,1]
        let result =
            dispatch_float_with_shapes(&FloatOp::Div, &[&a, &b], &[vec![2, 3], vec![2, 1]])
                .unwrap();
        let out: &[f32] = bytemuck::cast_slice(&result);
        assert_eq!(out, &[5.0, 10.0, 15.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_broadcast_incompatible_falls_back_to_cycling() {
        // [2,32] vs [1,64]: NOT broadcast-compatible.
        // Must not panic — falls back to cycling.
        let a = f32_bytes(&vec![1.0; 64]); // shape [2,32]
        let b = f32_bytes(&vec![2.0; 64]); // shape [1,64]
        let result =
            dispatch_float_with_shapes(&FloatOp::Add, &[&a, &b], &[vec![2, 32], vec![1, 64]]);
        assert!(result.is_ok()); // Must not panic
        let binding = result.unwrap();
        let out: &[f32] = bytemuck::cast_slice(&binding);
        assert_eq!(out.len(), 64); // cycling: max(64,64)
    }

    #[test]
    fn test_broadcast_shape_data_mismatch_falls_back() {
        // Shape says [2,4] (8 elements) but data has 6 f32s — shape mismatch
        // Must fall back to cycling, not panic.
        let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f32_bytes(&[10.0]);
        let result = dispatch_float_with_shapes(
            &FloatOp::Add,
            &[&a, &b],
            &[vec![2, 4], vec![1]], // shape [2,4] doesn't match 6 elements
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_broadcast_compare_2d() {
        // [2,3] > [1,3] => broadcast comparison
        let a = f32_bytes(&[1.0, 20.0, 3.0, 40.0, 5.0, 60.0]); // shape [2,3]
        let b = f32_bytes(&[10.0, 10.0, 10.0]); // shape [1,3]
        let result =
            dispatch_float_with_shapes(&FloatOp::Greater, &[&a, &b], &[vec![2, 3], vec![1, 3]])
                .unwrap();
        // 1>10=0, 20>10=1, 3>10=0, 40>10=1, 5>10=0, 60>10=1
        assert_eq!(result, vec![0, 1, 0, 1, 0, 1]);
    }
}
