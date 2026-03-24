//! Float-domain kernel dispatch for `FloatOp` graph operations.
//!
//! All kernels operate on `&[u8]` inputs interpreted as `&[f32]` via bytemuck,
//! matching the pattern used by `MatMulLut4`/`MatMulLut8`.

pub(crate) mod attention;
pub(crate) mod cast;
mod conv;
mod elementwise;
pub(crate) mod gather_concat;
mod helpers;
pub mod matmul;
mod misc;
pub(crate) mod norm;
pub(crate) mod pool;
mod reduce;
mod shape_ops;
mod spatial;
#[cfg(test)]
mod tests;

use hologram_core::op::{bits_to_f32, FloatOp, OpCategory};

use crate::error::ExecResult;
use crate::eval::executor::ExecutionContext;

// ── Re-exports (public API, unchanged) ───────────────────────────────────────

pub use gather_concat::dispatch_shape_sliced;
pub use helpers::{compute_strides, f32_vec_to_bytes};
pub use matmul::{dispatch_batched_matmul, dispatch_matmul, GemmParams};
pub use shape_ops::{dispatch_reshape_with_shape, dispatch_transpose};

// ── Dispatch entry points ────────────────────────────────────────────────────

/// Dispatch a `FloatOp` with the given byte-buffer inputs (no execution context).
#[inline]
pub fn dispatch_float(op: &FloatOp, inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    dispatch_float_ctx(op, inputs, None)
}

/// Dispatch with optional execution context (carries position offset for RoPE etc.).
pub fn dispatch_float_ctx(
    op: &FloatOp,
    inputs: &[&[u8]],
    ctx: Option<&ExecutionContext>,
) -> ExecResult<Vec<u8>> {
    match op.category() {
        OpCategory::UnaryElementwise => elementwise::unary_map(inputs, |v| op.apply_unary(v)),
        OpCategory::BinaryElementwise => {
            elementwise::binary_elementwise(inputs, |a, b| op.apply_binary(a, b))
        }
        OpCategory::BinaryCompare => {
            elementwise::binary_compare(inputs, |a, b| op.apply_compare(a, b))
        }
        OpCategory::BinaryByteBool => {
            elementwise::binary_byte_bool(inputs, |a, b| op.apply_byte_bool(a, b))
        }
        OpCategory::UnaryByteBool => {
            elementwise::unary_byte_bool(inputs, |a| if a != 0 { 0 } else { 1 })
        }
        OpCategory::UnaryToU8 => elementwise::dispatch_isnan(inputs),
        OpCategory::Custom => dispatch_custom(op, inputs, ctx),
    }
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
    // Vision ops need explicit input shapes (can't infer H/W from buffer length).
    if let FloatOp::Conv2d {
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        input_h: _,
        input_w: _,
    } = op
    {
        return conv::dispatch_conv2d_with_shapes(
            inputs,
            input_shapes,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
            *dilation_h as usize,
            *dilation_w as usize,
            *group as usize,
        );
    }
    match op.category() {
        OpCategory::BinaryElementwise if input_shapes.len() >= 2 => {
            elementwise::binary_elementwise_broadcast(inputs, input_shapes, |a, b| {
                op.apply_binary(a, b)
            })
        }
        OpCategory::BinaryCompare if input_shapes.len() >= 2 => {
            elementwise::binary_compare_broadcast(inputs, input_shapes, |a, b| {
                op.apply_compare(a, b)
            })
        }
        _ => dispatch_float_ctx(op, inputs, None),
    }
}

/// Dispatch a `FloatOp` into a pre-allocated output buffer.
///
/// The output buffer is cleared and reused — its backing allocation persists
/// across calls, eliminating repeated heap allocation in the hot path.
/// Falls back to `dispatch_float_ctx` for ops that don't yet support in-place output.
pub fn dispatch_float_into(
    op: &FloatOp,
    inputs: &[&[u8]],
    ctx: Option<&ExecutionContext>,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    match op.category() {
        OpCategory::UnaryElementwise => {
            // Fast path: monomorphized dispatch for common ops enables autovectorization.
            // The compiler can SIMD-ize x.max(0.0) for Relu, -x for Neg, etc. when the
            // operation is known at compile time instead of hidden behind a closure.
            match op {
                FloatOp::Relu => elementwise::unary_map_into(inputs, |v| v.max(0.0), out_buf),
                FloatOp::Neg => elementwise::unary_map_into(inputs, |v| -v, out_buf),
                FloatOp::Abs => elementwise::unary_map_into(inputs, |v| v.abs(), out_buf),
                FloatOp::Sigmoid => {
                    elementwise::unary_map_into(inputs, |v| 1.0 / (1.0 + (-v).exp()), out_buf);
                }
                FloatOp::Silu => {
                    elementwise::unary_map_into(
                        inputs,
                        |v| v * (1.0 / (1.0 + (-v).exp())),
                        out_buf,
                    );
                }
                FloatOp::Tanh => elementwise::unary_map_into(inputs, |v| v.tanh(), out_buf),
                FloatOp::Exp => elementwise::unary_map_into(inputs, |v| v.exp(), out_buf),
                FloatOp::Reciprocal => elementwise::unary_map_into(inputs, |v| 1.0 / v, out_buf),
                _ => elementwise::unary_map_into(inputs, |v| op.apply_unary(v), out_buf),
            }
            Ok(())
        }
        OpCategory::BinaryElementwise => {
            // Monomorphized dispatch for common binary ops — enables autovectorization.
            match op {
                FloatOp::Add => elementwise::binary_elementwise_into(inputs, |a, b| a + b, out_buf),
                FloatOp::Sub => elementwise::binary_elementwise_into(inputs, |a, b| a - b, out_buf),
                FloatOp::Mul => elementwise::binary_elementwise_into(inputs, |a, b| a * b, out_buf),
                FloatOp::Div => elementwise::binary_elementwise_into(inputs, |a, b| a / b, out_buf),
                FloatOp::Min => {
                    elementwise::binary_elementwise_into(inputs, |a, b| a.min(b), out_buf);
                }
                FloatOp::Max => {
                    elementwise::binary_elementwise_into(inputs, |a, b| a.max(b), out_buf);
                }
                _ => {
                    elementwise::binary_elementwise_into(
                        inputs,
                        |a, b| op.apply_binary(a, b),
                        out_buf,
                    );
                }
            }
            Ok(())
        }
        _ => {
            // Try dedicated _into dispatch for hot custom ops before falling back.
            if dispatch_custom_into(op, inputs, out_buf)? {
                return Ok(());
            }
            // Fall back to allocating dispatch for remaining ops.
            let result = dispatch_float_ctx(op, inputs, ctx)?;
            out_buf.extend_from_slice(&result);
            Ok(())
        }
    }
}

/// Resolve a size=0 sentinel to the actual element count from the input buffer.
///
/// The 0 sentinel means "infer at runtime". For ops like Softmax and RmsNorm,
/// the correct size is the number of f32 elements in the first input.
#[inline]
pub(crate) fn resolve_size(compiled_size: u32, inputs: &[&[u8]]) -> usize {
    let n_floats = inputs.first().map(|b| b.len() / 4).unwrap_or(0);
    if compiled_size == 0 || n_floats == 0 {
        n_floats
    } else {
        let cs = compiled_size as usize;
        // Use compiled size if it divides evenly; otherwise infer.
        if n_floats.is_multiple_of(cs) {
            cs
        } else {
            n_floats
        }
    }
}

/// Infer the full axis size for a Slice op from the buffer element count.
///
/// The baked `end` is the slice upper bound, not the full axis dimension.
/// When `end` evenly divides `n_elems`, it may be the axis size (fast path).
/// Otherwise, search upward for the smallest divisor >= `end` — this finds
/// the actual axis size even when leading dimensions (like seq_len) change
/// at runtime.
#[inline]
pub(crate) fn infer_slice_axis_size(n_elems: usize, end: usize) -> usize {
    if end == 0 || n_elems == 0 {
        return n_elems;
    }
    // Fast path: end divides evenly (covers the common case where end == axis_size).
    if n_elems.is_multiple_of(end) {
        return end;
    }
    // Search upward from end+1 for the smallest divisor of n_elems.
    // This finds the actual axis size when end < axis_size (e.g., partial slice
    // from combined QKV: n_elems = seq*2560, end = 2048 → finds 2560).
    for s in (end + 1)..=n_elems {
        if n_elems.is_multiple_of(s) {
            return s;
        }
    }
    n_elems
}

/// Attempt in-place dispatch for high-frequency custom ops.
///
/// Returns `Ok(true)` if handled, `Ok(false)` to fall back to allocating dispatch.
///
/// Handles size=0 sentinels by inferring from input buffer length at runtime.
fn dispatch_custom_into(op: &FloatOp, inputs: &[&[u8]], out_buf: &mut Vec<u8>) -> ExecResult<bool> {
    match op {
        // MatMul: output = M×N floats. Write directly into out_buf.
        FloatOp::MatMul { m, k, n } => {
            matmul::dispatch_matmul_into(inputs, *m as usize, *k as usize, *n as usize, out_buf)?;
            Ok(true)
        }
        // Softmax/LogSoftmax: output size == input size (element-preserving).
        // size=0 sentinel → infer from input buffer (full 1-D softmax).
        FloatOp::Softmax { size } => {
            let actual = resolve_size(*size, inputs);
            norm::dispatch_softmax_into(inputs, actual, out_buf)?;
            Ok(true)
        }
        FloatOp::LogSoftmax { size } => {
            let actual = resolve_size(*size, inputs);
            norm::dispatch_log_softmax_into(inputs, actual, out_buf)?;
            Ok(true)
        }
        // RmsNorm/LayerNorm: output size == input size.
        FloatOp::RmsNorm { size, epsilon } => {
            let actual = resolve_size(*size, inputs);
            norm::dispatch_rms_norm_into(inputs, actual, f32::from_bits(*epsilon), out_buf)?;
            Ok(true)
        }
        FloatOp::AddRmsNorm { size, epsilon } => {
            let actual = resolve_size(*size, inputs);
            norm::dispatch_add_rms_norm_into(inputs, actual, f32::from_bits(*epsilon), out_buf)?;
            Ok(true)
        }
        FloatOp::LayerNorm { size, epsilon } => {
            let actual = resolve_size(*size, inputs);
            norm::dispatch_layer_norm_into(inputs, actual, f32::from_bits(*epsilon), out_buf)?;
            Ok(true)
        }
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            ..
        } => {
            let result = attention::dispatch_attention(
                inputs,
                *head_dim as usize,
                *num_q_heads as usize,
                *num_kv_heads as usize,
                bits_to_f32(*scale),
                *causal,
                *heads_first,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(true)
        }
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
            input_h,
            input_w,
        } => {
            let result = conv::dispatch_conv2d_direct(
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
                *input_h as usize,
                *input_w as usize,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(true)
        }
        FloatOp::RotaryEmbedding { .. } => {
            // RoPE needs position offset from ctx — fall back to allocating dispatch.
            Ok(false)
        }
        _ => Ok(false),
    }
}

/// Dispatch a fused chain of unary element-wise f32 ops.
///
/// Applies each op in sequence to every element, avoiding intermediate buffers.
pub fn dispatch_fused_chain(chain: &[FloatOp], inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    let x = helpers::cast_f32(inputs[0])?;
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
    Ok(helpers::f32_vec_to_bytes(out))
}

/// Dispatch a fused chain into a pre-allocated output buffer.
pub fn dispatch_fused_chain_into(
    chain: &[FloatOp],
    inputs: &[&[u8]],
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    let x = helpers::cast_f32(inputs[0])?;
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
    out_buf.extend_from_slice(bytemuck::cast_slice(&out));
    Ok(())
}

/// Dispatch ops that need dedicated kernel logic.
fn dispatch_custom(
    op: &FloatOp,
    inputs: &[&[u8]],
    ctx: Option<&ExecutionContext>,
) -> ExecResult<Vec<u8>> {
    match op {
        FloatOp::MatMul { m, k, n } => {
            matmul::dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize)
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
        } => matmul::dispatch_gemm(
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
        FloatOp::Softmax { size } => norm::dispatch_softmax(inputs, resolve_size(*size, inputs)),
        FloatOp::LogSoftmax { size } => {
            norm::dispatch_log_softmax(inputs, resolve_size(*size, inputs))
        }
        FloatOp::RmsNorm { size, epsilon } => {
            norm::dispatch_rms_norm(inputs, resolve_size(*size, inputs), bits_to_f32(*epsilon))
        }
        FloatOp::AddRmsNorm { size, epsilon } => {
            norm::dispatch_add_rms_norm(inputs, resolve_size(*size, inputs), bits_to_f32(*epsilon))
        }
        FloatOp::LayerNorm { size, epsilon } => {
            norm::dispatch_layer_norm(inputs, resolve_size(*size, inputs), bits_to_f32(*epsilon))
        }
        FloatOp::ReduceSum { size } => {
            reduce::dispatch_reduce(inputs, resolve_size(*size, inputs), reduce::reduce_sum)
        }
        FloatOp::ReduceMean { size } => {
            reduce::dispatch_reduce(inputs, resolve_size(*size, inputs), reduce::reduce_mean)
        }
        FloatOp::ReduceMax { size } => {
            reduce::dispatch_reduce(inputs, resolve_size(*size, inputs), reduce::reduce_max)
        }
        FloatOp::ReduceMin { size } => {
            reduce::dispatch_reduce(inputs, resolve_size(*size, inputs), reduce::reduce_min)
        }
        FloatOp::Gather { dim, dtype } => {
            gather_concat::dispatch_gather(inputs, *dim as usize, *dtype)
        }
        FloatOp::Concat {
            size_a,
            size_b,
            dtype,
        } => gather_concat::dispatch_concat(inputs, *size_a as usize, *size_b as usize, *dtype),
        FloatOp::Reshape | FloatOp::GatherND => Ok(inputs[0].to_vec()),
        FloatOp::Transpose { .. } => {
            // Flat dispatch path has no shape metadata — passthrough.
            // The tape executor uses InlineTranspose with baked shapes instead.
            Ok(inputs[0].to_vec())
        }
        FloatOp::Cast { from, to } => cast::dispatch_cast(inputs, *from, *to),
        FloatOp::Embed { dim, quant } => cast::dispatch_embed(inputs, *dim as usize, *quant),
        FloatOp::Where => gather_concat::dispatch_where(inputs),
        FloatOp::Range => gather_concat::dispatch_range(inputs),
        FloatOp::Shape { dtype, start, end } => {
            gather_concat::dispatch_shape(inputs, *dtype, *start, *end)
        }
        FloatOp::RotaryEmbedding { dim, base, n_heads } => {
            let start_pos = ctx.map(|c| c.position_offset as usize).unwrap_or(0);
            attention::dispatch_rope(
                inputs,
                *dim as usize,
                bits_to_f32(*base),
                *n_heads as usize,
                start_pos,
            )
        }
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            ..
        } => attention::dispatch_attention(
            inputs,
            *head_dim as usize,
            *num_q_heads as usize,
            *num_kv_heads as usize,
            bits_to_f32(*scale),
            *causal,
            *heads_first,
        ),
        FloatOp::Dequantize => cast::dispatch_dequantize(inputs),
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
            input_h,
            input_w,
        } => conv::dispatch_conv2d_direct(
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
            *input_h as usize,
            *input_w as usize,
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
            input_h,
            input_w,
        } => conv::dispatch_conv_transpose(
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
            *input_h as usize,
            *input_w as usize,
        ),
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => pool::dispatch_max_pool_2d(
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
        } => pool::dispatch_avg_pool_2d(
            inputs,
            *kernel_h as usize,
            *kernel_w as usize,
            *stride_h as usize,
            *stride_w as usize,
            *pad_h as usize,
            *pad_w as usize,
        ),
        FloatOp::GlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => pool::dispatch_global_avg_pool_direct(
            inputs,
            *channels as usize,
            *spatial_h as usize,
            *spatial_w as usize,
        ),
        FloatOp::Resize { mode } => spatial::dispatch_resize(inputs, *mode),
        FloatOp::PadOp { mode } => spatial::dispatch_pad(inputs, *mode),
        FloatOp::InstanceNorm { size, epsilon } => {
            norm::dispatch_instance_norm(inputs, resolve_size(*size, inputs), bits_to_f32(*epsilon))
        }
        FloatOp::LRN {
            size,
            alpha,
            beta,
            bias,
        } => norm::dispatch_lrn(
            inputs,
            *size as usize,
            bits_to_f32(*alpha),
            bits_to_f32(*beta),
            bits_to_f32(*bias),
        ),
        // ── Utility ops ─────────────────────────────────────────────────
        FloatOp::ReduceProd { size } => {
            reduce::dispatch_reduce(inputs, *size as usize, reduce::reduce_prod)
        }
        FloatOp::TopK { axis, largest } => misc::dispatch_top_k(inputs, *axis as usize, *largest),
        FloatOp::ScatterND => misc::dispatch_scatter_nd(inputs),
        FloatOp::CumSum { axis } => misc::dispatch_cumsum(inputs, *axis as usize),
        FloatOp::NonZero => misc::dispatch_nonzero(inputs),
        FloatOp::Compress { axis } => misc::dispatch_compress(inputs, *axis as usize),
        FloatOp::ReverseSequence {
            batch_axis,
            time_axis,
        } => misc::dispatch_reverse_sequence(inputs, *batch_axis as usize, *time_axis as usize),
        // ── Slice ──────────────────────────────────────────────────────
        FloatOp::Slice {
            axis_from_end,
            start,
            end,
            axis_size: compiled_axis_size,
        } => {
            let data = inputs[0];
            let start = *start as usize;
            let end = *end as usize;
            let _afe = *axis_from_end as usize;
            let compiled_as = *compiled_axis_size as usize;

            // Without shape info, infer from buffer size and elem_size=4 (f32).
            let n_elems = data.len() / 4;
            if n_elems == 0 {
                return Ok(vec![]);
            }

            // Use the compiled axis_size if provided and it divides n_elems.
            // Otherwise fall back to heuristic inference from buffer size.
            let axis_size = if compiled_as > 0 && n_elems.is_multiple_of(compiled_as) {
                compiled_as
            } else {
                infer_slice_axis_size(n_elems, end)
            };
            let post = 1;
            let chunk = post * 4; // bytes per element along axis
            let src_stride = axis_size * chunk;
            let actual_end = end.min(axis_size);
            if start >= actual_end {
                return Ok(vec![]);
            }
            let slice_len = actual_end - start;
            let pre = (data.len()) / src_stride;
            let pre = pre.max(1);
            let dst_stride = slice_len * chunk;
            let out_bytes = pre * dst_stride;

            let mut out = vec![0u8; out_bytes];
            for i in 0..pre {
                let src_off = i * src_stride + start * chunk;
                let dst_off = i * dst_stride;
                let src_end = (src_off + dst_stride).min(data.len());
                let copy_len = src_end.saturating_sub(src_off).min(dst_stride);
                if copy_len > 0 {
                    out[dst_off..dst_off + copy_len]
                        .copy_from_slice(&data[src_off..src_off + copy_len]);
                }
            }
            Ok(out)
        }
        // ── KV cache ops ───────────────────────────────────────────────
        // Pass-through when no KvCacheState is available (non-pipeline execution).
        // When KvCacheState is wired in, the executor handles these before dispatch.
        FloatOp::KvWrite { .. } => Ok(inputs[0].to_vec()),
        FloatOp::KvRead { .. } => Ok(inputs.first().map(|b| b.to_vec()).unwrap_or_default()),
        _ => unreachable!("non-custom op {:?} routed to dispatch_custom", op),
    }
}
