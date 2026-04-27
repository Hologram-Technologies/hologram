//! Kernel dispatch: routes TapeKernel variants to execution code.

use hologram_core::op::{FloatOp, RingLevel};
use hologram_core::ring::byte_io::{read_le_u64, write_le_u64};
use hologram_graph::constant::ConstantId;

use crate::buffer::OutputBuffer;
use crate::error::ExecResult;
use crate::tape::{TapeContext, TapeKernel};

/// Result of kernel dispatch — output written to `out_buf`.
///
/// This is a unit struct; the execute loop discards it (only `?` propagation
/// for errors). Kept as a named type rather than `()` for readability in
/// dispatch_kernel match arms.
pub(crate) struct DispatchOk;

/// CPU-only kernel dispatch.
#[inline]
pub(crate) fn dispatch_kernel(
    kernel: &TapeKernel,
    inputs: &[&[u8]],
    input_metas: &crate::shape_resolve::InputMetas,
    input_shapes: &[Option<hologram_shape::TensorShape>],
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<DispatchOk> {
    use crate::float_dispatch;
    use crate::kv::KvStore;
    use crate::shape_resolve;

    // Debug: log first few non-trivial dispatches.
    static DK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dk = DK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if dk < 5 {
        let name = match kernel {
            TapeKernel::InlineMatMul { m, k, n } => format!("InlineMatMul m={m} k={k} n={n}"),
            TapeKernel::InlineGemm { m, k, n, .. } => format!("InlineGemm m={m} k={k} n={n}"),
            TapeKernel::Custom(_) => "Custom".into(),
            _ => format!("{:?}", std::mem::discriminant(kernel)),
        };
        tracing::debug!(dk, name, "dispatch_kernel");
    }

    // Helper: get the last dim from an input shape, if available.
    let shape_last_dim =
        |idx: usize| -> Option<usize> { input_shapes.get(idx)?.as_ref()?.last_dim() };

    // Helper: get spatial (H, W) from dims[-2], dims[-1] of a 4-D+ shape.
    // Validates the shape's volume against the data buffer length — runtime
    // shape inference at intermediate nodes can mistakenly populate a tensor's
    // shape from a sibling input (e.g. a conv weight's [out_c, in_c, k, k])
    // when the real data shape is unknown. Trust the shape only if it matches
    // the actual byte count.
    let shape_spatial_hw = |idx: usize| -> Option<(usize, usize)> {
        let s = input_shapes.get(idx)?.as_ref()?;
        if s.ndim() < 4 {
            return None;
        }
        let shape_vol: usize = s.dims.iter().copied().product();
        let data_floats = inputs.get(idx).map(|b| b.len() / 4).unwrap_or(0);
        if data_floats > 0 && shape_vol != data_floats {
            return None;
        }
        Some((s.dims[s.ndim() - 2], s.dims[s.ndim() - 1]))
    };

    // Helper: get (C, H, W) from dims[-3], dims[-2], dims[-1] of a 3-D+ shape.
    let shape_chw = |idx: usize| -> Option<(usize, usize, usize)> {
        let s = input_shapes.get(idx)?.as_ref()?;
        if s.ndim() >= 3 {
            let n = s.ndim();
            let c = s.dims[n - 3];
            let h = s.dims[n - 2];
            let w = s.dims[n - 1];
            if c > 0 && h > 0 && w > 0 {
                Some((c, h, w))
            } else {
                None
            }
        } else {
            None
        }
    };

    // Helper: resolve M for matmul from input_shapes[0].
    // M = total_elements / last_dim (where last_dim is K).
    let shape_matmul_m = |k_val: u32| -> Option<usize> {
        let s = input_shapes.first()?.as_ref()?;
        let total = s.total_elements();
        let k = k_val as usize;
        if k > 0 && total > 0 && total % k == 0 {
            Some(total / k)
        } else {
            None
        }
    };

    match kernel {
        TapeKernel::FusedFloatChain(chain) => {
            float_dispatch::dispatch_fused_chain_into(chain, inputs, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::Output => {
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchOk)
        }
        TapeKernel::LutView(view) | TapeKernel::PrimUnary(view) => {
            KvStore::apply_unary_into(view, inputs[0], out_buf);
            Ok(DispatchOk)
        }
        TapeKernel::LutView16(view) => {
            let input = inputs[0];
            let base = out_buf.len();
            out_buf.resize(base + input.len(), 0);
            out_buf[base..].copy_from_slice(input);
            let dst: &mut [u16] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
            view.apply_slice(dst);
            Ok(DispatchOk)
        }
        TapeKernel::PrimBinary(p) => {
            KvStore::apply_binary_into(*p, inputs[0], inputs[1], out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::RingPrimUnary { op, level } => {
            let input = inputs[0];
            let base = out_buf.len();
            // Dynamic precision: promote byte width if carry flux demands it.
            let compiled_bw = level.byte_width();
            let flux_bw = tape_ctx.flux.get().required_level().byte_width();
            let bw = compiled_bw.max(flux_bw) as usize;

            // Single loop for all quantum levels — parameterized by byte width.
            out_buf.resize(base + input.len(), 0);
            let dst = &mut out_buf[base..];
            for (c_in, c_out) in input.chunks_exact(bw).zip(dst.chunks_exact_mut(bw)) {
                let val = read_le_u64(c_in, bw);
                let result = op.apply_unary_u64(val, bw as u8);
                write_le_u64(c_out, result, bw);
            }

            // Accumulate curvature: XOR first byte of input/output, popcount.
            if !input.is_empty() && out_buf.len() > base {
                let curvature = (input[0] ^ out_buf[base]).count_ones() as u8;
                let mut flux = tape_ctx.flux.get();
                flux.accumulate_at(curvature, level.to_quantum());
                tape_ctx.flux.set(flux);
            }
            Ok(DispatchOk)
        }
        TapeKernel::RingPrimBinary { op, level } => {
            let (lhs, rhs) = (inputs[0], inputs[1]);
            if lhs.len() != rhs.len() {
                return Err(crate::error::ExecError::LengthMismatch {
                    expected: lhs.len(),
                    actual: rhs.len(),
                });
            }
            let base = out_buf.len();
            // Dynamic precision: promote byte width if carry flux demands it.
            let compiled_bw = level.byte_width();
            let flux_bw = tape_ctx.flux.get().required_level().byte_width();
            let bw = compiled_bw.max(flux_bw) as usize;

            // Single loop for all quantum levels — parameterized by byte width.
            out_buf.resize(base + lhs.len(), 0);
            let dst = &mut out_buf[base..];
            for i in (0..lhs.len()).step_by(bw) {
                let a = read_le_u64(&lhs[i..], bw);
                let b_val = read_le_u64(&rhs[i..], bw);
                let r = op.apply_binary_u64(a, b_val, bw as u8);
                write_le_u64(&mut dst[i..], r, bw);
            }

            // Accumulate curvature for binary op.
            if !lhs.is_empty() && out_buf.len() > base {
                let curvature = (lhs[0] ^ out_buf[base]).count_ones() as u8;
                let mut flux = tape_ctx.flux.get();
                flux.accumulate_at(curvature, level.to_quantum());
                tape_ctx.flux.set(flux);
            }
            Ok(DispatchOk)
        }
        TapeKernel::RingActivation { op, level } => {
            let input = inputs[0];
            let base = out_buf.len();
            let effective_level = {
                let flux = tape_ctx.flux.get();
                let flux_level = flux.required_level();
                if (flux_level as u8) > (*level as u8) {
                    flux_level
                } else {
                    *level
                }
            };
            out_buf.resize(base + input.len(), 0);
            let dst = &mut out_buf[base..];
            match effective_level {
                RingLevel::Q0 => {
                    for (i, &x) in input.iter().enumerate() {
                        dst[i] = op.apply::<hologram_ring::Q0>(x);
                    }
                }
                RingLevel::Q1 => {
                    for (c_in, c_out) in input.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
                        let val = u16::from_le_bytes([c_in[0], c_in[1]]);
                        let r = op.apply::<hologram_ring::Q1>(val);
                        c_out.copy_from_slice(&r.to_le_bytes());
                    }
                }
                RingLevel::Q2 => {
                    // Q2 (24-bit): treat as u32 masked to 24 bits
                    for (c_in, c_out) in input.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
                        let val = u32::from_le_bytes([c_in[0], c_in[1], c_in[2], 0]);
                        let r = op.apply::<hologram_ring::Q3>(val) & 0x00FF_FFFF;
                        let b = r.to_le_bytes();
                        c_out[0] = b[0];
                        c_out[1] = b[1];
                        c_out[2] = b[2];
                    }
                }
                RingLevel::Q3 => {
                    for (c_in, c_out) in input.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
                        let val = u32::from_le_bytes([c_in[0], c_in[1], c_in[2], c_in[3]]);
                        let r = op.apply::<hologram_ring::Q3>(val);
                        c_out.copy_from_slice(&r.to_le_bytes());
                    }
                }
            }
            // Curvature: O(1) — single XOR + popcount on first byte
            if !input.is_empty() && out_buf.len() > base {
                let curvature = (input[0] ^ out_buf[base]).count_ones() as u8;
                let mut flux = tape_ctx.flux.get();
                flux.accumulate(curvature, effective_level);
                tape_ctx.flux.set(flux);
            }
            Ok(DispatchOk)
        }
        TapeKernel::RingAccumulate { level } => {
            // Three inputs: acc, a, b. Element-wise: acc + a * b
            if inputs.len() < 3 {
                return Err(crate::error::ExecError::UnsupportedOp(
                    "RingAccumulate requires 3 inputs".into(),
                ));
            }
            let (acc, a, b) = (inputs[0], inputs[1], inputs[2]);
            let base = out_buf.len();
            out_buf.resize(base + acc.len(), 0);
            let dst = &mut out_buf[base..];
            match level {
                RingLevel::Q0 => {
                    for i in 0..acc.len() {
                        dst[i] = hologram_ring::accumulate(acc[i], a[i], b[i]);
                    }
                }
                RingLevel::Q3 => {
                    for i in (0..acc.len()).step_by(4) {
                        let va = u32::from_le_bytes([acc[i], acc[i + 1], acc[i + 2], acc[i + 3]]);
                        let vb = u32::from_le_bytes([a[i], a[i + 1], a[i + 2], a[i + 3]]);
                        let vc = u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
                        let r = hologram_ring::accumulate(va, vb, vc);
                        dst[i..i + 4].copy_from_slice(&r.to_le_bytes());
                    }
                }
                RingLevel::Q1 => {
                    for i in (0..acc.len()).step_by(2) {
                        if i + 1 < acc.len() {
                            let va = u16::from_le_bytes([acc[i], acc[i + 1]]);
                            let vb = u16::from_le_bytes([a[i], a[i + 1]]);
                            let vc = u16::from_le_bytes([b[i], b[i + 1]]);
                            let r = hologram_ring::accumulate(va, vb, vc);
                            dst[i..i + 2].copy_from_slice(&r.to_le_bytes());
                        }
                    }
                }
                RingLevel::Q2 => {
                    for i in (0..acc.len()).step_by(3) {
                        if i + 2 < acc.len() {
                            let va = u32::from_le_bytes([acc[i], acc[i + 1], acc[i + 2], 0])
                                & 0x00FF_FFFF;
                            let vb =
                                u32::from_le_bytes([a[i], a[i + 1], a[i + 2], 0]) & 0x00FF_FFFF;
                            let vc =
                                u32::from_le_bytes([b[i], b[i + 1], b[i + 2], 0]) & 0x00FF_FFFF;
                            let r = hologram_ring::accumulate(va, vb, vc) & 0x00FF_FFFF;
                            let bytes = r.to_le_bytes();
                            dst[i] = bytes[0];
                            dst[i + 1] = bytes[1];
                            dst[i + 2] = bytes[2];
                        }
                    }
                }
            }
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut4(cid) => {
            dispatch_lut_gemm_4(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut8(cid) => {
            dispatch_lut_gemm_8(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut4Activation(cid, activation) => {
            dispatch_lut_gemm_4(inputs, *cid, tape_ctx, out_buf)?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut8Activation(cid, activation) => {
            dispatch_lut_gemm_8(inputs, *cid, tape_ctx, out_buf)?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut16(cid) => {
            dispatch_lut_gemm_16(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut2(cid) => {
            dispatch_lut_gemm_2(inputs, *cid, tape_ctx, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::MatMulLut2Activation(cid, activation) => {
            dispatch_lut_gemm_2(inputs, *cid, tape_ctx, out_buf)?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
            heads_first,
        } => {
            dispatch_kv_write(
                inputs,
                KvWriteParams::new(*layer, *n_kv_heads, *head_dim)
                    .with_is_key(*is_key)
                    .with_heads_first(*heads_first),
                tape_ctx,
                out_buf,
            )?;
            Ok(DispatchOk)
        }
        TapeKernel::KvRead {
            layer,
            n_kv_heads,
            head_dim,
            heads_first,
        } => {
            dispatch_kv_read(
                *layer,
                *n_kv_heads,
                *head_dim,
                *heads_first,
                tape_ctx,
                out_buf,
            )?;
            Ok(DispatchOk)
        }

        // ── Trivial unary/binary float ops — delegated to float_dispatch ──
        //
        // All simple element-wise ops are converted to FloatOp and dispatched
        // through `dispatch_float_into`, which provides monomorphized closures
        // and SIMD autovectorization for common ops.
        TapeKernel::InlineRelu
        | TapeKernel::InlineNeg
        | TapeKernel::InlineSigmoid
        | TapeKernel::InlineSilu
        | TapeKernel::InlineTanh
        | TapeKernel::InlineGelu
        | TapeKernel::InlineExp
        | TapeKernel::InlineAbs
        | TapeKernel::InlineReciprocal
        | TapeKernel::InlineLog
        | TapeKernel::InlineSqrt
        | TapeKernel::InlineCos
        | TapeKernel::InlineSin
        | TapeKernel::InlineSign
        | TapeKernel::InlineFloor
        | TapeKernel::InlineCeil
        | TapeKernel::InlineRound
        | TapeKernel::InlineErf
        | TapeKernel::InlineAdd
        | TapeKernel::InlineMul
        | TapeKernel::InlineSub
        | TapeKernel::InlineDiv
        | TapeKernel::InlineMin
        | TapeKernel::InlineMax
        | TapeKernel::InlinePow
        | TapeKernel::InlineMod
        | TapeKernel::InlineIsNaN
        | TapeKernel::InlineNot
        | TapeKernel::InlineAnd
        | TapeKernel::InlineOr
        | TapeKernel::InlineXor
        | TapeKernel::InlineEqual
        | TapeKernel::InlineLess
        | TapeKernel::InlineLessOrEqual
        | TapeKernel::InlineGreater
        | TapeKernel::InlineGreaterOrEqual
        | TapeKernel::InlineFusedSwiGLU => {
            let float_op = tape_kernel_as_float_op(kernel);
            float_dispatch::dispatch_float_into(&float_op, inputs, tape_ctx.ctx.as_ref(), out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineClip { min, max } => {
            let float_op = FloatOp::Clip {
                min: *min,
                max: *max,
            };
            float_dispatch::dispatch_float_into(&float_op, inputs, tape_ctx.ctx.as_ref(), out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineLayerNorm { size, epsilon } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim_with_weight(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                    inputs.get(1).map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_layer_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineAddRmsNorm { size, epsilon } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_add_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineLogSoftmax { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_log_softmax_into(inputs, actual, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineAttention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            sparse_v,
        } => {
            let result = crate::float_dispatch::attention::dispatch_attention(
                inputs,
                crate::float_dispatch::attention::AttentionParams::new(
                    *head_dim as usize,
                    *num_q_heads as usize,
                    *num_kv_heads as usize,
                )
                .with_scale(f32::from_bits(*scale))
                .with_causal(*causal)
                .with_heads_first(*heads_first)
                .with_sparse_v(*sparse_v),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineRoPE { dim, base, n_heads } => {
            let start_pos = tape_ctx
                .ctx
                .as_ref()
                .map(|c| c.position_offset as usize)
                .unwrap_or(0);
            let result = crate::float_dispatch::attention::dispatch_rope(
                inputs,
                *dim as usize,
                f32::from_bits(*base),
                *n_heads as usize,
                start_pos,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineGather { dim, dtype } => {
            let result = crate::float_dispatch::gather_concat::dispatch_gather(
                inputs,
                *dim as usize,
                *dtype,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineConcat {
            size_a,
            size_b,
            dtype,
        } => {
            let result = crate::float_dispatch::gather_concat::dispatch_concat(
                inputs,
                *size_a as usize,
                *size_b as usize,
                *dtype,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineTranspose {
            perm,
            input_shape,
            ndim,
        } => {
            let n = *ndim as usize;
            let compiled_shape: Vec<usize> = input_shape[..n].iter().map(|&d| d as usize).collect();
            let perm_slice: &[u8] = &perm[..n];

            // Verify baked shape matches actual input size. If the input
            // is a different size (e.g., KV cache produced a runtime-sized
            // tensor), infer the actual shape by scaling the variable dim.
            let input_elems = inputs[0].len() / 4; // f32 elements
            let compiled_elems: usize = compiled_shape.iter().product();
            let shape = if compiled_elems > 0 && compiled_elems == input_elems {
                compiled_shape
            } else if compiled_elems > 0 && input_elems > 0 {
                // Find the dim that changed (variable-length dim like seq)
                // and scale it to match the actual input size.
                let mut adjusted = compiled_shape.clone();
                let ratio = input_elems as f64 / compiled_elems as f64;
                // Find the dim most likely to be variable (not head_dim, not n_heads).
                // Heuristic: the dim that, when scaled by ratio, gives an integer.
                for i in 0..adjusted.len() {
                    let scaled = (adjusted[i] as f64 * ratio).round() as usize;
                    let check: usize = adjusted
                        .iter()
                        .enumerate()
                        .map(|(j, &d)| if j == i { scaled } else { d })
                        .product();
                    if check == input_elems {
                        adjusted[i] = scaled;
                        break;
                    }
                }
                adjusted
            } else if input_elems > 0 {
                // compiled_elems == 0: one or more dims are 0-sentinels.
                // Resolve the sentinel dims from the buffer size and the
                // known (non-zero) dims. For a shape like [32, 0] with
                // input_elems=160, the 0-dim resolves to 160/32=5.
                let mut resolved = compiled_shape.clone();
                let nonzero_product: usize = resolved
                    .iter()
                    .filter(|&&d| d > 0)
                    .product::<usize>()
                    .max(1);
                let zero_count = resolved.iter().filter(|&&d| d == 0).count();
                if zero_count == 1
                    && nonzero_product > 0
                    && input_elems.is_multiple_of(nonzero_product)
                {
                    let inferred = input_elems / nonzero_product;
                    for d in &mut resolved {
                        if *d == 0 {
                            *d = inferred;
                        }
                    }
                } else if let Some(Some(shape)) = input_shapes.first() {
                    // Fallback: use tracked runtime shape from arena.
                    resolved = shape.dims.to_vec();
                } else {
                    // Last resort: passthrough.
                    out_buf.extend_from_slice(inputs[0]);
                    return Ok(DispatchOk);
                }
                resolved
            } else {
                // Empty input — passthrough.
                out_buf.extend_from_slice(inputs[0]);
                return Ok(DispatchOk);
            };

            let (result, _out_shape) =
                crate::float_dispatch::dispatch_transpose(inputs[0], perm_slice, &shape)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::Passthrough => {
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineReshape => {
            // Zero-copy: bytes unchanged, only metadata changes.
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineExpand { ndim, target_shape } => {
            let result = crate::float_dispatch::dispatch_float(
                &FloatOp::Expand {
                    ndim: *ndim,
                    target_shape: *target_shape,
                },
                inputs,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }

        // ── Complex ops (call existing handlers) ──────────────────────────
        TapeKernel::InlineGemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => {
            #[allow(unused_variables)]
            let baked_k = *k as usize;
            #[allow(unused_variables)]
            let baked_n = *n as usize;
            let alpha_f = hologram_core::op::bits_to_f32(*alpha);
            let beta_f = hologram_core::op::bits_to_f32(*beta);

            // Fast path: direct BLAS for standard Gemm (alpha=1, beta=0, no quant).
            // Handles both trans_b=false and trans_b=true (common ONNX pattern).
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            if !*trans_a
                && *quant_b == 0
                && alpha_f == 1.0
                && beta_f == 0.0
                && baked_k > 0
                && baked_n > 0
            {
                // B layout: [k, n] if !trans_b, [n, k] if trans_b.
                let b_expected = baked_k * baked_n * 4;
                let a_k = baked_k;
                if inputs[0].len().is_multiple_of(a_k * 4)
                    && inputs.get(1).map(|b| b.len()).unwrap_or(0) >= b_expected
                {
                    let actual_m = inputs[0].len() / (a_k * 4);
                    if actual_m > 0 {
                        let a: &[f32] = bytemuck::cast_slice(inputs[0]);
                        let b: &[f32] = bytemuck::cast_slice(&inputs[1][..b_expected]);
                        let out = crate::float_dispatch::helpers::alloc_f32_in(
                            out_buf,
                            actual_m * baked_n,
                        );
                        crate::float_dispatch::matmul::blas::sgemm_full(
                            crate::float_dispatch::matmul::GemmParams {
                                m: actual_m,
                                n: baked_n,
                                k: baked_k,
                                alpha: 1.0,
                                beta: 0.0,
                                trans_a: false,
                                trans_b: *trans_b,
                            },
                            a,
                            b,
                            out,
                        );
                        return Ok(DispatchOk);
                    }
                }
            }

            // General path.
            let (actual_m, actual_k, actual_n) = if let Some(am) = shape_matmul_m(*k) {
                (am, *k as usize, *n as usize)
            } else {
                shape_resolve::resolve_matmul_dims(
                    *m,
                    *k,
                    *n,
                    input_metas.first().and_then(|m| m.as_ref()),
                    input_metas.get(1).and_then(|m| m.as_ref()),
                    inputs[0].len(),
                    inputs.get(1).map(|b| b.len()).unwrap_or(0),
                )
            };
            let result = float_dispatch::matmul::dispatch_gemm(
                inputs,
                float_dispatch::matmul::GemmParams {
                    m: actual_m,
                    n: actual_n,
                    k: actual_k,
                    alpha: alpha_f,
                    beta: beta_f,
                    trans_a: *trans_a,
                    trans_b: *trans_b,
                },
                *quant_b,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReduceSum { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_sum,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReduceMean { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_mean,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReduceMax { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_max,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReduceMin { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                actual,
                float_dispatch::reduce::reduce_min,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReduceProd { size } => {
            let result = float_dispatch::reduce::dispatch_reduce(
                inputs,
                *size as usize,
                float_dispatch::reduce::reduce_prod,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineArgMax { axis, keepdims } => {
            let input = inputs.first().copied().unwrap_or(&[]);
            let floats = safe_cast_f32(input);
            // Resolve axis size from input shape, meta, or compiled value.
            let axis_size = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *axis,
                    input_metas.first().and_then(|m| m.as_ref()),
                    input.len(),
                )
            });
            if axis_size == 0 || floats.is_empty() {
                return Ok(DispatchOk);
            }
            let n_rows = floats.len() / axis_size;
            let argmax_row = |row: &[f32]| -> i64 {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as i64)
                    .unwrap_or(0)
            };
            let indices: Vec<i64> = {
                #[cfg(feature = "parallel")]
                {
                    if n_rows > 64 {
                        use rayon::prelude::*;
                        floats.par_chunks(axis_size).map(argmax_row).collect()
                    } else {
                        floats.chunks(axis_size).map(argmax_row).collect()
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = n_rows;
                    floats.chunks(axis_size).map(argmax_row).collect()
                }
            };
            let result_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
            out_buf.extend_from_slice(&result_bytes);
            let _ = keepdims; // Shape adjustment handled by output meta.
            Ok(DispatchOk)
        }
        TapeKernel::InlineCast { from, to } => {
            let result = float_dispatch::cast::dispatch_cast(inputs, *from, *to)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineEmbed { dim, quant } => {
            let result = float_dispatch::cast::dispatch_embed(inputs, *dim as usize, *quant)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineWhere => {
            let result = float_dispatch::gather_concat::dispatch_where(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineRange => {
            let result = float_dispatch::gather_concat::dispatch_range(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineShape { dtype, start, end } => {
            let result =
                float_dispatch::gather_concat::dispatch_shape(inputs, *dtype, *start, *end)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineSlice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => {
            let result = float_dispatch::dispatch_float_ctx(
                &FloatOp::Slice {
                    axis_from_end: *axis_from_end,
                    start: *start,
                    end: *end,
                    axis_size: *axis_size,
                },
                inputs,
                tape_ctx.ctx.as_ref(),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineGatherND => {
            // GatherND: pass-through (same as Reshape — data unchanged).
            if let Some(b) = inputs.first() {
                out_buf.extend_from_slice(b);
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineDequantize => {
            let result = float_dispatch::cast::dispatch_dequantize(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineConv2d {
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
            let (actual_h, actual_w) = shape_spatial_hw(0).unwrap_or_else(|| {
                shape_resolve::resolve_spatial_dims(
                    *input_h,
                    *input_w,
                    input_metas.first().and_then(|m| m.as_ref()),
                )
            });

            let kh = *kernel_h as usize;
            let kw = *kernel_w as usize;
            {
                let result = float_dispatch::conv::dispatch_conv2d_direct(
                    inputs,
                    float_dispatch::conv::Conv2dAttrs::new(kh, kw)
                        .with_stride(*stride_h as usize, *stride_w as usize)
                        .with_padding(*pad_h as usize, *pad_w as usize)
                        .with_dilation(*dilation_h as usize, *dilation_w as usize)
                        .with_group(*group as usize),
                    actual_h,
                    actual_w,
                )?;
                out_buf.extend_from_slice(&result);
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineConv2dActivation {
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
            activation,
        }
        | TapeKernel::InlineConv2dBiasActivation {
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
            activation,
        } => {
            let (actual_h, actual_w) = shape_spatial_hw(0).unwrap_or_else(|| {
                shape_resolve::resolve_spatial_dims(
                    *input_h,
                    *input_w,
                    input_metas.first().and_then(|m| m.as_ref()),
                )
            });
            let result = float_dispatch::conv::dispatch_conv2d_direct(
                inputs,
                float_dispatch::conv::Conv2dAttrs::new(*kernel_h as usize, *kernel_w as usize)
                    .with_stride(*stride_h as usize, *stride_w as usize)
                    .with_padding(*pad_h as usize, *pad_w as usize)
                    .with_dilation(*dilation_h as usize, *dilation_w as usize)
                    .with_group(*group as usize),
                actual_h,
                actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            // Epilogue: apply activation to cache-hot conv output.
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::InlineConv2dLut4 {
            cid,
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
            let (actual_h, actual_w) = shape_spatial_hw(0).unwrap_or_else(|| {
                shape_resolve::resolve_spatial_dims(
                    *input_h,
                    *input_w,
                    input_metas.first().and_then(|m| m.as_ref()),
                )
            });
            let result = float_dispatch::conv::dispatch_conv2d_lut4(
                inputs,
                *cid,
                tape_ctx,
                float_dispatch::conv::Conv2dAttrs::new(*kernel_h as usize, *kernel_w as usize)
                    .with_stride(*stride_h as usize, *stride_w as usize)
                    .with_padding(*pad_h as usize, *pad_w as usize)
                    .with_dilation(*dilation_h as usize, *dilation_w as usize)
                    .with_group(*group as usize),
                actual_h,
                actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineConvTranspose {
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
        } => {
            let (actual_h, actual_w) = shape_spatial_hw(0).unwrap_or_else(|| {
                shape_resolve::resolve_spatial_dims(
                    *input_h,
                    *input_w,
                    input_metas.first().and_then(|m| m.as_ref()),
                )
            });
            let result = float_dispatch::conv::dispatch_conv_transpose(
                inputs,
                float_dispatch::conv::Conv2dAttrs::new(*kernel_h as usize, *kernel_w as usize)
                    .with_stride(*stride_h as usize, *stride_w as usize)
                    .with_padding(*pad_h as usize, *pad_w as usize)
                    .with_dilation(*dilation_h as usize, *dilation_w as usize)
                    .with_group(*group as usize),
                float_dispatch::conv::ConvTransposeOutputPad::new()
                    .with_hw(*output_pad_h as usize, *output_pad_w as usize),
                actual_h,
                actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineMaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let result = float_dispatch::pool::dispatch_max_pool_2d(
                inputs,
                float_dispatch::pool::Pool2dAttrs::new(*kernel_h as usize, *kernel_w as usize)
                    .with_stride(*stride_h as usize, *stride_w as usize)
                    .with_padding(*pad_h as usize, *pad_w as usize),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineAvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => {
            let result = float_dispatch::pool::dispatch_avg_pool_2d(
                inputs,
                float_dispatch::pool::Pool2dAttrs::new(*kernel_h as usize, *kernel_w as usize)
                    .with_stride(*stride_h as usize, *stride_w as usize)
                    .with_padding(*pad_h as usize, *pad_w as usize),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineGlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => {
            let (actual_c, actual_h, actual_w) = shape_chw(0).unwrap_or_else(|| {
                shape_resolve::resolve_global_avg_pool_dims(
                    *channels,
                    *spatial_h,
                    *spatial_w,
                    input_metas.first().and_then(|m| m.as_ref()),
                )
            });
            let result = float_dispatch::pool::dispatch_global_avg_pool_direct(
                inputs, actual_c, actual_h, actual_w,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineResize { mode } => {
            // Prefer input_shapes, fall back to input_metas.
            let input_shape: Option<Vec<usize>> = input_shapes
                .first()
                .and_then(|s| s.as_ref())
                .map(|s| s.dims.to_vec())
                .or_else(|| {
                    input_metas
                        .first()
                        .and_then(|m| m.as_ref())
                        .map(|m| m.shape().iter().map(|&d| d as usize).collect())
                });
            let result = float_dispatch::spatial::dispatch_resize_with_shape(
                inputs,
                *mode,
                input_shape.as_deref(),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlinePad { mode } => {
            let result = float_dispatch::spatial::dispatch_pad(inputs, *mode)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineInstanceNorm { size, epsilon } => {
            // InstanceNorm normalizes across ALL spatial dims (H×W), not just
            // the last dim. Compute spatial size from input_shapes, TensorMeta,
            // or fall back to compiled size (which should be H×W from lowering).
            let actual = input_shapes
                .first()
                .and_then(|s| s.as_ref())
                .and_then(|s| {
                    if s.ndim() >= 3 {
                        let spatial: usize = s.dims[2..].iter().product();
                        if spatial > 0 {
                            Some(spatial)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    input_metas
                        .first()
                        .and_then(|m| m.as_ref())
                        .and_then(|meta| {
                            let shape = meta.shape();
                            if shape.len() >= 3 {
                                let spatial: usize =
                                    shape[2..].iter().map(|&d| d as usize).product();
                                if spatial > 0 {
                                    Some(spatial)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                })
                .unwrap_or_else(|| {
                    if *size > 0 {
                        *size as usize
                    } else {
                        // Last resort: total elements / n_channels.
                        let n_floats = inputs.first().map(|b| b.len() / 4).unwrap_or(0);
                        let n_channels = inputs.get(1).map(|b| b.len() / 4).unwrap_or(1).max(1);
                        n_floats / n_channels
                    }
                });
            let result = float_dispatch::norm::dispatch_instance_norm(
                inputs,
                actual,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineGroupNorm {
            num_groups,
            epsilon,
        } => {
            float_dispatch::norm::dispatch_group_norm_into(
                inputs,
                *num_groups as usize,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchOk)
        }

        // ── Fused norm + activation (epilogue fusion) ────────────────
        TapeKernel::InlineRmsNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::InlineLayerNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                shape_resolve::resolve_last_dim_with_weight(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                    inputs.get(1).map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_layer_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::InlineGroupNormActivation {
            num_groups,
            epsilon,
            activation,
        } => {
            float_dispatch::norm::dispatch_group_norm_activation_into(
                inputs,
                *num_groups as usize,
                f32::from_bits(*epsilon),
                activation,
                out_buf,
            )?;
            Ok(DispatchOk)
        }

        TapeKernel::InlineAddRmsNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let result = float_dispatch::norm::dispatch_add_rms_norm(
                inputs,
                *size as usize,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }
        TapeKernel::InlineInstanceNormActivation {
            size,
            epsilon,
            activation,
        } => {
            let result = float_dispatch::norm::dispatch_instance_norm(
                inputs,
                *size as usize,
                f32::from_bits(*epsilon),
            )?;
            out_buf.extend_from_slice(&result);
            apply_activation_to_out_buf(out_buf, activation);
            Ok(DispatchOk)
        }

        TapeKernel::InlineLRN {
            size,
            alpha,
            beta,
            bias,
        } => {
            let result = float_dispatch::norm::dispatch_lrn(
                inputs,
                *size as usize,
                hologram_core::op::bits_to_f32(*alpha),
                hologram_core::op::bits_to_f32(*beta),
                hologram_core::op::bits_to_f32(*bias),
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineTopK { axis, largest } => {
            let result = float_dispatch::misc::dispatch_top_k(inputs, *axis as usize, *largest)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineScatterND => {
            let result = float_dispatch::misc::dispatch_scatter_nd(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineCumSum { axis } => {
            let result = float_dispatch::misc::dispatch_cumsum(inputs, *axis as usize)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineNonZero => {
            let result = float_dispatch::misc::dispatch_nonzero(inputs)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineCompress { axis } => {
            let result = float_dispatch::misc::dispatch_compress(inputs, *axis as usize)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
        TapeKernel::InlineReverseSequence {
            batch_axis,
            time_axis,
        } => {
            let result = float_dispatch::misc::dispatch_reverse_sequence(
                inputs,
                *batch_axis as usize,
                *time_axis as usize,
            )?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }

        // ── Inline custom ops ─────────────────────────────────────────────
        TapeKernel::InlineMatMul { m, k, n } => {
            // Used by Accelerate BLAS fast path (cfg-gated on macOS).
            let baked_k = *k as usize;
            let baked_n = *n as usize;
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            let _ = (baked_k, baked_n);

            // ── CPU fast path (Accelerate BLAS) ───────────────────────
            // The BLAS block has side effects (writes to out_buf) despite
            // returning bool — clippy's needless_bool doesn't see them.
            #[allow(clippy::needless_bool)]
            let used_cpu_blas = {
                #[cfg(all(feature = "accelerate", target_os = "macos"))]
                {
                    if baked_k > 0
                        && baked_n > 0
                        && inputs[0].len().is_multiple_of(baked_k * 4)
                        && inputs[1].len() == baked_k * baked_n * 4
                    {
                        let actual_m = inputs[0].len() / (baked_k * 4);
                        if actual_m > 0 {
                            let a: &[f32] = bytemuck::cast_slice(inputs[0]);
                            let b: &[f32] =
                                bytemuck::cast_slice(&inputs[1][..baked_k * baked_n * 4]);
                            let out = crate::float_dispatch::helpers::alloc_f32_in(
                                out_buf,
                                actual_m * baked_n,
                            );
                            crate::float_dispatch::matmul::blas::sgemm(
                                actual_m, baked_n, baked_k, a, b, out,
                            );
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                {
                    false
                }
            };

            if !used_cpu_blas {
                // General path with dim resolution.
                let (am, ak, an) = if let Some(sm) = shape_matmul_m(*k) {
                    (sm, *k as usize, *n as usize)
                } else {
                    let meta_dims = shape_resolve::resolve_matmul_dims(
                        *m,
                        *k,
                        *n,
                        input_metas.first().and_then(|m| m.as_ref()),
                        input_metas.get(1).and_then(|m| m.as_ref()),
                        inputs[0].len(),
                        inputs[1].len(),
                    );
                    let a_floats = inputs[0].len() / 4;
                    let b_floats = inputs[1].len() / 4;
                    if meta_dims.1 > 0
                        && a_floats > 0
                        && b_floats > 0
                        && a_floats.is_multiple_of(meta_dims.1)
                        && b_floats.is_multiple_of(meta_dims.1)
                    {
                        meta_dims
                    } else {
                        crate::float_dispatch::matmul::infer_matmul_dims(
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            a_floats,
                            b_floats,
                        )
                    }
                };
                crate::float_dispatch::matmul::dispatch_matmul_into(inputs, am, ak, an, out_buf)?;
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineMatMulBiasActivation {
            m,
            k,
            n,
            activation,
        } => {
            // inputs: [activation_tensor, weight, bias] — all zero-copy from arena.
            let bias: &[f32] = bytemuck::try_cast_slice(inputs[2]).map_err(|_| {
                crate::error::ExecError::UnsupportedOp("bias not f32-aligned".into())
            })?;
            // Resolve runtime dimensions from input shapes, N-D input metas, or compiled values.
            let (actual_m, actual_k, actual_n) = if let Some(sm) = shape_matmul_m(*k) {
                (sm, *k as usize, *n as usize)
            } else {
                let meta_dims = shape_resolve::resolve_matmul_dims(
                    *m,
                    *k,
                    *n,
                    input_metas.first().and_then(|m| m.as_ref()),
                    input_metas.get(1).and_then(|m| m.as_ref()),
                    inputs[0].len(),
                    inputs[1].len(),
                );
                let a_floats = inputs[0].len() / 4;
                let b_floats = inputs[1].len() / 4;
                if meta_dims.1 > 0
                    && a_floats > 0
                    && b_floats > 0
                    && a_floats.is_multiple_of(meta_dims.1)
                    && b_floats.is_multiple_of(meta_dims.1)
                {
                    meta_dims
                } else {
                    crate::float_dispatch::matmul::infer_matmul_dims(
                        *m as usize,
                        *k as usize,
                        *n as usize,
                        a_floats,
                        b_floats,
                    )
                }
            };

            crate::float_dispatch::matmul::dispatch_matmul_bias_activation_into(
                &inputs[..2],
                actual_m,
                actual_k,
                actual_n,
                bias,
                activation,
                out_buf,
            )?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineMatMulActivation {
            m,
            k,
            n,
            activation,
        } => {
            let (actual_m, actual_k, actual_n) = if let Some(sm) = shape_matmul_m(*k) {
                (sm, *k as usize, *n as usize)
            } else {
                let meta_dims = shape_resolve::resolve_matmul_dims(
                    *m,
                    *k,
                    *n,
                    input_metas.first().and_then(|m| m.as_ref()),
                    input_metas.get(1).and_then(|m| m.as_ref()),
                    inputs[0].len(),
                    inputs[1].len(),
                );
                let a_floats = inputs[0].len() / 4;
                let b_floats = inputs[1].len() / 4;
                if meta_dims.1 > 0
                    && a_floats > 0
                    && b_floats > 0
                    && a_floats.is_multiple_of(meta_dims.1)
                    && b_floats.is_multiple_of(meta_dims.1)
                {
                    meta_dims
                } else {
                    crate::float_dispatch::matmul::infer_matmul_dims(
                        *m as usize,
                        *k as usize,
                        *n as usize,
                        a_floats,
                        b_floats,
                    )
                }
            };
            crate::float_dispatch::matmul::dispatch_matmul_activation_into(
                inputs, actual_m, actual_k, actual_n, activation, out_buf,
            )?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineSoftmax { size } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                crate::shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_softmax_into(inputs, actual, out_buf)?;
            Ok(DispatchOk)
        }
        TapeKernel::InlineRmsNorm { size, epsilon } => {
            let actual = shape_last_dim(0).unwrap_or_else(|| {
                crate::shape_resolve::resolve_last_dim(
                    *size,
                    input_metas.first().and_then(|m| m.as_ref()),
                    inputs.first().map(|b| b.len()).unwrap_or(0),
                )
            });
            crate::float_dispatch::norm::dispatch_rms_norm_into(
                inputs,
                actual,
                f32::from_bits(*epsilon),
                out_buf,
            )?;
            Ok(DispatchOk)
        }
        // ── Deep decode fusions (Plan 054) ──────────────────────────────
        //
        // At M=1 (decode), these kernels normalize into a reusable Vec
        // (sized once, reused across calls via thread-local or caller),
        // then project via BLAS sgemm. The norm intermediate never
        // enters the arena — saving one allocation per fused dispatch.
        TapeKernel::InlineNormProjectionGemv {
            norm_size,
            epsilon,
            k,
            n_total,
        } => {
            // Fused: RmsNorm(x, weight) → MatMul(normed, proj_weight)
            // inputs: [x, norm_weight, proj_weight]
            let k_val = *k as usize;
            let m_val = inputs[0].len() / 4 / k_val;

            if m_val == 1 {
                // M=1 fast path: RmsNorm into pre-sized Vec (no arena alloc).
                let x = safe_cast_f32(inputs[0]);
                let weight = safe_cast_f32(inputs[1]);
                let mut normed_f32 = x.into_owned();
                float_dispatch::norm::rms_norm_in_place(
                    &mut normed_f32,
                    &weight,
                    *norm_size as usize,
                    f32::from_bits(*epsilon),
                );
                let normed_bytes: &[u8] = bytemuck::cast_slice(&normed_f32);
                float_dispatch::matmul::dispatch_matmul_into(
                    &[normed_bytes, inputs[2]],
                    1,
                    k_val,
                    *n_total as usize,
                    out_buf,
                )?;
            } else {
                // M>1 fallback: decompose to separate ops.
                let normed = float_dispatch::norm::dispatch_rms_norm(
                    &[inputs[0], inputs[1]],
                    *norm_size as usize,
                    f32::from_bits(*epsilon),
                )?;
                float_dispatch::matmul::dispatch_matmul_into(
                    &[&normed, inputs[2]],
                    m_val,
                    k_val,
                    *n_total as usize,
                    out_buf,
                )?;
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineAddNormProjectionGemv {
            norm_size,
            epsilon,
            k,
            n_total,
        } => {
            // Fused: Add(x, residual) → RmsNorm(sum, weight) → MatMul(normed, proj_weight)
            // inputs: [x, residual, norm_weight, proj_weight]
            let k_val = *k as usize;
            let m_val = inputs[0].len() / 4 / k_val;

            if m_val == 1 {
                // M=1 fast path: Add + RmsNorm in-place, no arena alloc.
                let x = safe_cast_f32(inputs[0]);
                let residual = safe_cast_f32(inputs[1]);
                let weight = safe_cast_f32(inputs[2]);
                let mut normed_f32: Vec<f32> = x
                    .iter()
                    .zip(residual.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                float_dispatch::norm::rms_norm_in_place(
                    &mut normed_f32,
                    &weight,
                    *norm_size as usize,
                    f32::from_bits(*epsilon),
                );
                let normed_bytes: &[u8] = bytemuck::cast_slice(&normed_f32);
                float_dispatch::matmul::dispatch_matmul_into(
                    &[normed_bytes, inputs[3]],
                    1,
                    k_val,
                    *n_total as usize,
                    out_buf,
                )?;
            } else {
                // M>1 fallback: decompose to separate ops.
                let normed = float_dispatch::norm::dispatch_add_rms_norm(
                    &[inputs[0], inputs[1], inputs[2]],
                    *norm_size as usize,
                    f32::from_bits(*epsilon),
                )?;
                float_dispatch::matmul::dispatch_matmul_into(
                    &[&normed, inputs[3]],
                    m_val,
                    k_val,
                    *n_total as usize,
                    out_buf,
                )?;
            }
            Ok(DispatchOk)
        }
        TapeKernel::InlineSwiGluProjectionGemv { k, n } => {
            // Fused: SwiGLU(gate, up) → MatMul(activated, W_down)
            // inputs: [gate, up, down_weight]
            let k_val = *k as usize;
            let n_val = *n as usize;
            let m_val = inputs[0].len() / 4 / k_val;

            // SwiGLU activation: silu(gate) * up — computed into Vec, not arena.
            let gate = safe_cast_f32(inputs[0]);
            let up = safe_cast_f32(inputs[1]);
            let activated_f32: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| {
                    let sig = 1.0 / (1.0 + (-g).exp());
                    g * sig * u
                })
                .collect();
            let activated_bytes: &[u8] = bytemuck::cast_slice(&activated_f32);
            // Prevent the compiler from dropping activated_f32 before matmul uses it.
            float_dispatch::matmul::dispatch_matmul_into(
                &[activated_bytes, inputs[2]],
                m_val,
                k_val,
                n_val,
                out_buf,
            )?;
            drop(activated_f32);
            Ok(DispatchOk)
        }

        TapeKernel::Custom(handler) => {
            let result = handler(inputs, tape_ctx.constants)?;
            out_buf.extend_from_slice(&result);
            Ok(DispatchOk)
        }
    }
}

/// Cast `&[u8]` to `&[f32]`, handling misaligned buffers gracefully.
/// Returns a `Cow` — borrowed when aligned, owned copy when not.
#[inline(always)]
fn safe_cast_f32(bytes: &[u8]) -> std::borrow::Cow<'_, [f32]> {
    match bytemuck::try_cast_slice(bytes) {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => {
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            std::borrow::Cow::Owned(floats)
        }
    }
}

/// Binary elementwise with broadcasting. Fast paths avoid per-element modulo.
#[cfg(test)]
#[inline(always)]
pub(crate) fn binary_broadcast(a: &[f32], b: &[f32], dst: &mut [f32], f: impl Fn(f32, f32) -> f32) {
    if a.len() == b.len() {
        for (d, (&x, &y)) in dst.iter_mut().zip(a.iter().zip(b.iter())) {
            *d = f(x, y);
        }
    } else if b.len() == 1 {
        let bv = b[0];
        for (d, &x) in dst.iter_mut().zip(a.iter()) {
            *d = f(x, bv);
        }
    } else if a.len() == 1 {
        let av = a[0];
        for (d, &y) in dst.iter_mut().zip(b.iter()) {
            *d = f(av, y);
        }
    } else if !a.is_empty() && !b.is_empty() {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = f(a[i % a.len()], b[i % b.len()]);
        }
    }
    // If either input is empty, dst is left as zeros (from allocation).
}

/// Convert a trivial `TapeKernel` variant to its corresponding `FloatOp`.
///
/// Only handles the simple unary/binary/comparison/logical ops that have a
/// direct 1:1 mapping. Called from the consolidated dispatch arm.
///
/// # Panics
/// Panics if called with a non-trivial kernel variant (should never happen
/// since the match arm only routes the listed variants here).
fn tape_kernel_as_float_op(kernel: &TapeKernel) -> FloatOp {
    match kernel {
        // Unary activations / math
        TapeKernel::InlineRelu => FloatOp::Relu,
        TapeKernel::InlineNeg => FloatOp::Neg,
        TapeKernel::InlineSigmoid => FloatOp::Sigmoid,
        TapeKernel::InlineSilu => FloatOp::Silu,
        TapeKernel::InlineTanh => FloatOp::Tanh,
        TapeKernel::InlineGelu => FloatOp::Gelu,
        TapeKernel::InlineExp => FloatOp::Exp,
        TapeKernel::InlineAbs => FloatOp::Abs,
        TapeKernel::InlineReciprocal => FloatOp::Reciprocal,
        TapeKernel::InlineLog => FloatOp::Log,
        TapeKernel::InlineSqrt => FloatOp::Sqrt,
        TapeKernel::InlineCos => FloatOp::Cos,
        TapeKernel::InlineSin => FloatOp::Sin,
        TapeKernel::InlineSign => FloatOp::Sign,
        TapeKernel::InlineFloor => FloatOp::Floor,
        TapeKernel::InlineCeil => FloatOp::Ceil,
        TapeKernel::InlineRound => FloatOp::Round,
        TapeKernel::InlineErf => FloatOp::Erf,
        TapeKernel::InlineIsNaN => FloatOp::IsNaN,
        TapeKernel::InlineNot => FloatOp::Not,
        // Binary arithmetic
        TapeKernel::InlineAdd => FloatOp::Add,
        TapeKernel::InlineMul => FloatOp::Mul,
        TapeKernel::InlineSub => FloatOp::Sub,
        TapeKernel::InlineDiv => FloatOp::Div,
        TapeKernel::InlineMin => FloatOp::Min,
        TapeKernel::InlineMax => FloatOp::Max,
        TapeKernel::InlinePow => FloatOp::Pow,
        TapeKernel::InlineMod => FloatOp::Mod,
        // Binary logical
        TapeKernel::InlineAnd => FloatOp::And,
        TapeKernel::InlineOr => FloatOp::Or,
        TapeKernel::InlineXor => FloatOp::Xor,
        // Binary comparison
        TapeKernel::InlineEqual => FloatOp::Equal,
        TapeKernel::InlineLess => FloatOp::Less,
        TapeKernel::InlineLessOrEqual => FloatOp::LessOrEqual,
        TapeKernel::InlineGreater => FloatOp::Greater,
        TapeKernel::InlineGreaterOrEqual => FloatOp::GreaterOrEqual,
        // Fused
        TapeKernel::InlineFusedSwiGLU => FloatOp::FusedSwiGLU,
        _ => unreachable!(
            "tape_kernel_as_float_op called with non-trivial kernel: {:?}",
            std::mem::discriminant(kernel)
        ),
    }
}

/// Apply activation element-wise to an out_buf that contains f32 data.
/// Used for epilogue fusion on LUT-GEMM paths where the kernel writes
/// to out_buf first and we apply activation as an immediate post-pass.
fn apply_activation_to_out_buf(out_buf: &mut [u8], activation: &FloatOp) {
    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(out_buf) {
        for v in floats.iter_mut() {
            *v = activation.apply_unary(*v);
        }
    }
}

/// LUT-GEMM Q4 dispatch for tape kernels.
/// Q4 LUT-GEMM dispatch: LUT provides compact storage + fast dequant,
/// BLAS/AMX provides hardware-accelerated matmul where available.
///
/// Single pipeline: LUT dequant Q4→f32 (cached) → BLAS sgemm (AMX hardware).
/// Non-BLAS platforms: int8 LUT kernel (pure integer accumulation via NEON/scalar).
///
/// LUT is always the data layer — hologram's core value (compact Q4 weights
/// with centroid table lookup). BLAS augments LUT with hardware matrix compute.
fn dispatch_lut_gemm_4(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<()> {
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q4: activation not f32-aligned".into())
    })?;

    let mut cache = tape_ctx.weight_cache.write();
    let qw = cache.get_q4(cid, tape_ctx.constants, tape_ctx.weights)?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = activations.len().checked_div(k).unwrap_or(0);
    let out = crate::float_dispatch::helpers::alloc_f32_in(out_buf, m * n);

    // LUT + BLAS pipeline: LUT handles Q4→f32 dequant (cached), BLAS handles matmul.
    // AMX hardware outperforms software NEON for all matmul sizes including M=1.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        drop(cache);
        let mut cache = tape_ctx.weight_cache.write();
        let dequantized = cache.get_dequantized_f32(cid, tape_ctx.constants, tape_ctx.weights)?;
        crate::float_dispatch::matmul::blas::sgemm(m, n, k, activations, dequantized, out);
        return Ok(());
    }

    // Non-BLAS: int8 LUT kernel (NEON table lookup + integer accumulation).
    #[allow(unreachable_code)]
    {
        crate::lut_gemm::lut_gemm_4bit(activations, qw, out);
    }
    Ok(())
}

/// LUT-GEMM Q8 dispatch for tape kernels.
fn dispatch_lut_gemm_8(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<()> {
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q8: activation not f32-aligned".into())
    })?;

    let mut cache = tape_ctx.weight_cache.write();
    let qw = cache.get_q8(cid, tape_ctx.constants, tape_ctx.weights)?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = activations.len().checked_div(k).unwrap_or(0);
    let out = crate::float_dispatch::helpers::alloc_f32_in(out_buf, m * n);

    // Q8 + BLAS pipeline: dequant Q8 centroids → cached f32, then BLAS sgemm.
    // Same pattern as Q4 — AMX hardware outperforms software LUT kernels.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        drop(cache);
        let mut cache = tape_ctx.weight_cache.write();
        let dequantized =
            cache.get_dequantized_f32_q8(cid, tape_ctx.constants, tape_ctx.weights)?;
        crate::float_dispatch::matmul::blas::sgemm(m, n, k, activations, dequantized, out);
        return Ok(());
    }

    // Non-BLAS fallback: LUT-GEMM Q8 integer kernel.
    #[allow(unreachable_code)]
    {
        #[cfg(feature = "parallel")]
        crate::lut_gemm::lut_gemm_8bit_par(activations, qw, out);
        #[cfg(not(feature = "parallel"))]
        crate::lut_gemm::lut_gemm_8bit(activations, qw, out);
    }
    Ok(())
}

/// LUT-GEMM Q16 dispatch for tape kernels.
fn dispatch_lut_gemm_16(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<()> {
    let mut cache = tape_ctx.weight_cache.write();
    let qw = cache.get_q16(cid, tape_ctx.constants, tape_ctx.weights)?;
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q16: activation not f32-aligned".into())
    })?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = activations.len().checked_div(k).unwrap_or(0);
    let output_bytes = m * n * 4;
    let base = out_buf.len();
    out_buf.resize(base + output_bytes, 0);
    let output_slice: &mut [f32] = bytemuck::cast_slice_mut(&mut out_buf[base..]);
    crate::lut_gemm::lut_gemm_16bit(activations, qw, output_slice);
    Ok(())
}

/// LUT-GEMM Q2 dispatch for tape kernels.
///
/// Always uses the pure integer LUT kernel — no BLAS dequant path.
/// The whole point of Q2 is to bypass BLAS and use the pure integer kernel.
fn dispatch_lut_gemm_2(
    inputs: &[&[u8]],
    cid: ConstantId,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<()> {
    let mut cache = tape_ctx.weight_cache.write();
    let qw = cache.get_q2(cid, tape_ctx.constants, tape_ctx.weights)?;
    let activations: &[f32] = bytemuck::try_cast_slice(inputs[0]).map_err(|_| {
        crate::error::ExecError::UnsupportedOp("Q2: activation not f32-aligned".into())
    })?;
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let m = activations.len().checked_div(k).unwrap_or(0);
    let mut output = vec![0.0f32; m * n];
    crate::lut_gemm::lut_gemm_2bit(activations, qw, &mut output);
    out_buf.extend_from_slice(bytemuck::cast_slice(&output));
    Ok(())
}

/// Parameters for [`dispatch_kv_write`] that describe the KV slot being
/// written. Built with [`KvWriteParams::new`] (required: layer + head shape)
/// and chained [`Self::with_is_key`] / [`Self::with_heads_first`] to flip
/// the two layout/role flags. Default is "write V in seq-first layout".
#[derive(Debug, Clone, Copy)]
pub(crate) struct KvWriteParams {
    pub layer: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub is_key: bool,
    pub heads_first: bool,
}

impl KvWriteParams {
    #[inline]
    pub fn new(layer: u32, n_kv_heads: u32, head_dim: u32) -> Self {
        Self {
            layer,
            n_kv_heads,
            head_dim,
            is_key: false,
            heads_first: false,
        }
    }

    #[inline]
    pub fn with_is_key(mut self, is_key: bool) -> Self {
        self.is_key = is_key;
        self
    }

    #[inline]
    pub fn with_heads_first(mut self, heads_first: bool) -> Self {
        self.heads_first = heads_first;
        self
    }
}

/// KvWrite dispatch: store K/V to cache, output for downstream attention.
///
/// `heads_first` determines the input layout and output format:
/// - `true`: input is `[heads, seq, dim]`, transpose to seq-first for storage,
///   and transpose back to heads-first on output during decode.
/// - `false`: input is `[seq, heads, dim]`, store directly, output seq-first.
///
/// Returns the actual output TensorMeta (runtime shape, not compiled).
fn dispatch_kv_write(
    inputs: &[&[u8]],
    params: KvWriteParams,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<hologram_core::op::TensorMeta> {
    let KvWriteParams {
        layer,
        n_kv_heads,
        head_dim,
        is_key,
        heads_first,
    } = params;
    let Some(kv_cell) = &tape_ctx.kv_state else {
        return Err(crate::error::ExecError::UnsupportedOp(
            "KvWrite requires TapeContext with kv_state".into(),
        ));
    };
    let input = inputs.first().copied().unwrap_or(&[]);
    if input.is_empty() || input.len() % 4 != 0 {
        out_buf.extend_from_slice(input);
        return Ok(hologram_core::op::TensorMeta::infer_1d(input.len(), 4));
    }
    let floats = safe_cast_f32(input);
    let nkv = n_kv_heads as usize;
    let hd = head_dim as usize;
    let stride = nkv * hd;
    let seq = floats.len().checked_div(stride).unwrap_or(1);

    // Convert to seq-first for cache storage if input is heads-first.
    let seq_first_data: Vec<f32>;
    let cache_data: &[f32] = if heads_first {
        seq_first_data = transpose_heads_to_seq_first(&floats, nkv, seq, hd);
        &seq_first_data
    } else {
        &floats
    };

    let mut kv = kv_cell.borrow_mut();
    if is_key {
        kv.write_layer(layer, cache_data, &[]);
    } else {
        kv.write_layer(layer, &[], cache_data);
    }

    let out_meta = if kv.write_pos() == 0 {
        // Prefill: pass through original data in its original layout.
        out_buf.extend_from_slice(input);
        // Output shape matches input shape.
        if heads_first {
            hologram_core::op::TensorMeta::new(hologram_core::op::FloatDType::F32, &[nkv, seq, hd])
        } else {
            hologram_core::op::TensorMeta::new(hologram_core::op::FloatDType::F32, &[seq, nkv, hd])
        }
    } else {
        // Decode: read full cache (seq-first) and convert to output layout.
        // For quantized layers, read_*_through returns &[] — use the owned
        // (dequantizing) path instead.
        let total_seq = kv.write_pos() + seq;
        let borrowed = if is_key {
            kv.read_k_through(layer, seq)
        } else {
            kv.read_v_through(layer, seq)
        };
        if !borrowed.is_empty() {
            // F32 path: zero-copy.
            if heads_first {
                let heads = transpose_seq_first_to_heads(borrowed, nkv, total_seq, hd);
                out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&heads));
            } else {
                out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(borrowed));
            }
        } else {
            // Quantized path: dequantize into owned Vec.
            let owned = if is_key {
                kv.read_k_through_owned(layer, seq)
            } else {
                kv.read_v_through_owned(layer, seq)
            };
            if heads_first {
                let heads = transpose_seq_first_to_heads(&owned, nkv, total_seq, hd);
                out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&heads));
            } else {
                out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&owned));
            }
        }
        if heads_first {
            hologram_core::op::TensorMeta::new(
                hologram_core::op::FloatDType::F32,
                &[nkv, total_seq, hd],
            )
        } else {
            hologram_core::op::TensorMeta::new(
                hologram_core::op::FloatDType::F32,
                &[total_seq, nkv, hd],
            )
        }
    };
    Ok(out_meta)
}

/// KvRead dispatch: read full cached K/V from the KV cache.
///
/// `heads_first` determines output layout:
/// - `true`: transpose from seq-first cache to `[heads, seq, dim]`
/// - `false`: return seq-first `[seq, heads, dim]` directly
fn dispatch_kv_read(
    layer: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_first: bool,
    tape_ctx: &TapeContext<'_>,
    out_buf: &mut OutputBuffer,
) -> ExecResult<()> {
    let Some(kv_cell) = &tape_ctx.kv_state else {
        return Err(crate::error::ExecError::UnsupportedOp(
            "KvRead requires TapeContext with kv_state".into(),
        ));
    };
    let kv = kv_cell.borrow();
    let nkv = n_kv_heads as usize;
    let hd = head_dim as usize;
    let total_seq = kv.write_pos();

    // K: try borrowed (f32) first, fall back to owned (dequantized).
    let k_borrowed = kv.read_k(layer);
    let k_owned;
    let k_data: &[f32] = if !k_borrowed.is_empty() {
        k_borrowed
    } else {
        k_owned = kv.read_k_owned(layer);
        &k_owned
    };

    // V: try borrowed (f32) first, fall back to owned (dequantized).
    let v_borrowed = kv.read_v(layer);
    let v_owned;
    let v_data: &[f32] = if !v_borrowed.is_empty() {
        v_borrowed
    } else {
        v_owned = kv.read_v_owned(layer);
        &v_owned
    };

    if heads_first {
        let k_heads = transpose_seq_first_to_heads(k_data, nkv, total_seq, hd);
        let v_heads = transpose_seq_first_to_heads(v_data, nkv, total_seq, hd);
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&k_heads));
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&v_heads));
    } else {
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(k_data));
        out_buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(v_data));
    }
    Ok(())
}

/// Transpose KV data from heads-first `[heads, seq, dim]` to seq-first `[seq, heads, dim]`.
fn transpose_heads_to_seq_first(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for h in 0..n_heads {
        for s in 0..seq {
            let src = (h * seq + s) * head_dim;
            let dst = (s * n_heads + h) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}

/// Transpose KV data from seq-first `[seq, heads, dim]` to heads-first `[heads, seq, dim]`.
fn transpose_seq_first_to_heads(
    data: &[f32],
    n_heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let total = n_heads * seq * head_dim;
    if data.len() < total || seq == 0 || n_heads == 0 || head_dim == 0 {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; total];
    for s in 0..seq {
        for h in 0..n_heads {
            let src = (s * n_heads + h) * head_dim;
            let dst = (h * seq + s) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    out
}
