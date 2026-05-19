//! Single-path executor using hologram-backend.
//!
//! All buffers live on the target device. All ops dispatch through the
//! backend. One flush at the end. No CPU↔GPU transfers during execution.
//!
//! This replaces the dual-path logic in `tape.rs::execute_direct` with
//! a clean single-path loop.

use hologram_backend::{ComputeBackend, ComputeMemory, KernelParams, TensorBuffer};
use hologram_core::op::FloatOp;
use hologram_shape::{infer_output_shape, TensorShape};
use smallvec::SmallVec;

use crate::buffer::BufferArena;
use crate::tape::{EnumTape, TapeKernel};

/// Execute a tape on a device-native backend.
///
/// All weights and constants are uploaded to the device at the start.
/// All intermediate buffers live on the device. One flush at the end.
/// The output is downloaded to CPU for the caller.
///
/// This is the production execution path — no CPU fallback, no hybrid
/// GPU/CPU, no per-op readback.
pub fn execute_on_backend<M, B>(
    tape: &EnumTape,
    arena: &BufferArena<'_>,
    memory: &M,
    backend: &B,
) -> crate::error::ExecResult<Vec<Vec<u8>>>
where
    M: ComputeMemory,
    B: ComputeBackend<M>,
{
    let num_slots = tape
        .instructions
        .iter()
        .map(|i| i.output_idx as usize + 1)
        .max()
        .unwrap_or(0);

    // Allocate device buffers paired with shape metadata.
    // Every buffer carries its shape so downstream ops can resolve dimensions.
    let mut bufs: Vec<TensorBuffer<M::Buffer>> = (0..num_slots)
        .map(|_| TensorBuffer::unshared(memory.alloc(0)))
        .collect();
    // Parallel shape table — TensorBuffer.shape only carries `dims`, but
    // the inference step needs `dtype` too. Track full TensorShapes
    // here keyed by slot index so each dispatch can look up its
    // inputs and write its output without round-tripping the arena
    // (which is `&self` here).
    let mut shapes: Vec<Option<TensorShape>> = vec![None; num_slots];

    // Upload constants and graph inputs from the arena to device memory.
    // Seed the parallel shape table from the arena's `ShapeRegistry`
    // (Sprint 33 Phase 2) at the same time.
    for instr in &tape.instructions {
        for &idx in &instr.input_indices {
            let i = idx as usize;
            if i < bufs.len() {
                let node = hologram_graph::NodeId::new(idx, 0);
                if let Ok(data) = arena.get(node) {
                    if !data.is_empty() {
                        bufs[i] = TensorBuffer::unshared(memory.upload(data));
                    }
                }
                if shapes[i].is_none() {
                    if let Some(s) = arena.get_shape(node) {
                        bufs[i].shape = SmallVec::from_slice(&s.dims);
                        shapes[i] = Some(s.clone());
                    }
                }
            }
        }
    }

    // Execute: one dispatch per instruction, all on device.
    for instr in &tape.instructions {
        let out_idx = instr.output_idx as usize;

        // SAFETY: output_idx != any input_idx in a valid DAG.
        let bufs_ptr = bufs.as_mut_ptr();
        let input_refs: SmallVec<[&M::Buffer; 4]> = instr
            .input_indices
            .iter()
            .map(|&idx| unsafe { &(*bufs_ptr.add(idx as usize)).buffer })
            .collect();

        // Dispatch the kernel through the backend.
        let float_op = tape_kernel_to_float_op(&instr.kernel);
        if float_op.is_none() {
            static UNMAPPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if UNMAPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                eprintln!(
                    "[executor] unmapped kernel: {:?}",
                    std::mem::discriminant(&instr.kernel)
                );
            }
        }
        if let Some(op) = float_op {
            // Sprint 33 Phase 5.1 + 5.2: gather input shapes, infer the
            // output shape, then populate `KernelParams` with whatever
            // the dispatched op needs (e.g. MaxPool2d wants
            // `[channels, h_in, w_in]`). Sites that fall back to
            // byte-length inference still work; sites that hard-error
            // without params (MaxPool2d / AvgPool2d / Resize) now get
            // the right shapes routed through.
            drop(input_refs);
            let input_shapes_opt: Option<SmallVec<[TensorShape; 4]>> = instr
                .input_indices
                .iter()
                .map(|&idx| shapes.get(idx as usize).and_then(|s| s.clone()))
                .collect();
            let out_shape = input_shapes_opt.as_ref().and_then(|input_shapes| {
                let refs: SmallVec<[&TensorShape; 4]> = input_shapes.iter().collect();
                infer_output_shape(&op, &refs).ok()
            });
            let params = match (&input_shapes_opt, out_shape.as_ref()) {
                (Some(inputs), out) => {
                    let refs: SmallVec<[&TensorShape; 4]> = inputs.iter().collect();
                    kernel_params_for(&op, &refs, out)
                }
                _ => KernelParams::default(),
            };

            let input_refs: SmallVec<[&M::Buffer; 4]> = instr
                .input_indices
                .iter()
                .map(|&idx| unsafe { &(*bufs_ptr.add(idx as usize)).buffer })
                .collect();
            let out_tb = unsafe { &mut *bufs_ptr.add(out_idx) };
            let result = backend.dispatch(&op, &input_refs, &mut out_tb.buffer, &params);
            if let Err(e) = result {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                    eprintln!("[executor] unsupported op: {e}");
                }
            }
            drop(input_refs);

            // Phase 5.1: persist inferred output shape on both the
            // parallel table and the device buffer's metadata.
            if let Some(out_shape) = out_shape {
                let out_tb = unsafe { &mut *bufs_ptr.add(out_idx) };
                out_tb.shape = SmallVec::from_slice(&out_shape.dims);
                debug_assert_eq!(
                    memory.byte_len(&out_tb.buffer),
                    out_shape.dims.iter().product::<usize>() * out_shape.dtype.byte_size(),
                    "inferred shape volume mismatches backend buffer for op {:?}",
                    op,
                );
                shapes[out_idx] = Some(out_shape);
            }
            continue;
        }

        drop(input_refs);
    }

    // Single flush at the end.
    backend.flush();

    // Download all non-empty buffers to CPU.
    let outputs: Vec<Vec<u8>> = bufs
        .iter()
        .map(|tb| {
            if memory.byte_len(&tb.buffer) > 0 {
                memory.download(&tb.buffer)
            } else {
                Vec::new()
            }
        })
        .collect();

    Ok(outputs)
}

/// Build the `KernelParams` a given `FloatOp` expects from input
/// shapes (and the inferred output shape, when relevant).
///
/// The CPU backend's per-op convention varies — some ops want the
/// last-axis size, NCHW convs want `[channels, h_in, w_in]`,
/// `Resize` additionally needs `[h_out, w_out]`. This function
/// centralises the mapping so the executor can populate params
/// instead of always passing `KernelParams::default()`.
///
/// Ops that aren't covered fall back to the default params; the
/// CPU backend's "infer from byte_len" code paths handle them.
pub fn kernel_params_for(
    op: &FloatOp,
    input_shapes: &[&TensorShape],
    output_shape: Option<&TensorShape>,
) -> KernelParams {
    use smallvec::SmallVec;
    let mut params = KernelParams::default();

    // Helpers for common NCHW lookups.
    let nchw = |s: &TensorShape| -> Option<(u32, u32, u32)> {
        if s.dims.len() < 4 {
            return None;
        }
        Some((s.dims[1] as u32, s.dims[2] as u32, s.dims[3] as u32))
    };
    let last = |s: &TensorShape| s.dims.last().copied().map(|d| d as u32);

    match op {
        // ── NCHW pool / resize: [channels, h_in, w_in] (+ h_out, w_out) ──
        FloatOp::MaxPool2d { .. } | FloatOp::AvgPool2d { .. } => {
            if let Some(s) = input_shapes.first() {
                if let Some((c, h, w)) = nchw(s) {
                    params.u32s = SmallVec::from_slice(&[c, h, w]);
                }
            }
        }
        FloatOp::GlobalAvgPool { .. } => {
            if let Some(s) = input_shapes.first() {
                if let Some((c, h, w)) = nchw(s) {
                    params.u32s = SmallVec::from_slice(&[c, h, w]);
                }
            }
        }
        FloatOp::Resize { .. } => {
            if let (Some(s_in), Some(out)) = (input_shapes.first(), output_shape) {
                if let (Some((c, hi, wi)), 4) = (nchw(s_in), out.dims.len()) {
                    params.u32s =
                        SmallVec::from_slice(&[c, hi, wi, out.dims[2] as u32, out.dims[3] as u32]);
                }
            }
        }
        // ── Last-axis ops: u32s[0] = last_dim ───────────────────────────
        FloatOp::Softmax { .. }
        | FloatOp::LogSoftmax { .. }
        | FloatOp::ReduceSum { .. }
        | FloatOp::ReduceMean { .. }
        | FloatOp::ReduceMax { .. }
        | FloatOp::ReduceMin { .. }
        | FloatOp::ReduceProd { .. }
        | FloatOp::CumSum { .. }
        | FloatOp::LRN { .. } => {
            if let Some(d) = input_shapes.first().and_then(|s| last(s)) {
                params.u32s = SmallVec::from_slice(&[d]);
            }
        }
        // ── Transpose: full input dims fill u32s; perm via .perm ────────
        FloatOp::Transpose { perm, ndim } => {
            if let Some(s) = input_shapes.first() {
                let dims: SmallVec<[u32; 8]> = s.dims.iter().map(|&d| d as u32).collect();
                params.u32s = dims;
                params.perm = *perm;
                params.perm_len = *ndim;
            }
        }
        _ => {}
    }
    params
}

/// Map a TapeKernel to a FloatOp for backend dispatch.
///
/// This bridges the tape's instruction encoding to the backend's op type.
/// Eventually the tape should use FloatOp directly, eliminating this mapping.
pub fn tape_kernel_to_float_op(kernel: &TapeKernel) -> Option<hologram_core::op::FloatOp> {
    use hologram_core::op::FloatOp;

    match kernel {
        TapeKernel::InlineRelu => Some(FloatOp::Relu),
        TapeKernel::InlineNeg => Some(FloatOp::Neg),
        TapeKernel::InlineSigmoid => Some(FloatOp::Sigmoid),
        TapeKernel::InlineSilu => Some(FloatOp::Silu),
        TapeKernel::InlineTanh => Some(FloatOp::Tanh),
        TapeKernel::InlineGelu => Some(FloatOp::Gelu),
        TapeKernel::InlineExp => Some(FloatOp::Exp),
        TapeKernel::InlineAbs => Some(FloatOp::Abs),
        TapeKernel::InlineReciprocal => Some(FloatOp::Reciprocal),
        TapeKernel::InlineErf => Some(FloatOp::Erf),
        TapeKernel::InlineAdd => Some(FloatOp::Add),
        TapeKernel::InlineMul => Some(FloatOp::Mul),
        TapeKernel::InlineSub => Some(FloatOp::Sub),
        TapeKernel::InlineDiv => Some(FloatOp::Div),
        TapeKernel::InlineMatMul { m, k, n } => Some(FloatOp::MatMul {
            m: *m,
            k: *k,
            n: *n,
        }),
        TapeKernel::InlineSoftmax { size } => Some(FloatOp::Softmax { size: *size }),
        TapeKernel::InlineRmsNorm { size, epsilon } => Some(FloatOp::RmsNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineLayerNorm { size, epsilon } => Some(FloatOp::LayerNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineInstanceNorm { size, epsilon } => Some(FloatOp::InstanceNorm {
            size: *size,
            epsilon: *epsilon,
        }),
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
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineConv2dLut4 {
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
            ..
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineTranspose { perm, ndim, .. } => Some(FloatOp::Transpose {
            perm: *perm,
            ndim: *ndim,
        }),
        TapeKernel::InlineReshape => Some(FloatOp::Reshape),
        TapeKernel::InlineSlice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => Some(FloatOp::Slice {
            axis_from_end: *axis_from_end,
            start: *start,
            end: *end,
            axis_size: *axis_size,
        }),
        TapeKernel::InlineConcat {
            size_a,
            size_b,
            dtype,
        } => Some(FloatOp::Concat {
            size_a: *size_a,
            size_b: *size_b,
            dtype: *dtype,
        }),
        TapeKernel::InlineGemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => Some(FloatOp::Gemm {
            m: *m,
            k: *k,
            n: *n,
            alpha: *alpha,
            beta: *beta,
            trans_a: *trans_a,
            trans_b: *trans_b,
            quant_b: *quant_b,
        }),
        TapeKernel::InlineResize { mode } => Some(FloatOp::Resize { mode: *mode }),
        TapeKernel::InlineLog => Some(FloatOp::Log),
        TapeKernel::InlineSqrt => Some(FloatOp::Sqrt),
        TapeKernel::InlineCos => Some(FloatOp::Cos),
        TapeKernel::InlineSin => Some(FloatOp::Sin),
        TapeKernel::InlineSign => Some(FloatOp::Sign),
        TapeKernel::InlineFloor => Some(FloatOp::Floor),
        TapeKernel::InlineCeil => Some(FloatOp::Ceil),
        TapeKernel::InlineRound => Some(FloatOp::Round),
        TapeKernel::InlinePow => Some(FloatOp::Pow),
        TapeKernel::InlineMod => Some(FloatOp::Mod),
        TapeKernel::InlineMin => Some(FloatOp::Min),
        TapeKernel::InlineMax => Some(FloatOp::Max),
        TapeKernel::InlineClip { min, max } => Some(FloatOp::Clip {
            min: *min,
            max: *max,
        }),
        TapeKernel::InlineIsNaN => Some(FloatOp::IsNaN),
        TapeKernel::InlineNot => Some(FloatOp::Not),
        TapeKernel::InlineAnd => Some(FloatOp::And),
        TapeKernel::InlineOr => Some(FloatOp::Or),
        TapeKernel::InlineXor => Some(FloatOp::Xor),
        TapeKernel::InlineEqual => Some(FloatOp::Equal),
        TapeKernel::InlineLess => Some(FloatOp::Less),
        TapeKernel::InlineLessOrEqual => Some(FloatOp::LessOrEqual),
        TapeKernel::InlineGreater => Some(FloatOp::Greater),
        TapeKernel::InlineGreaterOrEqual => Some(FloatOp::GreaterOrEqual),

        // Reductions.
        TapeKernel::InlineReduceSum { size } => Some(FloatOp::ReduceSum { size: *size }),
        TapeKernel::InlineReduceMean { size } => Some(FloatOp::ReduceMean { size: *size }),
        TapeKernel::InlineReduceMax { size } => Some(FloatOp::ReduceMax { size: *size }),
        TapeKernel::InlineReduceMin { size } => Some(FloatOp::ReduceMin { size: *size }),
        TapeKernel::InlineReduceProd { size } => Some(FloatOp::ReduceProd { size: *size }),

        // Normalization.
        TapeKernel::InlineGroupNorm {
            num_groups,
            epsilon,
        } => Some(FloatOp::GroupNorm {
            num_groups: *num_groups,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineAddRmsNorm { size, epsilon } => Some(FloatOp::AddRmsNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineLogSoftmax { size } => Some(FloatOp::LogSoftmax { size: *size }),

        // Fused attention.
        TapeKernel::InlineAttention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            sparse_v,
        } => Some(FloatOp::Attention {
            head_dim: *head_dim,
            num_q_heads: *num_q_heads,
            num_kv_heads: *num_kv_heads,
            scale: *scale,
            causal: *causal,
            heads_first: *heads_first,
            qk_norm: false,
            rope: false,
            rope_base: 0,
            sparse_v: *sparse_v,
        }),

        // Rotary embedding.
        TapeKernel::InlineRoPE { dim, base, n_heads } => Some(FloatOp::RotaryEmbedding {
            dim: *dim,
            base: *base,
            n_heads: *n_heads,
        }),

        // Data movement / shape ops.
        TapeKernel::InlineGather { dim, dtype, .. } => Some(FloatOp::Gather {
            dim: *dim,
            dtype: *dtype,
        }),
        TapeKernel::InlineGatherND => Some(FloatOp::GatherND),
        TapeKernel::InlineWhere => Some(FloatOp::Where),
        TapeKernel::InlineRange => Some(FloatOp::Range),
        TapeKernel::InlineShape { dtype, start, end } => Some(FloatOp::Shape {
            dtype: *dtype,
            start: *start,
            end: *end,
        }),
        TapeKernel::InlineEmbed { dim, quant } => Some(FloatOp::Embed {
            dim: *dim,
            quant: *quant,
        }),
        TapeKernel::InlineCast { from, to } => Some(FloatOp::Cast {
            from: *from,
            to: *to,
        }),
        TapeKernel::InlineDequantize => Some(FloatOp::Dequantize),
        TapeKernel::InlineExpand { ndim, target_shape } => Some(FloatOp::Expand {
            ndim: *ndim,
            target_shape: *target_shape,
        }),
        TapeKernel::InlinePad { mode } => Some(FloatOp::PadOp { mode: *mode }),

        // Fused ops.
        TapeKernel::InlineFusedSwiGLU => Some(FloatOp::FusedSwiGLU),

        // Vision / spatial ops.
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
        } => Some(FloatOp::ConvTranspose {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            output_pad_h: *output_pad_h,
            output_pad_w: *output_pad_w,
            input_h: *input_h,
            input_w: *input_w,
        }),
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
            ..
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineConv2dBiasActivation {
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
            ..
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineMaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => Some(FloatOp::MaxPool2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
        }),
        TapeKernel::InlineAvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => Some(FloatOp::AvgPool2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
        }),
        TapeKernel::InlineGlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => Some(FloatOp::GlobalAvgPool {
            channels: *channels,
            spatial_h: *spatial_h,
            spatial_w: *spatial_w,
        }),
        TapeKernel::InlineLRN {
            size,
            alpha,
            beta,
            bias,
        } => Some(FloatOp::LRN {
            size: *size,
            alpha: *alpha,
            beta: *beta,
            bias: *bias,
        }),

        // KV cache ops.
        TapeKernel::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
            heads_first,
        } => Some(FloatOp::KvWrite {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
            is_key: *is_key,
            heads_first: *heads_first,
        }),
        TapeKernel::KvRead {
            layer,
            n_kv_heads,
            head_dim,
            heads_first,
        } => Some(FloatOp::KvRead {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
            heads_first: *heads_first,
        }),

        // Deep decode fusions.
        TapeKernel::InlineNormProjectionGemv {
            norm_size,
            epsilon,
            k,
            n_total,
        } => Some(FloatOp::NormProjectionGemv {
            norm_size: *norm_size,
            epsilon: *epsilon,
            k: *k,
            n_total: *n_total,
        }),
        TapeKernel::InlineAddNormProjectionGemv {
            norm_size,
            epsilon,
            k,
            n_total,
        } => Some(FloatOp::AddNormProjectionGemv {
            norm_size: *norm_size,
            epsilon: *epsilon,
            k: *k,
            n_total: *n_total,
        }),
        TapeKernel::InlineSwiGluProjectionGemv { k, n } => {
            Some(FloatOp::SwiGluProjectionGemv { k: *k, n: *n })
        }

        // Utility ops.
        TapeKernel::InlineTopK { axis, largest } => Some(FloatOp::TopK {
            axis: *axis,
            largest: *largest,
        }),
        TapeKernel::InlineScatterND => Some(FloatOp::ScatterND),
        TapeKernel::InlineCumSum { axis } => Some(FloatOp::CumSum { axis: *axis }),
        TapeKernel::InlineNonZero => Some(FloatOp::NonZero),
        TapeKernel::InlineCompress { axis } => Some(FloatOp::Compress { axis: *axis }),
        TapeKernel::InlineReverseSequence {
            batch_axis,
            time_axis,
        } => Some(FloatOp::ReverseSequence {
            batch_axis: *batch_axis,
            time_axis: *time_axis,
        }),
        TapeKernel::InlineArgMax { axis, keepdims } => Some(FloatOp::ArgMax {
            axis: *axis,
            keepdims: *keepdims,
        }),

        // Output is an identity op — pass the input through.
        TapeKernel::Output => Some(FloatOp::Reshape),
        // FusedFloatChain: apply chain of unary ops. Map to first op.
        TapeKernel::FusedFloatChain(ops) if !ops.is_empty() => Some(ops[0]),

        // Fused norm+activation and MatMul+activation variants: map to the
        // base op. The fused activation is lost here because FloatOp doesn't
        // encode activation epilogues. The new executor applies these via
        // separate dispatches or fused chains.
        TapeKernel::InlineMatMulActivation { m, k, n, .. } => Some(FloatOp::MatMul {
            m: *m,
            k: *k,
            n: *n,
        }),
        TapeKernel::InlineMatMulBiasActivation { m, k, n, .. } => Some(FloatOp::MatMul {
            m: *m,
            k: *k,
            n: *n,
        }),
        TapeKernel::InlineRmsNormActivation { size, epsilon, .. } => Some(FloatOp::RmsNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineLayerNormActivation { size, epsilon, .. } => Some(FloatOp::LayerNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineGroupNormActivation {
            num_groups,
            epsilon,
            ..
        } => Some(FloatOp::GroupNorm {
            num_groups: *num_groups,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineAddRmsNormActivation { size, epsilon, .. } => Some(FloatOp::AddRmsNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineInstanceNormActivation { size, epsilon, .. } => {
            Some(FloatOp::InstanceNorm {
                size: *size,
                epsilon: *epsilon,
            })
        }

        // Ring/LUT ops are byte-domain; they don't map to FloatOp.
        // The executor handles them separately via dispatch_ring.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::FloatDType;

    fn nchw(n: usize, c: usize, h: usize, w: usize) -> TensorShape {
        TensorShape::new(FloatDType::F32, &[n, c, h, w])
    }

    #[test]
    fn kernel_params_for_maxpool_emits_chw() {
        let s = nchw(1, 16, 32, 64);
        let p = kernel_params_for(
            &FloatOp::MaxPool2d {
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
                pad_h: 0,
                pad_w: 0,
            },
            &[&s],
            None,
        );
        assert_eq!(p.u32s.as_slice(), &[16, 32, 64]);
    }

    #[test]
    fn kernel_params_for_avgpool_emits_chw() {
        let s = nchw(2, 8, 16, 16);
        let p = kernel_params_for(
            &FloatOp::AvgPool2d {
                kernel_h: 3,
                kernel_w: 3,
                stride_h: 1,
                stride_w: 1,
                pad_h: 1,
                pad_w: 1,
            },
            &[&s],
            None,
        );
        assert_eq!(p.u32s.as_slice(), &[8, 16, 16]);
    }

    #[test]
    fn kernel_params_for_resize_includes_h_out_w_out() {
        let s_in = nchw(1, 3, 8, 8);
        let s_out = nchw(1, 3, 16, 16);
        let p = kernel_params_for(&FloatOp::Resize { mode: 0 }, &[&s_in], Some(&s_out));
        assert_eq!(p.u32s.as_slice(), &[3, 8, 8, 16, 16]);
    }

    #[test]
    fn kernel_params_for_softmax_emits_last_dim() {
        let s = TensorShape::new(FloatDType::F32, &[2, 4, 1024]);
        let p = kernel_params_for(&FloatOp::Softmax { size: 1024 }, &[&s], None);
        assert_eq!(p.u32s.as_slice(), &[1024]);
    }

    #[test]
    fn kernel_params_for_transpose_emits_full_dims_and_perm() {
        let s = TensorShape::new(FloatDType::F32, &[1, 4, 8, 16]);
        let p = kernel_params_for(
            &FloatOp::Transpose {
                perm: [0, 2, 1, 3, 0, 0, 0, 0],
                ndim: 4,
            },
            &[&s],
            None,
        );
        assert_eq!(p.u32s.as_slice(), &[1, 4, 8, 16]);
        assert_eq!(p.perm_len, 4);
        assert_eq!(p.perm[..4], [0, 2, 1, 3]);
    }

    #[test]
    fn kernel_params_for_unknown_op_returns_default() {
        let s = TensorShape::new(FloatDType::F32, &[3, 4]);
        let p = kernel_params_for(&FloatOp::Add, &[&s, &s], None);
        assert!(p.u32s.is_empty());
    }
}
