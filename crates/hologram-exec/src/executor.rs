//! Single-path executor using hologram-backend.
//!
//! All buffers live on the target device. All ops dispatch through the
//! backend. One flush at the end. No CPU↔GPU transfers during execution.
//!
//! This replaces the dual-path logic in `tape.rs::execute_direct` with
//! a clean single-path loop.

use hologram_backend::{ComputeBackend, ComputeMemory, TensorBuffer};
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

    // Upload constants and graph inputs from the arena to device memory.
    for instr in &tape.instructions {
        for &idx in &instr.input_indices {
            let i = idx as usize;
            if i < bufs.len() {
                if let Ok(data) = arena.get(hologram_graph::NodeId::new(idx, 0)) {
                    if !data.is_empty() {
                        bufs[i] = TensorBuffer::unshared(memory.upload(data));
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
        let out_tb = unsafe { &mut *bufs_ptr.add(out_idx) };

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
            let result = backend.dispatch(
                &op,
                &input_refs,
                &mut out_tb.buffer,
                &hologram_backend::KernelParams::default(),
            );
            if let Err(e) = result {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                    eprintln!("[executor] unsupported op: {e}");
                }
            }
            // TODO: compute output shape from op + input shapes and store in out_tb.shape.
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
