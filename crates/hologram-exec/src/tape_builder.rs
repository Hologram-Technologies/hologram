//! Tape builder: compiles a `SerializedGraph` + `ExecutionSchedule` into a tape.
//!
//! This is the "compile" step that pre-resolves kernel enum variants and
//! output element sizes for every node. The resulting tape can be executed
//! repeatedly per inference call without per-node op matching or HashMap lookups.
//!
//! Built once at model load time, amortized across all inference calls.

use std::collections::HashMap;

use hologram_archive::format::graph::SerializedGraph;
use hologram_core::op::{FloatDType, FloatOp};
use hologram_graph::graph::node::{InputSource, NodeId};
use hologram_graph::graph::GraphOp;
use hologram_graph::schedule::ExecutionSchedule;

use crate::error::{ExecError, ExecResult};
use crate::kv::CustomOpRegistry;
use crate::tape::{EnumTape, TapeInstruction, TapeKernel};

/// Build an [`EnumTape`] from a serialized graph and its execution schedule.
///
/// For each node in schedule order:
/// - Resolves a [`TapeKernel`] enum variant (no closure, no heap allocation)
/// - Pre-computes `output_elem_size` from the node's dtype
/// - Records input indices for zero-copy gathering at execution time
///
/// Constants and graph inputs are skipped (they are seeded into the arena
/// before tape execution).
pub fn build_tape(
    sg: &SerializedGraph,
    schedule: &ExecutionSchedule,
    registry: Option<&CustomOpRegistry>,
) -> ExecResult<EnumTape> {
    // Build flat lookup tables for O(1) access by node index.
    let max_idx = sg
        .nodes
        .iter()
        .map(|n| n.id.index() as usize + 1)
        .max()
        .unwrap_or(0);

    let mut node_by_idx: Vec<Option<usize>> = vec![None; max_idx];
    for (i, n) in sg.nodes.iter().enumerate() {
        let idx = n.id.index() as usize;
        if idx < max_idx {
            node_by_idx[idx] = Some(i);
        }
    }

    let dtypes: HashMap<NodeId, FloatDType> = sg.node_dtypes_map();
    let shapes: HashMap<NodeId, Vec<usize>> = sg.node_shapes_map();

    // Build lookup from graph-input index → Input node's NodeId.
    // Graph inputs are seeded into the arena at their node's index; compute ops
    // connected via InputSource::GraphInput need to reference that index.
    let graph_input_node_ids: Vec<NodeId> = sg
        .nodes
        .iter()
        .filter(|n| matches!(n.op, GraphOp::Input))
        .map(|n| n.id)
        .collect();

    let total_nodes: usize = schedule.levels.iter().map(|l| l.node_ids.len()).sum();
    let mut tape = EnumTape::with_capacity(total_nodes, schedule.levels.len());

    for level in &schedule.levels {
        for &node_id in &level.node_ids {
            let idx = node_id.index() as usize;
            let node_pos = if idx < max_idx {
                node_by_idx[idx]
            } else {
                None
            };
            let Some(node_pos) = node_pos else {
                continue;
            };
            let node = &sg.nodes[node_pos];

            // Skip constants and inputs — they're seeded into the arena.
            match &node.op {
                GraphOp::Constant(_) | GraphOp::Input => continue,
                _ => {}
            }

            // Resolve kernel enum variant.
            // For Transpose, we need input shapes to bake into InlineTranspose.
            let kernel = if let GraphOp::Float(FloatOp::Transpose { perm, ndim }) = &node.op {
                let n = *ndim as usize;
                // Get the input node's shape from compiled shapes.
                let input_node_id = node.inputs.first().and_then(|slot| match slot.source {
                    InputSource::Node(id) => Some(id),
                    InputSource::GraphInput { index } => {
                        graph_input_node_ids.get(index as usize).copied()
                    }
                    _ => None,
                });
                let input_shape_vec = input_node_id.and_then(|id| shapes.get(&id));
                if let Some(ishape) = input_shape_vec {
                    let mut shape_arr = [0u32; 8];
                    for (i, &d) in ishape.iter().take(8).enumerate() {
                        shape_arr[i] = d as u32;
                    }
                    TapeKernel::InlineTranspose {
                        perm: *perm,
                        input_shape: shape_arr,
                        ndim: n as u8,
                    }
                } else {
                    // No shape info — fall back to passthrough (legacy behavior).
                    TapeKernel::Passthrough
                }
            } else {
                resolve_kernel(&node.op, registry)?
            };

            // Pre-compute output elem_size.
            let output_elem_size = compute_elem_size(node_id, &node.op, &dtypes);

            // Collect input indices — resolve both Node and GraphInput sources.
            let input_indices: Vec<u32> = node
                .inputs
                .iter()
                .filter_map(|slot| match slot.source {
                    InputSource::Node(id) => Some(id.index()),
                    InputSource::GraphInput { index } => graph_input_node_ids
                        .get(index as usize)
                        .map(|id| id.index()),
                    InputSource::None => None,
                })
                .collect();

            // Pre-compute output byte size hint from compiled shapes.
            let output_byte_hint = compute_output_byte_hint(node_id, &shapes, output_elem_size);

            // Pre-compute weight offset for LUT-GEMM prefetching.
            let weight_offset_hint = compute_weight_offset(&kernel, &sg.constants);

            // Pre-compute output tensor metadata from compiled shapes + dtypes.
            let output_meta = shapes.get(&node_id).map(|shape| {
                let dtype = dtypes
                    .get(&node_id)
                    .copied()
                    .unwrap_or(hologram_core::op::FloatDType::F32);
                hologram_core::op::TensorMeta::new(dtype, shape)
            });

            tape.push(TapeInstruction {
                kernel,
                output_idx: node_id.index(),
                input_indices,
                output_elem_size,
                output_byte_hint,
                weight_offset_hint,
                passthrough: false,
                can_reuse_input: false,
                output_meta,
            });
        }
        tape.end_level();
    }

    // ── Post-pass: compute consumer counts and set optimization flags ──
    apply_reuse_flags(&mut tape);

    Ok(tape)
}

/// Scan instructions to compute per-node consumer counts, then set
/// `passthrough` (for Output ops with single-consumer inputs) and
/// `can_reuse_input` (for unary inline ops with single-consumer inputs).
fn apply_reuse_flags(tape: &mut EnumTape) {
    // Count how many instructions consume each node index.
    let max_idx = tape
        .instructions
        .iter()
        .map(|i| i.output_idx as usize)
        .max()
        .unwrap_or(0);
    let mut consumer_counts = vec![0u32; max_idx + 1];
    for instr in &tape.instructions {
        for &idx in &instr.input_indices {
            let i = idx as usize;
            if i < consumer_counts.len() {
                consumer_counts[i] += 1;
            }
        }
    }

    let is_single_consumer = |idx: u32| -> bool {
        let i = idx as usize;
        i < consumer_counts.len() && consumer_counts[i] == 1
    };

    for instr in &mut tape.instructions {
        match &instr.kernel {
            // Output passthrough: move buffer directly if input has one consumer.
            TapeKernel::Output if instr.input_indices.len() == 1 => {
                if is_single_consumer(instr.input_indices[0]) {
                    instr.passthrough = true;
                }
            }
            // Unary inline ops: reuse input buffer in-place if single consumer.
            TapeKernel::InlineRelu
            | TapeKernel::InlineNeg
            | TapeKernel::InlineAbs
            | TapeKernel::InlineSigmoid
            | TapeKernel::InlineSilu
            | TapeKernel::InlineTanh
            | TapeKernel::InlineGelu
            | TapeKernel::InlineExp
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
            | TapeKernel::InlineClip { .. }
            | TapeKernel::InlineNot
            | TapeKernel::InlineIsNaN
                if instr.input_indices.len() == 1 =>
            {
                if is_single_consumer(instr.input_indices[0]) {
                    instr.can_reuse_input = true;
                }
            }
            _ => {}
        }
    }
}

/// Resolve a `GraphOp` to a [`TapeKernel`] enum variant.
///
/// No closures, no heap allocation — just selects the right variant
/// and captures the op parameters inline.
fn resolve_kernel(op: &GraphOp, registry: Option<&CustomOpRegistry>) -> ExecResult<TapeKernel> {
    match op {
        GraphOp::Float(fop) => Ok(resolve_float_kernel(fop)),
        GraphOp::FusedFloatChain(chain) => Ok(TapeKernel::FusedFloatChain(chain.clone())),
        GraphOp::FusedMatMulActivation {
            m,
            k,
            n,
            activation,
        } => Ok(TapeKernel::InlineMatMulActivation {
            m: *m,
            k: *k,
            n: *n,
            activation: *activation,
        }),
        GraphOp::FusedMatMulBiasActivation {
            m,
            k,
            n,
            activation,
        } => Ok(TapeKernel::InlineMatMulBiasActivation {
            m: *m,
            k: *k,
            n: *n,
            activation: *activation,
        }),
        GraphOp::Output => Ok(TapeKernel::Output),
        GraphOp::Lut(_) | GraphOp::FusedView(_) => {
            let view = op
                .to_view()
                .ok_or_else(|| ExecError::UnsupportedOp("Lut/FusedView without view".into()))?;
            Ok(TapeKernel::LutView(view))
        }
        GraphOp::Prim(p) => {
            if p.arity() == 1 {
                let view = op
                    .to_view()
                    .ok_or_else(|| ExecError::UnsupportedOp("Prim without view".into()))?;
                Ok(TapeKernel::PrimUnary(view))
            } else {
                Ok(TapeKernel::PrimBinary(*p))
            }
        }
        GraphOp::MatMulLut4(cid) | GraphOp::BatchMatMulLut4(cid) => {
            Ok(TapeKernel::MatMulLut4(*cid))
        }
        GraphOp::MatMulLut8(cid) | GraphOp::BatchMatMulLut8(cid) => {
            Ok(TapeKernel::MatMulLut8(*cid))
        }
        GraphOp::FusedRmsNormActivation {
            size,
            epsilon,
            activation,
        } => Ok(TapeKernel::InlineRmsNormActivation {
            size: *size,
            epsilon: *epsilon,
            activation: *activation,
        }),
        GraphOp::FusedLayerNormActivation {
            size,
            epsilon,
            activation,
        } => Ok(TapeKernel::InlineLayerNormActivation {
            size: *size,
            epsilon: *epsilon,
            activation: *activation,
        }),
        GraphOp::FusedGroupNormActivation {
            num_groups,
            epsilon,
            activation,
        } => Ok(TapeKernel::InlineGroupNormActivation {
            num_groups: *num_groups,
            epsilon: *epsilon,
            activation: *activation,
        }),
        GraphOp::MatMulLut4Activation(cid, activation) => {
            Ok(TapeKernel::MatMulLut4Activation(*cid, *activation))
        }
        GraphOp::MatMulLut8Activation(cid, activation) => {
            Ok(TapeKernel::MatMulLut8Activation(*cid, *activation))
        }
        GraphOp::Custom { id, arity: _ } => {
            let reg = registry.ok_or_else(|| {
                ExecError::UnsupportedOp(format!(
                    "custom op {} requires a CustomOpRegistry",
                    id.raw()
                ))
            })?;
            let handler = reg.get_handler(*id).ok_or_else(|| {
                ExecError::UnsupportedOp(format!("custom op {} not registered", id.raw()))
            })?;
            Ok(TapeKernel::Custom(handler.clone()))
        }
        _ => Err(ExecError::UnsupportedOp(format!(
            "tape builder: unsupported op {:?}",
            op
        ))),
    }
}

/// Resolve a `FloatOp` to a [`TapeKernel`] variant.
///
/// Every FloatOp maps to a dedicated TapeKernel variant — no catch-all.
/// This ensures exhaustive coverage and explicit dispatch for every op.
fn resolve_float_kernel(fop: &FloatOp) -> TapeKernel {
    match fop {
        // ── Unary activations ────────────────────────────────────────────
        FloatOp::Relu => TapeKernel::InlineRelu,
        FloatOp::Neg => TapeKernel::InlineNeg,
        FloatOp::Sigmoid => TapeKernel::InlineSigmoid,
        FloatOp::Silu => TapeKernel::InlineSilu,
        FloatOp::Tanh => TapeKernel::InlineTanh,
        FloatOp::Gelu => TapeKernel::InlineGelu,
        FloatOp::Exp => TapeKernel::InlineExp,
        FloatOp::Abs => TapeKernel::InlineAbs,
        FloatOp::Reciprocal => TapeKernel::InlineReciprocal,
        FloatOp::Log => TapeKernel::InlineLog,
        FloatOp::Sqrt => TapeKernel::InlineSqrt,
        FloatOp::Cos => TapeKernel::InlineCos,
        FloatOp::Sin => TapeKernel::InlineSin,
        FloatOp::Sign => TapeKernel::InlineSign,
        FloatOp::Floor => TapeKernel::InlineFloor,
        FloatOp::Ceil => TapeKernel::InlineCeil,
        FloatOp::Round => TapeKernel::InlineRound,
        FloatOp::Erf => TapeKernel::InlineErf,
        FloatOp::Not => TapeKernel::InlineNot,
        FloatOp::IsNaN => TapeKernel::InlineIsNaN,
        FloatOp::Clip { min, max } => TapeKernel::InlineClip {
            min: *min,
            max: *max,
        },

        // ── Binary arithmetic ────────────────────────────────────────────
        FloatOp::Add => TapeKernel::InlineAdd,
        FloatOp::Mul => TapeKernel::InlineMul,
        FloatOp::Sub => TapeKernel::InlineSub,
        FloatOp::Div => TapeKernel::InlineDiv,
        FloatOp::Min => TapeKernel::InlineMin,
        FloatOp::Max => TapeKernel::InlineMax,
        FloatOp::Pow => TapeKernel::InlinePow,
        FloatOp::Mod => TapeKernel::InlineMod,

        // ── Boolean / comparison ─────────────────────────────────────────
        FloatOp::And => TapeKernel::InlineAnd,
        FloatOp::Or => TapeKernel::InlineOr,
        FloatOp::Xor => TapeKernel::InlineXor,
        FloatOp::Equal => TapeKernel::InlineEqual,
        FloatOp::Less => TapeKernel::InlineLess,
        FloatOp::LessOrEqual => TapeKernel::InlineLessOrEqual,
        FloatOp::Greater => TapeKernel::InlineGreater,
        FloatOp::GreaterOrEqual => TapeKernel::InlineGreaterOrEqual,

        // ── Linear algebra ───────────────────────────────────────────────
        FloatOp::MatMul { m, k, n } => TapeKernel::InlineMatMul {
            m: *m,
            k: *k,
            n: *n,
        },
        FloatOp::Gemm {
            m,
            k,
            n,
            alpha,
            beta,
            trans_a,
            trans_b,
            quant_b,
        } => TapeKernel::InlineGemm {
            m: *m,
            k: *k,
            n: *n,
            alpha: *alpha,
            beta: *beta,
            trans_a: *trans_a,
            trans_b: *trans_b,
            quant_b: *quant_b,
        },

        // ── Normalization / softmax ──────────────────────────────────────
        FloatOp::Softmax { size } => TapeKernel::InlineSoftmax { size: *size },
        FloatOp::LogSoftmax { size } => TapeKernel::InlineLogSoftmax { size: *size },
        FloatOp::RmsNorm { size, epsilon } => TapeKernel::InlineRmsNorm {
            size: *size,
            epsilon: *epsilon,
        },
        FloatOp::AddRmsNorm { size, epsilon } => TapeKernel::InlineAddRmsNorm {
            size: *size,
            epsilon: *epsilon,
        },
        FloatOp::LayerNorm { size, epsilon } => TapeKernel::InlineLayerNorm {
            size: *size,
            epsilon: *epsilon,
        },
        FloatOp::InstanceNorm { size, epsilon } => TapeKernel::InlineInstanceNorm {
            size: *size,
            epsilon: *epsilon,
        },
        FloatOp::GroupNorm {
            num_groups,
            epsilon,
        } => TapeKernel::InlineGroupNorm {
            num_groups: *num_groups,
            epsilon: *epsilon,
        },
        FloatOp::LRN {
            size,
            alpha,
            beta,
            bias,
        } => TapeKernel::InlineLRN {
            size: *size,
            alpha: *alpha,
            beta: *beta,
            bias: *bias,
        },

        // ── Reductions ───────────────────────────────────────────────────
        FloatOp::ReduceSum { size } => TapeKernel::InlineReduceSum { size: *size },
        FloatOp::ReduceMean { size } => TapeKernel::InlineReduceMean { size: *size },
        FloatOp::ReduceMax { size } => TapeKernel::InlineReduceMax { size: *size },
        FloatOp::ReduceMin { size } => TapeKernel::InlineReduceMin { size: *size },
        FloatOp::ReduceProd { size } => TapeKernel::InlineReduceProd { size: *size },

        // ── Attention / RoPE ─────────────────────────────────────────────
        FloatOp::Attention {
            head_dim,
            num_q_heads,
            num_kv_heads,
            scale,
            causal,
            heads_first,
            ..
        } => TapeKernel::InlineAttention {
            head_dim: *head_dim,
            num_q_heads: *num_q_heads,
            num_kv_heads: *num_kv_heads,
            scale: *scale,
            causal: *causal,
            heads_first: *heads_first,
        },
        FloatOp::RotaryEmbedding { dim, base, n_heads } => TapeKernel::InlineRoPE {
            dim: *dim,
            base: *base,
            n_heads: *n_heads,
        },
        FloatOp::FusedSwiGLU => TapeKernel::InlineFusedSwiGLU,

        // ── Shape manipulation ───────────────────────────────────────────
        FloatOp::Reshape => TapeKernel::InlineReshape,
        FloatOp::Gather { dim, dtype } => TapeKernel::InlineGather {
            dim: *dim,
            dtype: *dtype,
        },
        FloatOp::Concat {
            size_a,
            size_b,
            dtype,
        } => TapeKernel::InlineConcat {
            size_a: *size_a,
            size_b: *size_b,
            dtype: *dtype,
        },
        FloatOp::Cast { from, to } if from == to => TapeKernel::Passthrough,
        FloatOp::Cast { from, to } => TapeKernel::InlineCast {
            from: *from,
            to: *to,
        },
        FloatOp::Slice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => TapeKernel::InlineSlice {
            axis_from_end: *axis_from_end,
            start: *start,
            end: *end,
            axis_size: *axis_size,
        },
        FloatOp::Shape { dtype, start, end } => TapeKernel::InlineShape {
            dtype: *dtype,
            start: *start,
            end: *end,
        },
        FloatOp::GatherND => TapeKernel::InlineGatherND,

        // ── Embedding / quantization ─────────────────────────────────────
        FloatOp::Embed { dim, quant } => TapeKernel::InlineEmbed {
            dim: *dim,
            quant: *quant,
        },
        FloatOp::Dequantize => TapeKernel::InlineDequantize,

        // ── Conditional / utility ────────────────────────────────────────
        FloatOp::Where => TapeKernel::InlineWhere,
        FloatOp::Range => TapeKernel::InlineRange,
        FloatOp::TopK { axis, largest } => TapeKernel::InlineTopK {
            axis: *axis,
            largest: *largest,
        },
        FloatOp::ScatterND => TapeKernel::InlineScatterND,
        FloatOp::CumSum { axis } => TapeKernel::InlineCumSum { axis: *axis },
        FloatOp::NonZero => TapeKernel::InlineNonZero,
        FloatOp::Compress { axis } => TapeKernel::InlineCompress { axis: *axis },
        FloatOp::ReverseSequence {
            batch_axis,
            time_axis,
        } => TapeKernel::InlineReverseSequence {
            batch_axis: *batch_axis,
            time_axis: *time_axis,
        },

        // ── Vision / spatial ─────────────────────────────────────────────
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
        } => TapeKernel::InlineConv2d {
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
        },
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
        } => TapeKernel::InlineConvTranspose {
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
        },
        FloatOp::MaxPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => TapeKernel::InlineMaxPool2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
        },
        FloatOp::AvgPool2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        } => TapeKernel::InlineAvgPool2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
        },
        FloatOp::GlobalAvgPool {
            channels,
            spatial_h,
            spatial_w,
        } => TapeKernel::InlineGlobalAvgPool {
            channels: *channels,
            spatial_h: *spatial_h,
            spatial_w: *spatial_w,
        },
        FloatOp::Resize { mode } => TapeKernel::InlineResize { mode: *mode },
        FloatOp::PadOp { mode } => TapeKernel::InlinePad { mode: *mode },

        // ── KV cache ─────────────────────────────────────────────────────
        FloatOp::KvWrite {
            layer,
            n_kv_heads,
            head_dim,
            is_key,
            heads_first,
        } => TapeKernel::KvWrite {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
            is_key: *is_key,
            heads_first: *heads_first,
        },
        FloatOp::KvRead {
            layer,
            n_kv_heads,
            head_dim,
            heads_first,
        } => TapeKernel::KvRead {
            layer: *layer,
            n_kv_heads: *n_kv_heads,
            head_dim: *head_dim,
            heads_first: *heads_first,
        },

        // Transpose is handled separately (before this function).
        FloatOp::Transpose { .. } => TapeKernel::Passthrough,
    }
}

/// Pre-compute the output element size for a node.
///
/// Uses the compiled dtype when available, falling back to the op's
/// declared output dtype. Default: 4 (f32).
fn compute_elem_size(node_id: NodeId, op: &GraphOp, dtypes: &HashMap<NodeId, FloatDType>) -> u8 {
    // Try compiled dtype first (most reliable).
    if let Some(dtype) = dtypes.get(&node_id) {
        return dtype.byte_size() as u8;
    }
    // Infer from op's output dtype declaration.
    if let GraphOp::Float(fop) = op {
        match fop {
            FloatOp::IsNaN => return 1,
            FloatOp::Cast { to, .. } => return to.byte_size() as u8,
            FloatOp::Shape { .. } => return 8, // i64
            _ => {}
        }
    }
    4 // f32 default
}

/// Pre-compute the total output byte size for a node from compiled shapes.
///
/// Returns the product of shape dimensions × element size, or 0 if the
/// shape is unknown or contains a 0-sentinel (dynamic dimension).
fn compute_output_byte_hint(
    node_id: NodeId,
    shapes: &HashMap<NodeId, Vec<usize>>,
    elem_size: u8,
) -> u32 {
    let Some(shape) = shapes.get(&node_id) else {
        return 0;
    };
    if shape.is_empty() {
        return 0;
    }
    if shape.contains(&0) {
        return 0;
    }
    let n_elements: usize = shape.iter().product();
    let byte_size = n_elements.saturating_mul(elem_size as usize);
    if byte_size > u32::MAX as usize {
        0
    } else {
        byte_size as u32
    }
}

/// Compute the byte offset into the weight archive for LUT-GEMM constant prefetch.
///
/// Returns 0 for non-LUT-GEMM ops (no weight prefetch needed).
fn compute_weight_offset(
    kernel: &TapeKernel,
    constants: &hologram_graph::constant::ConstantStore,
) -> u32 {
    let cid = match kernel {
        TapeKernel::MatMulLut4(cid) | TapeKernel::MatMulLut8(cid) => *cid,
        _ => return 0,
    };
    match constants.get(cid) {
        Some(hologram_graph::constant::ConstantData::Deferred { source_id, .. }) => {
            *source_id as u32
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_graph::graph::edge;
    use hologram_graph::graph::Graph;

    fn make_simple_graph() -> (SerializedGraph, ExecutionSchedule) {
        let mut graph = Graph::new();
        let _input_id = graph.add_node(GraphOp::Input);
        let input_idx = graph.add_input("x");

        let relu_id = graph.add_node(GraphOp::Float(FloatOp::Relu));
        edge::connect_graph_input(&mut graph, input_idx, relu_id, 0);

        let out_id = graph.add_node(GraphOp::Output);
        edge::connect(&mut graph, relu_id, out_id, 0);
        graph.add_output("y", out_id);

        let sg = SerializedGraph::from_graph(&graph);
        let schedule = ExecutionSchedule::build(&graph).expect("schedule should build");
        (sg, schedule)
    }

    #[test]
    fn build_tape_from_simple_graph() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        assert!(
            !tape.instructions.is_empty(),
            "expected at least 1 instruction, got 0",
        );
    }

    #[test]
    fn tape_elem_size_defaults_to_f32() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        for instr in &tape.instructions {
            assert_eq!(instr.output_elem_size, 4);
        }
    }

    #[test]
    fn tape_kernel_is_enum_not_boxed() {
        let (sg, schedule) = make_simple_graph();
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");
        // Verify the Relu instruction is a Float variant (not a Box).
        for instr in &tape.instructions {
            match &instr.kernel {
                TapeKernel::InlineRelu | TapeKernel::Output => {}
                other => panic!(
                    "unexpected kernel variant: {:?}",
                    std::mem::discriminant(other)
                ),
            }
        }
    }

    /// Helper: build tape, seed arena, execute, and collect outputs.
    fn execute_graph(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        input_data: &[u8],
    ) -> Vec<(String, Vec<u8>)> {
        use crate::buffer::BufferArena;
        use crate::tape::TapeContext;
        use hologram_graph::constant::ConstantStore;

        let tape = build_tape(sg, schedule, None).expect("build_tape should succeed");
        let mut arena = BufferArena::with_capacity(sg.nodes.len());
        for node in &sg.nodes {
            if matches!(node.op, GraphOp::Input) {
                arena.insert_borrowed_with_elem_size(node.id, input_data, 4);
            }
        }
        tape.prewarm_arena(&mut arena);
        let constants = ConstantStore::default();
        let tape_ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &tape_ctx)
            .expect("tape execution should succeed");

        let mut outputs = Vec::new();
        for (i, name) in sg.output_names.iter().enumerate() {
            let node_id = sg.output_node_ids[i];
            let data = arena.take(node_id).unwrap_or_else(|_| {
                panic!("output '{}' at {:?} should be in arena", name, node_id)
            });
            outputs.push((name.clone(), data));
        }
        outputs
    }

    fn to_f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn from_f32_bytes(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// End-to-end: graph input → Relu → output, verify data flows through.
    #[test]
    fn tape_execute_and_collect_outputs() {
        let (sg, schedule) = make_simple_graph();
        let input_data = to_f32_bytes(&[-1.0, 2.0, -3.0, 4.0]);
        let outputs = execute_graph(&sg, &schedule, &input_data);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "y");
        assert!(!outputs[0].1.is_empty(), "output should not be empty");
        assert_eq!(from_f32_bytes(&outputs[0].1), vec![0.0, 2.0, 0.0, 4.0]);
    }

    /// Multi-op chain: Input → Relu → Neg → Output.
    /// Tests that data propagates through multiple ops.
    #[test]
    fn tape_execute_multi_op_chain() {
        let mut graph = Graph::new();
        let _input_id = graph.add_node(GraphOp::Input);
        let input_idx = graph.add_input("x");

        let relu_id = graph.add_node(GraphOp::Float(FloatOp::Relu));
        edge::connect_graph_input(&mut graph, input_idx, relu_id, 0);

        let neg_id = graph.add_node(GraphOp::Float(FloatOp::Neg));
        edge::connect(&mut graph, relu_id, neg_id, 0);

        let out_id = graph.add_node(GraphOp::Output);
        edge::connect(&mut graph, neg_id, out_id, 0);
        graph.add_output("y", out_id);

        let sg = SerializedGraph::from_graph(&graph);
        let schedule = ExecutionSchedule::build(&graph).expect("schedule should build");
        let input_data = to_f32_bytes(&[-1.0, 2.0, -3.0, 4.0]);
        let outputs = execute_graph(&sg, &schedule, &input_data);

        assert_eq!(outputs.len(), 1);
        // Relu([-1, 2, -3, 4]) = [0, 2, 0, 4]; Neg → [0, -2, 0, -4]
        assert_eq!(from_f32_bytes(&outputs[0].1), vec![0.0, -2.0, 0.0, -4.0]);
    }

    /// Output directly from graph input (identity pass-through).
    /// Tests that InputSource::GraphInput is correctly resolved.
    #[test]
    fn tape_execute_graph_input_passthrough() {
        let mut graph = Graph::new();
        let _input_id = graph.add_node(GraphOp::Input);
        let input_idx = graph.add_input("x");

        let out_id = graph.add_node(GraphOp::Output);
        edge::connect_graph_input(&mut graph, input_idx, out_id, 0);
        graph.add_output("y", out_id);

        let sg = SerializedGraph::from_graph(&graph);
        let schedule = ExecutionSchedule::build(&graph).expect("schedule should build");
        let input_data = to_f32_bytes(&[1.0, 2.0, 3.0]);
        let outputs = execute_graph(&sg, &schedule, &input_data);

        assert_eq!(outputs.len(), 1);
        assert_eq!(from_f32_bytes(&outputs[0].1), vec![1.0, 2.0, 3.0]);
    }

    /// ONNX-style graph: output_node_ids points to a compute node (no
    /// GraphOp::Output wrapper). Builds SerializedGraph directly, mimicking
    /// how ONNX import typically wires outputs.
    #[test]
    fn tape_execute_onnx_style_no_output_wrapper() {
        use crate::buffer::BufferArena;
        use crate::eval::schedule_bridge::build_schedule;
        use crate::tape::TapeContext;
        use hologram_graph::constant::ConstantStore;
        use hologram_graph::graph::node::{InputSlot, Node};

        fn nid(n: u32) -> NodeId {
            NodeId::new(n, 0)
        }

        // Input(0) → Relu(1), output registered at Relu node (no Output wrapper)
        let sg = SerializedGraph {
            nodes: vec![
                Node {
                    id: nid(0),
                    op: GraphOp::Input,
                    inputs: Default::default(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(1),
                    op: GraphOp::Float(FloatOp::Relu),
                    inputs: vec![InputSlot::from_node(nid(0))].into_iter().collect(),
                    num_outputs: 1,
                },
            ],
            input_names: vec!["input".into()],
            output_names: vec!["output".into()],
            output_node_ids: vec![nid(1)], // Points to compute node, no Output wrapper
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        };

        let schedule = build_schedule(&sg).expect("schedule should build");
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");

        let input_data = to_f32_bytes(&[-1.0, 2.0, -3.0, 4.0]);
        let mut arena = BufferArena::with_capacity(sg.nodes.len());
        arena.insert_borrowed_with_elem_size(nid(0), &input_data, 4);

        tape.prewarm_arena(&mut arena);
        let constants = ConstantStore::default();
        let tape_ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &tape_ctx)
            .expect("tape execution should succeed");

        // Output registered at Relu node (no Output wrapper)
        let node_id = sg.output_node_ids[0];
        let output_data = arena.take(node_id).expect("output should be in arena");
        assert!(
            !output_data.is_empty(),
            "ONNX-style output should not be empty"
        );
        assert_eq!(from_f32_bytes(&output_data), vec![0.0, 2.0, 0.0, 4.0]);
    }

    /// ONNX-style graph with Output wrapper: Input(0) → Relu(1) → Output(2).
    /// Output registered at the Output wrapper node.
    #[test]
    fn tape_execute_onnx_style_with_output_wrapper() {
        use crate::buffer::BufferArena;
        use crate::eval::schedule_bridge::build_schedule;
        use crate::tape::TapeContext;
        use hologram_graph::constant::ConstantStore;
        use hologram_graph::graph::node::{InputSlot, Node};

        fn nid(n: u32) -> NodeId {
            NodeId::new(n, 0)
        }

        // Input(0) → Relu(1) → Output(2), output registered at Output wrapper
        let sg = SerializedGraph {
            nodes: vec![
                Node {
                    id: nid(0),
                    op: GraphOp::Input,
                    inputs: Default::default(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(1),
                    op: GraphOp::Float(FloatOp::Relu),
                    inputs: vec![InputSlot::from_node(nid(0))].into_iter().collect(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(2),
                    op: GraphOp::Output,
                    inputs: vec![InputSlot::from_node(nid(1))].into_iter().collect(),
                    num_outputs: 1,
                },
            ],
            input_names: vec!["input".into()],
            output_names: vec!["output".into()],
            output_node_ids: vec![nid(2)], // Points to Output wrapper
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        };

        let schedule = build_schedule(&sg).expect("schedule should build");
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");

        let input_data = to_f32_bytes(&[-1.0, 2.0, -3.0, 4.0]);
        let mut arena = BufferArena::with_capacity(sg.nodes.len());
        arena.insert_borrowed_with_elem_size(nid(0), &input_data, 4);

        tape.prewarm_arena(&mut arena);
        let constants = ConstantStore::default();
        let tape_ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &tape_ctx)
            .expect("tape execution should succeed");

        let node_id = sg.output_node_ids[0];
        let output_data = arena.take(node_id).expect("output should be in arena");
        assert!(
            !output_data.is_empty(),
            "output with wrapper should not be empty"
        );
        assert_eq!(from_f32_bytes(&output_data), vec![0.0, 2.0, 0.0, 4.0]);
    }

    /// ONNX-style multi-layer chain with GraphInput source.
    /// Input(0) → Relu(1) → Neg(2) → Output(3), first op uses GraphInput.
    #[test]
    fn tape_execute_onnx_style_graph_input_to_compute() {
        use crate::buffer::BufferArena;
        use crate::eval::schedule_bridge::build_schedule;
        use crate::tape::TapeContext;
        use hologram_graph::constant::ConstantStore;
        use hologram_graph::graph::node::{InputSlot, Node};

        fn nid(n: u32) -> NodeId {
            NodeId::new(n, 0)
        }

        // Input(0) with GraphInput edge to Relu(1), then Node edge to Neg(2) → Output(3)
        let sg = SerializedGraph {
            nodes: vec![
                Node {
                    id: nid(0),
                    op: GraphOp::Input,
                    inputs: Default::default(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(1),
                    op: GraphOp::Float(FloatOp::Relu),
                    inputs: vec![InputSlot::from_graph_input(0)].into_iter().collect(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(2),
                    op: GraphOp::Float(FloatOp::Neg),
                    inputs: vec![InputSlot::from_node(nid(1))].into_iter().collect(),
                    num_outputs: 1,
                },
                Node {
                    id: nid(3),
                    op: GraphOp::Output,
                    inputs: vec![InputSlot::from_node(nid(2))].into_iter().collect(),
                    num_outputs: 1,
                },
            ],
            input_names: vec!["input".into()],
            output_names: vec!["result".into()],
            output_node_ids: vec![nid(3)],
            constants: ConstantStore::new(),
            constant_shapes: Vec::new(),
            node_shapes: Vec::new(),
            node_dtypes: Vec::new(),
        };

        let schedule = build_schedule(&sg).expect("schedule should build");
        let tape = build_tape(&sg, &schedule, None).expect("build_tape should succeed");

        let input_data = to_f32_bytes(&[-1.0, 2.0, -3.0, 4.0]);
        let mut arena = BufferArena::with_capacity(sg.nodes.len());
        arena.insert_borrowed_with_elem_size(nid(0), &input_data, 4);

        tape.prewarm_arena(&mut arena);
        let constants = ConstantStore::default();
        let tape_ctx = TapeContext::new(&constants, &[]);
        tape.execute(&mut arena, &tape_ctx)
            .expect("tape execution should succeed");

        let node_id = sg.output_node_ids[0];
        let output_data = arena.take(node_id).expect("output should be in arena");
        assert!(
            !output_data.is_empty(),
            "GraphInput→compute→output should produce data"
        );
        // Relu([-1,2,-3,4])=[0,2,0,4]; Neg→[0,-2,0,-4]
        assert_eq!(from_f32_bytes(&output_data), vec![0.0, -2.0, 0.0, -4.0]);
    }
}
