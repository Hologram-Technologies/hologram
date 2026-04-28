//! `SemanticOp` → forward `KernelCall` lowering.
//!
//! One match arm per semantic variant; per-op `build_*_call` helpers
//! package the planner-resolved `SlotSpan`s and attribute fields into
//! the matching `Call` struct. Helpers that need shape information
//! also consume the chain (via `super::require_tensor`).

use super::require_tensor;
use crate::chain::{TransformChain, TransformNode};
use crate::error::PlanError;
use crate::plan::{
    AddCall, AddRmsNormCall, AddressTable, AttentionCall, BinaryCall, ClipCall, ConcatCall,
    Conv2dCall, ConvTransposeCall, CumSumCall, ExpandCall, GemmCall, GlobalAvgPoolCall,
    GroupNormCall, KernelCall, LrnCall, MatMulCall, NormFullCall, NormScaleCall, PadCall,
    Pool2dCall, Pool2dKind, ReduceCall, ReduceKind, ReshapeCall, ResizeCall, RotaryEmbeddingCall,
    SliceCall, SlotSpan, SoftmaxCall, TransposeCall, UnaryCall, UnaryKind, WhereCall,
};
use hologram_ops::{
    AttentionAttrs, ClipAttrs, ConcatAttrs, Conv2dAttrs, ConvTransposeAttrs, CumSumAttrs,
    ExpandAttrs, GemmAttrs, GlobalAvgPoolAttrs, GroupNormAttrs, LrnAttrs, MatMulAttrs, NormAttrs,
    PadAttrs, Pool2dAttrs, ReduceAttrs, ResizeAttrs, RotaryEmbeddingAttrs, SemanticOp, SliceAttrs,
    TransposeAttrs,
};

/// Lower a single forward node to its `KernelCall`.
pub(super) fn lower_node(
    chain: &TransformChain,
    table: &AddressTable,
    node: &TransformNode,
) -> Result<KernelCall, PlanError> {
    check_arity(node)?;
    if let Some(kind) = unary_kind_of(node.op) {
        return Ok(KernelCall::Unary(unary_call(table, node), kind));
    }
    Ok(match node.op {
        SemanticOp::Add => KernelCall::Add(add_call(table, node)),
        SemanticOp::Sub => KernelCall::Sub(binary_call(table, node)),
        SemanticOp::Mul => KernelCall::Mul(binary_call(table, node)),
        SemanticOp::Div => KernelCall::Div(binary_call(table, node)),
        SemanticOp::Pow => KernelCall::Pow(binary_call(table, node)),
        SemanticOp::Mod => KernelCall::Mod(binary_call(table, node)),
        SemanticOp::Min => KernelCall::Min(binary_call(table, node)),
        SemanticOp::Max => KernelCall::Max(binary_call(table, node)),
        SemanticOp::Equal => KernelCall::Equal(binary_call(table, node)),
        SemanticOp::Less => KernelCall::Less(binary_call(table, node)),
        SemanticOp::LessOrEqual => KernelCall::LessOrEqual(binary_call(table, node)),
        SemanticOp::Greater => KernelCall::Greater(binary_call(table, node)),
        SemanticOp::GreaterOrEqual => KernelCall::GreaterOrEqual(binary_call(table, node)),
        SemanticOp::And => KernelCall::And(binary_call(table, node)),
        SemanticOp::Or => KernelCall::Or(binary_call(table, node)),
        SemanticOp::Xor => KernelCall::Xor(binary_call(table, node)),
        SemanticOp::FusedSwiGlu => KernelCall::FusedSwiGlu(binary_call(table, node)),
        SemanticOp::MatMul(a) => KernelCall::MatMul(matmul_call(table, node, a)),
        SemanticOp::Softmax(a) => KernelCall::Softmax(softmax_call(table, node, a.size)),
        SemanticOp::LogSoftmax(a) => KernelCall::LogSoftmax(softmax_call(table, node, a.size)),
        SemanticOp::Reshape => KernelCall::Reshape(reshape_call(table, node)),
        SemanticOp::Transpose(a) => KernelCall::Transpose(transpose_call(chain, table, node, a)?),
        SemanticOp::Slice(a) => KernelCall::Slice(slice_call(table, node, a)?),
        SemanticOp::Concat(a) => KernelCall::Concat(concat_call(table, node, a)),
        SemanticOp::RmsNorm(a) => KernelCall::RmsNorm(norm_scale_call(table, node, a)),
        SemanticOp::LayerNorm(a) => KernelCall::LayerNorm(norm_full_call(table, node, a)),
        SemanticOp::InstanceNorm(a) => KernelCall::InstanceNorm(norm_scale_call(table, node, a)),
        SemanticOp::GroupNorm(a) => KernelCall::GroupNorm(group_norm_call(table, node, a)),
        SemanticOp::AddRmsNorm(a) => KernelCall::AddRmsNorm(add_rms_norm_call(table, node, a)),
        SemanticOp::Conv2d(a) => KernelCall::Conv2d(conv2d_call(chain, table, node, a)?),
        SemanticOp::ReduceSum(a) => {
            KernelCall::Reduce(reduce_call(table, node, a), ReduceKind::Sum)
        }
        SemanticOp::ReduceMean(a) => {
            KernelCall::Reduce(reduce_call(table, node, a), ReduceKind::Mean)
        }
        SemanticOp::ReduceMax(a) => {
            KernelCall::Reduce(reduce_call(table, node, a), ReduceKind::Max)
        }
        SemanticOp::ReduceMin(a) => {
            KernelCall::Reduce(reduce_call(table, node, a), ReduceKind::Min)
        }
        SemanticOp::ReduceProd(a) => {
            KernelCall::Reduce(reduce_call(table, node, a), ReduceKind::Prod)
        }
        SemanticOp::MaxPool2d(a) => {
            KernelCall::Pool2d(pool2d_call(chain, table, node, a)?, Pool2dKind::Max)
        }
        SemanticOp::AvgPool2d(a) => {
            KernelCall::Pool2d(pool2d_call(chain, table, node, a)?, Pool2dKind::Avg)
        }
        SemanticOp::GlobalAvgPool(a) => {
            KernelCall::GlobalAvgPool(global_avg_pool_call(chain, table, node, a)?)
        }
        SemanticOp::Where => KernelCall::Where(where_call(table, node)),
        SemanticOp::Clip(a) => KernelCall::Clip(clip_call(table, node, a)),
        SemanticOp::CumSum(a) => KernelCall::CumSum(cumsum_call(chain, table, node, a)?),
        SemanticOp::Pad(a) => KernelCall::Pad(pad_call(chain, table, node, a)?),
        SemanticOp::Resize(a) => {
            let call = resize_call(chain, table, node, a)?;
            match a.mode {
                0 => KernelCall::ResizeNearest(call),
                1 => KernelCall::ResizeLinear(call),
                _ => {
                    return Err(PlanError::ShapeMismatch {
                        op: "resize",
                        detail: "supported modes: 0 (nearest), 1 (linear)",
                    });
                }
            }
        }
        SemanticOp::Lrn(a) => KernelCall::Lrn(lrn_call(chain, table, node, a)?),
        SemanticOp::ConvTranspose2d(a) => {
            KernelCall::ConvTranspose2d(conv_transpose_call(chain, table, node, a)?)
        }
        SemanticOp::Gemm(a) => KernelCall::Gemm(gemm_call(table, node, a)),
        SemanticOp::Expand(a) => KernelCall::Expand(expand_call(chain, table, node, a)?),
        SemanticOp::RotaryEmbedding(a) => {
            KernelCall::RotaryEmbedding(rotary_call(chain, table, node, a)?)
        }
        SemanticOp::Attention(a) => KernelCall::Attention(attention_call(chain, table, node, a)?),
        // The unary path is handled above; any remaining op is a planner gap.
        other => return Err(PlanError::UnsupportedOp(other.name())),
    })
}

fn check_arity(node: &TransformNode) -> Result<(), PlanError> {
    let expected = node.op.arity() as usize;
    if node.inputs.len() != expected {
        return Err(PlanError::ArityMismatch {
            op: node.op.name(),
            expected,
            actual: node.inputs.len(),
        });
    }
    Ok(())
}

fn unary_kind_of(op: SemanticOp) -> Option<UnaryKind> {
    Some(match op {
        SemanticOp::Neg => UnaryKind::Neg,
        SemanticOp::Relu => UnaryKind::Relu,
        SemanticOp::Gelu => UnaryKind::Gelu,
        SemanticOp::Silu => UnaryKind::Silu,
        SemanticOp::Tanh => UnaryKind::Tanh,
        SemanticOp::Sigmoid => UnaryKind::Sigmoid,
        SemanticOp::Exp => UnaryKind::Exp,
        SemanticOp::Log => UnaryKind::Log,
        SemanticOp::Sqrt => UnaryKind::Sqrt,
        SemanticOp::Abs => UnaryKind::Abs,
        SemanticOp::Reciprocal => UnaryKind::Reciprocal,
        SemanticOp::Cos => UnaryKind::Cos,
        SemanticOp::Sin => UnaryKind::Sin,
        SemanticOp::Sign => UnaryKind::Sign,
        SemanticOp::Floor => UnaryKind::Floor,
        SemanticOp::Ceil => UnaryKind::Ceil,
        SemanticOp::Round => UnaryKind::Round,
        SemanticOp::Erf => UnaryKind::Erf,
        SemanticOp::Not => UnaryKind::Not,
        SemanticOp::IsNaN => UnaryKind::IsNaN,
        _ => return None,
    })
}

// ── Trivial single-shape builders ────────────────────────────────────────────

fn unary_call(t: &AddressTable, n: &TransformNode) -> UnaryCall {
    UnaryCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
    }
}

fn binary_call(t: &AddressTable, n: &TransformNode) -> BinaryCall {
    BinaryCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        c: t.out_span(n, 0),
    }
}

fn add_call(t: &AddressTable, n: &TransformNode) -> AddCall {
    AddCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        c: t.out_span(n, 0),
    }
}

fn reshape_call(t: &AddressTable, n: &TransformNode) -> ReshapeCall {
    ReshapeCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
    }
}

fn where_call(t: &AddressTable, n: &TransformNode) -> WhereCall {
    WhereCall {
        condition: t.in_span(n, 0),
        x: t.in_span(n, 1),
        y: t.in_span(n, 2),
        output: t.out_span(n, 0),
    }
}

fn matmul_call(t: &AddressTable, n: &TransformNode, a: MatMulAttrs) -> MatMulCall {
    MatMulCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        c: t.out_span(n, 0),
        m: a.m as usize,
        k: a.k as usize,
        n: a.n as usize,
    }
}

fn softmax_call(t: &AddressTable, n: &TransformNode, axis_size: u32) -> SoftmaxCall {
    SoftmaxCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        size: axis_size as usize,
    }
}

fn reduce_call(t: &AddressTable, n: &TransformNode, a: ReduceAttrs) -> ReduceCall {
    ReduceCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        size: a.size as usize,
    }
}

fn norm_scale_call(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> NormScaleCall {
    NormScaleCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        output: t.out_span(n, 0),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn norm_full_call(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> NormFullCall {
    NormFullCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        bias: t.in_span(n, 2),
        output: t.out_span(n, 0),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn group_norm_call(t: &AddressTable, n: &TransformNode, a: GroupNormAttrs) -> GroupNormCall {
    GroupNormCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        bias: t.in_span(n, 2),
        output: t.out_span(n, 0),
        num_groups: a.num_groups,
        epsilon: a.epsilon,
    }
}

fn add_rms_norm_call(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> AddRmsNormCall {
    AddRmsNormCall {
        residual: t.in_span(n, 0),
        input: t.in_span(n, 1),
        weight: t.in_span(n, 2),
        output: t.out_span(n, 0),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn concat_call(t: &AddressTable, n: &TransformNode, a: ConcatAttrs) -> ConcatCall {
    ConcatCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        output: t.out_span(n, 0),
        size_a: a.size_a,
        size_b: a.size_b,
    }
}

fn clip_call(t: &AddressTable, n: &TransformNode, a: ClipAttrs) -> ClipCall {
    ClipCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        min_bits: a.min,
        max_bits: a.max,
    }
}

fn gemm_call(t: &AddressTable, n: &TransformNode, a: GemmAttrs) -> GemmCall {
    let c = if n.inputs.len() >= 3 {
        t.in_span(n, 2)
    } else {
        SlotSpan::empty(0)
    };
    GemmCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        c,
        y: t.out_span(n, 0),
        m: a.m as usize,
        k: a.k as usize,
        n: a.n as usize,
        trans_a: a.trans_a,
        trans_b: a.trans_b,
        alpha_bits: a.alpha,
        beta_bits: a.beta,
    }
}

// ── Shape-validating builders ────────────────────────────────────────────────

fn slice_call(t: &AddressTable, n: &TransformNode, a: SliceAttrs) -> Result<SliceCall, PlanError> {
    if a.axis_from_end != 0 {
        return Err(PlanError::ShapeMismatch {
            op: "slice",
            detail: "reference kernel supports last-axis slice only",
        });
    }
    Ok(SliceCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        axis_size: a.axis_size,
        start: a.start,
        end: a.end,
    })
}

fn transpose_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: TransposeAttrs,
) -> Result<TransposeCall, PlanError> {
    let nd = a.ndim as usize;
    if nd > 4 {
        return Err(PlanError::ShapeMismatch {
            op: "transpose",
            detail: "reference kernel supports rank ≤ 4",
        });
    }
    let dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if dims.len() != nd {
        return Err(PlanError::ShapeMismatch {
            op: "transpose",
            detail: "input rank must match TransposeAttrs.ndim",
        });
    }
    let mut input_dims = [0_u32; 4];
    let mut perm = [0_u8; 4];
    for i in 0..nd {
        input_dims[i] = dims[i] as u32;
        perm[i] = a.perm[i];
    }
    Ok(TransposeCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        input_dims,
        perm,
        ndim: a.ndim,
    })
}

fn cumsum_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    _a: CumSumAttrs,
) -> Result<CumSumCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let last = *in_dims.last().ok_or(PlanError::ShapeMismatch {
        op: "cum_sum",
        detail: "input must have at least one dimension",
    })?;
    Ok(CumSumCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        size: last,
    })
}

fn pad_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: PadAttrs,
) -> Result<PadCall, PlanError> {
    if a.mode > 2 {
        return Err(PlanError::ShapeMismatch {
            op: "pad",
            detail: "supported modes: 0 (constant), 1 (reflect), 2 (edge)",
        });
    }
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "pad",
            detail: "input must be 4-D (NCHW)",
        });
    }
    Ok(PadCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c: in_dims[1] as u32,
        h_in: in_dims[2] as u32,
        w_in: in_dims[3] as u32,
        pad_h: a.pad_h,
        pad_w: a.pad_w,
        value_bits: a.value,
        mode: a.mode,
    })
}

fn resize_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    _a: ResizeAttrs,
) -> Result<ResizeCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let out_dims = require_tensor(chain, n.outputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "resize",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    Ok(ResizeCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c: in_dims[1] as u32,
        h_in: in_dims[2] as u32,
        w_in: in_dims[3] as u32,
        h_out: out_dims[2] as u32,
        w_out: out_dims[3] as u32,
    })
}

fn pool2d_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: Pool2dAttrs,
) -> Result<Pool2dCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let out_dims = require_tensor(chain, n.outputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "pool2d",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    Ok(Pool2dCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c: in_dims[1] as u32,
        h_in: in_dims[2] as u32,
        w_in: in_dims[3] as u32,
        h_out: out_dims[2] as u32,
        w_out: out_dims[3] as u32,
        kernel_h: a.kernel_h,
        kernel_w: a.kernel_w,
        stride_h: a.stride_h,
        stride_w: a.stride_w,
        pad_h: a.pad_h,
        pad_w: a.pad_w,
    })
}

fn global_avg_pool_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: GlobalAvgPoolAttrs,
) -> Result<GlobalAvgPoolCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "global_avg_pool",
            detail: "input tensor must be 4-D (NCHW)",
        });
    }
    Ok(GlobalAvgPoolCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c: a.channels,
        h: a.spatial_h,
        w: a.spatial_w,
    })
}

fn lrn_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: LrnAttrs,
) -> Result<LrnCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "lrn",
            detail: "input must be 4-D (NCHW)",
        });
    }
    Ok(LrnCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c: in_dims[1] as u32,
        h: in_dims[2] as u32,
        w: in_dims[3] as u32,
        size: a.size,
        alpha_bits: a.alpha,
        beta_bits: a.beta,
        bias_bits: a.bias,
    })
}

fn conv2d_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: Conv2dAttrs,
) -> Result<Conv2dCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let out_dims = require_tensor(chain, n.outputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "conv2d",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    let bias = if n.inputs.len() >= 3 {
        t.in_span(n, 2)
    } else {
        SlotSpan::empty(0)
    };
    Ok(Conv2dCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        bias,
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c_in: in_dims[1] as u32,
        c_out: out_dims[1] as u32,
        h_in: in_dims[2] as u32,
        w_in: in_dims[3] as u32,
        h_out: out_dims[2] as u32,
        w_out: out_dims[3] as u32,
        kernel_h: a.kernel_h,
        kernel_w: a.kernel_w,
        stride_h: a.stride_h,
        stride_w: a.stride_w,
        pad_h: a.pad_h,
        pad_w: a.pad_w,
        dilation_h: a.dilation_h,
        dilation_w: a.dilation_w,
        group: a.group.max(1),
    })
}

fn conv_transpose_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: ConvTransposeAttrs,
) -> Result<ConvTransposeCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let out_dims = require_tensor(chain, n.outputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "conv_transpose_2d",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    let bias = if n.inputs.len() >= 3 {
        t.in_span(n, 2)
    } else {
        SlotSpan::empty(0)
    };
    Ok(ConvTransposeCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        bias,
        output: t.out_span(n, 0),
        n: in_dims[0] as u32,
        c_in: in_dims[1] as u32,
        c_out: out_dims[1] as u32,
        h_in: in_dims[2] as u32,
        w_in: in_dims[3] as u32,
        h_out: out_dims[2] as u32,
        w_out: out_dims[3] as u32,
        kernel_h: a.kernel_h,
        kernel_w: a.kernel_w,
        stride_h: a.stride_h,
        stride_w: a.stride_w,
        pad_h: a.pad_h,
        pad_w: a.pad_w,
        dilation_h: a.dilation_h,
        dilation_w: a.dilation_w,
        group: a.group.max(1),
    })
}

fn expand_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: ExpandAttrs,
) -> Result<ExpandCall, PlanError> {
    let nd = a.ndim as usize;
    if nd > 8 {
        return Err(PlanError::ShapeMismatch {
            op: "expand",
            detail: "reference kernel supports rank ≤ 8",
        });
    }
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != nd {
        return Err(PlanError::ShapeMismatch {
            op: "expand",
            detail: "input rank must match ExpandAttrs.ndim",
        });
    }
    let mut input_dims = [0_u32; 8];
    for (i, &d) in in_dims.iter().enumerate() {
        input_dims[i] = d as u32;
    }
    Ok(ExpandCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        input_dims,
        target_dims: a.target_shape,
        ndim: a.ndim,
    })
}

fn rotary_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: RotaryEmbeddingAttrs,
) -> Result<RotaryEmbeddingCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() < 3 {
        return Err(PlanError::ShapeMismatch {
            op: "rotary_embedding",
            detail: "input must be at least 3-D ([..., seq, n_heads, head_dim])",
        });
    }
    let head_dim = *in_dims.last().unwrap();
    let n_heads = in_dims[in_dims.len() - 2];
    let seq = in_dims[in_dims.len() - 3];
    if head_dim != a.dim as usize || n_heads != a.n_heads as usize {
        return Err(PlanError::ShapeMismatch {
            op: "rotary_embedding",
            detail: "input shape's [n_heads, head_dim] must match attrs",
        });
    }
    if !a.dim.is_multiple_of(2) {
        return Err(PlanError::ShapeMismatch {
            op: "rotary_embedding",
            detail: "rotation `dim` must be even",
        });
    }
    let batch: usize = in_dims[..in_dims.len() - 3]
        .iter()
        .product::<usize>()
        .max(1);
    Ok(RotaryEmbeddingCall {
        input: t.in_span(n, 0),
        output: t.out_span(n, 0),
        batch: batch as u32,
        seq: seq as u32,
        n_heads: a.n_heads,
        dim: a.dim,
        base_bits: a.base,
    })
}

fn attention_call(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: AttentionAttrs,
) -> Result<AttentionCall, PlanError> {
    if a.num_kv_heads == 0 || !a.num_q_heads.is_multiple_of(a.num_kv_heads) {
        return Err(PlanError::ShapeMismatch {
            op: "attention",
            detail: "num_q_heads must be a positive multiple of num_kv_heads",
        });
    }
    let q_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let k_dims = require_tensor(chain, n.inputs[1].tensor)?.dims.as_slice();
    let v_dims = require_tensor(chain, n.inputs[2].tensor)?.dims.as_slice();
    if q_dims.len() < 4 || k_dims.len() < 4 || v_dims.len() < 4 {
        return Err(PlanError::ShapeMismatch {
            op: "attention",
            detail: "Q/K/V must be 4-D ([batch, n_heads, seq, head_dim])",
        });
    }
    let head_dim = a.head_dim as usize;
    let nqh = a.num_q_heads as usize;
    let nkh = a.num_kv_heads as usize;
    if q_dims[1] != nqh
        || k_dims[1] != nkh
        || v_dims[1] != nkh
        || q_dims[3] != head_dim
        || k_dims[3] != head_dim
        || v_dims[3] != head_dim
    {
        return Err(PlanError::ShapeMismatch {
            op: "attention",
            detail: "Q/K/V shapes do not match attention attrs",
        });
    }
    let seq_q = q_dims[2];
    let seq_kv = k_dims[2];
    if v_dims[2] != seq_kv {
        return Err(PlanError::ShapeMismatch {
            op: "attention",
            detail: "K and V must share seq_kv length",
        });
    }
    let batch: usize = q_dims[..1].iter().product::<usize>().max(1);
    Ok(AttentionCall {
        q: t.in_span(n, 0),
        k: t.in_span(n, 1),
        v: t.in_span(n, 2),
        output: t.out_span(n, 0),
        batch: batch as u32,
        num_q_heads: a.num_q_heads,
        num_kv_heads: a.num_kv_heads,
        head_dim: a.head_dim,
        seq_q: seq_q as u32,
        seq_kv: seq_kv as u32,
        scale_bits: a.scale,
        causal: a.causal,
    })
}
