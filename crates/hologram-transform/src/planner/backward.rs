//! `BackwardRule` → `KernelCall` emission for differentiable ops.
//!
//! `emit` consumes one node + its rule and pushes the resulting
//! backward `KernelCall`(s) onto the planner's accumulator. The
//! per-rule helpers package planner-resolved span / grad-span
//! addresses into the matching `*GradCall` struct.

use super::require_tensor;
use crate::chain::{TransformChain, TransformNode};
use crate::error::PlanError;
use crate::plan::{
    AddGradCall, AddRmsNormGradCall, AddressTable, AttentionGradCall, ConcatGradCall,
    Conv2dGradCall, ConvTranspose2dGradCall, DivGradCall, FusedSwiGluGradCall,
    GlobalAvgPoolGradCall, GroupNormGradCall, InstanceNormGradCall, KernelCall, LayerNormGradCall,
    MatMulGradACall, MatMulGradBCall, MinMaxGradCall, MinMaxGradKind, MulGradCall, NegGradCall,
    Pool2dGradCall, Pool2dKind, PowGradCall, ReduceArgGradCall, ReduceArgGradKind, ReduceGradCall,
    ReduceGradKind, ReduceProdGradCall, RmsNormGradCall, SliceGradCall, SlotSpan, SoftmaxGradCall,
    SoftmaxGradKind, SubGradCall, TransposeGradCall, UnaryGradCall, UnaryGradKind,
};
use hologram_ops::{
    AttentionAttrs, BackwardRule, ConcatAttrs, Conv2dAttrs, ConvTransposeAttrs, GlobalAvgPoolAttrs,
    GroupNormAttrs, MatMulAttrs, NormAttrs, Pool2dAttrs, SemanticOp, SliceAttrs, TransposeAttrs,
};

/// Push the backward `KernelCall`(s) for a node onto `out`.
pub(super) fn emit(
    chain: &TransformChain,
    table: &AddressTable,
    node: &TransformNode,
    rule: BackwardRule,
    out: &mut Vec<KernelCall>,
) -> Result<(), PlanError> {
    require_grad_inputs(chain, node)?;

    // Helper: emit one `KernelCall::UnaryGrad` from a `UnaryGradKind`
    // and a `from_input` (true → forward input as `source`, false →
    // forward output).
    let emit_unary = |out: &mut Vec<KernelCall>, kind, from_input| {
        let call = if from_input {
            unary_grad_from_input(table, node)
        } else {
            unary_grad_from_output(table, node)
        };
        out.push(KernelCall::UnaryGrad(call, kind));
    };

    match rule {
        BackwardRule::AddBackward => out.push(KernelCall::AddGrad(add_grad(table, node))),
        BackwardRule::SubBackward => out.push(KernelCall::SubGrad(sub_grad(table, node))),
        BackwardRule::MulBackward => out.push(KernelCall::MulGrad(mul_grad(table, node))),
        BackwardRule::DivBackward => out.push(KernelCall::DivGrad(div_grad(table, node))),
        BackwardRule::NegBackward => out.push(KernelCall::NegGrad(neg_grad(table, node))),
        BackwardRule::ReluBackward => emit_unary(out, UnaryGradKind::Relu, true),
        BackwardRule::SigmoidBackward => emit_unary(out, UnaryGradKind::Sigmoid, false),
        BackwardRule::TanhBackward => emit_unary(out, UnaryGradKind::Tanh, false),
        BackwardRule::ExpBackward => emit_unary(out, UnaryGradKind::Exp, false),
        BackwardRule::LogBackward => emit_unary(out, UnaryGradKind::Log, true),
        BackwardRule::SqrtBackward => emit_unary(out, UnaryGradKind::Sqrt, false),
        BackwardRule::AbsBackward => emit_unary(out, UnaryGradKind::Abs, true),
        BackwardRule::ReciprocalBackward => emit_unary(out, UnaryGradKind::Reciprocal, false),
        BackwardRule::GeluBackward => emit_unary(out, UnaryGradKind::Gelu, true),
        BackwardRule::SiluBackward => emit_unary(out, UnaryGradKind::Silu, true),
        BackwardRule::MinBackward => out.push(KernelCall::MinMaxGrad(
            min_max_grad(table, node),
            MinMaxGradKind::Min,
        )),
        BackwardRule::MaxBackward => out.push(KernelCall::MinMaxGrad(
            min_max_grad(table, node),
            MinMaxGradKind::Max,
        )),
        BackwardRule::ReduceSumBackward => out.push(KernelCall::ReduceGrad(
            reduce_grad(chain, table, node)?,
            ReduceGradKind::Sum,
        )),
        BackwardRule::ReduceMeanBackward => out.push(KernelCall::ReduceGrad(
            reduce_grad(chain, table, node)?,
            ReduceGradKind::Mean,
        )),
        BackwardRule::MatMulBackward => {
            let attrs = matmul_attrs_of(node)?;
            out.push(KernelCall::MatMulGradA(matmul_grad_a(table, node, attrs)));
            out.push(KernelCall::MatMulGradB(matmul_grad_b(table, node, attrs)));
        }
        BackwardRule::ConcatBackward => {
            let attrs = concat_attrs_of(node)?;
            out.push(KernelCall::ConcatGrad(concat_grad(table, node, attrs)));
        }
        BackwardRule::SliceBackward => {
            let attrs = slice_attrs_of(node)?;
            out.push(KernelCall::SliceGrad(slice_grad(table, node, attrs)));
        }
        BackwardRule::TransposeBackward => {
            let attrs = transpose_attrs_of(node)?;
            out.push(KernelCall::TransposeGrad(transpose_grad(
                chain, table, node, attrs,
            )?));
        }
        BackwardRule::PowBackward => out.push(KernelCall::PowGrad(pow_grad(table, node))),
        BackwardRule::SoftmaxBackward => out.push(KernelCall::SoftmaxGrad(
            softmax_grad(chain, table, node)?,
            SoftmaxGradKind::Softmax,
        )),
        BackwardRule::LogSoftmaxBackward => out.push(KernelCall::SoftmaxGrad(
            softmax_grad(chain, table, node)?,
            SoftmaxGradKind::LogSoftmax,
        )),
        BackwardRule::ReduceMaxBackward => out.push(KernelCall::ReduceArgGrad(
            reduce_arg_grad(chain, table, node)?,
            ReduceArgGradKind::Max,
        )),
        BackwardRule::ReduceMinBackward => out.push(KernelCall::ReduceArgGrad(
            reduce_arg_grad(chain, table, node)?,
            ReduceArgGradKind::Min,
        )),
        BackwardRule::ReduceProdBackward => out.push(KernelCall::ReduceProdGrad(reduce_prod_grad(
            chain, table, node,
        )?)),
        BackwardRule::RmsNormBackward => {
            let attrs = norm_attrs_of(node, "rms_norm")?;
            out.push(KernelCall::RmsNormGrad(rms_norm_grad(table, node, attrs)));
        }
        BackwardRule::LayerNormBackward => {
            let attrs = norm_attrs_of(node, "layer_norm")?;
            out.push(KernelCall::LayerNormGrad(layer_norm_grad(
                table, node, attrs,
            )));
        }
        BackwardRule::InstanceNormBackward => {
            let attrs = norm_attrs_of(node, "instance_norm")?;
            out.push(KernelCall::InstanceNormGrad(instance_norm_grad(
                table, node, attrs,
            )));
        }
        BackwardRule::AddRmsNormBackward => {
            let attrs = norm_attrs_of(node, "add_rms_norm")?;
            out.push(KernelCall::AddRmsNormGrad(add_rms_norm_grad(
                table, node, attrs,
            )));
        }
        BackwardRule::GlobalAvgPoolBackward => {
            let attrs = global_avg_pool_attrs_of(node)?;
            out.push(KernelCall::GlobalAvgPoolGrad(global_avg_pool_grad(
                chain, table, node, attrs,
            )?));
        }
        BackwardRule::AvgPool2dBackward => {
            let attrs = pool2d_attrs_of(node, "avg_pool_2d")?;
            out.push(KernelCall::Pool2dGrad(
                pool2d_grad(chain, table, node, attrs)?,
                Pool2dKind::Avg,
            ));
        }
        BackwardRule::MaxPool2dBackward => {
            let attrs = pool2d_attrs_of(node, "max_pool_2d")?;
            out.push(KernelCall::Pool2dGrad(
                pool2d_grad(chain, table, node, attrs)?,
                Pool2dKind::Max,
            ));
        }
        BackwardRule::GroupNormBackward => {
            let attrs = group_norm_attrs_of(node)?;
            out.push(KernelCall::GroupNormGrad(group_norm_grad(
                table, node, attrs,
            )));
        }
        BackwardRule::FusedSwiGluBackward => {
            out.push(KernelCall::FusedSwiGluGrad(fused_swiglu_grad(table, node)));
        }
        BackwardRule::Conv2dBackward => {
            let attrs = conv2d_attrs_of(node)?;
            out.push(KernelCall::Conv2dGrad(conv2d_grad(
                chain, table, node, attrs,
            )?));
        }
        BackwardRule::ConvTranspose2dBackward => {
            let attrs = conv_transpose_2d_attrs_of(node)?;
            out.push(KernelCall::ConvTranspose2dGrad(conv_transpose_2d_grad(
                chain, table, node, attrs,
            )?));
        }
        BackwardRule::AttentionBackward => {
            let attrs = attention_attrs_of(node)?;
            out.push(KernelCall::AttentionGrad(attention_grad(
                chain, table, node, attrs,
            )?));
        }
    }
    Ok(())
}

fn require_grad_inputs(chain: &TransformChain, node: &TransformNode) -> Result<(), PlanError> {
    for r in &node.inputs {
        let t = require_tensor(chain, r.tensor)?;
        if !t.requires_grad {
            return Err(PlanError::MissingGradSlot(node.op.name()));
        }
    }
    Ok(())
}

/// Generate a single-variant attrs lookup. Each backward rule needs
/// to fish its `*Attrs` out of the node's `SemanticOp`; the pattern
/// is identical, so a macro saves the boilerplate.
macro_rules! attrs_of {
    ($name:ident, $variant:ident, $attrs:ty) => {
        fn $name(node: &TransformNode) -> Result<$attrs, PlanError> {
            match node.op {
                SemanticOp::$variant(a) => Ok(a),
                other => Err(PlanError::UnsupportedOp(other.name())),
            }
        }
    };
}

attrs_of!(matmul_attrs_of, MatMul, MatMulAttrs);
attrs_of!(concat_attrs_of, Concat, ConcatAttrs);
attrs_of!(slice_attrs_of, Slice, SliceAttrs);
attrs_of!(transpose_attrs_of, Transpose, TransposeAttrs);
attrs_of!(global_avg_pool_attrs_of, GlobalAvgPool, GlobalAvgPoolAttrs);
attrs_of!(group_norm_attrs_of, GroupNorm, GroupNormAttrs);
attrs_of!(conv2d_attrs_of, Conv2d, Conv2dAttrs);
attrs_of!(
    conv_transpose_2d_attrs_of,
    ConvTranspose2d,
    ConvTransposeAttrs
);
attrs_of!(attention_attrs_of, Attention, AttentionAttrs);

// `norm_attrs_of` and `pool2d_attrs_of` are intentionally hand-written
// — they accept several `SemanticOp` variants whose attrs all share
// one type (`NormAttrs` / `Pool2dAttrs`). A two-arity macro overload
// would be more cryptic than the explicit match.
fn norm_attrs_of(node: &TransformNode, expected: &'static str) -> Result<NormAttrs, PlanError> {
    match (expected, node.op) {
        ("rms_norm", SemanticOp::RmsNorm(a)) => Ok(a),
        ("layer_norm", SemanticOp::LayerNorm(a)) => Ok(a),
        ("instance_norm", SemanticOp::InstanceNorm(a)) => Ok(a),
        ("add_rms_norm", SemanticOp::AddRmsNorm(a)) => Ok(a),
        (_, other) => Err(PlanError::UnsupportedOp(other.name())),
    }
}

fn pool2d_attrs_of(node: &TransformNode, expected: &'static str) -> Result<Pool2dAttrs, PlanError> {
    match (expected, node.op) {
        ("max_pool_2d", SemanticOp::MaxPool2d(a)) => Ok(a),
        ("avg_pool_2d", SemanticOp::AvgPool2d(a)) => Ok(a),
        (_, other) => Err(PlanError::UnsupportedOp(other.name())),
    }
}

// ── Per-rule grad builders ───────────────────────────────────────────────────

fn add_grad(t: &AddressTable, n: &TransformNode) -> AddGradCall {
    AddGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn sub_grad(t: &AddressTable, n: &TransformNode) -> SubGradCall {
    SubGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn mul_grad(t: &AddressTable, n: &TransformNode) -> MulGradCall {
    MulGradCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn neg_grad(t: &AddressTable, n: &TransformNode) -> NegGradCall {
    NegGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
    }
}

fn div_grad(t: &AddressTable, n: &TransformNode) -> DivGradCall {
    DivGradCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn unary_grad_from_input(t: &AddressTable, n: &TransformNode) -> UnaryGradCall {
    UnaryGradCall {
        source: t.in_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
    }
}

fn unary_grad_from_output(t: &AddressTable, n: &TransformNode) -> UnaryGradCall {
    UnaryGradCall {
        source: t.out_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
    }
}

fn min_max_grad(t: &AddressTable, n: &TransformNode) -> MinMaxGradCall {
    MinMaxGradCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn reduce_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
) -> Result<ReduceGradCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let last = *in_dims.last().ok_or(PlanError::ShapeMismatch {
        op: "reduce_grad",
        detail: "input must have at least one dimension",
    })?;
    Ok(ReduceGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        size: last,
    })
}

fn matmul_grad_a(t: &AddressTable, n: &TransformNode, a: MatMulAttrs) -> MatMulGradACall {
    MatMulGradACall {
        dc: t.out_grad(n, 0),
        b: t.in_span(n, 1),
        da: t.in_grad(n, 0),
        m: a.m as usize,
        k: a.k as usize,
        n: a.n as usize,
    }
}

fn matmul_grad_b(t: &AddressTable, n: &TransformNode, a: MatMulAttrs) -> MatMulGradBCall {
    MatMulGradBCall {
        a: t.in_span(n, 0),
        dc: t.out_grad(n, 0),
        db: t.in_grad(n, 1),
        m: a.m as usize,
        k: a.k as usize,
        n: a.n as usize,
    }
}

fn concat_grad(t: &AddressTable, n: &TransformNode, a: ConcatAttrs) -> ConcatGradCall {
    ConcatGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
        size_a: a.size_a,
        size_b: a.size_b,
    }
}

fn slice_grad(t: &AddressTable, n: &TransformNode, a: SliceAttrs) -> SliceGradCall {
    SliceGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        axis_size: a.axis_size,
        start: a.start,
        end: a.end,
    }
}

fn transpose_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: TransposeAttrs,
) -> Result<TransposeGradCall, PlanError> {
    let nd = a.ndim as usize;
    if nd > 4 {
        return Err(PlanError::ShapeMismatch {
            op: "transpose_grad",
            detail: "reference kernel supports rank ≤ 4",
        });
    }
    let dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if dims.len() != nd {
        return Err(PlanError::ShapeMismatch {
            op: "transpose_grad",
            detail: "input rank must match TransposeAttrs.ndim",
        });
    }
    let mut input_dims = [0_u32; 4];
    let mut inv_perm = [0_u8; 4];
    for i in 0..nd {
        input_dims[i] = dims[i] as u32;
        let p = a.perm[i] as usize;
        if p >= nd {
            return Err(PlanError::ShapeMismatch {
                op: "transpose_grad",
                detail: "permutation entry out of range",
            });
        }
        inv_perm[p] = i as u8;
    }
    Ok(TransposeGradCall {
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        input_dims,
        inv_perm,
        ndim: a.ndim,
    })
}

fn pow_grad(t: &AddressTable, n: &TransformNode) -> PowGradCall {
    PowGradCall {
        a: t.in_span(n, 0),
        b: t.in_span(n, 1),
        out: t.out_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        db: t.in_grad(n, 1),
    }
}

fn last_axis_size(
    chain: &TransformChain,
    n: &TransformNode,
    op: &'static str,
) -> Result<usize, PlanError> {
    let dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let last = *dims.last().ok_or(PlanError::ShapeMismatch {
        op,
        detail: "input must have at least one dimension",
    })?;
    Ok(last)
}

fn softmax_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
) -> Result<SoftmaxGradCall, PlanError> {
    Ok(SoftmaxGradCall {
        output: t.out_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        size: last_axis_size(chain, n, "softmax_grad")?,
    })
}

fn reduce_arg_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
) -> Result<ReduceArgGradCall, PlanError> {
    Ok(ReduceArgGradCall {
        a: t.in_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        size: last_axis_size(chain, n, "reduce_arg_grad")?,
    })
}

fn reduce_prod_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
) -> Result<ReduceProdGradCall, PlanError> {
    Ok(ReduceProdGradCall {
        a: t.in_span(n, 0),
        out: t.out_span(n, 0),
        dc: t.out_grad(n, 0),
        da: t.in_grad(n, 0),
        size: last_axis_size(chain, n, "reduce_prod_grad")?,
    })
}

fn rms_norm_grad(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> RmsNormGradCall {
    RmsNormGradCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
        dw: t.in_grad(n, 1),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn layer_norm_grad(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> LayerNormGradCall {
    LayerNormGradCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
        dw: t.in_grad(n, 1),
        db: t.in_grad(n, 2),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn instance_norm_grad(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> InstanceNormGradCall {
    InstanceNormGradCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
        dw: t.in_grad(n, 1),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn add_rms_norm_grad(t: &AddressTable, n: &TransformNode, a: NormAttrs) -> AddRmsNormGradCall {
    AddRmsNormGradCall {
        residual: t.in_span(n, 0),
        input: t.in_span(n, 1),
        weight: t.in_span(n, 2),
        dy: t.out_grad(n, 0),
        d_residual: t.in_grad(n, 0),
        d_input: t.in_grad(n, 1),
        dw: t.in_grad(n, 2),
        size: a.size,
        epsilon: a.epsilon,
    }
}

fn pool2d_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: Pool2dAttrs,
) -> Result<Pool2dGradCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    let out_dims = require_tensor(chain, n.outputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "pool2d_grad",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    Ok(Pool2dGradCall {
        input: t.in_span(n, 0),
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
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

fn global_avg_pool_grad(
    chain: &TransformChain,
    t: &AddressTable,
    n: &TransformNode,
    a: GlobalAvgPoolAttrs,
) -> Result<GlobalAvgPoolGradCall, PlanError> {
    let in_dims = require_tensor(chain, n.inputs[0].tensor)?.dims.as_slice();
    if in_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "global_avg_pool_grad",
            detail: "input tensor must be 4-D (NCHW)",
        });
    }
    Ok(GlobalAvgPoolGradCall {
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
        n: in_dims[0] as u32,
        c: a.channels,
        h: a.spatial_h,
        w: a.spatial_w,
    })
}

fn group_norm_grad(t: &AddressTable, n: &TransformNode, a: GroupNormAttrs) -> GroupNormGradCall {
    GroupNormGradCall {
        input: t.in_span(n, 0),
        weight: t.in_span(n, 1),
        dy: t.out_grad(n, 0),
        dx: t.in_grad(n, 0),
        dw: t.in_grad(n, 1),
        db: t.in_grad(n, 2),
        num_groups: a.num_groups,
        epsilon: a.epsilon,
    }
}

fn fused_swiglu_grad(t: &AddressTable, n: &TransformNode) -> FusedSwiGluGradCall {
    FusedSwiGluGradCall {
        gate: t.in_span(n, 0),
        up: t.in_span(n, 1),
        dc: t.out_grad(n, 0),
        d_gate: t.in_grad(n, 0),
        d_up: t.in_grad(n, 1),
    }
}

fn conv2d_grad(
    chain: &TransformChain,
    t: &AddressTable,
    node: &TransformNode,
    a: Conv2dAttrs,
) -> Result<Conv2dGradCall, PlanError> {
    let in_dims = require_tensor(chain, node.inputs[0].tensor)?
        .dims
        .as_slice();
    let out_dims = require_tensor(chain, node.outputs[0].tensor)?
        .dims
        .as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "conv2d_grad",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    let (input, weight) = (t.in_span(node, 0), t.in_span(node, 1));
    let dx = t.in_grad(node, 0);
    let dw = t.in_grad(node, 1);
    let db = if node.inputs.len() >= 3 {
        t.in_grad(node, 2)
    } else {
        SlotSpan::empty(0)
    };
    Ok(Conv2dGradCall {
        input,
        weight,
        dy: t.out_grad(node, 0),
        dx,
        dw,
        db,
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

fn attention_grad(
    chain: &TransformChain,
    t: &AddressTable,
    node: &TransformNode,
    a: AttentionAttrs,
) -> Result<AttentionGradCall, PlanError> {
    if a.num_kv_heads == 0 || !a.num_q_heads.is_multiple_of(a.num_kv_heads) {
        return Err(PlanError::ShapeMismatch {
            op: "attention_grad",
            detail: "num_q_heads must be a positive multiple of num_kv_heads",
        });
    }
    let q_dims = require_tensor(chain, node.inputs[0].tensor)?
        .dims
        .as_slice();
    let k_dims = require_tensor(chain, node.inputs[1].tensor)?
        .dims
        .as_slice();
    if q_dims.len() < 4 || k_dims.len() < 4 {
        return Err(PlanError::ShapeMismatch {
            op: "attention_grad",
            detail: "Q/K must be 4-D ([batch, n_heads, seq, head_dim])",
        });
    }
    let batch: usize = q_dims[..1].iter().product::<usize>().max(1);
    Ok(AttentionGradCall {
        q: t.in_span(node, 0),
        k: t.in_span(node, 1),
        v: t.in_span(node, 2),
        d_out: t.out_grad(node, 0),
        dq: t.in_grad(node, 0),
        dk: t.in_grad(node, 1),
        dv: t.in_grad(node, 2),
        batch: batch as u32,
        num_q_heads: a.num_q_heads,
        num_kv_heads: a.num_kv_heads,
        head_dim: a.head_dim,
        seq_q: q_dims[2] as u32,
        seq_kv: k_dims[2] as u32,
        scale_bits: a.scale,
        causal: a.causal,
    })
}

fn conv_transpose_2d_grad(
    chain: &TransformChain,
    t: &AddressTable,
    node: &TransformNode,
    a: ConvTransposeAttrs,
) -> Result<ConvTranspose2dGradCall, PlanError> {
    let in_dims = require_tensor(chain, node.inputs[0].tensor)?
        .dims
        .as_slice();
    let out_dims = require_tensor(chain, node.outputs[0].tensor)?
        .dims
        .as_slice();
    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(PlanError::ShapeMismatch {
            op: "conv_transpose_2d_grad",
            detail: "input and output tensors must be 4-D (NCHW)",
        });
    }
    let db = if node.inputs.len() >= 3 {
        t.in_grad(node, 2)
    } else {
        SlotSpan::empty(0)
    };
    Ok(ConvTranspose2dGradCall {
        input: t.in_span(node, 0),
        weight: t.in_span(node, 1),
        dy: t.out_grad(node, 0),
        dx: t.in_grad(node, 0),
        dw: t.in_grad(node, 1),
        db,
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
