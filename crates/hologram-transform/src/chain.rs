//! Pure transform chain — semantic descriptors only.
//!
//! Building a chain never allocates a workspace, never dispatches a kernel,
//! and never touches a buffer. It is the compile-time input that the
//! planner consumes.

use smallvec::SmallVec;

use crate::address::{AddressRef, NodeId, TensorId};
use crate::error::PlanError;
use hologram_ops::{
    BackwardRule, ConcatAttrs, Conv2dAttrs, GroupNormAttrs, MatMulAttrs, NormAttrs, SemanticOp,
    SliceAttrs, SoftmaxAttrs, TransposeAttrs,
};

/// A tensor declared in a chain.
///
/// Shape is captured as a small dimension list. Dtype is implicitly
/// `f32` for this scaffold; ring / quantised dtypes are added later as
/// new fields rather than separate tensor types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    /// Stable id used by `AddressRef`s.
    pub id: TensorId,
    /// Concrete dimensions (row-major).
    pub dims: SmallVec<[usize; 4]>,
    /// Whether the planner should allocate a parallel gradient slot.
    pub requires_grad: bool,
}

impl Tensor {
    /// Construct a tensor with `requires_grad = false`.
    #[inline]
    #[must_use]
    pub fn new(id: TensorId, dims: &[usize]) -> Self {
        Self {
            id,
            dims: SmallVec::from_slice(dims),
            requires_grad: false,
        }
    }

    /// Mark this tensor as requiring a gradient slot.
    #[inline]
    #[must_use]
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Total element count (product of dims, or 1 for a scalar).
    #[inline]
    #[must_use]
    pub fn total_elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }
}

/// A single node in a transform chain.
///
/// Nodes are pure descriptors. They carry no buffers, no function pointers,
/// and no workspace handles. Inputs and outputs are stable `AddressRef`s
/// resolved by the planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformNode {
    /// Stable id (its index in `chain.nodes`).
    pub id: NodeId,
    /// Semantic op (canonical from `hologram-ops`).
    pub op: SemanticOp,
    /// Up to four input addresses (most ops are unary or binary).
    pub inputs: SmallVec<[AddressRef; 4]>,
    /// Up to two output addresses (most ops are single-output).
    pub outputs: SmallVec<[AddressRef; 2]>,
    /// Optional backward rule. Absent ⇒ this node is forward-only.
    pub backward: Option<BackwardRule>,
}

/// A chain of transforms.
///
/// `tensors` is the symbol table; `nodes` is an ordered list of operations.
/// Construction is purely a build step — the chain is then handed to the
/// planner to be lowered into a `CompiledPlan`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransformChain {
    /// Declared tensors, ordered so `tensors[i].id == TensorId(i as u32)`.
    pub tensors: Vec<Tensor>,
    /// Ordered list of operations.
    pub nodes: Vec<TransformNode>,
}

impl TransformChain {
    /// Start a new chain builder.
    #[inline]
    #[must_use]
    pub fn builder() -> TransformChainBuilder {
        TransformChainBuilder::default()
    }

    /// Look up a tensor by id. Returns `None` if the id is out of range.
    #[inline]
    #[must_use]
    pub fn tensor(&self, id: TensorId) -> Option<&Tensor> {
        self.tensors.get(id.0 as usize)
    }

    /// Number of declared tensors.
    #[inline]
    #[must_use]
    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }
}

/// Inputs to a `SemanticOp::Add` node.
#[derive(Debug, Clone, Copy)]
pub struct AddInputs {
    /// Left operand `A`.
    pub a: AddressRef,
    /// Right operand `B`.
    pub b: AddressRef,
    /// Output `C`.
    pub c: AddressRef,
}

/// Inputs to a unary canonical op (`Relu`, `Gelu`, `Tanh`, …).
#[derive(Debug, Clone, Copy)]
pub struct UnaryInputs {
    /// Input operand.
    pub input: AddressRef,
    /// Output.
    pub output: AddressRef,
}

/// Inputs to a 2-input weight-only norm (`RmsNorm`, `InstanceNorm`).
#[derive(Debug, Clone, Copy)]
pub struct NormScaleInputs {
    /// Input.
    pub input: AddressRef,
    /// Per-axis scale weight.
    pub weight: AddressRef,
    /// Output.
    pub output: AddressRef,
}

/// Inputs to a 3-input scale+bias norm (`LayerNorm`, `GroupNorm`).
#[derive(Debug, Clone, Copy)]
pub struct NormFullInputs {
    /// Input.
    pub input: AddressRef,
    /// Scale weight.
    pub weight: AddressRef,
    /// Bias.
    pub bias: AddressRef,
    /// Output.
    pub output: AddressRef,
}

/// Inputs to `AddRmsNorm` (residual add + RMSNorm).
#[derive(Debug, Clone, Copy)]
pub struct AddRmsNormInputs {
    /// Residual addend.
    pub residual: AddressRef,
    /// Second addend.
    pub input: AddressRef,
    /// Per-axis scale weight.
    pub weight: AddressRef,
    /// Output.
    pub output: AddressRef,
}

/// Inputs to a `Conv2d` op.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dInputs {
    /// Input data (NCHW).
    pub data: AddressRef,
    /// Kernel weight.
    pub weight: AddressRef,
    /// Bias (one element per output channel).
    pub bias: AddressRef,
    /// Output.
    pub output: AddressRef,
}

/// Inputs to a `SemanticOp::MatMul` node.
#[derive(Debug, Clone, Copy)]
pub struct MatMulInputs {
    /// Left operand `A` (`[m, k]`).
    pub a: AddressRef,
    /// Right operand `B` (`[k, n]`).
    pub b: AddressRef,
    /// Output `C` (`[m, n]`).
    pub c: AddressRef,
}

/// Builder for `TransformChain`. Allocations here are compile-time only.
#[derive(Debug, Default)]
pub struct TransformChainBuilder {
    chain: TransformChain,
}

impl TransformChainBuilder {
    /// Declare a tensor and return its id.
    #[must_use]
    pub fn add_tensor(&mut self, dims: &[usize], requires_grad: bool) -> TensorId {
        let id = TensorId(self.chain.tensors.len() as u32);
        let mut t = Tensor::new(id, dims);
        t.requires_grad = requires_grad;
        self.chain.tensors.push(t);
        id
    }

    /// Append an `Add` node with default backward (`AddBackward`).
    pub fn push_add(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Add, &[ins.a, ins.b], &[ins.c], true)
    }

    /// Append a forward-only `Add` node (no `BackwardRule`).
    pub fn push_add_forward_only(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Add, &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a `Sub` node with default backward (`SubBackward`).
    pub fn push_sub(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Sub, &[ins.a, ins.b], &[ins.c], true)
    }

    /// Append a forward-only `Sub` node (no `BackwardRule`).
    pub fn push_sub_forward_only(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Sub, &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a `Mul` node with default backward (`MulBackward`).
    pub fn push_mul(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Mul, &[ins.a, ins.b], &[ins.c], true)
    }

    /// Append a forward-only `Mul` node (no `BackwardRule`).
    pub fn push_mul_forward_only(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Mul, &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a forward-only `Div` node. Backward not yet supported.
    pub fn push_div(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Div, &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a forward-only unary canonical op node.
    ///
    /// Returns an error if `op` is not a unary canonical op (the planner
    /// rejects non-unary ops here at compile time).
    pub fn push_unary(&mut self, op: SemanticOp, ins: UnaryInputs) -> Result<NodeId, PlanError> {
        if op.arity() != 1 {
            return Err(PlanError::ArityMismatch {
                op: op.name(),
                expected: 1,
                actual: op.arity() as usize,
            });
        }
        Ok(self.push_node(op, &[ins.input], &[ins.output], false))
    }

    /// Append a forward-only `Softmax` node along the last axis.
    pub fn push_softmax(&mut self, axis_size: u32, ins: UnaryInputs) -> NodeId {
        self.push_node(
            SemanticOp::Softmax(SoftmaxAttrs { size: axis_size }),
            &[ins.input],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `LogSoftmax` node along the last axis.
    pub fn push_log_softmax(&mut self, axis_size: u32, ins: UnaryInputs) -> NodeId {
        self.push_node(
            SemanticOp::LogSoftmax(SoftmaxAttrs { size: axis_size }),
            &[ins.input],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `Reshape` node (lengths must match).
    pub fn push_reshape(&mut self, ins: UnaryInputs) -> NodeId {
        self.push_node(SemanticOp::Reshape, &[ins.input], &[ins.output], false)
    }

    /// Append a forward-only `Transpose` node (rank ≤ 4).
    pub fn push_transpose(&mut self, attrs: TransposeAttrs, ins: UnaryInputs) -> NodeId {
        self.push_node(
            SemanticOp::Transpose(attrs),
            &[ins.input],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only last-axis `Slice` node.
    pub fn push_slice(&mut self, attrs: SliceAttrs, ins: UnaryInputs) -> NodeId {
        self.push_node(SemanticOp::Slice(attrs), &[ins.input], &[ins.output], false)
    }

    /// Append a forward-only last-axis `Concat` node.
    pub fn push_concat(&mut self, attrs: ConcatAttrs, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::Concat(attrs), &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a forward-only `RmsNorm` node.
    pub fn push_rms_norm(&mut self, attrs: NormAttrs, ins: NormScaleInputs) -> NodeId {
        self.push_node(
            SemanticOp::RmsNorm(attrs),
            &[ins.input, ins.weight],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `LayerNorm` node.
    pub fn push_layer_norm(&mut self, attrs: NormAttrs, ins: NormFullInputs) -> NodeId {
        self.push_node(
            SemanticOp::LayerNorm(attrs),
            &[ins.input, ins.weight, ins.bias],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `InstanceNorm` node.
    pub fn push_instance_norm(&mut self, attrs: NormAttrs, ins: NormScaleInputs) -> NodeId {
        self.push_node(
            SemanticOp::InstanceNorm(attrs),
            &[ins.input, ins.weight],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `GroupNorm` node.
    pub fn push_group_norm(&mut self, attrs: GroupNormAttrs, ins: NormFullInputs) -> NodeId {
        self.push_node(
            SemanticOp::GroupNorm(attrs),
            &[ins.input, ins.weight, ins.bias],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `AddRmsNorm` node.
    pub fn push_add_rms_norm(&mut self, attrs: NormAttrs, ins: AddRmsNormInputs) -> NodeId {
        self.push_node(
            SemanticOp::AddRmsNorm(attrs),
            &[ins.residual, ins.input, ins.weight],
            &[ins.output],
            false,
        )
    }

    /// Append a forward-only `FusedSwiGlu` node (`out = silu(gate) * up`).
    pub fn push_fused_swiglu(&mut self, ins: AddInputs) -> NodeId {
        self.push_node(SemanticOp::FusedSwiGlu, &[ins.a, ins.b], &[ins.c], false)
    }

    /// Append a forward-only `Conv2d` node (NCHW).
    pub fn push_conv2d(&mut self, attrs: Conv2dAttrs, ins: Conv2dInputs) -> NodeId {
        self.push_node(
            SemanticOp::Conv2d(attrs),
            &[ins.data, ins.weight, ins.bias],
            &[ins.output],
            false,
        )
    }

    /// Append a `MatMul` node with default backward (`MatMulBackward`).
    ///
    /// Reads the operand and output tensors from the chain to populate
    /// `MatMulAttrs { m, k, n }` directly on the `SemanticOp`. Returns an
    /// error if any address is unresolved or shapes do not form a valid
    /// `[m,k] @ [k,n] = [m,n]` triple.
    pub fn push_matmul(&mut self, ins: MatMulInputs) -> Result<NodeId, PlanError> {
        let attrs = self.matmul_attrs_from(&ins)?;
        Ok(self.push_node(SemanticOp::MatMul(attrs), &[ins.a, ins.b], &[ins.c], true))
    }

    /// Append a forward-only `MatMul` node (no `BackwardRule`).
    pub fn push_matmul_forward_only(&mut self, ins: MatMulInputs) -> Result<NodeId, PlanError> {
        let attrs = self.matmul_attrs_from(&ins)?;
        Ok(self.push_node(SemanticOp::MatMul(attrs), &[ins.a, ins.b], &[ins.c], false))
    }

    fn matmul_attrs_from(&self, ins: &MatMulInputs) -> Result<MatMulAttrs, PlanError> {
        let a = self.tensor_dims(ins.a.tensor)?;
        let b = self.tensor_dims(ins.b.tensor)?;
        let c = self.tensor_dims(ins.c.tensor)?;
        let bad = || PlanError::ShapeMismatch {
            op: "matmul",
            detail: "expected A=[m,k], B=[k,n], C=[m,n]",
        };
        if a.len() != 2 || b.len() != 2 || c.len() != 2 {
            return Err(bad());
        }
        if a[1] != b[0] || c[0] != a[0] || c[1] != b[1] {
            return Err(bad());
        }
        Ok(MatMulAttrs {
            m: a[0] as u32,
            k: a[1] as u32,
            n: b[1] as u32,
        })
    }

    fn tensor_dims(&self, id: TensorId) -> Result<&[usize], PlanError> {
        self.chain
            .tensors
            .get(id.0 as usize)
            .map(|t| t.dims.as_slice())
            .ok_or(PlanError::UnknownTensor(id.0))
    }

    fn push_node(
        &mut self,
        op: SemanticOp,
        inputs: &[AddressRef],
        outputs: &[AddressRef],
        with_backward: bool,
    ) -> NodeId {
        let id = NodeId(self.chain.nodes.len() as u32);
        self.chain.nodes.push(TransformNode {
            id,
            op,
            inputs: SmallVec::from_slice(inputs),
            outputs: SmallVec::from_slice(outputs),
            backward: with_backward.then(|| op.backward()).flatten(),
        });
        id
    }

    /// Finalise the chain.
    #[inline]
    #[must_use]
    pub fn build(self) -> TransformChain {
        self.chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chain_has_no_tensors_or_nodes() {
        let chain = TransformChain::builder().build();
        assert_eq!(chain.n_tensors(), 0);
        assert!(chain.nodes.is_empty());
    }

    #[test]
    fn add_tensor_assigns_sequential_ids() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[4], false);
        let c = b.add_tensor(&[4], true);
        assert_eq!(a, TensorId(0));
        assert_eq!(c, TensorId(1));
        let chain = b.build();
        assert!(!chain.tensor(a).unwrap().requires_grad);
        assert!(chain.tensor(c).unwrap().requires_grad);
    }

    #[test]
    fn push_add_attaches_default_backward() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[4], true);
        let bb = b.add_tensor(&[4], true);
        let c = b.add_tensor(&[4], true);
        let n = b.push_add(AddInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        });
        assert_eq!(n, NodeId(0));
        let chain = b.build();
        assert_eq!(chain.nodes[0].op, SemanticOp::Add);
        assert_eq!(chain.nodes[0].backward, Some(BackwardRule::AddBackward));
    }

    #[test]
    fn push_matmul_attaches_default_backward_and_dims() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[2, 3], true);
        let bb = b.add_tensor(&[3, 5], true);
        let c = b.add_tensor(&[2, 5], true);
        b.push_matmul(MatMulInputs {
            a: AddressRef::of(a),
            b: AddressRef::of(bb),
            c: AddressRef::of(c),
        })
        .unwrap();
        let chain = b.build();
        assert_eq!(
            chain.nodes[0].op,
            SemanticOp::MatMul(MatMulAttrs { m: 2, k: 3, n: 5 })
        );
        assert_eq!(chain.nodes[0].backward, Some(BackwardRule::MatMulBackward));
    }

    #[test]
    fn push_matmul_rejects_shape_mismatch_at_build_time() {
        let mut b = TransformChain::builder();
        let a = b.add_tensor(&[2, 3], true);
        let bb = b.add_tensor(&[5, 4], true);
        let c = b.add_tensor(&[2, 4], true);
        let err = b
            .push_matmul(MatMulInputs {
                a: AddressRef::of(a),
                b: AddressRef::of(bb),
                c: AddressRef::of(c),
            })
            .unwrap_err();
        assert!(matches!(err, PlanError::ShapeMismatch { op: "matmul", .. }));
    }

    #[test]
    fn tensor_total_elements_handles_scalar() {
        let s = Tensor::new(TensorId(0), &[]);
        let v = Tensor::new(TensorId(0), &[8]);
        let m = Tensor::new(TensorId(0), &[3, 4]);
        assert_eq!(s.total_elements(), 1);
        assert_eq!(v.total_elements(), 8);
        assert_eq!(m.total_elements(), 12);
    }
}
