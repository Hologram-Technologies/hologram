//! Compiled, executable plan.
//!
//! `CompiledPlan` is the contract between the compile-time and run-time
//! worlds. It owns:
//!
//! - `address_table` — `AddressRef` → `SlotSpan`, O(1) lookup
//! - `grad_table`    — same shape as `address_table`, but for gradients
//! - `workspace`     — single contiguous allocation size
//! - `forward`       — pre-computed forward kernel calls
//! - `backward`      — pre-computed backward kernel calls
//!
//! At execution time everything in this struct is read-only. The
//! executor never allocates, never branches on shape, and never
//! traverses a graph.
//!
//! Per-op `Call` structs and the `KernelCall` enum live in
//! [`hologram_ops::kernels`]; this module owns only the planner-side
//! plan envelope.

pub use hologram_ops::{
    AddCall, AddGradCall, AddRmsNormCall, AddRmsNormGradCall, AttentionCall, AttentionGradCall,
    BinaryCall, ClipCall, ConcatCall, ConcatGradCall, Conv2dCall, Conv2dGradCall,
    ConvTranspose2dGradCall, ConvTransposeCall, CumSumCall, DivGradCall, ExpandCall,
    FusedSwiGluGradCall, GemmCall, GlobalAvgPoolCall, GlobalAvgPoolGradCall, GroupNormCall,
    GroupNormGradCall, InstanceNormGradCall, KernelCall, LayerNormGradCall, LrnCall, MatMulCall,
    MatMulGradACall, MatMulGradBCall, MinMaxGradCall, MinMaxGradKind, MulGradCall, NegGradCall,
    NormFullCall, NormScaleCall, PadCall, Pool2dCall, Pool2dGradCall, Pool2dKind, PowGradCall,
    ReduceArgGradCall, ReduceArgGradKind, ReduceCall, ReduceGradCall, ReduceGradKind, ReduceKind,
    ReduceProdGradCall, ReshapeCall, ResizeCall, RmsNormGradCall, RotaryEmbeddingCall, SliceCall,
    SliceGradCall, SlotSpan, SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind, SubGradCall,
    TransposeCall, TransposeGradCall, UnaryCall, UnaryGradCall, UnaryGradKind, UnaryKind,
    WhereCall,
};

/// Resolved address table: `tensor_id → SlotSpan`.
///
/// Indexed by `TensorId.0 as usize`. O(1) per lookup. Sized once by the
/// planner and never mutated thereafter.
#[derive(Debug, Clone)]
pub struct AddressTable {
    /// Forward (value) slot per tensor.
    pub spans: Box<[SlotSpan]>,
    /// Gradient slot per tensor (empty span if `requires_grad = false`).
    pub grads: Box<[SlotSpan]>,
}

impl AddressTable {
    /// Resolve a tensor's value span. Caller guarantees `id` is in range.
    #[inline]
    #[must_use]
    pub fn span(&self, id: crate::address::TensorId) -> SlotSpan {
        self.spans[id.0 as usize]
    }

    /// Resolve a tensor's gradient span. Empty if not `requires_grad`.
    #[inline]
    #[must_use]
    pub fn grad(&self, id: crate::address::TensorId) -> SlotSpan {
        self.grads[id.0 as usize]
    }

    /// Convenience: span of the `i`-th input of `node`.
    #[inline]
    #[must_use]
    pub fn in_span(&self, node: &crate::chain::TransformNode, i: usize) -> SlotSpan {
        self.span(node.inputs[i].tensor)
    }

    /// Convenience: span of the `i`-th output of `node`.
    #[inline]
    #[must_use]
    pub fn out_span(&self, node: &crate::chain::TransformNode, i: usize) -> SlotSpan {
        self.span(node.outputs[i].tensor)
    }

    /// Convenience: gradient span of the `i`-th input of `node`.
    #[inline]
    #[must_use]
    pub fn in_grad(&self, node: &crate::chain::TransformNode, i: usize) -> SlotSpan {
        self.grad(node.inputs[i].tensor)
    }

    /// Convenience: gradient span of the `i`-th output of `node`.
    #[inline]
    #[must_use]
    pub fn out_grad(&self, node: &crate::chain::TransformNode, i: usize) -> SlotSpan {
        self.grad(node.outputs[i].tensor)
    }
}

/// Total workspace allocation, in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceLayout {
    /// Total elements the executor must allocate.
    pub total_elements: usize,
}

/// The full compiled plan.
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    /// Forward kernel calls, in execution order.
    pub forward: Box<[KernelCall]>,
    /// Backward kernel calls, in execution order (already reversed).
    pub backward: Box<[KernelCall]>,
    /// Resolved tensor → slot mapping.
    pub address_table: AddressTable,
    /// Single contiguous workspace size.
    pub workspace: WorkspaceLayout,
}

impl CompiledPlan {
    /// Number of forward kernel calls.
    #[inline]
    #[must_use]
    pub fn forward_len(&self) -> usize {
        self.forward.len()
    }

    /// Number of backward kernel calls.
    #[inline]
    #[must_use]
    pub fn backward_len(&self) -> usize {
        self.backward.len()
    }

    /// Total workspace size in elements.
    #[inline]
    #[must_use]
    pub fn workspace_elements(&self) -> usize {
        self.workspace.total_elements
    }
}
