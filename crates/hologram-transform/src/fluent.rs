//! Fluent builder layer — `a.add(&b).matmul(&c)` style on top of
//! [`TransformChainBuilder`].
//!
//! `FluentChain` owns a `TransformChainBuilder` inside a `RefCell`
//! and hands out [`TensorRef`] handles that share that builder. Each
//! op method on `TensorRef` desugars to
//! [`TransformChainBuilder::push_op`], which validates arity, infers
//! the output shape, allocates a fresh tensor, and emits the node.
//!
//! Two principles:
//!
//! 1. **No new op surface.** This module adds *zero* per-op support
//!    code. The set of supported ops is whatever
//!    `TransformChainBuilder::push_op` covers (today: every
//!    `Elementwise`, `Fused`, `Normalisation`, `MatMul`, `Softmax`,
//!    `LogSoftmax`). Adding a new op anywhere — a new
//!    `SemanticOp` variant, a new `OpCategory` arm in
//!    `infer_output_shape` — automatically becomes available here
//!    via the [`TensorRef::op`] escape hatch.
//! 2. **Procedural builder is the lower layer.** Existing call sites
//!    (`builder.push_add`, `builder.push_matmul`, …) keep working
//!    unchanged. This module is purely additive ergonomics; nothing
//!    else depends on it.
//!
//! ## Sketch
//!
//! ```rust,ignore
//! use hologram_transform::FluentChain;
//! use hologram_ops::SemanticOp;
//!
//! let chain = FluentChain::new();
//! let a = chain.input(&[2, 3], false);
//! let b = chain.input(&[3, 4], false);
//! let c = chain.input(&[2, 4], false);
//!
//! let out = a.matmul(&b).unwrap().add(&c).unwrap();
//! // `out` is a `TensorRef` whose tensor id refers to the freshly
//! // allocated `[2, 4]` output of the residual add.
//!
//! let plan = chain.into_chain();   // hand off to `compile()`.
//! ```

use std::cell::RefCell;

use hologram_ops::{MatMulAttrs, SemanticOp, SoftmaxAttrs};

use crate::address::TensorId;
use crate::chain::{TransformChain, TransformChainBuilder};
use crate::error::PlanError;

/// Fluent wrapper around `TransformChainBuilder`.
///
/// The `RefCell` is the price of letting [`TensorRef`] handles share
/// one builder while still allowing each op method on a tensor to
/// mutably append a node. Borrows are runtime-checked, but only the
/// *build phase* uses this — the planner and executor see the
/// finalised `TransformChain` and never touch the cell.
pub struct FluentChain {
    builder: RefCell<TransformChainBuilder>,
}

impl FluentChain {
    /// Start a new fluent chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: RefCell::new(TransformChain::builder()),
        }
    }

    /// Start a new fluent chain pre-reserved for `n_ops` operations.
    /// See [`TransformChain::builder_with_capacity`] for the rationale.
    #[must_use]
    pub fn with_capacity(n_ops: usize) -> Self {
        Self {
            builder: RefCell::new(TransformChain::builder_with_capacity(n_ops)),
        }
    }

    /// Declare an input tensor with the given dimensions.
    pub fn input(&self, dims: &[usize], requires_grad: bool) -> TensorRef<'_> {
        let id = self.builder.borrow_mut().add_tensor(dims, requires_grad);
        TensorRef { id, chain: self }
    }

    /// Finalise the chain. Drops every outstanding [`TensorRef`]
    /// (they share the chain by reference, not by ownership).
    #[must_use]
    pub fn into_chain(self) -> TransformChain {
        self.builder.into_inner().build()
    }
}

impl Default for FluentChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to a tensor inside a [`FluentChain`].
///
/// Methods consume `&self` and return a fresh `TensorRef` — chaining
/// (`a.relu().matmul(&b)`) builds up a tree of intermediate nodes
/// without the caller pre-allocating each output. Cheap to clone /
/// pass around: it's just a `TensorId` plus a shared reference back
/// to the chain.
#[derive(Clone, Copy)]
pub struct TensorRef<'a> {
    id: TensorId,
    chain: &'a FluentChain,
}

impl<'a> TensorRef<'a> {
    /// The underlying [`TensorId`]. Use this to bridge back into the
    /// procedural [`TransformChainBuilder`] API when you need an op
    /// that the fluent layer doesn't surface yet.
    #[inline]
    #[must_use]
    pub fn id(self) -> TensorId {
        self.id
    }

    /// Generic op escape hatch. Lets fluent code call any op
    /// `TransformChainBuilder::push_op` supports — adding a new
    /// canonical op needs no per-method addition here.
    pub fn op(self, op: SemanticOp, others: &[Self]) -> Result<TensorRef<'a>, PlanError> {
        let mut inputs: smallvec::SmallVec<[TensorId; 4]> =
            smallvec::SmallVec::with_capacity(1 + others.len());
        inputs.push(self.id);
        inputs.extend(others.iter().map(|t| t.id));
        let id = self.chain.builder.borrow_mut().push_op(op, &inputs)?;
        Ok(TensorRef {
            id,
            chain: self.chain,
        })
    }

    // ── Elementwise binary ────────────────────────────────────────
    // Method names mirror the canonical op names. The
    // `should_implement_trait` lint suggests `std::ops::Add` etc.,
    // but those traits' `add` returns `Self::Output` — no error path
    // — and our methods need `Result<_, PlanError>` to surface
    // shape-mismatch / unknown-tensor failures. Inherent methods it is.
    /// `out = self + other`.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: &Self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Add, &[*other])
    }

    /// `out = self - other`.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: &Self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Sub, &[*other])
    }

    /// `out = self * other` (elementwise).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: &Self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Mul, &[*other])
    }

    /// `out = self / other`.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: &Self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Div, &[*other])
    }

    // ── Elementwise unary ─────────────────────────────────────────
    /// `out = relu(self)`.
    pub fn relu(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Relu, &[])
    }

    /// `out = silu(self)`.
    pub fn silu(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Silu, &[])
    }

    /// `out = sigmoid(self)`.
    pub fn sigmoid(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Sigmoid, &[])
    }

    /// `out = tanh(self)`.
    pub fn tanh(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Tanh, &[])
    }

    /// `out = -self`.
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Neg, &[])
    }

    /// `out = exp(self)`.
    pub fn exp(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Exp, &[])
    }

    /// `out = log(self)`.
    pub fn log(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Log, &[])
    }

    /// `out = sqrt(self)`.
    pub fn sqrt(self) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Sqrt, &[])
    }

    // ── Linear algebra / reductions ───────────────────────────────
    /// `out = self @ other` (matrix multiply, `[m,k] @ [k,n] = [m,n]`).
    /// `(m, k, n)` are derived from input shapes — the zero
    /// placeholder we pass here is rewritten inside `push_op`.
    pub fn matmul(self, other: &Self) -> Result<TensorRef<'a>, PlanError> {
        let placeholder = MatMulAttrs { m: 0, k: 0, n: 0 };
        self.op(SemanticOp::MatMul(placeholder), &[*other])
    }

    /// `out = softmax(self)` along the last axis. `axis_size` is the
    /// length of that axis (the canonical Softmax kernel takes it as
    /// an attribute rather than reading it from the shape).
    pub fn softmax(self, axis_size: u32) -> Result<TensorRef<'a>, PlanError> {
        self.op(SemanticOp::Softmax(SoftmaxAttrs { size: axis_size }), &[])
    }

    /// `out = log_softmax(self)` along the last axis.
    pub fn log_softmax(self, axis_size: u32) -> Result<TensorRef<'a>, PlanError> {
        self.op(
            SemanticOp::LogSoftmax(SoftmaxAttrs { size: axis_size }),
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluent_chain_builds_a_two_op_graph() {
        let chain = FluentChain::new();
        let a = chain.input(&[4], false);
        let b = chain.input(&[4], false);
        let c = a.add(&b).expect("add");
        let _d = c.relu().expect("relu");
        let built = chain.into_chain();
        // 4 tensors: a, b, c (add output), d (relu output).
        assert_eq!(built.tensors.len(), 4);
        // 2 nodes: Add then Relu.
        assert_eq!(built.nodes.len(), 2);
        assert_eq!(built.nodes[0].op.name(), "add");
        assert_eq!(built.nodes[1].op.name(), "relu");
    }

    #[test]
    fn fluent_matmul_then_add_chain() {
        let chain = FluentChain::new();
        // Pull the output id out of the borrow scope so we can call
        // `into_chain()` afterwards (the consuming method invalidates
        // any live `TensorRef`).
        let out_id = {
            let a = chain.input(&[2, 3], false);
            let b = chain.input(&[3, 4], false);
            let c = chain.input(&[2, 4], false);
            a.matmul(&b).expect("matmul").add(&c).expect("add").id()
        };
        let built = chain.into_chain();
        // a, b, c, matmul-out, add-out.
        assert_eq!(built.tensors.len(), 5);
        // Output of the chain is `[2, 4]`.
        assert_eq!(built.tensor(out_id).unwrap().dims.as_slice(), &[2, 4]);
        assert_eq!(built.nodes.len(), 2);
        assert_eq!(built.nodes[0].op.name(), "matmul");
        assert_eq!(built.nodes[1].op.name(), "add");
    }

    #[test]
    fn fluent_op_escape_hatch_dispatches_arbitrary_unary() {
        // The generic `op` method handles anything `push_op` covers
        // without needing a per-op fluent method. Demonstrate with a
        // direct `op(SemanticOp::Cos, &[])` call (no shorthand exists
        // on `TensorRef`). Inner block scopes the `TensorRef` so its
        // borrow drops before `into_chain()` consumes the FluentChain.
        let chain = FluentChain::new();
        {
            let x = chain.input(&[8], false);
            let _ = x.op(SemanticOp::Cos, &[]).expect("cos");
        }
        let built = chain.into_chain();
        assert_eq!(built.nodes[0].op.name(), "cos");
    }
}
