//! LUT-addressed transformation / mutation chains.
//!
//! This crate provides Hologram's transform/plan/execute split:
//!
//! 1. **Address layer** — `AddressRef`, `TensorId`, `RegionId`, `LayoutId`.
//!    These describe *which* object — never *how* to compute it. They are
//!    the bridge to the LUT layer (sourced from `uor-foundation`).
//!
//! 2. **Chain layer** — canonical `SemanticOp`, `BackwardRule`,
//!    `TransformNode`, `TransformChain`. Pure semantic descriptors;
//!    allocation-free; no execution.
//!
//! 3. **Plan layer** — `compile()` lowers a chain into a `CompiledPlan`
//!    with resolved `SlotSpan`s, a single `WorkspaceLayout`, and
//!    pre-computed `Box<[KernelCall]>`s for forward and backward.
//!
//! 4. **Execution layer** — [`Executor`] walks the compiled kernel calls
//!    via fixed `match` dispatch. No allocation, no virtual dispatch, no
//!    runtime algorithm selection.
//!
//! See [ADR-043](../../specs/adrs/043-lut-addressed-transform-chains.md)
//! and [Plan-043](../../specs/plans/043-lut-addressed-transform-chains.md).

#![deny(missing_docs)]

pub mod address;
pub mod backend;
pub mod buffer;
pub mod chain;
pub mod conformance;
pub mod error;
pub mod executor;
pub mod plan;
pub mod planner;

pub use address::{AddressRef, LayoutId, NodeId, RegionId, TensorId, DEFAULT_LAYOUT};
pub use backend::{kernel_call_name, CanonicalBackend, CpuBackend, TraceBackend, TraceEntry};
pub use buffer::BufferSet;
pub use chain::{
    AddInputs, AddRmsNormInputs, Conv2dInputs, MatMulInputs, NormFullInputs, NormScaleInputs,
    Tensor, TransformChain, TransformChainBuilder, TransformNode, UnaryInputs,
};
pub use conformance::{
    check_forward, check_forward_then_backward, compare, Conformance, Mismatch, Tolerance,
};
pub use error::{ExecError, PlanError};
pub use executor::Executor;
pub use hologram_ops::{BackwardRule, MatMulAttrs, Op, OpCategory, OpSignature, SemanticOp};
pub use plan::{
    AddCall, AddGradCall, AddRmsNormCall, AddressTable, BinaryCall, CompiledPlan, ConcatCall,
    Conv2dCall, GlobalAvgPoolCall, GroupNormCall, KernelCall, MatMulCall, MatMulGradACall,
    MatMulGradBCall, MulGradCall, NegGradCall, NormFullCall, NormScaleCall, Pool2dCall, Pool2dKind,
    ReduceCall, ReduceKind, ReshapeCall, SliceCall, SlotSpan, SoftmaxCall, SubGradCall,
    TransposeCall, UnaryCall, UnaryGradCall, UnaryGradKind, UnaryKind, WorkspaceLayout,
};
pub use planner::compile;
