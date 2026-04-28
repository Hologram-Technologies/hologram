//! Error types for planning and execution.

use thiserror::Error;

/// Errors that can occur while compiling a `TransformChain` into a plan.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    /// A node references a tensor id that is out of range.
    #[error("unknown tensor id {0}")]
    UnknownTensor(u32),
    /// A node's input/output count does not match the op's arity.
    #[error("op {op:?} expected {expected} inputs, got {actual}")]
    ArityMismatch {
        /// Name of the offending op.
        op: &'static str,
        /// Expected arity.
        expected: usize,
        /// Provided arity.
        actual: usize,
    },
    /// Add/MatMul shape mismatch between operands.
    #[error("shape mismatch in {op}: {detail}")]
    ShapeMismatch {
        /// Op name.
        op: &'static str,
        /// Diagnostic detail.
        detail: &'static str,
    },
    /// Backward planning attempted for an op whose inputs are not all
    /// `requires_grad`.
    #[error("backward requires gradient slots for all inputs of {0}")]
    MissingGradSlot(&'static str),
    /// The planner does not yet emit kernel calls for this canonical op.
    #[error("transform planner does not support op {0}")]
    UnsupportedOp(&'static str),
}

/// Errors that can occur while executing a `CompiledPlan`.
///
/// The reference CPU executor is allocation-free and side-effect-free,
/// so its only failure mode is `WorkspaceMismatch`. Device backends
/// (Metal, WebGPU, …) surface their own diagnostics through `Backend`.
#[derive(Debug, Error)]
pub enum ExecError {
    /// `BufferSet` was sized for a different plan.
    #[error("buffer-set capacity {actual} does not match plan workspace {expected}")]
    WorkspaceMismatch {
        /// Plan's expected workspace size, in elements.
        expected: usize,
        /// Buffer-set's actual capacity, in elements.
        actual: usize,
    },
    /// A device backend reported an error (init failure, unsupported
    /// kernel variant, queue submission error, …). The string carries
    /// a backend-specific diagnostic.
    #[error("backend error: {0}")]
    Backend(String),
}

impl PartialEq for ExecError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::WorkspaceMismatch {
                    expected: e1,
                    actual: a1,
                },
                Self::WorkspaceMismatch {
                    expected: e2,
                    actual: a2,
                },
            ) => e1 == e2 && a1 == a2,
            (Self::Backend(a), Self::Backend(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ExecError {}
