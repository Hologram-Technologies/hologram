//! Graph error types.

extern crate alloc;

use crate::graph::node::NodeId;

/// Error type for graph operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Node ID is stale or invalid.
    InvalidNode(NodeId),
    /// Cycle detected in the graph.
    CycleDetected,
    /// Subgraph ID is invalid.
    InvalidSubgraph(u32),
    /// Constant ID is invalid.
    InvalidConstant(u32),
    /// Node arity mismatch.
    ArityMismatch { expected: u8, got: u8 },
    /// Graph has no output nodes.
    NoOutputs,
    /// Graph input index out of bounds.
    InvalidGraphInput(u32),
}

impl core::fmt::Display for GraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidNode(id) => write!(f, "invalid node: {id:?}"),
            Self::CycleDetected => write!(f, "cycle detected"),
            Self::InvalidSubgraph(id) => write!(f, "invalid subgraph: {id}"),
            Self::InvalidConstant(id) => write!(f, "invalid constant: {id}"),
            Self::ArityMismatch { expected, got } => {
                write!(f, "arity mismatch: expected {expected}, got {got}")
            }
            Self::NoOutputs => write!(f, "graph has no outputs"),
            Self::InvalidGraphInput(i) => write!(f, "invalid graph input: {i}"),
        }
    }
}

/// Result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = GraphError::CycleDetected;
        assert_eq!(alloc::format!("{e}"), "cycle detected");
    }

    #[test]
    fn error_display_arity() {
        let e = GraphError::ArityMismatch {
            expected: 2,
            got: 1,
        };
        assert_eq!(alloc::format!("{e}"), "arity mismatch: expected 2, got 1");
    }

    #[test]
    fn error_eq() {
        assert_eq!(GraphError::NoOutputs, GraphError::NoOutputs);
        assert_ne!(GraphError::NoOutputs, GraphError::CycleDetected);
    }
}
