//! Error types for the execution engine.

use holo_graph::graph::node::NodeId;
use std::fmt;

/// Error type for execution operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecError {
    /// Node ID not found in the graph.
    NodeNotFound(NodeId),
    /// Missing input for a node at the given slot index.
    MissingInput { node: NodeId, slot: usize },
    /// Missing graph-level input by index.
    MissingGraphInput(u32),
    /// Buffer not available for a dependency node.
    BufferNotReady(NodeId),
    /// Constant not found in store.
    ConstantNotFound(u32),
    /// Graph contains a cycle.
    CycleDetected,
    /// Operation not supported in this phase.
    UnsupportedOp(String),
    /// Input/output length mismatch for binary ops.
    LengthMismatch { expected: usize, actual: usize },
    /// Error from the archive loader.
    ArchiveError(String),
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => {
                write!(f, "node not found: {id:?}")
            }
            Self::MissingInput { node, slot } => {
                write!(f, "missing input slot {slot} for node {node:?}")
            }
            Self::MissingGraphInput(idx) => {
                write!(f, "missing graph input at index {idx}")
            }
            Self::BufferNotReady(id) => {
                write!(f, "buffer not ready for node {id:?}")
            }
            Self::ConstantNotFound(id) => {
                write!(f, "constant not found: {id}")
            }
            Self::CycleDetected => write!(f, "cycle detected in graph"),
            Self::UnsupportedOp(op) => {
                write!(f, "unsupported operation: {op}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ArchiveError(msg) => {
                write!(f, "archive error: {msg}")
            }
        }
    }
}

impl std::error::Error for ExecError {}

/// Result type for execution operations.
pub type ExecResult<T> = Result<T, ExecError>;

impl From<holo_archive::ArchiveError> for ExecError {
    fn from(e: holo_archive::ArchiveError) -> Self {
        Self::ArchiveError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_node_not_found() {
        let e = ExecError::NodeNotFound(NodeId::new(5, 1));
        assert!(e.to_string().contains("node not found"));
    }

    #[test]
    fn display_missing_input() {
        let e = ExecError::MissingInput {
            node: NodeId::new(0, 0),
            slot: 2,
        };
        let s = e.to_string();
        assert!(s.contains("missing input"));
        assert!(s.contains("slot 2"));
    }

    #[test]
    fn display_missing_graph_input() {
        let e = ExecError::MissingGraphInput(3);
        assert!(e.to_string().contains("index 3"));
    }

    #[test]
    fn display_buffer_not_ready() {
        let e = ExecError::BufferNotReady(NodeId::new(1, 0));
        assert!(e.to_string().contains("buffer not ready"));
    }

    #[test]
    fn display_length_mismatch() {
        let e = ExecError::LengthMismatch {
            expected: 10,
            actual: 5,
        };
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn display_unsupported_op() {
        let e = ExecError::UnsupportedOp("CallSubgraph".into());
        assert!(e.to_string().contains("CallSubgraph"));
    }

    #[test]
    fn error_equality() {
        assert_eq!(ExecError::CycleDetected, ExecError::CycleDetected);
        assert_ne!(
            ExecError::ConstantNotFound(1),
            ExecError::ConstantNotFound(2)
        );
    }
}
