//! Compiler error types.

use std::fmt;

/// Error type for compilation operations.
///
/// The `InsufficientKernel` and `ContradictoryConstraint` variants map to the two infeasibility
/// classes from Prism PX_5:
/// - **Insufficient** (CB_5 sufficiency failure): a required kernel is absent.
/// - **Contradictory** (SR_5 ContradictionBoundary): two constraints conflict at the same address.
#[derive(Debug)]
pub enum CompileError {
    /// Graph validation failed.
    Validation(String),
    /// Fusion pass failed.
    Fusion(String),
    /// Archive emission failed.
    Emission(String),
    /// No registered kernel for this op/dtype combination.
    ///
    /// Corresponds to Prism PX_5 / CB_5 Insufficient infeasibility: the fiber-sufficiency check
    /// fails because no dispatcher covers the required (op, dtype) pair.
    InsufficientKernel {
        /// The operation name.
        op: String,
        /// The data type that lacks a kernel.
        dtype: String,
    },
    /// Conflicting type or shape constraints that cannot be simultaneously satisfied.
    ///
    /// Corresponds to Prism PX_5 / SR_5 Contradictory infeasibility: a ContradictionBoundary
    /// fires because two bindings conflict at the same address.
    ContradictoryConstraint {
        /// Human-readable description of the conflict.
        detail: String,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Fusion(msg) => write!(f, "fusion error: {msg}"),
            Self::Emission(msg) => write!(f, "emission error: {msg}"),
            Self::InsufficientKernel { op, dtype } => {
                write!(
                    f,
                    "insufficient kernel: no dispatcher for op '{op}' at dtype '{dtype}'"
                )
            }
            Self::ContradictoryConstraint { detail } => {
                write!(f, "contradictory constraint: {detail}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

impl From<hologram_graph::error::GraphError> for CompileError {
    fn from(e: hologram_graph::error::GraphError) -> Self {
        Self::Validation(e.to_string())
    }
}

impl From<hologram_archive::error::ArchiveError> for CompileError {
    fn from(e: hologram_archive::error::ArchiveError) -> Self {
        Self::Emission(e.to_string())
    }
}

/// Result type for compilation operations.
pub type CompileResult<T> = Result<T, CompileError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_validation() {
        let e = CompileError::Validation("bad graph".into());
        assert_eq!(format!("{e}"), "validation error: bad graph");
    }

    #[test]
    fn display_fusion() {
        let e = CompileError::Fusion("fuse failed".into());
        assert_eq!(format!("{e}"), "fusion error: fuse failed");
    }

    #[test]
    fn display_emission() {
        let e = CompileError::Emission("write failed".into());
        assert_eq!(format!("{e}"), "emission error: write failed");
    }

    #[test]
    fn error_trait() {
        let e = CompileError::Validation("test".into());
        assert!(e.source().is_none());
    }

    #[test]
    fn from_graph_error() {
        let ge = hologram_graph::error::GraphError::CycleDetected;
        let ce: CompileError = ge.into();
        assert!(matches!(ce, CompileError::Validation(_)));
        assert!(format!("{ce}").contains("cycle"));
    }

    #[test]
    fn from_archive_error() {
        let ae = hologram_archive::error::ArchiveError::InvalidMagic;
        let ce: CompileError = ae.into();
        assert!(matches!(ce, CompileError::Emission(_)));
        assert!(format!("{ce}").contains("magic"));
    }

    #[test]
    fn display_insufficient_kernel() {
        let e = CompileError::InsufficientKernel {
            op: "MatMul".into(),
            dtype: "f16".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("MatMul"));
        assert!(s.contains("f16"));
    }

    #[test]
    fn display_contradictory_constraint() {
        let e = CompileError::ContradictoryConstraint {
            detail: "shape [2,3] conflicts with [3,2] at node 5".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("contradictory constraint"));
        assert!(s.contains("node 5"));
    }
}
