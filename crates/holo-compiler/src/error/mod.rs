//! Compiler error types.

use std::fmt;

/// Error type for compilation operations.
#[derive(Debug)]
pub enum CompileError {
    /// Graph validation failed.
    Validation(String),
    /// Fusion pass failed.
    Fusion(String),
    /// Archive emission failed.
    Emission(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::Fusion(msg) => write!(f, "fusion error: {msg}"),
            Self::Emission(msg) => write!(f, "emission error: {msg}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<holo_graph::error::GraphError> for CompileError {
    fn from(e: holo_graph::error::GraphError) -> Self {
        Self::Validation(e.to_string())
    }
}

impl From<holo_archive::error::ArchiveError> for CompileError {
    fn from(e: holo_archive::error::ArchiveError) -> Self {
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
        let ge = holo_graph::error::GraphError::CycleDetected;
        let ce: CompileError = ge.into();
        assert!(matches!(ce, CompileError::Validation(_)));
        assert!(format!("{ce}").contains("cycle"));
    }

    #[test]
    fn from_archive_error() {
        let ae = holo_archive::error::ArchiveError::InvalidMagic;
        let ce: CompileError = ae.into();
        assert!(matches!(ce, CompileError::Emission(_)));
        assert!(format!("{ce}").contains("magic"));
    }
}
