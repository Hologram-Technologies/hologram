//! Archive error types.

use std::fmt;
use std::io;

/// Format a 32-byte hash as a short hex string (first 8 hex chars).
fn hex(hash: &[u8; 32]) -> String {
    hash.iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Error type for archive operations.
#[derive(Debug)]
pub enum ArchiveError {
    /// Invalid magic bytes (expected HOLO_MAGIC).
    InvalidMagic,
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// BLAKE3 checksum mismatch.
    ChecksumMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Section not found by kind.
    SectionNotFound(u32),
    /// Offset or size exceeds archive bounds.
    OutOfBounds { offset: u64, size: u64 },
    /// rkyv validation failed.
    ValidationFailed(String),
    /// I/O error (mmap, file read).
    Io(io::Error),
    /// Graph serialization error.
    GraphError(String),
    /// v3 archive is missing the required shape metadata for a node.
    /// Per ADR-053, v3 archives must populate
    /// [`SerializedGraph::node_shapes`] for every dispatch-producing node.
    MissingNodeShape { node_id: u32 },
    /// v3 archive is missing the required shape metadata for a constant.
    /// Per ADR-053, v3 archives must populate
    /// [`SerializedGraph::constant_shapes`] for every referenced constant.
    MissingConstantShape { constant_id: u32 },
}

impl fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid magic bytes"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported format version: {v}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "checksum mismatch: expected {}, got {}",
                    hex(expected),
                    hex(actual),
                )
            }
            Self::SectionNotFound(kind) => {
                write!(f, "section not found: kind {kind}")
            }
            Self::OutOfBounds { offset, size } => {
                write!(f, "out of bounds: offset {offset}, size {size}")
            }
            Self::ValidationFailed(msg) => {
                write!(f, "validation failed: {msg}")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::GraphError(msg) => {
                write!(f, "graph error: {msg}")
            }
            Self::MissingNodeShape { node_id } => {
                write!(
                    f,
                    "v3 archive missing required node_shapes entry for node {node_id}"
                )
            }
            Self::MissingConstantShape { constant_id } => {
                write!(
                    f,
                    "v3 archive missing required constant_shapes entry for constant {constant_id}"
                )
            }
        }
    }
}

impl std::error::Error for ArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ArchiveError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for archive operations.
pub type ArchiveResult<T> = Result<T, ArchiveError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_invalid_magic() {
        let e = ArchiveError::InvalidMagic;
        assert_eq!(format!("{e}"), "invalid magic bytes");
    }

    #[test]
    fn display_unsupported_version() {
        let e = ArchiveError::UnsupportedVersion(99);
        assert_eq!(format!("{e}"), "unsupported format version: 99");
    }

    #[test]
    fn display_checksum_mismatch() {
        let mut expected = [0u8; 32];
        expected[0] = 0xDE;
        expected[1] = 0xAD;
        expected[2] = 0xBE;
        expected[3] = 0xEF;
        let actual = [0u8; 32];
        let e = ArchiveError::ChecksumMismatch { expected, actual };
        let s = format!("{e}");
        assert!(s.contains("deadbeef"));
        assert!(s.contains("00000000"));
    }

    #[test]
    fn display_out_of_bounds() {
        let e = ArchiveError::OutOfBounds {
            offset: 100,
            size: 50,
        };
        assert_eq!(format!("{e}"), "out of bounds: offset 100, size 50");
    }

    #[test]
    fn from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "gone");
        let e: ArchiveError = io_err.into();
        assert!(matches!(e, ArchiveError::Io(_)));
        assert!(e.source().is_some());
    }

    #[test]
    fn error_source_none_for_non_io() {
        let e = ArchiveError::InvalidMagic;
        assert!(e.source().is_none());
    }
}
