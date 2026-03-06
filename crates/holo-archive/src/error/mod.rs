//! Archive error types.

use std::fmt;
use std::io;

/// Error type for archive operations.
#[derive(Debug)]
pub enum ArchiveError {
    /// Invalid magic bytes (expected HOLO_MAGIC).
    InvalidMagic,
    /// Unsupported format version.
    UnsupportedVersion(u32),
    /// CRC32 checksum mismatch.
    ChecksumMismatch { expected: u32, actual: u32 },
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
                    "checksum mismatch: expected {expected:#010x}, got {actual:#010x}"
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
        let e = ArchiveError::ChecksumMismatch {
            expected: 0xDEAD_BEEF,
            actual: 0x0000_0001,
        };
        let s = format!("{e}");
        assert!(s.contains("0xdeadbeef"));
        assert!(s.contains("0x00000001"));
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
