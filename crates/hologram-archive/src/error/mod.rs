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
///
/// `Clone` is implemented because `ArchiveError` is conventionally stored
/// in diagnostic chains, retried error events, and structured logs. The
/// `Io` variant flattens the underlying `std::io::Error` into its
/// `ErrorKind` discriminant plus a `String` message rather than holding
/// the non-`Clone` `io::Error` itself; the trade-off is loss of the
/// boxed source chain in `Error::source()` for the `Io` variant.
#[derive(Debug, Clone)]
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
    /// I/O error (mmap, file read). Captured as kind + display message
    /// rather than the underlying `io::Error` so the type stays `Clone`.
    Io {
        kind: io::ErrorKind,
        message: String,
    },
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
            Self::Io { kind, message } => write!(f, "I/O error ({kind:?}): {message}"),
            Self::GraphError(msg) => {
                write!(f, "graph error: {msg}")
            }
        }
    }
}

impl std::error::Error for ArchiveError {}

impl From<io::Error> for ArchiveError {
    fn from(e: io::Error) -> Self {
        Self::Io {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}

/// Result type for archive operations.
pub type ArchiveResult<T> = Result<T, ArchiveError>;

#[cfg(test)]
mod tests {
    use super::*;

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
        match e {
            ArchiveError::Io { kind, message } => {
                assert_eq!(kind, io::ErrorKind::NotFound);
                assert!(message.contains("gone"));
            }
            other => panic!("expected ArchiveError::Io, got {other:?}"),
        }
    }

    #[test]
    fn archive_error_is_clone() {
        // Compile-time check: Clone is available on every variant.
        fn _assert_clone<T: Clone>() {}
        _assert_clone::<ArchiveError>();
        let e = ArchiveError::InvalidMagic;
        let _cloned = e.clone();
    }
}
