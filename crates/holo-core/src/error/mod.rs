//! Error types for holo-core.

use core::fmt;

/// Core error type for holo-core operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreError {
    /// Value out of range for the target encoding.
    OutOfRange,
    /// Operation not supported for this datum.
    UnsupportedOp,
    /// Slice length mismatch.
    LengthMismatch,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange => f.write_str("value out of range"),
            Self::UnsupportedOp => f.write_str("unsupported operation"),
            Self::LengthMismatch => f.write_str("slice length mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        extern crate alloc;
        use alloc::format;
        assert_eq!(format!("{}", CoreError::OutOfRange), "value out of range");
        assert_eq!(format!("{}", CoreError::UnsupportedOp), "unsupported operation");
        assert_eq!(format!("{}", CoreError::LengthMismatch), "slice length mismatch");
    }

    #[test]
    fn error_eq() {
        assert_eq!(CoreError::OutOfRange, CoreError::OutOfRange);
        assert_ne!(CoreError::OutOfRange, CoreError::UnsupportedOp);
    }
}
