//! Parse error types.

use core::fmt;

/// Error produced during lexing or parsing of UOR term language source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset in the source where the error occurred.
    pub offset: usize,
    /// What was expected at this position.
    pub expected: &'static str,
    /// What was actually found (truncated to 20 chars).
    pub found: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at byte {}: expected {}, found {:?}",
            self.offset, self.expected, self.found
        )
    }
}

impl std::error::Error for ParseError {}
