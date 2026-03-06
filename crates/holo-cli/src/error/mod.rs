//! CLI error types.

use core::fmt;

/// Top-level CLI error.
#[derive(Debug)]
pub enum CliError {
    /// I/O error.
    Io(std::io::Error),
    /// Compilation error.
    Compile(holo_compiler::CompileError),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Compile(e) => write!(f, "compile error: {e}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<holo_compiler::CompileError> for CliError {
    fn from(e: holo_compiler::CompileError) -> Self {
        Self::Compile(e)
    }
}
