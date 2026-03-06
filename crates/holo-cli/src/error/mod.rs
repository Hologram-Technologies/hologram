//! CLI error types.

use core::fmt;

/// Top-level CLI error.
#[derive(Debug)]
pub enum CliError {
    /// I/O error.
    Io(std::io::Error),
    /// Compilation error.
    Compile(holo_compiler::CompileError),
    /// Execution error.
    Exec(holo_exec::ExecError),
    /// Archive error.
    Archive(holo_archive::ArchiveError),
    /// Invalid CLI input format.
    InvalidInput(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Compile(e) => write!(f, "compile error: {e}"),
            Self::Exec(e) => write!(f, "exec error: {e}"),
            Self::Archive(e) => write!(f, "archive error: {e}"),
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
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

impl From<holo_exec::ExecError> for CliError {
    fn from(e: holo_exec::ExecError) -> Self {
        Self::Exec(e)
    }
}

impl From<holo_archive::ArchiveError> for CliError {
    fn from(e: holo_archive::ArchiveError) -> Self {
        Self::Archive(e)
    }
}
