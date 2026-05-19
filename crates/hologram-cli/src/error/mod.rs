//! CLI error types.

use core::fmt;

/// Top-level CLI error.
#[derive(Debug)]
pub enum CliError {
    /// I/O error.
    Io(std::io::Error),
    /// Compilation error.
    Compile(hologram_compiler::CompileError),
    /// Execution error.
    Exec(hologram_exec::ExecError),
    /// Archive error.
    Archive(hologram_archive::ArchiveError),
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

impl CliError {
    /// Return the process exit code for this error.
    ///
    /// - 1: general / I/O / archive / invalid input
    /// - 2: compilation error
    /// - 3: execution error
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Compile(_) => 2,
            Self::Exec(_) => 3,
            _ => 1,
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<hologram_compiler::CompileError> for CliError {
    fn from(e: hologram_compiler::CompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<hologram_exec::ExecError> for CliError {
    fn from(e: hologram_exec::ExecError) -> Self {
        Self::Exec(e)
    }
}

impl From<hologram_archive::ArchiveError> for CliError {
    fn from(e: hologram_archive::ArchiveError) -> Self {
        Self::Archive(e)
    }
}
