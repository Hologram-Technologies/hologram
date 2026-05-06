use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("archive load failed: {0}")]
    Archive(#[from] hologram_archive::ArchiveError),
    #[error("backend dispatch failed")]
    Backend,
    #[error("input shape mismatch")]
    InputMismatch,
    #[error("workspace exhausted")]
    WorkspaceExhausted,
}
