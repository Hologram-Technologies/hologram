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
    /// `prism::pipeline::run` rejected the inference unit during the
    /// reduction-stage sequence (preflight feasibility / budget
    /// solvency / package coherence / dispatch coverage / timing).
    /// Per wiki ADR-022 D5 this is the canonical attestation-failure
    /// path; the compute may or may not have run before this.
    #[error("prism pipeline rejected the inference unit")]
    Pipeline(prism::pipeline::PipelineFailure),
}
