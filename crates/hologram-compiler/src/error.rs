use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("graph validation failed: {0}")]
    GraphValidation(&'static str),
    #[error("term emission overflowed arena for op {0}")]
    ArenaOverflow(&'static str),
    #[error("shape violation: {iri}")]
    ShapeViolation { iri: &'static str },
    #[error("completeness failure")]
    CompletenessFailure,
    #[error("archive build failed: {0}")]
    Archive(#[from] hologram_archive::ArchiveError),
    #[error("source parse failed: {0}")]
    SourceParse(&'static str),
    #[error("unsupported op kind: {0:?}")]
    UnsupportedOp(hologram_graph::OpKind),
}
