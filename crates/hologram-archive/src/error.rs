use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArchiveError {
    #[error("bad magic: {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported format version: {0}")]
    UnsupportedVersion(u16),
    #[error("section not found: {0:?}")]
    SectionMissing(crate::format::SectionKind),
    #[error("truncated archive (need {needed}, have {actual})")]
    Truncated { needed: usize, actual: usize },
    #[error("checksum mismatch")]
    ChecksumMismatch,
    #[error("io error: {0}")]
    Io(&'static str),
}
