//! Hologram `.holo` archive format (spec Part X).
//!
//! Wraps a compiler output for distribution: kernel-call sequence,
//! schedule, BLAKE3-deduped weights, shape registry, dtype registry,
//! per-node certificates, trace, metadata, plus a footer-fingerprint.

pub mod format;
pub mod writer;
pub mod loader;
pub mod weight;
pub mod kernel_codec;
pub mod decoder;
pub mod certificate_codec;
pub mod schedule_codec;
pub mod constant_codec;
pub mod error;

pub use format::{HoloHeader, SectionKind, MAGIC, FORMAT_VERSION};
pub use writer::{HoloWriter, PortDescriptor, decode_ports, decode_exec_plan, decode_weights};
pub use loader::{HoloLoader, LoadedPlan};
pub use weight::{WeightStore, WeightFingerprint};
pub use error::ArchiveError;
