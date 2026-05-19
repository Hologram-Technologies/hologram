//! Hologram `.holo` archive format (spec Part X).
//!
//! Wraps a compiler output for distribution: kernel-call sequence,
//! schedule, BLAKE3-deduped weights, shape registry, dtype registry,
//! per-node certificates, trace, metadata, plus a footer-fingerprint.

pub mod certificate_codec;
pub mod constant_codec;
pub mod decoder;
pub mod error;
pub mod format;
pub mod kernel_codec;
pub mod loader;
pub mod schedule_codec;
pub mod weight;
pub mod writer;

pub use error::ArchiveError;
pub use format::{HoloHeader, SectionKind, FORMAT_VERSION, MAGIC};
pub use loader::{HoloLoader, LoadedPlan};
pub use weight::{WeightFingerprint, WeightStore};
pub use writer::{decode_exec_plan, decode_ports, decode_weights, HoloWriter, PortDescriptor};
