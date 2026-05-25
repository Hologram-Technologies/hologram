//! Hologram `.holo` archive format (spec Part X).
//!
//! Wraps a compiler output for distribution: kernel-call sequence,
//! schedule, BLAKE3-deduped weights, shape registry, dtype registry,
//! per-node certificates, trace, metadata, plus a footer-fingerprint.
//!
//! `no_std` + `alloc` by default (matching prism / uor-addr) so the format
//! reader/writer runs in wasm and on embedded targets; the `std` feature
//! adds host-only amenities and the optional `memmap2` loader path.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod address;
pub mod certificate_codec;
pub mod compose;
pub mod constant_codec;
pub mod decoder;
pub mod error;
pub mod format;
pub mod kernel_codec;
pub mod loader;
pub mod schedule_codec;
pub mod warm_codec;
pub mod weight;
pub mod writer;

pub use address::{
    address_bytes, address_ring, compose_model, derive_label, derive_label_witnessed,
    AddressOutcome, AddressWitness, ContentLabel, KappaLabel,
};
pub use error::ArchiveError;
pub use format::{HoloHeader, SectionKind, FORMAT_VERSION, MAGIC};
pub use loader::{HoloLoader, LoadedPlan};
pub use warm_codec::{derive_cone_lattice, WarmEntry};
pub use weight::{WeightFingerprint, WeightStore};
pub use writer::{decode_exec_plan, decode_ports, decode_weights, HoloWriter, PortDescriptor};
