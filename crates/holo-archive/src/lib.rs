//! `.holo` archive format with execution entrypoints.
//!
//! Provides a single clean archive format for serializing compiled graphs,
//! weights, and execution metadata. Uses rkyv for zero-copy serialization
//! and supports memory-mapped loading.

pub mod checksum;
pub mod entrypoint;
pub mod error;
pub mod format;
pub mod layer;
pub mod loader;
pub mod section;
pub mod weight;
pub mod writer;
