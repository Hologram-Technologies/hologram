//! Archive loaders: byte-based (WASM-compatible) and mmap-based.

pub mod bytes;
pub mod pipeline;
pub mod plan;

#[cfg(feature = "std")]
mod mmap_loader;

#[cfg(feature = "std")]
pub use mmap_loader::HoloLoader;
