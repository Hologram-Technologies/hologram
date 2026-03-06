//! Hologram — O(1) compute acceleration via pre-computed lookup tables.
//!
//! This crate re-exports the public API from all workspace crates so consumers
//! only need to depend on `hologram`.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | yes | Standard library support |
//! | `simd` | yes | SIMD-accelerated LUT operations |
//! | `parallel` | yes | Rayon parallel level execution |
//! | `compiler` | yes | Graph → `.holo` archive compilation pipeline |
//! | `async` | no | Async execution wrappers (pulls in tokio) |
//! | `ffi` | no | C ABI and WASM bindings |
//! | `cli` | no | Command-line interface (pulls in tokio + clap) |
//! | `full` | no | All of the above |
//! | `wasm` | no | WASM bindings (implies `ffi`) |

pub use hologram_archive;
pub use hologram_core;
pub use hologram_exec;
pub use hologram_graph;

#[cfg(feature = "compiler")]
pub use hologram_compiler;

#[cfg(feature = "async")]
pub use hologram_async;

#[cfg(feature = "ffi")]
pub use hologram_ffi;

#[cfg(feature = "cli")]
pub use hologram_cli;
