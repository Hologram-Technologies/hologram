//! Tape builder — re-exports from `hologram_exec::tape_builder`.
//!
//! The canonical implementation lives in `hologram-exec`. This module
//! re-exports it for consumers that depend on `hologram-cascade`.

pub use hologram_exec::tape_builder::build_tape;
