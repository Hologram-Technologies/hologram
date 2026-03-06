//! Hologram FFI layer: C ABI + optional WASM bindings.
//!
//! Provides `extern "C"` functions for C/Python/Ruby consumers via opaque
//! handles and thread-local error propagation. WASM bindings are behind
//! the `wasm` feature flag.

// FFI functions take raw pointers by design — they are called from C.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod compiler;
pub mod encoding;
pub mod error;
pub mod exec;
pub mod graph;
pub(crate) mod handle;

#[cfg(feature = "wasm")]
pub mod wasm;
