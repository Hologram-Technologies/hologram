//! Async compilation and execution wrappers for hologram.
//!
//! Wraps the synchronous `CompilerBuilder` and tape executor behind
//! `tokio::task::spawn_blocking` so callers can drive the pipeline
//! from async contexts without blocking the executor thread.

pub mod compiler;
pub mod executor;
pub mod stream;

pub use compiler::AsyncCompiler;
pub use executor::AsyncExecutor;
pub use stream::{execute_stream, LevelResult};
