//! KV-lookup execution engine with parallel level scheduling.
//!
//! Every operation is an O(1) key-value lookup into precomputed tables.
//! Graphs are executed level-by-level, where nodes within a level have
//! all dependencies satisfied and can run concurrently (via rayon when enabled).

pub mod buffer;
pub mod error;
pub mod eval;
pub mod kv;
pub mod mmap;
pub mod parallel;

// Re-exports for convenience.
pub use buffer::BufferArena;
pub use error::{ExecError, ExecResult};
pub use eval::{build_schedule, GraphInputs, GraphOutputs, KvExecutor};
pub use kv::KvStore;
pub use mmap::{execute_bytes, execute_plan};

#[cfg(feature = "std")]
pub use mmap::execute_file;
