//! KV-lookup execution engine with parallel level scheduling.
//!
//! Every operation is an O(1) key-value lookup into precomputed tables.
//! Graphs are executed level-by-level, where nodes within a level have
//! all dependencies satisfied and can run concurrently (via rayon when enabled).

pub mod buffer;
pub mod error;
pub mod eval;
pub mod float_dispatch;
pub mod kv;
pub mod lut_gemm;
pub mod mmap;
pub mod parallel;
#[cfg(feature = "profile")]
pub mod profile;

// Re-exports for convenience.
pub use buffer::BufferArena;
pub use error::{ExecError, ExecResult};
pub use eval::{build_schedule, GraphInputs, GraphOutputs, KvExecutor};
pub use hologram_graph::graph::CustomOpId;
pub use kv::{CustomHandler, CustomOpRegistry, KvStore};
pub use mmap::{execute_bytes, execute_bytes_with_ops, execute_bytes_with_progress, execute_plan};

#[cfg(feature = "std")]
pub use mmap::execute_file;

/// Register a custom op handler in a `CustomOpRegistry`.
///
/// # Example
/// ```rust,ignore
/// register_op!(registry, id = 42, arity = 1, handler = |inputs, _| Ok(inputs[0].to_vec()));
/// ```
#[macro_export]
macro_rules! register_op {
    ($registry:expr, id = $id:expr, arity = $arity:expr, handler = $h:expr) => {
        $registry.register($crate::CustomOpId($id), $arity, ::std::sync::Arc::new($h))
    };
}
