//! Tape-based execution engine with parallel level scheduling.
//!
//! Every operation is an O(1) key-value lookup into precomputed tables.
//! Graphs are executed level-by-level, where nodes within a level have
//! all dependencies satisfied and can run concurrently (via rayon when enabled).

pub mod backend;
pub mod buffer;
pub mod error;
pub mod eval;
pub mod float_dispatch;
pub mod kv;
pub mod kv_cache;
pub mod lut_gemm;
pub mod mmap;
pub mod parallel;
pub mod shape_resolve;
pub mod tape;
pub mod tape_builder;

// Re-exports for convenience.
pub use buffer::BufferArena;
pub use error::{ExecError, ExecResult};
pub use eval::{build_schedule, GraphInputs, GraphOutputs};
pub use hologram_graph::graph::CustomOpId;
pub use kv::WeightCache;
pub use kv::{CustomHandler, CustomOpRegistry, KvStore};
pub use kv_cache::KvCacheState;
pub use mmap::{
    build_tape_from_plan, build_tape_from_plan_with_ops, execute_tape, execute_tape_with_kv,
    execute_tape_with_kv_and_shapes, execute_tape_with_kv_cached, execute_tape_with_shapes,
    InferenceSession,
};

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
