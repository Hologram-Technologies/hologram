//! Tape-based execution engine with parallel level scheduling.
//!
//! Every operation is an O(1) key-value lookup into precomputed tables.
//! Graphs are executed level-by-level, where nodes within a level have
//! all dependencies satisfied and can run concurrently (via rayon when enabled).
//!
//! # v0.2.0 reframing
//!
//! The v0.1.4 `CustomOpRegistry`, `CustomHandler`, `CustomOpId`, and the
//! `register_op!` macro were removed in the conformance-first refactor.
//! Custom ops violated the SCS carrying criterion (an op outside the
//! declared shape) and introduced runtime function-pointer indirection.
//! The replacement extensibility model is Prism-modules-as-extensions,
//! which dispatch at compile time. **Perf: WIN** — eliminates the function
//! pointer call from the kernel hot path.

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
pub mod prism_module;
pub mod shape_resolve;
pub mod tape;
pub mod tape_builder;

// Re-exports for convenience.
pub use buffer::BufferArena;
pub use error::{ExecError, ExecResult};
pub use eval::{build_schedule, GraphInputs, GraphOutputs};
pub use kv::{KvStore, WeightCache};
pub use kv_cache::{KvBits, KvCacheConfig, KvCacheState};
pub use mmap::{
    build_tape_from_plan, execute_tape, execute_tape_with_kv, execute_tape_with_kv_and_shapes,
    execute_tape_with_kv_cached, execute_tape_with_kv_shapes_cached, execute_tape_with_shapes,
    execute_tape_with_weight_cache, InferenceSession,
};
pub use prism_module::{FusedComponentModule, LoadedModel};
