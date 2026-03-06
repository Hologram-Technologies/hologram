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
