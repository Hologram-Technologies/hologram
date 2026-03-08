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

// ── Flat convenience re-exports ───────────────────────────────────────────────
// Consumers can use `hologram::Graph` instead of `hologram::hologram_graph::Graph`.

// Core primitives
pub use hologram_core::op::{bits_to_f32, f32_to_bits, FloatDType, FloatOp, LutOp, Op, PrimOp};
pub use hologram_core::view::ElementWiseView;

// Graph IR
pub use hologram_graph::{
    ConstantData, ConstantId, ConstantStore, CustomOpId, ExecutionSchedule, FusionStats, Graph,
    GraphBuilder, GraphOp, NodeId, SubgraphDef, SubgraphId,
};

// Execution
#[cfg(feature = "std")]
pub use hologram_exec::execute_file;
pub use hologram_exec::{
    execute_bytes, execute_bytes_with_ops, execute_plan, BufferArena, CustomHandler,
    CustomOpRegistry, GraphInputs, GraphOutputs, KvExecutor, KvStore,
};

// Archive
#[cfg(feature = "std")]
pub use hologram_archive::HoloLoader;
pub use hologram_archive::{
    load_from_bytes, ArchiveError, ArchiveResult, HoloHeader, HoloWriter, LayerDescriptor,
    LayerEntrypoint, LayerHeader, LayerId, LoadedPlan,
};

// Compiler (gated)
#[cfg(feature = "compiler")]
pub use hologram_compiler::{compile, CompilationOutput, CompilerBuilder};

// Async (gated)
#[cfg(feature = "async")]
pub use hologram_async::{execute_stream, AsyncCompiler, AsyncExecutor};
