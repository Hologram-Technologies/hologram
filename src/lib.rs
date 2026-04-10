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

#[cfg(feature = "cli")]
pub mod config;

pub use hologram_archive;
pub use hologram_core;
pub use hologram_fused_component;
pub use hologram_ir;
pub use hologram_shapes;

#[cfg(feature = "compiler")]
pub use hologram_compiler;

#[cfg(feature = "async")]
pub use hologram_async;

#[cfg(feature = "ffi")]
pub use hologram_ffi;

#[cfg(feature = "cli")]
pub use hologram_cli;

// ── Flat convenience re-exports ───────────────────────────────────────────────
// Consumers can use `hologram::Graph` instead of `hologram::hologram_ir::Graph`.

// Core primitives
pub use hologram_core::op::{bits_to_f32, f32_to_bits, FloatDType, FloatOp, LutOp, Op, PrimOp};
pub use hologram_core::view::ElementWiseView;

// Graph IR
pub use hologram_ir::{
    ConstantData, ConstantId, ConstantStore, ExecutionSchedule, Graph, GraphBuilder, GraphOp,
    NodeId, StructuralFindings, SubgraphDef, SubgraphId,
};

// Execution
pub use hologram_fused_component::kv_cache;
pub use hologram_fused_component::{
    build_tape_from_plan, execute_tape, execute_tape_with_kv, execute_tape_with_kv_and_shapes,
    execute_tape_with_kv_cached, execute_tape_with_kv_shapes_cached, execute_tape_with_shapes,
    execute_tape_with_weight_cache, BufferArena, GraphInputs, GraphOutputs, InferenceSession,
    KvCacheState, KvStore, WeightCache,
};

// Archive
pub use hologram_archive::loader::pipeline::LoadedPipeline;
#[cfg(feature = "std")]
pub use hologram_archive::HoloLoader;
pub use hologram_archive::{
    load_auto, load_from_bytes, load_from_bytes_unchecked, load_from_bytes_zero_copy, ArchiveError,
    ArchiveResult, HoloHeader, HoloWriter, LayerDescriptor, LayerEntrypoint, LayerHeader, LayerId,
    LoadedPlan,
};

// Compiler (gated)
#[cfg(feature = "compiler")]
pub use hologram_compiler::{
    compile, compile_from_source, unit_from_graph, unit_from_graph_with, CompilationOutput,
    CompilationStats, Compiler, PrismModuleRegistry, SourceInput,
};

// Async (gated)
#[cfg(feature = "async")]
pub use hologram_async::{execute_stream, AsyncCompiler, AsyncExecutor};
