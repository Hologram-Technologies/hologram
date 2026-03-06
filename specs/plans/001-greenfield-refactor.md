# Hologram Greenfield: Step-by-Step Refactoring Plan

## Context

V2 rewrite of `../categorical-x` (15+ crate workspace, 8-stage compiler, dual graph types, 60+ OpKind variants, two execution models). Strips away hacks and IRs, using `uor-foundation` (v3.5.0, git main) to turn execution into O(1) KV-lookups via precomputed LUT tables.

**Core requirements**:
- O(1) execution via LUT fusion tables and KV-lookups
- Zero-copy, memory-mapped data via rkyv (only serialization format)
- `.holo` archive format with execution entrypoints (hard requirement)
- SIMD for batch hot paths + rayon for parallel subgraph execution (no tokio in v1)
- CPU + WASM targets; tiny for constrained devices (ESP32, RPi)
- AI models with large weights via O(1) KV LUTs
- Scales beyond 8-bit via uor-foundation QuantumLevel (Q0–Q3)
- Subdirectory organization within each crate
- Root crate re-exports all public API — consumers use only the workspace root as dependency
- Max 3 args per function; builder pattern for more
- Prefer macros for repeated trait impls
- No backwards compatibility — single archive format
- Network distribution deferred — will port from categorical-x when core pipeline is stable

**Initial deliverable**: Super-fast calculator as `examples/calculator.rs` with benchmarks.

---

## Confirmed: uor-foundation Scaling Beyond 8-bit

| Level | Bits | Ring | States |
|-------|------|------|--------|
| Q0 | 8 | Z/256Z | 256 |
| Q1 | 16 | Z/65536Z | 65,536 |
| Q2 | 24 | Z/16777216Z | 16.7M |
| Q3 | 32 | Z/4294967296Z | 4.3B |

- `Ring` trait: general ring ops at any quantum level
- `Q1Ring` trait: 16-bit specialization
- `QuantumLevel` enum: Q0, Q1, Q2, Q3
- Implementations choose integer types per level (u8, u16, u32, etc.)

---

## Gap Analysis

1. **Encoding strategy** (pi-F-lambda) — must be formalized as traits
2. **`.holo` archive format** — single clean format, no V1/V2 compat, with execution entrypoints
3. **AI model weights** — deferred/streaming loading, quantization, LUT-GEMM
4. **Subgraph composition + parallel execution** — core to graph design
5. **Constrained device support** — `no_std`, feature-gated, WASM
6. **Root re-export** — single-crate consumer API

---

## Workspace Structure

```
hologram-greenfield/
  Cargo.toml                        # Workspace root + root crate re-exporting all public API
  AGENTS.md
  CLAUDE.md
  Justfile
  specs/
    project.md
    scratch.md
    SPRINT.md                       # Active sprint tracking
    sprints/                        # Archived sprints
  crates/
    holo-core/                      # Core: LUT, views, ring, encoding, types
    holo-graph/                     # Graph, subgraphs, fusion, scheduling
    holo-archive/                   # .holo format, rkyv, mmap, weights, entrypoints
    holo-exec/                      # KV executor, buffer, parallel levels
    holo-cli/                       # Async CLI with subcommands (exposes run())
    holo-bench/                     # Criterion benchmarks
  examples/
    calculator.rs                   # Scientific calculator example
  src/lib.rs                        # Root crate: re-exports all public API from subcrates
  src/main.rs                       # Binary: calls holo_cli::run()
```

**Root crate `src/lib.rs`**:
```rust
pub use holo_core::*;
pub use holo_graph::*;
pub use holo_archive::*;
pub use holo_exec::*;
```
Consumers add only `hologram-greenfield` as a dependency.

---

## Pre-Step: Plan Archival & Sprint Setup

Before any code, first:
1. Save this plan to `specs/plans/001-greenfield-refactor.md`
2. Create `specs/SPRINT.md` with Sprint 1 tasks (Phase 0 + Phase 1)
3. Create `specs/sprints/` directory
4. Keep `specs/SPRINT.md` updated as tasks are completed

---

## Phase 0: Foundation Setup (Sprint 1, Step 1)

**Goal**: Workspace, AGENTS.md, SPRINT.md, deps, build system, `no_std` structure.

### Tasks
1. Convert `Cargo.toml` to workspace + root crate (edition "2021")
2. Create all crate skeletons with subdirectory structure
3. Create `AGENTS.md` (see content below)
4. Create `specs/SPRINT.md` with Sprint 1 backlog
5. Create `specs/sprints/` directory
6. Create `Justfile`: `ci`, `bench`, `test`, `fmt`, `clippy`, `wasm`
7. Create `.githooks/pre-commit` hook:
   - `cargo fmt --all -- --check` (fast, read-only)
   - Detect changed crates from staged files (`git diff --cached --name-only`)
   - Run `cargo clippy -p <changed_crates> -- -D warnings` (incremental, not full workspace)
   - Handle workspace Cargo.toml changes → full clippy
   - Install via `git config core.hooksPath .githooks`
7. Workspace dependencies:
   ```toml
   uor-foundation = { git = "https://github.com/UOR-Foundation/UOR-Framework", tag = "v3.5.0" }
   rkyv = { version = "0.7", features = ["validation"] }
   bytemuck = { version = "1.14", features = ["derive"] }
   rayon = "1.8"
   criterion = "0.5"
   memmap2 = "0.9"
   crc32fast = "1.4"
   smallvec = "1.13"
   ```
8. Feature flags:
   ```toml
   default = ["std", "simd", "parallel"]
   std = []
   simd = []
   parallel = ["rayon"]
   wasm = []
   ```
9. Implement `Primitives` for `HoloPrimitives`
10. Root `src/lib.rs` re-exports all subcrate APIs

### AGENTS.md Rules

**Hard Rules**:
- All tests pass (`just ci`). Zero clippy warnings (`-D warnings`).
- 100% test coverage: unit + doc tests on all public items
- Zero-copy hot paths. No heap allocation in lookup functions
- No TODOs, no stubs, no `unimplemented!()`
- Functions <= 15 lines. Max 3 arguments per function — use builder-pattern structs for more.
- Traits for shared behavior; builder pattern for complex construction
- **Prefer macros** (`macro_rules!`) for repeated trait implementations and boilerplate patterns
- `holo-core` zero external deps except `uor-foundation` (no_std, traits-only)
- Every public operation has a Criterion benchmark
- SIMD behind `#[cfg(target_arch)]`, feature-gated
- Rayon for parallel subgraph execution, feature-gated
- Only rkyv for serialization; all persistent types derive rkyv traits
- Subdirectory organization — no loose files in `src/` beyond `lib.rs`
- All crates compile for `wasm32-unknown-unknown` with feature gates
- Data structures fit L1/L2 cache constraints
- No backwards compatibility formats — single current format only

**Sprint Workflow**:
- Active sprint in `specs/SPRINT.md` with checkboxes (`- [ ]` / `- [x]`)
- Task lifecycle: Backlog → In Progress → `just ci` → `/commit` → Completed
- Update `specs/SPRINT.md` immediately on task state change
- Archive completed sprints to `specs/sprints/<number>-<title>.md`
- "Completed (Running Log)" section in SPRINT.md (append-only, permanent)

### Verification
- `cargo build --workspace` succeeds
- `cargo test --workspace` passes
- `cargo clippy --workspace -- -D warnings` clean
- `cargo build --target wasm32-unknown-unknown -p holo-core` succeeds

---

## Phase 1: Core LUT Engine (Sprint 1, Step 2)

**Goal**: LUT tables, ElementWiseView, encodings, ring. rkyv-serializable. SIMD batch. `no_std`.

### Crate: `crates/holo-core/src/`
```
lib.rs
lut/
  mod.rs
  q0.rs             # 256-entry unary: stratum, curvature, domain, rank
  arith.rs          # 256x256 binary: add, sub, mul, pow
  activation.rs     # 21+ tables: sigmoid, tanh, relu, gelu, silu, sin, cos, etc.
view/
  mod.rs            # ElementWiseView: 256-byte, #[repr(align(64))], rkyv derives
  simd.rs           # AVX2 vpshufb (feature-gated)
  compose.rs        # .then() composition
ring/
  mod.rs
  byte_ring.rs      # Z/256Z, implements uor-foundation Ring
encoding/
  mod.rs            # Encoding trait: embed(f64)->u8, lift(u8)->f64
  angle.rs          # [0, 2pi) -> [0, 255]
  signed.rs         # [-1, 1] -> [0, 255]
  unsigned.rs       # [0, 1] -> [0, 255]
  raw.rs            # byte as-is
op/
  mod.rs            # Op types + macros for repeated impl patterns
  prim.rs           # 10 PrimOps matching uor-foundation PrimitiveOp
  lut_op.rs         # 21+ LutOps
datum/
  mod.rs            # ByteDatum: implements uor-foundation Datum
error/
  mod.rs
```

### Port Sources
| Source | Target |
|---|---|
| [crates/prism/src/lut.rs](../categorical-x/crates/prism/src/lut.rs) | `lut/` |
| [crates/prism/src/view/mod.rs](../categorical-x/crates/prism/src/view/mod.rs) | `view/mod.rs` |
| [crates/prism/src/view/simd.rs](../categorical-x/crates/prism/src/view/simd.rs) | `view/simd.rs` |
| [examples/rust/calculator/main.rs](../categorical-x/examples/rust/calculator/main.rs) encodings | `encoding/` |

### Tests & Benchmarks
- All LUT entries verified, ElementWiseView composition, ring axioms, SIMD parity, rkyv round-trip
- Benchmarks: lut.rs, view.rs, simd.rs

---

## Phase 2: Graph, Subgraphs & Fusion (Sprint 2, Step 3)

**Goal**: Single graph type, subgraphs, parallel levels, single-pass fusion. Auto-execute when deps satisfied.

### Crate: `crates/holo-graph/src/`
```
lib.rs
graph/
  mod.rs            # Graph struct (rkyv-serializable)
  node.rs           # Node with generational IDs, SmallVec inputs
  edge.rs
  validate.rs
builder/
  mod.rs            # GraphBuilder (fluent API, builder pattern)
subgraph/
  mod.rs            # SubgraphDef
  flatten.rs        # flatten_subgraph: template instantiation + ID remapping
fusion/
  mod.rs            # Single-pass orchestrator
  constant.rs       # Constant folding
  cse.rs            # Common subexpression elimination
  view_fusion.rs    # Unary chain → FusedView
schedule/
  mod.rs            # ExecutionSchedule, ParallelLevel
  levels.rs         # build_parallel_levels()
  toposort.rs
  critical_path.rs
constant/
  mod.rs            # ConstantStore
error/
  mod.rs
```

### Key Designs
- **Single Graph** (fixes v1 hacks #1, #2): one type for building and optimization
- **Op enum**: `Input | Output | Prim(PrimOp) | Lut(LutOp) | FusedView(ElementWiseView) | Constant(ConstantId) | CallSubgraph(SubgraphId)`
- **SubgraphDef** (port concept from [crates/graph/src/subgraph.rs](../categorical-x/crates/graph/src/subgraph.rs)): reusable templates, two-phase flatten with input bindings
- **Parallel levels** (port from [crates/compiler/src/partition/schedule.rs](../categorical-x/crates/compiler/src/partition/schedule.rs)): Level N nodes have all deps in levels < N → concurrent execution
- **Single-pass fusion** (fixes v1 hacks #3-5): one toposort walk → constant fold → CSE → view fusion

---

## Phase 3: .holo Archive Format (Sprint 2, Step 4)

**Goal**: Single clean .holo format with execution entrypoints. Page-aligned mmap. rkyv zero-copy. No backwards compat.

### Crate: `crates/holo-archive/src/`
```
lib.rs
format/
  mod.rs            # HOLO_MAGIC, PAGE_SIZE=4096, header struct
  header.rs         # HoloHeader: magic, version, graph/weights offsets, checksums,
                    #   sections table, entrypoint info
  pipeline.rs       # PipelineHeader for multi-model bundles
section/
  mod.rs            # EmbeddableSection trait
  table.rs          # Section table serialization
  weight_index.rs   # TensorMetadata, WeightDType, QuantizationParams
  layer_header.rs   # LayerHeader with execution entrypoints
entrypoint/
  mod.rs            # LayerDescriptor, LayerEntrypoint, TensorPort
  schedule.rs       # Embedded execution schedule (parallel levels)
writer/
  mod.rs
  holo_writer.rs    # HoloWriter: set_graph, set_weights, add_section → build
  pipeline_writer.rs # Multi-model composition
loader/
  mod.rs            # HoloLoader: sync mmap loading
  plan.rs           # LoadedPlan: mmap + ArchivedGraph pointer
  pipeline.rs       # Multi-model access
  bytes.rs          # load_from_bytes() fallback for WASM/embedded
layer/
  mod.rs            # LayerRef: Embedded | External | Registry { url, version }
weight/
  mod.rs
  quantize.rs       # QuantizationScheme
  compress.rs       # Zstd/LZ4 (feature-gated)
checksum/
  mod.rs            # CRC32
error/
  mod.rs
```

### Execution Entrypoints in Archive
The archive header includes `LayerHeader` with:
```rust
pub struct LayerDescriptor {
    pub id: LayerId,
    pub name: String,
    pub entrypoint: LayerEntrypoint,      // What to execute
    pub inputs: Vec<TensorPort>,           // I/O signature
    pub outputs: Vec<TensorPort>,
    pub group: u32,                        // Parallel execution group
    pub plan_offset: u64,                  // Embedded plan location in archive
    pub plan_size: u64,
}

pub struct LayerHeader {
    pub layers: Vec<LayerDescriptor>,
    pub schedule: Vec<Vec<LayerId>>,       // Execution order (parallel levels)
}
```

A `.holo` file is self-describing: it knows what graphs it contains, how they connect, and in what order to execute them.

### LayerRef for Network Distribution
```rust
pub enum LayerLocation {
    Embedded { offset: usize, size: usize },
    External(PathBuf),
    Registry { url: String, version: String },
}
```
Subgraphs can reference remote registries — enabling distributed execution where nodes fetch subgraphs from the network.

---

## Phase 4: KV-Lookup Execution Engine (Sprint 3, Step 5)

**Goal**: Every op = O(1) KV lookup. Rayon parallel levels. Loads from .holo via mmap.

### Crate: `crates/holo-exec/src/`
```
lib.rs
kv/
  mod.rs
  store.rs          # KvStore: table dispatch for all ops
eval/
  mod.rs            # Executor trait
  executor.rs       # KvExecutor: level-based graph evaluation
buffer/
  mod.rs
  arena.rs          # BufferArena: zero-copy intermediates
mmap/
  mod.rs            # MmapGraph: loads from .holo archive
  loader.rs         # Integration with holo-archive
parallel/
  mod.rs            # Rayon parallel level exec (feature-gated)
error/
  mod.rs
```

### Key Design
```rust
impl KvExecutor {
    pub fn execute(&self, plan: &LoadedPlan) -> Result<Vec<Vec<u8>>> {
        let schedule = &plan.schedule();
        for level in &schedule.levels {
            // feature: parallel → rayon par_iter
            // no feature → sequential iter
            execute_level(level, &self.store, &mut buffers);
        }
    }
}
```

---

## Phase 5: Calculator Example & Benchmarks (Sprint 3, Step 6)

### Files
- `examples/calculator.rs` — 29+ functions, pi-F-lambda, LUT vs f64, composition
- `crates/holo-bench/benches/` — calculator, lut_vs_f64, fusion, batch, kv_lookup, mmap, parallel, archive

---

## Future Sprints (Full Task Breakdown)

### Sprint 4: Q1/Q2/Q3 Quantum Level Scaling
- [ ] Design segmented LUT strategy for Q1 (16-bit): high-byte table + low-byte table + correction table
- [ ] Implement `Q1Ring` trait for `ByteRing16` (Z/65536Z)
- [ ] Implement Q1 `Datum` with 16-bit quantum level
- [ ] Build Q1 arithmetic tables (segmented approach to avoid 4GB tables)
- [ ] Build Q1 activation tables (segmented)
- [ ] Extend `ElementWiseView` to `ElementWiseView<N>` generic over table size or use segmented variant
- [ ] Add Q1 encoding types (16-bit angle, signed, unsigned)
- [ ] Design Q2/Q3 strategy (likely hierarchical segmentation or sparse tables)
- [ ] Benchmark Q1 vs Q0 lookup performance
- [ ] Tests: ring axioms at Q1, encoding round-trips, critical identity

### Sprint 5: LUT-GEMM for AI Model Inference
- [ ] Port Psumbook concept from [categorical-x/crates/backends/src/backends/cpu/lut_matmul.rs](../categorical-x/crates/backends/src/backends/cpu/lut_matmul.rs)
- [ ] Implement 4-bit weight quantization with partial sum books
- [ ] Implement 8-bit weight quantization
- [ ] Column-parallel psumbook construction (rayon-parallelized)
- [ ] Integrate LUT-GEMM into KvExecutor as MatMulLUT op
- [ ] Support deferred/streaming weight loading from mmap'd .holo archives
- [ ] Add `MatMul`, `BatchMatMul` op variants to graph
- [ ] Benchmark: LUT-GEMM vs naive matmul vs BLAS for small/medium/large matrices
- [ ] Tests: correctness against naive matmul for various sizes and quantization levels

### Sprint 6: Simplified Compiler Pipeline
- [ ] Design 3-stage pipeline: Parse → Fuse → Emit (replaces v1's 8-stage)
- [ ] Stage 1 (Parse): expression/graph input → internal Graph with validation
- [ ] Stage 2 (Fuse): single-pass fusion (already in holo-graph) + kernel cost integration
- [ ] Stage 3 (Emit): fused Graph → .holo archive with execution schedule
- [ ] Liveness analysis for buffer reuse (port concept from [crates/compiler/src/workspace/liveness.rs](../categorical-x/crates/compiler/src/workspace/liveness.rs))
- [ ] Workspace planning: compute buffer offsets and total workspace size
- [ ] Emit execution entrypoints into .holo header
- [ ] Benchmark: compilation time for various graph sizes
- [ ] Tests: compile → execute → verify for diverse graph topologies

### Sprint 7: FFI Bindings (was Sprint 8)
- [ ] Python bindings via UniFFI (auto-generated)
- [ ] TypeScript/WASM bindings via wasm-bindgen
- [ ] C bindings via cbindgen
- [ ] Thin translation layer — all logic in Rust crates
- [ ] FFI call overhead target: < 10 ns
- [ ] Integration tests in each target language
- [ ] Package and publish to PyPI, npm

### Sprint 8: Constrained Device Validation
- [ ] ESP32 cross-compilation and testing
- [ ] Raspberry Pi cross-compilation and testing
- [ ] Binary size budget analysis (target: < 100KB for core)
- [ ] Memory usage profiling on constrained targets
- [ ] `no_std` + `no_alloc` mode for ultra-constrained (static buffer only)
- [ ] Feature matrix: which features available on which targets

### Sprint 9: Tokio Integration + Async Execution
- [ ] Async graph compilation
- [ ] Async streaming evaluation for large models
- [ ] Async network transport for distributed execution
- [ ] Integration with holo-net for async P2P
- [ ] Benchmark: async vs sync execution overhead

### Sprint 10: Codegen from Descriptors
- [ ] Port ISA descriptor concept from [categorical-x/crates/holo/codegen/](../categorical-x/crates/holo/codegen/)
- [ ] Build-time code generation for instruction dispatch
- [ ] Generate op enum variants from descriptor files
- [ ] Generate dispatch tables from descriptors
- [ ] Proc-macro for op registration

### Sprint 11: Network Distribution (holo-net — deferred, will port from categorical-x)
- [ ] Port/adapt categorical-x's `crates/network/` and `crates/orchestrate/` crates
- [ ] Adapt to use holo-graph/holo-exec types
- [ ] Implement `Registry` variant of `LayerLocation`: fetch subgraphs by URL + version
- [ ] Distributed scheduler: assign subgraph levels to worker nodes
- [ ] Content-addressed subgraph storage
- [ ] Worker pool, P2P discovery, remote execution, fault tolerance

### Sprint 12+: hologram-ai (Separate Greenfield — ONNX/GGUF/GGML Support)
A separate project (`hologram-ai`) built on top of hologram-greenfield's core:
- [ ] Design extensible operation registry: consumers can register custom ops via trait impls
- [ ] ONNX operation support: parse ONNX protobuf, map ONNX ops → hologram Graph ops
- [ ] ONNX model import: load `.onnx` files → OperationGraph → fuse → .holo archive
- [ ] GGUF model support: parse GGUF format, extract quantized weights + graph topology
- [ ] GGML model support: parse GGML format, map operations to hologram primitives
- [ ] Consumer-extensible op system: `trait CustomOp` that users implement to add new operations
  - Custom ops provide their own LUT tables or compute functions
  - Registration via `OpRegistry::register::<MyOp>()`
  - Custom ops participate in fusion when they provide `ElementWiseView`
- [ ] Quantized weight import: ONNX Q8/Q4 → hologram LUT-GEMM quantization
- [ ] Shape inference for ONNX dynamic shapes → hologram symbolic dimensions
- [ ] End-to-end test: load ONNX model → compile → execute via KV lookups → verify accuracy
- [ ] Benchmark: hologram execution vs ONNX Runtime for small models

---

## v1 Hack Fix Summary

| # | v1 Problem | v2 Fix | Phase |
|---|-----------|--------|-------|
| 1 | Two graph types | Single `Graph` with builder | 2 |
| 2 | Separate constant stores | `ConstantStore` inside `Graph` | 2 |
| 3 | 8 independent fusion passes | Single-pass fusion engine | 2 |
| 4 | Kernel selection after fusion | Cost integrated into fusion | 2 |
| 5 | Separate workspace planning | Joint optimization in single pass | 2 |
| 6 | Two execution models | Single `KvExecutor` with parallel levels | 4 |
| 7 | 60+ OpKind variants | 10 `PrimOp` + `LutOp` + `FusedView` | 1 |
| 8 | Activation LUT as config | Automatic detection in fusion | 2 |
| 9 | Weight size thresholding | Archive format weight sections | 3 |

---

## Dependency Graph

```
uor-foundation (git v3.5.0, traits only, no_std)
       |
   holo-core (LUT, views, ring, encoding — no_std + alloc)
       |
   holo-graph (graph, subgraphs, fusion, scheduling)
       |
   holo-archive (.holo format, rkyv, mmap, entrypoints, weights)
       |
   holo-exec (KV executor, buffer, parallel levels)
       |
   holo-bench (criterion benchmarks)

Root crate (src/lib.rs) re-exports: holo-core, holo-graph, holo-archive, holo-exec
Future: holo-net (port from categorical-x when core pipeline is stable)
Examples: examples/calculator.rs
```

**Invariants**:
- `holo-core` depends ONLY on `uor-foundation`
- All crates compile for `wasm32-unknown-unknown` with appropriate feature gates
- Max 3 function args; builder pattern for more
- Macros for repeated trait implementations
- Single archive format — no backwards compat
- Root crate is the only consumer-facing dependency
