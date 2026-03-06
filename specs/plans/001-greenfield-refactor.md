# Hologram Greenfield: Step-by-Step Refactoring Plan

## Context

V2 rewrite of `../hologram-backup` (15+ crate workspace, 8-stage compiler, dual graph types, 60+ OpKind variants, two execution models). Strips away hacks and IRs, using `uor-foundation` (v3.5.0, git main) to turn execution into O(1) KV-lookups via precomputed LUT tables.

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
- Network distribution deferred — will port from hologram-backup when core pipeline is stable

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
    holo-compiler/                   # Compilation pipeline: Graph → .holo archive
    holo-ffi/                        # FFI layer: C ABI (extern "C", cbindgen) + WASM (wasm-bindgen, feature-gated)
    holo-cli/                       # Async CLI with subcommands (exposes run())
    holo-bench/                     # Criterion benchmarks
  include/
    hologram.h                      # Auto-generated C header (cbindgen)
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

## Phase 0: Foundation Setup (Sprint 1, Step 1) — COMPLETED

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

## Phase 1: Core LUT Engine (Sprint 1, Step 2) — COMPLETED

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
| [crates/prism/src/lut.rs](../hologram-backup/crates/prism/src/lut.rs) | `lut/` |
| [crates/prism/src/view/mod.rs](../hologram-backup/crates/prism/src/view/mod.rs) | `view/mod.rs` |
| [crates/prism/src/view/simd.rs](../hologram-backup/crates/prism/src/view/simd.rs) | `view/simd.rs` |
| [examples/rust/calculator/main.rs](../hologram-backup/examples/rust/calculator/main.rs) encodings | `encoding/` |

### Tests & Benchmarks
- All LUT entries verified, ElementWiseView composition, ring axioms, SIMD parity, rkyv round-trip
- Benchmarks: lut.rs, view.rs, simd.rs

---

## Phase 2: Graph, Subgraphs & Fusion (Sprint 2, Step 3) — COMPLETED

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
- **SubgraphDef** (port concept from [crates/graph/src/subgraph.rs](../hologram-backup/crates/graph/src/subgraph.rs)): reusable templates, two-phase flatten with input bindings
- **Parallel levels** (port from [crates/compiler/src/partition/schedule.rs](../hologram-backup/crates/compiler/src/partition/schedule.rs)): Level N nodes have all deps in levels < N → concurrent execution
- **Single-pass fusion** (fixes v1 hacks #3-5): one toposort walk → constant fold → CSE → view fusion

---

## Phase 3: .holo Archive Format (Sprint 2, Step 4) — COMPLETED

**Goal**: Single clean .holo format with execution entrypoints. Page-aligned mmap. rkyv zero-copy. No backwards compat.

**Notes**: HoloHeader uses bytemuck (80-byte `#[repr(C)]` fixed layout) instead of rkyv for the header — rkyv's root-at-end design doesn't work for a fixed-position header. rkyv is used for variable-length data (graph, section table). 83 tests, 279 workspace total.

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

## Phase 4: KV-Lookup Execution Engine (Sprint 3, Step 5) — COMPLETED

**Goal**: Every op = O(1) KV lookup. Rayon parallel levels. Loads from .holo via mmap.

**Notes**: KvStore is stateless/zero-sized — all LUT tables are static in holo-core. BufferArena uses `HashMap<NodeId, Vec<u8>>`. Schedule bridge builds `ExecutionSchedule` directly from `SerializedGraph` via Kahn's algorithm (avoids reconstructing full arena `Graph`). Parallel execution is feature-gated on `rayon`. 55 tests, 334 workspace total.

### Crate: `crates/holo-exec/src/`
```
lib.rs              # Re-exports: KvStore, KvExecutor, GraphInputs, GraphOutputs,
                    #   BufferArena, ExecError, build_schedule, execute_plan/bytes/file
error/
  mod.rs            # ExecError (9 variants), ExecResult, From<ArchiveError>
kv/
  mod.rs
  store.rs          # KvStore: stateless dispatch (apply_unary, apply_binary, dispatch)
eval/
  mod.rs
  schedule_bridge.rs # build_schedule(SerializedGraph) → ExecutionSchedule via Kahn's
  executor.rs       # KvExecutor, GraphInputs, GraphOutputs
buffer/
  mod.rs
  arena.rs          # BufferArena: HashMap<NodeId, Vec<u8>>
mmap/
  mod.rs            # execute_plan, execute_bytes, execute_file (convenience)
parallel/
  mod.rs            # execute_level: rayon par_iter when ≥ 4 nodes (feature-gated)
```

---

## Phase 5: Calculator Example & Benchmarks (Sprint 3, Step 6) — COMPLETED

### Implementation Notes
- `examples/calculator.rs`: 4 demos (pi-F-lambda encoding, LUT composition, graph I/O, full pipeline with error analysis)
- `tests/e2e.rs`: 8 E2E integration tests covering full pipeline (build → fuse → serialize → load → execute → verify)
- 4 new Criterion benchmark files: `kv_dispatch.rs` (6 benchmarks), `executor.rs` (5 benchmarks), `archive.rs` (5 benchmarks), `fusion.rs` (3 benchmarks)
- Root crate re-exports already complete from Phase 4

### Files
- `examples/calculator.rs` — scientific calculator with pi-F-lambda, LUT composition, graph I/O, full pipeline, error analysis
- `tests/e2e.rs` — 8 E2E integration tests (linear fused, diamond parallel, constants, chained folding, multi-input, long chain, wide fan-out, file roundtrip)
- `crates/holo-bench/benches/kv_dispatch.rs` — KvStore dispatch benchmarks (unary/binary, varying buffer sizes, all LutOp variants)
- `crates/holo-bench/benches/executor.rs` — KvExecutor benchmarks (linear/diamond/wide-parallel graphs, large buffers, schedule build)
- `crates/holo-bench/benches/archive.rs` — HoloWriter + load_from_bytes roundtrip benchmarks (varying sizes, diamond topology)
- `crates/holo-bench/benches/fusion.rs` — fusion pass benchmarks (10, 100, 1000 node graphs)

---

## Future Sprints (Full Task Breakdown)

### Sprint 4: Q1/Q2/Q3 Quantum Level Scaling — COMPLETED

**Notes**: Full 65536-entry tables (128KB each) instead of segmented approach — fits L3 cache at ~2.7MB total. Parallel Q1 types in `q1/` submodule (not generic over quantum level). Q2/Q3/Q4+ documented as scaling strategy in `quantum/mod.rs`. 130 new tests (472 total workspace), zero clippy warnings.

- [x] Q1 skeleton: `q1/mod.rs`, `q1/observables.rs` (7 functions), `q1/arith.rs` (4 wrapping ops)
- [x] `WordDatum` + `WordAddress` (16-bit, 3 Braille glyphs) in `q1/datum.rs` — rkyv derives, Datum/Address trait impls
- [x] `WordRing` (Z/65536Z) + `WordInvolution` (Neg/Bnot) in `q1/ring.rs` — Ring + Q1Ring trait impls
- [x] 21 Q1 activation tables (128KB each, 2.7MB total) in `q1/activation/` — sigmoid through cube
- [x] `ElementWiseView16` (heap-allocated 128KB table) in `q1/view.rs` — std-gated
- [x] `Encoding16` trait + 4 impls (angle, signed, unsigned, raw) in `q1/encoding.rs`
- [x] `PrimOp16` (10 ops), `LutOp16` (21 ops), `Op16` enum in `q1/op.rs`
- [x] Quantum scaling module in `quantum/mod.rs` — Q0-Q4+ strategy, Q2/Q3 helpers
- [x] Criterion benchmark `q1.rs` — Q1 vs Q0 vs f64 comparisons
- [x] 130 new tests (472 total workspace), zero clippy warnings

### Sprint 5: LUT-GEMM for AI Model Inference — COMPLETED
- [x] Psumbook4 (64B) + Psumbook8 (1KB) cache-aligned accumulators in `holo-exec/src/lut_gemm/psumbook.rs`
- [x] QuantizedWeights4/8 + k-means clustering, quantize_auto, dequantize_error in `holo-exec/src/lut_gemm/quantize.rs`
- [x] Sequential LUT-GEMM kernels (lut_gemm_4bit/8bit) in `holo-exec/src/lut_gemm/matmul.rs`
- [x] Column-parallel LUT-GEMM (rayon, PAR_COL_THRESHOLD=64) in `holo-exec/src/lut_gemm/parallel.rs`
- [x] 4 new GraphOp variants (MatMulLut4/8, BatchMatMulLut4/8) + KvStore::dispatch_with_constants
- [x] KvExecutor updated to pass &sg.constants through dispatch
- [x] ExecError::ShapeMismatch + ExecError::InvalidQuantization
- [x] GraphBuilder::matmul_lut_4bit/8bit builder helpers
- [x] QuantizationScheme::KMeansClustered { bits } archive weight scheme
- [x] Criterion benchmarks: lut_gemm.rs (Q4/Q8 at multiple sizes, naive comparison, quantization cost)
- [x] 6 E2E tests (Q4/Q8 pipeline, accuracy vs naive, matmul+activation, archive roundtrip)
- [x] 56 new tests (528 total workspace), zero clippy warnings

### Sprint 6: Compiler Pipeline — COMPLETED
- [x] New `holo-compiler` crate: compilation pipeline separate from execution (`holo-graph` + `holo-archive` deps, no `holo-exec`)
- [x] `CompileError` enum + `CompileResult` type with `From<GraphError>` + `From<ArchiveError>` conversions
- [x] `LivenessInterval` + `compute_liveness(schedule, graph)` — buffer lifetime tracking in schedule level order
- [x] `WorkspaceLayout` + `plan_workspace(intervals)` — first-fit-decreasing bin packing for buffer slot reuse
- [x] 3-stage pipeline: Parse (validate) → Fuse (constant folding, view fusion, CSE) → Emit (schedule, liveness, workspace, LayerHeader, .holo)
- [x] `CompilerBuilder::new(graph).fuse(bool).build()` → `CompilationOutput { archive, stats, schedule }`
- [x] `compile(graph)` convenience function
- [x] `SerializedGraph::to_graph()` reconstruction with ID remapping
- [x] CLI `hologram compile` wired to compiler pipeline with `--no-fuse` flag
- [x] Root crate re-exports `holo_compiler` public API
- [x] Criterion benchmarks `compiler.rs`: compile/liveness/workspace at 10/50/100 nodes
- [x] 7 E2E tests: compiler linear chain, diamond, constants, fusion toggle, large graph, workspace reuse, LayerHeader
- [x] 52 new tests (580 total workspace), zero clippy warnings

### Sprint 7: C FFI + WASM Bindings — COMPLETED

**Notes**: Single `holo-ffi` crate (`cdylib` + `rlib`). `FfiGraphBuilder` wraps `Graph` directly (non-consuming) with `index_to_id: Vec<NodeId>` for C-friendly indexing. Thread-local `LAST_ERROR: RefCell<Option<CString>>` for error propagation. Op mapping: kind 0=Input, 1=Output, 2=Prim(param 0–9), 3=Lut(param 0–20). cbindgen renames FFI types to `Holo*` in C header. WASM bindings feature-gated behind `wasm`. 56 new tests (636 total workspace), zero clippy warnings.

- [x] `crates/holo-ffi/` crate skeleton: `Cargo.toml`, `lib.rs`, all module declarations
- [x] `error/mod.rs`: `FfiStatus` enum (8 codes), thread-local `LAST_ERROR`, `holo_last_error()`, `holo_error_message()`, `ffi_catch()` wrapper
- [x] `handle/mod.rs`: `into_handle<T>()`, `borrow_handle()`, `borrow_handle_mut()`, `free_handle()`
- [x] `graph/mod.rs`: `FfiGraphBuilder` + all `holo_graph_builder_*` + `holo_graph_node_count/free`
- [x] `compiler/mod.rs`: `holo_compile()`, `holo_compile_no_fuse()`, archive ptr/len, stats, `holo_compilation_free()`
- [x] `exec/mod.rs`: `holo_inputs_new/set/free`, `holo_execute_bytes()`, `holo_outputs_*`
- [x] `encoding/mod.rs`: `holo_encoding_embed/lift()`, `holo_lut_apply()`, `holo_prim_apply_unary/binary()`
- [x] `cbindgen.toml` + `include/hologram.h` auto-generated C header
- [x] `wasm/mod.rs`: `WasmGraphBuilder`, `wasm_execute()`, `wasm_lut_apply()`, `wasm_encoding_embed/lift()` (feature-gated)
- [x] `crates/holo-bench/benches/ffi.rs`: graph build, lut_apply, encoding, full pipeline benchmarks
- [x] 6 FFI E2E tests in `tests/e2e.rs`: full pipeline, diamond, encoding round-trip, LUT ops, error handling, fusion toggle

### Sprint 8: Constrained Device Validation — COMPLETED

**Notes**: Fixed `f64::rem_euclid()` (std-only) with manual no_std modulo in `angle.rs`. Upgraded rkyv 0.7 → 0.8.15 across all 30+ workspace files — removes WASM32 const-eval overflow bug and eliminates manual `serialize` workaround; rkyv 0.8 auto-derives `CheckBytes` with `Archive`. `StaticBuf<const N: usize>` in `crates/holo-core/src/buffer/` with 15 unit tests. `Justfile` `wasm-nostd` / `embedded` recipes use rustup toolchain paths (needed because Homebrew Rust lacks cross targets). 15 new tests (StaticBuf), zero clippy warnings, holo-core no_std ~35–40 KB text (well under 100 KB).

**Step 1: no_std Validation**
- [x] Verify `cargo build --target wasm32-unknown-unknown -p holo-core --no-default-features` compiles cleanly
- [x] Verify `cargo build --target thumbv7em-none-eabihf -p holo-core --no-default-features` compiles cleanly
- [x] Fix `f64::rem_euclid()` std-only call in `encoding/angle.rs`

**Step 2: `no_alloc` Static Buffer Mode**
- [x] Add `no_alloc` marker feature to `holo-core/Cargo.toml`; make `rkyv` optional via `serialize` feature
- [x] `crates/holo-core/src/buffer/static_buf.rs` — `StaticBuf<const N: usize>`: fixed-size stack/static buffer, `push/pop/extend_from_slice/as_slice/clear/is_full/capacity`
- [x] `crates/holo-core/src/buffer/mod.rs` — re-export `StaticBuf`
- [x] `holo-core/src/lib.rs` — re-export `buffer` module
- [x] 15 unit tests: capacity boundary, overflow, extend, Q0 LUT use case

**Step 3: Binary Size Analysis**
- [x] Measured: wasm32 ~40 KB `.text`, thumbv7em ~35 KB — documented in `specs/feature-matrix.md`

**Step 4: Justfile + Feature Matrix**
- [x] `Justfile` `embedded` recipe (thumbv7em via rustup toolchain)
- [x] `Justfile` `wasm-nostd` recipe (wasm32 no_std via rustup toolchain)
- [x] `specs/feature-matrix.md` created: features vs targets matrix

**Step 5: rkyv 0.8 Upgrade (added mid-sprint)**
- [x] Upgrade workspace rkyv dep: `0.7` → `0.8.15`
- [x] Remove all `#[archive(check_bytes)]` / `#[rkyv(derive(CheckBytes))]` (auto-derived in 0.8)
- [x] Replace `to_bytes::<_, N>` → `to_bytes::<rkyv::rancor::Error>` across all crates + `tests/e2e.rs`
- [x] Replace `check_archived_root + deserialize` → `rkyv::from_bytes::<T, rkyv::rancor::Error>`
- [x] Replace `rkyv::Infallible` usage (removed in 0.8)

**Target**: ~15 new tests, ~651 total workspace, zero clippy warnings, holo-core no_std < 100KB. ✓

### Sprint 9: Tokio Integration + Async Execution — COMPLETED

**Notes**: `holo-async` crate with `AsyncCompiler` (wraps `CompilerBuilder` in `spawn_blocking`), `AsyncExecutor` (wraps `execute_bytes` in `spawn_blocking`), and `execute_stream` (per-level `mpsc` channel). `KvExecutor::execute_with_progress<F>` added to `holo-exec` — `execute` delegates to it (no duplication). `execute_bytes_with_progress` exported from `holo-exec`. `LevelResult { level_index, nodes_executed }` is the per-level progress type. Dropping the receiver does not cancel execution; the task completes and channel sends are silently discarded. 16 new holo-async tests + 2 new holo-exec tests. 669 total workspace tests, zero clippy warnings.

**Step 1: `holo-async` crate**
- [x] `crates/holo-async/Cargo.toml`: deps `holo-compiler`, `holo-exec`, `holo-graph`, `tokio`
- [x] `src/compiler.rs`: `AsyncCompiler { graph, enable_fusion }`, `.fuse(bool)`, `.compile() -> JoinHandle<CompileResult<CompilationOutput>>`
- [x] `src/executor.rs`: `AsyncExecutor::execute(archive, inputs) -> JoinHandle<ExecResult<GraphOutputs>>`
- [x] `src/lib.rs`: re-exports `AsyncCompiler`, `AsyncExecutor`, `execute_stream`, `LevelResult`

**Step 2: Streaming API**
- [x] `src/stream.rs`: `execute_stream(archive, inputs) -> (Receiver<LevelResult>, JoinHandle<ExecResult<GraphOutputs>>)`
- [x] `LevelResult { level_index: usize, nodes_executed: usize }`
- [x] `KvExecutor::execute_with_progress<F>` in `holo-exec/src/eval/executor.rs`
- [x] `execute_bytes_with_progress<F>` in `holo-exec/src/mmap/mod.rs`, exported from `holo-exec/src/lib.rs`

**Step 3: Benchmarks**
- [x] `crates/holo-bench/benches/async_exec.rs`: sync vs async compile + execute (10-node chain)
- [x] `crates/holo-bench/benches/async_stream.rs`: batch vs streaming (20-node chain)

**Step 4: Root re-export**
- [x] `src/lib.rs`: `pub use holo_async;`

**Result**: 18 new tests, 669 total workspace, zero clippy warnings. ✓

### Sprint 10: CLI Completeness — COMPLETED

**Notes**: `run` command now fully functional: reads `.holo` archive, parses `--input INDEX:HEX` flags via `parse_input`, executes via `execute_bytes`, prints `name: hex` per output. `inspect` command prints file size, node count, input/output names, schedule level count. `CliError` gained `Exec` and `Archive` variants with `From` impls. 15 new tests in `holo-cli`. 684 total workspace tests, zero clippy warnings.

- [x] `CliError::Exec(ExecError)` + `From<ExecError>` in `crates/holo-cli/src/error/mod.rs`
- [x] `CliError::Archive(ArchiveError)` + `From<ArchiveError>` in `crates/holo-cli/src/error/mod.rs`
- [x] `commands/run_cmd.rs`: real execution — load archive, parse `--input INDEX:HEX` flags, execute, print outputs
- [x] Input parser: `parse_input(s: &str) -> Result<(u32, Vec<u8>), CliError>` for `INDEX:HEX` format
- [x] Output printer: `print_outputs(outputs: &GraphOutputs)` — `name: hex` per output
- [x] `commands/inspect.rs`: `hologram inspect <file>` — print file size, node count, input/output names, level count
- [x] Register `Inspect` variant in `commands/mod.rs` + `dispatch`
- [x] 15 new tests in `holo-cli` (parse_input variants, inspect helpers)
- [x] Sprint 9 archived to `specs/sprints/9-tokio-async.md`
- [x] Zero clippy warnings; `just ci` green — **684 total workspace tests**

### Sprint 11: Custom Op Extension API — COMPLETED

**Notes**: `CustomOpId(u32)` newtype + `GraphOp::Custom { id, arity }` in `holo-graph`. `CustomOpRegistry` with `Arc<dyn Fn>` handlers in `holo-exec/src/kv/registry.rs`. Registry threaded through private `execute_core` → `dispatch_level` → `KvStore::dispatch_with_constants` without breaking existing caller signatures (all pass `None`). New public `KvExecutor::execute_with_registry` and `execute_bytes_with_ops` entry points. `register_op!` macro in `holo-exec/src/lib.rs`. Custom ops serialize cleanly via rkyv (id + arity only); handlers are re-registered at startup. 15 new tests (11 integration + 4 unit). 700 total workspace tests, zero clippy warnings.

- [x] `CustomOpId(pub u32)` with rkyv derives, `raw()` method, re-exported from `holo-graph`
- [x] `GraphOp::Custom { id: CustomOpId, arity: u8 }` variant; `arity`, `is_pure`, `to_view` updated
- [x] `GraphBuilder::custom_op(id, arity, inputs)` builder method
- [x] `CustomHandler` type alias + `CustomOpRegistry::register/dispatch/len/is_empty` + `Default`
- [x] `register_op!(registry, id = N, arity = A, handler = ...)` macro
- [x] `KvStore::dispatch` / `dispatch_with_constants` accept `Option<&CustomOpRegistry>` — no breaking change (existing callers pass `None`)
- [x] `KvExecutor::execute_with_registry` + private `execute_core` + `execute_bytes_with_ops`
- [x] 11 integration tests in `tests/custom_ops.rs` + 4 unit tests in `registry.rs`
- [x] Sprint 10 archived to `specs/sprints/10-cli-completeness.md`
- [x] 700 total workspace tests, zero clippy warnings

### Sprint 12 & beyond — Moved to separate consumer libraries
Network distribution (holo-net) and AI model support (hologram-ai / ONNX/GGUF/GGML) will be implemented as separate libraries that depend on hologram-greenfield as a consumer.

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
      / \
holo-compiler    holo-exec
(Graph → .holo)  (KV executor, buffer, parallel levels)
      \  |  /
   holo-cli (async CLI with subcommands)
       |
   holo-bench (criterion benchmarks)

holo-ffi  (C ABI + WASM: extern "C" + cbindgen header, wasm-bindgen feature-gated)

Root crate (src/lib.rs) re-exports: holo-core, holo-graph, holo-archive, holo-compiler, holo-exec
Examples: examples/calculator.rs

Consumer libraries (separate repos):
  hologram-ai (ONNX/GGUF/GGML support)
  holo-net (network distribution)
```

**Invariants**:
- `holo-core` depends ONLY on `uor-foundation`
- All crates compile for `wasm32-unknown-unknown` with appropriate feature gates
- Max 3 function args; builder pattern for more
- Macros for repeated trait implementations
- Single archive format — no backwards compat
- Root crate is the only consumer-facing dependency
