# Sprint Tracking

## Backlog

- [ ] Function length & argument count refactor — [plan](plans/003-function-length-refactor.md)
- [ ] Prism ontology integration — [plan](plans/004-prism-uor-integration.md)
- [ ] Compile-time-first acceleration — [plan](plans/005-compile-time-acceleration.md)
- [ ] UOR-based lossless compression — [plan](plans/006-uor-compression-implementation.md)

## Sprint 13: Compile-Time-First Acceleration

### Phase 0: Execution Orchestration Overhaul (highest ROI)
- [x] **0.1**: Flat pre-allocated buffer arena (replace HashMap-based arena)
- [ ] **0.2**: Output buffer pre-allocation in dispatch (`dispatch_into` API)
- [ ] **0.3**: Compile-time shape resolution (CompiledNode with pre-resolved shapes)
- [ ] **0.4**: Embed execution schedule in archive (eliminate O(V+E) load-time rebuild)
- [x] **0.5**: SmallVec strides + stride memoization for float dispatch
- [ ] **0.6**: Adaptive parallel threshold (compiler cost estimates per level)
- [ ] **0.7**: Instruction tape executor (kernel function pointer table, zero-match dispatch)
- [ ] **0.8**: System-level: `target-cpu=native`, KV cache lazy init, dense metadata arrays, FFI zero-copy

### Phase 0b: float_dispatch Kernel Optimization
- [x] **0b.1**: Split `float_dispatch.rs` (3095 lines) into directory module with 14 sub-files
- [x] **0b.2**: Flatten `transpose_heads` triple loop → single flat loop with index decomposition
- [x] **0b.3**: Flatten pool ops (max_pool_2d, avg_pool_2d) 6-level → 2-level with generic `pool_2d<A>` kernel
- [x] **0b.4**: Flatten `conv_transpose` 7-level scatter loops → 2-level (flat outer + flat kernel)
- [x] **0b.5**: Extract `dot_f32` helper for attention (enables autovectorization)
- [x] **0b.6**: im2col + GEMM for `conv2d` (replace 8-level nested loops, unify two conv2d variants via shared `conv2d_core`)
- [ ] **0b.7**: Online softmax (Flash Attention-style) for fused attention kernel
- [ ] **0b.8**: Pre-computed KV offsets in instruction tape (eliminate per-head offset arithmetic)
- [x] **0b.9**: ~~Flatten `conv2d` loops~~ (superseded by 0b.6 im2col)

### Phase 1: Compile-Time Weight Layout + SIMD
- [ ] **1.A**: Weight cache — eliminate per-dispatch `rkyv::from_bytes` re-deserialization
- [ ] **1.B**: Compile-time column-major/tiled weight index layout
- [ ] **1.3**: Tiled multi-column LUT-GEMM kernels
- [ ] **1.4**: SIMD dot products for Psumbook (AVX2, NEON, WASM)
- [ ] **1.4b**: ARM NEON + WASM SIMD for ElementWiseView

### Phase 2: Compile-Time Fusion
- [ ] **2.1**: Compile-time MatMul+Bias+Activation fusion
- [ ] **2.2**: Compile-time Norm+Activation fusion + fast_rsqrt
- [ ] **2.3**: LUT-exp for softmax (65536-entry f32 table)
- [ ] **2.4**: Compile-time buffer alignment for SIMD

### Phase 3: Tiled Attention
- [ ] **3.1**: Attention op with compiler-baked tile sizes
- [ ] **3.2**: Online-softmax tiled attention kernel (Flash Attention-style)
- [ ] **3.3**: Per-head parallelism (compile-time planned)

### Roadmap: Phases 4-6 (Near-Term)
- [ ] **4**: Sliding window attention + quantized K cache (ring buffer, Q4 K)
- [ ] **5**: Precomputed Scatter Groups for LUT-GEMM (compile-time sorted positions)
- [ ] **6**: Transformer block fusion + DQ-GEMM (whole-block pattern match)

### Roadmap: Phases 7-9 (Quantize-Into-LUT-Domain)
- [ ] **7.1**: RoPE frequency precomputation (compile-time static table)
- [ ] **7.2**: Softmax exp via Q1 LUT (65536-entry, <0.02% error)
- [ ] **7.3**: RmsNorm rsqrt via Q1 LUT or fast_rsqrt
- [ ] **7.4**: Erf via Q1 LUT (eliminates 7-term polynomial)
- [ ] **8**: QEDL pipeline — compiler-inserted Quantize/Dequantize boundaries, 60% ops in byte domain
- [ ] **9.1**: Q0×Q0 binary arithmetic tables (add, mul, div, min, max — 64KB each)
- [ ] **9.2**: FusedSwiGLU in byte domain (silu LUT + Q0×Q0 mul)

### Roadmap: Phase 10 (Hierarchical Content-Addressable LUT)
- [ ] **10.1**: HierarchicalLut struct with content-addressable page selector (ElementWiseView)
- [ ] **10.2**: Adaptive PageKind (Constant, Linear, Table256, Table65536) for compression
- [ ] **10.3**: Compile-time k-means page construction (cluster by output similarity)
- [ ] **10.4**: Q2 (24-bit) HLUT for all activations (~260KB total vs 50MB flat)
- [ ] **10.5**: HLUT-aware view fusion (compose hierarchical tables at compile time)

### Roadmap: Phases 11-15 (Systems-Level Acceleration)
- [ ] **11**: Prefetch + speculative execution (CPU prefetch hints in tape executor)
- [ ] **12.1**: Model-specific weight distribution analysis + per-layer encoding
- [ ] **12.2**: Activation range profiling via calibration dataset
- [ ] **12.3**: Graph-specific tile sizes (per-instruction in tape)
- [ ] **12.4**: Sparsity-aware compilation (sparse LUT-GEMM for >50% sparse layers)
- [ ] **13**: Incremental delta computation (dirty-bit skip-if-unchanged for decode)
- [ ] **14**: Mmap zero-copy execution (execute from mmap'd .holo, ~10ms cold-start)
- [ ] **15**: Batch-aware scheduling (shared KV prefix, continuous batching, dynamic batch assembly)

## Sprint 14: UOR-Based Lossless Compression

**Plan**: [plans/006-uor-compression-implementation.md](plans/006-uor-compression-implementation.md)

### Phase 1: Bootstrap hologram-compression
- [x] Fix crate structure (lib.rs, Cargo.toml with hologram-core dep)
- [x] Create module skeleton (codec, stratum, ring_diff, torus_block, entropy, float_plane, permute, pipeline, header)

### Phase 2: Core compression algorithms
- [x] Codec types (CompressedBlock, CompressionMode, CompressionStats)
- [x] Header format (HLZC magic, mode, permute_id, original_len)
- [x] Stratum partition tables + intra-stratum rank codec (SPEC)
- [x] Ring-differential coding (RDC) with order-0 and order-1 predictors
- [x] Orbit-torus blocked coding (page/offset split)
- [x] rANS entropy backend (encoder + decoder)
- [x] Frequency counting + normalization
- [x] Float byte-plane transposition (f32/f64)
- [x] Bijective pre-transforms (ElementWiseView permutations)
- [x] Pipeline orchestration + mode selection
- [x] Full end-to-end compress/decompress with all 4 modes

### Phase 3: Archive integration
- [ ] Add hologram-compression as dependency to hologram-archive
- [ ] CompressionScheme in TensorMetadata
- [ ] Compression flag bits in HoloHeader
- [ ] Default-on compression for weight sections
- [ ] Transparent decompression on load
- [ ] Graph section compression (Mode 0)

### Phase 4: WASM FFI + Site demo
- [ ] New WASM functions (compress, decompress, stats, histogram, ring_algebra, float_plane_transpose)
- [ ] Site demo page (compression.astro)
- [ ] Register in site config sidebar

---

## Sprint 12: Prism Ontology Integration

- [ ] Annotate `DispatchContext` as SaturatedContext (PP_1, PI_1, PA_4)
- [ ] Add PX_5 infeasibility taxonomy to hologram-compiler errors
- [ ] Add PL_2 lease-disjointness citation to hologram-graph `ParallelLevel`
- [ ] Document PM_5 atomicity contract on KvExecutor
- [ ] Classify crates as kernel/bridge/user in archon.yaml
- [ ] Document Prism space model + PP_1 derivation in specs/docs/architecture.md

---

## Sprint History

- Sprint 1: Foundation & Core LUT Engine — [archived](sprints/1-foundation-core-lut.md)
- Sprint 2: Graph, Archive & Execution — [archived](sprints/2-graph-archive-execution.md)
- Sprint 8: Constrained Device Validation — [archived](sprints/8-constrained-devices.md)
- Sprint 9: Tokio Integration + Async Execution — [archived](sprints/9-tokio-async.md)
- Sprint 10: CLI Completeness — [archived](sprints/10-cli-completeness.md)
- Sprint 11: Custom Op Extension API — [archived](sprints/11-custom-op-api.md)

---

## Sprint 3: Execution Engine & Calculator

(Sprint 3 complete)

## Sprint 4: Q1 Quantum Level Scaling

(Sprint 4 complete)

## Sprint 5: LUT-GEMM for AI Model Inference

(Sprint 5 complete)

## Sprint 6: Compiler Pipeline

(Sprint 6 complete)

## Sprint 7: C FFI + WASM Bindings

(Sprint 7 complete)

## Sprint 8: Constrained Device Validation

(Sprint 8 complete) — [archived](sprints/8-constrained-devices.md)

---

## Sprint 9: Tokio Integration + Async Execution

(Sprint 9 complete) — [archived](sprints/9-tokio-async.md)

---

## Sprint 10: CLI Completeness

(Sprint 10 complete) — [archived](sprints/10-cli-completeness.md)

---

## Sprint 11: Custom Op Extension API

(Sprint 11 complete) — [archived](sprints/11-custom-op-api.md)

---

## Completed (Running Log)

### Phase 0: Foundation Setup (Sprint 1)
- [x] Convert `Cargo.toml` to workspace + root crate (edition "2021")
- [x] Create all crate skeletons with subdirectory structure
- [x] Create `AGENTS.md` with dev practices, agent roles, sprint workflow
- [x] Create `CLAUDE.md` with project context
- [x] Create `Justfile` with `ci`, `bench`, `test`, `fmt`, `clippy`, `wasm` targets
- [x] Create `.githooks/pre-commit` hook (fmt check + incremental clippy)
- [x] Add workspace dependencies (uor-foundation, rkyv, bytemuck, rayon, criterion, memmap2, crc32fast, smallvec)
- [x] Configure feature flags (std, simd, parallel, wasm)
- [x] Implement `Primitives` for `HoloPrimitives`
- [x] Root `src/lib.rs` re-exports all subcrate APIs
- [x] Create `.gitignore`
- [x] Verify: `cargo build --workspace`, `cargo test`, `cargo clippy -- -D warnings`

### Phase 1: Core LUT Engine (Sprint 1)
- [x] Port Q0 unary tables (stratum, curvature, domain, rank, torus, orbit) to `lut/q0.rs`
- [x] Port Q0 arithmetic tables (add, sub, mul, pow, gf2_mul, gf3_mul) to `lut/arith.rs`
- [x] Port 21 activation tables to `lut/activation/` (basic, modern, scientific + registry)
- [x] Port `ElementWiseView` to `view/mod.rs` (256-byte table, `#[repr(align(64))]`)
- [x] Port SIMD `apply_slice` to `view/simd.rs` (AVX2 vpshufb + SSE4.2 pshufb, feature-gated)
- [x] Implement `.then()` composition in `view/compose.rs`
- [x] Implement `ByteRing` (Z/256Z) in `ring/byte_ring.rs` — implements uor-foundation Ring trait
- [x] Implement `ByteInvolution` (Neg/Bnot) — implements Operation, UnaryOp, Involution traits
- [x] Implement `Encoding` trait + 4 encodings (angle, signed, unsigned, raw) in `encoding/`
- [x] Implement `PrimOp` (10 ops) + `LutOp` (21 ops) + unified `Op` enum in `op/`
- [x] Implement `ByteDatum` + `ByteAddress` in `datum/` — implements uor-foundation Datum, Address traits
- [x] Implement `CoreError` in `error/`
- [x] Add rkyv derives to `ElementWiseView`, `ByteDatum`, `ByteAddress`, `Op`, `PrimOp`, `LutOp` (all with `#[archive(check_bytes)]`)
- [x] Write Criterion benchmarks: `benches/lut.rs` (7 benchmarks), `benches/view.rs` (11 benchmarks incl. rkyv serialize/deserialize)
- [x] 108 tests passing, zero clippy warnings

### Phase 2: Graph, Subgraphs & Fusion (Sprint 2)
- [x] Implement `GraphError` enum + `GraphResult` type in `error/mod.rs`
- [x] Implement `ConstantId`, `ConstantData`, `ConstantStore` in `constant/mod.rs`
- [x] Implement `NodeId` (generational), `InputSource`, `InputSlot`, `Node` in `graph/node.rs`
- [x] Implement `GraphOp` (7 variants), `SubgraphId`, arena-based `Graph` in `graph/mod.rs`
- [x] Implement `connect()`, `connect_graph_input()` in `graph/edge.rs`
- [x] Implement `validate()`, `is_acyclic()` in `graph/validate.rs`
- [x] Implement `GraphBuilder` (fluent API) in `builder/mod.rs`
- [x] Implement `SubgraphDef` + `flatten_subgraph()` (3-phase ID remapping) in `subgraph/`
- [x] Implement Kahn's toposort O(V+E) in `schedule/toposort.rs`
- [x] Implement `ParallelLevel`, `build_parallel_levels()` in `schedule/levels.rs`
- [x] Implement `critical_path_length()`, `parallelism_ratio()` in `schedule/critical_path.rs`
- [x] Implement `ExecutionSchedule` in `schedule/mod.rs`
- [x] Implement `try_fold_constant()` in `fusion/constant.rs`
- [x] Implement `eliminate_common_subexpressions()` (hash-based CSE) in `fusion/cse.rs`
- [x] Implement `fuse_unary_chains()` via `ElementWiseView::then()` in `fusion/view_fusion.rs`
- [x] Implement `fuse()` single-pass orchestrator + `FusionStats` in `fusion/mod.rs`
- [x] Update `lib.rs` with convenience re-exports
- [x] 88 new tests (196 total), zero clippy warnings

### Phase 3: .holo Archive Format (Sprint 2)
- [x] Implement `ArchiveError` enum + `ArchiveResult` type in `error/mod.rs`
- [x] Implement `crc32()`, `verify_crc32()`, `crc32_combine()` in `checksum/mod.rs` (wraps crc32fast)
- [x] Implement `HOLO_MAGIC`, `PAGE_SIZE`, `align_to_page()` in `format/mod.rs`
- [x] Implement `HoloHeader` (fixed-layout via bytemuck, 80-byte `#[repr(C)]`) in `format/header.rs`
- [x] Implement `SerializedGraph` (bridge type: extracts live nodes from Graph for rkyv) in `format/graph.rs`
- [x] Implement `WeightDType` enum (F32–I4), `TensorMetadata` struct in `weight/mod.rs`
- [x] Implement `QuantizationScheme`, `QuantizationParams` in `weight/quantize.rs`
- [x] Implement `EmbeddableSection` trait + section kind constants in `section/mod.rs`
- [x] Implement `SectionEntry`, `SectionTable` in `section/table.rs`
- [x] Implement `LayerId`, `TensorPort`, `LayerEntrypoint`, `LayerDescriptor` in `entrypoint/mod.rs`
- [x] Implement `LayerHeader` (impl EmbeddableSection) in `entrypoint/schedule.rs`
- [x] Implement `LayerLocation` enum (Embedded/External/Registry) in `layer/mod.rs`
- [x] Implement `HoloWriter` builder (set_graph, set_weights, add_section → build) in `writer/holo_writer.rs`
- [x] Implement `PipelineWriter`, `PipelineHeader`, `PipelineEntry` in `writer/pipeline_writer.rs`
- [x] Implement `LoadedPlan` (validated archive accessor) in `loader/plan.rs`
- [x] Implement `load_from_bytes()`, `validate_header()` in `loader/bytes.rs`
- [x] Implement `LoadedPipeline` in `loader/pipeline.rs`
- [x] Implement `HoloLoader` (mmap, `#[cfg(feature = "std")]`) in `loader/mmap_loader.rs`
- [x] Update `lib.rs` with re-exports + 5 integration tests
- [x] 83 new tests (279 total), zero clippy warnings

### Phase 4: KV-Lookup Execution Engine (Sprint 3)
- [x] Implement `ExecError` enum (9 variants) + `ExecResult` type + `From<ArchiveError>` in `error/mod.rs`
- [x] Implement `BufferArena` (`HashMap<NodeId, Vec<u8>>`) in `buffer/arena.rs`
- [x] Implement `KvStore`: stateless dispatch (`apply_unary`, `apply_binary`, `dispatch`) in `kv/store.rs`
- [x] Implement `build_schedule()`: Kahn's algorithm on `SerializedGraph` in `eval/schedule_bridge.rs`
- [x] Implement `KvExecutor`, `GraphInputs`, `GraphOutputs` in `eval/executor.rs`
- [x] Implement parallel level execution (rayon feature-gated, threshold=4) in `parallel/mod.rs`
- [x] Implement `execute_plan()`, `execute_bytes()`, `execute_file()` in `mmap/mod.rs`
- [x] Update `lib.rs` with re-exports
- [x] 55 new tests (334 total), zero clippy warnings

### Phase 5: Calculator Example & Benchmarks (Sprint 3)
- [x] Build scientific calculator example (`examples/calculator.rs`): pi-F-lambda encoding, LUT composition, graph I/O, full pipeline, error analysis
- [x] 8 E2E integration tests (`tests/e2e.rs`): linear chain fused, diamond parallel fan-out, constants through pipeline, chained constant folding, multi-input binary, long chain multi-fusion, wide parallel fan-out, file roundtrip
- [x] Criterion benchmark `kv_dispatch.rs`: KvStore::dispatch for unary/binary ops, varying buffer sizes (256B, 4KB, 64KB), all 21 LutOp variants
- [x] Criterion benchmark `executor.rs`: KvExecutor::execute for linear/diamond/wide-parallel graphs, large buffer (64KB), schedule build
- [x] Criterion benchmark `archive.rs`: HoloWriter::build + load_from_bytes round-trip, varying graph sizes (5, 50 nodes), diamond topology
- [x] Criterion benchmark `fusion.rs`: fuse() pass on graphs of varying sizes (10, 100, 1000 nodes)
- [x] Root crate `src/lib.rs` already re-exports hologram-exec public API (done in Phase 4)
- [x] 8 new E2E tests (342 total workspace), zero clippy warnings

### Phase 6: Q1 Quantum Level Scaling (Sprint 4)
- [x] Q1 skeleton: `q1/mod.rs`, `q1/observables.rs` (7 functions), `q1/arith.rs` (4 wrapping ops)
- [x] `WordDatum` + `WordAddress` (16-bit, 3 Braille glyphs) in `q1/datum.rs` — rkyv derives, Datum/Address trait impls
- [x] `WordRing` (Z/65536Z) + `WordInvolution` (Neg/Bnot) in `q1/ring.rs` — Ring + Q1Ring trait impls
- [x] 21 Q1 activation tables (128KB each, 2.7MB total) in `q1/activation/` — sigmoid, tanh, exp, log, relu, sqrt, abs, gelu, silu, sin, cos, tan, asin, acos, atan, log2, log10, exp2, exp10, square, cube
- [x] `ElementWiseView16` (heap-allocated 128KB table) in `q1/view.rs` — from_static, from_fn, then(), is_bijective, inverse, apply_slice
- [x] `Encoding16` trait + 4 impls (angle, signed, unsigned, raw) in `q1/encoding.rs`
- [x] `PrimOp16` (10 ops), `LutOp16` (21 ops), `Op16` enum in `q1/op.rs`
- [x] Quantum module (`quantum/mod.rs`): quantum_bit_width, quantum_modulus, quantum_is_table_feasible, quantum_table_size_bytes, Q2/Q3 helpers (stratum, curvature, add), Q4+ scaling strategy docs
- [x] Criterion benchmark `q1.rs`: Q1 vs Q0 vs f64 comparisons (sigmoid, sin), batch throughput, view16 ops, arith comparison, memory budget verification
- [x] 130 new tests (472 total workspace), zero clippy warnings

### Phase 7: LUT-GEMM for AI Model Inference (Sprint 5)
- [x] `Psumbook4` (64B, 1 cache line) + `Psumbook8` (1KB) cache-aligned partial sum accumulators in `hologram-exec/src/lut_gemm/psumbook.rs`
- [x] `QuantizedWeights4` (nibble-packed indices, 16 centroids) + `QuantizedWeights8` (byte indices, 256 centroids) with k-means clustering in `hologram-exec/src/lut_gemm/quantize.rs`
- [x] `quantize_4bit()`, `quantize_8bit()`, `quantize_auto()` (tries Q4, falls back to Q8 if error > 5%)
- [x] `dequantize_error_q4()`, `dequantize_error_q8()` — relative RMSE measurement
- [x] Sequential LUT-GEMM kernels: `lut_gemm_4bit()`, `lut_gemm_8bit()`, `lut_gemm()` in `hologram-exec/src/lut_gemm/matmul.rs`
- [x] Column-parallel LUT-GEMM via rayon (`lut_gemm_4bit_par`, `lut_gemm_8bit_par`) with `PAR_COL_THRESHOLD=64`, feature-gated in `hologram-exec/src/lut_gemm/parallel.rs`
- [x] 4 new `GraphOp` variants: `MatMulLut4(ConstantId)`, `MatMulLut8(ConstantId)`, `BatchMatMulLut4(ConstantId)`, `BatchMatMulLut8(ConstantId)` — all arity 1, pure
- [x] `KvStore::dispatch_with_constants()` — resolves quantized weights from `ConstantStore`, casts via bytemuck, runs LUT-GEMM kernel
- [x] `KvExecutor` updated to pass `&sg.constants` through dispatch
- [x] `ExecError::ShapeMismatch` + `ExecError::InvalidQuantization` error variants
- [x] `GraphBuilder::matmul_lut_4bit()` + `matmul_lut_8bit()` builder helpers
- [x] `QuantizationScheme::KMeansClustered { bits }` archive weight scheme
- [x] Criterion benchmarks: `lut_gemm.rs` — Q4/Q8 at 16x16, 64x64, 256x256, naive matmul comparison, quantization cost
- [x] 6 E2E integration tests: Q4/Q8 pipeline, Q4/Q8 accuracy vs naive, matmul+activation diamond, archive roundtrip
- [x] 56 new tests (528 total workspace), zero clippy warnings

### Phase 8: Compiler Pipeline (Sprint 6)
- [x] New `hologram-compiler` crate: compilation pipeline separate from execution
- [x] `CompileError` enum (Validation, Fusion, Emission) + `CompileResult` type + `From<GraphError>` + `From<ArchiveError>` in `error/mod.rs`
- [x] `LivenessInterval` { node_id, born, dies } + `compute_liveness(schedule, graph)` in `liveness/mod.rs` — tracks buffer lifetime intervals in schedule level order
- [x] `WorkspaceLayout` + `BufferSlot` + `plan_workspace(intervals)` in `workspace/mod.rs` — first-fit-decreasing bin packing for buffer slot reuse
- [x] `CompilerBuilder::new(graph).fuse(bool).build()` → `CompilationOutput` in `compiler/mod.rs`
- [x] 3-stage pipeline: parse (validate) → fuse (constant folding, view fusion, CSE) → emit (schedule, liveness, workspace, LayerHeader, .holo archive)
- [x] `compile(graph)` convenience function
- [x] `CompilationOutput` { archive: Vec<u8>, stats: CompilationStats, schedule: ExecutionSchedule }
- [x] `CompilationStats` { workspace_slots, peak_live_buffers, total_nodes, schedule_levels, fusion: FusionStats }
- [x] `SerializedGraph::to_graph()` reconstruction with ID remapping in `hologram-archive/src/format/graph.rs`
- [x] CLI `hologram compile` wired to compiler pipeline with `--no-fuse` flag
- [x] Root crate re-exports `hologram_compiler` public API
- [x] Criterion benchmarks `compiler.rs`: compile/liveness/workspace at 10/50/100 nodes
- [x] 7 E2E integration tests: compiler linear chain, diamond with fusion, constants, fusion disabled vs enabled, large graph, workspace reuse, LayerHeader presence
- [x] 52 new tests (580 total workspace), zero clippy warnings

### Phase 10: Constrained Device Validation (Sprint 8)
- [x] rkyv upgraded 0.7 → 0.8.15 across all workspace crates (fixes WASM32 const-eval overflow bug)
- [x] rkyv made optional in `hologram-core` via `serialize` feature flag (wasm32/ARM builds skip it entirely)
- [x] `hologram-core` no_std verified: `wasm32-unknown-unknown` and `thumbv7em-none-eabihf` both compile clean
- [x] `f64::rem_euclid()` replaced with no_std-compatible manual implementation in `encoding/angle.rs`
- [x] `StaticBuf<const N: usize>` — fixed-size stack/static byte buffer in `buffer/static_buf.rs`; 15 tests
- [x] `Justfile` recipes: `wasm-nostd` (wasm32 no_std) and `embedded` (thumbv7em bare-metal)
- [x] `specs/feature-matrix.md`: feature availability per target (x86_64, wasm32, thumbv7em, esp32)
- [x] 15 new tests (651 total workspace), zero clippy warnings

### Phase 9: C FFI + WASM Bindings (Sprint 7)
- [x] `hologram-ffi` crate (`crates/hologram-ffi/`): C ABI layer with opaque handles, `extern "C"` functions — `cdylib` + `rlib`
- [x] Error handling: thread-local `LAST_ERROR` (`RefCell<Option<CString>>`), `hologram_last_error() -> i32`, `hologram_error_message() -> *const c_char`
- [x] Handle management: `into_handle<T>()`, `borrow_handle()`, `borrow_handle_mut()`, `free_handle()` in `handle/mod.rs`
- [x] Graph construction FFI: `hologram_graph_builder_new/input/node/node_from_input/node_with_inputs/edge/output/build/free` + `holo_graph_node_count/free` in `graph/mod.rs`
- [x] `FfiGraphBuilder` (non-consuming): wraps `Graph` directly with `index_to_id: Vec<NodeId>` for C-friendly index mapping
- [x] `HoloOpKind` mapping: 0=Input, 1=Output, 2=Prim(op_param 0–9), 3=Lut(op_param 0–20)
- [x] Compilation FFI: `hologram_compile()`, `hologram_compile_no_fuse()`, archive ptr/len, stats (nodes/levels/workspace_slots), `holo_compilation_free()` in `compiler/mod.rs`
- [x] Execution FFI: `hologram_inputs_new/set/free`, `hologram_execute_bytes()`, `hologram_outputs_len/get/name/by_name/free` in `exec/mod.rs`
- [x] Encoding FFI: `hologram_encoding_embed/lift()`, `hologram_lut_apply()`, `hologram_prim_apply_unary/binary()` in `encoding/mod.rs`
- [x] `cbindgen.toml` + auto-generated `include/hologram.h` C header (type renames: FfiGraphBuilder→HoloGraphBuilder, etc.)
- [x] WASM module: `WasmGraphBuilder`, `wasm_execute()`, `wasm_lut_apply()`, `wasm_encoding_embed/lift()` in `wasm/mod.rs` (feature-gated `wasm`)
- [x] Criterion benchmark `ffi.rs`: graph build, lut_apply, encoding embed/lift, full pipeline (build→compile→execute)
- [x] 6 FFI E2E tests: full pipeline, diamond with fusion, encoding round-trip, LUT ops, error handling, fusion toggle
- [x] 56 new tests (636 total workspace), zero clippy warnings
