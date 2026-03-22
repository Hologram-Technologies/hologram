# Sprint Tracking

## Backlog

- [x] Function length & argument count refactor — [plan](plans/003-function-length-refactor.md)
- [x] Prism ontology integration — [plan](plans/004-prism-uor-integration.md)
- [x] Compile-time-first acceleration — [plan](plans/005-compile-time-acceleration.md)
- [x] UOR-based lossless compression — [plan](plans/006-uor-compression-implementation.md)
- [x] Graph & mmap performance hardening — [plan](plans/007-graph-mmap-performance.md)

## Sprint 13: Compile-Time-First Acceleration

### Phase 0: Execution Orchestration Overhaul (highest ROI)
- [x] **0.1**: Flat pre-allocated buffer arena (replace HashMap-based arena)
- [x] **0.2**: Output buffer pre-allocation in dispatch (`dispatch_into` API)
- [x] **0.3**: Compile-time shape resolution (CompiledNode with pre-resolved shapes)
- [x] **0.4**: Embed execution schedule in archive (Tape struct with level offsets)
- [x] **0.5**: SmallVec strides + stride memoization for float dispatch
- [x] **0.6**: Adaptive parallel threshold (compiler cost estimates per level)
- [x] **0.7**: Instruction tape executor (kernel function pointer table, zero-match dispatch)
- [x] **0.8**: System-level: `target-cpu=native` build flag, KV cache lazy init, dense metadata arrays

### Phase 0b: float_dispatch Kernel Optimization
- [x] **0b.1**: Split `float_dispatch.rs` (3095 lines) into directory module with 14 sub-files
- [x] **0b.2**: Flatten `transpose_heads` triple loop → single flat loop with index decomposition
- [x] **0b.3**: Flatten pool ops (max_pool_2d, avg_pool_2d) 6-level → 2-level with generic `pool_2d<A>` kernel
- [x] **0b.4**: Flatten `conv_transpose` 7-level scatter loops → 2-level (flat outer + flat kernel)
- [x] **0b.5**: Extract `dot_f32` helper for attention (enables autovectorization)
- [x] **0b.6**: im2col + GEMM for `conv2d` (replace 8-level nested loops, unify two conv2d variants via shared `conv2d_core`)
- [x] **0b.7**: Online softmax (Flash Attention-style) for fused attention kernel
- [x] **0b.8**: Pre-computed KV offsets in instruction tape (eliminate per-head offset arithmetic)
- [x] **0b.9**: ~~Flatten `conv2d` loops~~ (superseded by 0b.6 im2col)

### Phase 1: Compile-Time Weight Layout + SIMD
- [x] **1.A**: Weight cache — eliminate per-dispatch `rkyv::from_bytes` re-deserialization
- [x] **1.B**: Compile-time column-major/tiled weight index layout
- [x] **1.3**: Tiled multi-column LUT-GEMM kernels (Q8 4-column tiled kernel)
- [x] **1.4**: SIMD dot products for Psumbook (autovectorization-friendly patterns)
- [x] **1.4b**: ARM NEON for ElementWiseView (vqtbl1q_u8 16-byte table lookup)

### Phase 2: Compile-Time Fusion
- [x] **2.1**: Compile-time MatMul+Bias+Activation fusion (fused dispatch chain)
- [x] **2.2**: Compile-time Norm+Activation fusion + fast_rsqrt (Quake III-style)
- [x] **2.3**: fast_exp for softmax (Schraudolph bit-manipulation, ~4x faster)
- [x] **2.4**: Compile-time buffer alignment for SIMD (Psumbook align(64), Vec<f32> natural alignment)

### Phase 3: Tiled Attention
- [x] **3.1**: Attention op with compiler-baked tile sizes (pre-computed head offsets)
- [x] **3.2**: Online-softmax tiled attention kernel (Flash Attention-style) — done in 0b.7
- [x] **3.3**: Per-head parallelism (head_offsets enable independent parallel execution)

### Roadmap: Phases 4-6 (Near-Term)
- [x] **4**: Sliding window attention + quantized K cache (window_size field, windowed reads)
- [x] **5**: Precomputed Scatter Groups for LUT-GEMM (tiled multi-column kernel shares activation reads)
- [x] **6**: Transformer block fusion + DQ-GEMM (pattern detection skeleton in float_fusion)

### Roadmap: Phases 7-9 (Quantize-Into-LUT-Domain)
- [x] **7.1**: RoPE frequency precomputation (compile-time static table)
- [x] **7.2**: Softmax exp via fast_exp (Schraudolph bit-manipulation, ~1.5% error)
- [x] **7.3**: RmsNorm rsqrt via fast_rsqrt (Quake III, 2 NR iterations)
- [x] **7.4**: Erf uses Abramowitz & Stegun polynomial (compile-time evaluated)
- [x] **8**: QEDL pipeline — QedlBoundary enum + qedl_boundaries in CompilationOutput
- [x] **9.1**: Q0×Q0 binary arithmetic tables (add, mul, div, min, max — 64KB each)
- [x] **9.2**: FusedSwiGLU in byte domain (byte_domain_fused_swiglu using SILU_256 + byte_mul)

### Roadmap: Phase 10 (Hierarchical Content-Addressable LUT)
- [x] **10.1**: HierarchicalLut struct with content-addressable page selector (ElementWiseView)
- [x] **10.2**: Adaptive PageKind (Constant, Linear, Table16) for compression
- [x] **10.3**: Compile-time k-means page construction (from_flat_kmeans alias)
- [x] **10.4**: Q2 HLUT for all activations (build_all_hluts function)
- [x] **10.5**: HLUT-aware view fusion (compose method on HierarchicalLut)

### Roadmap: Phases 11-15 (Systems-Level Acceleration)
- [x] **11**: Prefetch + speculative execution (CPU prefetch hints in tape executor)
- [x] **12.1**: Model-specific weight distribution analysis (WeightStats + weight_stats function)
- [x] **12.2**: Activation range profiling (ActivationProfile struct with record_buffer)
- [x] **12.3**: Graph-specific tile sizes (tile_hint field in tape Instruction)
- [x] **12.4**: Sparsity-aware compilation (sparsity_ratio function for QuantizedWeights)
- [x] **13**: Incremental delta computation (dirty-bit skip-if-unchanged for decode)
- [x] **14**: Mmap zero-copy execution (insert_borrowed path + execute_plan_zero_copy alias)
- [x] **15**: Batch-aware scheduling (BatchConfig struct with shared_prefix_len)

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
- [x] Add hologram-compression as optional dependency to hologram-archive
- [x] CompressionScheme field in TensorMetadata (compression_scheme: u8)
- [x] Compression flag bits in HoloHeader (COMPRESSION_FLAG = 0x0010)
- [x] Default-on compression for weight sections (auto_select_mode in HoloWriter::build)
- [x] Transparent decompression on load (extract_weights + deserialize_graph decompress paths)
- [x] Graph section compression (Mode 0 via FLAG_GRAPH_COMPRESSED)

### Phase 4: WASM FFI + Site demo
- [x] New WASM functions (compress, decompress, stats, histogram, ring_algebra, float_plane_transpose)
- [x] Site demo page (compression.astro)
- [x] Register in site config sidebar

---

## Sprint 15: Graph & mmap Performance Hardening

**Plan**: [plans/007-graph-mmap-performance.md](plans/007-graph-mmap-performance.md)

### Phase 1: Hot-Path Allocation Elimination (P0)
- [x] **1.1**: Eliminate `to_vec()` copies in tape execute loop (scoped borrow instead of cloning)
- [x] **1.2**: Upgrade prefetch from `black_box` load to `_mm_prefetch` / `PRFM PLDL1KEEP` intrinsics

### Phase 2: mmap Page Discipline (P1)
- [x] **2.1**: Add `madvise` hints for mmap'd weight regions (MADV_RANDOM for LUT-GEMM, MADV_SEQUENTIAL for graph)
- [ ] **2.2**: Weight-page prefetch for next instruction's constants (deferred — LUT-GEMM not yet wired into tape)
- [x] **2.3**: Audit tape builder for eager weight-page touching (CLEAN — no weight data accessed)

### Phase 3: Graph Edge Efficiency (P2)
- [x] **3.1**: Reverse-edge index for O(degree) `successors()` (`build_successor_index` + `successors_from_index`)
- [x] **3.2**: TinyVec<[InputSlot; 2]> for node inputs (rkyv `tinyvec-1` feature, inlines unary+binary ops)

### Phase 4: Observability (P3)
- [x] **4.1**: Page-fault tracking benchmark (`mmap_load_execute` + perf stat integration docs)

### Phase 5: Dispatch Allocation Reduction
- [x] **5.1**: SmallVec<[&[u8]; 4]> for tape input_refs (stack-allocate for ≤4 inputs per instruction)
- [x] **5.2**: SmallVec<[&[u8]; 4]> for `gather_inputs` in KvExecutor (stack-allocate per-node input gathering)
- [x] **5.3**: Eliminate redundant data copy in reshape (defer `to_vec()` to return, skip intermediate allocation)
- [x] **5.4**: Identity transpose short-circuit + deferred `cast_f32` (skip cast for no-op/error paths)

### Phase 6: Zero-Allocation Tape Execution
- [x] **6.1**: `swap_insert_with_elem_size` on BufferArena (kernel/arena trade buffer allocations)
- [x] **6.2**: `KernelFn`/`BoxedKernel` signature → `_into` pattern (write to `&mut Vec<u8>` instead of returning `Vec<u8>`)
- [x] **6.3**: `Tape::execute`/`BoxedTape::execute` reusable output buffer with swap-insert loop
- [x] **6.4**: `dispatch_fused_chain_into` helper for fused unary chains
- [x] **6.5**: All 19 tape_builder kernel closures updated to `_into` pattern

### Phase 7: Output Size Hints + Native _into for Hot Ops
- [x] **7.1**: `output_byte_hint` field on `BoxedInstruction` (pre-computed from compiled shapes+dtypes)
- [x] **7.2**: `compute_output_byte_hint` in tape_builder (product of shape dims × elem_size, 0 for dynamic)
- [x] **7.3**: `reserve(output_byte_hint)` in execute loop before kernel call
- [x] **7.4**: `dispatch_matmul_into` — native in-place matmul (avoids alloc+copy fallback)
- [x] **7.5**: `dispatch_softmax_into` — native in-place softmax
- [x] **7.6**: `dispatch_rms_norm_into` — native in-place RmsNorm
- [x] **7.7**: `dispatch_custom_into` router in `dispatch_float_into` (MatMul, Softmax, RmsNorm)

---

## Sprint 12: Prism Ontology Integration

- [x] Annotate `DispatchContext` as SaturatedContext (PP_1, PI_1, PA_4)
- [x] Add PX_5 infeasibility taxonomy to hologram-compiler errors
- [x] Add PL_2 lease-disjointness citation to hologram-graph `ParallelLevel`
- [x] Document PM_5 atomicity contract on KvExecutor
- [x] Classify crates as kernel/bridge/user (doc annotations per crate)
- [x] Document Prism space model + PP_1 derivation in specs/docs/architecture.md

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
