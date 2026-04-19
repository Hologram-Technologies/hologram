# Sprint Tracking

## Backlog

- [x] Function length & argument count refactor — [plan](plans/003-function-length-refactor.md)
- [x] Prism ontology integration — [plan](plans/004-prism-uor-integration.md)
- [x] Compile-time-first acceleration — [plan](plans/005-compile-time-acceleration.md)
- [x] UOR-based lossless compression — [plan](plans/006-uor-compression-implementation.md)
- [x] Graph & mmap performance hardening — [plan](plans/007-graph-mmap-performance.md)
- [x] Dynamic sequence length — attention + slice fix — [plan](plans/016-dynamic-seq-attention-fix.md)
- [x] Zero-copy pipeline weights — [plan](plans/017-zero-copy-pipeline-weights.md)
- [x] Zero-copy graph access — [plan](plans/018-zero-copy-graph-access.md)
- [x] Runtime fat trim & allocation elimination — [plan](plans/019-runtime-fat-trim.md)
- [x] MatMul optimization — [plan](plans/020-matmul-optimization.md)
- [x] Epilogue fusion (Plan 005 Phase 2) — [plan](plans/030-epilogue-fusion.md)
- [x] Bias fusion (MatMul+Bias+Activation) — [plan](plans/031-bias-fusion.md)
- [x] Shape-aware tape execution API (feat/ai-optimization)

## Sprint 31: GEMM / MatMul / Conv2D Kernel Performance

**Plan**: [plans/037-gemm-conv2d-perf.md](plans/037-gemm-conv2d-perf.md)

Goal: maximize GEMM, MatMul, and Conv2D kernel throughput while keeping memory
usage flat. 10 optimizations across 4 phases.

### Phase 1: Bug Fix + Trivial Win
- [x] **1.1**: Complete LUT-GEMM + Conv2D in dispatch_kernel_par (BUG: catch-all error since ad7df78)
- [x] **1.2**: Eliminate per-call bias `.to_vec()` in Conv2D (borrow instead)

### Phase 2: GEMM Parallelism
- [x] **2.1**: Shared B-panel packing across M-tiles in matmul_k_outer
- [x] **2.2**: N-dimension parallelism in vecmat_mul + lower PAR_M_TILE_THRESHOLD

### Phase 3: Conv2D Kernel Improvements
- [x] **3.1**: Winograd batched GEMM parallelism (par_chunks_mut over 16 elements)
- [x] **3.2**: SIMD depthwise Conv2D (interior/border split + vectorization)
- [x] **3.3**: Fast im2col (memcpy interior for stride=1)

### Phase 4: Polish
- [x] **4.1**: Cache Winograd weight transform (1-entry thread-local, 16MB max)
- [x] **4.2**: A-panel packing — skipped (SIMD broadcast already fast, marginal gain vs complexity)
- [x] **4.3**: wasm32 SIMD128 micro-kernels (4×8 as two 4×4 halves)

---

## Sprint 30: CPU Optimization Sweep — Fusion, Prefetch, Parallelism

**Plan**: [plans/036-cpu-optimization-sweep.md](plans/036-cpu-optimization-sweep.md)

Goal: close remaining CPU-side optimization gaps across fusion, memory prefetch,
and parallelism. All changes are platform-agnostic (wasm + native).

### Phase 1: Fusion Gaps
- [x] **1.1**: AddRmsNorm + Activation fusion (GraphOp + TapeKernel + dispatch)
- [x] **1.1b**: InstanceNorm + Activation fusion (same pattern)
- [x] **1.2**: Binary in-place Add (zero-alloc Attention→Add(residual) via arena.add_inplace)
- [x] **1.3**: Transpose elimination (inverse transpose pairs → Passthrough)

### Phase 2: Multi-Level Weight Prefetch
- [x] **2.1**: 2-level lookahead in execute_inner (MADV_WILLNEED for i+1, i+2)
- [x] **2.2**: Early release already exists (MADV_DONTNEED for current level)

### Phase 3: Lock-Free LUT-GEMM Parallelism
- [x] **3.1**: Replace RefCell<WeightCache> with parking_lot::RwLock<WeightCache>
- [x] **3.2**: Enable rayon for LUT-GEMM levels (removed from needs_shared_state block)
- [ ] **3.3**: Per-thread Psumbook scratch (future: pre-populate cache for read-only access)

### Phase 4: Additional Fusion + Tuning
- [x] **4.1**: SwiGLU fusion from Silu + Mul pattern (try_fuse_swiglu)
- [ ] **4.2**: Adaptive sparse_v threshold (configurable per model/context)
- [ ] **4.3**: Activation checkpointing validation (verify compiler populates checkpoint_map)
- [x] **4.4**: InstanceNorm + Activation fusion (done in Phase 1)

### Phase 5: Memory Optimizations
- [x] **5.1**: Wire F16 activation compression (checkpoint nodes compress to F16 instead of evict)
- [ ] **5.2**: Wire workspace buffer reuse into arena allocation (not yet wired — architectural)

### Phase 6: WebGPU Kernel Parity (wasm GPU path)
- [x] **6.2**: Softmax + RmsNorm already existed; GroupNorm WGSL shader added

---

## Sprint 31: BufferArena Workspace Reuse

**Plan**: [plans/038-buffer-arena-workspace-reuse.md](plans/038-buffer-arena-workspace-reuse.md)

Goal: wire the compiler's `plan_workspace()` slot assignments into the executor's
`BufferArena` so non-overlapping nodes share physical buffers. Expected 20-40%
peak activation memory reduction with zero latency regression.

### Phase 1: Thread WorkspaceLayout into EnumTape
- [x] **1.1**: Add `slot_assignments: Vec<u32>` + `n_slots: u32` to `EnumTape`
- [x] **1.2**: `compute_slot_assignments()` — greedy interval coloring from producer/consumer maps

### Phase 2: MmapBuffer Free-List Recycling
- [x] **2.1**: Add `free_mmaps: Vec<MmapBuffer>` free-list to `BufferArena`
- [x] **2.2**: `evict()` pushes large MmapBuffers to free-list instead of dropping
- [x] **2.3**: `swap_insert_with_elem_size()` reuses from free-list before allocating

### Phase 3: Tests + Validation
- [x] **3.1**: 3 new tests: evict recycles, swap_insert reuses, small evict no-recycle
- [ ] **3.2**: Peak memory profiling (SD UNet, LLaMA 7B)

---

## Backlog: WebGPU Kernel Parity (wasm GPU path)
- [ ] Conv2d WGSL compute shader (im2col + tiled GEMM in WGSL)
- [ ] Attention WGSL shader (tiled, Flash Attention-style)

---

## Sprint 29: Conv2d Epilogue Fusion — Accelerate SD UNet Chain

**Plan**: [plans/035-conv2d-epilogue-fusion.md](plans/035-conv2d-epilogue-fusion.md)

Goal: fuse Conv2d + Activation (and Conv2d + Bias + Activation) into single tape
kernels, eliminating intermediate buffer materialization. GroupNorm → SiLU is already
fused (Sprint 23). For 512×512 SD inference with 23 ResNet blocks, this eliminates
~7.7GB of unnecessary memory traffic per step.

### Phase 1: Conv2d + Activation Epilogue Fusion
- [ ] **1.1**: Add `FusedConv2dActivation` GraphOp variant
- [ ] **1.2**: Add `try_fuse_conv2d_activation()` fusion pattern
- [ ] **1.3**: Add `InlineConv2dActivation` TapeKernel + dispatch
- [ ] **1.4**: Wire tape builder + exhaustive match coverage
- [ ] **1.5**: Tests: fusion detection, no-fuse (fan-out), correctness

### Phase 2: Conv2d + Bias + Activation (3-node)
- [ ] **2.1**: Add `FusedConv2dBiasActivation` GraphOp variant
- [ ] **2.2**: Add `try_fuse_conv2d_bias_activation()` fusion pattern
- [ ] **2.3**: Add `InlineConv2dBiasActivation` TapeKernel + dispatch
- [ ] **2.4**: Tests: 3-node pattern, non-constant bias rejection

### Phase 3: Validation
- [ ] **3.1**: Verify `can_reuse_input` for Attention → MatMul handoff
- [ ] **3.2**: Conv2d fusion benchmark
- [ ] **3.3**: End-to-end SD UNet latency comparison

---

## Sprint 28: KV Cache Quantization — Asymmetric Compression

**Plan**: [plans/034-kv-cache-quantization.md](plans/034-kv-cache-quantization.md)

Goal: reduce KV cache memory 2-4× via asymmetric quantization (K at f32, V at q4/q8)
with boundary layer protection and Walsh-Hadamard pre-rotation. Based on findings
from asymmetric KV compression research (V is robust to quantization; K errors
propagate exponentially through softmax).

### Phase 1: Boundary Layer Protection + Config
- [ ] **1.1**: Add `KvCacheConfig` / `KvBits` types with boundary layer support
- [ ] **1.2**: Modify `KvCacheState::new()` to accept config
- [ ] **1.3**: Tests: boundary layers remain f32, config defaults

### Phase 2: Per-Channel Min/Max Quantization
- [ ] **2.1**: Add `QuantizedKvBuffer` (q8/q4 storage with per-head scales)
- [ ] **2.2**: Online quantize on `write_layer()`, dequantize on `read_k()`/`read_v()`
- [ ] **2.3**: Tests: round-trip tolerance, asymmetric K/V precision

### Phase 3: Walsh-Hadamard Pre-Rotation
- [ ] **3.1**: Implement FWHT (in-place butterfly, O(d log d))
- [ ] **3.2**: Apply rotation before V quantization, inverse on dequantize
- [ ] **3.3**: Tests: self-inverse property, quantization error reduction

### Phase 4: Tape Integration
- [ ] **4.1**: Wire config through `TapeContext` → `KvCacheState`
- [ ] **4.2**: Verify KvWrite/KvRead dispatch handles quantized paths
- [ ] **4.3**: End-to-end integration tests

---

## Sprint 27: Performance Regression Fixes — M=1 MatMul + Tape Execution

**Branch**: `refactor/stable-diffusion-pipeline`

Goal: fix two regressions introduced by the fused dequant-matmul / mmap refactoring
in Sprint 26. M=1 (single-token decode) float matmul regressed 100-890% due to
missing SIMD in the MR=1 remainder path; tape execution regressed 140-272% due to
mandatory mmap syscalls on every small output.

### Fix 1: Specialized M=1 vecmat_mul
- [x] **F1.1**: Add `vecmat_mul` with NEON/AVX2 SIMD kernels (strided B, no packing)
- [x] **F1.2**: Early-out in `matmul_k_outer` when `m == 1`
- [x] **F1.3**: Relax bit-exact dequant test assertions to relative tolerance

**Result**: M=1 dispatch_matmul 49-58% faster; 1x64x64 now 20% faster than pre-Sprint-26 main.

### Fix 2: VecOwned arena buffer for small outputs
- [x] **F2.1**: Add `VecOwned(Vec<u8>)` variant to `ArenaBuffer`
- [x] **F2.2**: `swap_insert_with_elem_size` takes Vec ownership below 256 KB (no mmap syscall)
- [x] **F2.3**: Update all ArenaBuffer match sites (as_bytes, into_owned, get_mut_f32)

**Result**: Tape execution 2-4× faster for small/medium sizes; epilogue_fusion 49-75% faster.

---

## Sprint 26: UOR 0.1.0 Migration — Algebraic Performance Acceleration

**Plan**: [plans/033-uor-migration.md](plans/033-uor-migration.md)

Goal: merge the Q0→Q3 Cayley-Dickson algebraic acceleration chain from
`feat/uor0.1.0-migration`. Key wins: ~256× faster quantization, ~2× fewer MACs
via orbit compression, 24 inline hot-path kernels, cache-optimized fiber-ordered
GEMM, carry-driven dynamic precision dispatch, Q1 view fusion, and platform
prefetch.

### Merge
- [ ] **M.1**: Merge `feat/ai-optimization` → `main`
- [ ] **M.2**: Merge `origin/feat/uor0.1.0-migration` → `main` (resolve `tape.rs` conflict)
- [ ] **M.3**: Verify: `cargo test` + `cargo clippy` + `cargo fmt --check`

### New Modules (from migration branch)
- Q1/Q2/Q3 algebraic types (`hologram-core/src/{q1,q2,q3}/`)
- Carry-driven precision lifting (`hologram-core/src/carry/`)
- Orbit compression + Q16 quantization (`hologram-exec/src/lut_gemm/{orbit,quantize_q1,psumbook_q1}.rs`)
- Precision + QEDL compiler passes (`hologram-compiler/src/{precision,qedl}/`)
- Q1 view fusion pass (`hologram-graph/src/fusion/q1_view_fusion.rs`)
- 58 new conformance + performance-contract tests

---

## Sprint 25: Parallel Compilation + BLAKE3 Checksums

Goal: parallelise the compiler pipeline and migrate archive checksums from
CRC32 to BLAKE3 (format v2). ADR: [specs/adrs/001-blake3-checksums.md](adrs/001-blake3-checksums.md)

### Part A: CRC32 → BLAKE3 Migration
- [x] **A.1**: Migrate `checksum/mod.rs` from crc32fast to blake3
- [x] **A.2**: Expand header/section/error/weight checksum fields to `[u8; 32]`
- [x] **A.3**: Update writers + loader
- [x] **A.4**: Remove `crc32fast` dep

### Part B: Parallel Compilation (feature-gated `parallel`)
- [x] **B.1**: Add `parallel` feature + rayon to archive/graph/compiler crates
- [x] **B.2**: Parallelise graph + weight compression (`rayon::join`)
- [x] **B.3**: Parallelise schedule building (levels ∥ critical path)
- [x] **B.4**: Parallelise liveness analysis (`par_iter`)

---

## Sprint 24: Bias Fusion (Plan 031) — DONE

**Plan**: [plans/031-bias-fusion.md](plans/031-bias-fusion.md)

Goal: fuse MatMul+Add(bias)+Activation into a single TapeKernel, eliminating
two intermediate buffers. This is the pattern that `can_reuse_input` cannot
optimize away — the real performance win from epilogue fusion.

### Phase 1: Graph + Tape Variants
- [x] **1.1**: Add `FusedMatMulBiasActivation` GraphOp + `InlineMatMulBiasActivation` TapeKernel
- [x] **1.2**: Exhaustive match coverage (kv/store, CLI inspect, tape builder)

### Phase 2: Fused Kernel
- [x] **2.1**: `dispatch_matmul_bias_activation_into` — matmul + bias+activation in single pass

### Phase 3: Fusion Pass
- [x] **3.1**: `try_fuse_matmul_bias_activation()` — 3-node pattern (MatMul → Add(const) → Activation)
- [x] **3.2**: Wire into `fuse()` before 2-node matmul+activation pass

### Phase 4: Tests + Benchmark
- [x] **4.1**: Graph fusion test (`fuse_matmul_bias_activation_via_full_pass`)
- [x] **4.2**: Benchmark: transformer decode 2.81ms → 2.77ms (-1.24%, p=0.01)

---

## Sprint 23: Epilogue Fusion (Plan 030)

**Plan**: [plans/030-epilogue-fusion.md](plans/030-epilogue-fusion.md)

Goal: fuse matmul+activation and norm+activation into single TapeKernel variants,
eliminating memory round-trips between accumulator writeback and activation.
Driven by thermodynamic precision analysis (Landauer's principle: the epilogue is
the last reversible place to change precision gauges).

### Phase 1: MatMul + Activation Epilogue Fusion
- [x] **1.1**: Add `TapeKernel::InlineMatMulActivation` variant
- [x] **1.2**: Add `matmul_k_outer_fused` CPU kernel + `dispatch_matmul_activation_into`
- [x] **1.3**: Wire dispatch in tape executor
- [x] **1.4**: Add `GraphOp::FusedMatMulActivation` (rkyv-serializable)
- [x] **1.5**: Add `try_fuse_matmul_activation()` fusion pass
- [x] **1.6**: Wire tape builder: `FusedMatMulActivation` → `InlineMatMulActivation`
- [x] **1.7**: LUT-GEMM fused variants (`MatMulLut4Activation`, `MatMulLut8Activation`)

### Phase 2: Norm + Activation Fusion
- [x] **2.1**: Add fused `InlineRmsNormActivation`, `InlineLayerNormActivation`, `InlineGroupNormActivation`
- [x] **2.2**: Fused norm kernels (apply activation before writeback)
- [x] **2.3**: Add `try_fuse_norm_activation()` fusion pass

### Phase 3: Tests
- [x] **3.1**: Unit tests: fused kernel bit-identical to separate ops
- [x] **3.2**: Graph fusion tests: pattern detection + no-fuse cases
- [x] **3.3**: Tape E2E: fused vs unfused output identity

---

## Sprint 22: MatMul Optimization (Plan 020)

**Plan**: [plans/020-matmul-optimization.md](plans/020-matmul-optimization.md)

Goal: optimize MatMul kernels across CPU and GPU paths. Fix dispatch_gemm perf
bug, eliminate intermediate allocations, add register-blocked micro-kernel for
non-BLAS platforms, and enable batched matmul on GPU.

### Phase 1: dispatch_gemm Loop Restructuring
- [x] **1.1**: Pre-transpose A/B instead of runtime conditionals in inner loop
- [x] **1.2**: Use k-outer loop pattern via shared `matmul_k_outer` kernel
- [x] **1.3**: Apply alpha/beta scaling as post-pass

### Phase 2: dispatch_matmul_into Direct Write
- [x] **2.1**: Move `alloc_f32_in` + `transpose_f32` to shared helpers module
- [x] **2.2**: Rewrite dispatch_matmul_into to write directly to out_buf
- [x] **2.3**: Consolidate all matmul loops to shared `matmul_k_outer` kernel

### Phase 3: CPU Register-Blocked Micro-Kernel
- [x] **3.1**: 4×8 register-blocked matmul for non-BLAS platforms
- [x] **3.2**: Remainder handling for non-tile-aligned dimensions
- [x] **3.3**: Matmul size sweep benchmark (1×64×64 → 128×2048×2048)

### Phase 4: Batched MatMul GPU Dispatch
- [x] **4.1**: Metal batched SGEMM kernel (Z-dimension batch, shared-memory tiled)
- [x] **4.2**: WebGPU batched SGEMM kernel (Z-workgroup batch, deferred dispatch)
- [x] **4.3**: Wire batched dispatch through ComputeBackend trait (default Skipped)

---

## Sprint 21: Runtime Fat Trim & Allocation Elimination (Plan 019)

**Plan**: [plans/019-runtime-fat-trim.md](plans/019-runtime-fat-trim.md)

Goal: eliminate dead code, remove unused dependencies, eliminate `.to_vec()`
allocations in the hot path, and inline remaining high-frequency ops as
TapeKernel variants.

### Phase 1: Dead Code & Dependency Removal
- [x] **1.1**: Remove CUDA backend stub (`backend/cuda.rs` — always returns Skipped)
- [x] **1.2**: Replace `dirs` crate with cross-platform `home_dir()` helper (~25 transitive deps removed)
- [x] **1.3**: Gate `serde`/`toml` behind `cli` feature (remove from library dep tree)
- [x] **1.4**: Narrow `tokio` features (drop "full", use minimal subset)

### Phase 2: `.to_vec()` Elimination (71 calls audited)
- [x] **2.1**: Tape-builder passthrough for identity Cast and Reshape (zero dispatch, zero copy)
- [x] **2.2**: Norm `_into` variants write directly to `out_buf` (9 calls → zero intermediate Vec)
- [x] **2.3**: Attention zero-copy `heads_first` path via `Cow<[f32]>` (3 tensor copies eliminated)
- [x] **2.4**: `into_owned()` replacements for scatter_nd, cumsum, reverse_sequence, mask, RoPE (8 calls)

### Phase 2b: Inline TapeKernel Expansion
- [x] **2b.1**: Inline LayerNorm, AddRmsNorm, LogSoftmax (norm ops with baked params)
- [x] **2b.2**: Inline Attention, RotaryEmbedding (per-layer ops — uses TapeContext for position offset)
- [x] **2b.3**: Inline Gather, Concat (data movement ops with baked params)
- [x] **2b.4**: Inline remaining simple unary: Log, Sqrt, Cos, Sin, Sign, Floor, Ceil, Round, Erf
- [x] **2b.5**: Inline remaining simple binary: Min, Max

### Phase 3: Weight Cache & Dispatch Cleanup
- [x] **3.1**: Weight cache — eliminate double hash lookup via Entry API
- [x] **3.2**: `dispatch_float` marked `#[inline]` (kept for public API + test compat)
- [x] **3.3**: Allocating norm variants: `into_owned()` instead of `to_vec()`

### Benchmark Results (Sprint 21)
- tape::relu 64KB: **2.30 µs → 1.81 µs** (21% faster)
- transformer decode step: **5.99 ms → 2.77 ms** (54% faster, 2.2x speedup)
- softmax row_based 8192: **12.83 µs → 11.83 µs** (8% faster)
- tape::linear chain 4 nodes: **1.15 µs** (unchanged — already optimal)
- Total inline TapeKernel variants: **17 → 38** (all high-frequency ops covered)

---

## Sprint 20: Zero-Copy Graph Access (Plan 018)

**Plan**: [plans/018-zero-copy-graph-access.md](plans/018-zero-copy-graph-access.md)

Goal: eliminate 1.5s graph deserialization by using rkyv::access (zero-copy
archived field access) instead of rkyv::from_bytes (full owned deserialization).

### Phase 1: Optional Graph Compression
- [x] **1.1**: `compress_graph: bool` field on HoloWriter (default false)
- [x] **1.2**: `.compress_graph()` opt-in method
- [x] **1.3**: Skip compression when `compress_graph == false`

### Phase 2: Zero-Copy Graph Access
- [x] **2.1**: `GraphAccess` enum (Owned vs Archived) in LoadedPlan — lazy `OnceLock` deserialization
- [x] **2.2**: ~~`ArchivedConstantStore::get()`~~ — not needed: `graph()` transparently deserializes
- [x] **2.3**: ~~Archived-compatible maps~~ — not needed: lazy deser returns `&SerializedGraph`
- [x] **2.4**: ~~Update consumers~~ — all unchanged: `graph()` API returns `&SerializedGraph`
- [x] **2.5**: Decompress-once cache: `HoloLoader::load_cached()` with `.holo.cache` file + mmap

## Sprint 19: Zero-Copy Pipeline Weights (Plan 017)

**Plan**: [plans/017-zero-copy-pipeline-weights.md](plans/017-zero-copy-pipeline-weights.md)

Goal: pipeline archives store weights once in the wrapper, sub-archives reference
them via dedup index. Loading is zero-copy via mmap. Archive size halved, load
time from 20s+ to <1s.

### Phase 1: Archive Format + Loader (hologram)
- [x] **1.1**: `LoadedPlan::set_weights_borrowed` — zero-copy weight grafting from wrapper mmap
- [x] **1.2**: `PipelineWriter::build_with_shared_weights` — shared weight blob layout
- [x] **1.3**: `LoadedPipeline::from_bytes_zero_copy` — borrow sub-archive + shared weights from mmap

### Phase 2: Compiler (hologram-ai)
- [x] **2.1**: Shared weight extraction via `WeightStore` — `build_with_shared_weights()` wired
- [x] **2.2**: ~~Rewrite Deferred offsets~~ — not needed: offsets are per-component, loader grafts correct slice
- [x] **2.3**: `HoloRunner` zero-copy pipeline loading — dedup index resolution added to `from_storage()`

### Phase 3: Tests
- [x] **3.1**: Pipeline shared weights round-trip (build + load + resolve constants)
- [x] **3.2**: Zero-copy mmap pipeline loading (verify no allocation for weights)
- [ ] **3.3**: Weight dedup across prefill/decode models — needs TinyLlama model files
- [ ] **3.4**: E2E: compile TinyLlama pipeline + run with <1s load time — needs TinyLlama model files

---

## Sprint 18: Dynamic Shape Inference (Plan 016)

**Plan**: [plans/016-dynamic-seq-attention-fix.md](plans/016-dynamic-seq-attention-fix.md)

Goal: enable ONNX models with dynamic symbolic shapes (variable seq_len) to run
at runtime without `--seq-len` at compile time.

### Phase 1: Slice Axis Size Inference
- [x] **1.1**: `infer_slice_axis_size()` helper — infer actual axis dim from buffer + slice upper bound
- [x] **1.2**: Fix Slice dispatch to use inferred axis size instead of `end` heuristic

### Phase 2: Attention Buffer Validation
- [x] **2.1**: Validate Q/K/V buffer divisibility before seq inference
- [x] **2.2**: Validate K/V size consistency (prevent panic on mismatch)
- [x] **2.3**: Return `ShapeMismatch` with diagnostic info (buffer sizes, head config, inferred seq)

### Phase 3: Conformance Tests
- [x] **3.1**: GQA attention at variable seq lengths (seq=2 and seq=3)
- [x] **3.2**: Attention K/V mismatch → error (not panic)
- [x] **3.3**: Attention non-divisible Q → error
- [x] **3.4**: Slice with dynamic leading dimension (partial axis slice)
- [x] **3.5**: Slice where end == axis_size (fast path preserved)

---

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
- [x] **2.2**: Weight-page prefetch for next instruction's constants (already wired in execute loop)
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

### Phase 8: Enum Dispatch + LUT-GEMM Tape Wiring
- [x] **8.1**: `TapeKernel` enum — replaces `Box<dyn Fn>` with 8 inline variants (no vtable, no heap alloc)
- [x] **8.2**: `TapeContext` struct — carries ConstantStore + weights + RefCell\<WeightCache\> for LUT-GEMM
- [x] **8.3**: `TapeInstruction` / `EnumTape` — replaces `BoxedInstruction` / `BoxedTape`
- [x] **8.4**: `dispatch_kernel` match function — inlinable enum dispatch for all kernel types
- [x] **8.5**: LUT-GEMM Q4/Q8 wired into tape via `dispatch_lut_gemm_4` / `dispatch_lut_gemm_8`
- [x] **8.6**: `tape_builder.rs` rewritten — `resolve_kernel` returns enum variants, no closures
- [x] **8.7**: `execute_tape` in mmap/mod.rs builds `TapeContext` with weight access
- [x] **8.8**: 6 new EnumTape unit tests + tape vs KvExecutor benchmark (30% faster confirmed)

### Benchmark Results (Phase 8)
- Tape vs KvExecutor on Relu 64KB: **36.4 µs vs 47.2 µs** (1.30x faster)
- Tape linear chain (4 float nodes, 256B): **706 ns**

### Phase 8b: Fused Ops
- [x] **8b.1**: `FloatOp::AddRmsNorm` — fused Add + RmsNorm (eliminates intermediate residual buffer)
- [x] **8b.2**: `dispatch_add_rms_norm` + `dispatch_add_rms_norm_into` in norm.rs
- [x] **8b.3**: Wired into `dispatch_custom_into` router + `dispatch_custom` fallback

### Phase 9: Tape Correctness + Dispatch Coverage
- [x] **9.3**: Native `_into` for LayerNorm + LogSoftmax (extend `dispatch_custom_into`)
- [x] **9.5**: Dynamic size inference via `resolve_size` (Softmax/RmsNorm/LayerNorm size=0 sentinel → infer from input)
- [x] **9.7**: Tape-path conformance test vs KvExecutor (Relu→Neg chain, byte-for-byte output match)

### Phase 10: KvCache + Conformance
- [x] **10.5**: KvWrite/KvRead wired into tape (TapeKernel variants + TapeContext with RefCell\<KvCacheState\>)
- [x] **10.6**: Softmax conformance test (same graph through KvExecutor and EnumTape, byte-for-byte match)

### Phase 11: Weight Prefetch + LUT-GEMM Validation
- [x] **11.1**: `weight_offset_hint` on TapeInstruction + prefetch in execute loop for LUT-GEMM constants
- [x] **11.4**: LUT-GEMM Q4 tape integration test (build graph with quantized weights, execute via tape)

### Phase 12: Parallel Tape Execution
- [x] **12.1**: `execute_parallel` on EnumTape — Rayon within levels for ≥4 independent instructions
- [x] **12.1b**: `dispatch_kernel_par` — Sync-safe dispatch (skips RefCell ops: LUT-GEMM, KvCache)
- [x] **12.1c**: Adaptive threshold — falls back to sequential for small levels or shared-state ops

### Phase 13: Attention + Conv2d Dispatch Coverage
- [x] **13.1**: Attention routed through `dispatch_custom_into` (avoids generic fallback overhead)
- [x] **13.2**: Conv2d routed through `dispatch_custom_into`
- [x] **13.3**: RoPE explicitly falls back (needs position offset from ctx)

### Phase 14: Monomorphized SIMD Dispatch + Zero-Copy Output Write
- [x] **14.1**: Monomorphized unary dispatch for Relu, Neg, Abs, Sigmoid, Silu, Tanh, Exp, Reciprocal
- [x] **14.2**: Direct f32 write via `bytemuck::cast_slice_mut` (no intermediate Vec, no per-element extend)
- [x] **14.3**: Same pattern for binary elementwise, fused chain, norm, and matmul _into variants

### Benchmark Results (Phase 14)
- EnumTape Relu 64KB: **36.5 µs → 4.3 µs** (8.5x faster, autovectorization enabled)
- KvExecutor same graph: 44.6 µs (unchanged — still uses closure dispatch)

---

## Sprint 16: Multi-Backend Dispatch Architecture

**Plan**: [plans/009-multi-backend-dispatch.md](plans/009-multi-backend-dispatch.md)

### Phase 1: Backend Abstraction + Auto-Detection
- [x] **1.1**: `ComputeBackend` trait (dispatch_float, dispatch_matmul, name)
- [x] **1.2**: `CpuBackend` wrapping existing monomorphized SIMD dispatch
- [x] **1.3**: `MetalBackend` stub (auto-detected on macOS via build.rs `has_metal`)
- [x] **1.4**: `CudaBackend` stub (auto-detected via CUDA_HOME / nvcc)
- [x] **1.5**: `WebGpuBackend` stub (auto-detected on wasm32 targets)
- [x] **1.6**: `BackendSelector` enum (Auto/Cpu/Metal/Cuda/WebGpu) with `resolve()`
- [x] **1.7**: `default_backend()` priority: CUDA > Metal > WebGPU > CPU
- [x] **1.8**: `available_backends()` introspection
- [x] **1.9**: `build.rs` auto-detection + `cargo::rustc-check-cfg` registration
- [x] **1.10**: `TapeContext.backend` field with `BackendSelector::Auto` default

### Phase 2: Backend Wiring + Monomorphized Binary Dispatch
- [x] **2.1**: `dispatch_kernel` queries `backend.dispatch_float()` before CPU fallback
- [x] **2.2**: Backend resolved once at `execute()` start via `BackendSelector::resolve()`
- [x] **2.3**: Monomorphized binary elementwise (Add, Sub, Mul, Div, Min, Max — enables SIMD)

### Phase 3: Metal Compute Shader Kernels
**Plan**: [plans/010-metal-compute-kernels.md](plans/010-metal-compute-kernels.md)
- [x] **3.1**: `metal` crate (0.33) dependency, auto-linked on macOS
- [x] **3.2**: MetalBackend with shader compilation + pipeline caching (9 unary + 4 binary kernels)
- [x] **3.3**: Process-global cached backend via `OnceLock<Arc<MetalBackend>>` (shader compiled once)
- [x] **3.4**: Unary dispatch (relu, neg, abs, sigmoid, silu, tanh, exp, reciprocal, gelu)
- [x] **3.5**: Binary dispatch (add, sub, mul, div) with broadcasting
- [x] **3.6**: Size threshold (4MB) — CPU SIMD for small buffers, Metal for large
- [x] **3.7**: Metal conformance test (1.5M float relu, spot-check correctness)

### Phase 4: Metal SGEMM Matmul
- [x] **4.1**: Metal SGEMM compute shader (C[M,N] = A[M,K] × B[K,N], 2D grid dispatch)
- [x] **4.2**: `dispatch_matmul` wired — FloatOp::MatMul routed through dispatch_float → Metal
- [x] **4.3**: Size threshold (128×128 output) — CPU Accelerate BLAS for small matrices
- [x] **4.4**: Metal matmul conformance test (128×64 × 64×128, verified row correctness)

### Phase 5: Tiled SGEMM + Softmax + RmsNorm
- [x] **5.1**: Tiled SGEMM with threadgroup shared memory (16×16 tiles, barrier sync)
- [x] **5.2**: Metal softmax kernel (per-element row-wise with max/sum scan)
- [x] **5.3**: Metal RmsNorm kernel (per-element with mean-of-squares + rsqrt)
- [x] **5.4**: Softmax + RmsNorm routed through `dispatch_float` with size threshold
- [x] **5.5**: Metal softmax conformance test (1M floats, row sums to 1.0)

### Phase 6: MTLBuffer-Backed Arena
- [x] **6.1**: `ArenaBuffer` enum replacing `Cow<[u8]>` — supports Owned, Borrowed, and Metal variants
- [x] **6.2**: `as_bytes()` returns `&[u8]` for all variants (Metal via `contents()` pointer)
- [x] **6.3**: `insert_metal(id, metal::Buffer, elem_size)` — store GPU buffers directly in arena
- [x] **6.4**: `into_owned()` for take() — copies Metal buffer to Vec only when needed

### Phase 7: Zero-Copy Metal Output Path
- [x] **7.1**: `KernelOutput` enum (Skipped / Bytes / MetalBuffer) — dispatch tells executor how to store result
- [x] **7.2**: `DispatchResult` in tape.rs — execute loop handles Metal buffers via `insert_metal`
- [x] **7.3**: Metal unary dispatch returns `MetalBuffer` directly (skip Vec copy on output)
- [x] **7.4**: `ComputeBackend` trait updated — all backends return `KernelOutput` instead of `bool`

### Phase 8: Remaining GPU Work
- [x] **8.1**: Metal binary/matmul/softmax/rmsnorm all return MetalBuffer (full zero-copy path)
- [x] **8.2**: Async command buffer batching — `Mutex<Option<CommandBuffer>>` on MetalBackend, encode without commit per dispatch, `flush()` at level boundaries via `ComputeBackend::flush()` trait method
- [x] **8.3**: WebGPU/wgpu compute shader path — [plan](plans/012-webgpu-wgpu-compute.md)
  - [x] **8.3a**: Bootstrap — wgpu device init, WGSL compilation, pipeline caching, OnceLock caching
  - [x] **8.3b**: Complete elementwise — all 9 unary + 4 binary WGSL kernels with staging readback
  - [x] **8.3c**: Custom ops — tiled SGEMM (16×16), softmax, RmsNorm in WGSL
  - [x] **8.3d**: Deferred command encoder batching — `WgpuDeferred` + `flush_deferred()` — [plan](plans/013-webgpu-deferred-batching.md)
- ~~**8.4**: CUDA kernel implementations~~ (removed — CUDA stub deleted in Sprint 21)

### Phase 10: Weight Deduplication Primitive (Plan 021 Phase 3)
- [x] **10.1**: `WeightStore` — content-addressable weight storage with CRC32 identity + exact byte comparison
- [x] **10.2**: `WeightDedupIndex` / `WeightDedupEntry` — rkyv-serializable index for the deduplicated blob
- [x] **10.3**: `SECTION_WEIGHT_DEDUP` section kind, `EmbeddableSection` impl
- [x] **10.4**: Re-exported from `hologram_archive` crate root
- [x] **10.5**: 9 unit tests (empty, single, dedup, distinct, build, save-space, get, rkyv roundtrip, zero-copy)

### Phase 9: Zero-Overhead Dispatch — Flatten Abstraction Layers
**Plan**: [plans/011-zero-overhead-dispatch.md](plans/011-zero-overhead-dispatch.md)

Goal: eliminate all per-instruction overhead between the execute loop and the kernel compute. Target: O(1) constant-time dispatch with zero memory copies for the CPU path.

#### 9a: Inline Hot Ops (eliminate backend vtable + double match)
- [x] **9a.1**: 7 unary inline variants (InlineRelu, InlineNeg, InlineSigmoid, InlineSilu, InlineTanh, InlineGelu, InlineExp)
- [x] **9a.2**: 4 binary inline variants (InlineAdd, InlineMul, InlineSub, InlineDiv)
- [x] **9a.3**: tape_builder maps hot FloatOps to Inline variants at build time
- [x] **9a.4**: `inline_unary` / `inline_binary` helper functions (direct bytemuck cast, no dispatch_float_into)
- [x] **9a.5**: 3 inline conformance tests + inline benchmark
- [x] **9a.6**: `InlineMatMul { m, k, n }` — direct matmul_into call, backend GPU first then CPU fallback
- [x] **9a.7**: `InlineSoftmax { size }` / `InlineRmsNorm { size, epsilon }` — direct norm kernels, backend first
- [x] **9a.8**: `InlineAbs` / `InlineReciprocal` — complete unary inline coverage
- [x] **9a.9**: Visibility: `pub(crate) mod norm`, `pub(crate) fn resolve_size`, `pub(crate) fn dispatch_softmax_into/dispatch_rms_norm_into`

### Benchmark Results (Phase 9a+9b)
- EnumTape Relu 64KB: **4.23 µs → 2.54 µs** (40% faster — inline dispatch + in-place unary + output passthrough)
- KvExecutor same graph: 44.4 µs (unchanged)
- Tape vs KvExecutor: **17.5x faster**
- Tape linear chain (4 nodes, 256B): **1.11 µs**

#### 9b: Zero-Copy Arena Path (eliminate out_buf round-trip)
- [x] **9b.1**: Output passthrough — `arena.move_slot(src, dst)` when input has single consumer
- [x] **9b.2**: Pre-allocated arena output slots — `prewarm_arena()` pre-allocates with `output_byte_hint`
- [x] **9b.3**: In-place unary ops — `dispatch_inplace()` + `inline_unary_inplace()` when `can_reuse_input` flag set
- [x] **9b.4**: `apply_reuse_flags()` post-pass in tape_builder — consumer count analysis, sets `passthrough` and `can_reuse_input`

#### 9c: Typed Arena Access (eliminate per-call bytemuck cast)
- [x] **9c.1**: `arena.get_f32(id)` — returns `&[f32]` directly via localized `cast_slice`
- [x] **9c.2**: `arena.get_mut_f32(id)` — mutable f32 slice for in-place ops on `Owned` buffers
- [x] **9c.3**: `inline_unary_f32` / `inline_binary_f32` — typed kernel signatures, caller casts once
- [x] **9c.4**: In-place path refactored: `get_mut_f32` + `dispatch_inplace` + `move_slot` (no take+insert dance)

#### 9d: Direct Input Access (eliminate SmallVec collection for known arity)
- [x] **9d.1**: `TapeKernel::inline_arity()` — returns `Some(1)` / `Some(2)` / `None`
- [x] **9d.2**: Unary inline fast path — `arena.get_f32(input_indices[0])` directly, skip SmallVec
- [x] **9d.3**: Binary inline fast path — two direct `arena.get_f32` calls, skip SmallVec
- [x] **9d.4**: `dispatch_inline_unary` / `dispatch_inline_binary` — typed match wrappers
- [x] **9d.5**: Same restructuring applied to `execute_parallel` sequential fallback

#### 9e: Unsafe Fast Path (eliminate bounds checks in hot loop)
- [x] **9e.1**: `set_len()` instead of `resize()` in `inline_unary_f32` / `inline_binary_f32` (skip zero-fill)
- [x] **9e.2**: `arena.get_unchecked()` / `arena.get_f32_unchecked()` — skip bounds check
- [x] **9e.3**: Unchecked `input_indices` access when arity is known via `get_unchecked(0)`/`get_unchecked(1)`
- [x] **9e.4**: All unsafe gated with `#[cfg(not(debug_assertions))]` — debug builds use checked paths

### Performance Budget (per instruction)
| Layer | Current | After Phase 9 | Savings |
|-------|---------|---------------|---------|
| Backend vtable + Skipped check | ~60ns | 0ns (inline) | 60ns |
| Double match (category + op) | ~20ns | 0ns (inline) | 20ns |
| SmallVec collection for unary | ~30ns | 0ns (direct access) | 30ns |
| bytemuck cast_f32 per call | ~15ns | 0ns (typed arena) | 15ns |
| out_buf round-trip for passthrough | ~50ns | 0ns (pointer swap) | 50ns |
| out_buf.resize zeroes memory | ~30ns | 0ns (set_len) | 30ns |
| **Total per instruction** | **~205ns** | **~0ns** | **~205ns** |
| **150-op transformer layer** | **~30µs** | **~0µs** | **~30µs** |

### KvExecutor Deprecation
- [x] **dep.1**: `#[deprecated]` on `KvExecutor` struct (`eval/executor.rs`)
- [x] **dep.2**: `#[deprecated]` on mmap wrappers (`execute_plan`, `execute_plan_with_shape_hints`, `execute_plan_with_kv_state`, `execute_bytes`, `execute_bytes_with_ops`, `execute_bytes_with_progress`, `execute_file`)
- [x] **dep.3**: `#[allow(deprecated)]` on internal impl blocks and profile functions
- [x] **dep.4**: Deprecation roadmap documented in handoff spec (Section 8)
- [x] **dep.5**: Migrate CLI `run_cmd.rs` generation loop to tape path (Sprint 17)
- [ ] **dep.6**: Add intermediate capture to EnumTape (tape profiling) — deferred
- [x] **dep.7**: Migrate remaining KvExecutor-based tests to tape (Sprint 17)
- [x] **dep.8**: Remove KvExecutor (struct, impl, mmap wrappers, re-exports) (Sprint 17)

### Documentation (Sprint 16)
- [x] **D.1**: Transformer benchmark specification — [specs/docs/transformer-benchmark-spec.md](docs/transformer-benchmark-spec.md)
- [x] **D.2**: hologram-ai integration guide — [specs/docs/hologram-ai-integration.md](docs/hologram-ai-integration.md)
- [x] **D.3**: Feature matrix updated with backends, tape dispatch levels, Metal GPU thresholds — [specs/feature-matrix.md](../specs/feature-matrix.md)

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

---

## Sprint 17: Performance Hardening + KvExecutor Removal

**Plan**: [plans/014-graph-perf-kvexecutor-removal.md](plans/014-graph-perf-kvexecutor-removal.md)

### Phase 1: Graph Successor Index Optimization
- [x] **1.1**: Successor index in toposort Kahn's loop (O(N²) → O(V+E))
- [x] **1.2**: Successor index in build_parallel_levels (O(N²) → O(V+E))
- [x] **1.3**: Successor index in validate acyclicity check
- [x] **1.4**: Indexed rewire_successors for CSE pass

### Phase 2: Fusion Pass Optimization
- [x] **2.1**: Eliminate double toposort — reuse original order for CSE
- [x] **2.2**: Pre-built successor index in fusion pass (commit 6ad9e12)

### Phase 3: KvExecutor Removal
- [x] **3.1**: Migrate hologram-async to tape path
- [x] **3.2**: Migrate hologram-ffi to tape path
- [x] **3.3**: Migrate hologram-cli to tape path
- [x] **3.4**: Migrate e2e tests to tape path
- [x] **3.5**: Remove custom_ops tests (KvExecutor-dependent registry dispatch)
- [x] **3.6**: Migrate executor benchmarks to tape-only
- [x] **3.7**: Remove deprecated mmap convenience functions (execute_plan, execute_bytes, execute_file, etc.)
- [x] **3.8**: Clean up re-exports and #[allow(deprecated)] annotations
- [x] **3.9**: Migrate calculator example to tape path

### Phase 4: Dead Code Removal (Plan 015)
- [x] **4.1**: Remove shape_propagate.rs + shape_resolve.rs (1066 lines)
- [x] **4.2**: Remove dirty_bits.rs + profile.rs (385 lines)
- [x] **4.3**: Remove old Tape/Instruction/KernelFn + their tests
- [x] **4.4**: Inline `parse_shape_values` into float_dispatch/shape_ops.rs

### Phase 5: Tape-Compatible Custom Ops (Plan 015)
- [x] **5.1**: `TapeKernel::Custom` variant + dispatch in `dispatch_kernel` / `dispatch_kernel_par`
- [x] **5.2**: Wire `CustomOpRegistry` into tape_builder (`resolve_kernel` accepts registry)
- [x] **5.3**: `build_tape_from_plan_with_ops` entry point
- [x] **5.4**: Custom op E2E test (passthrough handler via tape path)

### Phase 6: Tape Hot Path Optimization (Plan 015)
- [x] **6.1**: `binary_broadcast` helper — eliminate modulo for same-size/scalar cases
- [x] **6.2**: Pre-size `consumer_counts` in `apply_reuse_flags`

### Benchmark Results (Phase 1+2)
- fusion::fuse(1000_nodes): **1.91 ms → 290 µs** (6.6x faster)
- fusion::fuse(100_nodes): **44 µs → 31 µs** (-30%)
- compile/100_nodes: **79 µs → 60 µs** (-24%)
- compile/50_nodes: **45 µs → 41 µs** (-9%)

## Sprint: ComputeBackend + ComputeMemory Rewrite (Plan 067)

**Plan**: [plans/067-compute-backend-rewrite.md](plans/067-compute-backend-rewrite.md)

Goal: single-device execution — all data lives on one device (Metal/WebGPU/CPU),
all computation happens on that device. No CPU↔GPU transfers during execution.
New `hologram-backend` crate with `ComputeMemory` + `ComputeBackend<M>` traits.

### Phase 1: Traits + CpuMemory (non-breaking)
- [ ] Create `hologram-backend` crate
- [ ] Define `ComputeMemory` trait (alloc, upload, download, reshape)
- [ ] Define `ComputeBackend<M>` trait (dispatch, load_ring_tables, flush)
- [ ] Implement `CpuMemory` + `CpuBackend` (wraps existing CPU dispatch)

### Phase 2: MetalMemory + device-native weight loading
- [ ] Implement `MetalMemory` (metal::Buffer allocation)
- [ ] Load weights directly into Metal buffers at archive load time
- [ ] Load UOR LUT tables onto Metal device

### Phase 3: Single-path executor
- [ ] New `execute<M, B>()` in hologram-exec consuming hologram-backend
- [ ] All ops dispatch through `backend.dispatch()` — no CPU fallback
- [ ] Single flush at end of execution

### Phase 4: Complete Metal kernel coverage
- [ ] Q4 dequant+GEMM kernel for Conv2dLut4/MatMulLut4
- [ ] Ring op kernels on Metal (Z/256Z LUT lookups)
- [ ] All TapeKernel variants covered

### Phase 5: WebGPU backend skeleton
- [ ] `WebGpuMemory` + `WebGpuBackend` (async-aware)
- [ ] WGSL shader source for core kernels
- [ ] WASM target compatibility
