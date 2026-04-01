# Plan 037: GEMM / MatMul / Conv2D Performance Optimization

## Context

GEMM, MatMul, and Conv2D are the dominant compute kernels in hologram — they account for 80%+ of inference time in both LLM (MatMul-heavy) and vision (Conv2D-heavy) workloads. The existing implementation already has BLIS-style tiling, NEON/AVX2 SIMD micro-kernels, Winograd F(2,3), and epilogue fusion. This plan targets the remaining performance gaps: redundant work, missing SIMD coverage, underutilized parallelism, and cache inefficiency.

Must work on wasm (no Metal/CUDA), aarch64, and x86_64.

**Memory constraint:** Optimizations must not balloon peak memory. Each item below includes a memory budget analysis. The existing TILE_CAP (16MB im2col) and stack-allocated packed_b (8KB) are the reference points. New allocations must be bounded and documented.

---

## Optimizations (priority order)

### 1. Shared B-Panel Packing Across M-Tiles
**Problem:** In `matmul_k_outer`, each M-tile independently packs the same B panel. For M=128, K=4096, N=4096: the same 8KB B panel is packed 32 times per N-tile per K-block.

**Fix:** Restructure loop order to K-block (outer) -> N-tile -> pack B once -> fan out M-tiles. Pack into a single reusable buffer sized `KC * NR` (one N-tile at a time, not all N-tiles), shared read-only across M-tile threads. This keeps the packed buffer at **8KB** (same as current per-tile stack allocation) — we just pack it once and share it instead of packing per M-tile.

**Memory:** Net zero — replaces N redundant 8KB stack allocs with 1 shared 8KB alloc. The key insight is packing one N-tile's B panel at a time (not the full `KC * N`), then processing all M-tiles for that panel before moving to the next N-tile. This preserves the current 8KB footprint while eliminating redundancy.

**Files:** `crates/hologram-exec/src/float_dispatch/matmul.rs` — `matmul_k_outer` (lines 1016-1145), `process_m_tile` closure
**Impact:** ~15-25% GEMM speedup for large matrices (eliminates ~30x redundant packing)
**Effort:** Medium

### 2. Enable LUT-GEMM + Conv2D in Parallel Level Dispatch
**Problem:** WeightCache already uses `parking_lot::RwLock` and `needs_shared_state` already excludes only KvWrite/KvRead (tape.rs:4097-4101). However, `dispatch_kernel_par` (tape.rs:2783-2934) does NOT handle `MatMulLut4`, `MatMulLut8`, `MatMulLut16`, `InlineConv2d*`, or `InlineConv2dLut4` — they fall through to the `_ => Err("non-parallelizable op")` catch-all at line 2931. So levels containing these ops can never run in parallel even though the lock type allows it.

**Fix:**
- Add LUT-GEMM dispatch arms to `dispatch_kernel_par`: `MatMulLut4(cid)`, `MatMulLut8(cid)`, `MatMulLut16(cid)`, and their `*Activation` variants. Each calls `dispatch_lut_gemm_*` which acquires a read lock on WeightCache (already `parking_lot::RwLock`, supports concurrent reads).
- Add Conv2D dispatch arms: `InlineConv2d`, `InlineConv2dActivation`, `InlineConv2dBiasActivation`, `InlineConv2dLut4`. These are stateless — just need `tape_ctx` passed to the parallel dispatch function.
- Change `dispatch_kernel_par` signature to accept `&TapeContext` instead of just `&ConstantStore`, so it can access the weight cache and weights bytes.
- Change WeightCache getters from `&mut self` to `&self` using `get_or_insert`-style with the RwLock (currently `cache.write()` at tape.rs:2955 takes exclusive lock — change to try read first, then write only on miss).

**Memory:** Zero new allocations — uses existing RwLock and HashMap. The only change is concurrent read access instead of sequential.

**Files:** `crates/hologram-exec/src/tape.rs` — `dispatch_kernel_par` (add ~60 lines of match arms), `execute_inner` parallel path (pass `tape_ctx` to par dispatch); `crates/hologram-exec/src/kv/weight_cache.rs` — change `get_q4`/`get_q8`/`get_q16` to use read-then-write pattern
**Impact:** 20-40% end-to-end latency on multi-core for quantized LLM inference (50%+ of levels become parallelizable)
**Effort:** Medium

### 3. Winograd Batched GEMM Parallelism
**Problem:** The 16 Winograd-domain GEMMs (`for e in 0..16` at conv.rs:251) run sequentially. Each is `[oc, ic] x [ic, n_tiles]` — substantial for oc=512, ic=512.

**Fix:** Use `m_buf.par_chunks_mut(oc_per_group * n_tiles)` with rayon to process the 16 Winograd-domain GEMMs concurrently. Each chunk `e` reads from disjoint `u_all` and `v_buf` slices and writes to its own `m_buf` chunk. Gate with a size threshold (e.g., `oc_per_group * n_tiles >= 1024`) to avoid rayon overhead on small convolutions.

**Memory:** Zero additional — `m_buf` and `v_buf` are already allocated at their full 16-element size. Parallelism just processes existing slices concurrently.

**Files:** `crates/hologram-exec/src/float_dispatch/conv.rs` — `conv2d_winograd_f23` (lines 250-287)
**Impact:** ~3-4x speedup on total Winograd conv time (3x3 convolutions, dominant in UNet/VAE)
**Effort:** Low

### 4. N-Dimension Parallelism + Lower PAR_M_TILE_THRESHOLD
**Problem:** `PAR_M_TILE_THRESHOLD=8` means M<32 runs sequential. LLM decode (M=1) gets zero parallelism.

**Fix:**
- Lower `PAR_M_TILE_THRESHOLD` to 2 (M >= 8)
- Add N-parallel path for `vecmat_mul` (M=1): partition `n_tiles` across rayon threads, each writes non-overlapping output columns
- For small M (< threshold): add N-dimension parallel fallback that partitions N-tiles

**Memory:** Zero additional — just changes which dimension is partitioned across threads.

**Files:** `crates/hologram-exec/src/float_dispatch/matmul.rs` — `matmul_k_outer`, `vecmat_mul`, constants
**Impact:** 2-3x speedup for M=1 GEMM (LLM decode) on multi-core
**Effort:** Low-Medium

### 5. SIMD Depthwise Conv2D
**Problem:** `conv2d_depthwise` is fully scalar with per-element bounds checking. Branch-heavy padding checks prevent auto-vectorization.

**Fix:**
- Split spatial loop into interior (no padding check needed) and border regions
- Interior: branch-free inner loop enables auto-vectorization; optionally add explicit NEON/AVX2 for 4/8-wide output positions
- Border: keep current bounds-checked scalar path

**Memory:** Zero additional — same output buffer, just changes iteration pattern. The interior/border split is computed from existing dimensions, not stored.

**Files:** `crates/hologram-exec/src/float_dispatch/conv.rs` — `conv2d_depthwise` (lines 12-68)
**Impact:** 3-4x depthwise conv speedup (MobileNet, EfficientNet, VAE decoders)
**Effort:** Medium

### 6. A-Panel Packing
**Problem:** A is accessed with stride K — each load pulls a 64B cache line but uses only 4B. For K=4096, adjacent A rows are 16KB apart.

**Fix:** Add `pack_a_panel<MR>()` that copies A[i..i+MR, kc..kc+KC] into contiguous MR-strided buffer. Stack-allocated: `MR * KC = 4 * 256 = 1KB`. Update SIMD micro-kernels to read from packed A.

**Memory:** +1KB stack per thread (stack-allocated like existing `packed_b`). Negligible.

**Files:** `crates/hologram-exec/src/float_dispatch/matmul.rs` — add `pack_a_panel`, update micro-kernel signatures
**Impact:** 5-15% GEMM speedup for K >= 512
**Effort:** Medium (requires updating all SIMD variants)

### 7. wasm32 SIMD Micro-Kernels
**Problem:** wasm falls through to scalar code everywhere. SIMD128 (128-bit, 4-float) is widely supported.

**Fix:** Add `#[cfg(target_arch = "wasm32")]` paths using `std::arch::wasm32::*`:
- MR=4, NR=4 micro-kernel (128-bit vectors = 4 floats)
- Vecmat NR=4 kernel
- Depthwise conv 4-wide interior loop

**Memory:** Zero additional — same buffers, different instruction sequences.

**Files:** `crates/hologram-exec/src/float_dispatch/matmul.rs`, `crates/hologram-exec/src/float_dispatch/conv.rs`
**Impact:** ~3-4x wasm GEMM throughput
**Effort:** Medium

### 8. Fast Im2col (memcpy interior)
**Problem:** Im2col loop (conv.rs:482-509) does per-element division, modulo, and bounds check. For stride=1, consecutive output positions map to consecutive input.

**Fix:** For stride=1, no-dilation interior region: replace element-wise gather with `copy_from_slice` from `data[base_offset..]`. Keep scalar path for borders and non-unit-stride.

**Memory:** Zero additional — writes into the same `col` buffer. The `copy_from_slice` path replaces element-wise writes, same destination.

**Files:** `crates/hologram-exec/src/float_dispatch/conv.rs` — im2col loop in `conv2d_core`
**Impact:** 8-15% conv2d speedup for stride=1 (im2col is 10-20% of total time)
**Effort:** Medium

### 9. Eliminate Per-Call Allocations in Conv2D Hot Path
**Problem:** `conv2d_core` allocates 4 buffers per call: `col` (up to 16MB), `tile_out`, `col_t_buf`, `lut_out_buf`. For diffusion models doing 20-50 steps with identical Conv2d shapes, these are allocated and freed thousands of times. Additionally, `dispatch_conv2d_direct` copies bias into a new `Vec<f32>` every call (line 619-623) when it could just borrow.

**Fix:**
- Change bias handling to borrow `cast_f32(bias_bytes)?` directly instead of `.to_vec()` — zero-copy, bias is already f32-aligned in the constant archive.
- For repeated-inference scratch reuse: add an optional `&mut ConvScratch` parameter (or thread-local) that holds pre-sized `col`, `tile_out`, `col_t_buf`, `lut_out_buf`. On first call, allocate to size; on subsequent calls, reuse if capacity suffices. This avoids per-call `vec![]` + `drop`.

**Memory:** Net negative — eliminates the bias copy (saves `oc * 4` bytes per call) and reuses scratch buffers instead of alloc+dealloc. Peak memory unchanged (same buffers, just not freed between calls).

**Files:** `crates/hologram-exec/src/float_dispatch/conv.rs` — `dispatch_conv2d_direct` (bias borrow), `conv2d_core` (scratch reuse)
**Impact:** 5-10% latency reduction for small-spatial Conv2d (where alloc overhead is significant relative to compute). Larger impact on wasm where allocator is slower.
**Effort:** Low-Medium

### 10. Cache Winograd Weight Transform
**Problem:** `conv2d_winograd_f23` recomputes `u_all` (the Winograd weight transform) every call. For inference with static weights (the common case), this is pure redundant work — `u_all` depends only on `weight`, not on input data.

**Fix:** Cache `u_all` keyed by (weight pointer, weight length, oc, ic, group) in a small LRU or direct-map cache. On cache hit, skip the weight transform entirely. For SD UNet with ~22 Conv2d-3x3 layers at 20 steps, this eliminates 440 weight transforms.

**Memory:** The cached `u_all` is `group * 16 * oc_per_group * ic_per_group` floats. For a typical layer (oc=512, ic=512, group=1): 16 * 512 * 512 * 4 = 16MB. With ~22 layers cached: ~352MB — **too large to cache all layers simultaneously.** Instead, use a 1-entry cache (last used) or LRU(2). Most diffusion loops re-execute the same layer repeatedly before moving to the next, so a 1-entry cache has high hit rate. Memory: 16MB max for the largest layer.

**Files:** `crates/hologram-exec/src/float_dispatch/conv.rs` — `conv2d_winograd_f23` (add cache check before weight transform)
**Impact:** Eliminates weight transform phase (~15% of Winograd time for large layers). More impactful on smaller spatial dims where transform cost is proportionally higher.
**Effort:** Low

---

## Memory Impact Summary

| Optimization | Peak Memory Delta | Notes |
|---|---|---|
| 1. Shared B-pack | **0** | Same 8KB panel, packed once not N times |
| 2. LUT-GEMM/Conv2D parallel | **0** | Uses existing RwLock and HashMap |
| 3. Winograd parallel | **0** | Existing buffers processed concurrently |
| 4. N-dim parallel | **0** | Different partitioning of existing work |
| 5. Depthwise SIMD | **0** | Same output buffer, new iteration pattern |
| 6. A-panel pack | **+1KB/thread** | Stack-allocated, freed on return |
| 7. wasm SIMD | **0** | Different instructions, same buffers |
| 8. Fast im2col | **0** | Same `col` buffer, memcpy instead of element writes |
| 9. Conv2D alloc elimination | **net negative** | Eliminates bias copy; scratch reuse replaces alloc+dealloc |
| 10. Winograd weight cache | **+16MB max** | 1-entry cache for largest layer's transformed weights |

**Total worst case:** +16MB (Winograd weight cache) + 1KB/thread (A-panel). The Winograd cache is opt-in and bounded. All other optimizations are memory-neutral or negative.

---

## Implementation Phases

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 1** | 1 (shared B-pack) + 2 (parallel LUT-GEMM/Conv2D dispatch) | Highest impact, independent, foundational |
| **Phase 2** | 3 (Winograd parallel) + 4 (N-dim parallel) + 9 (alloc elimination) | Quick wins, large impact on specific paths |
| **Phase 3** | 5 (depthwise SIMD) + 8 (fast im2col) + 10 (Winograd weight cache) | Conv2D kernel improvements |
| **Phase 4** | 6 (A-pack) + 7 (wasm SIMD) | Polish: cache optimization + platform coverage |

---

## Verification

After each phase:
- `cargo test --workspace`
- `cargo clippy -- -D warnings`
- `cargo bench --bench matmul` (GEMM regression/improvement)
- `cargo bench --bench executor` (end-to-end tape execution)
- `cargo bench --bench lut_gemm` (quantized path validation)
- **Memory check:** Compare peak RSS before/after using `/usr/bin/time -l` (macOS) or `valgrind --tool=massif` (Linux) on a representative model. Regression threshold: <1% peak memory increase.
- Manual profiling with `instruments` (macOS) or `perf` (Linux) on representative model shapes
