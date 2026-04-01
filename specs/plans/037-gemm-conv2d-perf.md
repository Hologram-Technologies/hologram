# Plan 037: GEMM / MatMul / Conv2D Performance Optimization

## Context

GEMM, MatMul, and Conv2D are the dominant compute kernels in hologram — they
account for 80%+ of inference time in both LLM (MatMul-heavy) and vision
(Conv2D-heavy) workloads. The existing implementation has BLIS-style tiling,
NEON/AVX2 SIMD micro-kernels, Winograd F(2,3), and epilogue fusion. This plan
targets the remaining performance gaps.

Must work on wasm (no Metal/CUDA), aarch64, and x86_64.

**Memory constraint:** No peak memory increase beyond +16MB (opt-in Winograd
cache). All kernel-level changes are memory-neutral.

### What's Already Done (Sprint 30 + prior work)
- parking_lot::RwLock on WeightCache (commit ad7df78)
- needs_shared_state only blocks KvWrite/KvRead (LUT-GEMM removed from guard)
- Conv2d tile buffer pre-allocation (commit a406ebb) — no per-tile alloc
- Conv2d epilogue fusion: Conv2d+Act, Conv2d+Bias+Act (commit 13ab183)
- Rayon tile parallelism tried + reverted (commit 3ec60de) — oversubscribed threads with inner GEMM rayon
- LUT-GEMM Q4 Conv2d for WASM (commit c00d25b)
- MmapBuffer free-list recycling in arena (uncommitted, in progress)
- Workspace slot assignment via greedy interval coloring (uncommitted, in progress)

### Critical Bug Found During Review
**`dispatch_kernel_par` doesn't handle LUT-GEMM or Conv2D ops** (tape.rs:2931
catch-all returns error). But `needs_shared_state` no longer blocks them
(ad7df78 removed the guard). If a level has 4+ instructions including LUT-GEMM,
parallel dispatch will hit `UnsupportedOp` error. The comment at line 4230
("parallel levels never contain LUT-GEMM") is now stale. **This must be fixed
first** (item 1 below).

---

## Optimizations (priority order)

### 1. Complete LUT-GEMM + Conv2D Parallel Dispatch [BUG FIX]
**Problem:** `dispatch_kernel_par` (tape.rs:2783-2934) has no match arms for
`MatMulLut4`, `MatMulLut8`, `MatMulLut16`, their `*Activation` variants,
`InlineConv2d*`, or `InlineConv2dLut4`. These fall through to the error at
line 2931. Since commit ad7df78 removed these from the `needs_shared_state`
guard, any parallel level containing them will crash.

**Fix:**
- Add LUT-GEMM arms to `dispatch_kernel_par` that call existing
  `dispatch_lut_gemm_*` functions. Requires passing `&TapeContext` instead of
  just `&ConstantStore` to `dispatch_kernel_par` (and through the parallel
  closure at line 4250).
- Add Conv2D arms: `InlineConv2d`, `InlineConv2dActivation`,
  `InlineConv2dBiasActivation`, `InlineConv2dLut4`.
- Update the stale comment at line 4230.
- WeightCache getters use `&mut self` but are behind `parking_lot::RwLock`.
  Current call site uses `.write()` (exclusive lock). Change to: try `.read()`
  first (cache hit), fall back to `.write()` only on miss. This allows
  concurrent reads after warmup.

**Memory:** Zero — uses existing RwLock and HashMap.
**Files:** `tape.rs` (dispatch_kernel_par + execute_parallel), `weight_cache.rs`
**Impact:** Unlocks 20-40% multi-core latency improvement for quantized inference
**Effort:** Medium
**Priority:** CRITICAL — must be done first (correctness fix)

### 2. Shared B-Panel Packing Across M-Tiles
**Problem:** Each M-tile independently packs the same B panel via stack-allocated
`packed_b` in `process_m_tile`. For M=128, K=4096, N=4096: the same 8KB panel
is packed 32 times per (N-tile, K-block).

**Fix:** Restructure loop: K-block → N-tile → pack B once → fan out all M-tiles
over that panel. Pack into a single buffer (`KC * NR` = 8KB), shared read-only
across rayon M-tile threads.

**Memory:** Net zero — same 8KB, packed once not N times.
**Files:** `matmul.rs` — `matmul_k_outer` (lines 1016-1145)
**Impact:** ~15-25% GEMM speedup for large matrices
**Effort:** Medium

### 3. Winograd Batched GEMM Parallelism
**Problem:** 16 Winograd-domain GEMMs (`for e in 0..16`, conv.rs:251) run
sequentially. Each is `[oc, ic] x [ic, n_tiles]`.

**Fix:** `m_buf.par_chunks_mut(oc_per_group * n_tiles)` to process the 16 GEMMs
concurrently. Each chunk reads disjoint `u_all`/`v_buf` slices. Gate with
threshold (`oc_per_group * n_tiles >= 1024`) to avoid rayon overhead on small
convolutions.

**Note:** Rayon *tile-level* parallelism was tried and reverted (commit 3ec60de)
because it oversubscribed threads with the inner GEMM's own rayon. This is
different — the 16 element-wise GEMMs are independent and each uses BLAS
(on macOS) or `matmul_k_outer`. For BLAS, the parallelism is internal and
non-conflicting. For `matmul_k_outer`, we should disable its inner rayon when
called from the Winograd parallel path (pass a flag or check nesting depth).

**Memory:** Zero — existing `m_buf`/`v_buf` processed concurrently.
**Files:** `conv.rs` — `conv2d_winograd_f23` (lines 250-287)
**Impact:** ~3-4x Winograd conv speedup on multi-core
**Effort:** Low-Medium

### 4. N-Dimension Parallelism + Lower PAR_M_TILE_THRESHOLD
**Problem:** `PAR_M_TILE_THRESHOLD=8` means M<32 runs sequential. LLM decode
(M=1) gets zero thread parallelism in the GEMM kernel itself.

**Fix:**
- Lower `PAR_M_TILE_THRESHOLD` to 2 (M >= 8)
- Add N-parallel path for `vecmat_mul` (M=1): partition `n_tiles` across rayon
  threads. Each writes non-overlapping output columns.

**Memory:** Zero.
**Files:** `matmul.rs` — `matmul_k_outer`, `vecmat_mul`
**Impact:** 2-3x for M=1 GEMM (LLM decode) on multi-core
**Effort:** Low-Medium

### 5. SIMD Depthwise Conv2D
**Problem:** `conv2d_depthwise` (conv.rs:12-69) is fully scalar with per-element
bounds checking. No auto-vectorization possible due to branch-heavy padding.

**Fix:** Split into interior (no bounds check) and border regions. Interior loop
is branch-free → auto-vectorizes. Optionally add explicit NEON/AVX2 for 4/8-wide
output positions.

**Memory:** Zero.
**Files:** `conv.rs` — `conv2d_depthwise`
**Impact:** 3-4x depthwise conv speedup
**Effort:** Medium

### 6. Eliminate Per-Call Bias Copy in Conv2D
**Problem:** `dispatch_conv2d_direct` (conv.rs:620) copies bias to `Vec<f32>`
every call via `.to_vec()`. Bias is already f32-aligned in the constant archive.

**Fix:** Borrow `cast_f32(bias_bytes)?` directly instead of `.to_vec()`.

**Memory:** Net negative — eliminates `oc * 4` bytes per call.
**Files:** `conv.rs` — `dispatch_conv2d_direct` (line 620)
**Impact:** Small latency reduction, removes unnecessary allocation
**Effort:** Trivial

### 7. Fast Im2col (memcpy for stride=1 interior)
**Problem:** Im2col (conv.rs:482-509) does per-element division/modulo/bounds
check. For stride=1, consecutive output positions map to consecutive input.

**Fix:** For stride=1, no-dilation interior: replace element-wise gather with
`copy_from_slice`. Keep scalar for borders and non-unit-stride.

**Memory:** Zero — same `col` buffer.
**Files:** `conv.rs` — im2col loop in `conv2d_core`
**Impact:** 8-15% conv2d speedup for stride=1
**Effort:** Medium

### 8. Cache Winograd Weight Transform
**Problem:** `conv2d_winograd_f23` recomputes `u_all` every call. For static
weights (inference), this is redundant.

**Fix:** 1-entry cache keyed by (weight pointer, len, oc, ic, group). On hit,
skip transform. Memory: 16MB max for the largest layer.

**Memory:** +16MB max (opt-in, 1-entry cache).
**Files:** `conv.rs` — `conv2d_winograd_f23`
**Impact:** ~15% of Winograd time eliminated for repeated inference
**Effort:** Low

### 9. A-Panel Packing
**Problem:** A accessed with stride K (16KB between adjacent rows for K=4096).

**Fix:** `pack_a_panel<MR>()` copies A[i..i+MR, kc..kc+KC] into 1KB stack
buffer. Update SIMD micro-kernels.

**Memory:** +1KB stack per thread.
**Files:** `matmul.rs` — add `pack_a_panel`, update micro-kernels
**Impact:** 5-15% GEMM speedup for K >= 512
**Effort:** Medium

### 10. wasm32 SIMD128 Micro-Kernels
**Problem:** wasm falls through to scalar everywhere. SIMD128 widely supported.

**Fix:** `#[cfg(target_arch = "wasm32")]` paths: MR=4, NR=4 micro-kernel,
vecmat NR=4, depthwise 4-wide interior.

**Memory:** Zero.
**Files:** `matmul.rs`, `conv.rs`
**Impact:** ~3-4x wasm GEMM throughput
**Effort:** Medium

---

## Memory Impact Summary

| Optimization | Peak Memory Delta | Notes |
|---|---|---|
| 1. LUT-GEMM/Conv2D parallel | **0** | Existing RwLock + HashMap |
| 2. Shared B-pack | **0** | Same 8KB panel, packed once |
| 3. Winograd parallel | **0** | Existing buffers concurrently |
| 4. N-dim parallel | **0** | Different partitioning |
| 5. Depthwise SIMD | **0** | Same buffer, new iteration |
| 6. Bias borrow | **negative** | Eliminates `.to_vec()` |
| 7. Fast im2col | **0** | Same `col` buffer |
| 8. Winograd weight cache | **+16MB max** | 1-entry opt-in cache |
| 9. A-panel pack | **+1KB/thread** | Stack, freed on return |
| 10. wasm SIMD | **0** | Different instructions |

**Total worst case:** +16MB (Winograd cache, opt-in) + 1KB/thread (A-panel).

---

## Implementation Phases

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 1** | 1 (parallel dispatch fix) + 6 (bias borrow) | Bug fix + trivial win |
| **Phase 2** | 2 (shared B-pack) + 4 (N-dim parallel) | Highest GEMM impact |
| **Phase 3** | 3 (Winograd parallel) + 5 (depthwise SIMD) + 7 (fast im2col) | Conv2D improvements |
| **Phase 4** | 8 (Winograd cache) + 9 (A-pack) + 10 (wasm SIMD) | Polish |

---

## Verification

After each phase:
- `cargo test --workspace`
- `cargo clippy -- -D warnings`
- `cargo bench --bench matmul` (GEMM regression/improvement)
- `cargo bench --bench executor` (end-to-end tape execution)
- `cargo bench --bench lut_gemm` (quantized path validation)
- **Memory check:** `/usr/bin/time -l` (macOS) or `valgrind --tool=massif`.
  Regression threshold: <1% peak memory increase.
