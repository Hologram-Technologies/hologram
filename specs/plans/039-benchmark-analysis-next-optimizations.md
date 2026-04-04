# Plan 039: Benchmark Analysis & Next Optimization Opportunities

## Context

The hologram runtime has undergone aggressive optimization on the `feat/deep-decode-fusion`
branch, progressing from **2.4 tok/s → ~81 tok/s** (synthetic transformer decode step,
hidden=2048, FFN=5632, M=1) through zero-copy input gathering, NEON int8 Q4 kernels,
hybrid BLAS+LUT dispatch, and fused decode kernels.

This plan documents the current benchmark state and identifies remaining performance
opportunities, prioritized by impact and effort.

---

## Current Benchmark Snapshot

### Transformer Decode Step (headline metric)
| Metric | Value |
|--------|-------|
| Mean | 12.345 ms |
| Median | 12.189 ms |
| Stddev | 0.565 ms |
| Throughput | **~81 tok/s** |

### LUT-GEMM Micro-Benchmarks
| Size | Q4 | Q8 |
|------|----|----|
| 1×16×16 | 0.3 µs | 0.5 µs |
| 4×64×64 | 1 µs | 9 µs |
| 4×256×256 | 10 µs | 83 µs |

### Other Benchmarks (sorted by time)
| Benchmark | Mean |
|-----------|------|
| quantize_q8(64×64) | 10.2 ms |
| quantize_q4(64×64) | 0.2 ms |
| fusion(1000 nodes) | 0.285 ms |
| mmap_load_execute(256KB) | 0.146 ms |
| tape relu(64KB) | 6 µs |

### Time Distribution (from HOLOGRAM_PROFILE=1)
- **MatMulLut4/8: 75-85%** of wall time
- Attention: 5-10%
- Normalization: 5-10%
- Activations: ~5%

---

## What's Already Been Done

1. Zero-copy input gathering — 16.8× speedup (commit 0dafd70)
2. NEON int8 Q4 kernel with `vqtbl1q_s8` table lookup — 3× vs f32 centroids
3. Hybrid BLAS+LUT dispatch — Accelerate for f32, LUT-GEMM for quantized
4. Compile-time centroid tables — eliminated runtime quantization (commit 4d0468e)
5. Deep decode fusion (Plan 054 Wave 1) — NormProjectionGemv, SwiGluProjectionGemv
6. M=1 fast path — normalize into caller Vec, skip arena allocation
7. Epilogue fusion — MatMul+Activation fused kernels (Plan 030)
8. Column-parallel LUT-GEMM — rayon over N > 64 columns
9. Buffer arena with mmap/vec hybrid, F16 compression infrastructure
10. KV cache quantization — asymmetric K=f32/V=Q4 (Plan 034)

---

## Remaining Optimization Opportunities

### Tier 1: Critical / High Impact

#### 1. Fix `dispatch_kernel_par` bug — CORRECTNESS FIX
**Ref:** Plan 037, Item 1

`dispatch_kernel_par` (tape.rs) has no match arms for `MatMulLut4`, `MatMulLut8`,
`MatMulLut16`, their `*Activation` variants, `InlineConv2d*`, or `InlineConv2dLut4`.
Since commit ad7df78 removed these from `needs_shared_state`, parallel levels
containing them will crash with `UnsupportedOp`.

- Add LUT-GEMM + Conv2D arms to `dispatch_kernel_par`
- Change WeightCache access: try `.read()` first, fall back to `.write()` on miss
- Update stale comment at tape.rs line ~4230
- **Impact**: Unlocks 20-40% multi-core latency for quantized inference
- **Effort**: Medium
- **Files**: `tape.rs`, `weight_cache.rs`

#### 2. N-Dimension Parallelism for M=1 GEMM
**Ref:** Plan 037, Item 4

LLM decode is always M=1 (vecmat). Currently `PAR_M_TILE_THRESHOLD=8` means
M<32 gets zero thread parallelism in the GEMM kernel.

- Add N-parallel path for `vecmat_mul`: partition `n_tiles` across rayon threads
- Each thread writes non-overlapping output columns
- Lower `PAR_M_TILE_THRESHOLD` to 2
- **Impact**: 2-3× decode GEMM on multi-core
- **Effort**: Low-Medium
- **Files**: `matmul.rs` — `vecmat_mul`, `matmul_k_outer`

#### 3. Multi-Level Weight Prefetch
**Ref:** Plan 036, Phase 2

Current prefetch issues `MADV_WILLNEED` for next level's weights. For large models
(7B+ params), the OS needs more lead time to page in 4-8MB of weights.

- Issue `MADV_WILLNEED` for levels i+1 AND i+2
- Issue `MADV_DONTNEED` for level i-2 (release pages back to OS sooner)
- **Impact**: 5-15% latency on large models
- **Effort**: Low (~20 lines)
- **Files**: `tape.rs` (execute_inner, ~lines 3569-3580)

### Tier 2: Significant Gains

#### 4. AddRmsNorm + Activation Fusion
**Ref:** Plan 036, §1.1

`AddRmsNorm` (residual + normalize) has no fused activation variant, unlike
`RmsNorm`/`LayerNorm`/`GroupNorm`. This pattern appears in every LLM transformer block.

- Add `FusedAddRmsNormActivation` GraphOp + `InlineAddRmsNormActivation` TapeKernel
- Extend `try_fuse_norm_activation()` to handle `FloatOp::AddRmsNorm`
- **Impact**: 3-5% latency
- **Effort**: Low
- **Files**: `graph/mod.rs`, `float_fusion.rs`, `tape.rs`, `tape_builder.rs`

#### 5. Attention + Residual Add Fusion
**Ref:** Plan 036, §1.2

Most common 2-node pattern in transformers. Fusing eliminates materializing
the full attention output buffer.

- Add `FusedAttentionResidualAdd` GraphOp
- Fusion pattern: Attention with single successor Add, where Add's other input
  is not a descendant of Attention
- **Impact**: 5-8% latency
- **Effort**: Medium
- **Files**: `graph/mod.rs`, `float_fusion.rs`, `tape.rs`, `attention.rs`

#### 6. Shared B-Panel Packing Across M-Tiles
**Ref:** Plan 037, Item 2

Each M-tile independently packs the same B panel. For M=128, K=4096, N=4096:
the same 8KB panel is packed 32 times per (N-tile, K-block).

- Restructure loop: K-block → N-tile → pack B once → fan out M-tiles
- **Impact**: 15-25% GEMM speedup for large matrices
- **Effort**: Medium
- **Files**: `matmul.rs` — `matmul_k_outer`

#### 7. SwiGLU Fusion from Separate Ops
**Ref:** Plan 036, §4.1

`FusedSwiGLU` exists as a primitive but the fusion pass doesn't recognize the
`Split → Silu → Mul` pattern. Only works if the model importer explicitly creates it.

- Add `try_fuse_swiglu()` to detect gate/up split + Silu(gate) * up
- **Impact**: 3-5% on LLaMA/Mistral-family models
- **Effort**: Medium
- **Files**: `float_fusion.rs`

### Tier 3: Memory & Platform

#### 8. Wire Workspace Buffer Reuse
**Ref:** Plan 038

Compiler's `plan_workspace()` computes optimal buffer slot assignments via greedy
interval coloring. Not wired to executor arena.

- Map `WorkspaceLayout` assignments to `BufferArena` pre-allocation at tape start
- **Impact**: 20-40% peak activation memory reduction
- **Effort**: Low
- **Files**: `compiler/workspace/mod.rs`, `tape.rs`

#### 9. Wire F16 Activation Compression
**Ref:** Plan 036, Phase 5

`ArenaBuffer::F16Compressed` exists with `compress_f16()` / `expand_f32()` but
is not wired into the execution loop.

- Wire `compress_f16()` into swap_insert for buffers with liveness gap > N instructions
- Auto-expand on read via `get_f32_or_expand()`
- **Impact**: ~50% activation memory for large buffers
- **Effort**: Medium
- **Files**: `buffer/arena.rs`, `tape.rs`

#### 10. Adaptive sparse_v Threshold
**Ref:** Plan 036, §4.2

Currently hardcoded at 1e-6. At long contexts (32K+), more positions have
near-zero attention weight.

- Make threshold configurable per `KvCacheConfig` or per Attention op
- Add diagnostic mode counting skipped positions per layer
- **Impact**: 5-20% decode speedup at long contexts
- **Effort**: Low
- **Files**: `attention.rs`, `float_op.rs`

#### 11. wasm32 SIMD128 Micro-Kernels
**Ref:** Plan 037, Item 10

Wasm falls through to scalar everywhere. SIMD128 is widely supported in browsers.

- Add `#[cfg(target_arch = "wasm32")]` paths: MR=4, NR=4 micro-kernel
- Add vecmat NR=4, depthwise 4-wide interior
- **Impact**: ~3-4× wasm GEMM throughput
- **Effort**: Medium
- **Files**: `matmul.rs`, `conv.rs`

---

## Estimated Cumulative Impact

| Category | Items | Expected Impact |
|----------|-------|-----------------|
| Multi-core unlock | #1 + #2 | 2-3× decode GEMM on multi-core |
| Fusion gaps | #4 + #5 + #7 | 11-18% latency reduction |
| Memory bandwidth | #3 + #6 | 20-40% GEMM + 5-15% prefetch |
| Memory footprint | #8 + #9 | 20-50% peak memory reduction |
| Long context | #10 | 5-20% decode at 32K+ |
| Wasm | #11 | 3-4× wasm GEMM |

---

## Assessment

The single-threaded hot path (NEON int8 Q4 kernel) is well-optimized. The runtime
is not leaving huge single-thread performance on the table — 81 tok/s on the synthetic
decode step is strong for a pure-CPU, no-Metal, wasm-compatible runtime.

The biggest remaining wins are:
1. **Multi-core parallelism** — the `dispatch_kernel_par` bug blocks this entirely
   for quantized ops. Fixing it + adding N-parallelism is the single largest opportunity.
2. **Fusion gaps** — several common transformer patterns remain unfused, each
   individually small but cumulatively significant.
3. **Memory bandwidth** — prefetch distance and shared B-panel packing reduce
   memory stalls on large models.

---

## Implementation Phases

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 1** | #1 (parallel dispatch fix) | Correctness fix, prerequisite for #2 |
| **Phase 2** | #2 (N-parallel M=1) + #3 (prefetch) | Highest decode throughput impact |
| **Phase 3** | #4 + #5 + #7 (fusion gaps) | Latency wins, independent of each other |
| **Phase 4** | #6 (shared B-pack) | GEMM throughput for prefill |
| **Phase 5** | #8 + #9 (memory) | Capacity, not latency |
| **Phase 6** | #10 + #11 (long context + wasm) | Specialized improvements |

## Verification

After each phase:
```bash
cargo test --workspace
cargo clippy -- -D warnings
cargo bench --bench matmul          # GEMM regression/improvement
cargo bench --bench executor        # End-to-end tape execution
cargo bench --bench lut_gemm        # Quantized path
cargo bench --bench epilogue_fusion # Fusion correctness
HOLOGRAM_PROFILE=1 cargo test --test tinyllama  # Profile real model
```
