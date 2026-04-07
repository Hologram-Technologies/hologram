# Plan 040: HC1 Gap Analysis — CPU-First Performance Roadmap

## Context

**Goal**: Identify architectural and optimization gaps between hologram's software
runtime and a representative HC1-class hardware inference accelerator, and chart a
CPU-first roadmap to close the gap where possible.

**HC1 reference accelerator** is a custom silicon inference accelerator (TSMC 6nm,
815mm^2, 53B transistors) class device claiming **17,000 tokens/sec/user** on
Llama 3.1 8B at 1k/1k sequence length, in a 2.5kW server. Benchmarks above H200,
B200, Groq, Sambanova, Cerebras.

**Hologram** is a software O(1) compute acceleration runtime achieving ~81 tok/s on
a synthetic transformer (hidden=2048, FFN=5632) via LUT-GEMM, tape-based execution,
and five fusion passes. Targets CPU and WASM.

**Branch**: `feat/hc1-gap-closure`

---

## Performance Gap Summary

| Metric | HC1 reference | Hologram (current) | Gap |
|--------|-----------|-------------------|-----|
| Throughput (tok/s, 8B model) | 17,000 | ~81 (synthetic, smaller model) | ~210x |
| Parallelism | Massive on-die (53B transistors) | Rayon work-stealing | Critical |
| Memory bandwidth | Custom on-die SRAM/HBM | System RAM + mmap | 10-100x |
| Compute density | Fixed-function datapath | General-purpose CPU ALUs | 50-100x |
| Datapath width | 512-bit+ custom buses | 128-512 bit SIMD | 4-16x |

---

## What HC1 Gets "For Free" — And What Hologram Can Do

### 1. Massive Parallel Compute (CRITICAL)

HC1 runs thousands of compute units simultaneously. Hologram parallelizes across
graph levels via rayon but has gaps:

**Already done** (Sprints 30-31):
- dispatch_kernel_par fixed for LUT-GEMM variants (Sprint 31, 1.1)
- N-parallel vecmat for M=1 decode (Sprint 31, 2.2)
- Lock-free LUT-GEMM parallelism via RwLock (Sprint 30, 3.1-3.2)

**Remaining gaps**:
- Per-thread Psumbook scratch (Sprint 30, 3.3 — not done)
- Batch execution across multiple sequences (amortize weight loads)
- Intra-attention parallelism (parallel across heads)

### 2. Memory Bandwidth Dominance

Custom silicon places compute next to memory. Hologram relies on system RAM + mmap.

**Already done** (Sprint 30):
- Multi-level prefetch, levels i+1 and i+2 (Sprint 30, 2.1)
- F16 activation compression for large buffers (Sprint 30, 5.1)

**Remaining gaps**:
- Workspace buffer reuse not wired into arena (Sprint 30, 5.2 — not done)
- Cache-tiled GEMM (block to fit L1/L2, minimize DRAM round-trips)
- NUMA-aware thread pinning for multi-socket servers

### 3. Fixed-Function Datapaths

HC1 has hardwired matmul/attention/norm units. Hologram approximates with LUT-GEMM
and enum dispatch.

**Already done** (Sprints 30-31):
- ARM NEON int8 LUT-GEMM kernel
- WASM SIMD128 micro-kernels (Sprint 31, 4.3)
- Shared B-panel packing (Sprint 31, 2.1)

**Remaining gaps**:
- x86 AVX2/AVX-512 GEMM micro-kernels (no optimized x86 path exists)
- Fused decode kernels could go deeper (multi-layer pipeline)

### 4. Batch-Level Throughput

HC1's 17k tok/s is likely batched (many sequences). Hologram optimizes for
single-sequence decode only.

**Remaining gap** (new work):
- Batch execution in tape engine (load weights once, apply to N sequences)
- Continuous batching for server workloads
- This is the single biggest throughput multiplier available on CPU

### 5. Dedicated Datapath Width

HC1 pushes data through wide custom buses. CPU is limited to SIMD register width.

**Already done**: ARM NEON (128-bit), WASM SIMD128 (128-bit)

**Remaining gaps**:
- x86 AVX2 (256-bit) — no optimized kernels
- x86 AVX-512 (512-bit) — completely untapped

---

## What Hologram Already Does Well (Preserve)

| Strength | Detail |
|----------|--------|
| O(1) per-op dispatch | Enum tape, no graph traversal at runtime |
| View fusion | Activation chains -> single 256-byte LUT |
| Epilogue fusion | MatMul+Bias+Activation in-register |
| Zero-copy loading | rkyv + mmap for weights/constants |
| Cross-platform | CPU + WASM (unique vs hardware accelerators) |
| Compile-first | All shapes/dtypes resolved ahead of time |
| LUT-GEMM | Quantized matmul via partial-sum booklets |

---

## Prioritized Roadmap (CPU-Only)

### Phase 1: Remaining Low-Hanging Fruit (1-2 weeks)

| # | Item | Source | Impact |
|---|------|--------|--------|
| 1.1 | Wire workspace buffer reuse into arena | Sprint 30, 5.2 | 20-40% peak memory |
| 1.2 | Per-thread Psumbook scratch | Sprint 30, 3.3 | Eliminates contention in parallel LUT-GEMM |
| 1.3 | Adaptive sparse_v threshold | Sprint 30, 4.2 | 5-20% at long contexts |

### Phase 2: x86 SIMD Parity (2-3 weeks)

| # | Item | Impact |
|---|------|--------|
| 2.1 | AVX2 LUT-GEMM micro-kernel (256-bit, int8) | 2x throughput vs scalar on x86 |
| 2.2 | AVX-512 LUT-GEMM micro-kernel (512-bit) | 4x on supported CPUs (Zen4, SPR) |
| 2.3 | AVX2/512 fused attention kernel | Match NEON quality on x86 |

### Phase 3: Batch Execution (1-2 months)

| # | Item | Impact |
|---|------|--------|
| 3.1 | Batch tape execution API (N sequences, shared weights) | 10-50x aggregate throughput |
| 3.2 | Batched LUT-GEMM (M>1 with shared weight pages) | Amortize weight loading |
| 3.3 | Batched KV cache management | Per-sequence KV with shared attention |

### Phase 4: Advanced CPU Techniques (2-4 months, aspirational)

| # | Item | Impact |
|---|------|--------|
| 4.1 | Cache-tiled GEMM (L1/L2 blocking) | Minimize DRAM round-trips |
| 4.2 | Speculative decoding | Draft model predicts, main model verifies in parallel |
| 4.3 | Continuous batching / paged attention | Server-grade multi-user on CPU |
| 4.4 | NUMA-aware execution | Pin threads + memory to socket |

---

## Realistic CPU-Only Targets

| Scenario | Current | Target (after Phase 1-3) |
|----------|---------|--------------------------|
| Single-sequence decode (8B) | ~81 tok/s | ~300-500 tok/s |
| Batched throughput (8B, N users) | N/A | ~1,000-3,000 tok/s aggregate |
| WASM browser (1-3B) | untested | ~50-150 tok/s |

Competitive with llama.cpp CPU mode while preserving hologram's LUT-GEMM, O(1)
dispatch, and cross-platform advantages.

**Key insight from HC1**: Parallelism and memory bandwidth matter more than raw
clock speed. The CPU roadmap maximizes both.

---

## Backlog (Future, Not In Scope)

- **Metal GPU backend** — wire up `BackendSelector::Metal` for macOS GPU offload
- **CUDA backend** — NVIDIA GPU support for datacenter workloads
- **WebGPU backend** — browser GPU acceleration
- GPU backend stubs: `hologram-exec/src/backend/mod.rs`

---

## Verification

- Run benchmarks: `cargo bench -p hologram-bench`
- Profile decode: `cargo run --features profile,cli -- run model.holo --prompt "..."`
- Compare before/after each phase
- Track: tok/s, peak memory, time-to-first-token, GEMM utilization %

---

## Key Files

- Parallel dispatch: `crates/hologram-exec/src/parallel/mod.rs`
- Tape execution: `crates/hologram-exec/src/tape.rs`
- GEMM kernels: `crates/hologram-exec/src/lut_gemm/`
- Buffer arena: `crates/hologram-exec/src/buffer/arena.rs`
- Fusion passes: `crates/hologram-graph/src/fusion/`
- SIMD: `crates/hologram-core/src/view/simd.rs`
- Benchmarks: `crates/hologram-bench/benches/`
