# Plan 033: UOR 0.1.0 Migration — Algebraic Performance Acceleration

## Context

The hologram runtime's LUT-GEMM pipeline operates at Q0 (8-bit, 256-entry tables) with float escape hatches for operations that exceed byte-domain precision. This ceiling limits throughput: quantization uses O(N×256) k-means, matmul kernels suffer cache-line thrashing, and unary/binary ops pay vtable dispatch overhead on every invocation.

Branch `feat/uor0.1.0-migration` implements the complete Q0→Q3 Cayley-Dickson algebraic chain, replacing ad-hoc precision workarounds with a mathematically grounded hierarchy (Real→Complex→Quaternion→Octonion) and delivering concrete performance wins across quantization, GEMM, and tape execution.

## Performance Wins

| Area | Improvement | Mechanism |
|------|------------|-----------|
| Quantization | ~256× faster | O(N) uniform floor-division replaces O(N×256) k-means |
| LUT-GEMM (symmetric weights) | ~2× fewer MACs | Orbit compression exploits dihedral symmetry |
| LUT-GEMM (cache) | No cache-line thrashing | Fiber-ordered Q8: 16-pass radix, 1 L1 line per pass |
| Unary/Binary ops | Zero dispatch overhead | 24 inline hot-path kernels bypass vtable |
| Memory latency | Hidden | Platform prefetch (x86 `_mm_prefetch`, aarch64 `PRFM`) |
| First-inference alloc | Eliminated | `output_byte_hint` + `prewarm_arena()` |
| Unary chain fusion (Q1) | Compile-time collapse | Consecutive Q1 ops → single 128KB LUT |
| Precision dispatch | No runtime branching | Carry-driven `CurvatureFlux` (<10ns/op) |
| MatMul | Fewer allocations | `dispatch_matmul_into()` direct to pre-allocated buffer |

## Performance Contracts (tested)

| Operation | Target | Per-op |
|-----------|--------|--------|
| Q0 activation lookup | 1M ops < 10ms | < 10ns |
| CurvatureFlux query | 10M queries < 50ms | < 5ns |
| Carry lift Q0→Q1 | 1M lifts < 10ms | Zero-copy |
| Q3 wrapping arithmetic | 1M ops < 10ms | Native u32 |
| Cayley-Dickson multiply | 1M ops < 50ms | < 50ns |
| Q8 GEMM (4×256×256) | 10 iters < 50ms | < 5ms each |
| Streaming accumulate+query | 10M ops < 100ms | < 10ns |

## Implementation Summary

### New modules (hologram-core)
- `q1/` — Q1 WordRing (16-bit, Z/65536Z)
- `q2/` — Q2 TripleRing (24-bit, Z/2^24Z)
- `q3/` — Q3 OctonionRing (32-bit, Z/2^32Z) with Cayley-Dickson arithmetic
- `carry/` — CurvatureFlux carry-driven precision lifting (DC_5 protocol)
- `ring/` — ByteRing + ring trait interface
- `quantum/` — Quantum level utilities + table feasibility hierarchy
- `view/` — ElementWiseView16 for Q1 LUT operations

### New modules (hologram-exec)
- `lut_gemm/orbit.rs` — Orbit compression (dihedral symmetry maps)
- `lut_gemm/quantize_q1.rs` — Q16 hierarchical quantization (256 pages × 256 sub-centroids)
- `lut_gemm/psumbook_q1.rs` — Hierarchical partial-sum accumulator
- `float_dispatch/gather_concat.rs` — Gather, concat, where, range, shape ops

### New compiler passes (hologram-compiler)
- `precision/pass.rs` — Static precision promotion from curvature analysis
- `qedl/pass.rs` — Byte↔float domain boundary insertion

### New graph fusion (hologram-graph)
- `fusion/q1_view_fusion.rs` — Q1 unary chain fusion + involution cancellation

### Tests (58 new, 8 files)
- Ring algebraic axioms, carry protocol, performance contracts
- Quantization correctness, orbit compression, Q3 octonion properties
- Streaming dynamic dispatch scenarios

## Merge Strategy

1. Merge `feat/ai-optimization` → `main` first (smaller, 7 commits)
2. Merge `origin/feat/uor0.1.0-migration` → `main`
3. Resolve `tape.rs` conflict (only overlapping file)
4. Verify: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`
