# Benchmarks & throughput

Captured on the `cleanup-arbitrary-limits` branch (criterion, release, 100
samples/bench; CPU). Absolute times are machine-dependent (a shared CI VM); the
**ratios** — content-addressing reuse and matmul scaling efficiency — are the
load-bearing results. Re-run with `cargo bench -p hologram-bench` and the
release perf-floor V&V with `cargo test --release -p hologram-backend --test
performance --features cpu -- --nocapture`.

## Headline: UOR content-addressing is the win

| Path | Cold / recompute | Reused (κ-label memo hit) | Speedup |
|---|---|---|---|
| 8-op chain (d=128) | 584 µs | **150 ns** | **~3900×** |
| Production MLP (seq=64, d=256, 4 layers) | 3.00 ms | **3.02 µs** | **~1947×** |

The memo hit is O(1) in graph size — an identical input set returns the cached
output κ-labels without touching the graph. This is the "perf is content
addressing, not micro-opt" thesis, measured.

## Matmul throughput (f32, zero-copy blocked kernel)

| size | time | throughput |
|---|---|---|
| 64³ | 5.43 µs | ~96 GFLOP/s |
| 128³ | 45.4 µs | ~92 GFLOP/s |
| 256³ | 417 µs | ~80 GFLOP/s |
| 512³ | 3.59 ms | ~75 GFLOP/s |

Efficiency is retained across scale (PV-1 floor: ≥60% of peak from 128³→512³),
demonstrating the cache-oblivious recursion leverages the cache hierarchy
uniformly — no breakdown at size. `matmul_w8` (byte-domain reference) 64³ =
19.2 µs.

## Production MLP stack (cold, all-novel; PV-4)

| width | latency/infer | throughput |
|---|---|---|
| d=128 (4 layers) | 2.11 ms | 31.9 GFLOP/s |
| d=256 (4 layers) | 5.80 ms | 46.3 GFLOP/s |

The matmul→add→activation epilogue fuses one op per layer (`MatMulAddActivation`,
FU-5), eliding the product, post-add sum, and activation intermediates.

## Runtime overhead

| stage | time |
|---|---|
| compile (decode_step) | 14.0 µs |
| session load (decode + fuse passes) | 4.8 µs |
| per-execute dispatch | 191 ns |

Content addressing every node adds no measurable cost to the compute path; the
three fusion passes (matmul-epilogue, dequant→matmul, expand→binary) run once at
load and are O(calls), no-ops when nothing matches.

## Fusion micro-benchmarks (fused vs unfused, head-to-head)

| pattern | unfused | fused | note |
|---|---|---|---|
| Expand→Mul (512²) | 1.84 ms | ~1.8–2.0 ms | `BroadcastBinary`: elides the 1 MB broadcast intermediate; wall-clock neutral (within VM noise) for this memory-bound op, memory footprint strictly lower |
| dequantize→matmul (256³) | 557 µs | 597 µs | `MatMulDequant`: the matmul dominates; the win is **memory** — the dense f32 weight is never materialized as a pool slot |

Both fused kernels are verified **zero heap allocation per call** after warm-up
(`tests/zero_overhead.rs`) and bit-equal to the unfused path (FU-6/FU-7). Their
value is memory elision and scheduler simplification, not raw FLOP/s at these
sizes.
