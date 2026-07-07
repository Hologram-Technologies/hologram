# Benchmarks & throughput

Captured at v0.5.0 (criterion, release, 100
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

## Refinement prototype notes

The compiled refinement strategy adds no backend dispatch path. A refinement
pass is a normal `InferenceSession::execute_addressed` call; pass-to-pass state
flows by κ-label. Prototype overhead is therefore:

| component | bound |
|---|---|
| pass loop | O(`max_passes + repair_passes`) over plan constants |
| per pass | normal addressed execution cost |
| `StableLabels` validator | O(number of state ports) |
| `StableBytes` validator | O(logical state bytes), zero-copy slice comparison |
| reporting | fixed counters plus final labels |

`RefinementReport` records total passes, repair passes, dispatched kernels,
resident-reuse skips, final labels, and resident-memory counters. No dedicated
Criterion benchmark is added for the prototype; once the archive/API shape is
stable, useful benchmark cases are one-pass identity label convergence,
two-pass byte convergence, repair retry, and graph-memo refinement replay.

## Fusion micro-benchmarks (fused vs unfused, head-to-head)

| pattern | unfused | fused | note |
|---|---|---|---|
| Expand→Mul (512²) | 1.84 ms | ~1.8–2.0 ms | `BroadcastBinary`: elides the 1 MB broadcast intermediate; wall-clock neutral (within VM noise) for this memory-bound op, memory footprint strictly lower |
| dequantize→matmul (256³) | 557 µs | 597 µs | `MatMulDequant`: the matmul dominates; the win is **memory** — the dense f32 weight is never materialized as a pool slot |

Both fused kernels are verified **zero heap allocation per call** after warm-up
(`tests/zero_overhead.rs`) and bit-equal to the unfused path (FU-6/FU-7). Their
value is memory elision and scheduler simplification, not raw FLOP/s at these
sizes.

## PM_7 tiered execution + LUT activations (`pm7-unified-memory` branch)

### LUT-accelerated low-precision activations — the genuine win

A transcendental activation over a finite quantum level is fully materialized
as a content-addressed table (Q0 = 256-entry, Q1 = 65536-entry), so dispatch is
one load instead of `widen → exp/tanh → narrow`. Bit-identical to compute.

| activation (1M elements) | computed | **LUT** | speedup |
|---|---|---|---|
| bf16 GELU | 20.2 ms | **712 µs** | **~28×** (≈1.4 G elem/s vs 50 M) |

Byte (Q0) Sigmoid/Tanh/Gelu/Silu/Exp/Erf likewise dispatch via a 256-byte table.
The table is built once (`OnceLock`); the loop scales to any element count.

### Densification keyed on the realized quantum level (quantized inference)

The same win, generalized off the 16-bit storage domain: a `Dequantize →
activation` chain stores f32 (no table — f32 is 2³²) but its *realized* domain is
the quantized source's (256 for i8, 16 for i4). `activation((q − zp)·scale)`
densifies into a ≤256-entry table indexed by the quantized byte
(`KernelCall::DequantActivation`), bit-identical to the unfused pair.

| activation (1M i8 elements) | unfused (dequant + scalar) | **densified table** | speedup |
|---|---|---|---|
| i8 → GELU | 18.9 ms | **695 µs** | **~27×** (≈1.44 G elem/s vs 53 M) |

This removes the scalar transcendental path for the f32 quantized-inference case;
the table tracks the quantum domain (≤256), not the f32 storage domain, so it
scales to any element count. Per-tensor; fired by a runtime fusion pass.

### No regression vs main (dropped the branch's slower fusion engine)

| workload | this branch | original PR branch |
|---|---|---|
| MLP cold d256 (4 layers) | **2.93 ms** (≈47 GFLOP/s) | 7.72 ms (2.6× slower) |
| MLP cold d128 | 1.16 ms | 2.58 ms |
| f32 matmul 256³ / 512³ | 422 µs / 3.50 ms (~79 / ~77 GFLOP/s) | — |

Content-addressing reuse intact: memo hit 150 ns vs 579 µs recompute (~3860×);
MLP served 3.97 µs vs 2.93 ms cold (~740×).

### PM_7 tiering — zero execution overhead

Tiers are classified at load (pure function of the quantum level); `execute()`
never consults them. Per-execute dispatch is unchanged:

| | time/execute |
|---|---|
| tiered 2-op | 155 ns |
| tiered 6-op chain | 157 ns |
| 6-op cached re-execute | 169 ns |

## Decode int8 GEMV (output-major W8A8, wasm SIMD128 lane)

Single-token decode (m = 1) at representative projection shapes sampled
across model scales — **samples only**: the kernel and the compile-time
fusion are shape-generic. Reported as **GB/s of int8 weight bytes streamed**
(the numerator of the downstream bandwidth-ratio witness). wasmtime +
`-Ctarget-feature=+simd128`, release; `omajor_w8a8` is the decode kernel
(output-major weight, per-token W8A8, exact integer accumulation),
`kn_w8a32` the prior fused path at the same shapes.

| shape (1×k×n) | omajor W8A8 | prior [k,n] W8A32 | ratio |
|---|---|---|---|
| 1×896×896 | 46 µs, 17.4 GB/s | 139 µs, 5.8 GB/s | ~3.0× |
| 1×896×4864 | 263 µs, 16.6 GB/s | 786 µs, 5.5 GB/s | ~3.0× |
| 1×4864×896 | 227 µs, 19.2 GB/s | 745 µs, 5.9 GB/s | ~3.3× |
| 1×1536×8960 | 739 µs, 18.6 GB/s | 2811 µs, 4.9 GB/s | ~3.8× |
| 1×3584×18944 | 5917 µs, 11.5 GB/s | 17391 µs, 3.9 GB/s | ~2.9× |

The largest shape drops toward DRAM bandwidth as the weight leaves cache —
the kernel is entering the bandwidth-bound regime. Output is bit-identical
across scalar / NEON / wasm (exact integer accumulation). These lanes are
iteration signals; acceptance is witnessed downstream by hologram-ai's
performance contract, which exercises the deployed browser build. Re-run:
`cargo bench -p hologram-bench --bench decode_gemv` (native) and
`RUSTFLAGS="-Ctarget-feature=+simd128" cargo run --release --example
wasm_matmul_timing --target wasm32-wasip1 -p hologram-backend --features
std,cpu` under wasmtime.

## Regression gate (CI)

PRs to `main` are gated by [`.github/workflows/perf-gate.yml`](.github/workflows/perf-gate.yml).
The job **interleaves** the **PR merge result** and the **target branch tip** on
the same runner — each round benchmarks both seconds apart, so the runner's
throughput drift cancels rather than reading as a regression — then reduces each
side to its per-benchmark minimum median across rounds and runs
[`scripts/compare-benchmarks.py`](scripts/compare-benchmarks.py), which **fails
the job** if any benchmark's median regresses past the gate.

A regression must clear two bars, so CI noise doesn't block honest PRs:

1. **Relative** — `pr_median > base_median × (1 + threshold)` (default 10%).
2. **Noise** — the slowdown also exceeds `noise-sigmas × √(base_std² + pr_std²)`
   (default 2σ), i.e. it is outside the measured jitter.

Both knobs are env-tunable in the workflow (`REGRESSION_THRESHOLD`,
`NOISE_SIGMAS`). New benchmarks (no baseline) are reported but never gate;
benchmarks missing from the PR (renamed/removed) are flagged as warnings.
`main` keeps publishing its post-merge numbers via `benchmarks.yml`.

Run the same gate locally before pushing:

```bash
scripts/perf-gate-local.sh                 # vs origin/main, 10% / 2σ
scripts/perf-gate-local.sh origin/main 0.15 2.0
```

To enforce it, make **“Benchmark regression gate”** a required status check
(Settings → Branches → `main` → branch protection).
