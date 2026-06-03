# Fused-int8 matmul (decode speed) — Scope

**Repo:** `hologram` (backend kernels). Branch spike: `fused-int8`.
**Status:** scoping. Spike done + validated; this is the productionization plan.

## Why (validated by the spike)

int8 weight quantization (hologram-ai PR #5) is currently **dequant-then-matmul**:
it expands the int8 weight to a full f32 scratch buffer, then runs the normal f32
matmul → ~2× *slower* than f32 on compute. So today int8 is a size win only.

The spike (`cpu::simd::matmul_i8_per_channel`, on this branch) reads the int8
weight directly and dequantizes each 16-wide column tile in registers; the
per-column scale factors out of the k-loop to the writeback
(`out[i][j] = scale[j]·Σ_k a[i][k]·qb[k][j]`, zp=0). Measured (M4 Max, single-thread):

| shape | f32 | fused-int8 | result |
|---|---|---|---|
| 256³ (prefill) | 311 µs | 1017 µs | **3.3× slower** |
| 1×2048×2048 (decode) | 435 µs | 270 µs | **1.6× faster** |
| 1×4096×4096 (decode) | 3262 µs | 1990 µs | **1.6× faster** |
| 1×2048×8192 (decode) | 3057 µs | 2072 µs | **1.5× faster** |

**Conclusion:** fused-int8 is ~1.5–1.6× faster than f32 (and >2× faster than the
current dequant-then-matmul) on the **decode (M=1)** shape that dominates token
generation — *plus* the 4× size reduction. It is ~3× slower on compute-bound
prefill (the spike is GEMV-style, no A-reuse). So the kernel must be **M-gated**:
fused-int for decode, the existing f32-tiled / dequant path for prefill.

## Goal

Make int8 a genuine **decode speed** win on the wasm-primary target: dispatch a
fused-int kernel for small-M (decode), on aarch64 **and** wasm SIMD128, with no
prefill regression and unchanged numerics (the existing int8 accuracy gate still
passes).

## Work items

1. **Promote + harden the NEON kernel** (`cpu/simd.rs`)
   - Spike `matmul_i8_per_channel` exists + has a correctness test. Keep GEMV
     form (right for decode M=1; loop rows for small M).
   - Scope to **per-channel, symmetric (zp=0), i8, f32 output** — exactly what
     `quantize_weights` emits. Everything else falls back.

2. **wasm SIMD128 kernel** (primary target) — mirror NEON with `core::arch::wasm32`:
   `v128_load` i8x16 → `i16x8_extend_low/high_i8x16` → `i32x4_extend_low/high_i16x8`
   → `f32x4_convert_i32x4` → `f32x4_add(acc, f32x4_mul(av, b))` (no FMA on wasm)
   → scale at writeback. Gate `#[cfg(all(target_arch="wasm32", target_feature="simd128"))]`.
   Portable scalar fallback for the rest.

3. **M-gated dispatch in `matmul_dequant_float`** (`cpu/float_kernels.rs:311`)
   - When `per_channel && zero_point==0 && quant_dtype==I8 && dtype==F32 && m <= M_GATE`
     and target is aarch64 or wasm+simd128 → call the fused kernel.
   - Else → the existing dequant-then-matmul (unchanged; covers prefill, x86,
     per-tensor, zp≠0, i4).
   - **Measure the crossover** to set `M_GATE` (spike: wins at M=1, loses by M=256;
     find where it flips — expect single digits). Start conservative (`M_GATE=8`).

4. **Tests**
   - Kernel vs naive (NEON have it; add wasm run under wasmtime).
   - `matmul_dequant` integration: fused path matches dequant-then-matmul within
     f32 tol across M on both sides of `M_GATE`.
   - Re-run hologram-ai `int8_accuracy` (cosine ≥ 0.999) and `hologram-exec`
     quantization suite — unchanged.
   - Zero-alloc on the hot path (`tests/zero_cost.rs` style).

5. **Benchmark + manifest**
   - Productionize the spike harness as a hologram-bench decode bench
     (int8 vs f32, M=1 large-N); register in `benches/manifest.toml`.
   - Confirm decode speedup + no prefill regression (gating).

## Open questions / risks

- **`M_GATE` value** — needs a crossover sweep (M = 1,2,4,8,16,32 …).
- **wasm engine variance** — measured under wasmtime/Cranelift; V8/SpiderMonkey
  will differ. The widen-heavy inner loop may behave differently; verify codegen.
- **Decode N can be large** (e.g. 8192) — single-thread is fine to start; a
  multi-core (parallel over N) or a register-tiled variant are later options.
- **Numerics**: factoring the scale to writeback changes the f32 reduction order
  vs the dequant-then-matmul path — keep the tolerance test (≤ ~1e-4 rel).

## Out of scope (explicit follow-ons)

i4 fused (¼-byte, nibble unpack); asymmetric / zp≠0 (needs a per-row A-sum
correction term); x86 fused-int (keep dequant-then-matmul there); a register-tiled
fused-int for small-batch prefill; multi-core fused-int.

## Effort

One focused PR: NEON (spike done) + wasm SIMD128 kernel + M-gated wiring +
~4 tests + 1 bench + CI. Medium.
