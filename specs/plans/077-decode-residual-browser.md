# Plan 077: Decode Residual (Browser) — wasm int8 GEMV and the Path to Stream Bandwidth

## Context

hologram-ai's deployed browser decode (0.5B int8, wasm SIMD128, single 32-bit
heap) attributes its residual upstream: a seq-1 pass streams ~0.3 GB of
weights at ~7 MB/s effective against calibrated stream bandwidth in the GB/s
range. The kernel is compute-bound, not bandwidth-bound. Grounded at hologram
HEAD `a912335`, hologram-ai HEAD `8c6402a`; ordered by leverage.

hologram does not benchmark in the browser. **Acceptance for every item is
witnessed downstream by hologram-ai's performance contract** (target lane
ratio, deployed headless-Chromium journey), which exercises the actual wasm
SIMD128 code. hologram-local runs (native, qemu-aarch64, wasmtime) are
iteration signals and regression mirrors only.

**Constraint (κ-operation).** Every item stays within k-operation. Layouts,
tables, and plans are derived content under their own κ (derivation-keyed,
re-derivable, fail-closed); kernels execute over materialized κ content with
deterministic reduction order so CE derivation keys stay valid; no item
introduces a classical fallback tier.

## Phase 1 — items 1, 2, 9 (DONE)

### Item 1: wasm int8 GEMV — weight layout and integer accumulation

The prior `matmul_i8_pc_wasm` walked `bq[kk*n + j]` k-inner with 16-column
tiles: stride-n access used 16 of every 64 bytes per line with no line reuse
between tiles at decode shapes, and every product was float (W8A32, via a
10-op i8→i16→i32→f32 widening ladder).

Landed:
- `matmul_i8_pc_omajor` (`hologram-backend::cpu::simd`): GEMV over
  **output-major** `[n,k]` weights (each output's k-vector contiguous),
  per-token symmetric dynamic i8 activation quantization (**W8A8**,
  `scale = max|a|/127`, deterministic trunc-cast rounding), and **exact
  integer accumulation** — wasm `i16x8_extend` + `i32x4_dot_i16x8`, NEON
  `vmull_s8` + `vpadalq_s16`, identical integer function on scalar targets.
  Output is bit-identical across scalar / NEON / wasm (verified under
  qemu-aarch64 and wasmtime in addition to native): integer sums are
  associative, so reduction order cannot perturb CE derivation keys. Exact-i32
  bound `k ≤ mm_act_quant::K_MAX` (~133k) enforced loudly.
- `MatMulDequantCall { bq_omajor, act_quant }`: `bq_omajor` is layout-only and
  **excluded from `op_signature`** (the `b_packed` rule — the operand's κ-label
  reflects its transposed bytes); W8A8 is a different function and takes a
  **new signature tag (116)**, leaving W8A32 signatures byte-identical (no
  re-keying of existing content). Archive wire: new discriminant `D_MMDQ2 =
  116` emitted only when a field is non-default; unchanged compilations stay
  byte-identical, unknown tags fail closed.
- Compile-time fusion pass `fuse_const_i8_decode` (`hologram-compiler`): a
  constant symmetric per-channel i8 weight uniquely consumed by
  `Dequantize → MatMul(B)` at decode shapes (static m ≤ 4) fuses in the
  archive into one `MatMulDequant { bq_omajor, W8A8 }`, with the constant
  transposed `[k,n] → [n,k]` — derived content under its own κ, the quantized
  analog of the f32 panel packing. The fused call is the transposed layout's
  only reader. Dynamic quantized weights keep load-time fusion (W8A32,
  `[k,n]`), which no-ops on already-fused archives; warm-fold is unaffected
  (the fused call has a dynamic input, so it is never a constant-only cone).
- Dispatch (`matmul_dequant_float`): omajor+W8A8 routes to the new kernel on
  **all** targets, fail-closed — no W8A32 downgrade, no `[k,n]`
  misinterpretation possible.

Conformance: `wl2_*` tests prove the fusion fires in the archive, execution is
bit-identical to an independent W8A8 integer reference over the original
`[k,n]` bytes (which also witnesses the transposition), prefill shapes stay on
the runtime W8A32 path, and asymmetric zero-points never take the W8A8 path.

### Item 2: GEMV-specialized dispatch

The omajor kernel is GEMV-shaped by construction: at m = 1 it runs 4 output
rows in flight with independent i32x4 accumulators over contiguous weight
rows, unrolled k, no output tiling; the activation extends amortize over the
rows in flight. Small m (≤ 4) loops rows through the same core.

### Item 9: decode-shaped benches (upstream contract mirror)

- `decode_gemv` criterion bench: m = 1 at deployed projections — 0.5B
  896×896 / 896×4864 / 4864×896, 1.5B 1536×8960, 7B 3584×18944 — reporting
  **bytes of int8 weight streamed per second** for both the omajor W8A8
  kernel and the prior `[k,n]` W8A32 path, plus a full-pipeline
  `session_step_novel` bench (novel input per step so the memo cannot elide
  the kernel — the seq-1 per-op-overhead surface for item 7). Registered in
  `benches/manifest.toml`.
- `wasm_matmul_timing` example extended with the int8 GEMV shapes so the
  wasmtime + simd128 lane exercises the actual wasm kernel.

Iteration signals at landing (not acceptance): wasmtime SIMD128 streams
16–19 GB/s int8 on 0.5B–1.5B shapes vs 3.9–5.9 GB/s for the prior kernel
(~3×), falling to 11.5 GB/s at the 7B shape as the working set leaves cache —
i.e., the kernel is entering the bandwidth-bound regime item 1 targets.
Native x86 (scalar vs scalar) shows ~5× from layout + integer accumulation
alone.

## Item 3: acceptance is witnessed downstream (standing)

Kernel changes are accepted by hologram-ai's performance contract, which is
where the browser exists. hologram's benches catch shape-level regressions
before they reach it.

## Phase 2 — item 4: relaxed SIMD tier

`simd.rs` baseline simd128 has no fused multiply-add. Add a relaxed-simd tier
(`f32x4_relaxed_madd`; `i32x4_relaxed_dot_i8x16_i7x16_add_s` only where the
dot maps exactly — one operand provably in i7 range) behind build/runtime
detection, baseline kept as the witnessed fallback. The i8-dot relaxed
instruction removes the extend chain from the W8A8 inner loop entirely.

## Phase 3 — item 7 (+ item 8): seq-1 dispatch, fusion, and mathf

Fixture-scale decode attributes 20–50× floor to per-op overhead at seq = 1.
Levers: fuse dequant+matmul+bias+activation as one call (extend the epilogue
fusion onto `MatMulDequant`, mirroring `MatMulAddActivation`); a pre-bound
plan handle keyed by the graph's κ that validates once and replays per step;
arena reuse across steps. The `decode_gemv/session_step_novel` bench is the
measurement surface. Item 8 rides along: SIMD exp for the softmax path
(`cpu/mathf.rs` breadth), or the Q-tier exp table once item 6 lands.

## Phase 4 — item 5: wasm threads

`cpu/parallel.rs` is `std::thread`, compiled out on wasm. Decode GEMV is
row-parallel with zero synchronization inside a step. wasm threads
(SharedArrayBuffer + atomics; hologram-ai serves its own COOP/COEP headers)
need an embedder-provided worker contract; determinism is preserved by static
row partitioning (per-output reduction order unchanged, structural-ce
unaffected). Near-linear scaling until Phase 1's kernel is bandwidth-bound.

## Phase 5 — item 6: Q0/LUT-GEMM tier to main

The kernel-floor tier (fiber-ordered radix passes, one L1 line per pass,
per-element cost = table lookup) exists only as plan 033 on the migration
branch; `cpu/lut.rs` on main is Q1 unary only. This is the structural lever
below item 1's ceiling: it removes the multiply entirely and cuts the
streamed bytes, not just the widening. Sequence after Phases 2–3.
