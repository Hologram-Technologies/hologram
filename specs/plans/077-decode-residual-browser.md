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

**ISA completeness.** Every decode kernel — the i8/i4 omajor GEMVs and the
deterministic softmax exp — carries a full inner on each supported ISA:
x86_64 **AVX2** (`_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` exact-i32
dot; `_mm256_shuffle_epi8` nibble LUT; the no-FMA exp sequence 8-wide),
aarch64 **NEON**, wasm **SIMD128** (baseline + relaxed), selected by runtime
CPUID on x86 / baseline elsewhere — no arch falls to the scalar reference in
a stock build, per this module's dispatch contract. All four lanes are
**bit-identical** (integer madd is exact and associative; the exp replays
one fixed op sequence), pinned by the same bit-exactness suite run natively,
under qemu-aarch64, and under wasmtime on both wasm tiers. Native AVX2 now
streams ~37 GiB/s int8 at 0.5B shapes (the fastest lane, and the honest
number the native `decode_gemv` bench reports).

## Item 3: acceptance is witnessed downstream (standing)

Kernel changes are accepted by hologram-ai's performance contract, which is
where the browser exists. hologram's benches catch shape-level regressions
before they reach it.

## Phase 2 — item 4: relaxed SIMD tier (DONE)

Landed as an **exact-only** tier: `gemv_i8_omajor_wasm_relaxed` (built with
`-Ctarget-feature=+simd128,+relaxed-simd`) computes the **same W8A8
function** with `i32x4_relaxed_dot_i8x16_i7x16_add` by splitting the signed
activation row `q = q⁺ − q⁻` with both halves in the i7 range `[0, 127]` —
exactly where the relaxed dot is exact and engine-deterministic (products
≤ 127², internal pairwise i16 sums ≤ 32258 cannot saturate). Each 16-wide
step is two relaxed dots per row instead of two extends + two dots + two
adds, with no activation extends at all. Output stays bit-identical to the
baseline and scalar paths (the exactness suite passes on both builds under
wasmtime), so the tier is a pure execution speedup with zero semantic
surface — no call-surface, signature, codec, or compiler change. The
baseline simd128 build remains the witnessed fallback; `just wasm` builds
both tiers so neither bit-rots.

`f32x4_relaxed_madd` was measured (wasmtime, x86-64 FMA host) and
**regressed the f32 kernels ~30%** — the register-tile accumulator chains
are latency-bound and the fused op lengthens the dependency chain — so it is
deliberately excluded (documented at the wasm_simd module comment). Re-open
only with an in-browser V8 measurement from hologram-ai showing otherwise.

wasmtime relaxed-tier signal: 17.6–19.6 GB/s int8 at cache-resident decode
shapes and 14.5 GB/s at the 7B shape (vs 11.5 baseline, +26%) — the kernel
is bandwidth-limited on this host.

## Phase 3 — item 7 (+ item 8): seq-1 dispatch, fusion, and mathf (IN PROGRESS)

Fixture-scale decode attributes 20–50× floor to per-op overhead at seq = 1.

Landed:
- Fusion pass ordering: dequant→matmul fuses before the matmul epilogue, so
  a dynamic quantized weight followed by an activation keeps streaming in
  place instead of re-materializing the dense f32 weight each step.
- dequant+matmul+bias+activation as ONE call: `MatMulDequantCall` carries a
  fused epilogue (`act`, `residual`) — signature-visible (extended tag),
  wire-carried on the extended discriminant, applied in place at dispatch
  while the `m·n` results are hot. The load-time epilogue pass absorbs
  activation-only, bias-add-only, and the three-op `matmul → add → act`
  chain into fused dequant-matmuls, including archive-carried compile-time
  omajor W8A8 calls. Conformance: `gelu(A·dequant(Bq) + bias)` is one call;
  exact epilogues (relu) stay bit-identical to the W8A8 reference.

Also landed — validate-once / replay-per-step for the walk: callgrind
attributed the fixed per-step residual (~100 µs even on a 1-node graph;
~60% of all instructions at small shapes) to the boundary-address mint —
`derive_label_witnessed` grounded a full ψ-tower composition per operand
per step, and the walk immediately dropped the TC-05 witness. The grounded
address is definitionally the σ-axis fold of the composed digests, so
`derive_label_boundary` / `compose_ordered_blake3_address` now mint the
**same address** with plain streaming hashes; pointwise equality with the
witnessed authority is pinned by tests (fail-closed against future algebra
changes), and the witness remains re-derivable on demand. Measured: walk
overhead ~100 µs → ~10–28 µs per step; `session_step_novel` −6% end-to-end
at 896×4864 (kernel-dominated).

Item 8 landed: `exp_f32_det` — one fixed IEEE mul/add sequence (range
reduction with trunc-cast round-half-away, degree-6 Horner polynomial,
exponent-bit 2^k scale; deliberately no FMA) with NEON and wasm SIMD128
lanes replaying it exactly, so scalar / NEON / wasm are **bit-identical**
(pinned by tests on all lanes incl. the relaxed tier). Wired into
`softmax_float` and the attention inner softmax; sequential reduction
order unchanged; masked −∞ scores map to exactly 0. ~2× over the scalar
libm loop on the wasm lane, and it removes a pre-existing determinism
split (std builds used the platform libm, no_std the libm crate). The
Q-tier exp table stays an item-6 follow-up.

Phase-3 residual: constant rebinding (O(constants) map hits per step) —
revisit with a many-hundred-weight model.

## Phase 4 — item 5: wasm threads (DONE)

Landed as `cpu/wasm_pool.rs` behind the `wasm-threads` feature (plain
simd128 builds byte-unchanged — the witnessed fallback):

- **Embedder contract**: the host (hologram-ai; it already serves
  COOP/COEP) builds shared-memory
  (`+simd128,+atomics,+bulk-memory,+mutable-globals`), instantiates the
  module on N web workers sharing one linear memory, and each worker calls
  the exported `hologram_worker_run(id)` once, before the first execute
  (late registration traps — fail-loud). Work flows through a single
  fork-join job slot in linear memory: epoch published by the executing
  thread, drained by workers, `done` joined by the publisher. Workers never
  allocate. `hologram_pool_shutdown` / `hologram_pool_workers` complete the
  lifecycle surface.
- **Embedder futex**: wasm's native `memory.atomic.wait32`/`notify`
  intrinsics are unstable on stable Rust (rust-lang/rust#77839), so no_std
  builds import `hologram_host_wait32`/`hologram_host_notify` — one-line JS
  wrappers over `Atomics.wait`/`Atomics.notify`. The std test lane
  (wasm32-wasip1-threads) parks by spin + OS yield; the synchronization
  algebra is identical.
- **Determinism is structural**: the GEMV partitions output rows into
  contiguous per-participant ranges; every row is computed whole by one
  participant running the identical single-threaded inner, so per-output
  reduction order — and every CE derivation key — is unchanged. Locked by
  `parallel_gemv_matches_serial_bitwise`, which runs real threads under
  wasmtime (`-W threads=y -S threads`) and compares bits against serial.
- A `POOL_MIN_WEIGHT_BYTES` latency floor (structural: wake+join round-trip
  vs. per-slice work, not model-derived) keeps tiny GEMVs serial.
- Scaling signal (wasmtime, 3 workers + main, `wasm_threads_timing`):
  896×4864 18.5 → 45.6 GB/s (2.5×), 1536×8960 19.0 → 71.8 GB/s (3.8×,
  near-linear), 3584×18944 14.2 → 35.1 GB/s (DRAM-saturated) — the
  aggregate now sits at memory bandwidth, which is item 6's cue: cut the
  streamed bytes.

## Phase 5 — item 6: Q0/LUT tier — decode core to main (DONE)

The migration branch (`origin/port/uor-foundation-0.3.0`) carries the full
LUT-GEMM machinery — `hologram-core/src/lut/`, `hologram-ring`,
`hologram-exec/src/lut_gemm/` (matmul, orbit compression, psumbook,
fiber-ordered radix) — in an architecture main no longer has, ~2900 lines
across crates that don't exist here. Rather than port that whole surface, we
landed **the piece the decode residual actually names**: the LUT-tier GEMV
that removes the stored multiply and **halves the streamed weight bytes**,
inside the exact κ-discipline items 1–5 established.

Landed:
- `matmul_i4_pc_omajor` (`hologram-backend::cpu::simd`): output-major
  **packed-i4** W4A8 GEMV. The stored-weight multiply becomes an in-register
  16-entry table lookup — `i8x16_swizzle` (wasm) / `vqtbl1q_s8` (NEON), the
  value grid `I4_VALUES` living in one SIMD register — after which the
  looked-up i8 values flow through the *identical* integer dot pipeline as
  the i8 kernel (baseline extends + `i32x4_dot_i16x8`; relaxed `q⁺/q⁻` i7
  dots; scalar MACs). Output is **bit-identical** across scalar / NEON / wasm
  on both SIMD tiers (`matmul_i4_pc_omajor_matches_integer_reference`,
  verified natively, under qemu-aarch64, and under wasmtime on both tiers).
  The activation is de-interleaved once per token (`[q_even | q_odd]`, or the
  four-way `q⁺/q⁻` split on the relaxed tier) so the packed nibbles need no
  lane shuffle in the inner loop — the swizzle overhead lives outside the hot
  path.
- Dispatch: `matmul_dequant_float` routes `bq_omajor + W8A8 + quant_dtype ==
  i4` to the packed kernel, fail-closed (even-k guard); the W8A8 signature
  tag and archive discriminant are unchanged — i4 is just another
  `quant_dtype` under the existing extended `MatMulDequant`, no new
  call-surface, codec, or signature.
- Compiler: `fuse_const_i8_decode` gained the i4 arm — a constant packed-i4
  weight (even `k`, whole packed bytes) uniquely consumed by
  `Dequantize → MatMul(B)` at decode shapes fuses in the archive with the
  nibbles **repacked** into output-major `[n, k/2]` (derived content under
  its own κ, the i4 analog of the i8 transpose). Odd-k i4 falls through to
  the generic W8A32 path (`wl3_odd_k_i4_stays_on_generic_path`).
- Pool: the fork-join job carries a `kind` (0 = i8, 1 = i4); the i4 path
  passes its de-interleaved activation through the shared slot, and
  `parallel_gemv_matches_serial_bitwise` now locks bit-identity for **both**
  kinds under real wasmtime threads.

Conformance: `wl3_const_i4_decode_weight_fuses_lut_tier_and_conforms` proves
the archive fusion fires and execution is bit-identical to an independent
W4A8 integer reference over the *original* `[k,n]` nibble packing (also
witnessing the repack).

Signal (wasmtime, relaxed tier): W4A8 streams 5.9–7.2 GB/s of int4 bytes at
decode shapes — i.e. it moves **half** the bytes of the W8A8 line at
comparable step time where compute-bound, and at the DRAM-saturated 7B shape
under the pool (3 workers + main) W4A8 finishes in 1434 µs vs W8A8's 1551 µs
while resident model footprint halves — the decisive lever for the single
32-bit heap. The full-fat orbit/psumbook/fiber-radix port (dihedral MAC
compression, non-uniform codebooks) remains available on the migration
branch as a future sprint; this landed the byte-cutting core the resource
model named.
