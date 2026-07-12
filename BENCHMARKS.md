# Benchmarks & throughput

Provenance is per-section. The criterion tables below were **captured at
v0.5.0** and have not been re-run since; the "Decode kernels" section carries its
own, later capture. A number is never restamped with a version it was not
measured at.

Captured at v0.5.0 (criterion, release, 100
samples/bench; CPU). Absolute times are machine-dependent (a shared CI VM); the
**ratios** — content-addressing reuse and matmul scaling efficiency — are the
load-bearing results. Re-run with `cargo bench -p hologram-bench` and the
release perf-floor V&V with `cargo test --release -p hologram-backend --test
performance --features cpu -- --nocapture`.


---

## Decode kernels (captured at v0.7.3)

Machine: AMD EPYC 7763, 4 physical cores / 8 threads, AVX2 (no AVX-512), shared
CI VM. Method: `cargo run --release --example gemv_micro -p hologram-bench`
(single-threaded unless noted). `GMAC/s` is the precision-invariant metric — all
tiers perform the same `k·n` multiply-accumulates, so it compares them directly;
`GB/s` is the streamed weight-byte view.

**A — L3-resident weight, reused (the compute ceiling):**

| kernel | GMAC/s | GB/s (weight) |
|---|---|---|
| i8 W8A8 (`matmul_i8_pc_omajor`) | 41.8 | 41.8 |
| i4 W4A8 (`matmul_i4_pc_omajor`) | 22.7 | 11.3 |
| e8cb 1-bit (`matmul_e8cb_omajor`) | 27.1 | 3.4 |

**B — weight ≫ L3, DRAM-streamed (the real decode regime):**

| kernel | GMAC/s | GB/s (weight) |
|---|---|---|
| i8 W8A8 | 14.4 | 14.4 |
| i4 W4A8 | 13.1 | 6.5 |
| e8cb 1-bit | **27.0** | 3.4 |

Three results worth stating plainly:

- **The E8-codebook (VQ) tier wins only when memory-bound.** Cache-warm it is
  gather-compute-bound and loses to i8 (0.65×); DRAM-streamed its 8× byte
  reduction dominates and it is ~1.9× faster. The 8× smaller weight is the
  browser memory/download lever regardless of throughput.
- **i4 is slower than i8 in *every* regime measured** (0.54× and 0.91×). Its
  nibble unpack costs more than the bytes it saves, and no cost model would ever
  prefer it for speed. It remains a *memory* tier, never a speed tier — this is
  documented rather than silently assumed.
- Fewer weight bits only helps where bytes, not MACs, are the wall.

### Prefill on the output-major integer GEMV (`m > 1`)

The i8 / packed-i4 / E8CB output-major GEMVs handled `m > 1` as
`for i in 0..m { gemv(row_i) }`. Each row streamed the whole `[n,k]` weight, so
the weight was never amortized: **per-row cost stayed flat at every `m`**, pinned
at the weight's memory bandwidth.

Read `ms per row`, not total time and not GB/s. Total time grows with `m` because
the work does; an earlier version of this note mistook that for the defect and
quoted a meaningless "GB/s collapse" (a constant `k·n` divided by `m`-proportional
time). The real signal is per-row cost *failing to fall* as `m` rises.

The fix blocks the output columns, so a block's weight slab is read once and
reused by all `m` rows. x86-64 AVX2, `ms per row`:

| weight | | m=1 | m=4 | m=16 | GMAC/s @ m=16 |
|---|---|---|---|---|---|
| 4 MB (L3) | per-row loop | 0.120 | 0.097 | 0.095 | 44.3 |
| | blocked | 0.115 | 0.087 | **0.084** | 50.1 |
| 32 MB (~LLC) | per-row loop | 1.713 | 1.570 | 1.698 | 19.8 |
| | blocked | 1.783 | 0.937 | **0.718** | 46.8 |
| 64 MB (DRAM) | per-row loop | 2.900 | 2.744 | 2.798 | 24.0 |
| | blocked | 2.688 | 1.617 | **1.377** | 48.8 |

At `m = 16` that is **2.37× at 32 MB** and **2.03× at 64 MB** of total time;
blocked prefill reaches the same ~48–50 GMAC/s compute ceiling the cache-resident
case hits, while the per-row loop is stuck at the weight's DRAM bandwidth
forever. At 4 MB the GEMV is compute-bound and the blocking is worth 1.13× —
that is physics, not a defect, and the pin says so.

Decode (`m = 1`) is untouched: one row has nothing to amortize over, and it keeps
the pooled dispatch. Pinned by
`cargo run --release --example i8_m_scaling -p hologram-backend`.

**Reordering is free here, and only here.** Every output cell is one whole dot
over the same `k`-vector; only the order in which cells are visited changes. The
accumulation is an exact i32 sum and integer addition is associative *and*
commutative, so no tiling, blocking, or completion order can move a bit — the
schedule-independence the integer path has and the f32 path does not. Witness:
`batched_integer_gemv_equals_row_by_row_bit_for_bit` asserts a batched call
equals `m` independent single-row calls byte for byte, for all three tiers, on
AVX2 / NEON / wasm SIMD128 / wasm relaxed-SIMD.

### Pooled prefill on wasm (captured at v0.8.2) — and where the ceiling actually is

TTFT is the serial `m > 1` GEMM. The wasm worker pool — previously decode-only —
now partitions **any** `m` by output column: each participant computes its
columns for all `m` rows, reading its weight tile once. (`m` pooled single-row
jobs would reload the weight `m` times and lose to serial batched; the batched
kernel itself is what gets partitioned.) Bit-identity is structural and pinned
at `m ∈ {1,2,5,8}` for all three tiers by `parallel_gemv_matches_serial_bitwise`.

wasm32-wasip1-threads under wasmtime, 3 workers + main (4 participants):

| shape (i8 W8A8) | serial | pooled | speedup |
|---|--:|--:|--:|
| decode `1×1536×8960` | 759.6 µs | 187.1 µs | 4.06× |
| prefill `32×896×4864` | 7.37 ms | 1.91 ms | **3.87×** |
| prefill `128×896×4864` | 29.2 ms | 7.66 ms | **3.81×** |
| prefill `128×1536×8960` | 86.0 ms | 21.9 ms | **3.93×** |

Near-linear on 4 participants; pooled prefill sustains ~73–80 GMAC/s where
serial sat at ~19–20. Reproduce with `wasm_threads_timing` (wasmtime,
`-W threads=y -S threads=y`).

### Fused decode attention: the long-context ceiling, removed (captured at v0.9.0)

hologram-ai's measurement: with the GEMV pool fixed in context, attention + the
per-step KV recopy grow with context and the pool speedup collapses (3.09× at
L=128 → 1.14× at L=32768; ~440 ms/token of pure `Concat`+`Transpose` recopy at
32K on a 1.5B model). Two substrate contracts forced that: `AttentionCall`'s
single-`k` signature (⇒ in-graph `Concat(past, new)` recopies the whole bucket
every step) and `causal`-only masking over one `seq` (⇒ a runtime realized
length was inexpressible, and `m = 1` against `L` keys wasn't even a legal
shape).

`DecodeAttentionCall` (new κ discriminant 119; the legacy call's wire is
frozen) removes both: **split KV** read in place — the kernel iterates
`past ∥ new` so the concatenation is never materialized, witnessed bit-for-bit
against the legacy kernel over the precatenated buffer — and a **required
additive mask** `[m, past+new]`, where `-inf` erases a key *exactly* (padded
bucket ≡ tight computation, bit for bit). The κ split is deliberate: the fixed
bucket is structure in the call; the realized length is content in the mask
operand's bytes. Every `(batch, head, query-row)` is one whole
score→softmax→context pipeline — the partition unit, pooled on wasm (fork-join
by row, publisher-carried score scratch so workers never allocate) and native
(`parallel` column… row tiles), bit-identical to serial by construction.

wasm32-wasip1-threads under wasmtime, 1.5B-class heads (h=12, kv=2, d=128),
`m = 1` decode step. Three runs on the shared CI VM; the table shows the
observed range (run-to-run pool contention moves the pooled absolutes, the
serial baseline is stable to ~4%):

| context L | serial | pooled (4 participants) | speedup | KV-read GB/s |
|---|--:|--:|--:|--:|
| 128 | 74–76 µs | (declined: below work floor) | 1.0× | ~3.5 |
| 8 192 | 4.73–4.93 ms | 1.20–1.75 ms | **2.8–4.0×** | 3.4 → 9.6–13.9 |
| 32 768 | 23.6–24.3 ms | 7.29–8.09 ms | **2.9–3.2×** | 2.8 → 8.3–9.2 |

And the recopy the split form deletes is gone *by construction*, not by
optimization: there is no `Concat` in the lowered graph to pay for. Per-token
attention cost is now the inherent O(context) KV **read**, divided across
participants — the parametric no-arbitrary-ceiling shape the request asked for.
Decode (`m = 1`), chunked prefill (`m = C`, causal-within-chunk in the mask)
and speculative verify (`m = K`) all ride the same call.

### Resident KV: the cache write is a κ move; the byte boundary disappears (captured at v0.9.0)

The third decode ceiling hologram-ai measured was invisible to kernel
benchmarks: carrying `past_k`/`past_v` as host bytes re-hashes (BLAKE3) and
recopies the **entire** cache across the byte boundary every token — a second
O(bucket) per-step cost on top of the recopy (their numbers: ~28 ms/tok @2K →
~442 ms/tok @32K on a 1.5B model, native SIMD, before the kernel even runs).
The substrate already had the addressed path (`execute_addressed`: *"nothing
is rehashed"*) — what was missing was a way to **append to a resident cache**
so the updated value could be retained by label and bound next step.

`KvCacheWrite` (OpKind + κ discriminant 120) is that append: a fixed-bucket
row write at a **runtime** position (ring wrap; the position is a 4-byte
operand — content, like the decode mask's realized length, so one compiled
step-graph serves every step). The kernel's contract is an honest
O(bucket) copy. The *executor* realizes an eligible write as an in-place
**move**: the old cache label is retired (a moved value is never
re-addressed), the buffer is mutated at O(new_rows), and the result is
retained under the derived output label — bit-identical to the copy by
construction, pinned by `hologram-exec/tests/kv_cache_write.rs` including a
two-step decode loop against directly-dispatched kernels. Eligibility is
sound, not assumed: the load-time analysis re-derives it from the decoded
plan (every other toucher of the cache slot scheduled strictly earlier — a
hand-built archive cannot spoof the flag), and at steal time the pool
declines unless the buffer is owned by exactly this node's operands (view
aliases, duplicate-label ports, pinned/lazy tiers all decline to the honest
copy — each pathology witnessed).

One decode-step graph (`DecodeAttention` + 2×`KvCacheWrite`, h=12 kv=2 d=128,
m=1), driven both ways — native x86-64, release, serial kernels; the byte
column is the *optimistic lower bound* on what the addressed loop removes
(deployed wasm32 hashes slower):

| bucket L | byte loop (µs/step) | addressed loop (µs/step) | speedup |
|---|--:|--:|--:|
| 2 048 | 3 852–4 017 | 578–589 | **6.5–7.0×** |
| 8 192 | 23 748–23 915 | 2 378–2 583 | **9.2–10.1×** |
| 32 768 | 144 870–165 022 | 18 186–19 487 | **8.0–8.5×** |

(Two runs on the shared CI VM; the second run includes best-fit buffer
recycling — see below — which also trimmed the byte loop's allocator churn
at 32K.)

The addressed column is essentially the attention kernel itself; everything
else — 2×O(bucket) re-hash, 2×O(bucket) copy-in, 2×O(bucket) copy-out, and
the 2×O(bucket) honest-copy writes — became label binds plus two O(1)-row
in-place moves. Reproduce with
`cargo run --release -p hologram-exec --example addressed_decode_timing`.

The confinement witness (`confinement.rs`) then found the pool's free list
was size-blind LIFO: in a steady-state decode loop a 4-byte position request
could shrink-realloc a cache-sized buffer and the next large request would
grow one back — per-step malloc/free churn, and unbounded-looking allocation
drift (+64 B/step in the witness). Recycling is now **best-fit by capacity**,
so each request pairs with its own size class and the loop holds total pool
allocation *exactly* constant after warmup — pinned, not observed.

Falling out of the same hardening pass: a refused `execute_addressed` no
longer rotates the transient generations — input bindability is validated
*before* any state change, so a failed call cannot age out the resident
KV labels a retrying decode loop still needs.

### Pool admission: work admits, width declines (captured at v0.8.2)

Adversarial pass over the pooled GEMM found the admission gate wrong twice, in
opposite directions — both measured, both fixed, both pinned by
`pool_floor_probe` (wasmtime, 3 workers + main):

- **Byte-keyed floor, blind to the batch.** `k·n < 256 KiB` declined a 112 KiB
  per-head projection at `m = 128` — 14.7 MMAC of parallel work ran serial.
  Work-based admission (`m·k·n`) recovers it; at `m = 1` the two gates are the
  same number, so decode admission is unchanged:

  | shape | serial | pooled (work-based floor) |
  |---|--:|--:|
  | `128×896×128` (112 KiB) | 878 µs | 514 µs |
  | `128×256×896` (224 KiB) | 2.01 ms | 462–615 µs |
  | `64×896×256` (224 KiB) | 844 µs | 260–264 µs |

- **Work-only floor, blind to width.** At `n = 8`, 4 participants get 2 columns
  each — every row runs the SIMD column tail, and pooling *lost* (473 µs vs
  342 µs serial). Admission now also requires `n ≥ 8 × participants`; the shape
  runs serial again. The bitwise witness covers both edges: `n = 3` (declined,
  equals the exact oracle) and a ragged admitted `n = 33` (9/8/8/8 columns,
  bit-identical through the uneven split).

- **The publisher's serial share is measured, not guessed.** Quantizing
  `m = 128` rows plus dispatch is ~342 µs against a 21.7 ms pooled full-width
  call — ~1.6%. Parallelising the publisher would buy ≤ ~1.2% (Amdahl) and is
  deliberately not done.

The native `parallel` gates use the same work-based admission
(`m·k·n ≥ GEMV_PAR_THRESHOLD`), and the native column-tile prefill now covers
**all three tiers** (i8 / packed-i4 / E8CB) on x86-64 and aarch64 — previously
i8 only. Pinned bit-exact against the integer oracle at parallel scale by
`batched_gemm_all_tiers_match_the_exact_oracle_at_parallel_scale`, which runs
with and without the feature.

### Roofline verdicts: the kernels have no headroom left (captured at v0.8.2)

"Faster than yesterday" cannot answer *are we done*. `cargo run --release
--example roofline -p hologram-backend` measures the machine's ceilings in the
same process and places each kernel against them. This host (EPYC, shared VM):

| kernel | achieved | ceiling | verdict |
|---|--:|--:|---|
| i8 decode, 64 MB weight | 24.6 GB/s | 22.8 GB/s streaming-read probe | **≥100% — bandwidth-bound, done** |
| i8 prefill m=128, 64 MB | 52.9 GMAC/s | 50.3 GMAC/s cache-resident | **~100% — compute-bound, done** |
| i4 decode | 17.3 GMAC/s | (decode-compute-bound) | unpack is the wall, not bytes |
| e8cb decode | 28.0 GMAC/s | (gather-bound) | codebook gather is the wall |

(>100% means the kernel out-streams the single-stream probe — it is itself the
better bandwidth probe; the no-headroom verdict holds a fortiori.)

**Consequence.** With decode at the memory wall and prefill at the compute wall,
further *kernel* tuning on these paths is chasing noise. The remaining levers are
structural, and they are the UOR levers:

1. **Fewer bytes** — deeper weight codecs (e8cb is 8× fewer bytes; its win is
   gated on gather cost, not on kernel polish).
2. **More participants** — pooling, now covering all of inference (above).
3. **Not recomputing** — content addressing. A κ-matched re-execution is a graph
   memo hit: **zero kernel dispatches, zero weight bytes paged**, output returned
   by address, cost independent of model size. Witnessed through the shipped
   weightless + paged binding by
   `repeated_prefill_through_the_shipped_binding_is_a_memo_hit`. This is the
   super-linear axis: a shared system prompt re-executing across requests costs
   a hash, not a prefill.

### Small-`m` f32 matmul (the decode/short-prefill shape)

`matmul_f32_blocked`'s micro-kernel works on an `MR = 4` register tile. The
remainder rows (`m mod 4`) were processed **one row at a time**, and each row
re-streamed the entire `k×n` weight. B — not the FMAs — sets the time at small
`m`, so `m = 3` moved 3× the bytes of `m = 4` and ran *slower in absolute time
while doing less arithmetic*.

Two fixes, both **bit-identical** per cell (each output keeps its `kk`-ascending
FMA chain — f32 result bytes are content-addressed, so reassociating the
reduction would re-key every κ):

1. The 1–3 remainder rows share a *single* pass over B.
2. The remainder is monomorphized on the row count (`rem_rows::<R>`), so `R = 1`
   compiles to a dedicated GEMV, and low `R` gets a wide-column tier — with only
   `2·R` accumulators the 16-column loop is FMA-**latency** bound, not bandwidth
   bound. (A runtime `.take(rem)` bound leaves the row loop rolled; that cost
   wasm 1.42× at `m = 1`, which is why step 2 exists.)

`k = n = 1024`, x86-64 AVX2, best-of-3:

| m | before | after | speedup |
|---|---|---|---|
| 1 | 0.448 ms | 0.159 ms | **2.82×** |
| 2 | 0.605 ms | 0.178 ms | **3.40×** |
| 3 | 0.890 ms | 0.256 ms | **3.48×** |
| 4 | 0.238 ms | 0.240 ms | — (unchanged tile path) |
| 6 | 0.608 ms | 0.299 ms | 2.03× |
| 7 | 0.786 ms | 0.350 ms | 2.25× |

`m = 3` now costs what `m = 4` costs, as it should: both are one pass over B.
Low GFLOP/s at small `m` is physics (B dominates the traffic), not a defect —
the pathology was absolute time *rising* as `m` fell. Pinned by
`cargo run --release --example matmul_small_m -p hologram-bench`.

**The lane that ships.** The first version of this fix landed only in the x86
leaves; NEON and wasm still re-streamed B per leftover row, so none of it reached
a single-threaded-wasm consumer. Both are now fixed. wasm32 + SIMD128 under
wasmtime, `m = 1`, best-of-3:

| shape | blocked before | blocked after | packed before | packed after |
|---|---|---|---|---|
| k=n=512  |  33.5 µs |  37.2 µs |  25.6 µs |  26.4 µs |
| k=n=1024 | 208.2 µs | **162.1 µs** | 103.5 µs | 105.7 µs |
| k=n=2048 | 872.5 µs | **715.9 µs** | 452.2 µs | 439.3 µs |

plus `m = 2` 0.429 → 0.227 ms (1.89×) and `m = 3` 0.634 → 0.317 ms (2.00×) at
`k = n = 1024`. Pinned by the `small-m sweep` section of
`cargo run --release --example wasm_matmul_timing -p hologram-backend --target
wasm32-wasip1` (criterion, a hologram-bench dependency, does not build for wasm).
The `k = n = 512` blocked row is ~11% *slower*: at 1 MB the weight is
cache-resident, so the extra tier is overhead rather than latency cover. Recorded,
not hidden.

The wide-column tier for the **packed** leaf is enabled on x86 only (native
`m = 1`, `k = n = 2048`: 319.5 → 254.6 µs, 1.25×). On wasm the same tier measured
1.16–1.25× *slower* — there are no registers, and 8 v128 accumulators spill — so
it is not applied there by symmetry. NEON keeps the strided tier on the same
register-machine premise and is correctness-verified under qemu; it is not timed
here, so no NEON speedup is claimed.

`m = 5..7` legitimately take two passes over B (one `MR` tile plus a remainder).
`MR` cannot grow past 4 without exceeding the 16 YMM registers the 4×16 tile
already fills, so that is the floor, not a defect.


Multi-threaded (`--features parallel`) and end-to-end per-token figures move with
the machine's memory subsystem; re-run `decode_profile` on the target rather than
quoting these.


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
the kernel is entering the bandwidth-bound regime. The relaxed-SIMD build
(`+relaxed-simd`, wasmtime `-W relaxed-simd=y`) computes the same exact
function via `i32x4_relaxed_dot_i8x16_i7x16_add` over a `q⁺ − q⁻` i7 split
and lifts the 7B shape to 14.5 GB/s (+26%); cache-resident shapes reach
17.6–19.6 GB/s. Output is bit-identical
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
