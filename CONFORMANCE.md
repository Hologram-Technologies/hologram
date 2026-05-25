# hologram — Conformance

> **Purpose.** The normative invariant catalog hologram must uphold to
> claim it is a correct, bottleneck-free realization of prism / uor-addr.
> Each invariant has a class + number, a normative statement, an
> enforcement mechanism, and a traced artifact (test/bench/proof).
> Mirrors `prism-btc`'s `CONFORMANCE.md`. Reproduced by `just vv`.
>
> **Status legend:** ✅ enforced & passing · 🟡 partial · ⛔ gap (tracked).

## Classes

| Class | Scope | Enforcement |
|---|---|---|
| **AS** | Addressing / σ-axis correctness vs external authority | conformance tests vs BLAKE3 reference |
| **MA** | Model addressing vs format spec + cross-tool + replay | conformance tests vs GGUF/ONNX |
| **KC** | Kernel numeric conformance vs ONNX op spec + IEEE-754 | conformance tests vs reference vectors/runtime |
| **CA** | Content-addressed execution operates on κ-labels, verifiably | exec tests + replay |
| **SG** | Sub-graph addressing — shared computation is recognized by κ-label and elided | exec reuse tests (dispatch counting) |
| **FU** | Content-addressed fusion — composable sub-graphs collapse to one κ-addressed op, eliding intermediates | exec fusion tests (fused/kernel counts + f64 ref) |
| **WS** | Warm-start — the compiled object carries the constant-derivation lattice (+ materialized fold results) so the runtime cache is never cold; warm ≡ cold byte-identically | compiler/exec tests (baked==derived, cone-complete, dispatch elision, warm==cold) |
| **PV** | Performance — every part within budget, no bottlenecks | benches with baselines/budgets |
| **NS** | `no_std` portability (wasm + bare-metal) | cross-target builds |
| **RP** | Replay — TC-05 witnesses verify (QS-05 equivalence) | witness round-trip tests |

## AS — Addressing / σ-axis (external: BLAKE3 reference)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **AS-1** | hologram's σ-axis (`HologramHasher` = `prism::crypto::Blake3Hasher`) equals the upstream BLAKE3 reference byte-for-byte over inputs spanning chunk (1024 B) and subtree-merge boundaries. | cross-impl test vs `blake3` crate | `hologram-archive/tests/conformance.rs::as1_sigma_axis_equals_blake3_reference` | ✅ |
| **AS-2** | `address_bytes` emits exactly `blake3:<64 lowercase-hex>` (71 B) of the reference digest — the canonical κ-label wire form. | cross-impl test | `…::as2_address_bytes_is_canonical_kappa_label` | ✅ |
| **AS-3** | Equal content ⇒ equal κ-label (determinism). | test | `…::as3_addressing_is_deterministic` | ✅ |
| **AS-4** | A single-bit input change changes the κ-label (collision-sensitivity). | test | `…::as4_single_bit_change_changes_label` | ✅ |
| **AS-5** | Incremental `fold_bytes` (ADR-060 streaming) equals a one-shot hash and the reference. | test | `…::as5_streaming_equals_one_shot` | ✅ |

## MA / RP — Model addressing + replay (external: GGUF/ONNX spec, TC-05)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **MA-1** | A well-formed GGUF model addresses to a verifiable 71-byte `blake3:` κ-label whose TC-05 witness re-certifies to the same label. | test | `hologram-archive/tests/model_address.rs::gguf_model_addresses_to_verifiable_kappa_label` | ✅ |
| **MA-2** | Distinct model content ⇒ distinct κ-labels; CS-G2 composition of shard labels is order-independent. | test | `…::distinct_weights_yield_distinct_labels`, `…::gguf_label_composes_with_e8_operations` | ✅ |
| **MA-3** | Cross-tool agreement (hologram's GGUF/ONNX addresses match an independent loader) is **inherited by construction**: hologram addresses models *through* uor-addr's `gguf`/`onnx` realization, which uor-addr itself cross-validates against independent tools (`uor-addr/tests/gguf_cross_tool.rs`, `onnx_cross_tool.rs`). hologram's responsibility — correct invocation + witness replay — is MA-1/MA-2; re-running uor-addr's cross-tool suite here would duplicate the realization owner's V&V. | architectural boundary + MA-1/2 | uor-addr cross-tool suite (upstream) + `model_address.rs` | ✅ |

## KC — Kernel numeric conformance (external: ONNX op spec + IEEE-754)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **KC-1** | f32 matmul equals an **independent f64-reference** product (the IEEE-754 definitional ground truth, à la BLAS/NumPy validation — not hologram's own evaluator) within `√k·ε_f32`, and reads all operands (no short-cut). | conformance test vs f64 reference | `hologram-backend/tests/conformance.rs::kc1_sc1_matmul_conforms_across_scale`, `::kc2_matmul_reads_all_operands_no_shortcut` | ✅ |
| **KC-2** | Softmax, LayerNorm, RMSNorm, Gelu (ONNX `approximate="tanh"`), Silu, Conv2d (NCHW valid cross-correlation), Attention (scaled dot-product), DequantizeLinear each match their independent ONNX-definition f64 reference across scale (incl. non-power-of-2). | conformance tests vs f64 reference | `hologram-backend/tests/conformance.rs::kc3…kc9_*` | ✅ |
| **KC-3** | ReduceSum/Mean/Max + MaxPool/AveragePool (NCHW) match their ONNX-definition f64 reference across scale. | conformance tests vs f64 reference | `hologram-backend/tests/conformance.rs::kc10_reduce_*`, `::kc11_pooling_*` | ✅ |
| **KC-0** | _Supplementary internal cross-check_ (external conformance is KC-1/2/3): kernels also agree with hologram's Term-tree reference evaluator. | test | `hologram-backend/tests/kernel_equivalence.rs` | ✅ |

## SC — Scaling (no short-cut / no breakdown at arbitrary size)

> The V&V must *demonstrate* hologram holds at arbitrary scale — that it
> never silently degrades to a wrong/partial result or breaks down
> (tail-handling, precision, overflow) as sizes grow. Mirrors prism-btc's
> CM (mainnet-readiness) / CP (scaling-across-decades) classes.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **SC-1** | f32 matmul stays conformant to the f64 reference across `(m,k,n)` from 2³ through 512×64×384, **including non-power-of-2 and rectangular shapes** (which expose tail/blocking bugs); normalized error stays `≪ 1e-4`, not diverging with size. | scale-parameterized conformance test | `hologram-backend/tests/conformance.rs::kc1_sc1_matmul_conforms_across_scale` | ✅ |
| **SC-2** | Content-addressed execution holds at scale: at each size 8³…128³ the executed output matches the f64 reference, and a memoized re-execution is byte-identical to the first (no degenerate short-cut at scale). | scale-parameterized exec test | `hologram-exec/tests/conformance.rs::sc2_content_addressed_matmul_conforms_and_reuses_across_scale` | ✅ |
| **SC-3** | The transient buffer pool stays byte-bounded across arbitrarily long runs (generational eviction); pinned constants survive churn. | unit test | `hologram-exec/src/buffer.rs::tests::transient_bytes_are_bounded_regardless_of_run_length`, `::pinned_survives_transient_churn` | ✅ |
| **SC-4** | A matmul against a **constant weight** (the inference case) matches the f64 reference and is non-zero. _Regression guard: the V&V exposed a `lower.rs` bug — constant-operand shapes were unresolved, so weight-matmuls inferred `m=k=n=0` and silently no-op'd to zeros; now fixed._ | exec conformance test | `hologram-exec/tests/conformance.rs::sc4_matmul_against_constant_weight_conforms` | ✅ |

## CA — Content-addressed execution

> **Zero-movement substrate.** The runtime has a *single* substrate — a
> content-addressed buffer pool (`hologram-exec/src/buffer.rs`). Each value
> lives in exactly one 64-byte-aligned buffer; a slot is a *binding* to a
> buffer, not a copy. A produced value is written once by its kernel and
> thereafter referred to by binding: sub-graph reuse rebinds the slot to the
> existing buffer (no copy-back), retention keeps the buffer keyed by its
> κ-label (no copy-out), constants are pinned buffers (no second copy of a
> weight). The legacy design kept a fixed byte arena *and* a separate
> content store and copied every tensor between them per node; that movement
> is eliminated. Weights-loading through output resolution operate on
> κ-labels; the only byte copies are the unavoidable I/O boundary (caller
> bytes in, output bytes out) and the one-time constant load.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CA-1** | `execute_addressed` operates on κ-labels end-to-end; identical input addresses return cached output addresses without recompute or rehash. | exec test | `hologram-exec/tests/content_addressed.rs::addressed_io_matches_byte_io_and_never_rehashes` | ✅ |
| **CA-2** | Memoized output ≡ recomputed output, byte-identical (cross-validation of the reuse path). | exec test | `…::identical_reexecution_is_fully_memoized` (+ byte path) | ✅ |
| **CA-3** | Derived/output κ-labels are built by **witnessed composition** of the input addresses (a CS-G2 fold from a per-graph/port identity base, via uor-addr's `compose_g2_product_blake3`, as `prism-btc` demonstrates) — carrying a replayable TC-05 witness, not a bare derivation hash. Determinism + order-sensitivity are *proven* by the suite. An `InferenceSession`'s **output-port** addresses are minted this way (the witnessed composition of the producing node's operand labels with its op signature) — the replayable boundary address. Interior nodes use the cheap, order-sensitive `derive_label` reuse key (see SG) so the prism-pipeline grounding cost is paid once per output, not per node. | archive replay test + exec | `hologram-archive/tests/conformance.rs::ca3_derived_address_carries_replayable_witness`, `::ca3_derivation_is_deterministic_and_order_sensitive` | ✅ |

## SG — Sub-graph content addressing (reuse beyond whole-graph exact match)

> Whole-graph memoization (CA-1/2) only fires when the *entire* input set
> repeats. Sub-graph addressing keys **every node** on the cheap,
> order-sensitive derivation (`derive_label`: op signature ‖ ordered
> operand labels) of its operands' κ-labels — an `O(operands)` fold paid on
> the hot path with no measurable per-node overhead. Any shared
> computation — a prefix unchanged across runs (the KV-cache case) or a
> common subexpression within one run — is recognized by that key and its
> kernel dispatch is **elided**, raising the reuse rate `r` (and thus the
> effective ops/core-cycle, which is `peak/(1−r)`) above what exact-match
> whole-graph reuse can reach. Eliding is observable via the per-walk
> dispatch/skip counters; correctness of the reused result is held to the
> f64 reference. (Output-port addresses remain the *witnessed* form — CA-3.)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **SG-1** | Each node is keyed by the order-sensitive derivation (`derive_label`: op signature ‖ ordered operand labels) of its operands. When the top-level input set differs but a sub-graph's operands are unchanged, the whole-graph memo **misses** yet the unchanged sub-graph's kernels are **elided** (recognized by resident key), and the result still equals the independent f64 reference. _The prefix / KV-cache reuse case._ | sub-graph reuse test (dispatch counting + f64 ref) | `hologram-exec/tests/conformance.rs::sg1_subgraph_reuse_elides_unchanged_branch_across_inputs` | ✅ |
| **SG-2** | An identical computation appearing twice in one graph (same op, same operand labels — a common subexpression) is computed **once**; the duplicate is elided within a single `execute`, and the output equals the f64 reference. | intra-graph CSE test (skip counting + f64 ref) | `hologram-exec/tests/conformance.rs::sg2_common_subexpression_elided_within_one_execution` | ✅ |

## FU — Content-addressed fusion (the UOR-native execution pass)

> Where SG elides a *redundant* computation, FU elides an *intermediate*.
> The executor's fusion pass collapses a `matmul → elementwise-activation`
> sub-graph into one fused op whose activation runs in the matmul epilogue
> (while the result is hot in cache), so the activation's intermediate is
> never materialized as a distinct buffer nor addressed as a distinct
> κ-value — the fused node carries a **single κ-derivation**. This is the
> UOR-native answer to the production-throughput gap PV-4 found
> (matmul→activation interleaving): the composite op is the addressable
> unit, not its pieces. Fusion is **guarded** — it fires only when the
> matmul's output has exactly one observer (the activation) and is not a
> graph output, so it never changes observable semantics.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **FU-1** | A fused `matmul → activation` (relu / gelu / silu) equals the independent f64 reference `activation(matmul(a,b))` across scale, **and the intermediate is elided** — the pair becomes one kernel (`fused_count == 1`, `kernel_count == 1`), one content-addressed op. | fusion conformance test (f64 ref + fused/kernel counts) | `hologram-exec/tests/conformance.rs::fu1_fused_matmul_activation_conforms_and_elides_intermediate` | ✅ |
| **FU-2** | Fusion is **semantics-preserving** — the fused result is byte-identical to the unfused computation — and **guarded**: a matmul whose output has a second observer is *not* fused (`fused_count == 0`), so the intermediate it still needs is preserved. | fused-vs-unfused equality + guard test | `hologram-exec/tests/conformance.rs::fu2_fusion_is_semantics_preserving_and_guarded` | ✅ |
| **FU-3** | Fusion fires on the production workload: a stacked transformer-MLP fuses one `matmul → gelu` per layer (the 4·d-wide activation intermediate is never materialized), reducing kernel count, while PV-4's throughput/scaling floors still hold. | production perf test (fused_count == layers) | `hologram-exec/tests/performance.rs::pv4_production_mlp_throughput_latency_and_scaling` | ✅ |

## WS — Warm-start (the compiled object is never cold)

> A κ-label is a **deterministic function of the compiled graph**:
> `derive_label(opcode ‖ params, ordered operand labels)`. The op signature
> and params are fixed at compile time, and a constant/weight leaf addresses
> to a fixed label by its content — so **every node whose transitive inputs
> are all constants** (the *constant-only cone*: weight preprocessing,
> dequant, bias/transpose folds) has a fully compile-time-determined κ-label
> *and* result. Warm-start ships those into the compiled object so load is
> not cold:
>
> * **Lattice (WS-1).** Because the labels are a deterministic function of
>   the compiled graph, the runtime **derives** the cone lattice itself at
>   load (post-fusion, so it always matches what the walk dispatches) — no
>   redundant copy is baked into the archive. These labels are the keys the
>   fold pins under (WS-2) and the persisted store resolves (WS-3).
> * **Fold (WS-2).** The archive carries the cone's *materialized results*
>   (the non-recomputable part), baked by the post-compile fold pass
>   `hologram_exec::fold_archive`. At load they are pinned under their lattice
>   labels; the **existing** residency check in the node walk
>   (`pool.resident(label)`) then elides every cone node on the very first
>   run — **no walk changes, no second code path**. Because the lattice is
>   derived post-fusion, even a fused constant-only `matmul→activation` warms.
> * **Persisted κ-store (WS-3).** A content-addressed store (`WarmStore`)
>   keyed by κ-label lets results computed in one process warm a later one,
>   extending reuse past the constant cone to repeated input-dependent work.
>
> Warm-start is **observationally invisible**: a warm-loaded session is
> byte-identical to a cold one (determinism ground, ADR-030; same discipline
> as FU-2 "semantics-preserving" and CA-2 "memoized ≡ recomputed"). The
> values are held to the independent f64 reference (KC), never to a label
> hologram itself produced.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **WS-1** | The runtime **derives** the constant-only-cone lattice at load (`InferenceSession::warm_lattice`) — for every cone node, the κ-label is `derive_label(op-signature, operand labels)`, equal to an independent reference derivation, and the lattice is **complete** (a node is present iff all its transitive operands are constants; input-dependent nodes are absent). It is deterministic across loads and a warm-loaded session produces the f64 reference. No redundant copy is baked into the archive. | exec lattice accessor + independent-derivation cross-check + completeness | `hologram-exec/tests/conformance.rs::ws1_warm_lattice_matches_runtime_derivation_and_is_complete` | ✅ |
| **WS-2** | The archive carries the cone's materialized results (`hologram_exec::fold_archive` runs the cone through the real runtime and re-emits the archive with a `WarmStart` section); at load they are pinned under their lattice labels, so the first cold-input run **elides every cone node** (`last_dispatched` drops by the cone size) with output equal to the f64 reference and byte-identical to a no-warm run. Because the lattice is derived post-fusion, a fused constant-only `matmul→activation` warms too. | exec tests (dispatch counting + f64 ref + warm==cold; fused-cone case) | `hologram-exec/tests/conformance.rs::ws2_materialized_fold_elides_cone_on_first_run`, `::ws2_fused_constant_cone_is_warmed` | ✅ |
| **WS-3** | A persisted κ-store (`WarmStore` trait; in-memory `MemWarmStore`, `std`-gated file-backed `FileWarmStore`) keyed by lattice labels warms a fresh session from a prior one's computed results — even from a *labels-only* archive — so the warmed cone is elided on the first run with output equal to the f64 reference. A **missing** entry recomputes (miss-safe), and a **corrupt** entry fails its integrity check and recomputes (never a wrong result). | exec test (cross-session reuse + miss + corruption safety) | `hologram-exec/tests/warm_store.rs::ws3_persisted_store_warms_across_sessions` | ✅ |

## PV — Performance / no-bottleneck

> Two layers: **microbenches** (PV-1/3) bound a single kernel; **production
> workload** V&V (PV-2/4) bounds a full mixed-op inference graph under
> sustained serving load — what a deployment actually pays. PV-4 reports
> throughput (GFLOP/s), efficiency (FLOP/core-cycle), and per-inference
> latency, and proves they hold across sizes (arbitrary models). _Observed
> on Zen3 (AVX2+FMA, 32-FLOP/cycle peak) at the 3.24 GHz core clock: bare
> 128³ matmul ≈ 69 GFLOP/s (~21 FLOP/cycle, ~66% of peak); a production
> transformer-MLP stack ≈ 26 GFLOP/s (~8 FLOP/cycle, ~25% of peak) — the
> gap is the production reality of skinny/rectangular matmuls (seq ≪ d)
> interleaved with activation/residual bandwidth. Under serving reuse, a
> repeated request resolves in single-digit µs (≫1000× vs cold)._

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **PV-1** | f32 matmul clears a conservative throughput floor (≥1 GFLOP/s at 256³, best-of-N) and stays within the cubic scaling envelope (256³/64³ time ∈ [16,400]) — catching scalar-fallback / super-cubic / degenerate-short-cut bottlenecks. Release-only. | release perf test | `hologram-backend/tests/performance.rs::pv1_matmul_throughput_floor_and_scaling` | ✅ |
| **PV-2** | Content-addressed reuse (graph-memo hit) is ≥8× faster than recompute — a same-machine ratio proving the reuse path is O(1)-ish and not secretly recomputing. Release-only. | release perf test | `hologram-exec/tests/performance.rs::pv2_content_addressed_reuse_beats_recompute` | ✅ |
| **PV-3** | The other heavy compute kernels (Conv2d, Attention) each carry a conservative throughput floor (best-of-N) so no part is a silent bottleneck. Release-only. | release perf test | `hologram-backend/tests/performance.rs::pv3_conv_and_attention_throughput_floors` | ✅ |
| **PV-4** | A **production-representative** workload — a stacked transformer-MLP (`matmul → gelu → matmul → residual` per layer) — sustains a throughput floor under cold (all-novel) load, does **not break down across sizes** (d=256 keeps ≥¼ of d=128's per-FLOP throughput, so arbitrary sizes hold), and under serving reuse collapses to a whole-graph memo hit ≥8× faster than cold. Reports GFLOP/s, FLOP/core-cycle, and ms/inference. Release-only; mirrored by the `production` criterion bench. | release perf test + bench | `hologram-exec/tests/performance.rs::pv4_production_mlp_throughput_latency_and_scaling`, `hologram-bench/benches/production.rs` | ✅ |

## OV — No fixed byte ceiling (ADR-060: operations don't overflow at scale)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **OV-1** | Byte offsets/lengths (`BufferRef`, `SlotSpan`) and element counts (`PortDescriptor`, kernel calls, compiler `element_counts`/`byte_lengths`) are **u64** — no 4 GiB / 4.29 B-element cap, no saturating/truncating size arithmetic. Values beyond `u32::MAX` survive the archive codec round-trip without truncation. _Regression fix: these were u32 with `saturating_mul` + `offset: total as u32`, so a >4 GiB workspace silently truncated slot offsets (corruption) and a >4 GiB tensor's byte length capped._ | codec round-trip + type widening | `hologram-archive/tests/conformance.rs::ov_codec_roundtrips_beyond_u32_without_truncation` | ✅ |

## NS — Portability

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **NS-1** | The library stack builds `no_std` on `wasm32-unknown-unknown` and `thumbv7em-none-eabi`. | cross-target build | `just wasm`, `just embedded` | ✅ |

## What conformance does and does not claim

It claims, with a traced artifact for each: the σ-axis and κ-labels are
BLAKE3-correct against the reference implementation (AS); model addresses
are spec-correct and replay-verifiable, inheriting uor-addr's cross-tool
validation (MA); every numeric kernel — matmul, softmax, layer/RMS norm,
gelu/silu, conv, attention, dequantize, reduce, pooling — matches its
ONNX-definition reference across scale incl. non-power-of-2 (KC/SC);
content-addressed execution is correct, reuses byte-identically, and
addresses outputs by **witnessed non-commutative composition** (CA);
every node is addressed by that composition so shared sub-graphs (cross-run
prefixes, intra-run common subexpressions) are recognized by κ-label and
their compute is elided (SG); composable sub-graphs (matmul→activation)
fuse into one κ-addressed op that elides the intermediate, semantics
preserved and guarded (FU); the compiled object carries the constant-only
cone's deterministic κ-label lattice and its materialized fold so the runtime
cache is never cold, with cross-process warming via a persisted κ-store —
warm ≡ cold byte-identically through the single residency mechanism (WS);
performance carries budgets on every heavy part with no silent bottleneck
(PV); and the libraries are `no_std`-portable (NS).

Every numbered invariant is enforced and passing. A possible future
*hardening* (beyond this contract, not a gap) is to additionally pin the
kernel references to the official ONNX node-test vectors + a cross-runtime
oracle — the same ONNX-op-semantics authority KC already validates
against, in a heavier form.
