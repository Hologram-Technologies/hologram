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
| **WL** | Weight-layout monomorphism — constant matmul weights are panel-packed at compile time into the kernel's contiguous-read layout (zero runtime copy), semantics-preserving | compiler/exec test (packing fires + packed==f64) |
| **EL** | Algebraic elision — computation UOR's algebra proves unnecessary (identity elements, involutions, relabels, dead nodes) is removed at compile time so it is never scheduled/dispatched, result-preserving | graph unit tests + exec result-equality tests |
| **CN** | κ-label canonicalization — commutative ops get operand-order-independent addresses so `a∘b`≡`b∘a` reuse each other's compute at runtime | exec reuse test (dispatch elision) |
| **AD** | Autodiff by composition — gradients are pipelines of forward ops (chain rule = composition); verified against finite differences | graph + exec grad-check tests |
| **PV** | Performance — every part within budget, no bottlenecks | benches with baselines/budgets |
| **PA** | Parallel execution — the lattice recursion leverages the whole processor (disjoint output tiles, one sequential recursion per core), observationally invisible | exec/backend tests (parallel≡sequential≡f64, pool concurrency) |
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
| **KC-2** | Softmax, LayerNorm, RMSNorm, **GroupNorm / InstanceNorm** (per-group normalization with per-channel affine; InstanceNorm = `num_groups == channels`), Gelu (ONNX `approximate="tanh"`), Silu, Conv2d (NCHW valid cross-correlation), Attention (scaled dot-product), DequantizeLinear each match their independent ONNX-definition f64 reference across scale (incl. non-power-of-2). _GroupNorm/InstanceNorm are realized as a true grouped kernel (`group_norm_float`, keyed on `num_groups`), not a LayerNorm stand-in._ | conformance tests vs f64 reference | `hologram-backend/tests/conformance.rs::kc3…kc9_*`, `::kc4b_group_norm_conforms_across_scale` | ✅ |
| **KC-3** | ReduceSum/Mean/Max + MaxPool/AveragePool (NCHW) match their ONNX-definition f64 reference across scale. | conformance tests vs f64 reference | `hologram-backend/tests/conformance.rs::kc10_reduce_*`, `::kc11_pooling_*` | ✅ |
| **KC-0** | _Supplementary internal cross-check_ (external conformance is KC-1/2/3): kernels also agree with hologram's Term-tree reference evaluator. | test | `hologram-backend/tests/kernel_equivalence.rs` | ✅ |
| **KC-4** | Low-precision (bf16/f16) matmul, conv2d, attention route through the **one** f32 engine (widen→engine→narrow) and match the f64 reference over the dtype-quantized operands. | conformance vs f64 reference | `conformance.rs::kc1b_low_precision_matmul_routes_through_engine`, `::kc7b_bf16_conv_routes_through_engine`, `::kc8b_bf16_attention_routes_through_engine` | ✅ |
| **KC-5** | dtype policy is single-sourced and never silently wrong: f64 is **rejected** (`UnsupportedOp`, not a 0-output), and div/mod by zero are IEEE-native (±∞/NaN, not 0). | conformance | `conformance.rs::kcdt_f64_rejected_never_silent_zero`, `::kcdt_div_mod_by_zero_is_ieee` | ✅ |
| **KC-6** | Every UOR-native layout / re-indexing / constructor / composite op equals its definitional reference: Reshape & Slice (`ProjectField`, zero-movement view — `last_skipped`), Pad (offset placement), Concat (`PrimitiveOp::Concat`), Transpose (n-d permute), Expand (broadcast), Resize (nearest), Clip (`Min∘Max`), SwiGLU (`MatMul·Silu·MatMul·Mul`), RoPE (rotate-half), Lrn (windowed-channel). | exec conformance + graph unit tests | `hologram-exec/tests/desugar.rs::*`, `hologram-graph/src/graph.rs::desugar_tests` | ✅ |

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
| **SC-5** | Softmax / LayerNorm / RMSNorm normalize over the last axis at **arbitrary rank**, not only rank-2: `feature` = last dim, `batch` = product of preceding dims. _Regression guard: the V&V exposed a `lower.rs` bug — the norm/softmax shape derivation fired only for rank-2 inputs, so the common rank-3 `[batch, seq, hidden]` transformer layout left `feature = 0` and the kernel silently emitted a zeroed output; now fixed._ | exec end-to-end test | `hologram-exec/tests/real_execution.rs::softmax_rank3_normalizes_over_last_axis` | ✅ |

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
> unit, not its pieces. The pass fuses **three** matmul epilogues — an
> elementwise activation (`MatMulActivation`, FU-1/2/3), the transformer
> residual add (`MatMulAdd`, FU-4), and the full `add → activation` chain
> (`MatMulAddActivation`, FU-5) — so an MLP layer's `act(A·B + bias)`
> collapses from three ops to one. Fusion is **guarded** —
> it fires only when the matmul's output has exactly one observer and is not
> a graph output (and, for the residual, when the skip tensor is ready no
> later than the matmul's level), so it never changes observable semantics.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **FU-1** | A fused `matmul → activation` (relu / gelu / silu) equals the independent f64 reference `activation(matmul(a,b))` across scale, **and the intermediate is elided** — the pair becomes one kernel (`fused_count == 1`, `kernel_count == 1`), one content-addressed op. | fusion conformance test (f64 ref + fused/kernel counts) | `hologram-exec/tests/conformance.rs::fu1_fused_matmul_activation_conforms_and_elides_intermediate` | ✅ |
| **FU-2** | Fusion is **semantics-preserving** — the fused result is byte-identical to the unfused computation — and **guarded**: a matmul whose output has a second observer is *not* fused (`fused_count == 0`), so the intermediate it still needs is preserved. | fused-vs-unfused equality + guard test | `hologram-exec/tests/conformance.rs::fu2_fusion_is_semantics_preserving_and_guarded` | ✅ |
| **FU-3** | Fusion fires on the production workload: a stacked transformer-MLP fuses one `matmul → gelu` per layer (the 4·d-wide activation intermediate is never materialized), reducing kernel count, while PV-4's throughput/scaling floors still hold. | production perf test (fused_count == layers) | `hologram-exec/tests/performance.rs::pv4_production_mlp_throughput_latency_and_scaling` | ✅ |
| **FU-4** | The transformer **residual** fuses too: `matmul → add(out, residual)` collapses into one `MatMulAdd` (residual added in the matmul epilogue), eliding the matmul intermediate *and* the separate bandwidth-bound add pass. Equals the independent f64 reference; **guarded** — a matmul whose output has a second observer is not fused; and the residual operand is only absorbed when it is ready no later than the matmul's schedule level (never observes a not-yet-computed tensor). The production MLP-stack fuses one per layer (12 → 8 kernels). | residual-fusion conformance test (f64 ref + count + guard) | `hologram-exec/tests/conformance.rs::fu4_residual_add_fuses_and_is_guarded` | ✅ |
| **FU-5** | The full MLP epilogue **`matmul → add → activation`** (`act(A·B + bias)`) collapses into one `MatMulAddActivation` — the matmul product, the post-add sum, *and* the activation intermediate are all elided (`add_activation_fused_count == 1`, `kernel_count == 1`). Equals the independent f64 reference `act(A·B + b)` for relu/gelu/silu; **guarded** — when the intermediate add has a second observer the activation is not absorbed and it degrades to a plain `MatMulAdd`. | three-op fusion conformance test (f64 ref + counts + guard) | `hologram-exec/tests/conformance.rs::fu5_matmul_add_activation_fuses_and_conforms` | ✅ |

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

## WL — Weight-layout monomorphism (efficiency via the data representation)

> hologram reaches kernel efficiency by **lattice recursion** — the matmul is
> a cache-oblivious recursive decomposition (`cpu::simd::matmul_f32_recursive`)
> whose subdivision blocks for every cache level with **no per-cache constant**
> (the term/shape-recursion analog at the kernel). The one remaining strided
> access — gathering the weight's columns — is removed by choosing the
> weight's **data representation** at compile time: a matmul's *constant*
> weight is panel-packed (`layout::pack_b_panels_bytes`) into exactly the order
> the leaf streams, baked into the archive. This is part of the single
> monomorphism the model compiles to: at runtime the kernel reads B
> contiguously with **zero copy** (the zero-cost/zero-copy contract holds —
> the recursion is pointer-only, the packing is compile-time). The packed
> weight is content-addressed by its bytes like any constant, so it composes
> with warm-start (WS) and fusion (FU: a packed `matmul→gelu` still fuses).

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **WL-1** | The compiler panel-packs a matmul's **constant** f32 weight (consumed by that matmul alone) into the kernel's contiguous layout — the compiled `MatMul` carries `b_packed` and the stored weight body is the packed extent — and the packed-weight result equals the **independent f64 reference** (semantics-preserving; incl. `n` not a multiple of the panel width). | compiler emit + decode check + f64 ref | `hologram-exec/tests/conformance.rs::wl1_constant_weight_is_panel_packed_and_conforms` | ✅ |
| **WL-0** | _Kernel-level_: the packed-panel matmul leaf equals a naïve product across odd shapes (panel padding, partial-column store, m-remainder). | unit test | `hologram-backend/src/cpu/simd.rs::tests::packed_b_matmul_matches_naive` | ✅ |

## EL — Algebraic elision (compute UOR proves unnecessary)

> UOR's algebra identifies *invariant facets* of a computation — values the
> result does not depend on computing. `Graph::elide_invariants` (run in
> `compile()` after desugaring, before scheduling) removes them so they are
> never scheduled, dispatched, or addressed: **identity elements** (`x+0`,
> `x−0`, `x·1`, `x/1`, `x¹` → `x`), **involutions** (`Neg∘Neg`, `Bnot∘Bnot`
> → `x`), **relabels** (a same-shape `Reshape` drops; `Reshape∘Reshape`
> collapses), and **dead nodes** (anything unreachable from a graph output;
> inputs are retained as the call ABI). Every rule is value-preserving within
> the runtime's accuracy contract — and only those: annihilators (`x·0→0`),
> reciprocal round-trips, and the softmax-sum-is-1 facet are **deliberately
> excluded** because they are *not* bit-exact under IEEE (`∞·0=NaN`,
> `1/(1/x)≠x`), so eliding them would silently change results.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **EL-1** | Each identity/involution/relabel rule collapses its node and redirects consumers to the surviving value; annihilators and non-bit-exact identities are preserved. | graph unit tests | `hologram-graph/src/graph.rs::elision_tests` | ✅ |
| **EL-2** | Dead nodes (unreachable from any output) are removed; graph inputs are retained. | graph unit test | `hologram-graph/src/graph.rs::elision_tests::dead_nodes_are_eliminated` | ✅ |
| **EL-3** | Elision is transparent end-to-end: an identity-padded graph executes to bit-identical output as its reduced form and compiles to the same node count. | exec result-equality test | `hologram-exec/tests/elision.rs` | ✅ |

## CN — Algebraic κ-label canonicalization (commutativity → runtime reuse)

> A commutative op's value is independent of operand order, so the executor
> sorts operand labels into a canonical order before deriving the content
> address (`KernelCall::is_commutative` gates it: Add/Mul/Min/Max/And/Or/
> Xor/Equal — never Sub/Div/Pow/comparisons). `a∘b` and `b∘a` then collapse to
> one κ-label, so the second is recognized as already-resident and its compute
> is **elided at runtime** (the resident-reuse path), and its boundary address
> is order-independent. This is UOR's algebra turned into reuse: the same
> mechanism as SG/WS, now reaching operand-reordered duplicates that an
> order-sensitive hash would miss. Cost is a 2–4 element sort on commutative
> nodes only; the served whole-graph memo-hit path skips the walk entirely, so
> there is no hot-path regression (content-reuse + production benches unchanged).

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CN-1** | `a+b` and `b+a` (reversed operands) derive one κ-label; the duplicate is elided at runtime (`last_skipped ≥ 1`), result preserved. | exec reuse test | `hologram-exec/tests/canonicalization.rs::commutative_reordering_dedups_at_runtime` | ✅ |
| **CN-2** | Non-commutative ops (Sub/Div/…) keep operand order — `a−b` and `b−a` stay distinct and both compute. | exec test | `hologram-exec/tests/canonicalization.rs::noncommutative_order_is_preserved` | ✅ |

## AD — Autodiff by composition (gradients are forward-op pipelines)

> Per the UOR framework, a gradient is a **composition** of the forward
> primitives a value decomposes into — the chain rule *is* categorical
> composition. `hologram_graph::append_backward` composes each op's
> vector-Jacobian product from existing forward ops (e.g.
> `dA = MatMul(g, Bᵀ)`; `σ'(x) = y·(1−y)`), summing where a value fans out.
> No new "backward kernel" exists — gradients run on the already-verified
> forward kernels, so there is no second silent-wrong surface — and an op
> whose VJP is not yet composed fails loud (`BackwardError::NoGradient`)
> rather than being approximated.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **AD-1** | Composed gradients match central finite differences (the derivative's definition) across the **entire differentiable catalog**: arithmetic (Add/Sub/Mul/Div/Neg); all elementwise activations (Sigmoid/Tanh/Relu/Silu/Gelu/Elu/Selu/Exp/Log/**Log1p**/Sqrt/Reciprocal/Erf/Abs/Sin/Cos/Tan/**Asin/Acos/Atan**); **Ceil/Floor/Round** (0 a.e.); Min/Max/Pow; **Mod** (floored: `dy/db=−floor(a/b)`); the comparison/`IsNaN` predicates (0 a.e.); **Where**; Reshape/Transpose; 2-D MatMul; **Gemm**; reductions (Sum/Mean/Prod/Min/Max) — **full *and* over an arbitrary `axes` subset** — and **CumSum**; Softmax/LogSoftmax; LayerNorm/RmsNorm/GroupNorm/InstanceNorm/**AddRmsNorm** (incl. the **dγ/dβ** scale/shift-parameter gradients, not just dx); Expand; GlobalAvgPool/AvgPool2d/MaxPool2d; Concat/Slice/Pad; **Resize** (nearest); **Lrn**; **RotaryEmbedding**; Attention; Conv2d/**ConvTranspose2d**. Mechanisms (all from forward ops, no backward kernel): softmax/norm row-mean = matmul-ones; axis-reduction grads broadcast back via reshape-to-keepdims→Expand; dγ/dβ = ReduceSum(g⊙x̂)/ReduceSum(g) over the broadcast axes; Expand grad = ReduceSum(g) over the broadcast axes (any rank, multi-axis); pool grads = Reshape→Expand→Reshape upsample; LayerNorm/RmsNorm use the last-axis VJP **at any rank** (matching the rank-general forward), GroupNorm/InstanceNorm a true **per-group** VJP (group-mean broadcast + per-channel γ broadcast, matching the grouped forward); Attention unrolls per (batch,head) into rank-2 VJPs; Conv = `dW=Σ gᵦ·im2col(xᵦ)ᵀ`, `dX=Σ col2im(Wᵀ·gᵦ)`; CumSum = `total(g)−cumsum(g)+g`; Resize = `Sᵀ·g` (selection matrix); Lrn = windowed-channel band-matrix matmuls; RoPE = `g⊙cos − RoPE(g⊙sin,0,1)`; Where reuses the Where kernel to mask. | exec grad-check | `hologram-exec/tests/autodiff.rs` | ✅ |
| **AD-2** | `append_backward` emits only forward ops (composites desugar first; no `*Grad` op-kind). The forward primitives a VJP needs but the catalog lacked are added as honest **forward** ops — `Im2Col` (receptive-field gather) and `Col2Im` (its accumulating adjoint) — never backward kernels. Every op with a real-valued input now has a VJP; only the **discrete byte-algebra** ops (`And/Or/Xor/Bnot/Succ/Pred`) and `Dequantize` (integer source domain) fail loud (`BackwardError::NoGradient`) — they have no calculus derivative, which is a correctness statement, not a deferral. | graph/compiler tests | `hologram-compiler/tests/backward_emission.rs` | ✅ |
| **AD-4** | _Forward fix surfaced by AD V&V_: `Gemm` lowered with α=β=0 (so `Y = α·A·B + β·C` computed **zero**); now carries `GemmAttrs` defaulting to α=β=1 (the plain `A·B + C`), and the `Lrn` kernel's `f32::powf` → `libm::powf` so the no_std/bare-metal build links. | exec grad-check + cross-compile | `hologram-exec/tests/autodiff.rs::gemm_*` ; `just embedded` | ✅ |
| **AD-3** | _Forward fix surfaced by AD V&V_: a compiled full-reduction kernel folds over its **input** element count (was the scalar output count → only element 0 summed). | exec grad-check (reductions) | `hologram-exec/tests/autodiff.rs::reduction_gradients_match_finite_difference` | ✅ |

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
| **PV-1c** | **Cache-hierarchy V&V (machine-independent).** The cache-oblivious recursion keeps per-FLOP throughput from collapsing as the working set grows past each cache level (compulsory-only misses): 512³ (~3 MiB operands ≫ L2) retains ≥60% of 128³'s (L2-resident) GFLOP/s — a cache-blind kernel cliffs far below; the recursion holds ~90%. Verifies the UOR hierarchical model maximizes cache hits across workloads, without hardware counters. Release-only. | release perf test | `hologram-backend/tests/performance.rs::pv1c_cache_oblivious_efficiency_holds_across_scale` | ✅ |
| **PV-2** | Content-addressed reuse (graph-memo hit) is ≥8× faster than recompute — a same-machine ratio proving the reuse path is O(1)-ish and not secretly recomputing. Release-only. | release perf test | `hologram-exec/tests/performance.rs::pv2_content_addressed_reuse_beats_recompute` | ✅ |
| **PV-3** | The other heavy compute kernels (Conv2d, Attention) each carry a conservative throughput floor (best-of-N) so no part is a silent bottleneck. Release-only. | release perf test | `hologram-backend/tests/performance.rs::pv3_conv_and_attention_throughput_floors` | ✅ |
| **PV-4** | A **production-representative** workload — a stacked transformer-MLP (`matmul → gelu → matmul → residual` per layer) — sustains a throughput floor under cold (all-novel) load, does **not break down across sizes** (d=256 keeps ≥¼ of d=128's per-FLOP throughput, so arbitrary sizes hold), and under serving reuse collapses to a whole-graph memo hit ≥8× faster than cold. Reports GFLOP/s, FLOP/core-cycle, and ms/inference. Release-only; mirrored by the `production` criterion bench. | release perf test + bench | `hologram-exec/tests/performance.rs::pv4_production_mlp_throughput_latency_and_scaling`, `hologram-bench/benches/production.rs` | ✅ |
| **PV-Z** | **Zero-overhead / zero-copy contract (executable).** After warm-up, matmul / gemm / conv2d / attention perform **0 heap allocations per call** — the cache-oblivious engine and every dtype-widen / im2col / score buffer is a reused thread-local, so an inference loop pays O(1) allocations, not O(calls). A per-thread counting allocator catches any reintroduced per-call `Vec`/copy. | allocation-counting test | `hologram-backend/tests/zero_overhead.rs` | ✅ |

## PA — Parallel execution (the whole processor surface)

> Parallelism is the **same lattice recursion**, read as a task DAG: an M/N
> split of the cache-oblivious matmul yields disjoint-output, independent
> sub-products (lattice nodes). The pool (`cpu::parallel`, `std`-only, behind
> `--features parallel`) cuts the recursion at the parallel grain — ≈one tile
> per core (`output_tiles`) — and each tile runs the *sequential* recursion on
> its **private L2**, so per-core cache locality is preserved (the hierarchy
> compounds: each tile's working set fits one core's cache). Small ops stay
> single-core (below `PAR_THRESHOLD`) so the single-core path stays optimal.
> The feature is off by default (single-core unaffected); on, a single matmul
> `dispatch` uses every core with no workspace splitting.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **PA-1** | Multi-core execution is **observationally invisible**: matmul (row-major and packed) run across the pool is byte-equal to the f64 reference and **deterministic** across repeated runs (the determinism loop is the data-race stress — disjoint output tiles must never alias). Holds for the sequential path too (feature off). | parallel conformance test (f64 ref + determinism) | `hologram-backend/tests/parallel.rs::pa1_parallel_matmul_matches_reference_and_is_deterministic` | ✅ |
| **PA-2** | The pool **actually runs tasks concurrently** — `width` independent spins complete in `< 0.6×` their serial sum (a fully-serial drain == serial fails decisively). Regression guard for the `while let Some(t)=lock().pop_front()` footgun that held the queue lock across `t()` and silently serialized the pool. `output_tiles` exactly partitions the output (race-freedom witness). | pool concurrency + tile-partition unit tests | `hologram-backend/src/cpu/parallel.rs::{pool_diag::run_distributes_across_workers_concurrently, tests::output_tiles_partition_exactly_and_align}` | ✅ |

## OV — No fixed byte ceiling (ADR-060: operations don't overflow at scale)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **OV-1** | Byte offsets/lengths (`BufferRef`, `SlotSpan`) and element counts (`PortDescriptor`, kernel calls, compiler `element_counts`/`byte_lengths`) are **u64** — no 4 GiB / 4.29 B-element cap, no saturating/truncating size arithmetic. Values beyond `u32::MAX` survive the archive codec round-trip without truncation. _Regression fix: these were u32 with `saturating_mul` + `offset: total as u32`, so a >4 GiB workspace silently truncated slot offsets (corruption) and a >4 GiB tensor's byte length capped._ | codec round-trip + type widening | `hologram-archive/tests/conformance.rs::ov_codec_roundtrips_beyond_u32_without_truncation` | ✅ |
| **OV-2** | The host boundary never reports a truncated transfer as success: the C-ABI compile entry points return the **full** archive length (snprintf-style, so `ret > capacity` ⇒ caller retries larger), and `hologram_session_execute` **fails loud** (-1) on an undersized output buffer instead of silently writing a partial result. Callers size buffers from `input_byte_len`/`output_byte_len` (the archive's declared shape × dtype), not a fixed cap. _Regression fix: the FFI copied `min(len, capacity)` and returned success; the CLI fed a hard-coded 4096-byte input buffer; an unknown `--backend` silently downgraded to CPU._ | C-ABI integration tests | `hologram-ffi/tests/c_abi.rs::{compile_signals_truncation_via_required_length, execute_fails_loud_on_undersized_output_buffer}` | ✅ |

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
their compute is elided (SG); composable sub-graphs (matmul→activation and
matmul→residual-add) fuse into one κ-addressed op that elides the
intermediate, semantics preserved and guarded (FU); the compiled object carries the constant-only
cone's deterministic κ-label lattice and its materialized fold so the runtime
cache is never cold, with cross-process warming via a persisted κ-store —
warm ≡ cold byte-identically through the single residency mechanism (WS);
performance carries budgets on every heavy part with no silent bottleneck
(PV); and the libraries are `no_std`-portable (NS).

Every numbered invariant is enforced and passing — there are no deferred
items. KC validates every kernel against the **ONNX operator definition**
(the authority) via an independent f64 reference across scale, including
non-power-of-2 and rectangular shapes; the official ONNX node-test corpus is
that *same* authority repackaged as fixed vectors, so reproducing it would be
a redundancy, not a remaining conformance obligation. Performance is
structural, not hand-tuned: the matmul is a cache-oblivious recursion (no
per-cache block constant) over a compile-time panel-packed weight layout
(zero runtime copy), with activation and residual epilogues fused — so
efficiency holds across scale by construction, the same way the runtime's
addressing and warm-start do.
