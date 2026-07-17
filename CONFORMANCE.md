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
| **RF** | Refinement execution is bounded, deterministic, and label-addressed | exec refinement tests |
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
| **CC** | Component conformance (holospaces) — each component vs its external authority (hash KATs, native-executor oracle, substrate TCK, WebAssembly/VirtIO/OCI/Dev-Container specs, QEMU, Playwright) — spec 06 §V&V (MG-7) | conformance tests (`spaces/holospaces/tests/cc*.rs`) + CC bijection audit |
| **CS** | Specification conformance (holospaces docs) — the documentation vs arc42 / C4 / OPM ISO 19450 / ISO 15288, via validators V1–V8 — spec 06 §docs (MG-8) | validator scripts (`specs/holospaces/scripts/v*-*`) |
| **LAW** | Repo-wide laws (SPINE-1..6, κ-only identity, capability attenuation, async/sync, one surface) — refactor spec 00 | BDD scenarios (features/suites/s0_laws) |
| **SP** | Space contract trait set + laws + TCK battery; external-repo parity (D21) — spec 02 | BDD scenarios (s1_space_contract) |
| **HF** | `.holo` v3 container, attenuated nesting, per-layer certificates — spec 03 | BDD scenarios (s2_holo_format) |
| **NW** | Network κ-realization, KappaSync/DHT, public/restricted/private tiers — spec 04 | BDD scenarios (s3_networks) |
| **TL** | One binary, one public facade crate, FFI over Client — spec 05 | BDD scenarios (s4_tooling) |
| **MG** | Phased always-green migration gates (P0–P6) — spec 06 | BDD scenarios (s5_migration) |
| **GV** | Governance R1–R4 boundary rules — spec 07 | BDD scenarios (s6_governance) |

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
| **KC-1** | f32 matmul equals an **independent f64-reference** product (the IEEE-754 definitional ground truth, à la BLAS/NumPy validation — not hologram's own evaluator) within `√k·ε_f32`, and reads all operands (no short-cut). | conformance test vs f64 reference | `hologram-compute/tests/conformance.rs::kc1_sc1_matmul_conforms_across_scale`, `::kc2_matmul_reads_all_operands_no_shortcut` | ✅ |
| **KC-2** | Softmax, LayerNorm, RMSNorm, **GroupNorm / InstanceNorm** (per-group normalization with per-channel affine; InstanceNorm = `num_groups == channels`), Gelu (ONNX `approximate="tanh"`), Silu, Conv2d (NCHW valid cross-correlation), Attention (scaled dot-product), DequantizeLinear (**per-tensor and per-axis/per-channel** scale + zero-point) each match their independent ONNX-definition f64 reference across scale (incl. non-power-of-2). _GroupNorm/InstanceNorm are realized as a true grouped kernel (`group_norm_float`, keyed on `num_groups`), not a LayerNorm stand-in._ | conformance tests vs f64 reference | `hologram-compute/tests/conformance.rs::kc3…kc9_*`, `::kc4b_group_norm_conforms_across_scale`, `::kc9b_per_channel_dequantize_conforms` | ✅ |
| **KC-3** | ReduceSum/Mean/Max + MaxPool/AveragePool (NCHW) match their ONNX-definition f64 reference across scale. | conformance tests vs f64 reference | `hologram-compute/tests/conformance.rs::kc10_reduce_*`, `::kc11_pooling_*` | ✅ |
| **KC-0** | _Supplementary internal cross-check_ (external conformance is KC-1/2/3): kernels also agree with hologram's Term-tree reference evaluator. | test | `hologram-compute/tests/kernel_equivalence.rs` | ✅ |
| **KC-4** | Low-precision (bf16/f16) matmul, conv2d, attention are computed by the f32 engine and match the f64 reference over the dtype-quantized operands. Large `m` widens the weight into the engine; decode shapes take the streamed `matmul_lowp_gemv`, which widens in-register — bit-identical, never a scalar fallback on a first-class target. | conformance vs f64 reference | `conformance.rs::kc1b_low_precision_matmul_routes_through_engine`, `::kc7b_bf16_conv_routes_through_engine`, `::kc8b_bf16_attention_routes_through_engine` | ✅ |
| **KC-5** | dtype policy is single-sourced and never silently wrong: f64 is **rejected** (`UnsupportedOp`, not a 0-output), and div/mod by zero are IEEE-native (±∞/NaN, not 0). | conformance | `conformance.rs::kcdt_f64_rejected_never_silent_zero`, `::kcdt_div_mod_by_zero_is_ieee` | ✅ |
| **KC-6** | Every UOR-native layout / re-indexing / constructor / composite op equals its definitional reference: Reshape & Slice (`ProjectField`, zero-movement view — `last_skipped`), Pad (offset placement), Concat (`PrimitiveOp::Concat`), Transpose (n-d permute), Expand (broadcast), Resize (nearest), Clip (`Min∘Max`), SwiGLU (`MatMul·Silu·MatMul·Mul`), RoPE (rotate-half), Lrn (windowed-channel). | exec conformance + graph unit tests | `hologram-exec/tests/desugar.rs::*`, `hologram-graph/src/graph.rs::desugar_tests` | ✅ |

## SC — Scaling (no short-cut / no breakdown at arbitrary size)

> The V&V must *demonstrate* hologram holds at arbitrary scale — that it
> never silently degrades to a wrong/partial result or breaks down
> (tail-handling, precision, overflow) as sizes grow. Mirrors prism-btc's
> CM (mainnet-readiness) / CP (scaling-across-decades) classes.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **SC-1** | f32 matmul stays conformant to the f64 reference across `(m,k,n)` from 2³ through 512×64×384, **including non-power-of-2 and rectangular shapes** (which expose tail/blocking bugs); normalized error stays `≪ 1e-4`, not diverging with size. | scale-parameterized conformance test | `hologram-compute/tests/conformance.rs::kc1_sc1_matmul_conforms_across_scale` | ✅ |
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

## RF — Refinement execution

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **RF-1** | A refinement plan is a bounded execution strategy over an addressed session, not a graph op or backend kernel. It validates an explicit or derived state contract before execution and feeds pass outputs back as next-pass κ-label inputs. | exec integration tests | `hologram-exec/tests/refinement.rs::successful_label_convergence_on_identity_graph`, `::successful_byte_convergence_after_two_relu_passes`, `::explicit_state_contract_matches_session`, `::explicit_state_contract_rejects_shape_mismatch` | ✅ |
| **RF-2** | Validation failure and repair are terminally bounded: normal pass exhaustion reports `PassBoundReached`, and retry repair reports `Repaired(..)` or `RepairBoundReached` without an unbounded loop. | exec integration tests | `…::validation_failure_returns_pass_bound_status`, `…::repair_flow_uses_bounded_retry_pass` | ✅ |
| **RF-3** | Refinement is deterministic and does not mutate the loaded graph schedule or grow resident memory without bound across repeated runs. | exec integration tests | `…::planner_generated_plan_runs_deterministically`, `…::refinement_does_not_mutate_loaded_graph_schedule`, `…::repeated_refinement_keeps_resident_memory_bounded` | ✅ |

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
| **FU-6** | A **`dequantize → matmul`** on a quantized weight collapses into one `MatMulDequant` (`dequant_fused_count == 1`): the weight is dequantized into a transient panel inside the matmul, so the **dense f32 weight is never materialized** in the pool. Equals the unfused dequantize-then-matmul; **single-consumer guarded** (the dequant output must feed only the matmul's B). Constant quantized weights are warm-start-folded (no runtime dequant), so the fusion targets dynamic quantized operands. The matmul perf floors (PV-1) still hold. | fused-dequant exec test (fires + equals unfused) + perf floors | `hologram-exec/tests/quantization.rs::dequant_matmul_fuses_and_matches_unfused` | ✅ |
| **FU-7** | An **`expand → elementwise-binary`** (`Add`/`Sub`/`Mul`) collapses into one `BroadcastBinary` (`broadcast_binary_fused_count == 1`) that reads the pre-Expand operand with **stride-0 broadcast indexing in place** — the materialized broadcast tensor is never produced (the zero-movement realization of Expand for its dominant consumer; e.g. the norm-VJP `Expand → Mul` and bias/scale broadcasts). Equals the explicit broadcast-then-binary; **single-consumer guarded**; float-domain. | fused-broadcast exec test (fires + equals unfused) | `hologram-exec/tests/real_execution.rs::expand_binary_fuses_and_elides_broadcast` | ✅ |

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
| **WL-0** | _Kernel-level_: the packed-panel matmul leaf equals a naïve product across odd shapes (panel padding, partial-column store, m-remainder). | unit test | `hologram-compute/src/cpu/simd.rs::tests::packed_b_matmul_matches_naive` | ✅ |

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
| **PV-1** | f32 matmul clears a conservative throughput floor (≥1 GFLOP/s at 256³, best-of-N) and stays within the cubic scaling envelope (256³/64³ time ∈ [16,400]) — catching scalar-fallback / super-cubic / degenerate-short-cut bottlenecks. Release-only. | release perf test | `hologram-compute/tests/performance.rs::pv1_matmul_throughput_floor_and_scaling` | ✅ |
| **PV-1c** | **Cache-hierarchy V&V (machine-independent).** The cache-oblivious recursion keeps per-FLOP throughput from collapsing as the working set grows past each cache level (compulsory-only misses): 512³ (~3 MiB operands ≫ L2) retains ≥60% of 128³'s (L2-resident) GFLOP/s — a cache-blind kernel cliffs far below; the recursion holds ~90%. Verifies the UOR hierarchical model maximizes cache hits across workloads, without hardware counters. Release-only. | release perf test | `hologram-compute/tests/performance.rs::pv1c_cache_oblivious_efficiency_holds_across_scale` | ✅ |
| **PV-2** | Content-addressed reuse (graph-memo hit) is ≥8× faster than recompute — a same-machine ratio proving the reuse path is O(1)-ish and not secretly recomputing. Release-only. | release perf test | `hologram-exec/tests/performance.rs::pv2_content_addressed_reuse_beats_recompute` | ✅ |
| **PV-3** | The other heavy compute kernels (Conv2d, Attention) each carry a conservative throughput floor (best-of-N) so no part is a silent bottleneck. Release-only. | release perf test | `hologram-compute/tests/performance.rs::pv3_conv_and_attention_throughput_floors` | ✅ |
| **PV-4** | A **production-representative** workload — a stacked transformer-MLP (`matmul → gelu → matmul → residual` per layer) — sustains a throughput floor under cold (all-novel) load, does **not break down across sizes** (d=256 keeps ≥¼ of d=128's per-FLOP throughput, so arbitrary sizes hold), and under serving reuse collapses to a whole-graph memo hit ≥8× faster than cold. Reports GFLOP/s, FLOP/core-cycle, and ms/inference. Release-only; mirrored by the `production` criterion bench. | release perf test + bench | `hologram-exec/tests/performance.rs::pv4_production_mlp_throughput_latency_and_scaling`, `hologram-bench/benches/production.rs` | ✅ |
| **PV-Z** | **Zero-overhead / zero-copy contract (executable).** After warm-up, matmul / gemm / conv2d / attention perform **0 heap allocations per call** — the cache-oblivious engine and every dtype-widen / im2col / score buffer is a reused thread-local, so an inference loop pays O(1) allocations, not O(calls). A per-thread counting allocator catches any reintroduced per-call `Vec`/copy. | allocation-counting test | `hologram-compute/tests/zero_overhead.rs` | ✅ |

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
| **PA-1** | Multi-core execution is **observationally invisible**: matmul (row-major and packed) run across the pool is byte-equal to the f64 reference and **deterministic** across repeated runs (the determinism loop is the data-race stress — disjoint output tiles must never alias). Holds for the sequential path too (feature off). | parallel conformance test (f64 ref + determinism) | `hologram-compute/tests/parallel.rs::pa1_parallel_matmul_matches_reference_and_is_deterministic` | ✅ |
| **PA-2** | The pool **actually runs tasks concurrently** — `width` independent spins complete in `< 0.6×` their serial sum (a fully-serial drain == serial fails decisively). Regression guard for the `while let Some(t)=lock().pop_front()` footgun that held the queue lock across `t()` and silently serialized the pool. `output_tiles` exactly partitions the output (race-freedom witness). | pool concurrency + tile-partition unit tests | `hologram-compute/src/cpu/parallel.rs::{pool_diag::run_distributes_across_workers_concurrently, tests::output_tiles_partition_exactly_and_align}` | ✅ |

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

## LAW — repo-wide laws (refactor spec 00; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **LAW-0** | Harness smoke: the conformance runner discovers and executes feature files. | BDD scenario | `s0_laws/_smoke.feature::the harness discovers and runs feature files` | ✅ |
| **LAW-1** | SPINE-1: a realization with no canonical bytes is unrepresentable; identity is verified by re-derivation, never trusted. | BDD scenario (witnessed against `hologram-substrate-core::verify_kappa` + a `ContainerManifest`) | `s0_laws/spine.feature::canonical bytes or nothing` | ✅ |
| **LAW-2** | κ-only identity: no contract or stored form exposes a UUID / PeerId / Multiaddr / path / hostname as identity; transport ids never leak. | BDD scenario | `s0_laws/identity.feature::no second naming surface` | ⛔ |
| **LAW-3** | Contracts are hologram's, spaces are anyone's: the space contract has no sealed traits or crate-private seams; a space may live in any repository (D2/D21). | BDD scenario (witnessed by `hologram-spike-sp3` — a separate crate — implementing the contract and being accepted by `Client`) | `s0_laws/open_contract.feature::the space contract is open to any repo` | ✅ |
| **LAW-4** | Sync storage + compute, async network/lifecycle: `KappaStore` and the tensor hot path are synchronous (sync OPFS in a Worker → wasm-safe); network sync + lifecycle are async; the async↔sync seam is the network/boot boundary, never storage; Send-bound is maybe-Send (D14; P0.5 spike). | BDD scenario | `s0_laws/async_sync_seam.feature::the session boundary is the only async-sync seam` | ⛔ |
| **LAW-5** | Capability attenuation only: a delegated capability is always a subset of the grantor's; amplification is unrepresentable. | BDD scenario | `s0_laws/attenuation.feature::delegation cannot amplify` | ⛔ |
| **LAW-6** | One programmatic surface: CLI / FFI / SDK are thin shells over the `Client` facade; behavior lives in exactly one place. | BDD scenario | `s0_laws/one_surface.feature::entry points are thin shells` | ⛔ |

## SP — space contract + TCK (refactor spec 02; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **SP-1** | Every space implements the identical contract surface; passing `hologram-tck` is the definition of conformance. | BDD scenario (witnessed against the reference `MemKappaStore` via the shared `hologram-tck` battery) | `s1_space_contract/tck.feature::passing the TCK is conformance` | ✅ |
| **SP-2** | An external-repo space passes the TCK as a dev-dependency and is accepted by `Client` with no facade change (D21). | BDD scenario | `s1_space_contract/external_parity.feature::external space is first-class` | ⛔ |
| **SP-3** | A `Space` composes a synchronous store + sync compute with an async network/boot seam; `Client` drives compile→store→boot end to end through the one async↔sync boundary (D14/D28; witnessed by `hologram-spike-sp3`). | BDD scenario | `s1_space_contract/composition.feature::a space composes async network with sync storage and compute` | ✅ |
| **SP-4** | The reference HAL seams (`Entropy`/`Clock`/`Spawner`, spec 02 §4) are hermetic and deterministic — equally-seeded entropy reproduces the same stream, the clock advances only when told, and the background spawner is inert — so V&V is reproducible. | BDD scenario (witnessed against `hologram-space`'s `SeededEntropy`/`ManualClock`/`NoopSpawner`) | `s1_space_contract/hal_seams.feature::the reference HAL seams are hermetic and deterministic` | ✅ |
| **SP-5** | Headless is a first-class conformance profile (spec 02 §5): a space with no display satisfies `Surface` via the null projection — `project` yields the canonical empty-projection κ and `intent` refuses with a typed headless error. | BDD scenario (witnessed against `hologram-space`'s `NullSurface`) | `s1_space_contract/surface_headless.feature::a headless space satisfies the Surface contract via the null projection` | ✅ |

## HF — .holo v3 format (refactor spec 03; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **HF-1** | `.holo` v3 is the one application container; a tensor-only archive is the degenerate single-layer case. | BDD scenario | `s2_holo_format/container.feature::single format covers tensor-only` | ✅ |
| **HF-2** | App nesting is capability-attenuated: a child's κ refs + delegated CapabilitySet are a subset of the parent's. | BDD scenario | `s2_holo_format/nesting.feature::nested app cannot exceed parent` | ✅ |
| **HF-3** | v3 per-layer certificates verify; inspection APIs never strip them. | BDD scenario | `s2_holo_format/certificates.feature::per-layer certificates verify` | ⛔ |

## NW — networks (refactor spec 04; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **NW-1** | A Network is a κ-realization embedding its membership + policy operand κs (SPINE-2/3); no side tables. | BDD scenario | `s3_networks/realization.feature::network embeds operand κs` | ⛔ |
| **NW-2** | Network tiers (public / restricted / private) gate capability at the protocol boundary, never in business logic. | BDD scenario | `s3_networks/tiers.feature::tiers gate at the boundary` | ⛔ |

## TL — tooling (refactor spec 05; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **TL-1** | Exactly one binary named `hologram` ships. | BDD scenario | `s4_tooling/one_binary.feature::exactly one binary` | ⛔ |
| **TL-2** | Exactly one public crate (`hologram`) is imported with features; users never import subcrates. | BDD scenario | `s4_tooling/one_facade.feature::one public crate` | ⛔ |
| **TL-3** | Leaf tier (D22): dependencies flow core → spaces → leaf {facade+Client, cli, packaging}; nothing depends on a leaf crate. | BDD scenario | `s4_tooling/leaf_tier.feature::nothing depends on a leaf crate` | ⛔ |
| **TL-4** | Deploy is `put` + `announce` (+ `--page`): `hologram app publish` makes an app reachable and the same κ resolves and runs across every access rung (D25, spec 08). | BDD scenario | `s4_tooling/deploy.feature::one app publishes to every rung by κ` | ⛔ |

## MG — migration gates (refactor spec 06; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **MG-1** | Every phase boundary P0–P6 is always-green: the full holospaces V&V passes before the next phase starts. | BDD scenario | `s5_migration/always_green.feature::each phase boundary is green` | ⛔ |
| **MG-2** | P0 sync exit criteria (D23) are met before any refactor move: holospaces ports to hologram HEAD, V&V green, bridge tag cut. | BDD scenario | `s5_migration/p0_sync.feature::p0 exit criteria met` | ⛔ |
| **MG-3** | P0.5 de-risk spike (D28): the Space+Client vertical slice compiles and runs on native AND wasm32, resolving the Send-bound question, before any P1 move. | BDD scenario | `s5_migration/p05_spike.feature::the de-risk spike proves composition before P1` | ⛔ |
| **MG-4** | Perf gate (D27): hologram-bench roofline/kernel baselines are captured at P1 preflight and re-run each release; a regression past threshold blocks the release. | BDD scenario | `s5_migration/perf_gate.feature::perf regression blocks a release` | ⛔ |
| **MG-5** | κ-stability (ground rule 5): golden vectors re-derive bit-identically across every crate move; a κ break is a versioned format change, never a move. | BDD scenario (frozen σ-axis + realization κs re-derived vs `hologram-substrate-core`/`-realizations`) | `s5_migration/kappa_stability.feature::golden vectors re-derive bit-identically across moves` | ✅ |
| **MG-6** | P0 gate (D24/D29): written MIT→dual relicense consent and a holospaces-restructuring spec review are recorded before any code moves. | BDD scenario | `s5_migration/p0_license_review.feature::license consent and restructuring review precede any move` | ⛔ |
| **MG-7** | holospaces' V&V is absorbed into hologram's unified conformance ledger (spec 06): its component-conformance (CC) catalog + spec-conformance (CS) suites run under the one meta-gate, each witnessed against its external authority (hash KATs, the native-executor oracle, the substrate TCK, QEMU, Playwright) and never by self-reference; the `vv/` artifacts are content-addressed and verified on import. | BDD scenario | `s5_migration/vv_absorption.feature::the holospaces CC catalog is absorbed into the unified conformance ledger` | ✅ |
| **MG-8** | holospaces' specification conformance (CS) is absorbed into the unified ledger (spec 06 §docs): the docs V&V (validators V1–V8) runs under the one framework, each CS row witnessed against its external standard (arc42 / C4 / OPM ISO 19450 / ISO 15288) and never by self-reference; the toolchain + pins are content-addressed on import. | BDD scenario | `s5_migration/cs_absorption.feature::the holospaces CS catalog is absorbed into the unified conformance ledger` | ✅ |

## GV — governance requirements (refactor spec 07; BDD)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **GV-1** | R1 traceability: every new realization embeds its operand κs so `references()` yields the full provenance closure — no side tables. | BDD scenario (witnessed against `hologram-realizations::ContainerManifest`) | `s6_governance/traceability.feature::references yields full provenance` | ✅ |
| **GV-2** | R2 auditability: lifecycle transitions emit through one seam that can be pointed at the κ-chain; no lifecycle path bypasses it. | BDD scenario | `s6_governance/auditability.feature::one audit seam, no bypass` | ⛔ |
| **GV-3** | R3 attestation: signing keys are bound to κ-addressed identities as published content; certificates are never a second identity surface. | BDD scenario | `s6_governance/attestation.feature::keys bind to κ-identity` | ⛔ |
| **GV-4** | R4 data governance: capability checks stay at the import/protocol boundary; resource accounting is per-capability, not global. | BDD scenario | `s6_governance/data_governance.feature::capability checks at the boundary` | ⛔ |

## CC — component conformance (holospaces V&V; external: per-row authority; cargo-witnessed, non-BDD)

Absorbed from holospaces' V&V (spec 06 §V&V; MG-7). Each row is a component validated against
its own external authority (hash KATs, the native-executor oracle, the substrate TCK, the
WebAssembly/VirtIO/OCI/Dev-Container specs, QEMU, Playwright), witnessed by a cargo test in the
ported `holospaces` space — never by self-reference. Like the other non-BDD classes (AS/KC/…) the
honesty meta-gate does not bind these to Gherkin; instead the **CC bijection audit**
(`crates/hologram-conformance/tests/cc_gate.rs`) binds every row to a present witness, and **MG-7**
witnesses the absorption. Enforcement tiers: `test` (fast, no artifact), `test (artifact: …)`, and
`test (heavy: QEMU/browser · #[ignore])`. 🟡 = present + audit-bound; ✅ once CI gates it green.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CC-1** | kappa kat — kappa digest equals reference implementation | test | `spaces/holospaces/tests/cc1_kappa_kat.rs::kappa_digest_equals_reference_implementation` | ✅ |
| **CC-2** | holo engine — identical holo yields identical kappa across independent builds | test | `spaces/holospaces/tests/cc2_holo_engine.rs::identical_holo_yields_identical_kappa_across_independent_builds` | ✅ |
| **CC-3** | substrate tck — in memory store obeys the substrate contract | test | `spaces/holospaces/tests/cc3_substrate_tck.rs::in_memory_store_obeys_the_substrate_contract` | ✅ |
| **CC-4** | devcontainer — real configs conform to the dev container schema | test (artifact: vv/artifacts/cc4) | `spaces/holospaces/tests/cc4_devcontainer.rs::real_configs_conform_to_the_dev_container_schema` | ✅ |
| **CC-5** | wasm — validator agrees with the webassembly spec suite | test (artifact: vv/artifacts/cc5) | `spaces/holospaces/tests/cc5_wasm.rs::validator_agrees_with_the_webassembly_spec_suite` | ✅ |
| **CC-6** | execution surface — surface validator enforces the contract | test | `spaces/holospaces/tests/cc6_execution_surface.rs::surface_validator_enforces_the_contract` | ✅ |
| **CC-7** | kdisk — the ext4 artifact matches its recorded digest | test (artifact: vv/artifacts/cc7) | `spaces/holospaces/tests/cc7_kdisk.rs::the_ext4_artifact_matches_its_recorded_digest` | ✅ |
| **CC-8** | import run — a forged import is refused by re derivation | test | `spaces/holospaces/tests/cc8_import_run.rs::a_forged_import_is_refused_by_re_derivation` | ✅ |
| **CC-9** | emulator — the emulator core conforms to the risc v isa | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc9_emulator.rs::the_emulator_core_conforms_to_the_risc_v_isa` | ✅ |
| **CC-10** | ingestion — a real oci image ingests as verified kappa content | test (artifact: vv/artifacts/cc10) | `spaces/holospaces/tests/cc10_ingestion.rs::a_real_oci_image_ingests_as_verified_kappa_content` | ✅ |
| **CC-11.term** | raw terminal — the deployed terminal echoes edits and handles ctrl c | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc11_raw_terminal.rs::the_deployed_terminal_echoes_edits_and_handles_ctrl_c` | ✅ |
| **CC-11** | workspace — intent edits are content addressed in the store | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc11_workspace.rs::intent_edits_are_content_addressed_in_the_store` | ✅ |
| **CC-12** | manager console — the console signs in a self sovereign content addressed identity | test | `spaces/holospaces/tests/cc12_manager_console.rs::the_console_signs_in_a_self_sovereign_content_addressed_identity` | ✅ |
| **CC-13** | vscode workspace — the vscode components re derive to their pinned kappa | test (artifact: vv/artifacts/cc13) | `spaces/holospaces/tests/cc13_vscode_workspace.rs::the_vscode_components_re_derive_to_their_pinned_kappa` | ✅ |
| **CC-14.asm** | assembly — the oci layers assemble into a clean mountable ext4 rootfs | test (artifact: vv/artifacts/cc14) | `spaces/holospaces/tests/cc14_assembly.rs::the_oci_layers_assemble_into_a_clean_mountable_ext4_rootfs` | ✅ |
| **CC-14** | virtio block — the generated device tree is valid | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc14_virtio_block.rs::the_generated_device_tree_is_valid` | ✅ |
| **CC-15** | workspace — the os and holospaces share the workspace over virtio 9p | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc15_workspace.rs::the_os_and_holospaces_share_the_workspace_over_virtio_9p` | ✅ |
| **CC-16** | network — the os reaches the internet through the userspace nat | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc16_network.rs::the_os_reaches_the_internet_through_the_userspace_nat` | ✅ |
| **CC-17** | workspace fs — the editor lists writes and reads the shared workspace by kappa | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc17_workspace_fs.rs::the_editor_lists_writes_and_reads_the_shared_workspace_by_kappa` | ✅ |
| **CC-18** | lsp — the language server and session are in the assembled rootfs | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc18_lsp.rs::the_language_server_and_session_are_in_the_assembled_rootfs` | ✅ |
| **CC-18.bridge** | lsp bridge — the in os language server serves the workbench over the bridge | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc18_lsp_bridge.rs::the_in_os_language_server_serves_the_workbench_over_the_bridge` | ✅ |
| **CC-20** | import — a devcontainer provisions from a repository url | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc20_import.rs::a_devcontainer_provisions_from_a_repository_url` | ✅ |
| **CC-21** | port forward — a server in the devcontainer is reachable through a forwarded port | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc21_port_forward.rs::a_server_in_the_devcontainer_is_reachable_through_a_forwarded_port` | ✅ |
| **CC-22** | lifecycle — the lifecycle init is built from the parsed config | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc22_lifecycle.rs::the_lifecycle_init_is_built_from_the_parsed_config` | ✅ |
| **CC-23** | personalization — the operators dotfiles are injected into the assembled rootfs | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc23_personalization.rs::the_operators_dotfiles_are_injected_into_the_assembled_rootfs` | ✅ |
| **CC-24** | auth — the devcontainer authenticates with github over the network | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc24_auth.rs::the_devcontainer_authenticates_with_github_over_the_network` | ✅ |
| **CC-25** | features — the feature is staged and scheduled in the rootfs | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc25_features.rs::the_feature_is_staged_and_scheduled_in_the_rootfs` | ✅ |
| **CC-26** | build — the dockerfile build is assembled | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc26_build.rs::the_dockerfile_build_is_assembled` | ✅ |
| **CC-27** | compose — the import resolves the compose service from a repo | test (artifact: vv/artifacts/cc27) | `spaces/holospaces/tests/cc27_compose.rs::the_import_resolves_the_compose_service_from_a_repo` | ✅ |
| **CC-28** | reconfigure — the control plane reconfigures an instance over the substrate | test | `spaces/holospaces/tests/cc28_reconfigure.rs::the_control_plane_reconfigures_an_instance_over_the_substrate` | ✅ |
| **CC-29** | read verify boundary — the boundary check rejects a liar but the trusted read trusts the store | test | `spaces/holospaces/tests/cc29_read_verify_boundary.rs::the_boundary_check_rejects_a_liar_but_the_trusted_read_trusts_the_store` | ✅ |
| **CC-30** | resume — restore is the inverse of snapshot and continues identically | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc30_resume.rs::restore_is_the_inverse_of_snapshot_and_continues_identically` | ✅ |
| **CC-31** | resume terminal — a resumed idle devcontainer is live and its scrollback is a terminal concern | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc31_resume_terminal.rs::a_resumed_idle_devcontainer_is_live_and_its_scrollback_is_a_terminal_concern` | ✅ |
| **CC-33** | guest bridge — a guest server is reachable over the in process substrate bridge | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc33_guest_bridge.rs::a_guest_server_is_reachable_over_the_in_process_substrate_bridge` | ✅ |
| **CC-35** | aarch64 — the a64 data processing battery passes | test (artifact: vv/artifacts/cc35) | `spaces/holospaces/tests/cc35_aarch64.rs::the_a64_data_processing_battery_passes` | ✅ |
| **CC-36** | aarch64 — the emulator boots real arm64 linux to userspace | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc36_aarch64.rs::the_emulator_boots_real_arm64_linux_to_userspace` | ✅ |
| **CC-37** | aarch64 — an arm64 devcontainer runs a stock linux arm64 binary | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc37_aarch64.rs::an_arm64_devcontainer_runs_a_stock_linux_arm64_binary` | ✅ |
| **CC-38** | content net — a browser peer fetches content from a bare metal peer | test | `spaces/holospaces/tests/cc38_content_net.rs::a_browser_peer_fetches_content_from_a_bare_metal_peer` | ✅ |
| **CC-40** | product security — sec integrity tampered content does not re derive | test | `spaces/holospaces/tests/cc40_product_security.rs::sec_integrity_tampered_content_does_not_re_derive` | ✅ |
| **CC-44** | x64 boot — an amd64 linux kernel boots to userspace | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc44_x64_boot.rs::an_amd64_linux_kernel_boots_to_userspace` | ✅ |
| **CC-45** | dogfood — holospaces builds in its own real devcontainer | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc45_dogfood.rs::holospaces_builds_in_its_own_real_devcontainer` | 🟡 |
| **CC-46** | devbus parity — the aarch64 core mounts a 9p workspace over the shared devbus | test | `spaces/holospaces/tests/cc46_devbus_parity.rs::the_aarch64_core_mounts_a_9p_workspace_over_the_shared_devbus` | ✅ |
| **CC-46.boot** | realboot — the aarch64 core serves 9p net and bridge to a real arm64 boot | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc46_realboot.rs::the_aarch64_core_serves_9p_net_and_bridge_to_a_real_arm64_boot` | ✅ |
| **CC-50** | streaming assembly — a sparse large rootfs streams with bounded peak memory | test | `spaces/holospaces/tests/cc50_streaming_assembly.rs::a_sparse_large_rootfs_streams_with_bounded_peak_memory` | ✅ |
| **CC-51** | nested workspace — the host and os share a nested workspace tree over virtio 9p | test (heavy: QEMU/browser · #[ignore]) | `spaces/holospaces/tests/cc51_nested_workspace.rs::the_host_and_os_share_a_nested_workspace_tree_over_virtio_9p` | ✅ |

## CS — specification conformance (holospaces docs V&V; external: arc42 / C4 / OPM ISO 19450 / ISO 15288; validator-witnessed, non-BDD)

Absorbed from holospaces' docs V&V (spec 06 §docs; MG-8). Each row validates the *documentation*
(now at `specs/holospaces/`) against an external standard via a pinned validator (V1–V8, run by
`specs/holospaces/scripts/validate.sh`); the authorities + pins are in `specs/holospaces/tools/`.
Like CC/AS/KC, these are non-BDD (the honesty meta-gate does not bind them to Gherkin); **MG-8**
witnesses the CS absorption. Enforcement needs the docs toolchain (JDK 21 · Ruby 3 · Structurizr ·
cmark-gfm · pandoc), so 🟡 = present + imported; ✅ once the docs-conformance CI job gates it green.

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CS-1** | Architecture structure conforms to arc42 (pinned arc42 template + generator). | validators V1+V2 | `specs/holospaces/scripts/v2-arc42-build.sh` | ✅ |
| **CS-2** | The C4 model is well-formed (Structurizr DSL parses + renders). | validator V3 | `specs/holospaces/scripts/v3-structurizr.sh` | ✅ |
| **CS-3** | Rendered docs are valid GitHub-flavoured Markdown (CommonMark/GFM + github-markup). | validators V4+V5 | `specs/holospaces/scripts/v4-cmark-gfm.sh` | ✅ |
| **CS-4** | The conceptual model is valid OPM — every OPL parses against the ISO 19450 grammar. | validator V6 | `specs/holospaces/scripts/v6-opl-syntax.sh` | ✅ |
| **CS-5** | Each OPD agrees with its OPL (OPD↔OPL coherence). | validator V7 | `specs/holospaces/scripts/v7-opd-opl-coherence.sh` | ✅ |
| **CS-6** | The lifecycle covers the standard ISO/IEC/IEEE 15288 processes (superset check). | validator V8 | `specs/holospaces/scripts/v8-iso-15288-superset.sh` | ✅ |
