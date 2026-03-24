# Coherence-Native Model Architecture Exploration Prompt

You are helping design a speculative but technically grounded **coherence-native neural architecture** that could serve as an alternative or hybrid successor to transformer-based models.

Your job is to explore, critique, and refine a model family where computation is based on **field evolution, phase relationships, interference, resonance, spectral mixing, and selective memory**, rather than token-to-token attention as the primary mechanism.

We are not looking for sci-fi language. We want a serious, engineering-oriented architecture exploration with:
- mathematical intuition
- architecture decomposition
- training strategy
- systems implications
- backend/runtime implications
- hybridization strategy
- failure modes
- evaluation plan

## Core constraint: All computation as LUT

**All computation in this architecture must resolve to lookup-table composition on finite rings.** Field evolution, interference, spectral transforms, resonance, memory access, and output probing must each be expressible as O(1) table lookups or compositions thereof. Where a primitive cannot be LUT-reduced, the design must explain why and propose the minimal non-LUT escape hatch.

This is the central design thesis. The architecture is not "inspired by" LUTs — it IS LUT computation on finite rings. Every section below must be evaluated through this lens.

## Context

We want to explore whether a model can be built around these ideas:

- inputs are lifted into **complex-valued or phase-aware states** encoded on finite rings
- tokens / patches / graph nodes deposit energy into a shared latent **coherence field** represented as ring-valued arrays
- computation proceeds via:
  - phase evolution (LUT on phase byte)
  - interference (ring addition on Z/nZ)
  - resonance gating (threshold LUT + masking)
  - spectral transforms (NTT — number-theoretic transform on Z/nZ)
  - selective state-space memory (LUT-addressed persistent buffers)
  - energy normalization (norm LUT + scaling table)
- outputs are obtained by probing local and global field structure via LUT readout
- attention is not the default primitive; it may appear only as a narrow auxiliary mechanism if absolutely necessary

We are especially interested in architectures that combine ideas from:
- complex-valued neural networks
- state space models
- neural operators / Fourier-domain models (NTT on finite rings, not floating-point FFT)
- energy-based models
- graph-to-field representations
- finite ring geometry (Z/256Z, Z/65536Z)
- sparse controllers for discrete routing

## Substrate: Hologram LUT hierarchy

This architecture targets the **Hologram runtime**, which already implements a multi-level LUT execution model. The design must compile down to these existing primitives or propose minimal extensions.

### Existing LUT levels

| Level | Ring | Table size | Storage | Composition | Status |
|-------|------|-----------|---------|-------------|--------|
| **Q0** | Z/256Z | 256 entries (256 bytes) | Stack, L1-aligned | `view.then(other)` → `result[i] = b[a[i]]` | Production |
| **Q1** | Z/65536Z | 65536 entries (128 KB) | Heap | Same `.then()` semantics | Production |
| **Q2** | Z/2²⁴ | Hierarchical segments (~50 MB) | Heap | Segmented composition | Design phase |
| **Q4-GEMM** | 16 centroids | 16 entries per codebook | Packed (2 indices/byte) | Psumbook partial-sum accumulation | Production |
| **Q8-GEMM** | 256 centroids | 256 entries per codebook | 1 index/byte | Psumbook partial-sum accumulation | Production |

### Existing execution model

- **Flat tape**: Pre-compiled `EnumTape` — instruction array with O(1) arena indexing, no runtime graph traversal
- **BufferArena**: Zero-copy shared memory plane with swap-insert recycling, O(1) flat indexing by node ID
- **View fusion**: Chains of unary ops fused into single LUT at compile time (`Sigmoid → Relu → Tanh` → one 256-byte table)
- **ParallelLevel scheduling**: Topological sort into disjoint execution levels (buffer-lease disjointness)
- **Multi-backend dispatch**: CPU (SIMD: AVX2/SSE4.2/NEON) / Metal / WebGPU
- **Ring arithmetic**: GF(2), GF(3), and Z/nZ tables already in static `.rodata` (~519 KB total)

### Key principle

Every new coherence primitive must answer: **"What is the LUT?"** — specifically, what ring, what table size, and can it compose with adjacent operations via `.then()`?

## Desired output

Produce a detailed design document with the following sections.

### 1. Executive summary
Explain:
- what a coherence-native model is
- what it is trying to replace or augment in transformers
- why LUT-reducibility makes this feasible (O(1) per operation, deterministic, composable)
- why this might fail (precision limits of finite rings, non-reducible operations)

### 2. Coherence-to-LUT reducibility mapping

For each coherence primitive, fill in the complete table. Every "?" must become a concrete proposal or a proof of impossibility.

| Coherence Primitive | Current Hologram Analog | LUT-Reducible? | Proposed LUT Strategy | Ring | Table Size | Composable? |
|---|---|---|---|---|---|---|
| Field evolution (unary) | ElementWiseView | Yes (proven) | — | Q0/Q1 | 256B/128KB | Yes |
| Operator composition | view.then() | Yes (proven) | — | Q0/Q1 | 256B/128KB | Yes |
| Complex-valued states | — | ? | ? | ? | ? | ? |
| Phase rotation | — | ? | ? | ? | ? | ? |
| Interference (additive) | — | ? | ? | ? | ? | ? |
| Resonance gating | — | ? | ? | ? | ? | ? |
| Spectral transform (FFT/NTT) | — | ? | ? | ? | ? | ? |
| Field interaction (bilinear) | LUT-GEMM | Yes (proven) | Centroid lookup + Psumbook | Q4/Q8 | 16/256 centroids | N/A |
| Energy normalization | — | ? | ? | ? | ? | ? |
| Standing-wave memory read | KvCacheState | Partial | ? | ? | ? | ? |
| Standing-wave memory write | KvCacheState | Partial | ? | ? | ? | ? |
| Dynamic routing / MoE | — | ? | ? | ? | ? | ? |
| Output probing / readout | — | ? | ? | ? | ? | ? |

Also include the standard transformer-to-coherence mapping (token embeddings, positional encoding, attention, softmax, feedforward, residuals, KV cache, MoE routing, recurrence) but with LUT strategy for each.

### 3. Core mathematical model

Formalize the core state representation such that **every operation is a function `Z/nZ → Z/nZ`** (or `(Z/nZ)^k → (Z/nZ)^k` for multi-channel ops) that can be precomputed as a lookup table.

For operations where full tabulation is infeasible (e.g., bilinear interactions on large domains), propose a **factored representation** — e.g., centroid quantization + partial-sum accumulation (as LUT-GEMM already does for matmul).

Specifically address:
- How complex-valued states are encoded on Z/256Z or Z/65536Z (paired bytes? amplitude-phase? real-imaginary?)
- How phase evolution is a ring automorphism expressible as a LUT
- How interference maps to ring addition (and whether Z/256Z addition is sufficient or a different ring is needed)
- How spectral transforms decompose into LUT-composable stages (NTT butterfly decomposition on Z/nZ)
- How energy normalization avoids floating-point (ring-valued norm tables?)

Be explicit about candidate equations using ring notation, not real-valued approximations.

### 4. Layer/block design
Design one or more candidate "coherence blocks" that could replace a transformer block.

For each block, specify:
- inputs (ring-valued, which ring)
- internal sub-operations (each with its LUT strategy)
- outputs
- residual structure (ring addition?)
- normalization strategy (LUT-based)
- computational complexity (in table lookups, not FLOPs)
- likely hardware characteristics

**LUT budget analysis** (required for each block):
- Total LUT bytes (static `.rodata`)
- Per-inference working set (must fit L2 for hot path)
- Composition depth (how many `.then()` chains before precision loss matters on the chosen ring)
- Fallback strategy when composition depth exceeds precision budget

### 5. Memory architecture
Design a multi-tier memory model, including:
- local transient coherence (arena buffers — already exists)
- persistent standing-wave or modal memory (how does this map to LUT-addressable state?)
- selective recurrent state (LUT-gated read/write?)
- optional external symbolic / retrieval memory

Explain how this differs from KV caching and how it supports long-context reasoning. All memory access patterns must be LUT-addressable or justify why not.

### 6. Hybrid control strategy
Assume pure coherence dynamics may not be sufficient for discrete reasoning.

Propose one or more hybrid strategies that combine:
- coherence field computation (LUT-based)
- sparse routing
- symbolic constraints
- tool usage
- program-like control
- optional minimal attention mechanisms at boundaries only

**Critical constraint**: The primary execution model is a **static pre-compiled tape**. All data paths are resolved at compile time. Dynamic routing must be expressible as **LUT-gated tape selection** — a threshold lookup that selects among pre-compiled execution paths. No runtime graph mutation is permitted.

Be explicit about where discreteness enters the system and how it interacts with the static-tape model.

### 7. Dataflow and runtime model
Describe the runtime as a multi-plane system:
- representation plane (ring-valued state arrays in BufferArena)
- field plane (LUT evolution on arena buffers)
- memory plane (persistent modal state, extending KvCacheState)
- control plane (EnumTape instruction sequencing)
- execution plane (backend dispatch: CPU SIMD / Metal / WebGPU)

Show how data moves between them — all transfers must be zero-copy or LUT-mediated.

Discuss how this maps to the existing Hologram backend-agnostic runtime targeting:
- CPU (with SIMD: AVX2, SSE4.2, NEON)
- Metal (unified memory on Apple Silicon)
- WebGPU (compute shaders)
- CUDA (future)

### 8. Training strategy
Propose staged training, for example:
- self-supervised substrate pretraining
- masked reconstruction or next-state prediction
- curriculum for long-range dependencies
- hybrid fine-tuning for language / multimodal tasks
- energy or stability regularization
- auxiliary objectives for phase consistency or modal sparsity

Discuss:
- How training works when inference is LUT-based (train in float, quantize to ring? train directly on ring? straight-through estimators?)
- How LUT tables are learned vs. fixed
- How composition depth affects gradient flow
- Likely optimization problems and how to mitigate them

### 9. Evaluation plan
Design a benchmark plan that tests:
- long-context retention
- compositional reasoning
- retrieval-like behavior
- sequence efficiency
- multimodal fusion
- scaling behavior
- interpretability of field structure
- **LUT utilization efficiency** (what fraction of table entries are actually hit?)
- **precision vs. ring size** (Q0 vs. Q1 vs. Q2 quality curves)

Include:
- synthetic tasks
- realistic language tasks
- ablation studies (especially: ring size, composition depth, LUT-GEMM centroid count)

### 10. Failure modes and criticism
Give a serious critique of the design.

Address:
- training instability (especially ring quantization noise)
- lack of discrete reasoning
- interpretability limitations
- hardware inefficiency (table thrashing, cache pressure)
- inability to match transformer scaling
- unclear benefit over existing SSMs or hybrids
- **precision ceiling**: what tasks fundamentally cannot be done at Q0 (8-bit) resolution?
- **composition depth limits**: when do chained LUTs lose too much information?
- **NTT feasibility**: is number-theoretic transform on Z/256Z practically useful or toy-scale?

### 11. Research roadmap
Propose a staged roadmap:
- prototype A: single coherence block, Q0 LUTs only, synthetic tasks
- prototype B: multi-block stack, Q1 LUTs, language modeling
- prototype C: full architecture with NTT, modal memory, hybrid control
- benchmark phase: head-to-head vs. transformer baselines
- systems phase: integration with Hologram tape compilation
- production-readiness criteria

For each stage, specify:
- scope
- success metrics
- what to discard if it fails

### 12. Hologram compilation target (required)

Specify concretely how this architecture compiles to the Hologram runtime:

- **TapeKernel mapping**: Which new `TapeKernel` variants are needed? (e.g., `CoherenceEvolution(ElementWiseView)`, `NTTButterfly(...)`, `ResonanceGate(...)`)
- **Arena buffer layout**: How are complex field states laid out in `BufferArena`? (byte pairs? interleaved amplitude/phase? separate buffers?)
- **NTT decomposition**: How does the number-theoretic transform decompose into composable LUT stages that can be fused by view fusion?
- **Modal memory**: How does standing-wave memory extend `KvCacheState`? What new arena buffer types are needed?
- **LUT budget**: Total static `.rodata` for all coherence tables. Target: fit in L2 cache for hot path.
- **Zero-allocation**: Does the design stay within Hologram's zero-allocation steady-state execution model (swap-insert buffer recycling)?
- **Fusion opportunities**: Which sequences of coherence ops can be fused into single LUTs via `.then()`?

## Constraints

- Do not assume quantum hardware.
- Keep the design classical-first, even if quantum-inspired mathematically.
- **All primitives must be LUT-reducible or explicitly justified as escape hatches.**
- Do not hand-wave over training or implementation difficulty.
- Prefer explicit tradeoffs over optimism.
- Treat this as an R&D architecture memo, not marketing.
- When uncertain, present multiple competing design options and compare them.
- Assume the execution target is the Hologram runtime with its existing LUT hierarchy and tape model.

## Implementation effort analysis

The design must be grounded in realistic implementation cost. The Hologram codebase has been analyzed to determine the exact gap between current infrastructure and each coherence primitive. **Your design document should factor in these constraints — propose architectures achievable in Phase 1 first, with Phase 2/3 extensions clearly marked.**

### Pillar-by-pillar gap analysis

| Coherence Pillar | What Exists | What's Needed | Cheapest Path | Effort | Full Path | Effort |
|---|---|---|---|---|---|---|
| **Complex-valued states** | `ElementWiseView` is scalar `[u8; 256]` only | Amplitude + phase encoding | Paired Q0 views (two tables, convention only) | 1 week | Joint Q0 `[u8; 65536]` or ComplexQ1 encoding | 4–6 weeks |
| **NTT / spectral** | No FFT/NTT. Ring is Z/65536 (not prime — no primitive roots) | Spectral transforms on finite rings | Reframe LUT-GEMM as spectral interaction primitive | 0 weeks | Z/257 ring module + butterfly kernel + twiddle tables | 3–4 weeks |
| **Resonance gating** | `FloatOp::Where` (element-wise), threshold LUTs trivial to bake | Data-dependent field masking | Threshold LUT → `InlineMul` mask (already expressible) | 0 weeks | Tape bank selector (multi-tape MoE) | 2–3 weeks |
| **Modal memory** | `KvCacheState` — hardcoded for K/V pairs only | Persistent named field patterns across executions | Generalize KvCacheState to named mode channels | 2–3 weeks | — | — |
| **Energy normalization** | `InlineRmsNorm`, `InlineLayerNorm` (float) | Ring-valued norms without float | Keep float norms as escape hatch | 0 weeks | Precomputed norm² + sqrt tables on Z/257 | 1 week |
| **Training pipeline** | No training in hologram (inference only) | Float training → ring quantization | — | — | Straight-through estimators, learned LUT entries | 6–8 weeks |

### Key technical risks

- **Q0 precision ceiling**: Z/256Z has 8 bits of resolution. For tasks requiring fine discrimination (language modeling perplexity), this may be fundamentally insufficient. Q1 (16-bit) helps but costs 512× more memory per joint operation.
- **NTT ring incompatibility**: NTT requires a prime modulus for primitive roots of unity. Z/256Z and Z/65536Z are powers of 2 — no primitive roots exist. Z/257 works (primitive root: 3, supports NTT up to length 256) but introduces a **parallel ring** alongside existing Z/256Z infrastructure, not a replacement.
- **Static tape vs. dynamic routing**: The tape is fully pre-compiled with no runtime branching. Element-wise LUT masking (threshold → multiply) handles most gating needs, but true MoE-style expert selection requires a tape bank selector — a new executor concept.
- **Training convergence**: Learning discrete LUT entries via straight-through estimators is fragile. No published results demonstrate coherence-native models outperforming transformers on any task. The entire training strategy is unvalidated.
- **Opportunity cost**: Full implementation is ~31 weeks. That's time not spent on proven optimizations (matmul tuning, larger model support, standard training pipeline).

### Three-phase implementation plan

**Phase 1 — Minimum Viable Coherence (~5 weeks)**
- Paired Q0 views (convention only, no struct changes)
- Reframe LUT-GEMM as spectral interaction (no code change, just semantic interpretation)
- Threshold LUT + InlineMul for resonance gating (bake new tables, existing executor)
- Generalize KvCacheState → named modal memory channels
- Single coherence block as custom op via existing `CustomOpRegistry`
- Synthetic benchmark (associative recall, sequence copy)
- **No new `TapeKernel` variants needed**

*Go/no-go criteria:*
- Coherence block runs on EnumTape with ≤3 new LUT tables
- Synthetic task >80% accuracy
- LUT utilization >30% (≥77 of 256 entries hit during inference)
- Throughput within 2× of equivalent float custom op

**Phase 2 — Ring-Native Coherence (~11 weeks)** *(gate on Phase 1 results)*
- ComplexQ1 encoding (`amplitude:phase` in u16)
- Z/257 ring module + NTT butterfly kernel + `TapeKernel::NTTButterfly`
- Tape bank selector for MoE-style routing
- LUT-based norms on Z/257
- Small coherence language model vs. transformer baseline

**Phase 3 — Production Coherence (~15 weeks)** *(gate on Phase 2 language modeling results)*
- Training pipeline (float → ring quantization, straight-through estimators)
- Metal/WebGPU NTT compute kernels
- Scaling benchmarks (head-to-head vs. transformers)
- Architecture hyperparameter search (ring size, depth, centroid count)

### ROI signals to watch

- **LUT utilization**: If <30% of table entries are hit, the model uses a tiny fraction of the ring — a simpler mechanism may suffice.
- **Composition depth**: If accuracy degrades after >3 chained `.then()` calls, Q0 precision is the bottleneck.
- **Spectral vs. centroid**: If LUT-GEMM (centroid-based) matches NTT (spectral) quality, skip the Z/257 ring entirely.
- **Training loss curves**: If ring-quantized models don't converge within 2× the epochs of float baselines, the training strategy needs rethinking.

## Deliverable style

Write the answer like a production-quality internal architecture research memo:
- clear section headers
- dense but readable
- tables where useful
- concrete examples
- candidate equations (in ring notation, not just reals)
- no fluff
- no hype language

## Reference: Key source files

These files contain the existing LUT and execution infrastructure. If you have codebase access, ground your proposals in these implementations.

```
crates/hologram-core/src/view/mod.rs       — ElementWiseView (Q0, 256-byte LUT, SIMD apply)
crates/hologram-core/src/q1/view.rs        — ElementWiseView16 (Q1, 128KB LUT)
crates/hologram-core/src/lut/              — All precomputed activation/arithmetic tables
crates/hologram-graph/src/fusion/          — View fusion (unary chain → single LUT)
crates/hologram-exec/src/tape.rs           — TapeKernel enum, EnumTape, execution loop
crates/hologram-exec/src/tape_builder.rs   — Graph → tape compilation
crates/hologram-exec/src/buffer/arena.rs   — BufferArena (O(1) flat indexing, zero-copy)
crates/hologram-exec/src/lut_gemm/         — Quantized matmul via centroid LUT + Psumbook
crates/hologram-exec/src/kv/               — KV cache state (modal memory starting point)
specs/docs/architecture.md                 — Prism identities, three-space model
```
