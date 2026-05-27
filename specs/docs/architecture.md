# Hologram Architecture

## Overview

Hologram is a content-addressed, UOR-native tensor runtime built on the UOR Foundation. It compiles
a tensor graph to a `.holo` archive and executes it through a single content-addressed buffer pool:
every value carries a UOR-ADDR κ-label, so identical computation is addressed once and reused
(memoized, deduplicated, replayed) rather than recomputed, and a function over a finite quantum
domain is materialized once as a lookup table. (ONNX/GGUF models are realized into this graph by the
downstream `hologram-ai` layer via the `model-formats` feature.)

---

## Prism Algebraic Grounding

Hologram's design is formally grounded in the **Prism ontology v1.3.0** — the "Polymorphic
Resolution and Isometric Symmetry Machine" — which is the algebraic runtime layer extending the
UOR Foundation operations graph.

### PP_1: Pipeline Unification (hologram's O(1) claim)

Prism identity PP_1 states:

```
κ(λ_k(α*(ι(s,·))),C) = resolve(s,C)
```

The composed pipeline (dispatch → inference → accumulation → composition) collapses to a single
O(1) resolution on a **saturated context**. Hologram's compiled `.holo` archive is this saturated
context: all shapes, dtypes, and constants are resolved at compile time, so identical computation is
addressed once by its κ-label and a graph-level memo hit is O(1) in graph size.

**Derivation chain** (each step traces to a UOR Foundation axiom):

| Step | Prism Identity | Foundation Axiom | Role in hologram |
|------|---------------|-----------------|------------------|
| 0 | PI_3 (inference monotonicity) | SR_1 (freeCount non-increasing) | Shape propagation only converges |
| 1 | PA_1 (accumulation associativity) | SR_10 (Church-Rosser confluence) | Parallel level order doesn't affect final state |
| 2 | PL_3 (lease completeness recovery) | MC_6 (full coverage → σ=1) | All levels compose back to full saturation |
| 3 | PK_2 (composition O(1) resolution) | MC_7 (stepCount=0 on saturated context) | κ-label memo hit is O(1) |

### Additional Identities in Use

| Prism Identity | Foundation Basis | Hologram Component |
|---------------|-----------------|-------------------|
| PA_4 (base binding preservation) | SR_1 + bitmask OR irreversibility | Compiled `.holo` immutability (`KernelCall`s fixed post-compile); PM_5 rollback |
| PI_1 (inference idempotence) | CC_1 + SC_5 | κ-label result memoization in the `BufferArena` pool |
| PD_1 (dispatch determinism) | AD_1 (addressing bijection) | `CpuBackend::dispatch` exhaustive-match determinism |
| PD_2 (dispatch type safety) | CB_5 (fiber sufficiency) | dtype-gated dispatch |
| PL_2 (lease disjointness) | SR_9 (ContextLease fiber disjointness) | `ParallelLevel` isolation |
| PX_5 (infeasibility detection) | CB_5 + SR_5 (ContradictionBoundary) | `CompileError` taxonomy |
| PM_5 (transaction atomicity) | PA_4 (base preservation = free rollback) | `InferenceSession::execute()` error contract |
| PK_3 (parallelism bound) | MC_8 (work ≤ ⌈n/k⌉ for k leases) | Level fusion quality criterion |

---

## Three-Space Model

Hologram's crates follow the Prism space classification. Each space has distinct mutability and
deployment guarantees:

| Space | Prism Definition | Hologram Crates |
|-------|-----------------|-----------------|
| **kernel** | Deployment-immutable; contains foundation operations and algebraic laws | `hologram-host`, `hologram-types`, `hologram-ops`, `hologram-graph`, `hologram-archive` |
| **bridge** | Prism-computed; derives from kernel crates via explicit composition laws | `hologram-exec`, `hologram-compiler`, `hologram-backend` |
| **user** | Application-configurable; exposed at system boundaries | `hologram-ffi`, `hologram-cli`, `hologram-bench` |

**Rule**: kernel crates must not depend on bridge or user crates. Bridge crates must not depend on
user crates. This enforces the one-way information flow required by the Prism space hierarchy.

---

## Crate Dependency Graph

The workspace is 11 crates (`hologram-core`, `hologram-async`, and
`hologram-transform` were earlier-design crates that no longer exist):

```
hologram-host ─┐
hologram-types ┤
               └── hologram-ops (kernel: canonical op vocabulary)
                       └── hologram-graph (kernel: tensor graph IR)
                               └── hologram-archive (kernel: .holo format, κ-labels)
                                       ├── hologram-backend (bridge: CPU/GPU kernels)
                                       └── hologram-exec (bridge: content-addressed executor)
                                               └── hologram-compiler (bridge: graph → .holo)
                                                       ├── hologram-ffi (user: C ABI)
                                                       └── hologram-cli (user: CLI)
                                       └── hologram-bench (user: benchmarks)
```

`hologram-ops` is the canonical semantic operation vocabulary. It describes
what an operation means (`OpKind`, `BackwardRule`, semantic signature) without
committing to any graph encoding, executable kernel format, or device backend.
`hologram-graph` now has a matching `GraphOp::Compute(...)` path for canonical
semantic compute nodes, while `GraphOp::Float(...)` remains as a legacy
execution-compatibility encoding during the migration.

---

## Content-Addressed Execution Pipeline

Hologram compiles a dataflow graph into a `.holo` archive of `KernelCall`s plus an execution
schedule. At load time the archive is decoded and the load-time fusion passes run; at execution
time the backend dispatches each `KernelCall` against a single content-addressed buffer pool. Every
value carries a UOR-ADDR κ-label, a slot *binds* to a buffer by that label, and identical
computation is memoized rather than recomputed. This eliminates per-node op matching against the
graph, HashMap lookups on the hot path, and vtable indirection at execution time — realising the
PP_1 O(1) resolution claim.

### Stage 1: Graph — Edges Define Data Paths

Each `Node` in the graph connects to its inputs via `InputSlot`, which names a source `NodeId` and
an `output_port`. The graph exposes `predecessors()` and `successors()` for traversal, plus
`build_successor_index()` for O(1) reverse-edge lookups used by the fusion and scheduling passes.

**Key types**: `Node`, `InputSlot`, `InputSource` (`hologram-graph/src/graph/node.rs`)

### Stage 2: Schedule — Paths Become an Ordered Plan

A modified Kahn's topological sort partitions the graph into `ParallelLevel`s. Nodes within a level
have no mutual dependencies, and all predecessors reside in strictly earlier levels. This satisfies
**PL_2 (lease disjointness)**: nodes in a level hold non-overlapping buffer leases. The flattened
level order is the deterministic execution schedule carried in the archive.

Critical-path analysis (DP over the topological order) computes the longest dependency chain, giving
the parallelism ratio `total_nodes / critical_path_length`.

**Key types**: `ExecutionSchedule`, `ParallelLevel` (`hologram-graph/src/schedule/`)

### Stage 3: Compilation — Ops Become `KernelCall`s

The compiler lowers each scheduled graph op (`OpKind`, and its per-op marker type in
`hologram-ops`) into a variant of the `KernelCall` enum (`hologram-backend/src/kernel_call.rs`).
A `KernelCall` is a fully-resolved, self-describing instruction: it carries the operand κ-labels,
the resolved shapes/dtypes, and the output label. There is no boxed trait object and no runtime
shape-inference module — shapes are resolved at compile time and travel inside the `KernelCall`.

The compiler emits these `KernelCall`s plus the schedule into a `.holo` archive. All graph edges are
resolved to κ-labels at this stage; no graph traversal occurs at runtime.

**Key types**: `OpKind` and per-op marker types (`hologram-ops`), `KernelCall`
(`hologram-backend/src/kernel_call.rs`), `.holo` archive (`hologram-archive`)

### Stage 4: Load — Decode + Fuse

`InferenceSession::load` (`hologram-exec/src/session.rs`) decodes the archive, then runs the
load-time content-addressed fusion passes (see below). The result is the ordered list of
`KernelCall`s the session will dispatch, with constants pinned into the pool for the session
lifetime. Warm-start may pre-populate the pool from a persisted κ-store (`WarmStore`), so the
compiled object is never cold.

**Key types**: `InferenceSession`, `WarmStore` (`hologram-exec/src/session.rs`,
`hologram-exec/src/warm.rs`)

### Stage 5: Execution — κ-Label Binding Against the Pool

`InferenceSession::execute` dispatches each `KernelCall` against `BufferArena`
(`hologram-exec/src/buffer.rs`), the single content-addressed buffer pool. A value lives in exactly
one aligned buffer; a slot *binds* to it by κ-label. The pool holds two buffer classes:

| Class | Purpose | Lifetime |
|-------|---------|----------|
| pinned | model constants/weights, deduped by content κ-label | session lifetime |
| transient | activations, byte-bounded so memory holds for arbitrary models | reused/recycled |

The CPU backend (`CpuBackend`, `hologram-backend/src/cpu.rs`) dispatches by an **exhaustive `match`
over `KernelCall`** — no virtual dispatch, no function-pointer tables, no runtime algorithm
selection. Before computing, the pool checks whether the output κ-label is already resident
(pinned or transient); if so the compute is **elided** and the slot rebinds to the existing buffer
(the κ-label memo). This is the single mechanism behind result caching — there is no separate KV
store.

Steady-state execution is **zero-allocation** on the hot path: a κ-label miss writes into a recycled
transient buffer reclaimed from the pool rather than allocating a fresh one. Identical computation
across nodes (or across runs, via warm-start) is an O(1) rebind.

**Key types**: `BufferArena` (`hologram-exec/src/buffer.rs`), `CpuBackend`
(`hologram-backend/src/cpu.rs`)

### Fusion — Content-Addressed Path Shortening

Fusion happens in two phases. The compiler first desugars composite ops to primitives and applies
**algebraic elision** (bit-exact-sound identities/involutions, Reshape relabel, DCE) — compute the
κ-label algebra proves unnecessary is removed. Then, at load time, content-addressed fusion passes
collapse adjacent `KernelCall`s into fused variants that elide the intermediate buffer:

| Pattern | Fused `KernelCall` |
|---------|--------------------|
| MatMul → Activation | `MatMulActivation` |
| MatMul → Add(bias) → Activation | `MatMulAddActivation` |
| Dequantize → MatMul | `MatMulDequant` (never materialises the dense f32 weight) |
| Dequantize → Activation | `DequantActivation` |
| Expand → elementwise-binary | `BroadcastBinary` (zero-movement Expand) |

A single topological pass over the graph also applies constant folding, view fusion, and CSE.
Topological order guarantees predecessors are processed before successors, so forward-looking
(epilogue) and backward-walking (view) fusions compose correctly in one pass.

**Key entry point**: `fusion::fuse()` (`hologram-graph/src/fusion/mod.rs`)

#### 1. Constant Folding

Operations whose inputs are all compile-time constants are evaluated immediately and replaced with
a single `Const` node. Handles unary/binary `LutOp`, `PrimOp`, `FusedView`, and
`RingPrimUnary`/`RingPrimBinary` variants.

**Effect**: eliminates computation entirely and shrinks the graph.

**Key file**: `hologram-graph/src/fusion/constant.rs`

#### 2. View Fusion (Q0 — byte domain)

Backward-walks each node to find chains of byte-domain unary ops and composes them into a single
256-byte lookup table (`ElementWiseView`). A chain like `Sigmoid → Relu → Gelu` becomes one
`FusedView` node — one array access regardless of chain length.

- **Algebraic fast path**: involutions such as `Neg∘Neg` and `Bnot∘Bnot` are detected and replaced
  with `Passthrough` (zero-cost identity) without materialising a table.
- **Size**: each `ElementWiseView` is exactly 256 bytes, cache-line aligned for efficient L1 access.

**Effect**: replaces N-node chains with a single node; eliminates all intermediate buffers.

**Key file**: `hologram-graph/src/fusion/view_fusion.rs`

#### 3. Q1 View Fusion (16-bit domain)

Mirrors Q0 view fusion for operations at the Q1 ring level (`RingPrimUnary`/`RingPrimBinary` with
`RingLevel::Q1`). Produces a 128 KB `ElementWiseView16` table (heap-allocated via `Box` to avoid
stack overflow). The same involution fast path applies at Q1.

- **Ring-level safety**: never fuses across ring-level boundaries (Q0 → Q1 remains intact).

**Key file**: `hologram-graph/src/fusion/q1_view_fusion.rs`

#### 4. Epilogue Fusion (MatMul / Conv2d / Norm + Activation)

Detects linear-algebra ops whose sole successor is an element-wise activation (and optionally a
bias add in between) and merges them into a single fused node. The activation is applied
**in-register** during accumulation, avoiding a round-trip to memory for the intermediate buffer.

Supported patterns:

| Pattern | Fused node |
|---------|------------|
| `MatMul → Activation` | `FusedMatMulActivation` |
| `MatMul → Add(bias) → Activation` | `FusedMatMulBiasActivation` |
| `Conv2d → Activation` | `FusedConv2dActivation` |
| `Conv2d → Add(bias) → Activation` | `FusedConv2dBiasActivation` |
| `RmsNorm / LayerNorm / GroupNorm → Activation` | `Fused{Norm}Activation` |
| `MatMulLut4 / MatMulLut8 → Activation` | `MatMulLut{4,8}Activation` |

Detection rule: the predecessor has exactly one successor, and that successor has exactly one
predecessor (no fan-out / fan-in). The successor absorbs the predecessor's inputs and the
predecessor is removed.

**Effect**: eliminates large intermediate buffers. For example, in Stable Diffusion's UNet
(512×512, 320 channels), one Conv2d + Activation fusion saves ~335 MB of memory bandwidth;
across 23 ResNet blocks that totals ~7.7 GB saved per inference step.

**Key file**: `hologram-graph/src/fusion/float_fusion.rs`

#### 5. Common Subexpression Elimination (CSE)

Hash-based deduplication of nodes with identical `(op, sorted_predecessors)` signatures. All
successors of a duplicate node are rewired to the canonical node.

- **Commutative-aware**: predecessor lists are only sorted for commutative ops (`Add`, `Mul`) to
  preserve semantics of non-commutative ops (`Sub`, `Div`).
- **Purity check**: only deduplicates pure ops; skips `Input`, `Output`, `CallSubgraph`.

**Key file**: `hologram-graph/src/fusion/cse.rs`

#### Fusion → `KernelCall` mapping

Each fused graph node maps to a specific `KernelCall` variant — the view-fusion tables, the
epilogue variants (`MatMulActivation`, `MatMulAddActivation`), the quantized variants
(`MatMulDequant`, `DequantActivation`), and `BroadcastBinary` — so the backend sees fully resolved
instructions with **zero pattern-detection overhead** at runtime.

**Key types**: `KernelCall` (`hologram-backend/src/kernel_call.rs`); load-time fusion in
`InferenceSession::load` (`hologram-exec/src/session.rs`)

### Prism Grounding

The content-addressed pipeline realises several Prism identities:

- **PP_1** — Pre-resolution of all paths at compile time means execution is a single O(1) κ-label
  rebind/memo per `KernelCall` on the saturated context (the archive + buffer pool).
- **PL_2** — Schedule level boundaries guarantee buffer-lease disjointness within each level.
- **PA_1** — Accumulation associativity means the order of operations within a level does not affect
  the final result, enabling safe parallelism.

---

## Cascade Compilation Pipeline

All compilation routes through the 7-stage cascade engine (`hologram-cascade`), implementing
the UOR cascade pipeline from uor-foundation v0.1.3. The cascade is the sole compilation path;
there is no separate imperative pipeline.

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `compile(graph)` | Raw Graph — wraps in CompileUnit via `unit_from_graph()` → cascade |
| `CompilerBuilder::from_unit(unit, graph)` | Pre-built CompileUnit + pre-lowered Graph → cascade |
| `CompilerBuilder::from_source(source, ...)` | UOR term language → parse → preflight → lower → cascade |
| `compile_from_source(source, ...)` | Convenience wrapper for `from_source` |

### 7 Cascade Stages

| Stage | Name | Ontology | Role |
|-------|------|----------|------|
| 0 | Init (Ω⁰) | `stage_initialization` | Certificate cache check; on hit, skip to Extract |
| 1 | Declare (Ω¹) | `stage_declare` | Ring-level precision promotion via `promote_prim_ring_levels()` |
| 2 | Factorize (Ω²) | `stage_factorize` | Fusion passes: constant folding, view fusion, CSE |
| 3 | Resolve (Ω³) | `stage_resolve` | Build execution schedule via Kahn's algorithm |
| 4 | Attest (Ω⁴) | `stage_attest` | Liveness analysis, workspace planning, QEDL boundaries, assertion verification |
| 5 | Extract (Ω⁵) | `stage_extract` | Build the `KernelCall` sequence from graph + schedule |
| 6 | Converge (π) | `stage_convergence` | Emit `.holo` archive with LayerHeader + CompileUnitMeta sections |

### Certificate Memoization

Keyed by `(unit_address: [u8; 32], quantum_level: RingLevel)`. On cache hit at Stage 0,
stages 1-4 are skipped entirely. `unit_from_graph()` computes deterministic BLAKE3 hashes
from graph structure (node ops + edges), enabling memoization across identical graphs.
The `CertificateStore` supports dynamic resizing (doubles at 75% load) and disk persistence.

### Archive Sections

Compiled archives include:
- `SECTION_LAYER_HEADER` (kind 2): Layer descriptors and execution schedule
- `SECTION_COMPILE_UNIT_META` (kind 5): Unit address, quantum level, budget, term/binding/assertion counts

**Key types**: `CascadeState`, `CascadeStage`, `CascadeResult` (`hologram-cascade/src/stage.rs`),
`run_cascade_with_graph()` (`hologram-cascade/src/engine.rs`),
`CompilerBuilder` (`hologram-compiler/src/compiler/mod.rs`)

---

## Quantum Level Strategy

Hologram implements UOR's quantum level hierarchy for ring-arithmetic acceleration:

| Level | Bits | Ring | Strategy |
|-------|------|------|----------|
| Q0 | 8 | Z/256Z | Full LUT (256 B per table) |
| Q1 | 16 | Z/65536Z | Full LUT (128 KB per table) |
| Q2 | 24 | Z/16777216Z | Hierarchical segmentation (~50 MB) |
| Q3 | 32 | Z/4294967296Z | Algorithmic only (17 GB full LUT infeasible) |
| Q4+ | 40+ | Z/2^nZ | Algorithmic with optional LRU cache |

Q0 and Q1 are fully realised as lookup tables in `hologram-backend` (`cpu::lut`).
Q2+ are computed algorithmically.

---

## Error Taxonomy (Prism PX_5)

Compilation failures are classified according to Prism PX_5's two infeasibility classes:

- **Insufficient** (`CompileError::InsufficientKernel`): the CB_5 fiber-sufficiency check fails
  because no dispatcher covers the required (op, dtype) pair. Resolution: register a kernel or
  lower to a supported dtype.

- **Contradictory** (`CompileError::ContradictoryConstraint`): the SR_5 ContradictionBoundary
  fires because two shape or type constraints conflict at the same node. Resolution: fix the model
  topology or add an explicit cast.
