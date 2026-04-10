# Hologram Architecture

## Overview

Hologram is an O(1) compute acceleration runtime built on UOR-Framework. It compiles ONNX models
into a content-addressed, KV-lookup execution graph where every operation resolves in O(1) on a
pre-saturated context.

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
O(1) resolution on a **saturated context**. Hologram's `DispatchContext` is this saturated context:
all shapes, dtypes, and constants are resolved at compile time, so every execution is a single KV
lookup.

**Derivation chain** (each step traces to a UOR Foundation axiom):

| Step | Prism Identity | Foundation Axiom | Role in hologram |
|------|---------------|-----------------|------------------|
| 0 | PI_3 (inference monotonicity) | SR_1 (freeCount non-increasing) | Shape propagation only converges |
| 1 | PA_1 (accumulation associativity) | SR_10 (Church-Rosser confluence) | Parallel level order doesn't affect final state |
| 2 | PL_3 (lease completeness recovery) | MC_6 (full coverage → σ=1) | All levels compose back to full saturation |
| 3 | PK_2 (composition O(1) resolution) | MC_7 (stepCount=0 on saturated context) | KV lookup is O(1) |

### Additional Identities in Use

| Prism Identity | Foundation Basis | Hologram Component |
|---------------|-----------------|-------------------|
| PA_4 (base binding preservation) | SR_1 + bitmask OR irreversibility | `DispatchContext` immutability; PM_5 rollback |
| PI_1 (inference idempotence) | CC_1 + SC_5 | `KvStore` result caching |
| PD_1 (dispatch determinism) | AD_1 (addressing bijection) | `float_dispatch.rs` determinism |
| PD_2 (dispatch type safety) | CB_5 (fiber sufficiency) | dtype-gated dispatch |
| PL_2 (lease disjointness) | SR_9 (ContextLease fiber disjointness) | `ParallelLevel` isolation |
| PX_5 (infeasibility detection) | CB_5 + SR_5 (ContradictionBoundary) | `CompileError` taxonomy |
| PM_5 (transaction atomicity) | PA_4 (base preservation = free rollback) | `PrismModule::execute()` error contract |
| PK_3 (parallelism bound) | MC_8 (work ≤ ⌈n/k⌉ for k leases) | Level fusion quality criterion |

---

## Three-Space Model

Hologram's crates follow the Prism space classification. Each space has distinct mutability and
deployment guarantees:

| Space | Prism Definition | Hologram Crates |
|-------|-----------------|-----------------|
| **kernel** | Deployment-immutable; contains foundation operations and algebraic laws | `hologram-core`, `hologram-ir`, `hologram-archive`, `hologram-shapes` |
| **bridge** | Prism-computed; derives from kernel crates via explicit composition laws | `hologram-fused-component`, `hologram-compiler`, `hologram-async` |
| **user** | Application-configurable; exposed at system boundaries | `hologram-ffi`, `hologram-cli`, `hologram-bench` |

**Rule**: kernel crates must not depend on bridge or user crates. Bridge crates must not depend on
user crates. This enforces the one-way information flow required by the Prism space hierarchy.

---

## Crate Dependency Graph

```
hologram-foundation (kernel — re-export shim for uor-foundation v0.2.0)
hologram-core (kernel)
    └── hologram-ir (kernel)
            └── hologram-archive (kernel)
hologram-shapes (kernel — Shape declarations + PrismModule trait)
    └── hologram-fused-component (bridge — first PrismModule impl)
            └── hologram-compiler (bridge — Compiler + SourceInput)
                    └── hologram-async (bridge)
                            └── hologram-ffi (user)
                            └── hologram-cli (user)
                            └── hologram-bench (user)
```

---

## Tape Execution Pipeline

Hologram compiles a dataflow graph into a flat, pre-resolved instruction tape where every data path
is an integer index into a buffer arena. This eliminates per-node op matching, HashMap lookups, and
vtable indirection at execution time — realising the PP_1 O(1) resolution claim.

### Stage 1: Graph — Edges Define Data Paths

Each `Node` in the graph connects to its inputs via `InputSlot`, which names a source `NodeId` and
an `output_port`. The graph exposes `predecessors()` and `successors()` for traversal, plus
`build_successor_index()` for O(1) reverse-edge lookups used by the fusion and scheduling passes.

**Key types**: `Node`, `InputSlot`, `InputSource` (`hologram-ir/src/graph/node.rs`)

### Stage 2: Schedule — Paths Become Parallel Levels

A modified Kahn's topological sort partitions the graph into `ParallelLevel`s. Nodes within a level
have no mutual dependencies and can execute concurrently. This satisfies **PL_2 (lease
disjointness)**: nodes in a level hold non-overlapping buffer leases, and all predecessors reside in
strictly earlier levels.

Critical-path analysis (DP over the topological order) computes the longest dependency chain, giving
the parallelism ratio `total_nodes / critical_path_length`.

**Key types**: `ExecutionSchedule`, `ParallelLevel` (`hologram-ir/src/schedule/`)

### Stage 3: Tape Compilation — Paths Become Arena Indices

`build_tape()` compiles the schedule into a flat `EnumTape`:

```
EnumTape {
    instructions: Vec<TapeInstruction>,   // flat array in execution order
    level_offsets: Vec<usize>,            // boundaries between parallel levels
}
```

Each `TapeInstruction` pre-resolves all data routing:

| Field | Purpose |
|-------|---------|
| `kernel: TapeKernel` | Operation as an enum variant (not boxed trait) |
| `input_indices: Vec<u32>` | Arena slots to read inputs from |
| `output_idx: u32` | Arena slot to store the result |
| `passthrough: bool` | Zero-copy move (identity/reshape ops) |
| `can_reuse_input: bool` | In-place mutation for single-consumer unary ops |
| `weight_offset_hint: u32` | Prefetch hint for LUT-GEMM weight pages |
| `output_byte_hint: u32` | Pre-computed output size for arena pre-warming |

All graph edges are resolved to integer indices at this stage. No graph traversal occurs at runtime.

**Key types**: `EnumTape`, `TapeInstruction`, `TapeKernel` (`hologram-fused-component/src/tape.rs`),
`build_tape()` (`hologram-fused-component/src/tape_builder.rs`)

### Stage 4: Execution — Index-Based Data Routing

`BufferArena` is a flat `Vec<Option<ArenaBuffer>>` indexed by `NodeId::index()`, giving O(1) lookup
without hashing. The executor processes instructions level-by-level, selecting one of four fast
paths per instruction:

| Fast Path | Condition | Mechanism |
|-----------|-----------|-----------|
| Passthrough | `passthrough = true` | `arena.move_slot(src → dst)` — zero-copy |
| In-place unary | `can_reuse_input = true` | Mutate input buffer, then move slot |
| Inline dispatch | Simple unary/binary ops | Direct f32 access, compute into recycled buffer |
| General dispatch | Everything else | Gather input refs into SmallVec, dispatch to backend |

After arena pre-warming (`prewarm_arena()`), steady-state execution is **zero-allocation**: the
`swap_insert_with_elem_size()` method exchanges output buffers with the arena's existing allocation,
so the kernel writes into a recycled `Vec<u8>` and the arena reclaims the old one.

The executor also **prefetches ahead**: while instruction N executes, instruction N+1's input data
and weight pages are prefetched into cache.

**Key types**: `BufferArena`, `ArenaBuffer` (`hologram-fused-component/src/buffer/arena.rs`)

### Fusion — Path Shortening

Before tape compilation, a single topological pass applies five complementary optimisation passes
that shorten the graph, eliminate intermediate buffers, and reduce dispatch overhead. Topological
order guarantees that predecessors are processed before their successors, so both forward-looking
(epilogue) and backward-walking (view) fusions compose correctly in one pass.

**Key entry point**: `analysis::analyze()` (`hologram-ir/src/analysis/mod.rs`)

#### 1. Constant Folding

Operations whose inputs are all compile-time constants are evaluated immediately and replaced with
a single `Const` node. Handles unary/binary `LutOp`, `PrimOp`, `FusedView`, and
`RingPrimUnary`/`RingPrimBinary` variants.

**Effect**: eliminates computation entirely and shrinks the graph.

**Key file**: `hologram-ir/src/analysis/constant_folding.rs`

#### 2. View Fusion (Q0 — byte domain)

Backward-walks each node to find chains of byte-domain unary ops and composes them into a single
256-byte lookup table (`ElementWiseView`). A chain like `Sigmoid → Relu → Gelu` becomes one
`FusedView` node — one array access regardless of chain length.

- **Algebraic fast path**: involutions such as `Neg∘Neg` and `Bnot∘Bnot` are detected and replaced
  with `Passthrough` (zero-cost identity) without materialising a table.
- **Size**: each `ElementWiseView` is exactly 256 bytes, cache-line aligned for efficient L1 access.

**Effect**: replaces N-node chains with a single node; eliminates all intermediate buffers.

**Key file**: `hologram-ir/src/analysis/view_detection.rs`

#### 3. W16 View Detection (16-bit domain)

Mirrors W8 view detection for operations at the W16 ring level
(`RingPrimUnary`/`RingPrimBinary` with `RingLevel::Q1`). Produces a 128 KB
`ElementWiseView16` table (heap-allocated via `Box` to avoid stack overflow).
The same involution fast path applies at W16.

- **Ring-level safety**: never fuses across ring-level boundaries (W8 → W16 remains intact).

**Key file**: `hologram-ir/src/analysis/q1_view_detection.rs`

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

**Key file**: `hologram-ir/src/analysis/float_fusion.rs`

#### 5. Common Subexpression Elimination (CSE)

Hash-based deduplication of nodes with identical `(op, sorted_predecessors)` signatures. All
successors of a duplicate node are rewired to the canonical node.

- **Commutative-aware**: predecessor lists are only sorted for commutative ops (`Add`, `Mul`) to
  preserve semantics of non-commutative ops (`Sub`, `Div`).
- **Purity check**: only deduplicates pure ops; skips `Input`, `Output`, `CallSubgraph`.

**Key file**: `hologram-ir/src/analysis/cse.rs`

#### Fusion → Tape mapping

The fused graph is lowered into a flat tape of `TapeKernel` instructions. Each fused graph node
maps to a specific kernel variant — `LutView`, `LutView16`, `InlineMatMulActivation`,
`InlineConv2dBiasActivation`, `FusedRmsNormActivation`, etc. — so the executor sees fully
resolved instructions with **zero pattern-detection overhead** at runtime.

**Key types**: `TapeKernel` (`hologram-fused-component/src/tape.rs`), tape builder
(`hologram-fused-component/src/tape_builder.rs`)

### Prism Grounding

The tape pipeline realises several Prism identities:

- **PP_1** — Pre-resolution of all paths at compile time means execution is a single O(1) lookup
  per instruction on the saturated context (the arena + tape).
- **PL_2** — Level boundaries in `level_offsets` guarantee buffer-lease disjointness within each
  level.
- **PA_1** — Accumulation associativity means the order of operations within a level does not affect
  the final result, enabling safe parallelism.

---

## Compilation Pipeline (v0.2.0 Structure-Finder)

Per Prism section 4 of the SCS framework, hologram's compiler is a *structure-finder*, not a
constructor. The pipeline is a flat sequence of finder passes that produce a `.holo` archive — no
state machine, no certificate store, no per-stage budget tracking. This shape is the result of the
v0.2.0 conformance-first refactor that deleted the v0.1.4 7-stage cascade engine.

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `Compiler::compile(SourceInput::Graph(g))` | Raw Graph — wrapped via `unit_from_graph()` |
| `Compiler::compile(SourceInput::Term { unit, graph })` | Pre-built CompileUnit + pre-lowered Graph |
| `Compiler::compile(SourceInput::TermSource { source, level, ... })` | UOR term language → parse → preflight → lower → emit |
| `compile(graph)` | Convenience wrapper for the Graph variant |
| `compile_from_source(source, ...)` | Convenience wrapper for the TermSource variant |

### Finder Passes

The pipeline runs the analyses in `hologram-ir::analysis` directly, plus archive emission:

| # | Pass | Purpose |
|---|------|---------|
| 1 | `precision::promote_prim_ring_levels` | Observable-guided ring-level inference (W8/W16/W24) |
| 2 | `analysis::analyze` | Pattern detection: constant folding, view detection, float fusion, CSE |
| 3 | `verify_assertions` | Compile-time assertion check on the unit's term arena |
| 4 | `ExecutionSchedule::build` | Topological-level scheduling (Kahn's algorithm) |
| 5 | `liveness::compute_liveness` + `workspace::plan_workspace` | Buffer lifetime intervals + slot reuse |
| 6 | `qedl::insert_qedl_boundaries` | Domain-crossing detection + grounding validation |
| 7 | `tape_builder::build_tape` | Build the execution tape |
| 8 | Archive emission | `LayerHeader` + `CompileUnitMeta` + `ConformanceShapeSection` via `HoloWriter` |

### Conformance Shape Declaration

Every archive emitted by `hologram-compiler` carries a `ConformanceShapeSection`
(`SECTION_CONFORMANCE_SHAPE`, kind 6) declaring which `Shape` the compiled tape conforms to. The
loading `PrismModule` validates this declaration against its expected shape before any execution
proceeds. This is the v0.2.0 conformance-first contract: an archive is not just a blob of compiled
bytes, it is a value in `Val(F)` for a specific declared `F`.

### Archive Sections

Compiled archives include:
- `SECTION_LAYER_HEADER` (kind 2): Layer descriptors and execution schedule
- `SECTION_COMPILE_UNIT_META` (kind 5): Unit address, witt length, budget, term/binding/assertion counts
- `SECTION_CONFORMANCE_SHAPE` (kind 6): Shape ID + IRI + primitive count + Witt length range

**Key types**: `Compiler`, `SourceInput`, `PrismModuleRegistry`
(`hologram-compiler/src/compiler/mod.rs`),
`compile_via_finder_pipeline()` (`hologram-compiler/src/compiler/mod.rs`),
`ConformanceShapeSection` (`hologram-archive/src/section/conformance_shape.rs`)

---

## Witt Level Strategy

Hologram implements uor-foundation v0.2.0's Witt level hierarchy for ring-arithmetic acceleration.
Witt levels are named by bit width (`W8`, `W16`, `W24`, `W32`, …) rather than the v0.1.4 quantum
indices (`Q0`, `Q1`, `Q2`, `Q3`).

| Level | Bits | Ring | Strategy |
|-------|------|------|----------|
| W8 | 8 | Z/256Z | Full LUT (256 B per table) |
| W16 | 16 | Z/65536Z | Full LUT (128 KB per table) |
| W24 | 24 | Z/16777216Z | Hierarchical segmentation (~50 MB) |
| W32 | 32 | Z/4294967296Z | Algorithmic only (17 GB full LUT infeasible) |
| W40+ | 40+ | Z/2^nZ | Algorithmic with optional LRU cache |

W8 and W16 are fully realised in `hologram-core`. W24+ are algorithmic fallbacks.

### Term-language source syntax

The UOR term-language DSL — the surface that `compile_from_source(...)` and
`Compiler::compile(SourceInput::TermSource { ... })` parse — accepts **both** spellings for ring-level
annotations:

| Preferred (v0.2.0) | Legacy (v0.1.4) | Maps to |
|--------------------|-----------------|---------|
| `W8`  | `Q0` | `RingLevel::Q0` (8-bit ring) |
| `W16` | `Q1` | `RingLevel::Q1` (16-bit ring) |
| `W24` | `Q2` | `RingLevel::Q2` (24-bit ring) |
| `W32` | `Q3` | `RingLevel::Q3` (32-bit ring) |

Both forms produce **byte-identical archives**. New scripts should prefer the `W*` spelling because it
matches the rest of the type system (`WittLevel::W8`, etc.); the `Q*` spelling is retained as a
compatibility shim for existing source files. There is no plan to deprecate `Q*` — both spellings
parse to the same `Token::*` variants and route through the same `parse_witt_level()` arm in
`hologram-compiler/src/term_parser/parser.rs`.

Examples:

```text
let x : W8 = 42 ; neg(x)         # preferred
let x : Q0 = 42 ; neg(x)         # legacy, byte-identical
1000@W16                          # preferred Witt-level literal
1000@Q1                           # legacy, byte-identical
```

---

## Error Taxonomy (Prism PX_5)

Compilation failures are classified according to Prism PX_5's two infeasibility classes:

- **Insufficient** (`CompileError::InsufficientKernel`): the CB_5 fiber-sufficiency check fails
  because no dispatcher covers the required (op, dtype) pair. Resolution: register a kernel or
  lower to a supported dtype.

- **Contradictory** (`CompileError::ContradictoryConstraint`): the SR_5 ContradictionBoundary
  fires because two shape or type constraints conflict at the same node. Resolution: fix the model
  topology or add an explicit cast.
