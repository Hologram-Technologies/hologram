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
| PM_5 (transaction atomicity) | PA_4 (base preservation = free rollback) | `KvExecutor::execute()` error contract |
| PK_3 (parallelism bound) | MC_8 (work ≤ ⌈n/k⌉ for k leases) | Level fusion quality criterion |

---

## Three-Space Model

Hologram's crates follow the Prism space classification. Each space has distinct mutability and
deployment guarantees:

| Space | Prism Definition | Hologram Crates |
|-------|-----------------|-----------------|
| **kernel** | Deployment-immutable; contains foundation operations and algebraic laws | `hologram-core`, `hologram-graph`, `hologram-archive` |
| **bridge** | Prism-computed; derives from kernel crates via explicit composition laws | `hologram-exec`, `hologram-compiler`, `hologram-async` |
| **user** | Application-configurable; exposed at system boundaries | `hologram-ffi`, `hologram-cli`, `hologram-bench` |

**Rule**: kernel crates must not depend on bridge or user crates. Bridge crates must not depend on
user crates. This enforces the one-way information flow required by the Prism space hierarchy.

---

## Crate Dependency Graph

```
hologram-core (kernel)
    └── hologram-graph (kernel)
            └── hologram-archive (kernel)
                    └── hologram-exec (bridge)
                    │       └── hologram-compiler (bridge)
                    │               └── hologram-async (bridge)
                    │                       └── hologram-ffi (user)
                    │                       └── hologram-cli (user)
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

**Key types**: `Node`, `InputSlot`, `InputSource` (`hologram-graph/src/graph/node.rs`)

### Stage 2: Schedule — Paths Become Parallel Levels

A modified Kahn's topological sort partitions the graph into `ParallelLevel`s. Nodes within a level
have no mutual dependencies and can execute concurrently. This satisfies **PL_2 (lease
disjointness)**: nodes in a level hold non-overlapping buffer leases, and all predecessors reside in
strictly earlier levels.

Critical-path analysis (DP over the topological order) computes the longest dependency chain, giving
the parallelism ratio `total_nodes / critical_path_length`.

**Key types**: `ExecutionSchedule`, `ParallelLevel` (`hologram-graph/src/schedule/`)

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

**Key types**: `EnumTape`, `TapeInstruction`, `TapeKernel` (`hologram-exec/src/tape.rs`),
`build_tape()` (`hologram-exec/src/tape_builder.rs`)

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

**Key types**: `BufferArena`, `ArenaBuffer` (`hologram-exec/src/buffer/arena.rs`)

### Fusion — Path Shortening

Before tape compilation, optimisation passes shorten the graph:

- **View fusion**: chains of unary ops (e.g. Sigmoid → Relu) are composed into a single 256-byte
  LUT via `fuse_unary_chains()`, replacing multiple nodes with one `FusedView` node.
- **CSE**: duplicate subexpressions are merged, eliminating redundant computation paths.

These reduce both the instruction count and the critical path length.

**Key types**: `fuse_unary_chains()` (`hologram-graph/src/fusion/view_fusion.rs`)

### Prism Grounding

The tape pipeline realises several Prism identities:

- **PP_1** — Pre-resolution of all paths at compile time means execution is a single O(1) lookup
  per instruction on the saturated context (the arena + tape).
- **PL_2** — Level boundaries in `level_offsets` guarantee buffer-lease disjointness within each
  level.
- **PA_1** — Accumulation associativity means the order of operations within a level does not affect
  the final result, enabling safe parallelism.

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

Q0 and Q1 are fully realised in `hologram-core`. Q2+ are algorithmic fallbacks.

---

## Error Taxonomy (Prism PX_5)

Compilation failures are classified according to Prism PX_5's two infeasibility classes:

- **Insufficient** (`CompileError::InsufficientKernel`): the CB_5 fiber-sufficiency check fails
  because no dispatcher covers the required (op, dtype) pair. Resolution: register a kernel or
  lower to a supported dtype.

- **Contradictory** (`CompileError::ContradictoryConstraint`): the SR_5 ContradictionBoundary
  fires because two shape or type constraints conflict at the same node. Resolution: fix the model
  topology or add an explicit cast.
