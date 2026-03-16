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
