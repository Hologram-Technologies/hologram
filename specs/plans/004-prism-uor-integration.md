# 004 — Prism Ontology Integration

## Context

The UOR team published the Prism ontology v1.3.0 — the "Polymorphic Resolution and Isometric
Symmetry Machine" — as the algebraic runtime layer extending the UOR Foundation operations graph.
Every Prism operation is a composition of foundation operations; every Prism identity derives from
foundation axioms.

Hologram already satisfies most Prism identities structurally — it just doesn't name them.
This sprint grounds hologram's design in Prism's formal vocabulary, adds the missing invariant
assertions, and extends the error taxonomy. No functional changes; purely documentation and
lightweight structural additions.

## Key Prism Identities

**PP_1 (Pipeline Unification)** — `κ(λ_k(α*(ι(s,·))),C) = resolve(s,C)`.
The entire dispatch→inference→accumulate→compose chain collapses to O(1) on a saturated context.
Derivation chain: PI_3 (monotone inference) → PA_1 (Church-Rosser accumulation) → PL_3 (lease
completeness) → PK_2 (O(1) on saturated context). This is the formal proof of hologram's O(1) claim.

**PA_4 (Base Binding Preservation)** — SR_1 monotonicity + OR-bitmask irreversibility guarantee
pinned fibers are never unpinned. Grounds `DispatchContext` immutability.

**PI_1 (Inference Idempotence)** — CC_1 + SC_5: repeated resolution on a saturated context
yields no state change. Grounds `KvStore` caching.

**PD_1/PD_2 (Dispatch Determinism/Type Safety)** — AD_1 bijection (same content → same address
→ same binding) and CB_5 fiber sufficiency. Grounds `float_dispatch.rs`.

**PL_2 (Lease Disjointness)** — SR_9: leased fibers are disjoint. Grounds `ParallelLevel` isolation.

**PX_5 (Infeasibility Detection)** — two failure modes: Insufficient (CB_5 sufficiency fails →
missing kernel) and Contradictory (SR_5 ContradictionBoundary → shape/type conflict).

**PM_5 (Transaction Atomicity)** — PA_4 base preservation makes rollback free. Grounds
`KvExecutor` error recovery contract.

## Tasks

### Task 1: Annotate DispatchContext as SaturatedContext
**File**: `crates/hologram-exec/src/eval/executor.rs`

Expand the `DispatchContext` doc comment to cite PP_1, PA_4, and PI_1. No code changes.

### Task 2: PX_5 Error Taxonomy in hologram-compiler
**File**: `crates/hologram-compiler/src/error/mod.rs`

Add two new `CompileError` variants mapping to Prism PX_5's two infeasibility classes:
- `InsufficientKernel { op, dtype }` — CB_5 sufficiency failure (missing kernel)
- `ContradictoryConstraint { detail }` — SR_5 contradiction (shape/type conflict)

### Task 3: PL_2 Disjointness on ParallelLevel
**File**: `crates/hologram-graph/src/schedule/levels.rs`

Add PL_2 citation to `ParallelLevel` doc comment. The topological invariant (Kahn's algorithm
assigns each node to exactly one level) structurally enforces PL_2 — add a doc note explaining
this.

### Task 4: PM_5 Atomicity Contract on KvExecutor
**File**: `crates/hologram-exec/src/eval/executor.rs`

Extend `KvExecutor::execute()` doc comment: on error, `DispatchContext` is unchanged (PA_4 base
preservation = free rollback); caller may safely retry with different `GraphInputs`.

### Task 5: Prism Space Classification in archon.yaml
**File**: `archon.yaml`

Add `prism_space` field to each crate entry:
- kernel: hologram-core, hologram-graph, hologram-archive
- bridge: hologram-exec, hologram-compiler, hologram-async
- user: hologram-ffi, hologram-cli, hologram-bench

### Task 6: Architecture Documentation
**File**: `specs/docs/architecture.md` (new file)

Document:
- Prism algebraic grounding (PP_1 derivation chain)
- Three-space model (kernel/bridge/user) with crate mapping
- Conceptual mapping table (Prism concept → hologram component)

## Acceptance Criteria

- `just ci` passes (tests + clippy + fmt)
- `DispatchContext` doc cites PP_1, PA_4, PI_1
- `CompileError` has `InsufficientKernel` and `ContradictoryConstraint` with Prism citations
- `ParallelLevel` doc cites PL_2
- `archon.yaml` has `prism_space` for every crate
- `specs/docs/architecture.md` exists with Prism grounding section
