# ADR-045: `hologram-ops` as the Single Source of Truth for Ops

## Status

Accepted (2026-04-27)

## Context

After ADR-044 (the `Op` trait), an op's identity (marker struct, trait
impl, semantic facts) lived in `hologram-ops`, but its executable form
(`Call` struct) and its kernel function lived in `hologram-transform`.
Adding one canonical op still touched two crates and six files:

- `hologram-ops/src/lib.rs` — marker struct + `Op` impl + `SemanticOp`
  variant + macro arm
- `hologram-transform/src/plan.rs` — `Call` struct + `KernelCall` variant
- `hologram-transform/src/kernels/<file>.rs` — kernel function
- `hologram-transform/src/planner.rs` — planner arm + helper
- `hologram-transform/src/chain.rs` — builder method
- `hologram-transform/src/executor.rs` — dispatch arm

This is the same conflation ADR-044 set out to solve, just one level
deeper. It is also at odds with the design direction stated in the
ADR-044 architecture sketch, which wants `hologram-ops` to be a single
home for "what ops exist in Hologram" — including, eventually, their
LUT generators (Plan 074 / `uor-foundation` 0.3.0).

## Decision

**`hologram-ops` owns everything about an op:** semantic identity, the
executable form (per-op `Call` struct), the reference kernel, and
(when implemented) the LUT generator. `hologram-transform` becomes a
pure orchestration crate — chain construction, planner-time address
resolution, and the executor loop that walks `KernelCall`s.

```
hologram-ops/                          orchestration:
  src/                                 hologram-transform/
    lib.rs                               src/
    span.rs           SlotSpan             address.rs        AddressRef, TensorId
    kernels/                               buffer.rs         BufferSet
      mod.rs          KernelCall + dispatch chain.rs         TransformChain
      add.rs          AddCall, fwd, bwd     error.rs
      binary.rs       BinaryCall, sub/mul/div executor.rs    walks CompiledPlan
      unary.rs        UnaryCall + 18 ops    plan.rs          CompiledPlan envelope
      matmul.rs       MatMulCall, fwd, bwds planner.rs       chain → CompiledPlan
      softmax.rs      SoftmaxCall + log
      reshape.rs      ReshapeCall
      shape.rs        Transpose/Slice/Concat
      norm.rs         5 norm variants
      fused.rs        FusedSwiGlu
      conv.rs         Conv2dCall
```

The dependency direction stays one-way: `hologram-transform` depends on
`hologram-ops`. Nothing depends in the other direction.

### Why move the kernels here

1. **Cohesion.** A canonical op is *defined* by its semantic identity
   *and* its computation. Splitting them across crates means a reader
   chasing "what does Add actually do?" follows two `cd`s. Co-locating
   them collapses that to one file.
2. **LUT alignment.** The user's stated invariant is that every op needs
   *both* a LUT (identity) *and* a transformation (computation). When
   the LUT layer is implemented, it belongs alongside the kernel — not
   in a third crate. ADR-045 puts ops in the right home for that
   landing.
3. **Architecture-layer correctness.** `hologram-ops` is in the
   *kernel* tier per `architecture.md`. Per-op kernels are kernel
   artefacts; they belong here, not in a bridge-tier orchestration
   crate.
4. **One file per op (almost).** Adding a new op is now: edit the per-op
   kernel module + add a `SemanticOp` variant + `KernelCall` variant +
   `dispatch` arm + planner arm + builder method. The semantic and
   executable parts of the op are both in one file in `hologram-ops`.

### What `hologram-transform` keeps

- `AddressRef`, `TensorId`, `LayoutId` — chain-level identity (these
  are about the *plan*, not about ops).
- `TransformChain`, builder, `CompiledPlan`, `AddressTable`,
  `WorkspaceLayout` — orchestration types.
- `Executor`, `BufferSet` — runtime walk + storage.
- `planner.rs` — `SemanticOp` → `KernelCall` lowering, including
  shape-derived attribute propagation (e.g. reading tensor shapes to
  populate `Conv2dCall.h_in`).

These are not op-related; they are the chain → plan → execute
orchestration on top of the ops vocabulary.

### Bridge to legacy paths

`hologram-graph::GraphOp::Compute(SemanticOp)` and
`hologram-graph::graph::op::legacy_float_op()` are unchanged. The
legacy `FloatOp` execution path through `hologram-exec` continues to
work. ADR-045 only restructures *new* code; it does not migrate exec
or backend to consume `KernelCall` directly. Those moves are tracked
separately.

### Alignment with hologram invariants

| Invariant                            | How this ADR preserves it                              |
|--------------------------------------|--------------------------------------------------------|
| Closed serialisation surface         | `SemanticOp` enum unchanged on the wire                |
| Exhaustive matching                  | `KernelCall` enum exhaustive at the `dispatch` site    |
| No virtual dispatch in kernels       | Kernels are concrete fns; `dispatch` is enum match     |
| O(1) lookup, zero-copy               | `SlotSpan` + flat-buffer model unchanged               |
| No allocation in hot paths           | Executor still iterates a pre-built `Box<[KernelCall]>`|
| Single source of truth for ops       | `hologram-ops` now owns identity + kernel + (future) LUT|

## Consequences

- `hologram-ops` grows from ~1k lines (semantic-only) to ~3k lines
  (semantic + kernels). It now depends on `libm`. This is
  proportionate: the crate's scope expanded to match its name.
- `hologram-transform` shrinks from ~1.5k lines (orchestration +
  kernels) to ~600 lines (orchestration only). Its only dependency on
  the op layer is via `hologram_ops::*`.
- The "to add an op" recipe becomes: write one file in
  `hologram-ops/src/kernels/<name>.rs` containing the `Call` struct +
  forward kernel (+ backward where applicable), then six small touch
  points: `SemanticOp` variant, `Op` impl, `dispatch` arm, planner
  arm, builder method, integration test. The bulk of the logic is in
  one file.
- Future LUT generators per op land in the same per-op file, completing
  the "one place per op" structure. Concretely: when Plan 074 lands the
  `uor-foundation` 0.3.0 address API, an `Op::lut()` trait method (and
  per-op `lut()` implementations) will be added to each
  `hologram-ops/src/kernels/<op>.rs` alongside the existing `Call`
  struct and kernel function. At that point each op file holds *all
  three* aspects — identity, transformation, addressing — and the
  user's invariant ("every op needs a LUT *and* be transformed") is
  satisfied by construction. Tracked as Sprint 37 Phase 2.
- 65 workspace test suites pass, 0 failures across the move (39 ops
  unit tests + 14 transform unit tests + 30 transform integration
  tests, plus the rest of the workspace).

## Alternatives considered

- **Keep kernels in `hologram-transform`.** Status quo before this ADR.
  Rejected for the cohesion / LUT-alignment / layering reasons above.
- **Per-op crates (`hologram-op-add`, `hologram-op-matmul`, …).**
  Rejected as over-fragmentation; 36 sub-crates would inflate the build
  graph for no real isolation gain.
- **Keep `Call` structs in `hologram-transform`, move only the kernel
  functions.** Rejected because the `Call` struct is the executable
  form of the op — it belongs with the kernel that consumes it.
  Splitting them would maintain the original spread.
