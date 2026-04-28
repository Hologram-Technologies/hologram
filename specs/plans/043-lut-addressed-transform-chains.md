# Plan 043: LUT-Addressed Transformation / Mutation Chains

**Status:** Phase 1–4 in progress
**Date:** 2026-04-27
**ADR:** [adrs/043-lut-addressed-transform-chains.md](../adrs/043-lut-addressed-transform-chains.md)

## Goal

Build a Hologram-native transformation system in which:

- **LUT** (sourced from `uor-foundation`) gives **addresses**.
- **Transform chains** define **computation** (forward + backward).
- **Planning** lowers a chain into a fully-resolved **CompiledPlan**.
- **Execution** runs only a slice of `KernelCall`s — no allocation, no
  virtual dispatch, no runtime algorithm selection.

The first deliverable supports forward + backward for **ADD** and **MatMul**.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  uor-foundation (LUT / identity)                                 │
│    AddressRef ← TensorId ← LayoutId ← RegionId                   │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  hologram-transform: chain (semantics)                           │
│    OpKind, BackwardRule, TransformNode, TransformChain           │
└──────────────────────────────────────────────────────────────────┘
                       │      compile-time only
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  hologram-transform: planner                                     │
│    AddressTable, WorkspaceLayout, Box<[KernelCall]>              │
└──────────────────────────────────────────────────────────────────┘
                       │      run-time only
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  hologram-transform: executor                                    │
│    fixed dispatch, single contiguous BufferSet                   │
└──────────────────────────────────────────────────────────────────┘
```

## Crate layout

```
crates/hologram-transform/
  Cargo.toml
  src/
    lib.rs           — module surface + re-exports
    address.rs       — AddressRef, TensorId, NodeId, RegionId, LayoutId
    op.rs            — OpKind, BackwardRule
    chain.rs         — Tensor, TransformNode, TransformChain (+ builder)
    error.rs         — PlanError, ExecError
    plan.rs          — CompiledPlan, KernelCall, SlotSpan, AddressTable,
                       WorkspaceLayout
    planner.rs       — TransformChain → CompiledPlan
    buffer.rs        — BufferSet (single owned allocation)
    executor.rs      — Executor::run_forward / run_backward
    kernels/
      mod.rs
      add.rs         — add, add_grad
      matmul.rs      — matmul, matmul_grad_a, matmul_grad_b
  tests/
    transform_chain.rs
```

Workspace `Cargo.toml` gains `hologram-transform` as a member and as a
workspace dependency.

---

## Phase 1 — Semantic transform chain

**Deliverable:** `OpKind`, `BackwardRule`, `Tensor`, `TransformNode`,
`TransformChain` with a builder API.

- `OpKind` is `Copy + 'static`. Initial variants: `Add`, `MatMul`.
- `BackwardRule` is `Copy + 'static`. Initial variants: `AddBackward`,
  `MatMulBackward`.
- `TransformNode` carries `inputs: SmallVec<[AddressRef; 4]>` and
  `outputs: SmallVec<[AddressRef; 2]>`.
- `TransformChain` stores tensors and nodes as `Vec`s — allocation here is
  fine because chain construction is a compile-time operation.

**Tests:** chain construction, node lookup, op-arity sanity, backward-rule
attachment.

---

## Phase 2 — Address table and planner

**Deliverable:** `AddressTable`, `WorkspaceLayout`, planner that resolves
every `AddressRef` to a `SlotSpan`.

- `SlotSpan { offset, len }` indexes into the single `BufferSet` storage.
- `AddressTable: Box<[SlotSpan]>` indexed by `TensorId.0 as usize` — O(1) lookup.
- `WorkspaceLayout { total_elements }` is the single allocation size for the
  forward pass.
- For each tensor with `requires_grad = true`, the planner allocates a
  parallel "grad" slot and stores it in `grad_table: Box<[SlotSpan]>`.
- Planner clears tape state before re-planning (planner is pure: same input
  ⇒ same plan).

**Tests:** address resolution stability, total-elements accounting,
grad-slot allocation only when `requires_grad` is set.

---

## Phase 3 — Forward and backward lowering

**Deliverable:** chain → `CompiledPlan { forward, backward }`.

- Forward lowering walks `chain.nodes` in order and emits one or more
  `KernelCall`s per node.
- Backward lowering walks `chain.nodes` **in reverse** and emits backward
  `KernelCall`s based on `node.backward`.
- `KernelCall` variants (initial set):
  - `Add { a, b, c, len }`
  - `AddGrad { dc, da, db, len }` — accumulate `dC` into both `dA` and `dB`
  - `MatMul { a, b, c, m, k, n }`
  - `MatMulGradA { dc, b, da, m, k, n }` — `dA += dC @ Bᵀ`
  - `MatMulGradB { a, dc, db, m, k, n }` — `dB += Aᵀ @ dC`
- All offsets, dimensions, and lengths in `KernelCall` are precomputed;
  the executor only dereferences.

**Tests:** ADD forward, ADD backward, MatMul forward, MatMul backward —
each verified against a hand-rolled reference computation.

---

## Phase 4 — CPU reference executor

**Deliverable:** `Executor::run_forward(plan, buffers)` and
`Executor::run_backward(plan, buffers)`.

- Executor is a unit struct; methods are `pub fn`, take only references.
- Hot loop is a `for call in plan.forward.iter()` + `match call`. No `dyn`,
  no `Box`, no `Vec::push`.
- `BufferSet` owns a single `Box<[f32]>`. Subslices are produced via
  `split_at_mut` for write+read aliasing safety.
- `run_backward` calls `plan.backward.iter()` only — chain traversal is
  never re-derived at runtime.

**Tests:** end-to-end ADD train step, end-to-end MatMul train step,
non-allocation assertion (no `Vec` constructed inside the executor's hot
loop, verified by a `#[deny(clippy::disallowed_methods)]`-style review +
inline `debug_assert!`s on capacity).

---

## Phase 5 — Fusion and backend specialisation (deferred)

Out of scope for this PR but designed in:

- Add `KernelCall::FusedAddRelu`, `KernelCall::MatMulBias`, etc. as new
  variants. Fusion is a planner pass that rewrites the
  `Box<[KernelCall]>` slice — the executor is unchanged.
- Backend specialisation (Metal, WebGPU, Atlas ISA) is implemented as an
  alternative executor that consumes the *same* `CompiledPlan`. The chain
  and planner are backend-agnostic.

---

## Phase 6 — UOR / `uor-foundation` integration (deferred)

- `AddressRef` will become an alias for / wrapper around
  `uor_foundation`'s LUT-resolved address type, once Plan 074 lands and
  exposes a stable trait.
- For now, `AddressRef` is a Hologram-native struct with the same
  semantics: a typed pointer into a LUT layer that the executor never
  touches.

---

## Validation

- `cargo fmt --all`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test -p hologram-transform`
- Workspace `just ci`

## Out of scope

- ONNX / `.holo` archive integration.
- Quantised / ring-arithmetic kernels (those continue to live in
  `hologram-exec`).
- Multi-device scheduling.
