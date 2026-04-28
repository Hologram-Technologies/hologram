# ADR-043: LUT-Addressed Transformation / Mutation Chains

## Status

Accepted (2026-04-27)

## Context

`hologram` distinguishes itself from conventional tensor runtimes by:

1. Treating identity / addressing as an **ontological lookup**, sourced from
   `uor-foundation`, rather than an ad-hoc index into a Vec.
2. Pushing every algorithmic decision (kernel choice, fusion, layout, workspace
   sizing, backward graph) to **compile time**.
3. Maintaining hot-path invariants of **O(1) lookup**, **zero-copy**, and
   **fixed dispatch** during execution.

`uor-foundation` provides identity, ring algebra, and LUT-style addressing
primitives but is deliberately **execution-free**: it cannot describe a
forward pass, materialise a gradient, or own a workspace. Hologram has, until
this ADR, conflated three concerns into the existing `op`/`Tape` stack:

- *what is this object* (identity / address),
- *what computation will produce it* (transform),
- *how is that computation executed* (planned kernel calls).

That conflation is the source of recurring friction:

- Backward computation has no first-class representation (it is reconstructed
  ad-hoc by graph traversal at runtime).
- Address resolution and kernel dispatch are interleaved, making it hard to
  reason about whether a code path allocates.
- Adding a backend (Metal, WebGPU, Atlas ISA) requires re-deriving the
  transform graph instead of lowering a single planned representation.

## Decision

Introduce a new crate, **`hologram-transform`**, that splits the runtime into
four explicit layers:

```
UOR / LUT layer            (identity)
  ─ symbolic objects, typed addresses, shape / layout / region metadata
  ─ forward and backward address derivation
  ─ no execution, no allocation

hologram-transform: chain  (semantics)
  ─ TransformNode { OpKind, inputs, outputs, BackwardRule }
  ─ pure descriptors — no allocation, no execution, no dispatch

hologram-transform: planner (compile-time)
  ─ resolves AddressRefs → concrete (offset, len, layout)
  ─ allocates workspace, precomputes strides
  ─ emits forward and backward KernelCall slices

hologram-transform: executor (run-time)
  ─ runs compiled kernels only
  ─ fixed dispatch (enum match), no virtual calls in kernels
  ─ no heap allocation in the hot loop
```

### Why LUT is not execution

A LUT entry is a *name* for a piece of data (a tensor, a region, a layout). It
answers "*which* object" — never "*how* to compute it". Bundling execution into
the address layer would couple identity to a particular backend and make the
same tensor unrepresentable across CPU, GPU, and Atlas without re-keying. The
LUT is therefore strictly an immutable source of truth for `AddressRef`
resolution, never a kernel dispatcher.

### Why transform descriptors are semantic only

`TransformNode` is a value type. It is `Copy` where possible, owns no
allocation beyond its `inputs`/`outputs` `SmallVec`, and contains **no**
function pointers, trait objects, or workspace handles. This is what makes a
chain fully analysable at compile time: a planner can rewrite, fuse, or
schedule it without losing fidelity, and an audit can prove that no chain
construction allocates from a kernel hot path.

### Why backward computation is planned ahead of time

Backward passes are notorious for runtime graph traversal, dynamic dispatch,
and surprise allocation. By lifting `BackwardRule` into the chain and lowering
it to ordinary `KernelCall`s during planning, the executor sees no difference
between forward and backward — both are just sequences of pre-resolved kernel
calls. This preserves the O(1) and zero-allocation invariants for training as
well as inference.

For ADD:

```
forward  : C  = A + B
backward : dA += dC
           dB += dC
```

For MatMul:

```
forward  : C  = A @ B
backward : dA += dC @ Bᵀ
           dB += Aᵀ @ dC
```

Both backward forms are emitted as concrete kernel calls (`AddGrad`,
`MatMulGradA`, `MatMulGradB`) by the planner.

### Why compiled plans own dispatch / workspace / address resolution

`CompiledPlan` is the contract between the compile-time and run-time worlds:

- `address_table` maps every `AddressRef` to a concrete `SlotSpan`.
- `workspace` declares the single contiguous allocation the executor needs.
- `forward` / `backward` are `Box<[KernelCall]>` — sized once at planning,
  iterated by index at execution.

This makes the executor's job mechanical: walk a slice, dispatch via a fixed
enum match, read and write into a single pre-allocated buffer. There is no
opportunity for the executor to introduce dynamic behaviour.

### Alignment with hologram invariants

| Invariant                          | Where it is enforced                            |
|------------------------------------|-------------------------------------------------|
| O(1) address lookup                | `AddressTable` is `Box<[SlotSpan]>` indexed by `TensorId.0 as usize` |
| Zero-copy hot paths                | `BufferSet` owns one `Box<[f32]>`; kernels take subslices |
| No dynamic allocation in execution | Planner sizes everything; `Executor::run_*` borrows only |
| No virtual dispatch in kernels     | `KernelCall` is an enum, dispatched by match; no `Box<dyn …>` |
| No runtime algorithm selection     | Planner picks variants; executor never branches on shape |
| Backward as planned kernels        | `BackwardRule` lowered to `KernelCall` at compile time |

## Consequences

- A new crate, `hologram-transform`, sits beside `hologram-graph` and
  `hologram-exec`. Existing crates are unchanged by this ADR.
- `BackwardRule` is the canonical place to add new differentiable ops. Adding
  Conv, Reduce, etc. is a planner-side change only.
- A future GPU / Atlas backend is implemented by adding a new
  `KernelCall` consumer (executor variant) — the chain and planner do not
  change. This keeps the chain → plan boundary stable across backends.
- `TransformChain` is intentionally simpler than the existing `Tape` (no
  cancellation, no kv-cache, no profiling). It is meant to be the foundation
  the existing tape eventually lowers into, not a replacement for it on day
  one. ADR-067 (compute backend rewrite) defines the longer-term migration.

## Alternatives considered

- **Add backward support directly to `Tape`.** Rejected because the tape is
  already coupled to weight loading, KV caches, and profiling; bolting on
  symbolic differentiation grows it in the wrong direction.
- **Re-use `hologram-graph::Graph` as the chain.** Rejected because `Graph`
  carries archive-loading and serialisation concerns that have nothing to do
  with transform semantics.
- **Generate backward code at compile time via macros.** Rejected as
  premature; an explicit `BackwardRule` enum keeps the data shape inspectable
  and forward-compatible with autodiff over a richer op set.
