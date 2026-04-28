# ADR-050: Canonical Kernels as Semantic Contract, Not Execution Path

## Status

Accepted (2026-04-27)

## Context

ADR-047 originally framed Stage 3 of the `FloatOp` deprecation as
"reorganise `hologram-exec/src/float_dispatch/` to dispatch on
`SemanticOp` directly". On closer inspection of the existing exec
dispatch, that framing turns out to be wrong:

- `hologram-exec`'s float dispatch is heavily optimised:
  monomorphised inner loops for hot ops (Relu, Add, Mul…), in-place
  `OutputBuffer` writes that avoid per-call allocation, and closure
  arguments structured to hint at autovectorisation. The dispatch
  is performance-critical for inference workloads.
- Canonical kernels in `hologram-ops` are reference implementations.
  They use a `&mut [f32]` storage + `SlotSpan` model with explicit
  scratch allocation in some cases (`attention.rs`), prioritise
  clarity over performance, and have no SIMD or in-place
  optimisations.
- Forcing exec to call canonical kernels via "lift to canonical →
  build temp workspace → run kernel → copy back" would **regress**
  every canonical-covered op's performance with no semantic gain.

The migration's actual goal was always **semantic clarity** — one
place to look up "what does op X mean" — not redundant execution.
Stage 3 needs reframing.

## Decision

Canonical kernels are the **semantic contract** for each op, not the
execution path:

- `hologram-ops/src/kernels/<op>.rs` defines what op X *means*. It
  is the reference correctness implementation, the test oracle, and
  the documentation of op semantics. Anyone implementing op X for
  any backend conforms to this kernel's behaviour.
- `hologram-exec/src/float_dispatch/` and any future
  `hologram-backend` (Metal, WebGPU, Atlas) provide *optimised
  implementations* that conform to the canonical contract. They are
  not required to call canonical kernels at runtime.
- **Conformance is enforced by tests**, not by call-graph topology.
  The conformance test asserts: for any input, the optimised
  implementation produces the same output (within tolerance) as the
  canonical kernel.

This reframes Stage 3 from "rewrite exec to call canonical" to
"establish a conformance test contract between exec and canonical".

### Conformance test infrastructure

A new `tests/canonical_conformance.rs` in `hologram-exec` provides
the cross-check infrastructure:

```rust
fn assert_conformance<F>(
    op_name: &str,
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    legacy_dispatch: F,
    canonical_kernel: impl FnOnce(&mut [f32]),
    workspace_layout: WorkspaceLayout,
)
where
    F: FnOnce(&[&[u8]], &[Vec<usize>]) -> Vec<u8>,
```

Per-op cross-check tests:
- Run exec's `dispatch_float_with_shapes` on byte-encoded inputs.
- Build an equivalent canonical `KernelCall` against a fresh f32
  workspace with the same input values.
- Run `hologram_ops::dispatch`.
- Assert outputs match within `1e-5` tolerance.

The infrastructure is parameterised so adding a new op's
conformance check is one new test function pointing at the relevant
canonical / exec entry points.

### What this preserves and what it gives up

**Preserves:**
- Exec's hot-path performance — no rewrite, no extra copies, no
  allocator churn in the inner loop.
- Backend independence — Metal / WebGPU executors provide their
  own optimised paths and conform to the same canonical contract.
- The single-source-of-truth property — questions of "what does
  this op mean" have one answer (the canonical kernel).

**Gives up:**
- The vision of "exec literally calls `hologram_ops::dispatch`".
  That was an implementation idea; the underlying goal (one source
  of truth) is met without it.
- Compile-time enforcement that exec's behaviour matches canonical.
  Replaced by test-time enforcement.

### Stage 4 implications

`FloatOp` doesn't migrate to "internal-only" via path replacement
anymore. Instead:

- `FloatOp` stays as exec's *internal* dispatch encoding.
- Public API surface (`hologram::FloatOp` re-export) is removed in
  Stage 4 — *that* part of ADR-047 still applies.
- New code constructs canonical (`GraphOp::Compute(SemanticOp)`)
  via `GraphOp::from_float`. New backends consume canonical
  contracts, not `FloatOp`.
- The legacy `GraphOp::Float(FloatOp)` variant remains for archive
  compatibility (rkyv format).

## Consequences

- Stage 3 ships as **conformance test infrastructure** rather than
  a 1k-2k-line exec rewrite.
- The migration's goals (canonical as single source of truth,
  ADR-046 one-way bridge, ADR-048 permanent FloatOp surface,
  ADR-049 canonical attention) are **already complete**. Stage 3's
  reframing is the closing semantic move, not a rewrite.
- New canonical ops added in future sprints get a conformance test
  free with the addition: the per-op kernel test in `hologram-ops`
  + the exec conformance cross-check together specify the contract.
- LLM inference performance is unchanged (no exec rewrite).
- The "rewrite exec to consume canonical directly" idea is
  explicitly **rejected** as architecturally wrong. Future
  contributors won't waste effort on it.

## Alternatives considered

- **Rewrite exec to call canonical kernels.** Rejected — measurable
  perf regression for no semantic gain. The canonical kernels are
  reference, not optimised; making them the runtime path makes
  inference slower.
- **Make canonical kernels SIMD-optimised in place.** Rejected —
  the canonical layer is the *semantic contract*. Optimisation
  belongs to the backend layer where target-specific tuning makes
  sense (NEON on ARM, AVX on x86, Metal compute shaders on Apple
  Silicon, etc.).
- **Have two implementations side by side with no contract.**
  That's the status quo before this ADR, and the migration's
  whole point was to fix it. Conformance tests are the contract.
- **Per-op feature flag for canonical vs legacy dispatch.**
  Rejected — adds combinatorial test surface, makes performance
  characteristics depend on flag state, defeats the "exec is the
  fast path" simplification.
