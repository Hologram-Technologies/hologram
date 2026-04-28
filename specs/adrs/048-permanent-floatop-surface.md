# ADR-048: Permanent FloatOp Surface

## Status

Accepted (2026-04-27)

## Context

ADR-047 staged the `FloatOp` deprecation in 4 phases. While migrating
canonical ops in Sprint 37 Phase 3.3 Stage 2 it became clear that not
every `FloatOp` variant *should* migrate to `SemanticOp`. The
canonical layer has a deliberate shape — pure-f32 compute, dtype-uniform,
single-output — and several `FloatOp` variants violate that contract
in ways that aren't accidental.

This ADR pins the boundary so future "migrate everything" sprints
don't waste effort trying to fit ops that don't belong.

## Decision

The canonical `SemanticOp` surface comprises **pure-f32 single-output
compute ops**. The following `FloatOp` variants stay in `FloatOp`
permanently as the execution-side encoding, *not* as a Stage-2-pending
migration:

### Multi-output ops

- `TopK { axis, largest }` — returns `(values, indices)`. Canonical
  ops are single-output by construction (the planner emits one
  `KernelCall` per node, with one output `SlotSpan`). Multi-output
  semantics need a separate canonical layer or graph rewrite that
  splits into two single-output ops.
- `NonZero` — returns variable-length `i64` indices. Canonical ops
  have static output shapes derivable at planning time; NonZero's
  output length is data-dependent.

### Integer-index ops

- `Gather { dim, dtype }` — index input is `i64`.
- `GatherND` — multi-dim `i64` indices.
- `ScatterND` — `i64` indices + scatter pattern.
- `Embed { dim, quant }` — token id input is `u32`/`i64`.
- `ArgMax { axis, keepdims }` — output is `i64` indices.

These need an integer-tensor type in the canonical layer to be
represented honestly. An eventual "canonical IntOp" or a
multi-dtype `SemanticOp` is the right home — *not* `FloatOp`'s f32
variant carrying integer payloads. Until that ADR lands, they stay
on `FloatOp`.

### Boolean / mask-driven ops

- `Compress { axis }` — boolean condition mask input.

Same reasoning as integer-index ops: needs a typed bool tensor in
the canonical layer.

### Sequence-length-driven ops

- `ReverseSequence { batch_axis, time_axis }` — second input is a
  per-batch `i64` sequence-lengths tensor. Same integer-tensor
  problem as the index-driven ops above; promotion deferred until
  the canonical layer can carry integer tensors honestly.

### Dtype-conversion ops

- `Cast { from, to }` — explicit dtype conversion. The canonical
  layer assumes uniform f32 storage; "casting" within an all-f32
  layer is a no-op or a silent precision change. Cast is genuinely
  an *execution-layer* concern (where dtypes are tracked alongside
  spans), not a semantic compute op.

### Metadata-output ops

- `Shape { dtype, start, end }` — output is `i64` shape values.
  Belongs to a graph-tooling / metadata layer, not the canonical
  compute layer.
- `Range` — output values are *generated* (not computed from input
  tensor values); inputs are scalars. Different semantic shape than
  a compute op; could fit in a "tensor builders" layer if one is
  added.

### Quantisation adapters

- `Dequantize` — quant→f32 conversion. Pure execution adapter; the
  semantic op above it (`MatMulLut*`, `Conv2dLut4`) is what the
  canonical layer should eventually express, not the dequant step.

### KV-cache state ops

- `KvWrite { … }` and `KvRead { … }` — these mutate / read external
  cache state. Canonical compute ops are pure (no side effects on
  cache state). KV cache is an execution-side concern that the
  planner orchestrates around, not a semantic op.

### Specialised fused projections

- `NormProjectionGemv { … }`
- `AddNormProjectionGemv { … }`
- `SwiGluProjectionGemv { k, n }`

These are runtime-fused composite ops produced by the legacy fusion
pass. Per ADR-044, fused variants stay on `GraphOp` (or here, on
`FloatOp` as the legacy execution encoding). They're planner
products, not canonical sources.

### Execution-shape tags

- `FloatOpShape::*` (`UnaryElementwise`, `BinaryElementwise`, etc.)
  — these aren't ops at all; they're the dispatch-shape enum used
  inside `hologram-exec/float_dispatch/`. Already renamed from
  `OpCategory` in Sprint 37 Phase 1.5.4 to make the role explicit.

## Variants that DO migrate (stage 2 remaining)

Pure-f32 single-output compute that's just unmigrated work:

- `Attention { … }` — multi-head attention. ADR-worthy because the
  canonical surface design has to decide on multi-head structure,
  optional masking, and whether to integrate `RotaryEmbedding` or
  not.
- `Resize` (cubic mode — nearest + linear landed in Sprint 37
  Phase 3.3 Stage 2 Round 5).

## Consequences

- Stage 2's "remaining work" list shrinks: the permanent-FloatOp set
  is taken out of the migration target. The canonical surface end
  state has roughly **75 ops**, not 96.
- ADR-047 Stage 4 ("remove `FloatOp` from public API") becomes a
  more nuanced move: `FloatOp` doesn't disappear — it becomes the
  execution-side encoding for the variants listed above, properly
  scoped (likely renamed to `ExecOp` or `LegacyOp` and marked
  `pub(crate)` inside an internal exec module).
- Eventual canonical `IntOp` or multi-dtype `SemanticOp` is the
  long-term home for the integer-index / boolean-mask variants. A
  separate ADR will design that layer.
- The Stage 2 backlog is now a finite, auditable list — every entry
  can be implemented as a self-contained kernel addition.

## Alternatives considered

- **Force every `FloatOp` variant into `SemanticOp` via type erasure
  / `bytes` payload.** Rejected — defeats the type-safety
  contract that makes the canonical layer a useful source of truth.
- **Add a parallel `IntOp` enum now.** Rejected as premature; the
  in-tree consumers don't need it yet, and the canonical-surface
  expansion shows what the integer-tensor surface actually needs to
  cover.
- **Merge `MatMul` into `Gemm` when canonical Gemm lands.** Open
  question; will be decided in the Gemm-specific ADR (Stage 2).
- **Keep `FloatOp` as the eternal source of truth.** Rejected —
  ADR-045 made `hologram-ops` the single source of truth; this ADR
  only carves out which variants belong in *which layer* of that
  source.
