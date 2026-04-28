# ADR-047: Staged FloatOp Deprecation

## Status

Accepted (2026-04-27) — implementation in progress (Sprint 37 Phase 3.3+)

## Context

After ADR-044/045/046, the canonical-op layer (`hologram-ops`) is the
single source of truth for ops it covers, and the
`Compute(SemanticOp) → Float(FloatOp)` bridge is one-way. The end
state stated in earlier design discussions is:

> "FloatOp disappears entirely. The backend consumes a lower-level
> executable form derived from canonical ops, not FloatOp."

But `FloatOp` has **96 variants**; `SemanticOp` has **36**. The
canonical layer doesn't yet cover roughly 60 ops the production
runtime depends on (comparisons, reductions, pooling, attention,
type ops, embedding, etc.). A one-shot rewrite to consume
`SemanticOp` directly inside `hologram-exec/src/float_dispatch/`
would either drop those ops or duplicate dispatch logic across two
encodings — both regressive.

This ADR records the staged plan to get there safely.

## Decision

`FloatOp` is deprecated as a *top-level semantic source* (where new
graphs should never be constructed against it) but remains as the
*execution-side encoding* until two things are true:

1. Every production `FloatOp` variant has a `SemanticOp` counterpart.
2. `hologram-exec/src/float_dispatch/` is reorganised to dispatch on
   `SemanticOp` directly (kernels in `hologram-ops` are the source
   of truth for canonical ops; exec-side dispatch becomes a thin
   pattern match over `SemanticOp` variants).

Until both hold, the bridge in `legacy_float_op()` is the supported
architecture per ADR-046 — `Compute(SemanticOp)` enters, gets lowered
to `FloatOp`, and runs through the existing dispatch.

### Stage 1 — Smart construction (this ADR, landed)

- `GraphOp::from_float(FloatOp) -> GraphOp` smart constructor:
  returns `Compute(SemanticOp)` when canonical covers the variant,
  falls back to `Float(FloatOp)` otherwise.
- All in-tree lowering sites (compiler `term_lower`, future ONNX
  importers, hand-built test graphs that don't deliberately exercise
  the legacy path) migrate to call `GraphOp::from_float`.
- New code is *forbidden* from constructing `GraphOp::Float` directly
  except inside `hologram-exec`/fusion (where it's the execution
  encoding, not a semantic choice).

### Stage 2 — Canonical surface expansion (Sprint 37 Phase 3.4)

Add `SemanticOp` variants + `Op` trait impls + per-op kernel files
in `hologram-ops/src/kernels/` for the variants that block stage 3:

**Easy (can land incrementally):**
- Comparison: `Equal`, `Less`, `LessOrEqual`, `Greater`, `GreaterOrEqual`
- Bitwise / bool: `And`, `Or`, `Xor`, `Not`
- Binary math: `Pow`, `Mod`, `Min`, `Max`
- Reductions: `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`,
  `ReduceProd`
- Type / shape: `IsNaN`, `Cast`, `Clip`, `Shape`, `Range`,
  `CumSum`, `Where`

**Medium:**
- Pooling: `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`, `LRN`
- `Embed`, `Gather`, `GatherND`, `ScatterND`, `TopK`
- `Resize`, `PadOp`

**Heavy (ADR-worthy):**
- `Attention` (and the rotary embedding it bundles)
- `ConvTranspose`
- `Gemm` (matmul + bias + alpha/beta — wider than `MatMul`)
- `Dequantize`

Each addition:
- New `SemanticOp` variant + `Attrs` struct
- New marker struct + `Op` impl in `hologram-ops/src/kernels/<op>.rs`
- New `Call` struct + reference kernel
- New `KernelCall` variant + `dispatch` arm
- New planner arm in `hologram-transform`
- Update `from_float` and `legacy_float_op` mappings

Adding a single op is bounded; the count is the work.

### Stage 3 — Reorganise exec around SemanticOp

Once `SemanticOp` covers every production `FloatOp` variant:

- `hologram-exec/src/float_dispatch/` rewrites to dispatch on
  `SemanticOp` (or directly on `KernelCall` for fully-canonical
  paths).
- `legacy_float_op()` shrinks to a thin compat shim used only by
  `GraphOp::Float(_)` legacy construction sites that haven't been
  removed.
- `hologram-backend` similarly rewrites.

### Stage 4 — Remove FloatOp from public API

- `hologram-core::op::FloatOp` becomes `pub(crate)` inside whatever
  exec/backend module needs it as the on-the-wire dispatch
  encoding, or is renamed to something internal-flavoured (e.g.
  `ExecOp`, `DispatchOp`).
- Public re-exports of `FloatOp` from `hologram::*` are removed.
- `GraphOp::Float` variant: either removed (rkyv-archive
  format-breaking change, requires migration tooling) or retained as
  a frozen "legacy archive readback" variant.

This stage is intentionally last — it's the most disruptive and only
worth taking after the canonical surface is genuinely complete.

### Why not all-at-once

- **Archive compatibility.** `GraphOp::Float` is on the wire format.
  Removing it without canonical coverage would break archive readers
  for any model that uses ops not yet in `SemanticOp`.
- **Surface area.** ~60 unmigrated ops × {variant + struct + Op impl
  + Call struct + kernel + planner arm + tests} is 5–10 sprints of
  focused work. A flag day move would either ship 60 stub kernels
  (regressive) or block until everything's done (ships nothing).
- **Risk isolation.** Each canonical-op addition is independently
  tested and merged; a bug in the new `Equal` kernel doesn't risk
  the `Conv2d` path.

## Consequences

- Stage 1 is the contract for all *new* code: `GraphOp::from_float`
  is the smart constructor; direct `GraphOp::Float` construction is
  reserved for fusion products + exec-internal code.
- Production `FloatOp` callers continue to work unchanged through
  the bridge.
- Each stage 2 op-addition follows the established ADR-045 recipe —
  the marginal cost has been engineered down by the prior ADRs.
- Stage 3 is a real rewrite (~1k–2k lines in `float_dispatch/`) but
  it's mechanical once stage 2 is done.
- Stage 4 needs a separate "archive format migration" ADR if
  `GraphOp::Float` is to be removed.

## Alternatives considered

- **Rip out `FloatOp` now, ship stubs for the gap.** Rejected —
  silently regresses production ops.
- **Keep `FloatOp` forever as a parallel encoding.** Rejected — the
  whole point of the canonical layer is to be *the* source of truth.
  Two encodings drift; ADR-045 was about preventing exactly that.
- **Auto-derive `SemanticOp` from `FloatOp` via macro.** Rejected —
  the canonical surface is a *deliberate semantic contract*, not a
  mechanical projection. Ops like `Gemm` (which subsumes `MatMul`
  with bias + alpha + beta) need explicit canonical design, not
  auto-mirroring.
