# ADR-046: Canonical → Legacy as a One-Way Adapter

## Status

Accepted (2026-04-27)

## Context

The migration to canonical ops (ADR-044, ADR-045) introduced two
helper methods on `GraphOp`:

- `legacy_float_op() -> Option<FloatOp>` — used to lower a
  `GraphOp::Compute(SemanticOp)` into the legacy `FloatOp` so it can
  flow through the existing tape, exec, and backend paths.
- `semantic_op() -> Option<SemanticOp>` — speculative reverse: lift a
  `GraphOp::Float(FloatOp)` into the canonical form.

In practice every production caller uses `legacy_float_op()` only.
`semantic_op()` had zero non-test callers across the workspace. Two
exec dispatch sites (`hologram-exec/src/kv/store.rs` and
`tape_builder.rs`) also kept *separate* match arms for `Compute` and
`Float` even though both arms ran the identical lowering — the one
difference being a no-op `legacy_float_op()` call on the `Compute`
side.

This ADR locks in the actual shape of the bridge so future work
doesn't drift back toward bidirectional plumbing.

## Decision

The canonical → legacy bridge is **one-way** and has a **single
collapsed dispatch site per consumer**:

1. **Remove `GraphOp::semantic_op()`** and its `semantic_from_float`
   helper. Future code that needs `FloatOp → SemanticOp` may add a
   focused helper at the call site, but the speculative public API is
   gone.
2. **Collapse `Compute` / `Float` dispatch arms** wherever they were
   duplicated. Both variants resolve through `op.legacy_float_op()`
   into a single shared lowering path. The two pre-collapse arms in
   `hologram-exec` (`kv/store.rs`'s `dispatch_with_shapes` and
   `tape_builder.rs`'s `resolve_kernel`) are now one arm each.
3. **`legacy_float_op()` keeps its name** even though the bridge is no
   longer bidirectional — the name accurately describes its job
   (resolve to legacy `FloatOp`) and is already pervasive in
   downstream crates.

### Why one-way is the right shape

- `FloatOp` is the *execution-side* encoding (per the user's stated
  philosophy: "FloatOp remains only as a legacy lowering type inside
  execution/backend code"). Code below the graph layer dispatches on
  `FloatOp`. A one-way `Compute → Float` adapter at the graph/exec
  boundary is therefore correct architecture.
- The reverse direction (`Float → Compute`) would be a *promotion* —
  taking an execution-encoded op back to a canonical semantic one.
  No consumer ever wanted that. Lifting an op is the graph builder's
  job; it should construct `Compute` directly, not rewrite `Float`
  nodes after the fact.
- A bidirectional bridge implicitly says "either form is acceptable
  anywhere". That undermines the migration: the canonical layer is
  supposed to *be* the source of truth, not one of two
  interchangeable encodings.

### Why the collapsed dispatch arms

Before this ADR, `kv/store.rs` and `tape_builder.rs` each had
duplicated `Compute(_) => …` and `Float(_) => …` arms with literally
identical bodies (modulo the `legacy_float_op()` call). That coupling
was the kind of "you can drift between two encodings without anyone
noticing" risk the ADR-045 single-source-of-truth move was trying to
prevent.

Collapsing both to `Compute(_) | Float(_) => …` makes the canonical
path strictly subordinate to the legacy execution dispatch — exactly
the relationship the bridge represents.

## Consequences

- Public API surface shrinks: `GraphOp::semantic_op` is gone. Any
  external caller relying on it (none observed in-tree) needs a
  focused helper.
- The dead `semantic_from_float` helper (~95 lines) is deleted.
- Exec dispatch sites are simpler — one arm each instead of two
  duplicates.
- The naming `legacy_float_op` is intentionally preserved: it
  signals to readers that this is the *adapter into the legacy
  execution form*, which is also why we want to eventually replace
  the float-op-shaped exec path entirely (Sprint 37 Phase 3.3).

## Alternatives considered

- **Keep `semantic_op()` for symmetry.** Rejected — symmetry isn't a
  goal; correctness of the bridge's direction is. Speculative public
  API rots.
- **Rename `legacy_float_op` to something neutral like `float_form`.**
  Rejected for now — the "legacy" qualifier is informative; renaming
  it implies the FloatOp form is permanent, which it isn't (Sprint
  37 Phase 3.3 has its eventual deprecation as a goal).
- **Eagerly resolve `Compute` to `Float` at graph-construction time.**
  Rejected — that would lose the canonical `SemanticOp` payload, and
  makes graph-level rewrites that want canonical form (e.g. the LUT
  generator integration in Plan 074) harder. The bridge stays at the
  exec boundary, not at construction.
