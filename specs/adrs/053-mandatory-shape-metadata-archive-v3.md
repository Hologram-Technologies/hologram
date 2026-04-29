# ADR-053: Mandatory Shape Metadata in Archive Format v3

**Status:** Proposed
**Date:** 2026-04-28
**Deciders:** Ari (project lead)
**Related:** ADR-001 (BLAKE3 archive checksums, format v2), Sprint 33 Phase 4.3 (shape_resolve.rs deletion)

## Context

`hologram-exec` currently maintains a 355-line shape-resolution
fallback module
([`crates/hologram-exec/src/shape_resolve.rs`](../../crates/hologram-exec/src/shape_resolve.rs))
referenced from 22 dispatch sites in `kernel_dispatch.rs`. Every site
has the shape:

```rust
let actual = shape_last_dim(idx).unwrap_or_else(|| {
    shape_resolve::resolve_last_dim(
        compiled_size,                                  // baked at compile time
        input_metas.first().and_then(|m| m.as_ref()),   // optional runtime meta
        inputs.first().map(|b| b.len()).unwrap_or(0),   // byte-length heuristic
    )
});
```

The fast path (`shape_last_dim(idx)`) reads the runtime
`ShapeRegistry` populated by the executor before dispatch. The
fallback chain exists because **the archive format does not require
shape metadata to be present** — a v2 archive can omit
`SectionedGraph::node_shapes` and `constant_shapes`, in which case the
runtime falls through to byte-length/compiled-size heuristics. Three
of the four fallback strategies are guesses that work most of the time
but mask real bugs (e.g. a wrong-dtype meta accidentally aligning with
a buffer size that divides cleanly).

Sprint 33 Phase 4 introduced the fast path in commit `1d1e870` but
left the fallback. Phase 4.3 ("delete shape_resolve.rs") was marked
complete in error — the file still exists at 355 lines because no
contract guarantees archives carry shape metadata.

This ADR proposes the contract.

## Decision

### 1. Bump archive format version to **v3**

`FORMAT_VERSION = 3` ([`crates/hologram-archive/src/format/mod.rs`](../../crates/hologram-archive/src/format/mod.rs)).
The header parser already accepts both `2` and `3` for read; the
writer becomes the only producer of v3.

### 2. v3 mandates per-tensor shape coverage

Every persisted graph in a v3 archive **must** populate
`SectionedGraph::node_shapes` for every non-constant tensor produced
by a dispatch node, and `constant_shapes` for every weight tensor
referenced by the graph. Validators reject v3 archives missing
either coverage during load.

Specifically, in
[`crates/hologram-archive/src/format/graph.rs`](../../crates/hologram-archive/src/format/graph.rs):
- `node_shapes` — one entry per `NodeId` in execution order.
- `constant_shapes` — one entry per `ConstantId` referenced by the
  weight section.

Both fields are non-optional in v3. The validator surfaces missing
entries as `ArchiveError::MissingShape { node_id }` /
`MissingConstantShape { constant_id }`.

### 3. Strict load — no heuristic fallbacks

`hologram-exec`'s loader pre-seeds `BufferArena::shapes` from
`SectionedGraph::node_shapes` and `constant_shapes` at archive load
time. Once seeded, dispatch reads shapes directly from
`ShapeRegistry` — no heuristics, no `unwrap_or_else` fallbacks.

After this is wired:

- **Delete** `crates/hologram-exec/src/shape_resolve.rs` (355 lines).
- **Delete** `InputMetas` typedef and the `input_metas` parameter
  threading through `dispatch_kernel` (29 sites).
- **Replace** the 22 `shape_*(idx).unwrap_or_else(...)` patterns
  with direct `shape_*(idx)?` returning `DispatchError::MissingShape`
  on absence (which the loader has just guaranteed cannot happen for
  v3 archives).

### 4. v2 archives: read-only, deprecated

v2 archives continue to load (existing readers tolerate them) but
**only via a compatibility loader path** that pre-populates the
`ShapeRegistry` with whatever metadata is available, then **synthesises
shapes from the compiled-size and buffer-length pair** at load time
using the current `shape_resolve::resolve_*` logic — **moved out of
the dispatch hot path into the v2 compat shim**.

This means the heuristic logic survives in `hologram-archive`'s v2
compat shim, but `hologram-exec` itself no longer carries it.
Eventually (a separate ADR) v2 read support gets removed entirely.

### 5. Writer behaviour

`HoloWriter` always emits v3. The four `tape_builder.rs` test sites
currently constructing graphs with `node_shapes: Vec::new()` get
audited and either:
- populated with the actual shapes (preferred — tests are in tree, the
  shape data exists at construction time); or
- routed through a `for_tests_v2` constructor that emits a v2 archive
  for back-compat coverage.

## Consequences

### Positive

- **355 lines + 22 dispatch fallback paths deleted** from the
  hot path. Sprint 33 Phase 4.3 actually finishes.
- Bug masking is gone: a missing shape becomes an explicit error at
  archive load, not a heuristic guess at dispatch.
- One canonical source of truth for shapes — the `ShapeRegistry`,
  populated from the archive once.
- Dispatch becomes pure `ShapeRegistry` lookup; no more `Option`
  threading through the dispatch parameter list.

### Negative

- v2 archives produced before this ADR no longer dispatch via the
  same path as v3 — they take the compat shim. Acceptable: shim is
  smaller than the current hot-path fallback chain, and v2 is
  pre-1.0 anyway.
- Writers that previously skipped `node_shapes` (the four test sites
  in `tape_builder.rs`) need to be updated. Mechanical.
- The v3 format is not byte-compatible with v2 — `is_supported_version`
  changes from `version == 2 || version == 3` to a true
  branch-on-version with separate code paths.

### Alternatives considered

1. **Keep heuristic fallbacks; just trust them.** Rejected — the
   fallbacks are 355 lines, masking real bugs (the dtype/byte-length
   conflation in `resolve_last_dim` is a known foot-gun), and they
   add `Option` threading through 29 callsites that should be
   straight reads.
2. **Make `node_shapes` optional in v3 but log a warning.** Rejected
   — pushes the same heuristic logic into the runtime, just with
   noisier observability. No incentive to populate shapes correctly.
3. **Compute shapes at load time from the graph's IR rather than
   persisting them.** Rejected — would require rerunning shape
   inference at every load, which the planner already paid for at
   compile time. Persisting is cheap (a few KB per graph).

## Implementation plan

Tracked separately. Skeleton:

1. **v3 writer** — bump `FORMAT_VERSION = 3`; `HoloWriter` validates
   coverage at write time and errors out if the graph is missing
   shapes. (One PR, no consumer changes.)
2. **v2 compat shim** — move `shape_resolve::resolve_*` into a new
   `hologram-archive::compat::v2_shapes` module; the loader
   synthesises shapes for v2 archives there. (One PR.)
3. **Strict loader** — pre-populate `BufferArena::shapes` from
   archive metadata; add `MissingShape` errors. (One PR, gated on
   v3 writer being live.)
4. **Delete** `shape_resolve.rs` + `InputMetas` + 22 `unwrap_or_else`
   sites + 29 `input_metas` threadings. Sprint 33 Phase 4.3 closes.
5. **Test suite** — round-trip test asserting every node and constant
   in a representative graph has a shape in the v3 archive.

## References

- ADR-001 — BLAKE3 archive checksums (precedent for breaking-change
  format bumps).
- [`crates/hologram-exec/src/shape_resolve.rs`](../../crates/hologram-exec/src/shape_resolve.rs)
  — module to be deleted.
- [`crates/hologram-archive/src/format/graph.rs`](../../crates/hologram-archive/src/format/graph.rs)
  — `node_shapes`/`constant_shapes` fields.
- [`crates/hologram-exec/src/kernel_dispatch.rs`](../../crates/hologram-exec/src/kernel_dispatch.rs)
  — 22 fallback sites.
- Sprint 33 Phase 4 commit `1d1e870` — direct-read fast path.
- Sprint 33 status correction commit `e897895` — Phase 4.3 marked partial.
