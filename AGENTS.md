# AGENTS.md — Hologram Greenfield

## Hard Rules

- All tests pass (`just ci`). Zero clippy warnings (`-D warnings`).
- 100% test coverage: unit + doc tests on all public items.
- Zero-copy hot paths. No heap allocation in lookup functions.
- No TODOs, no stubs, no `unimplemented!()`.
- Functions <= 15 lines. Max 3 arguments per function — use builder-pattern structs for more.
- Traits for shared behavior; builder pattern for complex construction.
- Prefer macros (`macro_rules!`) for repeated trait implementations and boilerplate patterns.
- `holo-core` zero external deps except `uor-foundation` (no_std, traits-only).
- Every public operation has a Criterion benchmark.
- SIMD behind `#[cfg(target_arch)]`, feature-gated (`simd`).
- Rayon for parallel subgraph execution, feature-gated (`parallel`).
- Only rkyv for serialization; all persistent types derive rkyv traits.
- Subdirectory organization — no loose files in `src/` beyond `lib.rs`.
- All crates compile for `wasm32-unknown-unknown` with feature gates.
- Data structures fit L1/L2 cache constraints.
- No backwards compatibility formats — single current format only.
- Root crate (`src/lib.rs`) re-exports all public API from subcrates.

## Workspace Structure

```
hologram-greenfield/
  Cargo.toml         # Workspace root + root crate
  AGENTS.md          # This file
  CLAUDE.md          # Project context for AI agents
  Justfile           # Build/test/bench commands
  .githooks/         # Git hooks
  specs/
    project.md       # Project specification
    SPRINT.md        # Active sprint tracking
    sprints/         # Archived sprints
    plans/           # Implementation plans
  crates/
    holo-core/       # LUT tables, views, ring, encoding (no_std)
    holo-graph/      # Graph, subgraphs, fusion, scheduling
    holo-archive/    # .holo format, rkyv, mmap, entrypoints
    holo-exec/       # KV executor, buffer, parallel levels
    holo-bench/      # Criterion benchmarks
  examples/
    calculator.rs    # Scientific calculator example
  src/lib.rs         # Root: re-exports all subcrate APIs
```

## Agent Roles

### Implementer
- Write code that passes `just ci` before committing.
- Follow all hard rules above.
- Update `specs/SPRINT.md` immediately on task state change.
- Mark SPRINT tasks as `- [x]` only after `just ci` passes.

### Reviewer
- Verify hard rules compliance.
- Check for zero-copy violations in hot paths.
- Ensure macro usage for repeated patterns.
- Validate function argument counts (<= 3).

### Architect
- Maintain `specs/plans/` with implementation plans.
- Ensure dependency graph integrity (holo-core depends only on uor-foundation).
- Review fusion and scheduling designs for correctness.

## Sprint Workflow

- Active sprint tracked in `specs/SPRINT.md` with checkboxes (`- [ ]` / `- [x]`).
- Task lifecycle: Backlog → In Progress → `just ci` → `/commit` → Completed.
- Update `specs/SPRINT.md` immediately on task state change.
- Archive completed sprints to `specs/sprints/<number>-<title>.md`.
- "Completed (Running Log)" section in SPRINT.md is append-only, permanent.

## Dependency Graph

```
uor-foundation (git v3.5.0, traits only, no_std)
       │
   holo-core (LUT, views, ring, encoding — no_std + alloc)
       │
   holo-graph (graph, subgraphs, fusion, scheduling)
       │
   holo-archive (.holo format, rkyv, mmap, entrypoints, weights)
       │
   holo-exec (KV executor, buffer, parallel levels)
       │
   holo-bench (criterion benchmarks)
```
