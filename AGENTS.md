# AGENTS.md — Hologram Greenfield

## Hard Rules

- All tests pass (`just ci`). Zero clippy warnings (`-D warnings`).
- 100% test coverage: unit + doc tests on all public items.
- Zero-copy hot paths. No heap allocation in lookup functions.
- No TODOs, no stubs, no `unimplemented!()`.
- Functions <= 15 lines. Max 3 arguments per function — use builder-pattern structs for more.
- Traits for shared behavior; builder pattern for complex construction.
- Prefer macros (`macro_rules!`) for repeated trait implementations and boilerplate patterns.
- `hologram-core` zero external deps except `uor-foundation` (no_std, traits-only).
- Every public operation has a Criterion benchmark.
- SIMD behind `#[cfg(target_arch)]`, feature-gated (`simd`).
- Rayon for parallel subgraph execution, feature-gated (`parallel`).
- Only rkyv for serialization; all persistent types derive rkyv traits.
- Subdirectory organization — no loose files in `src/` beyond `lib.rs`.
- All crates compile for `wasm32-unknown-unknown` with feature gates.
- Data structures fit L1/L2 cache constraints.
- No backwards compatibility formats — single current format only.
- Root crate (`src/lib.rs`) re-exports all public API from subcrates.
- When adding or changing public API, update the corresponding `site/pages/crates/*.mdx` doc page.

## Workspace Structure

```
hologram/
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
    hologram-core/       # LUT tables, views, ring, encoding (no_std)
    hologram-graph/      # Graph, subgraphs, fusion, scheduling
    hologram-archive/    # .holo format, rkyv, mmap, entrypoints
    hologram-exec/       # KV executor, buffer, parallel levels
    hologram-bench/      # Criterion benchmarks
  examples/
    calculator.rs    # Scientific calculator example
  site/
    pages/           # Nextra docs pages (MDX)
    next.config.mjs  # Site configuration
  src/lib.rs         # Root: re-exports all subcrate APIs
```

## Agent Roles

### Implementer
- Write code that passes `just ci` before committing.
- Follow all hard rules above.
- **After every task**: update `specs/SPRINT.md` AND `specs/plans/001-greenfield-refactor.md`.
- If public API changed: update the relevant `site/pages/crates/*.mdx` page.
- Mark tasks `- [x]` in BOTH documents only after `just ci` passes.
- Add implementation notes to `001-greenfield-refactor.md` sprint section when sprint completes.

### Reviewer
- Verify hard rules compliance.
- Check for zero-copy violations in hot paths.
- Ensure macro usage for repeated patterns.
- Validate function argument counts (<= 3).

### Architect
- Maintain `specs/plans/` with implementation plans.
- Ensure dependency graph integrity (hologram-core depends only on uor-foundation).
- Review fusion and scheduling designs for correctness.

## Sprint Workflow

- Active sprint tracked in `specs/SPRINT.md` with checkboxes (`- [ ]` / `- [x]`).
- Task lifecycle: Backlog → In Progress → `just ci` → `/commit` → Completed.
- **MANDATORY**: After every task, update BOTH tracking documents before committing:
  - `specs/SPRINT.md` — mark deliverables `[x]`, add Sprint 9+ section when sprint completes
  - `specs/plans/001-greenfield-refactor.md` — mark sprint tasks `[x]`, add implementation notes
- Archive completed sprints to `specs/sprints/<number>-<title>.md`.
- "Completed (Running Log)" section in SPRINT.md is append-only, permanent.

## Dependency Graph

```
uor-foundation (git v3.5.0, traits only, no_std)
       │
   hologram-core (LUT, views, ring, encoding — no_std + alloc)
       │
   hologram-graph (graph, subgraphs, fusion, scheduling)
       │
   hologram-archive (.holo format, rkyv, mmap, entrypoints, weights)
       │
   hologram-exec (KV executor, buffer, parallel levels)
       │
   hologram-bench (criterion benchmarks)
```


## Architecture Reference

Architecture decisions for this project are synced from `hologram-architecture`.
Before implementing significant functionality, read:

```
specs/docs/
```

To pull the latest:

```bash
holoarch pull
```
