# CLAUDE.md — Hologram

## Project Overview

V2 rewrite of `../hologram-backup`. O(1) compute acceleration via precomputed LUT tables and KV-lookups, built on `uor-foundation` (v3.5.0).

## Build Commands

```bash
just ci        # fmt check + clippy + test (full CI)
just test      # cargo test --workspace
just bench     # criterion benchmarks
just fmt       # cargo fmt --all
just clippy    # cargo clippy --workspace -- -D warnings
just wasm      # build hologram-core for wasm32-unknown-unknown
```

## Architecture

- **hologram-core**: LUT tables, ElementWiseView, ring algebra, encoding. Zero deps except uor-foundation. `no_std`.
- **hologram-graph**: Expression graph, subgraphs, fusion, parallel scheduling.
- **hologram-archive**: `.holo` archive format, rkyv zero-copy, mmap, execution entrypoints.
- **hologram-exec**: KV-lookup executor, buffer arena, parallel level execution.
- **hologram-bench**: Criterion benchmarks.

Root crate (`src/lib.rs`) re-exports all subcrate APIs.

## Conventions

- Max 3 function arguments; use builder pattern for more.
- Prefer `macro_rules!` for repeated trait implementations.
- Functions <= 15 lines.
- Subdirectory organization within crates (no loose files beyond `lib.rs`).
- Only rkyv for serialization. All persistent types derive rkyv traits.
- SIMD feature-gated behind `simd`. Rayon behind `parallel`.
- No TODOs, stubs, or `unimplemented!()`.
- No backwards compatibility — single format version.

## Sprint Tracking

Active sprint: `specs/SPRINT.md`
Archived sprints: `specs/sprints/`
Plans: `specs/plans/`

<!-- HOLOARCH:MANAGED:BEGIN -->
## Relationship to hologram-architecture

This project is part of the Hologram ecosystem. Architecture decisions,
ADRs, and planning artifacts are maintained in `hologram-architecture`.

Before implementing significant functionality:

1. Read `specs/docs/architecture.md` and `specs/docs/upstream-architecture.md`.
2. Check `specs/docs/development.md` for the local development workflow.
3. Pull updated architecture docs with:
```bash
holoarch pull
```

## Important Commands

```bash
holoarch check       # validate repository conformance
holoarch pull        # pull latest docs + refresh managed sections
holoarch doc <name>  # generate a new doc template in specs/docs/
```

_This section is managed by `holoarch pull`. Repo: hologram_
<!-- HOLOARCH:MANAGED:END -->
