# CLAUDE.md — Hologram Greenfield

## Project Overview

V2 rewrite of `../hologram-backup`. O(1) compute acceleration via precomputed LUT tables and KV-lookups, built on `uor-foundation` (v3.5.0).

## Build Commands

```bash
just ci        # fmt check + clippy + test (full CI)
just test      # cargo test --workspace
just bench     # criterion benchmarks
just fmt       # cargo fmt --all
just clippy    # cargo clippy --workspace -- -D warnings
just wasm      # build holo-core for wasm32-unknown-unknown
```

## Architecture

- **holo-core**: LUT tables, ElementWiseView, ring algebra, encoding. Zero deps except uor-foundation. `no_std`.
- **holo-graph**: Expression graph, subgraphs, fusion, parallel scheduling.
- **holo-archive**: `.holo` archive format, rkyv zero-copy, mmap, execution entrypoints.
- **holo-exec**: KV-lookup executor, buffer arena, parallel level execution.
- **holo-bench**: Criterion benchmarks.

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
