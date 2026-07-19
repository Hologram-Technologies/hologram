# hologram-compiler

> Compiles a hologram Graph into Prism pipelines (a per-node CompileUnit) and emits a `.holo` archive.

The compiler runs a per-node CompileUnit pipeline: look up the op marker for `node.op_kind`, resolve concrete shape/dtype/host-bounds generics, emit a Term tree into a `TermArena`, build and validate a `CompileUnit`, run tower-completeness, cache by `ContentFingerprint<32>`, lower to a backend `KernelCall`, and emit `(kernel_call, certificate, fingerprint)` into the archive.

## What it provides

- `Compiler` — drives the compile pipeline over a `hologram_graph::Graph`.
- `BackendKind`, `CompilationOutput`, `CompilationStats` — the backend selector and compile results.
- `compile` — compile a pre-built graph.
- `compile_from_source` — parse UOR source into a Graph, then compile.
- `compile_from_source_language` — parse a selected source language (`source::SourceLanguage`) into a Graph, then compile.
- `compile_with_backward` — desugar composites, append a backward subgraph, expose gradients as outputs, and compile the augmented graph (spec V.4 / ADR-043); returns the gradient `NodeId`s alongside the archive.
- `CertificateCache`, `CachedCertificate` — certificate caching keyed by fingerprint.
- `CompileError` — the compile error surface.

## Features

- `std` (default) — enables `tracing` diagnostics and the std error surface; propagates `std` to prism, foundation, types, compute, and archive.
- `frontend-rust` — Rust source frontend (`proc-macro2`, `syn`; implies `std`).
- `frontend-python` — Python source frontend (implies `std`).
- `frontend-typescript` — TypeScript source frontend (`swc_common`, `swc_ecma_ast`, `swc_ecma_parser`; implies `std`).

## Targets & build notes

`no_std` + `alloc` when `std` is disabled (matching prism / uor-addr), so the core compile pipeline runs on wasm and embedded targets. The `std` feature only adds `tracing` diagnostics. Frontend features require `std`.

Part of the [hologram](../../README.md) workspace.
