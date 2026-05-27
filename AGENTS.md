# AGENTS.md

This document provides guidance for automated agents operating in **`hologram`**.

---

## Repository Purpose

`hologram` is a **library** repository in the ecosystem.

Standards version: `2026.03`

---

## Repository Structure

```
crates/         — the 11 workspace crates (hologram-host, -types, -ops,
                  -graph, -compiler, -exec, -backend, -archive, -cli,
                  -ffi, -bench)
specs/
  docs/         — project documentation (see architecture.md)
  adrs/         — architecture decision records
site/           — Astro documentation website + wasm demo
```

---

## Development

- Run tests: `cargo test --workspace`
- Check lints: `cargo clippy --workspace -- -D warnings`
- Format code: `cargo fmt --all`
- Common workflows live in the `Justfile` (`just ci`, `just test`, `just bench`).
- Project-specific architecture: `specs/docs/architecture.md`.

---

## Rules for Agents

1. Follow the architecture standards defined in the architecture repo
2. Do not modify files outside this repository unless explicitly instructed
3. Run `cargo clippy -- -D warnings` before committing Rust changes
4. Use a consistent naming prefix for all crate names

### Runtime Performance (hologram-exec / hologram-backend)
- **Zero allocation in hot paths**: `InferenceSession::execute` (`hologram-exec/src/session.rs`) dispatches one `KernelCall` per scheduled node, potentially thousands of times per inference. A κ-label miss must write into a recycled transient buffer reclaimed from the `BufferArena` pool — no `Vec::new`/`Box::new`/`to_vec`/`String::new` inside the dispatch loop.
- **Result caching is the κ-label memo, not a side store**: identical computation is addressed once by κ-label; before computing, the pool checks whether the output label is resident (pinned or transient) and rebinds the slot if so (an O(1) elision). Do not add a separate result cache, KV store, or HashMap on the hot path — the `BufferArena` pool *is* the cache.
- **Prefer compile-time solutions over runtime inference**: shapes are resolved at compile time and carried inside the `KernelCall`/`.holo` archive. There is no runtime shape-inference module; do not add per-op shape heuristics at execution time. If a new op's output shape cannot be expressed by the compiler's shape machinery, extend that, not the executor.
- **Fast-path first**: the common case is that compiled shapes and κ-labels are correct. Any recovery/validation logic must be guarded by a cheap check that short-circuits to a no-op.
- **All ops must dispatch through `KernelCall`**: every operation — float, quantized, fused, or custom — must have a corresponding `KernelCall` variant (`hologram-backend/src/kernel_call.rs`) and be handled by the backend's exhaustive `match` (`CpuBackend::dispatch`, `hologram-backend/src/cpu.rs`). Do not introduce execution paths that bypass the enum (no ad-hoc closures, `Box<dyn Fn>`, function-pointer tables, or boxed/dyn kernels). New ops require a new `KernelCall` variant and a `dispatch` arm; there is no tape and no virtual dispatch.

### Canonical ops (hologram-ops) and the `KernelCall` dispatch layer

These rules apply to the canonical-op stack (ADR-044, ADR-045) and the
`hologram-exec` / `hologram-backend` execution layer that runs on top of
it.

- **`hologram-ops` is the single source of truth for ops.** Every
  canonical op's full definition — marker struct, `Op` trait impl,
  `Call` struct, kernel function(s), and (when Plan 074 lands) LUT
  generator — lives in one file under `hologram-ops/src/kernels/<op>.rs`.
  Do not split an op's definition across crates. The op taxonomy is
  `OpKind` plus the per-op marker types in `hologram-ops`; the
  `KernelCall` enum (`hologram-backend/src/kernel_call.rs`) and the
  backend `dispatch()` (`CpuBackend::dispatch`, `hologram-backend/src/cpu.rs`)
  are the small touch points that route to the per-op file; everything
  else is local to that file.
- **Adding a new op is a checklist, not a search.** New op = new
  `kernels/<op>.rs` file with the full definition + `OpKind`/marker entry
  in `hologram-ops` + variant in `KernelCall` (with `dispatch` arm) +
  compiler lowering + integration test. No incidental edits in
  `hologram-graph` or `hologram-exec` unless that op needs a non-canonical
  lowering.
- **`OpKind` and `KernelCall` are closed surfaces.** `KernelCall` variants
  are on the `.holo` wire format; `OpKind` is the graph IR vocabulary.
  Adding/removing/reordering variants is an archive-format change.
- **No virtual dispatch in the backend.** `KernelCall` is an enum matched
  exhaustively by `CpuBackend::dispatch`. Do not add `Box<dyn Kernel>`,
  function-pointer tables, or trait-object kernel registries on the hot
  path. There is no tape and no boxed/dyn kernel layer.
- **κ-labels (LUT/AddressRef) are identity, never execution.** Address
  layers describe *which* value — never *how* to compute it. Do not embed
  kernel pointers, function pointers, or workspace/buffer handles in
  address or label types.
- **Backward computation is planned, not traversed.** Reverse-mode
  autodiff is forward-op composition: backward passes are emitted as
  ordinary `KernelCall`s ahead of time. The session must never traverse a
  graph to compute gradients at runtime.
- **No heap allocations in executor hot paths.** The `InferenceSession::execute`
  dispatch loop must not call `Vec::new`, `Box::new`, `to_vec`,
  `String::new`, or any allocator-touching API per `KernelCall`.
  Allocations belong at load/compile time.
- **No runtime algorithm selection inside kernels.** Choosing between
  variants (e.g., padded vs unpadded MatMul) is a compile/load-time
  decision that produces a different `KernelCall` variant. Kernels must
  not branch on shape or dtype to pick an algorithm.
- **No TODOs or `unimplemented!()` stubs in `hologram-exec` / `hologram-backend`.**
  Every public item must be fully implemented or removed.
- **Functions ≤ 15 lines and ≤ 3 args.** When a function naturally needs
  more arguments, introduce a builder struct or a parameter struct.
- **Every public item must have tests.** New `KernelCall` variants must be
  exercised end-to-end (graph → compile → execute) in the test suite.
- **Docs and specs must be updated with behaviour changes.** Any new op,
  backward rule, or `KernelCall` variant requires an update to the
  relevant ADR (or a new ADR) and `specs/docs/architecture.md`.

## Problem-Solving Philosophy

**Think like a principal systems architect, not a patch author.** When encountering bugs, build failures, or design issues, do not apply narrow band-aid fixes that address only the immediate symptom. Instead:

1. **Diagnose the root cause.** Before writing any fix, understand *why* the problem exists. Trace it back to the underlying design decision, missing abstraction, or architectural gap that allowed it to surface.

2. **Assess the blast radius.** Ask: is this a one-off mistake, or a symptom of a systemic pattern? If the same class of bug could occur elsewhere, the fix must address the class, not just the instance.

3. **Propose a production-ready solution.** Design the fix as if this code will run in production under load, across tenants, for years. Consider concurrency, error propagation, backward compatibility, and operational debuggability. A correct fix that is fragile or hard to reason about is not production-ready.

4. **Refactor when the problem is structural.** If the root cause is that a function does too much, a type doesn't enforce its invariants, or responsibilities are in the wrong module — fix the structure. Moving code, splitting types, introducing a new abstraction, or changing an API boundary are all valid (and often necessary) responses to a bug.

5. **Never play whack-a-mole.** If you find yourself fixing the same kind of issue in multiple places with small, repetitive patches, stop. That pattern means the underlying design needs to change. Propose the holistic fix, not N localized patches.

6. **Validate the fix is complete.** After implementing a solution, check whether the same class of issue exists anywhere else in the codebase. A fix that leaves known instances of the same bug untouched is incomplete.

---

<!-- ARCHON:MANAGED:BEGIN -->
## Ecosystem Rules

These rules apply to all repositories in the Hologram ecosystem.

### Naming
- Use the `hologram-` prefix for all crate names (never `holo-`)
- Follow kebab-case for crate and repo names

### Code Quality
- Run `cargo clippy -- -D warnings` before committing Rust changes
- Run `cargo fmt --check` before committing Rust changes
- All public APIs must have documentation comments
- No `unwrap()` in library code — use proper error handling
- Use traits at API boundaries; use macros to eliminate boilerplate
- Functions with >3 parameters must use the builder pattern
- Use `thiserror` for library errors; `anyhow` only in binaries
- See ADR-0007 for the full set of Rust development standards

### Architecture
- Follow ADR decisions from `hologram-architecture`
- Declare contracts in `hologram.repo.yaml`
- Do not introduce cross-repo dependencies without an ADR

### Documentation
- Keep `specs/docs/architecture.md` up to date with structural changes
- Update `AGENTS.md` when adding new conventions or rules
<!-- ARCHON:MANAGED:END -->

<!-- ARCHON:CONTEXT:BEGIN -->
## Ecosystem Context (auto-generated by archon)

See [`.archon/context.md`](.archon/context.md) for full dependency graph, public API surface, and contract details for this repo.
<!-- ARCHON:CONTEXT:END -->
