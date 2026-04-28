# AGENTS.md

This document provides guidance for automated agents operating in **`hologram`**.

---

## Repository Purpose

`hologram` is a **library** repository in the ecosystem.

Standards version: `2026.03`

---

## Repository Structure

```
specs/
  docs/         — project documentation
  adrs/         — architecture decision records
```

---

## Rules for Agents

1. Follow the architecture standards defined in the architecture repo
2. Do not modify files outside this repository unless explicitly instructed
3. Run `cargo clippy -- -D warnings` before committing Rust changes
4. Use a consistent naming prefix for all crate names

### Runtime Performance (hologram-exec)
- **Zero allocation in hot paths**: shape resolution runs per-node per-level (thousands of times per inference). No `Vec` allocations inside shape-resolution functions except when constructing the output shape itself.
- **Prefer compile-time solutions over runtime inference**: stale-shape recovery in `ShapeContext` is a fallback; the correct long-term fix is ensuring the compiler emits accurate shapes via the ONNX Shape Oracle (see plan). Do not grow `shape_resolve.rs` with new per-op heuristics — add oracle coverage instead.
- **No speculative corrections**: `correct_stale_shape` scans at most `ndim` integers (≤8 for all current ops). It must not call external functions, allocate, or recurse.
- **Fast-path first**: the common case is that compiled shapes are correct. All correction logic must be guarded by a cheap identity check (`prod == actual_count`) that short-circuits to a no-op.
- **Avoid growing shape_resolve.rs**: new op support belongs in the compiler's shape oracle, not in runtime shape inference. If a new op's output shape cannot be expressed via `ShapeSpec`, add a `ShapeSpec` variant rather than a new `resolve_*` function.
- **All ops must dispatch through `TapeKernel`**: every operation — float, quantized, fused, or custom — must have a corresponding `TapeKernel` variant and go through `dispatch_kernel()`. Do not introduce op execution paths that bypass the tape (e.g., ad-hoc closures, `Box<dyn Fn>`, or direct kernel calls outside the enum match). New ops require a new `TapeKernel` variant in `tape.rs` and a mapping in `tape_builder.rs`.

### Canonical ops (hologram-ops) and Transformation Chains (hologram-transform)

These rules apply to the canonical-op stack (ADR-044, ADR-045) and the
transform / planner / executor layer on top of it (ADR-043).

- **`hologram-ops` is the single source of truth for ops.** Every
  canonical op's full definition — marker struct, `Op` trait impl,
  `Call` struct, kernel function(s), and (when Plan 074 lands) LUT
  generator — lives in one file under `hologram-ops/src/kernels/<op>.rs`.
  Do not split an op's definition across crates. The `SemanticOp`
  enum, dispatch macro, `KernelCall` enum, and `dispatch()` function
  are the small touch points that route to the per-op file; everything
  else is local to that file.
- **Adding a new op is a checklist, not a search.** New op = new
  `kernels/<op>.rs` file with the full definition + variant in
  `SemanticOp` (with macro arm) + variant in `KernelCall` (with
  `dispatch` arm) + planner arm + builder method + integration test.
  No edits in `hologram-graph`, `hologram-exec`, or `hologram-backend`
  unless that op needs a non-canonical lowering.
- **`SemanticOp` is the closed serialisation surface.** Variants are
  on the wire format. Adding/removing/reordering variants is a
  graph-archive-format change.
- **No virtual dispatch in `dispatch()`.** `KernelCall` is an enum
  matched exhaustively. Do not add `Box<dyn Kernel>`, function-pointer
  tables, or trait-object kernel registries on the hot path.

The rules below apply specifically to the chain → plan → execute
layer in `hologram-transform`:

- **LUT is identity, never execution.** Address layers (`AddressRef`,
  `TensorId`, `RegionId`, `LayoutId`) describe *which* object — never *how*
  to compute it. Do not embed kernel pointers, function pointers, or
  workspace handles in address types.
- **Transform descriptors are semantic only.** `TransformNode` and
  `TransformChain` must not allocate a workspace, hold a buffer, or perform
  any computation. They are pure descriptors safe to clone and rewrite.
- **Backward computation is planned, not traversed.** Backward passes must
  be emitted as ordinary `KernelCall`s by the planner. The executor must
  never traverse a graph to compute gradients at runtime.
- **No heap allocations in executor hot paths.** `Executor::run_forward`
  and `Executor::run_backward` must not call `Vec::new`, `Box::new`,
  `to_vec`, `String::new`, or any allocator-touching API inside the kernel
  loop. Allocations belong in the planner.
- **No runtime algorithm selection inside kernels.** Choosing between
  variants (e.g., padded vs unpadded MatMul) is a planner-time decision
  that produces a different `KernelCall` variant. Kernels must not branch
  on shape or dtype to pick an algorithm.
- **No virtual dispatch in kernels.** `KernelCall` is an enum; dispatch is
  `match`. Do not introduce `Box<dyn Kernel>`, function-pointer arrays, or
  trait-object kernel registries on the hot path.
- **No TODOs or `unimplemented!()` stubs in `hologram-transform`.** Every
  public item must be fully implemented or removed.
- **Functions ≤ 15 lines and ≤ 3 args.** When a function naturally needs
  more arguments, introduce a builder struct or a parameter struct. The
  planner and executor are deliberately written this way.
- **Every public item must have tests.** New `KernelCall` variants must be
  exercised end-to-end (chain → plan → execute) in the crate's test suite.
- **Docs and specs must be updated with behaviour changes.** Any new op,
  backward rule, or kernel variant requires an update to ADR-043 (or a new
  ADR) and the matching plan.

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
