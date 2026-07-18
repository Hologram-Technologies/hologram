# hologram-exec

> The hologram runtime executor: an `InferenceSession` that drives compiled kernel calls over a content-addressed buffer pool.

`hologram-exec` is the runtime executor described in spec Part VIII. It takes a
compiled/loaded plan and runs its kernel-call sequence against a pluggable
`SessionBackend`, feeding inputs and outputs through a reusable buffer arena. It
also carries the refinement (iterative validate-and-repair) machinery and a warm
store for reloading archives.

The crate is `no_std` + `alloc` by default (matching prism / uor-addr) so
inference can run in wasm and on embedded targets; the `std` feature adds
host-only amenities and the std error surface.

## What it provides

- `InferenceSession` / `SessionBackend` — the executor and the trait a compute backend implements to run kernel calls.
- `BufferArena`, `InputBuffer`, `OutputBuffer`, `SlotSpan` — the content-addressed buffer pool and its I/O views.
- `WarmStore`, `MemWarmStore`, `FileWarmStore`, `fold_archive` — warm-reload of `.holo` archives (`FileWarmStore` is `std`-only).
- `RefinementPlan`, `RefinementRunner`, `RefinementReport`, `CompiledRefinement`, and related validator/repair types — the iterative refinement (convergence) surface.
- `AttestedExecution` — prism-routed, attested execution results.
- `ExecError` — the crate error type.

## Features

- `std` (default) — enables the std error surface and forwards `std` to the runtime deps.
- `parallel` — forwards to `hologram-compute/parallel` for intra-kernel multi-core matmul; also tightens `SessionBackend` bounds with `Clone + Send + Sync`.
- `tiered-exec` — enables the tiered-execution/device-coherence module (`SlotCoherence`, `TierPolicy`, `LevelMigration`, `DeviceOwner`, `TierReport`).

## Targets & build notes

`no_std` + `alloc` by default; opt into `std` for host builds. The single-thread
path is the default; `parallel` is host-only.

Part of the [hologram](../../README.md) workspace.
