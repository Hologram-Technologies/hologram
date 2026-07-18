# hologram-compute

> Per-target kernel dispatch for hologram, with CPU, Metal, and wgpu backends.

`hologram-compute` is the per-target dispatch layer described in spec Part IX.
Each backend declares its `HostBounds` and a `Backend` impl whose `dispatch`
consumes a `KernelCall` and writes into a runtime `Workspace`; the hot loop holds
zero virtual dispatch. The crate also exposes hologram's f32 CPU kernels through
the prism-tensor axis interface (per wiki ADR-031).

The crate is `no_std` + `alloc` by default (matching prism / uor-addr) so
hologram-ai runs in wasm and on embedded targets; the `std` feature adds
host-only backends and amenities (wgpu, runtime SIMD detection, thread-local
scratch).

## What it provides

- `Backend` — the dispatch trait; `dispatch` consumes a `KernelCall` and writes into a `Workspace`.
- `KernelCall` and friends (re-exported from `kernel_call`) — the unit of work handed to a backend.
- `Workspace`, `BufferRef`, `SplitReads` — the runtime buffer surface a backend reads and writes.
- `CpuBackend` (feature `cpu`), `MetalBackend` (feature `metal`, macOS), `WgpuBackend` (feature `wgpu`) — the concrete backends.
- `BackendError` — the crate error type.
- Prism-canonical axis impls — `HologramF32MatmulSquare`, `HologramF32Tensor{4x4,8x8,16x16}Matmul`, `HologramF32VectorActivation{,16,64,256}`, plus `HOLOGRAM_MAX_TENSOR_DIM` / `HOLOGRAM_MAX_ACTIVATION_LEN`.

## Features

- `std` (default) — std-only backends and runtime-SIMD / thread-local-scratch amenities.
- `cpu` (default) — the CPU backend and its kernels.
- `parallel` — schedules the cache-oblivious matmul recursion's disjoint sub-products across an in-tree persistent worker pool (std threads, host-only); the single-thread path stays byte-identical when off.
- `wasm-threads` — uses an embedder-provided wasm worker pool; requires a shared-memory build (atomics/simd128/bulk-memory).
- `metal` — the Metal GPU backend (macOS, host-only).
- `wgpu` — the wgpu GPU backend (host-only).

## Targets & build notes

`no_std` + `alloc` by default; the GPU backends (`metal`, `wgpu`) and the
threaded paths are host-only. `wasm-threads` needs
`RUSTFLAGS="-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals"`
(or the `wasm32-wasip1-threads` target).

Part of the [hologram](../../README.md) workspace.
