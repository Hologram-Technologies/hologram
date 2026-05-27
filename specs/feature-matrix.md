# Feature Matrix — Hologram Target Compatibility (v0.5.0)

All library crates are `no_std` + `alloc` by default. A `std` feature opts in
to the host build (std error surface, host-only backends, mmap, diagnostics).
Backends are selected by **Cargo feature**, not by `build.rs` autodetection.

## Library defaults

| Trait | Value |
|-------|-------|
| Default environment | `no_std` + `alloc` |
| Host build | enable each crate's `std` feature |
| Backend selection | explicit Cargo feature (`cpu` / `parallel` / `wgpu` / `metal`) |

## Crate feature flags

### hologram-backend

| Feature | Description | Default |
|---------|-------------|---------|
| `cpu` | CPU backend (cache-oblivious recursive matmul, monomorphic kernels) | on |
| `parallel` | Multi-core level execution via an in-tree std-thread worker pool (no external dep); single-thread path is byte-identical | on |
| `std` | Host std support (enables std-only backends and amenities) | on |
| `wgpu` | WebGPU backend; implies `std` | off |
| `metal` | Apple GPU backend; implies `std` (macOS) | off |

`cpu` is the default compute backend; `parallel` is on by default and is
intra-kernel (a single `dispatch` fans its disjoint output tiles across cores).
`wgpu` and `metal` are host-only and each pull `std`.

### hologram-exec

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Host std error surface; forwards `std` to backend/archive/compiler | on |
| `tiered-exec` | PM_7 memory-affinity tier classification + observability | off |
| `parallel` | Forwards to `hologram-backend/parallel` (intra-kernel multi-core) | off |

### hologram-archive

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Host std support (mmap via `memmap2`, std forwarding) | on |
| `model-formats` | GGUF / ONNX UOR-ADDR realizations (`uor-addr/gguf`, `uor-addr/onnx`) for hologram-ai | off |
| `compression` | (reserved) | off |

### hologram-compiler

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Host std support; enables `tracing` diagnostics + std error surface | on |

### hologram-ffi

| Feature | Description | Default |
|---------|-------------|---------|
| `wasm` | WebAssembly build of the C-ABI FFI | off |

### hologram-host

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Host std support | off |

### hologram-graph / hologram-ops / hologram-types

These crates are `no_std` + `alloc` with no optional features (`default = []`).

There is **no** `accelerate`, `profile`, `serialize`, `no_alloc`, or `simd`
feature, and there is **no** `hologram-core` crate. GPU backends are gated by
the `wgpu` / `metal` Cargo features rather than detected at build time.

## Target-compatibility matrix

| Target | Env | CPU | parallel | wgpu | metal | std |
|--------|-----|:---:|:--------:|:----:|:-----:|:---:|
| `x86_64` (AVX2 / AVX-512) | std | yes | yes | yes | no | yes |
| `aarch64` (NEON) | std | yes | yes | yes | macOS only | yes |
| macOS (`aarch64`/`x86_64`) | std | yes | yes | yes | yes | yes |
| Windows (`x86_64`) | std | yes | yes | yes | no | yes |
| `wasm32-unknown-unknown` | no_std + alloc | yes | no | no | no | no |
| `thumbv7em-none-eabi` (embedded) | no_std + alloc | yes | no | no | no | no |

Notes:
- `no_std` targets (`wasm32-unknown-unknown`, `thumbv7em-none-eabi`) run the
  CPU backend single-threaded; `parallel`, the GPU backends, mmap, and the std
  error surface require `std`.
- `metal` is available only on macOS; `wgpu` is the portable GPU path on
  std-capable desktop targets.

## Execution model

There is **no** `KvExecutor` and **no** `execute_plan` / tape API. The executor
is the content-addressed `InferenceSession`
(`crates/hologram-exec/src/session.rs`) running over a single `BufferArena`
content-addressed buffer pool (`crates/hologram-exec/src/buffer.rs`): a value
lives in one aligned buffer and a slot binds to it; reuse rebinds, constants
are pinned, and residency checks elide redundant compute. The compiler lowers
the graph IR (`hologram_graph::OpKind`) into a sequence of `KernelCall`s
(`crates/hologram-backend/src/kernel_call.rs`) that the CPU backend dispatches
by exhaustive match.

## Build recipes

```bash
# Host build (std + cpu + parallel)
cargo build

# WebGPU on desktop
cargo build -p hologram-backend --features wgpu

# Apple GPU (macOS)
cargo build -p hologram-backend --features metal

# no_std core (wasm / embedded): default-features off, no std
cargo build -p hologram-backend --no-default-features --features cpu

# WebAssembly FFI
cargo build -p hologram-ffi --features wasm

# Benches with the multi-core worker pool engaged
cargo bench -p hologram-bench --features parallel
```
