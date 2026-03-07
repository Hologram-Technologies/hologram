# Hologram

**O(1) compute acceleration via precomputed lookup tables**

[![CI](https://github.com/UOR-Foundation/hologram/actions/workflows/ci.yml/badge.svg)](https://github.com/UOR-Foundation/hologram/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Hologram replaces iterative computation with single-cycle array lookups. Any unary function — activation, trigonometric, logarithmic — is precomputed into a 256-entry table. Chains of such operations are fused at compile time into a single table, so `sigmoid(relu(gelu(x)))` costs the same as `identity(x)`. Quantized matrix multiplication is implemented via partial-sum booklets (LUT-GEMM), enabling GPU-free neural network inference with 16–256× fewer FLOPs than naive float matmul.

The pipeline runs on x86_64, WebAssembly, and ARM bare-metal (`no_std`) with the same `.holo` archive format across all targets.

---

## How it works

### Pi-F-lambda encoding

Continuous values are mapped into the byte domain, operated on in O(1), then mapped back:

```
f64 ──[embed: pi]──► u8 ──[LUT: F]──► u8 ──[lift: lambda]──► f64
      AngleEncoding        sin/cos             SignedEncoding
      SignedEncoding       relu/sigmoid        UnsignedEncoding
      UnsignedEncoding     gelu/silu/exp …
```

Four encoding strategies cover periodic (angle), signed-range (signed), unit-interval (unsigned), and pass-through (raw) domains.

### View fusion

`ElementWiseView` composes arbitrary chains of unary tables into a single 256-byte lookup at build time:

```rust
let fused = view_sin.then(view_relu).then(view_sigmoid);
// fused.apply(x) performs one array access regardless of chain length
```

### LUT-GEMM

Quantized weight matrices are stored as 4-bit or 8-bit indices into a codebook. Matrix–vector products are computed by accumulating precomputed partial sums rather than multiply-accumulate, achieving constant FLOP reduction over the matrix rank.

---

## Workspace crates

| Crate | Role | `no_std` | Key types |
|---|---|:---:|---|
| `hologram-core` | LUT tables, `ElementWiseView`, ring algebra, encoding | ✓ | `ElementWiseView`, `ByteRing`, `LutOp`, `PrimOp`, `AngleEncoding` |
| `hologram-graph` | Expression graph, subgraphs, fusion passes, scheduling | — | `Graph`, `GraphBuilder`, `GraphOp`, `ExecutionSchedule` |
| `hologram-archive` | `.holo` binary format, rkyv zero-copy, mmap | — | `HoloWriter`, `HoloLoader`, `HoloHeader`, `WeightDType` |
| `hologram-exec` | KV-lookup executor, buffer arena, LUT-GEMM kernels | — | `KvExecutor`, `KvStore`, `BufferArena`, `QuantizedWeights4/8` |
| `hologram-compiler` | Graph → optimised `.holo` (liveness, workspace planning) | — | `CompilerBuilder`, `CompilationStats`, `WorkspaceLayout` |
| `hologram-async` | Tokio async/await wrappers for compile + execute | — | `AsyncCompiler`, `AsyncExecutor` |
| `hologram-ffi` | C ABI (`cbindgen`) and WASM (`wasm-bindgen`) bindings | — | `HoloGraphBuilder`, `wasm_execute` |
| `hologram-cli` | `hologram compile / execute / bench` subcommands | — | — |
| `hologram-bench` | Criterion benchmarks (12 suites) | — | — |

The root `hologram` crate re-exports the full public API as a single dependency.

---

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
hologram = { git = "https://github.com/UOR-Foundation/hologram" }
```

Run the scientific calculator example, which demonstrates the full pi-F-lambda pipeline, view fusion, graph I/O, and round-trip serialisation:

```bash
cargo run --example calculator
```

Minimal usage — apply a sigmoid LUT to a byte buffer:

```rust
use hologram::core::{op::LutOp, view::ElementWiseView};

let view = ElementWiseView::from_op(LutOp::Sigmoid);
let output: Vec<u8> = input.iter().map(|&b| view.apply(b)).collect();
```

Build and execute a fused graph:

```rust
use hologram::{
    graph::builder::GraphBuilder,
    graph::fusion,
    archive::HoloWriter,
    exec::{execute_bytes, GraphInputs},
};

let mut b = GraphBuilder::default();
let x = b.input("x");
let y = b.lut(x, hologram::core::op::LutOp::Relu);
let graph = b.output("y", y).build();

let fused = fusion::fuse(graph);
let archive = HoloWriter::default().graph(&fused).build().unwrap();

let mut inputs = GraphInputs::default();
inputs.insert("x", input_bytes);
let outputs = execute_bytes(&archive, inputs).unwrap();
```

---

## Build & development

Requires: Rust stable, [`just`](https://github.com/casey/just).

| Command | What it does |
|---|---|
| `just ci` | fmt check + clippy + full test suite |
| `just test` | `cargo test --workspace` |
| `just bench` | Criterion benchmarks |
| `just fmt` | `cargo fmt --all` |
| `just clippy` | `cargo clippy --workspace -- -D warnings` |
| `just wasm` | Build `hologram-core` for `wasm32-unknown-unknown` |

---

## Feature flags

| Flag | Default | Enables |
|---|:---:|---|
| `std` | ✓ | Standard library, mmap, rkyv std |
| `simd` | ✓ | AVX2 `vpshufb` / SSE4.2 `pshufb` bulk LUT apply |
| `parallel` | ✓ | Rayon work-stealing within execution levels |
| `compiler` | ✓ | Full compilation pipeline (`hologram-compiler`) |
| `async` | — | Tokio async wrappers (`hologram-async`) |
| `ffi` | — | C ABI + WASM bindings (`hologram-ffi`) |
| `cli` | — | `hologram` binary (`hologram-cli`) |
| `wasm` | — | `wasm-bindgen` JS exports (implies `ffi`) |
| `full` | — | All of the above |

For `no_std` targets disable `std` and `simd`:

```toml
hologram = { ..., default-features = false, features = ["parallel"] }
```

---

## Platform support

| Target | Tier | Notes |
|---|---|---|
| `x86_64-unknown-linux-gnu` | Full | AVX2 SIMD, all features |
| `x86_64-apple-darwin` | Full | CI-tested on macOS |
| `x86_64-pc-windows-msvc` | Full | CI-tested on Windows |
| `wasm32-unknown-unknown` | Full | Browser + WASM runtime, `no_std` |
| `aarch64-unknown-linux-gnu` | Full | CI cross-compiled |
| `thumbv7em-none-eabihf` | Core | `no_std`, no heap — `hologram-core` only |

---

## Benchmarks

Twelve Criterion suites cover every layer:

| Suite | Measures |
|---|---|
| `lut` | Table generation, single-byte apply, 21 `LutOp` variants |
| `view` | Composition chains, SIMD `apply_slice`, rkyv round-trip |
| `kv_dispatch` | `KvStore` unary/binary at 256 B – 64 KB |
| `executor` | Linear, diamond, and wide-parallel graph topologies |
| `lut_gemm` | Q4/Q8 matmul at 16×16 – 256×256; quantisation overhead |
| `compiler` | Full compile pipeline at 10/50/100 nodes |
| `fusion` | Constant fold + CSE + view fusion at 10 – 1 000 nodes |
| `archive` | `HoloWriter` build + `HoloLoader` round-trip |
| `q1` | 16-bit quantum scaling vs Q0 and f64 |
| `async_exec` | Tokio batch execution throughput |
| `async_stream` | Token-streaming scheduling |
| `ffi` | C/WASM interface call overhead |

```bash
just bench                        # run all
cargo bench -p hologram-bench lut_gemm  # specific suite
```

CI publishes benchmark results to the docs site on every push to `main`.

---

## CLI

```bash
# compile a graph description to a .holo archive
hologram compile graph.json --output model.holo

# execute a .holo archive with named inputs
hologram execute model.holo --input x=data.bin

# profile execution
hologram bench model.holo
```

Install from workspace:

```bash
cargo install --path . --features cli
```

---

## C FFI & WebAssembly

`hologram-ffi` exposes the full pipeline via a C ABI. Headers are generated automatically by `cbindgen`:

```c
#include "include/hologram.h"

HoloGraphBuilder *b = hologram_graph_builder_new();
HoloGraphNode    x = hologram_input(b, "x");
HoloGraphNode    y = hologram_lut(b, x, HOLO_LUT_RELU);
hologram_output(b, "y", y);
HoloArchive *archive = hologram_compile(b);
hologram_graph_builder_free(b);
// … execute, free …
```

The same crate builds to a WASM module with `wasm-bindgen` JS exports when compiled with `--features wasm`.

---

## Architecture

See [`site/src/content/docs/architecture.mdx`](site/src/content/docs/architecture.mdx) for a detailed walkthrough of the execution model, quantum levels (Q0/Q1), the `.holo` format layout, and the compilation pipeline stages (parse → fuse → emit).

---

## Contributing

- Clippy is enforced with `-D warnings` — zero warnings required.
- Functions ≤ 15 lines; max 3 arguments (use the builder pattern for more).
- No `TODO`, `unimplemented!()`, or stubs — every merged feature is complete.
- Serialisation uses rkyv exclusively; no serde.
- SIMD behind `simd` feature gate; parallelism behind `parallel`.

Run the full quality gate before submitting:

```bash
just ci
```

---

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.

© UOR Foundation
