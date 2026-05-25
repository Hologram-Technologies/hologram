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

### Compile-time fusion

Five optimisation passes run during graph compilation, before any code executes:

1. **Constant folding** — ops on compile-time constants are evaluated and replaced with a single `Const` node.
2. **View fusion (Q0)** — chains of byte-domain unary ops are composed into a single 256-byte lookup table. Involutions like `Neg∘Neg` cancel to zero-cost identity.
3. **Q1 view fusion** — same composition for 16-bit ring operations (128 KB table). Never fuses across ring-level boundaries.
4. **Epilogue fusion** — `MatMul`, `Conv2d`, and normalisation ops absorb their successor activation (and optional bias add) so the activation is applied in-register, eliminating intermediate buffers.
5. **Common subexpression elimination** — duplicate subexpressions are hash-deduplicated.

View fusion example:

```rust
let fused = view_sin.then(view_relu).then(view_sigmoid);
// fused.apply(x) performs one array access regardless of chain length
```

Epilogue fusion is the biggest memory-bandwidth win: in Stable Diffusion's UNet (512×512, 320 channels), Conv2d + Activation fusion saves ~7.7 GB of memory traffic per inference step across 23 ResNet blocks.

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

Run the end-to-end pipeline example, which parses a graph, compiles it to a
`.holo` archive, executes it on the CPU backend, and mints + composes
UOR-ADDR κ-labels:

```bash
cargo run -p hologram-cli --example pipeline
```

Minimal usage — compile a graph to a `.holo` archive and execute it:

```rust
use hologram_compiler::{source, BackendKind, Compiler};
use hologram_backend::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

// Parse a line-oriented hologram source into a Graph and compile it.
let graph = source::parse("input x\nop relu x as=y\noutput y\n").unwrap();
let compiled = Compiler::new(graph, BackendKind::Cpu, WittLevel::new(32))
    .compile()
    .unwrap();

// Load the archive and execute against the CPU backend.
let mut session =
    InferenceSession::load(&compiled.archive, CpuBackend::<BufferArena>::new()).unwrap();
let zeros = vec![0u8; 4096];
let inputs: Vec<InputBuffer> =
    (0..session.input_count()).map(|_| InputBuffer { bytes: &zeros }).collect();
let outputs = session.execute(&inputs).unwrap();
```

Content-address and compose model parts as UOR-ADDR κ-labels:

```rust
use hologram_archive::address::{address_ring, compose_model};

let a = address_ring(&[1, 0x02, 0x01]).unwrap().address;
let b = address_ring(&[2, 0x10, 0x20, 0x30]).unwrap().address;
// CS-G2 commutative product — order-independent model identity.
let model = compose_model(&[a, b]).unwrap();
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
| `just wasm` | Build the `no_std` library stack for `wasm32-unknown-unknown` |
| `just embedded` | Build the `no_std` library stack for bare-metal ARM (`thumbv7em-none-eabi`) |

---

## Feature flags

Every library crate is `no_std` + `alloc` by default (so hologram-ai runs in
wasm and on embedded targets) and exposes a `std` feature for host builds.

| Flag | Crate(s) | Default | Enables |
|---|---|:---:|---|
| `std` | all libs | ✓ | Standard library: file I/O, runtime SIMD detection, thread-local scratch, `tracing` |
| `cpu` | `hologram-backend` | ✓ | The native CPU kernel backend (`CpuBackend`) |
| `wgpu` | `hologram-backend` | — | The wgpu GPU backend (implies `std`) |
| `metal` | `hologram-backend` | — | The Apple Metal GPU backend (implies `std`, macOS) |
| `model-formats` | `hologram-archive` | — | GGUF / ONNX UOR-ADDR realizations for model addressing (hologram-ai) |
| `async` | `hologram-exec` | — | Async execution wrapper (implies `std`) |
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

## Profiling

Enable the `profile` feature to collect per-op timing, per-level breakdown, and shape propagation overhead during execution:

```bash
cargo run --features profile,cli -p hologram -- run model.holo --prompt "Hello"
```

The profile summary is printed to stderr when execution completes:

```
═══════════════════════════════════════════════════════════════
  EXECUTION PROFILE
═══════════════════════════════════════════════════════════════
  Total wall time: 1234.567ms

  OP TIMING (sorted by total time)
  ─────────────────────────────────────────────────────────────
  Op                    Calls   Total(ms)    Avg(µs)  Out(MB)  Pct(%)
  ─────────────────────────────────────────────────────────────
  MatMul                   64    890.123     13908.2    24.50    72.1%
  Attention                32    210.456      6576.8    12.25    17.0%
  RMSNorm                  64     45.678       713.7     6.00     3.7%
  ...

  LEVEL TIMING (top 10 by dispatch time)
  ─────────────────────────────────────────────────────────────
  Level    Nodes     Shape(ms)  Dispatch(ms)
  ─────────────────────────────────────────────────────────────
      0        3        0.012        0.045
     12        5        0.008       42.315
  ...
═══════════════════════════════════════════════════════════════
```

The profiling infrastructure has zero overhead when the `profile` feature is disabled. On macOS with Apple Silicon, enable `accelerate` alongside `profile` to benchmark with BLAS-accelerated MatMul and Attention:

```bash
cargo run --features profile,accelerate,cli -p hologram -- run model.holo --prompt "Hello"
```

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

## Configuration

Hologram loads settings from TOML config files. Files are checked in priority order (highest first):

1. `--config <path>` flag (explicit override)
2. `.hologram/config.toml` in the current directory (project-local)
3. `~/.hologram/config.toml` (user-global)
4. Built-in defaults

### Example `~/.hologram/config.toml`

```toml
[cache]
# Directory for decompressed archive caches.
# Compressed archives are decompressed once on first run,
# then mmap'd from cache for instant loading.
# Default: cache next to the archive file.
dir = "~/.hologram/cache"

[archive]
# Whether to compress weights/graph in new archives.
# false = larger files but instant mmap loading (default).
# true  = smaller files but requires decompression on load.
compress_weights = false
compress_graph = false

[inference]
# Default inference parameters (overridden by CLI flags).
temperature = 0.7
top_k = 40
max_tokens = 128
```

### Programmatic access

```rust
use hologram::config::HologramConfig;

// Load from standard locations (~/.hologram/config.toml, .hologram/config.toml)
let config = HologramConfig::load();

// Load from a specific file
let config = HologramConfig::load_file(Path::new("my-config.toml"))
    .unwrap_or_default();

// Access settings
if let Some(cache_dir) = config.cache_dir() {
    println!("Cache: {}", cache_dir.display());
}
```

---

## Archive loading

Hologram archives (`.holo`) support two loading modes:

| Mode | Archive type | Load time | Memory |
|------|-------------|-----------|--------|
| **Zero-copy mmap** | Uncompressed | Instant | On-demand (page faults) |
| **Decompress + cache** | Compressed | First run: seconds. Subsequent: instant | Cache file on disk |

By default, `HoloWriter` produces uncompressed archives for instant loading. Use `.compress_weights()` and `.compress_graph()` for smaller archives (e.g., for distribution), and the runtime will decompress once to a cache file.

```rust
// Uncompressed (default) — instant mmap loading
let archive = HoloWriter::new()
    .set_graph(&graph)
    .set_weights(weights)
    .build()?;

// Compressed — smaller file, decompressed on first load
let archive = HoloWriter::new()
    .set_graph(&graph)
    .set_weights(weights)
    .compress_weights()
    .compress_graph()
    .build()?;
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
