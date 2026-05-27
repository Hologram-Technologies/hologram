# Hologram

**A content-addressed, UOR-native tensor runtime**

[![CI](https://github.com/Hologram-Technologies/hologram/actions/workflows/ci.yml/badge.svg)](https://github.com/Hologram-Technologies/hologram/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Hologram compiles a tensor graph to a `.holo` archive and executes it through a
content-addressed runtime: every value carries a UOR-ADDR κ-label, so identical
computation is addressed once and reused (memoized, deduplicated, replayed)
instead of recomputed. Where a function has a finite quantum domain it is
**materialized once as a lookup table** — the compute-once form of the function —
and dispatched in O(1). The same `.holo` archive runs on x86_64, WebAssembly, and
ARM bare-metal (`no_std`).

---

## How it works

### Content-addressed execution

The runtime is one content-addressed buffer pool. A value lives in a single
aligned buffer; a slot *binds* to it by κ-label. Re-executing identical inputs
rebinds rather than recomputes (a graph-level memo hit is O(1) in graph size),
and constants are pinned for the session's lifetime. This is the "performance is
content-addressing, not micro-optimization" principle: redundant compute is
eliminated by identity, not by hand-tuning.

### LUT materialization over finite quantum domains

A pure function over a finite quantum domain is its own content-addressed table,
built bit-identically from the reference implementation:

- **f16 / bf16 transcendentals** (Sigmoid, Tanh, GELU, SiLU, Exp, Erf): the
  16-bit domain has 65536 points, so the activation is materialized once as a
  `[u16; 65536]` table (128 KB, L2-resident). Dispatch is one lookup instead of
  `widen → transcendental → narrow` — **~28× faster** on bf16 GELU, bit-identical.
- **Byte (≤8-bit) domain**: a 256-entry table.
- **Quantized inference**: a `Dequantize → activation` chain stores f32 but its
  *realized* domain is the quantized source's (256 for i8/u8, 16 for i4), so it
  densifies into a ≤256-entry table indexed by the quantized byte — **~27×**
  faster, keyed on realized information content rather than storage width.
- **f32** is computed (a 4 GB table is infeasible); reuse is structural, via the
  κ-label memo at the graph level.

### Fusion

Compile-time, the graph is desugared to primitive ops and algebraically elided
(bit-exact-sound identities / involutions / `Reshape` relabels + dead-code
elimination). At session load, content-addressed fusion passes collapse
sub-graphs so intermediates are never separately materialized:

- **Matmul epilogue** — `MatMul` / `Conv2d` absorb a following activation and/or
  bias add (`MatMulActivation`, `MatMulAddActivation`), applied in-register.
- **Dequantize → matmul** (`MatMulDequant`) — the quantized weight is
  dequantized inside the kernel; the dense f32 weight is never materialized.
- **Dequantize → activation** — densified to a quantized-domain table (above).
- **Expand → elementwise-binary** (`BroadcastBinary`) — the broadcast operand is
  read with stride-0 indexing in place; the broadcast tensor is never built.

### Matmul

f32 matmul is a cache-oblivious blocked SIMD kernel (AVX-512 → AVX2 → NEON →
portable scalar, selected at runtime) with compile-time panel-packed constant
weights (zero runtime copy). Quantized weights — **i8, u8 (ONNX's default
asymmetric type), and i4** — flow through the fused `MatMulDequant` path, with
per-tensor or per-channel scale/zero-point. f16 / bf16 widen into the same f32
engine — no scalar fallback; f64 is rejected loudly.

---

## Workspace crates

Every library crate is `no_std` + `alloc` by default and exposes a `std` feature
for host builds.

| Crate | Role | Key types |
|---|---|---|
| `hologram-host` | Platform/host bounds (register widths, capacities) | `HologramHostBounds` |
| `hologram-types` | Shared types: dtype tags, memory tiers | `MemoryTier` |
| `hologram-ops` | UOR-native op taxonomy + per-op reference semantics | `OpKind`, `ReferenceEvaluator`, `emit_op_term` |
| `hologram-graph` | Tensor graph IR, desugaring, algebraic elision, scheduling | `Graph`, `Node`, `GraphOp`, `OpKind`, `ConstantStore`, `ExecutionSchedule` |
| `hologram-compiler` | Graph → `.holo` (lowering, fusion, workspace planning) | `Compiler`, `compile`, `BackendKind`, `source` |
| `hologram-archive` | `.holo` binary format, UOR-ADDR κ-labels, BLAKE3 footer | `HoloWriter`, `HoloLoader`, `address::{address_ring, compose_model}` |
| `hologram-backend` | Kernel backends (CPU SIMD + LUT; optional wgpu/Metal) | `CpuBackend`, `Backend`, `KernelCall`, `Workspace` |
| `hologram-exec` | Content-addressed executor, buffer pool, warm-start | `InferenceSession`, `BufferArena`, `InputBuffer`, `WarmStore` |
| `hologram-ffi` | C ABI bindings (`hologram_session_*`) | C functions |
| `hologram-cli` | `hologram` binary: `compile` / `execute` / `inspect` / `bench` | — |
| `hologram-bench` | Criterion benchmark suites | — |

Depend on the individual crates you need (e.g. `hologram-compiler`,
`hologram-exec`, `hologram-backend`); there is no umbrella crate.

---

## Quick start

Add the crates you need to `Cargo.toml`:

```toml
[dependencies]
hologram-compiler = { git = "https://github.com/Hologram-Technologies/hologram" }
hologram-exec     = { git = "https://github.com/Hologram-Technologies/hologram" }
hologram-backend  = { git = "https://github.com/Hologram-Technologies/hologram" }
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
| `tiered-exec` | `hologram-exec` | — | PM_7 memory-affinity tier classification + observability |
| `parallel` | `hologram-backend` | ✓ | Rayon parallel level execution |
| `wasm` | `hologram-ffi` | — | WebAssembly build of the C-ABI FFI (browser demo) |

For `no_std` targets (wasm / embedded) disable default features on the library
crates:

```toml
hologram-backend = { ..., default-features = false }
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
| `thumbv7em-none-eabihf` | Core | `no_std` + `alloc` — library crates (no CLI / host I/O) |

---

## Benchmarks

Criterion suites under `hologram-bench` (`just bench`, or `cargo bench -p
hologram-bench --bench <suite>`):

| Suite | Measures |
|---|---|
| `matmul` | f32 blocked-SIMD matmul throughput across sizes |
| `production` | End-to-end MLP stack (cold + content-addressed served) |
| `fusion` | Fused vs unfused kernels (matmul epilogue, dequant, broadcast) |
| `lut_activation` | f16/bf16 LUT vs computed transcendentals |
| `dequant_activation` | Densified `Dequantize → activation` vs unfused |
| `content_reuse` | Content-addressed memo hit vs recompute |
| `tiered_executor` | Per-execute dispatch overhead (PM_7 tiering) |
| `compiler` | Compile pipeline |
| `decode_step` | Archive decode + session load |

Recorded results live in [`BENCHMARKS.md`](BENCHMARKS.md).

---

## CLI

`hologram-cli` builds the `hologram` binary:

```bash
# compile hologram-source (or an empty graph) to a .holo archive
hologram compile --source graph.txt --output model.holo

# inspect an archive's section table
hologram inspect --archive model.holo

# execute against zero-byte inputs; prints each output port's byte length
hologram execute --archive model.holo

# micro-bench: run an archive N times, report wall-clock per iteration
hologram bench --archive model.holo --iterations 100
```

Install:

```bash
cargo install --path crates/hologram-cli
```

---

## C FFI

`hologram-ffi` exposes the pipeline through a C ABI. A session is referenced by
an integer handle into a process-local table:

```c
// compile hologram-source into a .holo archive (written to `out`)
int len = hologram_compile_source(src, src_len, out, out_capacity);

// load an archive into a session, returning a handle (or a negative error)
int h = hologram_session_load(archive, archive_len);
int in_count  = hologram_session_input_count(h);
int out_count = hologram_session_output_count(h);

// ports carry a semantic name + shape (multi-input models map by identity)
hologram_session_input_name(h, 0, name_buf, name_cap);   // snprintf-style copy
int rank = hologram_session_input_shape(h, 0, dims, dim_cap);
// (and hologram_session_output_name / hologram_session_output_shape)

// open producer-defined metadata (tokenizer, gen config, …) travels in the archive
int n = hologram_session_extension(h, key, key_len, out, out_cap); // bytes, or -1

// execute (inputs/outputs marshalled as byte buffers), then release
hologram_session_execute(h, /* … */);
hologram_session_close(h);
```

Built for `wasm32-unknown-unknown` with `--features wasm`; the browser demo
under `site/` loads the resulting module.

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
