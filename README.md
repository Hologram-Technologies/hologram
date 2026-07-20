# Hologram

**One content address for everything — from tensor kernels to a portable, resumable execution substrate.**

[![CI](https://github.com/Hologram-Technologies/hologram/actions/workflows/ci.yml/badge.svg)](https://github.com/Hologram-Technologies/hologram/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Hologram compiles a tensor graph to a `.holo` archive and executes it through a
content-addressed runtime: every value carries a UOR-ADDR κ-label, so identical
computation is addressed once and reused — memoized, deduplicated, replayed —
instead of recomputed. Where a function has a finite quantum domain it is
**materialized once as a lookup table** (its compute-once form) and dispatched in
O(1). The same `.holo` runs on x86_64, WebAssembly, and ARM bare-metal (`no_std`).

The **consolidation** (`specs/refactor/`) extends that same κ-identity past the
tensor engine into a full portable substrate. A `.holo` becomes an application of
κ-addressed layers; content-addressed storage (`KappaStore`), a capability-secured
container runtime, a κ-native network (SPINE-4 `KappaSync` + a κ-XOR Kademlia DHT),
and deterministic system emulators all sit behind one **space contract** — the
trait surface any host (browser, native, or bare-metal) implements to become *a
place Hologram runs*. A single `hologram::Client` drives `compile → provision →
run` over any space, so the CLI, C ABI, and SDKs are thin shells over exactly one
type.

> **κ (kappa)** is a content-addressed label (BLAKE3 σ-axis derivation via
> `uor-addr`) — the only identity in the system: for bytes, apps, peers,
> operators, and networks. There is no second naming surface (no UUIDs, PeerIds,
> or hostnames as identity).

**Highlights**

- **One identity.** Every byte, model, container, peer, and network is a κ. No parallel naming scheme to keep in sync.
- **Compute once.** Identical work is addressed once and reused; finite-domain functions become lookup tables dispatched in O(1) — **~28×** on bf16 GELU, bit-identical.
- **One archive, every target.** The same `.holo` runs on x86_64, WebAssembly, and ARM bare-metal (`no_std` + `alloc`).
- **One contract, any host.** A browser tab, a server, or a bare-metal board becomes a *space* by implementing one trait — and a session suspends to a κ here and resumes there.
- **One surface.** The CLI, C ABI, and language SDKs are thin shells over a single `hologram::Client`, so bindings cannot drift.

---

## Contents

- **Get started** — [Quickstart](#quickstart)
- **Concepts** — [How it works: the tensor engine](#how-it-works--the-tensor-engine) · [The space substrate](#the-space-substrate)
- **Use it** — [Tensor engine (library)](#using-the-tensor-engine) · [Opening a holospace](#opening-a-holospace) · [Programmatic control](#programmatic-control) · [CLI](#cli) · [C FFI](#c-ffi)
- **Reference** — [Workspace crates](#workspace-crates) · [Feature flags](#feature-flags) · [Platform support](#platform-support) · [Benchmarks](#benchmarks) · [Build & development](#build--development)
- **Project** — [Architecture](#architecture) · [Contributing](#contributing) · [License](#license)

---

## Quickstart

**Prerequisites:** Rust (stable) and [`just`](https://github.com/casey/just).

```bash
# 1. Clone and build the workspace
git clone https://github.com/Hologram-Technologies/hologram
cd hologram
just build                                   # or: cargo build --workspace

# 2. Run the end-to-end example (parse → compile → execute → κ-address a graph)
cargo run -p hologram-cli --example pipeline

# 3. Install the single `hologram` binary (tensor engine + substrate node)
cargo install --path crates/hologram-cli

# 4. Compile native Hologram source to a content-addressed .holo, then run it
hologram compile --source graph.txt --output model.holo
hologram execute --archive model.holo
```

To use Hologram as a library, add the `hologram` facade crate and enable the
surfaces you need:

```toml
[dependencies]
hologram = {
  git = "https://github.com/Hologram-Technologies/hologram",
  features = ["archive", "backend", "compiler", "exec"],   # the tensor engine
  # add "space", "client" for the substrate contract + the programmatic `Client`
}
```

From here: run tensors from [Rust / Python / TS](#using-the-tensor-engine),
provision and boot containers with [holospaces](#opening-a-holospace), drive the
whole pipeline from one [`Client`](#programmatic-control), reach for the
[CLI](#cli), or jump to [Contributing](#contributing).

---

## How it works — the tensor engine

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
asymmetric type), i4, and the E8 lattice-codebook (1 bit/weight) VQ tier** —
flow through the fused `MatMulDequant` path, with per-tensor or per-channel
scale/zero-point; a weight bound *after* compile reaches the fused output-major
W8A8 decode GEMV by declaring its layout (`QuantAttrs::weight_layout`) rather
than by shipping constant bytes. f16 / bf16 route through the f32 engine: large
`m` widens the weight into it, and decode shapes take a dedicated streamed GEMV
(`matmul_lowp_gemv`) that widens in-register and never materializes the f32
weight. Both are bit-identical, and every first-class target has a SIMD lane —
no scalar fallback. f64 is rejected loudly.

---

## The space substrate

The refactor dissolved the transitional `substrate/` tree and merged `holospaces`
into this repo. The result is one contract layer over which any platform becomes a
*space*.

### The space contract

A **space** is any host that implements the `hologram-space` contract: the `Space`
aggregate trait naming a platform's parts — a `KappaStore`, a network `KappaSync`,
a `ContainerRuntime`, and a hardware-abstraction layer (`BlockDevice`, `Clock`,
`Entropy`, `Spawner`). *Contracts are hologram's; spaces are anyone's*: platform
differences live **behind** the traits, never in them, and conformance means
passing the `hologram-tck` battery. A space may live in an external repository
depending only on published crates — in-tree spaces are a convenience, not a
privilege.

### κ-addressed storage

`KappaStore` is the one content-addressed blob store: put content, get its κ; get
by κ, verify-on-receipt by re-derivation (SPINE-4). `hologram-store` ships three
backends behind features — `bare` (`no_std` block device), `native` (redb B-tree
with sharding + a read-through cache), and `opfs` (wasm32 browser OPFS) — each
passing the shared TCK **identically** to the in-memory reference
(`MemKappaStore`). Storage is synchronous and `no_std`-capable (law 4).

### Container runtime

`hologram-runtime` orchestrates container lifecycle over a `ContainerEngine` seam
plus a `KappaStore`: `boot` / `suspend` / `resume` / `terminate`, where the
**snapshot is itself a κ** — a session suspended on one space can be resumed on
another. Capabilities *attenuate only* (a delegated capability is always a subset
of its grantor's; amplification is unrepresentable). Engines are feature-gated:
`engine-wasmtime` (std, Wasmtime) and `engine-wasmi` (`no_std` interpreter).

### κ-native network

`hologram-net` is the SPINE-4 `KappaSync` layer: a bare-metal frame protocol, an
HTTP-CAS gateway, and a κ-XOR Kademlia DHT over TCP/QUIC. κ is the only identity
on the wire — transport-internal identifiers never leak into contracts or stored
forms. The protocol core is `no_std`; live transports are host-only features
(`live` HTTP-CAS, `tcp` DHT, `quic`).

### System emulation

`hologram-emulator` provides deterministic RISC-V / x86-64 / aarch64 cores that
boot an OS on the substrate, depending only on the space contract — the machinery
a space uses to actually run a workload's binary.

### The `Client` facade

All of it composes behind one type, `hologram::Client<S: Space>` (the `client`
feature). The CLI, C ABI, and SDKs wrap this single surface, so bindings cannot
drift:

```rust
use hologram::Client;

// A space supplies the platform: its KappaStore, network KappaSync, and container
// runtime. `Client` accepts any `impl Space` — you compose one from a `Runtime`
// + the HAL stubs + a `KappaSync` seam (see "Programmatic control" below).
let client = Client::builder().space(space).build()?;

let holo   = client.compile(graph)?;               // sync compute      (law 4)
let kappa  = client.provision(&holo)?;             // sync storage      (law 4)
let out    = client.run(&kappa, &[input]).await?;  // async resolve → sync execute

// The snapshot is a κ, so suspend-here / resume-there works across spaces.
let mut session = client.open(&container_kappa, &caps_kappa);

// Plus store/GC operations: get / pin / unpin / ls / gc / verify, and
// `.holo` v3 app tooling: inspect / is_fat / thin / fat / all_verified.
```

A full, compilable walkthrough — including the minimal `Space` wiring — is in
[Programmatic control](#programmatic-control).

---

## Using the tensor engine

Add the facade crate and enable the surfaces you need:

```toml
[dependencies]
hologram = {
  git = "https://github.com/Hologram-Technologies/hologram",
  features = ["archive", "backend", "compiler", "exec"],
}
```

`full` enables every **tensor-engine** facade module (`types`, `ops`, `graph`,
`compiler`, `exec`, `backend`, `archive`, `ffi`, `cli`, `bench`). Add `space` and
`client` on top for the substrate contract and the `Client` surface:

```toml
[dependencies]
hologram = {
  git = "https://github.com/Hologram-Technologies/hologram",
  features = ["full", "space", "client"],
}
```

Enable host-language source frontends on the root facade when your build needs
to parse embedded Hologram graph functions from Python, TypeScript, or Rust
source files:

```toml
[dependencies]
hologram = {
  git = "https://github.com/Hologram-Technologies/hologram",
  features = ["compiler", "frontend-python", "frontend-typescript", "frontend-rust"],
}
```

Run the end-to-end pipeline example, which parses a graph, compiles it to a
`.holo` archive, executes it on the CPU backend, and mints + composes
UOR-ADDR κ-labels:

```bash
cargo run -p hologram-cli --example pipeline
```

Minimal usage — compile a graph to a `.holo` archive and execute it directly on
the tensor engine (the `Client` facade wraps this same path over a space):

```rust
use hologram::backend::CpuBackend;
use hologram::compiler::{source, BackendKind, Compiler};
use hologram::exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

// Parse native Hologram source into a Graph and compile it.
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

### Source frontends

Hologram source frontends all lower through the same compile-time boundary:

```text
source text -> SourceDocument -> selected SourceProgram -> Graph -> Compiler
```

Native Hologram source is always available. Python, TypeScript, and Rust
frontends are feature-gated and host-only. They parse the host language AST and
extract restricted Hologram builder functions from larger application files;
they do not import, compile, link, evaluate, or execute host-language code.
Unrelated functions are ignored. Unsupported statements inside an inferred graph
function fail loudly with source-position diagnostics.

The CLI detects source language from the file extension (`.txt`, `.py`, `.ts`,
`.tsx`, `.rs`). Use `--source-language <lang>` only when overriding detection or
when the file uses an unusual extension. If a source file contains one inferred
graph function, `--graph` can be omitted. If it contains multiple graph
functions, pass `--graph <name>` to select one.

Programmatic graph selection uses `SourceParseOptions`:

```rust
use hologram::compiler::source::{self, SourceLanguage, SourceParseOptions};
use hologram::compiler::{BackendKind, Compiler};
use prism::vocabulary::WittLevel;

let options = SourceParseOptions::new().graph("encoder");
let program = source::parse_ir_with_options(
    python_source,
    SourceLanguage::Python,
    &options,
)?;
let graph = source::lower_ir(&program)?;
let compiled = Compiler::new(graph, BackendKind::Cpu, WittLevel::W32).compile()?;
```

When graph selection is not needed, use `compile_from_source_language`:

```rust
use hologram::compiler::source::SourceLanguage;
use hologram::compiler::{compile_from_source_language, BackendKind};
use prism::vocabulary::WittLevel;

let output = compile_from_source_language(
    python_source,
    SourceLanguage::Python,
    WittLevel::W32,
    BackendKind::Cpu,
)?;
```

#### Python source frontend

The Python frontend is feature-gated behind `frontend-python`. It parses Python
source as an AST and extracts only restricted Hologram builder functions; it
does not import, evaluate, or execute user Python code. Unrelated application
code is ignored, one inferred graph compiles by default, and files with multiple
graph functions require `--graph`.

```python
def ordinary_app_code():
    return 42

def encoder(h):
    x = h.input("x", dtype="f32", shape=[2, 3])
    w = h.const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    y = h.ops.matmul(x, w, shape=[2, 2])
    h.output("y", y)
```

```bash
cargo run -p hologram-cli --features frontend-python -- compile \
  --source graph.py \
  --graph encoder \
  --output model.holo
```

Current Python support covers `h.input`, `h.const` / `h.constant`,
`h.ops.<op>`, and `h.output` with literal `shape`, `dtype`, constant `values`,
and the same op attributes accepted by native Hologram source.

For files without a `.py` extension, pass `--source-language python`
explicitly.

#### TypeScript source frontend

The TypeScript frontend is feature-gated behind `frontend-typescript`. It uses
the TypeScript AST to extract restricted Hologram builder functions from normal
`.ts` / `.tsx` files; it does not execute user code. Plain or exported
functions with builder usage become named graph regions, unrelated application
code is ignored, and files with multiple graph functions require `--graph`.

```ts
function ordinaryAppCode() {
    return 42;
}

export function encoder(h: HologramBuilder) {
    const x = h.input("x", { dtype: "f32", shape: [2, 3] });
    const w = h.const("w", { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] });
    const y = h.ops.matmul(x, w, { shape: [2, 2] });
    h.output("y", y);
}
```

```bash
cargo run -p hologram-cli --features frontend-typescript -- compile \
  --source graph.ts \
  --graph encoder \
  --output model.holo
```

Current TypeScript support covers `h.input`, `h.const` / `h.constant`,
`h.ops.<op>`, and `h.output` with object-literal `shape`, `dtype`, constant
`values`, and the same op attributes accepted by native Hologram source.

For files without a `.ts` or `.tsx` extension, pass `--source-language
typescript` explicitly.

#### Rust source frontend

The Rust frontend is feature-gated behind `frontend-rust`. It uses `syn` to
parse Rust source as an AST and extracts only restricted Hologram builder
functions; it does not compile, link, or execute user Rust code. Plain or
exported functions with builder usage become named graph regions, unrelated
application code is ignored, and files with multiple graph functions require
`--graph`.

```rust
fn ordinary_app_code() -> i32 {
    42
}

pub fn encoder(h: &mut HologramBuilder) {
    let x = h.input("x", dtype("f32"), shape([2, 3]));
    let w = h.constant("w", shape([3, 2]), values([1, 2, 3, 4, 5, 6]));
    let y = h.ops().matmul(x, w, shape([2, 2]));
    h.output("y", y);
}
```

```bash
cargo run -p hologram-cli --features frontend-rust -- compile \
  --source graph.rs \
  --graph encoder \
  --output model.holo
```

Current Rust support covers `h.input`, `h.constant` / `h.const_`,
`h.ops().<op>`, and `h.output` with helper-call `shape`, `dtype`, constant
`values`, and the same op attributes accepted by native Hologram source.

For files without a `.rs` extension, pass `--source-language rust` explicitly.

### SDK packages

Source frontends parse explicit graph regions from host-language files. SDKs
build the same graph contract directly. The initial package scaffolds live under
[`sdk/`](sdk/):

The demos below use only package-root exports: Python re-exports `Graph`,
`Session`, `f32`, `compile_source`, and `compile_source_file` from
`hologram`; TypeScript re-exports `Graph`, `Session`, `f32`, and
`compileSource` from `@uor-foundation/sdk`; the Node adapter re-exports
`createNativeBinding` and `compileSourceFile` from `@uor-foundation/native`; the
WASM adapter re-exports `loadWasmBinding` and `createWasmBinding` from
`@uor-foundation/wasm`.

```python
import hologram as hg

g = hg.Graph("encoder")
x = g.input("x", dtype=hg.f32, shape=[2, 3])
w = g.const_ref("w", dtype=hg.f32, shape=[3, 2], file="weights.bin", blake3="0" * 64)
y = x.matmul(w, shape=[2, 2]).relu()
archive = g.output("y", y).compile()

with hg.Session.load(archive) as session:
    assert session.input_dtype(0) == hg.f32
    assert session.output_dtype(0) == hg.f32
    outputs = session.execute({"x": input_bytes})
    y_bytes = outputs["y"]
```

```ts
import { Graph, Session, f32 } from "@uor-foundation/sdk";
import { createNativeBinding } from "@uor-foundation/native";

const native = createNativeBinding();
const g = new Graph("encoder");
const x = g.input("x", { dtype: f32, shape: [2, 3] });
const w = g.constRef("w", {
  dtype: f32,
  shape: [3, 2],
  file: "weights.bin",
  blake3: "0".repeat(64),
});
const y = x.matmul(w, { shape: [2, 2] }).relu();
const archive = await g.output("y", y).compile(native);
const session = await Session.load(archive, native);
console.log(session.inputDType(0), session.outputDType(0));
const outputs = await session.execute({ x: inputBytes });
await session.close();
```

Python packages as `hologram` from `sdk/python/`; TypeScript packages as
`@uor-foundation/sdk` from `sdk/typescript/`. `@uor-foundation/native` provides the Node
N-API binding, while `@uor-foundation/wasm` provides the browser-safe adapter plus
WASM driver crate. The prebuild and installed-package smoke matrix is tracked
in [`sdk/PREBUILD.md`](sdk/PREBUILD.md).

SDKs also expose native Hologram `.txt` source compilation through the same
FFI boundary:

```python
archive = hg.compile_source_file("graph.txt")
```

```ts
import { compileSource } from "@uor-foundation/sdk";
import { compileSourceFile, createNativeBinding } from "@uor-foundation/native";

const native = createNativeBinding();
const archive = await compileSourceFile("graph.txt", native);
const inlineArchive = await compileSource("input x\nop relu x as=y\noutput y\n", native);
```

SDKs map stable FFI error codes to language-native exceptions/classes, so
callers can catch categories instead of parsing message text. Python exports
`hg.ParseError`, `hg.GraphError`, `hg.UnsupportedOpError`,
`hg.BadAttrError`, `hg.ShapeError`, `hg.ExternalTensorError`,
`hg.ArchiveLoadError`, `hg.ExecutionError`, `hg.AbiMismatchError`,
`hg.InvalidArgumentError`, `hg.UnsupportedDTypeError`, and
`hg.CompileError`. TypeScript exports the same classes plus
`errorFromCode(code, message)` and `ERROR_*` constants from `@uor-foundation/sdk`.
Where a frontend can identify source position, errors also preserve
`line`, `column`, and `rejected` fields. File-backed `const_ref` values are
read and hash-checked during compile; runtime sessions never reopen source
paths. Set `HOLOGRAM_EXTERNAL_TENSOR_ROOT` to constrain relative and absolute
external tensor paths to an explicit compile root.

Content-address and compose model parts as UOR-ADDR κ-labels:

```rust
use hologram::archive::address::{address_ring, compose_model};

let a = address_ring(&[1, 0x02, 0x01]).unwrap().address;
let b = address_ring(&[2, 0x10, 0x20, 0x30]).unwrap().address;
// CS-G2 commutative product — order-independent model identity.
let model = compose_model(&[a, b]).unwrap();
```

---

## Opening a holospace

A **holospace** is a κ-addressed application you *provision into a store* and
*boot on a runtime* — a `.holo` compute artifact, a Wasm userland, or a git
devcontainer. There is **no CLI verb** for holospaces: you open them
programmatically (Rust) or in the browser peer. The lifecycle lives in
[`spaces/holospaces`](spaces/holospaces/); the authoritative end-to-end flows are
in [`spaces/holospaces/tests/e2e.rs`](spaces/holospaces/tests/e2e.rs).

A `Source` is one of three provisioning forms:

| `Source` | What it addresses |
|---|---|
| `Source::HoloFile { artifact }` | a `.holo` compute artifact (κ) |
| `Source::Userland { entry }` | a Wasm-recompiled userland — the execution surface (κ) |
| `Source::Devcontainer { repo, reference, config, userland, arch, … }` | a git repo + `devcontainer.json` |

### From Rust — the Platform Manager

Sign in with a self-sovereign key, provision a holospace from a `Source`, then
**open** it into a session and drive its lifecycle. The suspend snapshot is a κ,
so a session suspended on one peer resumes byte-identically on any other.

```rust
use hologram_runtime::{Runtime, WasmtimeEngine};
use hologram_space::MemKappaStore;
use holospaces::identity::Operator;
use holospaces::manager::Manager;
use holospaces::peer::Peer;
use holospaces::substrate::{Capabilities, KappaStore};
use holospaces::Source;

// A self-sovereign key unlocks a content-addressed operator identity.
let operator = Operator::from_public_key(b"operator-public-key");

// A peer = a κ-store + a container runtime (the real Wasmtime engine).
let runtime = Runtime::new(WasmtimeEngine::new(), MemKappaStore::new());
let code = runtime.store().put("blake3", &wasm_userland_bytes)?; // a hologram.* Wasm userland
let peer = Peer::new(runtime.store(), &runtime);

// Capabilities attenuate only — quotas and grants a delegate can never widen.
let caps = Capabilities {
    storage_roots: Vec::new(),
    storage_quota_bytes: 0,
    network_fetch: false,
    network_announce: false,
    publish_channels: Vec::new(),
    subscribe_channels: Vec::new(),
    memory_max_bytes: 4 << 20,
    cpu_time_per_event_ms: 1000,
    priority_weight: 0,
};

// Sign in to the Platform Manager, provision, then OPEN → a session.
let mut manager = Manager::sign_in(peer, operator);
let holospace = manager.provision(Source::Userland { entry: code }, caps)?;

let mut session = manager.open(&holospace).await?; // resolves + verifies by κ (L5)
session.boot().await?;                              // now Phase::Running
let snapshot = session.suspend().await?;            // κ snapshot of live state
session.terminate().await?;
```

Prefer the lower level? Skip the Manager and use the `boot` free functions
directly: `boot::provision(store, source, caps)` mints the holospace and
`boot::Session::provision(&runtime, holospace)` (or `Peer::session(holospace)`)
begins the lifecycle. `Session::adopt(&runtime, holospace, snapshot)` resumes a
*migrated* session from its snapshot κ — the basis for suspend-here / resume-there.

Depend on it as a workspace crate — `holospaces = { workspace = true }` (default
`std`; `default-features = false` for a `no_std` peer, `features = ["net"]` for the
internet import boundary). Booting a holospace needs a `ContainerRuntime` —
`hologram_runtime::Runtime` with an engine (`WasmtimeEngine` natively; the
`wasmi`/bare-metal engines for browser and embedded).

### In the browser — the tab *is* the substrate

`spaces/holospaces-browser` compiles the Platform Manager to `wasm32` via
`wasm-bindgen`. Loading the bundle makes the browser tab a peer — no server. The
JS-facing `Console` mirrors the Rust Manager:

```js
// Bindings generated from `spaces/holospaces-browser` (wasm-bindgen).
import init, { Console } from "./holospaces_browser.js";

await init();                                  // the tab is now a Hologram peer
const console = new Console();
console.sign_in(publicKeyBytes);               // content-addressed operator identity

// Provision + boot a Wasm userland entirely in-browser; returns the κ snapshot.
const snapshot = console.boot_userland(wasmModuleBytes, 4 << 20);

// …or provision from a git devcontainer, then read the operator's view.
console.provision_devcontainer(devcontainerJson, "x64", 64 << 20);
const view = JSON.parse(console.view());       // { operator, holospaces: [κ, …] }
```

---

## Programmatic control

`hologram::Client<S: Space>` is the one typed surface the CLI, C ABI, and SDKs are
all thin shells over — so bindings cannot drift. It drives `compile → provision →
run` (and container `open`) over any space, with the store/GC and `.holo` app
tooling hanging off the same handle. Enable it with `features = ["client"]` (which
pulls `space` + `compiler` + `exec` + `backend` + `archive` + `hologram-runtime`).

A `Space` names a platform's parts (a `KappaStore`, a `KappaSync`, a
`ContainerRuntime`, and the HAL). `Client` accepts **any** `impl Space`; there is
no built-in one, so you compose a minimal space from the reference pieces — a
`Runtime` over `MemKappaStore`, the HAL stubs, and a `KappaSync` seam. The full
~40-line recipe is [`tests/client.rs`](tests/client.rs):

```rust
use hologram::space::{
    ManualClock, MemKappaStore, NoopSpawner, NullSurface, SeededEntropy, Space,
};

// The composition that makes a type a `Space` (condensed from tests/client.rs):
impl Space for MinimalSpace {
    type Store   = MemKappaStore;
    type Runtime = hologram_runtime::Runtime<hologram_runtime::MockEngine, MemKappaStore>;
    type Sync    = NullSync;       // your KappaSync — a network layer, or a no-op seam
    type Entropy = SeededEntropy;
    type Clock   = ManualClock;
    type Spawner = NoopSpawner;
    type Surface = NullSurface;
    // …seven accessors returning &self.<field>; store() delegates to runtime.store().
}
```

`MockEngine` is a deterministic in-process engine (enough to prove compute +
lifecycle); swap in `WasmtimeEngine` to actually boot Wasm containers. With a space
in hand, everything composes behind the one `Client`:

```rust
use hologram::Client;

let client = Client::builder().space(MinimalSpace::new()).build()?;

let holo   = client.compile(graph)?;               // sync compute   (law 4)
let kappa  = client.provision(&holo)?;             // sync storage   (law 4)
let out    = client.run(&kappa, &[input]).await?;  // async resolve → sync CPU execute

// Content-addressed store ops on the same handle:
let bytes = client.get(&kappa)?;      // Option<Bytes>
client.pin(&kappa)?;                   // pin / unpin / ls / verify / gc
let all   = client.ls();               // Vec<KappaLabel71>

// .holo v3 app tooling — inspect layer certificates, thin ⇄ fat (app κ unchanged):
let report = client.inspect(&holo)?;
assert!(report.all_verified());
let thin   = client.thin(&holo)?;

// The snapshot is a κ, so suspend-here / resume-there works across spaces:
let mut session = client.open(&container_kappa, &caps_kappa);
session.boot().await?;
```

Under `client`, `run` executes on the CPU backend; the builder's `.target(...)` /
`.level(...)` set the compile backend and Witt level. `gc()` is available whenever
the space's store implements `GarbageCollect` (the reference `MemKappaStore` does).

---

## CLI

`hologram-cli` builds the single `hologram` binary — the one programmatic surface
for the tensor engine *and* the substrate node. Install it with `cargo install
--path crates/hologram-cli`.

```bash
# compile native Hologram source (or an empty graph) to a .holo archive
hologram compile --source graph.txt --output model.holo

# compile a Python / TypeScript / Rust file containing Hologram builder functions
cargo run -p hologram-cli --features frontend-python -- compile \
  --source graph.py --graph encoder --output model.holo

# override extension-based language detection when needed
hologram compile --source embedded.txt --source-language python --graph encoder --output model.holo

# inspect an archive's section table
hologram inspect --archive model.holo

# execute against zero-byte inputs; prints each output port's byte length
hologram execute --archive model.holo

# micro-bench: run an archive N times, report wall-clock per iteration
hologram bench --archive model.holo --iterations 100
```

Substrate tooling is unified into the same binary (D13):

```bash
# node — a content store / router / server over a KappaStore (redb)
hologram node put weights.bin           # store bytes, print the κ-label
hologram node get <kappa>               # write a κ's canonical bytes to stdout
hologram node verify <kappa> weights.bin  # re-derive and check (SPINE-4)
hologram node serve --listen 127.0.0.1:8080 --tcp 127.0.0.1:9000  # HTTP-CAS + κ-XOR DHT
# (also: pin / unpin / gc / ls / inspect / manifest / spawn / caps — see `hologram node --help`)

# app — .holo v3 application tooling (inspect layers + certificates; fat<->thin, app κ unchanged)
hologram app inspect --archive app.holo
hologram app thin --input app.holo --output app.thin.holo
hologram app fat  --input app.holo --output app.fat.holo --store node.redb

# network — Network (VPC-analogue) realizations: membership / policy / key are all κ
hologram network create --member founder.key --policy caps.bin --tier restricted --output net.bin
hologram network show --network net.bin
hologram network delegate --parent parent.caps --child child.caps --output deleg.bin  # attenuation only
```

---

## C FFI

`hologram-ffi` exposes the pipeline through a C ABI. A session is referenced by
an integer handle into a process-local table:

```c
// compile native Hologram source into a .holo archive (written to `out`)
int len = hologram_compile_source(src, src_len, out, out_capacity);

// or build the same SourceProgram through the ABI without parsing source text
HologramSourceBuilder *b = hologram_source_builder_new();
hologram_source_builder_input(b, &input_desc);
hologram_source_builder_const(b, &small_inline_const);
hologram_source_builder_const_ref(b, &file_backed_const); // path + byte range + BLAKE3
hologram_source_builder_op(b, &op_desc);
hologram_source_builder_output(b, output_name);
int built_len = hologram_source_builder_compile(b, out, out_capacity);
if (built_len < 0) {
    int code = hologram_last_error_code();
    const char *message = hologram_last_error_message();
    size_t line = hologram_last_error_line();
    size_t column = hologram_last_error_column();
    const char *rejected = hologram_last_error_rejected();
}
hologram_source_builder_free(b);

// load an archive into a session, returning a handle (or a negative error)
int h = hologram_session_load(archive, archive_len);
int in_count  = hologram_session_input_count(h);
int out_count = hologram_session_output_count(h);

// ports carry a semantic name + shape (multi-input models map by identity)
hologram_session_input_name(h, 0, name_buf, name_cap);   // snprintf-style copy
int rank = hologram_session_input_shape(h, 0, dims, dim_cap);
int dtype = hologram_session_input_dtype(h, 0);
// (and hologram_session_output_name / output_shape / output_dtype)

// open producer-defined metadata (tokenizer, gen config, …) travels in the archive
int n = hologram_session_extension(h, key, key_len, out, out_cap); // bytes, or -1

// execute (inputs/outputs marshalled as byte buffers), then release
hologram_session_execute(h, /* … */);
hologram_session_close(h);
```

Ownership, versioning, feature probing, and error-code rules are captured in
[`specs/docs/ffi-abi-contract.md`](specs/docs/ffi-abi-contract.md).
SDK bindings should check `hologram_abi_version()` and required
`hologram_feature_supported(...)` strings before calling optional builder APIs.

Built for `wasm32-unknown-unknown` with `--features wasm`; the browser demo
under `site/` loads the resulting module.

---

## Workspace crates

Every library crate is `no_std` + `alloc` by default and exposes a `std` feature
for host builds. Applications depend on the one `hologram` facade crate and opt
into the surfaces they need (see [Root facade crate](#root-facade-crate)).

Dependencies flow in three tiers (a repo law): **core** (`crates/`, tensor engine
+ substrate) → **spaces** (`spaces/`, depend on core only) → **leaf** (facade +
`Client`, CLI, SDK packaging — may depend on anything; nothing depends on a leaf).

### Tensor engine (the content-addressed compute core)

| Crate | Role | Key types |
|---|---|---|
| `hologram-types` | Type vocabulary: dtype markers, shape markers, host/σ-axis selection (absorbs the former `hologram-host`) | `DType`, `Shape1`/`Shape2`, `Digest`, `host::HologramHasher` |
| `hologram-ops` | The closed 64-op catalog: Term-tree emitters + per-op reference evaluators | `OpKind`, `emit_op_term`, `ReferenceEvaluator` |
| `hologram-graph` | Arena DAG IR, schedules, registries, backward-graph construction | `Graph`, `Node`, `GraphOp`, `Schedule`, `ShapeRegistry` |
| `hologram-compiler` | Graph → `.holo` (lowering, fusion, fingerprint caching, source frontends) | `Compiler`, `compile`, `BackendKind`, `source` |
| `hologram-archive` | `.holo` format: sections, BLAKE3-deduped weights, per-layer certificates, footer, κ-labels | `HoloWriter`, `HoloLoader`, `SectionKind`, `compose_model` |
| `hologram-compute` | Per-target kernel dispatch (CPU SIMD/parallel, Metal, wgpu) — was `hologram-backend` | `Backend`, `KernelCall`, `Workspace`, `CpuBackend` |
| `hologram-exec` | Content-addressed sync executor, buffer pool, warm-start | `InferenceSession`, `BufferArena`, `InputBuffer`, `WarmStore` |

### The κ substrate (storage / containers / network / emulation)

| Crate | Role | Key types |
|---|---|---|
| `hologram-space` | **The space contract** + the portable σ-axis κ-addressing core | `Space`, `KappaStore`, `KappaSync`, `MemKappaStore`, `verify_kappa` |
| `hologram-tck` | Technology Compatibility Kit: the conformance battery every `KappaStore` backend must pass | `store_battery`, `MemKappaStore` |
| `hologram-store` | The `KappaStore` backends in one feature-gated crate — `bare` / `native` (redb) / `opfs` | `native::*` (redb + cache), `bare`, `opfs::OpfsKappaStore` |
| `hologram-net` | uor-native network (SPINE-4 `KappaSync`): bare frames, HTTP-CAS, κ-XOR Kademlia DHT | `bare::BareNetSync`, `http::live::HttpKappaSync`, `tcp`, `quic` |
| `hologram-runtime` | Container-runtime orchestration: lifecycle (`Session`), snapshot-as-κ, capability enforcement | `Runtime`, `lifecycle::Session`, `ContainerEngine` |
| `hologram-emulator` | Deterministic RISC-V / x86-64 / aarch64 cores that boot an OS on the substrate | `Arch`, `Emulator`, `MachineSpec` |
| `hologram-efi` | Bare-metal UEFI boot binary `hologram.efi` (workspace-excluded; `x86_64-unknown-uefi`) | measured-boot self-test over `BareMetalKappaStore` |

### Facade, tooling & conformance (leaf tier)

| Crate | Role | Key types |
|---|---|---|
| `hologram` | Facade: feature-gated re-exports + the `Client` type | `Client`, `ClientBuilder`, `Holo` |
| `hologram-cli` | The one `hologram` binary: `compile` / `execute` / `inspect` / `bench` + `node` / `app` / `network` | `cmd::run_from_env` |
| `hologram-ffi` | C ABI + WASM bindings over the CPU backend | `hologram_session_*`, `hologram_source_builder_*` |
| `hologram-bench` | Criterion performance battery (roofline, matmul, fusion, decode, …) | `[[bench]]` targets |
| `hologram-conformance` | BDD (cucumber) conformance runner + honesty meta-gate | `ConformanceWorld`, `bdd` / `meta_gate` |

### Spaces (platform implementations of the contract)

| Crate | Role | Key types |
|---|---|---|
| `spaces/holospaces` | Portable `Space` + Platform Manager: provisions & boots content-addressed environments | `Holospace`, `boot::Session`, `peer` / `manager` |
| `spaces/holospaces-browser` | wasm32 browser peer (excluded): loading the bundle makes the browser *be* the substrate — no server | `Workspace`, `WebRtcLink`, `#[wasm_bindgen]` console |
| `spaces/holospaces-node` | Edge/native peer: NIC egress + durable storage a browser routes through, OTA-updated | `EgressServer`, `storage`, `ota` |

### Root facade crate

The root `hologram` package is the application-facing import surface. It does not
add execution logic; [`src/lib.rs`](src/lib.rs) owns the export policy — each
enabled Cargo feature creates a module and re-exports the matching backing crate.

| Feature | Public surface | Backing crate |
|---|---|---|
| `types` | `hologram::types` | `hologram-types` |
| `ops` | `hologram::ops` | `hologram-ops` |
| `graph` | `hologram::graph` | `hologram-graph` |
| `compiler` | `hologram::compiler` | `hologram-compiler` |
| `archive` | `hologram::archive` | `hologram-archive` |
| `backend` | `hologram::backend` | `hologram-compute` |
| `exec` | `hologram::exec` | `hologram-exec` |
| `space` | `hologram::space` | `hologram-space` |
| `ffi` | `hologram::ffi` | `hologram-ffi` |
| `cli` | `hologram::cli` | `hologram-cli` |
| `bench` | `hologram::bench` | `hologram-bench` |
| `client` | `hologram::Client` (+ `Holo`, `ClientBuilder`) | composition over `hologram-space` + `hologram-runtime` |

The remaining substrate crates (`hologram-net`, `-store`, `-tck`, `-emulator`, and
`-runtime` beyond what `client` pulls) are currently consumed directly, wired by
the CLI's `node` command and the `spaces/` peers rather than re-exported as facade
modules. Direct dependencies on individual crates remain supported for low-level
crate authors, but applications should prefer the root facade.

---

## Feature flags

The root `hologram` crate has same-named features for the tensor-engine crates
(`types`, `ops`, `graph`, `compiler`, `exec`, `backend`, `archive`, `ffi`, `cli`,
`bench`) — `full` enables all of those. The substrate contract (`space`) and the
programmatic surface (`client`) are opt-in on top; `client` composes the space
contract, the compute hot path, and `hologram-runtime`.

Every library crate is `no_std` + `alloc` by default (so the stack runs in wasm
and on embedded targets) and exposes a `std` feature for host builds. The facade
defaults to `std` and forwards it only to enabled optional crates.

| Flag | Crate(s) | Default | Enables |
|---|---|:---:|---|
| `std` | facade + enabled libs | ✓ | Standard library: file I/O, runtime SIMD detection, thread-local scratch, `tracing` |
| `space` | `hologram-space` | — | The space contract (`Space`, `KappaStore`, `KappaSync`, HAL, realizations) |
| `client` | facade | — | The `hologram::Client<S: Space>` surface (pulls `space` + `compiler` + `exec` + `backend` + `archive` + `hologram-runtime`) |
| `backend` / `backend-cpu` | `hologram-compute` | — | The native CPU kernel backend (`CpuBackend`) |
| `backend-wgpu` | `hologram-compute` | — | The wgpu GPU backend (implies `std`) |
| `backend-metal` | `hologram-compute` | — | The Apple Metal GPU backend (implies `std`, macOS) |
| `archive-model-formats` | `hologram-archive` | — | GGUF / ONNX UOR-ADDR realizations for model addressing |
| `archive-compression` | `hologram-archive` | — | Archive compression support |
| `exec-tiered` | `hologram-exec` | — | Memory-affinity tier classification + observability |
| `backend-parallel` / `exec-parallel` | backend / exec | — | In-tree multi-core kernel dispatch |
| `frontend-python` | `hologram-compiler` | — | Python AST source frontend (implies `compiler` + `std`) |
| `frontend-rust` | `hologram-compiler` | — | Rust AST source frontend (implies `compiler` + `std`) |
| `frontend-typescript` | `hologram-compiler` | — | TypeScript AST source frontend (implies `compiler` + `std`) |
| `ffi-wasm` | `hologram-ffi` | — | WebAssembly build of the C-ABI FFI (browser demo) |

The substrate crates carry their own feature axes, consumed directly:
`hologram-store` (`bare` / `native` / `opfs`), `hologram-net` (`live` / `tcp` /
`quic`), and `hologram-runtime` (`engine-wasmtime` / `engine-wasmi`).

For `no_std` targets (wasm / embedded) disable facade default features:

```toml
hologram = { ..., default-features = false, features = ["backend", "compiler", "exec"] }
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
| `x86_64-unknown-uefi` | Boot | `hologram.efi` bare-metal boot self-test (QEMU/OVMF) |

The κ substrate (`hologram-space`, `-tck`, `-net`, `-runtime`) builds `no_std`
for both `wasm32` and `thumbv7em`; the browser space (`spaces/holospaces-browser`)
ships the wasm32 peer, and `spaces/holospaces-node` cross-compiles to small Linux
edge boards.

---

## Benchmarks

Criterion suites under `hologram-bench` (`just bench`, or `cargo bench -p
hologram-bench --bench <suite>`):

| Suite | Measures |
|---|---|
| `kernel_perf` | Kernel/roofline baselines (the release perf gate, D27) |
| `matmul` | f32 blocked-SIMD matmul throughput across sizes |
| `production` | End-to-end MLP stack (cold + content-addressed served) |
| `fusion` | Fused vs unfused kernels (matmul epilogue, dequant, broadcast) |
| `lut_activation` | f16/bf16 LUT vs computed transcendentals |
| `dequant_activation` | Densified `Dequantize → activation` vs unfused |
| `decode_gemv` | Low-precision streamed decode GEMV (`matmul_lowp_gemv`) |
| `content_reuse` | Content-addressed memo hit vs recompute |
| `tiered_executor` | Per-execute dispatch overhead (tiering) |
| `compiler` | Compile pipeline |
| `source_lowering` | Source parsing allocations, source IR lowering, and archive equivalence guards |
| `decode_step` | Archive decode + session load |

The κ-store performance floors (`sp_floors`, under `hologram-store` /
`hologram-tck`) run in `just perf`. Recorded results live in
[`BENCHMARKS.md`](BENCHMARKS.md).

---

## Build & development

Requires: Rust stable, [`just`](https://github.com/casey/just).

| Command | What it does |
|---|---|
| `just ci` | fmt check + clippy + test suite + supply-chain gate (`cargo deny`) |
| `just test` | `cargo nextest run --workspace` + the cucumber `bdd` suite |
| `just vv` | full Verification & Validation: conformance + bdd + parallel + perf + deny + wasm + embedded |
| `just conformance` | external-authority conformance suites (archive/compute/exec) |
| `just bdd` | cucumber conformance runner + honesty meta-gate (catalog ↔ scenario bijection) |
| `just bench` | Criterion benchmarks |
| `just fmt` | `cargo fmt --all` |
| `just clippy` | `cargo clippy --workspace -- -D warnings` |
| `just wasm` | Build the `no_std` stack (tensor engine + substrate) for `wasm32-unknown-unknown` |
| `just embedded` | Build the `no_std` stack for bare-metal ARM (`thumbv7em-none-eabi`) |
| `just examples` | Run the container-substrate examples (CAS cache, event bus, least-privilege, Wasm inference, live migration) |
| `just uefi-boot` | Build `hologram.efi` and boot it in QEMU/OVMF (no OS), asserting the storage self-check passes |
| `just release <version>` / `just release-auto [bump]` | Cut a lockstep workspace release via the `version-bump` GitHub workflow |

---

## Architecture

The consolidation is specified under [`specs/refactor/`](specs/refactor/) — start
with [`00-overview.md`](specs/refactor/00-overview.md) (the laws, ubiquitous
language, and decision record) and [`01-crate-map.md`](specs/refactor/01-crate-map.md)
(the target crate map + tiers + facade feature matrix).

See [`site/src/content/docs/architecture.mdx`](site/src/content/docs/architecture.mdx)
for a detailed walkthrough of the execution model, quantum levels (Q0/Q1), the
`.holo` format layout, and the compilation pipeline stages (parse → fuse → emit).

---

## Contributing

Contributions are welcome. Fork the repository, create a topic branch off `main`,
make your change, and open a pull request. `main` is governed by a merge queue,
and every PR must pass the aggregate `CI Success` check before it can land — so
run the full quality gate locally first (CI runs the same one, and it must be
green):

```bash
just ci   # fmt check + clippy (-D warnings) + test suite + supply-chain gate (cargo deny)
```

House rules (enforced in review and CI):

- Clippy is enforced with `-D warnings` — zero warnings required.
- Functions ≤ 15 lines; max 3 arguments (use the builder pattern for more).
- No `TODO`, `unimplemented!()`, or stubs — every merged feature is complete.
- `κ`-only identity: no second naming surface (no UUIDs, PeerIds, or hostnames as
  identity); transport-internal identifiers never leak into contracts or stored forms.
- `thiserror` in libraries; `anyhow` only in `hologram-cli`. No `unwrap()`/`expect()`
  on production paths. Zero `unsafe` outside documented FFI/SIMD boundaries.
- Serialisation uses rkyv exclusively; no serde on stored forms.
- SIMD behind the `simd`/`cpu` feature gate; parallelism behind `parallel`.

New crates land in the right tier (**core → spaces → leaf**; nothing depends on a
leaf) and ship with a `README.md`. The consolidation laws and the target crate map
live in [`specs/refactor/`](specs/refactor/).

---

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your option.

© UOR Foundation
