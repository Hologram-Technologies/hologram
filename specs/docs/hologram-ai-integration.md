# hologram-ai Integration Guide

## Overview

`hologram-ai` is the downstream AI-inference layer; `hologram` (v0.5.0) is the
compile-and-execute runtime it builds on. hologram has **zero knowledge of AI
model formats** — it compiles an op `Graph` into a content-addressed `.holo`
archive and executes that archive **synchronously** through an
`InferenceSession`. hologram-ai owns everything above the graph layer (format
parsing, graph lowering, tokenization, sampling).

The integration surface is small and stable:

1. Build (or lower a model into) a `hologram::graph::Graph`.
2. Compile it to a `.holo` archive with `hologram::compiler`.
3. Load + execute the archive with `hologram::exec`'s `InferenceSession`.
4. Read back raw output bytes.

Use the root `hologram` facade crate and enable the crate surfaces needed by the
integration. The implementation crates remain available for consumers that need
direct dependencies, but the facade is the recommended application-author
surface.

> **Migration note (removed architecture).** Earlier drafts described a "tape
> execution path" (`build_tape_from_plan` / `execute_tape` / `EnumTape` /
> `TapeKernel`), a `KvExecutor` (`execute_plan`), a `Float(FloatOp::…)` op
> encoding, a LUT-GEMM codebook, `build.rs` backend autodetection, and the
> `hologram-core` / `hologram-async` crates. **None of these exist in v0.5.0.**
> See "Migration from the removed tape / KvExecutor path" at the end of this
> guide.

## 1. Crates and features

| Facade module | Backing crate | Role |
|---|---|---|
| `hologram::compiler` | `hologram-compiler` | `Graph` → `.holo` archive (`Compiler` / `compile`). |
| `hologram::exec` | `hologram-exec` | Load + run an archive (`InferenceSession`, `BufferArena`, `InputBuffer`/`OutputBuffer`). |
| `hologram::backend` | `hologram-compute` | Compute backends (`CpuBackend`, plus optional GPU). |
| `hologram::archive` | `hologram-archive` | `.holo` format + content addressing (`address_ring`, `compose_model`); model-file realization under the `archive-model-formats` feature. |
| `hologram::graph` | `hologram-graph` | The op graph (`Graph`, `Node`, `GraphOp`, `OpKind`). Re-exported through the compiler's inputs. |

```toml
[dependencies]
hologram = {
  version = "0.6",
  features = ["archive", "archive-model-formats", "backend", "compiler", "exec", "graph"],
}
```

Relevant Cargo features:

- `archive-model-formats` — enables `hologram::archive::{onnx, gguf}`
  (UOR-ADDR realization of ONNX / GGUF model files; see §5). Off by default.
- `exec-tiered` — enables `InferenceSession::tier_report()` and the
  PM_7 memory-tier accessors (see §4). Off by default.
- `exec-parallel` — intra-kernel multi-core dispatch (forwards to
  `backend-parallel`).
- `backend-wgpu` / `backend-metal` — optional GPU backends
  (`CpuBackend` is always available; GPU is opt-in, *not* autodetected).

## 2. Compile a graph to a `.holo` archive

A graph is built with `hologram::graph` (there is no fluent `GraphBuilder` in
the Rust API — construct `Node`s directly, or lower your model IR into a
`Graph`).
Then compile it:

```rust
use hologram::compiler::{compile, BackendKind, CompilationOutput};
use prism::vocabulary::WittLevel;

// `graph: hologram::graph::Graph` produced by hologram-ai's lowering.
let out: CompilationOutput = compile(graph, BackendKind::Cpu, WittLevel::W32)?;
let archive: Vec<u8> = out.archive;        // the `.holo` bytes
// out.stats: CompilationStats { total_nodes, schedule_levels, cache_hits, ... }
```

The builder form is equivalent and exposes the per-compile certificate cache:

```rust
use hologram::compiler::{BackendKind, Compiler};
use prism::vocabulary::WittLevel;

let out = Compiler::new(graph, BackendKind::Cpu, WittLevel::W32).compile()?;
```

`BackendKind` selects the lowering target (`Cpu`, `Avx2`, `Avx512`, `Neon`,
`Metal`, `Wgpu`). `WittLevel::W32` is the standard quantum level for f32
inference. Compilation desugars composite ops into primitive pipelines, elides
algebraically-unnecessary computation, schedules, validates each node, and emits
the archive (kernel calls, schedule, dedup'd weights, constants, port
descriptors, certificates).

The `hologram` CLI compiles hologram-source files the same way
(`hologram compile --source m.holosrc --output m.holo`) and, by default, bakes
the warm-start fold so the runtime cache is never cold.

## 3. Load and execute via `InferenceSession`

Execution is **synchronous**. Load the archive with a backend, then call
`execute` with one `InputBuffer` per declared input port; read each
`OutputBuffer.bytes`.

```rust
use hologram::backend::CpuBackend;
use hologram::exec::{BufferArena, InferenceSession, InputBuffer, OutputBuffer};

// Backend is generic over the workspace; the session uses `BufferArena`.
let backend: CpuBackend<BufferArena> = CpuBackend::new();
let mut session = InferenceSession::load(&archive, backend)?;

// One InputBuffer per input port, in declared order. `bytes` is a raw
// little-endian tensor payload (e.g. f32 LE for an f32 port).
let x_bytes: Vec<u8> = /* token embeddings etc., little-endian */;
let outputs: Vec<OutputBuffer> = session.execute(&[InputBuffer { bytes: &x_bytes }])?;

let logits: &[u8] = &outputs[0].bytes; // raw LE bytes; hologram-ai decodes
```

For a content-addressed pipeline (e.g. autoregressive decode), prefer the
address-level surface so values flow by κ-label and are never rehashed:

- `session.intern_input(bytes) -> ContentLabel` — byte → address (hashes once).
- `session.execute_addressed(&[label, ..]) -> Vec<ContentLabel>` — runs on
  addresses; on a repeat it is an O(1) memo hit with no graph walk and no byte
  movement.
- `session.resolve(&label) -> Option<&[u8]>` — address → bytes for reading an
  output.

Port sizing helpers: `input_byte_len(i)` / `output_byte_len(i)` and
`input_ports()` / `output_ports()` (each `PortDescriptor` carries `name`, `slot`,
`element_count`, `dtype`, and full `shape`).

Multi-input models (e.g. `input_ids` / `attention_mask` / `pixel_values`) are
addressed **by identity, not by guessing positions**:
`input_port_by_name(name) -> Option<(usize, &PortDescriptor)>` and
`output_port_by_name(name)` map a model's semantic port names to the `execute`
positions. Builders register the names with `Graph::add_named_input(node, name)` /
`Graph::add_named_output(node, name)` (the older `add_input`/`add_output` still
work and leave the name empty).

Producer-defined metadata travels with the archive as **open `Extension`
sections** — a length-prefixed string `key` + arbitrary `bytes`, repeatable, one
per key. A frontend attaches a tokenizer, generation config, class labels, or a
calibration table via `Graph::add_extension(key, bytes)` (flows through
`compile()`) and reads them back at runtime with
`session.extension(key) -> Option<&[u8]>` / `session.extension_keys()`. The format
does not enumerate consumers, so hologram-ai owns the key namespace.

## 4. Observability

`InferenceSession` exposes the content-addressed runtime's footprint and the
last walk's reuse:

| Accessor | Meaning |
|----------|---------|
| `input_count()` / `output_count()` | Declared port counts. |
| `resident_bytes()` | Deduplicated content-addressed footprint (pinned constants/weights + transient inputs/intermediates). Identical content occupies one buffer. |
| `resident_count()` | Number of distinct resident values (deduped by κ-label). |
| `content_store_len()` | Distinct addressed values in the store (same count as `resident_count`; grows with novel values, flat across all-memo-hit re-runs). |
| `last_dispatched()` / `last_skipped()` | Kernels dispatched vs. elided (sub-graph reuse) in the most recent walk. |
| `kernel_count()` / `schedule_levels()` | Static schedule size. |
| `fused_count()`, `dequant_fused_count()`, `broadcast_binary_fused_count()`, … | Counts of each load-time fusion (see §6). |

With the `tiered-exec` feature:

```rust
#[cfg(feature = "tiered-exec")]
let report = session.tier_report(); // per-tier kernel histogram + migration stats
```

## 5. Content addressing and model composition

Every value the runtime operates on carries a UOR-ADDR **κ-label** — a typed,
σ-projection-grounded, replayable 71-byte content address (`blake3:<64 hex>`),
not a bare hash. Identical re-execution is recognized by label and served from
the graph memo in O(1) (no walk, no movement); identical sub-graphs / weights
collapse to one buffer.

For model **identity** (decomposition → composition), use
`hologram::archive::address`:

```rust
use hologram::archive::address::{address_ring, compose_model};

// Address each part (an Amendment-43 ring element) to a κ-label, then fold
// the parts into one model identity. compose_model uses the CS-G2 *commutative*
// product, so the model identity is independent of assembly order.
let part_a = address_ring(&canonical_bytes_a)?.address;
let part_b = address_ring(&canonical_bytes_b)?.address;
let model_id = compose_model(&[part_a, part_b])?;
```

Other addressing primitives in the same module: `address_bytes` (leaf identity
for arbitrary bytes), `derive_label` (cheap ordered reuse key — the runtime's
internal memo key), and `derive_label_witnessed` (replayable TC-05 boundary
address). The full per-axis surface (sha256, sha3-256, …) is re-exported via
`address::{ring, composition}`.

With the `archive-model-formats` feature, `hologram::archive::{onnx, gguf}` (UOR-ADDR's
ONNX / GGUF realizations) address a *model file* into κ-labels. Note these
**address** model files — they do not lower them into a `Graph`. Parsing a model
into hologram's op graph is hologram-ai's responsibility.

## 6. Quantization

hologram dequantizes packed integer weights with
`output = (q − zero_point) · scale`, in both **per-tensor** and **per-channel
(per-axis)** modes (the per-channel form supplies `scale`/`zero_point` vectors as
extra operands). Supported quantized source dtypes are **`i8`, `u8`, and `i4`**
(`u8` is ONNX's default asymmetric type — the byte is read unsigned; `i4` = two
sign-extended nibbles per byte). The `zero_point` is a full `i32`, so asymmetric
(ONNX-style) zero-points are handled.

The compiler reads a node's `QuantAttrs` (`quant_dtype`, `scale_bits`,
`zero_point`, `axis`) and the executor fuses quantized patterns at load:

- **`Dequantize → MatMul`** fuses into `MatMulDequant` — the quantized weight is
  dequantized into a transient panel *inside* the matmul, so the dense f32 weight
  is never materialized in the pool. (Count via `dequant_fused_count()`.)
- **`Dequantize → unary activation`** densifies into `DequantActivation` — a
  table over the finite quantum domain (≤256 entries for `i8`, ≤16 for `i4`),
  removing the per-element transcendental path. (Count via
  `dequant_activation_fused_count()`.)

Constant quantized weights are folded at warm-start (no runtime dequant); a
runtime dequant feeding a matmul is the dynamic case the fusion targets.

`Dequantize` is quantization-specific — it decodes a packed `i4`/`i8`/`u8` value
with scale/zero-point. A general numeric dtype conversion (e.g. an `i32`/`i64`
index or count → float) is `Cast`, **not** `Dequantize`. Likewise, an embedding
lookup lowers to a first-class `Gather` (`out[…,i,…] = data[…,indices[i],…]`, a
direct indexed row copy), not a `OneHot(indices) · table` matmul — the matmul form
does `axis_dim×` more work.

## 7. Op mapping

hologram-ai's lowering targets `hologram::graph::OpKind` (the canonical op set
defined in `hologram::ops`). Construct nodes as `GraphOp::Op(OpKind::…)`. There is
**no `Float(FloatOp::…)` encoding** — use the `OpKind` variants directly.

| AI operation | `OpKind` |
|---|---|
| Matrix multiply | `MatMul` (or `Gemm`) |
| RMS norm / layer norm | `RmsNorm`, `LayerNorm`, `AddRmsNorm` |
| Group / instance norm | `GroupNorm`, `InstanceNorm` |
| Softmax / log-softmax | `Softmax`, `LogSoftmax` |
| Activations | `Gelu`, `Silu`, `Relu`, `Sigmoid`, `Tanh`, `Elu`, `Selu`, `Erf`, … |
| Elementwise binary | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`, … |
| Attention (fused) | `Attention` |
| SwiGLU (fused) | `FusedSwiGlu` |
| Rotary position embedding | `RotaryEmbedding` |
| Embedding lookup | `Gather` (first-class, *not* `OneHot · MatMul`) |
| Numeric dtype conversion (int→float, etc.) | `Cast` (*not* `Dequantize`) |
| Dequantize a *quantized* weight | `Dequantize` (`i4`/`i8`/`u8` only — quantization-specific) |
| Layout / shape | `Reshape`, `Transpose`, `Concat`, `Slice`, `Pad`, `Expand`, `Resize` |
| Reductions | `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd` |
| Convolution / pooling | `Conv2d`, `ConvTranspose2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool` |

All float dtypes route through the one f32 engine: `f16`/`bf16` inputs widen into
it; `f64` is rejected. (See `hologram-ops` for the authoritative `OpKind` list.)

## What hologram-ai still owns

hologram provides the compile + execute engine only. hologram-ai implements:

- **Model parsers** — ONNX, safetensors, GGUF → an in-memory model IR. (hologram's
  `archive-model-formats` addresses these files but does not parse them into graphs.)
- **Graph lowering** — model IR → `hologram::graph::Graph` (mapping ops to the
  `OpKind` set in §7).
- **Tokenization & sampling** — BPE / SentencePiece, top-k / top-p / temperature,
  and the token-by-token generation loop (driving `execute` / `execute_addressed`).

## Migration from the removed tape / KvExecutor path

The "tape" and `KvExecutor` execution paths were **removed** in v0.5.0. Replace:

```rust
// OLD — removed. Does not compile against v0.5.0.
let plan   = HoloLoader::open(&path)?.load()?;
let tape   = build_tape_from_plan(&plan)?;
let outputs = execute_tape(&tape, &plan, &inputs)?;
// (and the KvExecutor form: execute_plan(&plan, &inputs)?)
```

with the single synchronous `InferenceSession` surface:

```rust
// NEW — v0.5.0.
let mut session = InferenceSession::load(&archive, CpuBackend::<BufferArena>::new())?;
let outputs = session.execute(&[InputBuffer { bytes: &input_bytes }])?;
```

Specifically removed (do not reference them): the tape path
(`build_tape_from_plan`, `execute_tape`, `EnumTape`, `TapeKernel`), `KvExecutor`
and `execute_plan`, the `Float(FloatOp::…)` / `FloatOp` op encoding, the
LUT-GEMM codebook, `build.rs` backend autodetection, and the `hologram-core` /
`hologram-async` crates. The performance story is now content addressing
(O(1) memo hits, sub-graph reuse, deduplicated residency) plus load-time fusion
(§6), not a tape; the old "17.5x / 140x tape vs KvExecutor" figures no longer
apply.
