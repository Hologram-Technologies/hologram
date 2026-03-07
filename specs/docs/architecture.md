# hologram — Full Architecture

---

## 1. System Purpose

Hologram is an O(1) compute acceleration framework that replaces iterative computation with precomputed lookup table (LUT) operations. It solves the problem of making neural network inference and scientific computation dramatically faster by eliminating multiply-accumulate operations in favor of array lookups. Any unary function (activation, trigonometric, logarithmic) is precomputed into a 256-entry byte-indexed lookup table. Chains of operations are fused at compile time into a single table, so `sigmoid(relu(gelu(x)))` costs the same as a single array access—O(1) regardless of composition depth.

The canonical pipeline:

```
Graph construction
        │
   ┌────▼─────────────────────────┐
   │   Graph IR                   │  hologram-graph
   │   (nodes, edges, constants)  │
   └────┬─────────────────────────┘
        │ Graph
   ┌────▼─────────────────────────┐
   │   Compiler                   │  hologram-compiler
   │   (parse → fuse → plan)     │
   └────┬─────────────────────────┘
        │ ExecutionSchedule + .holo archive
   ┌────▼─────────────────────────┐
   │   Archive                    │  hologram-archive
   │   (serialize / load)        │
   └────┬─────────────────────────┘
        │ SerializedGraph + weights
   ┌────▼─────────────────────────┐
   │   KvExecutor                 │  hologram-exec
   │   (O(1) dispatch per node)  │
   └────┬─────────────────────────┘
        │ GraphOutputs
        ▼
```

---

## 2. System Boundaries

### hologram owns

- **Graph IR**: `Graph`, `Node`, `GraphOp`, `NodeId`, `Edge`
- **Constants**: `ConstantId`, `ConstantStore`, `ConstantData`
- **Scheduling**: `ExecutionSchedule`, `ParallelLevel`
- **Execution**: `KvExecutor`, `KvStore`, `BufferArena`
- **Extension**: `CustomOpRegistry`, `CustomOpId`, `CustomHandler`
- **LUT primitives**: `ElementWiseView`, `LutOp`, `PrimOp`, `Encoding`
- **Compilation**: `compile()`, fusion passes, workspace planning
- **Archive**: `HoloWriter`, `HoloLoader`, `HoloHeader`
- **Quantized matmul**: `lut_gemm_4bit()`, `lut_gemm_8bit()`, `QuantizedWeights4/8`
- **CLI**: `hologram compile`, `hologram run`, `hologram inspect`
- **LUT generation and composition**: Precomputed 256-byte lookup tables for unary functions (21+ activations, 10 primitives)
- **Pi-F-Lambda encoding**: Continuous-to-byte domain mapping strategies (Angle, Signed, Unsigned, Raw)
- **Fusion passes**: Compile-time optimization (constant folding, view fusion, CSE)
- **Cross-platform execution**: x86_64 (SIMD), ARM, WebAssembly, bare-metal (`no_std`)

### hologram does NOT own

- **AI model format parsing** (ONNX, GGUF, GGML) — belongs to `hologram-ai`
- **AI-specific operations** (attention, norm, rope) — registered via `CustomOpRegistry`
- **AI-specific semantics**: Attention heads, KV-cache, token generation are `hologram-ai` concerns
- **Sandbox isolation** (processes, WASM, microVMs) — belongs to `hologram-sandbox`
- **Tokenization, sampling, streaming** — belongs to `hologram-ai`
- **GPU backends**: Metal, CUDA, WebGPU kernels are future extensions
- **Training**: Hologram is inference-first; training is out of scope
- **Model serving infrastructure**: HTTP APIs, batching, load balancing are application-layer concerns

---

## 2a. Consumer Contract

Consumer projects (`hologram-ai`, `hologram-sandbox`, and any third-party
crate) depend on hologram as a library. The following rules govern this
relationship:

### Read-only dependency

Consumer projects MUST NOT modify files in the `hologram` repository.
Hologram is consumed via `Cargo.toml` — not co-edited. If a hologram type or
API is insufficient, the correct action is to propose an ADR in
`hologram-architecture`, not to fork or patch hologram directly.

### Extension-only interface

Consumers extend hologram exclusively through these defined extension points:

| Extension Point | Purpose |
|----------------|---------|
| `CustomOpRegistry` | Register domain-specific operation handlers |
| `HoloWriter::add_section(kind >= SECTION_CUSTOM_BASE)` | Add custom archive sections |
| `GraphBuilder` | Construct graphs from consumer-owned IRs |
| `hologram::compile()` | Invoke the compiler as a library |
| `GraphInputs` / `GraphOutputs` | Provide data to and receive results from execution |

No other mechanism is sanctioned. Consumers MUST NOT use unsafe code to
access hologram internals, bypass visibility modifiers, or depend on
undocumented behavior.

### No parallel types

Consumers MUST NOT define types that duplicate hologram types. Examples of
prohibited duplication:

- A consumer-defined graph IR alongside `hologram::Graph`
- A consumer-defined execution plan alongside `ExecutionSchedule`
- A consumer-defined buffer arena alongside `BufferArena`
- A consumer-defined archive format alongside `.holo`

If a hologram type is insufficient, propose an ADR — do not work around it.

### No direct subcrate imports

All types are accessed via `hologram::TypeName`, never via
`hologram_graph::TypeName` or `hologram_exec::TypeName` directly. This
allows hologram to restructure its internal subcrate layout without breaking
consumers.

### Agent directive

AI agents (Claude, Cursor, Codex, or any other framework) operating in
consumer repositories MUST NOT edit, create, or delete files in the
`hologram` repository or any other sibling repository unless the user
explicitly instructs them to do so.

---

## 3. Core Abstraction: ElementWiseView

The fundamental unit of computation in hologram is the 256-byte lookup table.

```rust
#[repr(align(64))]
pub struct ElementWiseView {
    table: [u8; 256],
}
```

Every unary operation (activation function, trigonometric function, etc.) is
precomputed into a 256-entry table mapping `u8 → u8`. The key operations:

- **`apply(byte: u8) -> u8`** — single array lookup, O(1)
- **`then(&self, other: &Self) -> Self`** — compose two tables into one (O(256) one-time)
- **`apply_slice(&self, data: &mut [u8])`** — SIMD-accelerated batch apply (AVX2 `vpshufb`)

Composition is the foundation of the fusion optimization: any chain of unary
operations collapses into a single table, regardless of chain length. The table is 256 bytes (4 cache lines) for optimal memory access patterns.

---

## 4. Encoding System

The `Encoding` trait bridges continuous floating-point values and the byte
domain where all hologram computation occurs.

```rust
pub trait Encoding {
    fn embed(&self, value: f64) -> u8;   // continuous → byte
    fn lift(&self, byte: u8) -> f64;     // byte → continuous
    fn name(&self) -> &'static str;
}
```

Four implementations:

| Encoding | Domain | Use case |
|----------|--------|----------|
| `AngleEncoding` | Periodic | sin, cos, tan |
| `SignedEncoding` | [-1, 1] | tanh, signed activations |
| `UnsignedEncoding` | [0, 1] | sigmoid, probabilities |
| `RawEncoding` | Pass-through | raw byte operations |

---

## 5. Graph IR

The graph is an arena-based directed acyclic graph. See [graph-ir.md](graph-ir.md)
for the full specification.

Key properties:

- Arena-allocated nodes with generational `NodeId` for safe access
- Named inputs and outputs for external interface
- `ConstantStore` for immutable data (weights, precomputed tables)
- `SubgraphDef` / `CallSubgraph` for hierarchical composition
- Fluent `GraphBuilder` API for construction
- Typed operations: `Lut`, `Prim`, `FusedView`, `MatMulLut`, `Custom`

---

## 6. Compilation Pipeline

Three-stage pipeline. See [compilation.md](compilation.md) for the full specification.

```
Parse → Fuse → Plan & Emit
```

- **Parse**: validate structure, check arities, verify acyclicity
- **Fuse**: constant folding + view fusion + CSE in a single topological walk
- **Plan & Emit**: liveness analysis → workspace slot allocation → scheduling → `.holo` archive

### Compilation Flow

```
GraphBuilder API        →  Graph (arena-based)
                              ↓
CompilerBuilder.build() →  Validate (structure check)
                              ↓
                           Fuse (constant fold, view fusion, CSE)
                              ↓
                           Schedule (topo sort → parallel levels)
                              ↓
                           Liveness (compute buffer lifetimes)
                              ↓
                           Emit (HoloWriter → .holo archive)
```

---

## 7. Execution Model

Stateless O(1) dispatch. See [execution.md](execution.md) for the full specification.

```
KvExecutor::execute(graph, schedule, inputs) → GraphOutputs
```

- Nodes dispatched via `KvStore` (static dispatch table)
- `BufferArena` provides scratch memory between levels
- Nodes within a level can run concurrently (Rayon, behind `parallel` feature)
- Custom ops dispatched via `CustomOpRegistry`

### Execution Flow

```
.holo archive           →  HoloLoader (mmap, zero-copy)
                              ↓
                           LoadedPlan (graph + weights + metadata)
                              ↓
build_schedule()        →  ExecutionSchedule (Vec<ParallelLevel>)
                              ↓
GraphInputs             →  KvExecutor.execute()
                              ↓
                           For each level (Rayon parallel):
                             For each node:
                               Read inputs from BufferArena
                               Dispatch via KvStore (O(1) lookup)
                               Write output to BufferArena
                              ↓
                           GraphOutputs (named results)
```

### Memory Layout During Execution

```
BufferArena:
  ┌─────────────────────────────────┐
  │ slot_0 [256 B] ← input          │
  │ slot_1 [256 B] ← temp (reused)  │
  │ slot_2 [256 B] ← temp (reused)  │
  │ slot_3 [256 B] ← output         │
  └─────────────────────────────────┘
       ↑
   Liveness-based reuse (computed at compile time)
```

---

## 8. Archive Format

Binary `.holo` format with page-aligned sections. See [archive-format.md](archive-format.md)
for the full specification.

- 80-byte header with CRC checksums
- rkyv zero-copy serialization (no serde)
- Page-aligned sections (4 KB) enable mmap for instant O(1) loading
- Sections: graph, weights, custom, section table

---

## 9. CLI

Command-line interface for compiling, running, and inspecting `.holo` archives.
See [cli.md](cli.md) for the full specification.

```sh
hologram compile --input graph.bin --output model.holo
hologram run model.holo --input 0:deadbeef
hologram inspect model.holo
```

---

## 10. LUT-GEMM (Quantized Matrix Multiplication)

Replaces O(k) multiply-accumulate with O(Q) lookups using partial-sum booklets.

```
For C = A × W where W is quantized to Q levels:
1. K-means clusters weights into Q centroids (compile-time)
2. For each C[i,j]:
   - Build Psumbook: sums[q] = Σ A[i,l] for all l where index[l,j] == q
   - C[i,j] = Σ sums[q] × centroid[q]
```

Two quantization levels:

| Variant | Levels | Index size | Booklet |
|---------|--------|-----------|---------|
| Q4 (`MatMulLut4`) | 16 | 4-bit (2 per byte) | 16 accumulators |
| Q8 (`MatMulLut8`) | 256 | 8-bit | 256 accumulators |

Parallel variants behind the `parallel` feature flag.

---

## 11. Ring Arithmetic

Primitive operations (`PrimOp`) are implemented as ring operations closed under
mod 256. This means all binary operations (add, sub, mul, xor, and, or) operate
on byte values and produce byte values with no overflow or precision concerns.

---

## 12. Major Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / FFI / WASM                         │
│  hologram-cli (compile, run, inspect)                           │
│  hologram-ffi (C ABI, wasm-bindgen)                             │
├─────────────────────────────────────────────────────────────────┤
│                      Async Wrappers (optional)                  │
│  hologram-async (Tokio spawn_blocking, streaming)               │
├─────────────────────────────────────────────────────────────────┤
│                       Compilation Pipeline                       │
│  hologram-compiler (parse → fuse → emit)                        │
│  • validate graph structure                                     │
│  • fusion: constant fold, view composition, CSE                 │
│  • liveness analysis, workspace planning                        │
│  • schedule generation, archive emission                        │
├─────────────────────────────────────────────────────────────────┤
│                         Execution Layer                          │
│  hologram-exec (KvExecutor, BufferArena, CustomOpRegistry)      │
│  • level-by-level dispatch via KvStore                          │
│  • Rayon parallel execution within levels                       │
│  • LUT-GEMM kernels (Q4/Q8 matmul)                              │
├─────────────────────────────────────────────────────────────────┤
│                      Serialization Layer                         │
│  hologram-archive (.holo format, HoloWriter, HoloLoader)        │
│  • rkyv zero-copy serialization                                 │
│  • mmap for instant archive loading                             │
│  • 4 KB page alignment, CRC32 checksums                         │
├─────────────────────────────────────────────────────────────────┤
│                          Graph IR Layer                          │
│  hologram-graph (Graph, GraphBuilder, ExecutionSchedule)        │
│  • arena-based node storage with generation versioning          │
│  • subgraph templates for reusable patterns                     │
│  • topological sort into parallel levels                        │
├─────────────────────────────────────────────────────────────────┤
│                           Core Layer                             │
│  hologram-core (ElementWiseView, LutOp, ByteRing, encoding)     │
│  • 256-byte cache-line-aligned lookup tables                    │
│  • SIMD-accelerated bulk apply (AVX2 vpshufb)                   │
│  • no_std compatible, zero external dependencies                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Feature Flags

| Feature | Enables | Default |
|---------|---------|---------|
| `std` | Standard library, mmap, rkyv std | Yes |
| `simd` | AVX2 / SSE4.2 LUT acceleration | Yes |
| `parallel` | Rayon parallel level execution | Yes |
| `compiler` | Full compilation pipeline | Yes |
| `async` | Tokio async wrappers | No |
| `ffi` | C ABI + WASM bindings | No |
| `cli` | CLI subcommands | No |
| `wasm` | WASM bindings (implies `ffi`) | No |
| `full` | All features | No |

For `no_std` targets:

```toml
hologram = { ..., default-features = false, features = ["parallel"] }
```

---

## 14. Platform Support

| Target | Tier | Notes |
|--------|------|-------|
| `x86_64-unknown-linux-gnu` | Full | AVX2 SIMD, all features |
| `x86_64-apple-darwin` | Full | CI-tested on macOS |
| `x86_64-pc-windows-msvc` | Full | CI-tested on Windows |
| `wasm32-unknown-unknown` | Full | Browser + WASM runtime, `no_std` |
| `aarch64-unknown-linux-gnu` | Full | CI cross-compiled |
| `thumbv7em-none-eabihf` | Core | `no_std`, no heap — `hologram-core` only |

---

## 15. Serialization

All serialization uses **rkyv** exclusively. There is no serde dependency anywhere
in the hologram codebase. rkyv provides zero-copy deserialization, which is critical
for fast archive loading and mmap compatibility.

---

## 16. Design Rules

Enforced throughout the codebase:

1. **O(1) per operation**: All unary functions execute as single array lookups regardless of mathematical complexity
2. **Zero-copy serialization**: Archives use rkyv for instant deserialization without intermediate copies; mmap enables O(1) loading
3. **Max 3 function arguments** — use builder pattern for more
4. **Functions ≤ 15 lines** — keep functions focused
5. **No TODOs, stubs, or `unimplemented!()`** — all code is complete
6. **Serialization**: rkyv exclusively; no serde
7. **SIMD**: behind `simd` feature gate
8. **Parallelism**: behind `parallel` feature gate
9. **Zero warnings** with `-D warnings` clippy enforcement
10. **Cache-line alignment**: `ElementWiseView` is 256 bytes (4 cache lines) for optimal memory access patterns
11. **Single format version**: No backwards compatibility; `.holo` format has one version at any time
12. **no_std compatibility**: `hologram-core` has zero dependencies beyond `uor-foundation` and runs on bare-metal ARM (thumbv7em)

---

## 17. Integration Points

| Integration | Direction | Mechanism |
|-------------|-----------|-----------|
| **hologram-ai** | Consumer | Lowers `AiGraph` to `hologram::Graph`, calls `KvExecutor` |
| **hologram-sandbox** | Consumer | Loads `.holo` archives, executes in isolated runtime |
| **hologram-architecture** | Upstream | ADRs and docs sync via `holoarch pull` |
| **uor-foundation** | Dependency | Trait-only foundation (`Primitives` trait) |
| **C/C++ consumers** | FFI | `hologram-ffi` provides opaque handle API |
| **Browser/JS** | FFI | `hologram-ffi` with `wasm-bindgen` feature |

---

## 18. Types Provided to Subprojects

| Type | Crate | Purpose |
|------|-------|---------|
| `Graph` | `hologram-graph` | Directed graph of operations |
| `ExecutionSchedule` | `hologram-graph` | Topological execution order |
| `ConstantId` / `ConstantStore` | `hologram-graph` | Immutable data storage |
| `KvExecutor` | `hologram-exec` | Stateless graph executor |
| `BufferArena` | `hologram-exec` | Scratch memory management |
| `CustomOpRegistry` | `hologram-exec` | Extension point for domain ops |
| `compile()` | `hologram-compiler` | Graph → optimized schedule |
| `HoloLoader` / `HoloWriter` | `hologram-archive` | `.holo` archive I/O |

All accessed via `hologram::TypeName` — never via internal subcrates directly.

---

## 19. Crate Access Rule

All types defined in hologram subcrates (`hologram-core`, `hologram-graph`,
`hologram-exec`, `hologram-archive`, etc.) are accessible **only via the
`hologram` root crate**. Consumers MUST import `hologram::Graph`, never
`hologram_graph::Graph` directly.

This allows hologram to restructure its internal subcrate layout without
breaking consumers.
