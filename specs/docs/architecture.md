# hologram — Architecture

## System Purpose

Hologram is an O(1) compute acceleration framework that replaces iterative computation with precomputed lookup table (LUT) operations. It solves the problem of making neural network inference and scientific computation dramatically faster by eliminating multiply-accumulate operations in favor of array lookups. Any unary function (activation, trigonometric, logarithmic) is precomputed into a 256-entry byte-indexed lookup table. Chains of operations are fused at compile time into a single table, so `sigmoid(relu(gelu(x)))` costs the same as a single array access—O(1) regardless of composition depth.

---

## System Boundaries

### What hologram owns

- **LUT generation and composition**: Precomputed 256-byte lookup tables for unary functions (21+ activations, 10 primitives)
- **Pi-F-Lambda encoding**: Continuous-to-byte domain mapping strategies (Angle, Signed, Unsigned, Raw)
- **Graph IR**: Arena-based expression graph with typed operations (Lut, Prim, FusedView, MatMulLut, Custom)
- **Fusion passes**: Compile-time optimization (constant folding, view fusion, CSE)
- **Parallel scheduling**: Topological sort into dependency-aware parallel levels
- **Archive format**: `.holo` binary format with rkyv zero-copy serialization and mmap support
- **KV execution**: Level-by-level dispatch via precomputed operation tables
- **Buffer management**: Liveness-based workspace allocation and reuse
- **LUT-GEMM kernels**: Quantized matrix multiplication via 4-bit/8-bit codebook lookup
- **Cross-platform execution**: x86_64 (SIMD), ARM, WebAssembly, bare-metal (`no_std`)

### What hologram does NOT own

- **AI model format parsing**: ONNX, GGUF, GGML importers belong in `hologram-ai`
- **AI-specific semantics**: Attention heads, KV-cache, token generation are `hologram-ai` concerns
- **Process/VM isolation**: Sandbox and runtime targets belong in `hologram-sandbox`
- **GPU backends**: Metal, CUDA, WebGPU kernels are future extensions
- **Training**: Hologram is inference-first; training is out of scope
- **Model serving infrastructure**: HTTP APIs, batching, load balancing are application-layer concerns

---

## Major Layers

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

## Key Data Flows

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

## Integration Points

| Integration | Direction | Mechanism |
|-------------|-----------|-----------|
| **hologram-ai** | Consumer | Lowers `AiGraph` to `hologram::Graph`, calls `KvExecutor` |
| **hologram-sandbox** | Consumer | Loads `.holo` archives, executes in isolated runtime |
| **hologram-architecture** | Upstream | ADRs and docs sync via `holoarch pull` |
| **uor-foundation** | Dependency | Trait-only foundation (`Primitives` trait) |
| **C/C++ consumers** | FFI | `hologram-ffi` provides opaque handle API |
| **Browser/JS** | FFI | `hologram-ffi` with `wasm-bindgen` feature |

---

## Design Constraints

1. **O(1) per operation**: All unary functions execute as single array lookups regardless of mathematical complexity.

2. **Zero-copy serialization**: Archives use rkyv for instant deserialization without intermediate copies; mmap enables O(1) loading.

3. **no_std compatibility**: `hologram-core` has zero dependencies beyond `uor-foundation` and runs on bare-metal ARM (thumbv7em).

4. **Cache-line alignment**: `ElementWiseView` is 256 bytes (4 cache lines) for optimal memory access patterns.

5. **Single format version**: No backwards compatibility; `.holo` format has one version at any time.

6. **SIMD feature-gated**: SIMD acceleration (`vpshufb`) behind `simd` feature; parallel execution behind `parallel` feature.

7. **Functions ≤ 15 lines**: Code convention enforces small, focused functions.

8. **Max 3 function arguments**: Builder pattern required for more complex construction.

9. **rkyv-only serialization**: All persistent types derive rkyv traits; no serde, no custom binary formats.

10. **No stubs or TODOs**: Implementation is complete or not present; no placeholder code.