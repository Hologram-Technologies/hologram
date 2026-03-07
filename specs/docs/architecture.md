# hologram — Architecture

## System Purpose

Hologram is an O(1) compute acceleration substrate that executes computation graphs via precomputed lookup tables (LUTs) and KV-store dispatch. Rather than computing operations at runtime, all unary byte-to-byte functions are precomputed into 256-byte tables, reducing every element-wise operation to a single array index access regardless of the mathematical complexity of the original function.

---

## System Boundaries

### What hologram owns

- **LUT tables and ring algebra**: Precomputed 256-byte tables for unary operations (activations, mathematical functions), stored in `.rodata` at compile time.
- **ElementWiseView function composition**: The `then()` combinator that fuses arbitrary chains of byte-to-byte functions into a single LUT at zero runtime cost.
- **Graph IR**: The unified `Graph` type for representing computation DAGs with operations, subgraphs, and constants.
- **Execution scheduling**: Dependency-aware parallel level scheduling via `ExecutionSchedule`.
- **KV-store dispatch**: The `KvStore` type that routes `GraphOp` nodes to their corresponding O(1) kernels.
- **Buffer arena**: Memory management for intermediate values during graph execution.
- **`.holo` archive format**: rkyv-based zero-copy serialization of compiled graphs, weights, and execution metadata.
- **Compilation pipeline**: Graph validation, fusion passes, liveness analysis, and `.holo` emission.
- **Custom op registry**: Extensibility point for user-defined operations.

### What hologram does NOT own

- **AI model formats and semantics**: ONNX, GGUF, GGML import and AI-specific IR live in `hologram-ai`.
- **Runtime isolation and sandboxing**: Process, WASM, and microVM targets belong in `hologram-sandbox`.
- **Training**: Hologram is inference-first; training is out of scope.
- **Hardware-specific backends**: GPU (CUDA, Metal, WebGPU) acceleration is future scope, not core.

---

## Major Layers

1. **hologram-core**: Mathematical foundation. LUT tables, `ElementWiseView`, byte-ring algebra, encoding pipelines. Zero dependencies except `uor-foundation`. Supports `no_std` for embedded targets.

2. **hologram-graph**: Expression DAG representation. `Graph` type, `GraphBuilder`, subgraph templates, single-pass fusion engine, and `ExecutionSchedule` for parallel level ordering.

3. **hologram-archive**: Persistence layer. `.holo` binary format with rkyv zero-copy serialization, mmap loading, checksum validation, and section-based extensibility.

4. **hologram-exec**: Runtime execution. `KvStore` dispatch, `BufferArena` for intermediates, level-parallel execution via rayon, and custom op registry.

5. **hologram-compiler**: Compilation pipeline. Graph validation, constant folding, view fusion, liveness analysis, workspace layout planning, and `.holo` emission.

6. **hologram-async**: Optional async wrappers around compilation and execution for non-blocking integration with tokio runtimes.

7. **hologram-cli**: Command-line interface with `compile`, `run`, and `inspect` subcommands.

8. **hologram-ffi**: C ABI and WASM bindings for cross-language integration.

---

## Key Data Flows

1. **Graph construction**: Callers build a `Graph` via `GraphBuilder`, adding nodes (`Input`, `Lut`, `Prim`, `Output`, etc.) and edges.

2. **Compilation**: `CompilerBuilder` takes the graph, runs fusion passes to collapse chains of `LutOp` nodes into `FusedView` nodes, computes liveness intervals, plans workspace layout, and emits a `.holo` archive.

3. **Loading**: `HoloLoader` memory-maps or reads the `.holo` archive. `load_from_bytes` returns a `LoadedPlan` containing the deserialized graph, execution schedule, weights, and section table.

4. **Execution**: `KvExecutor` iterates through the `ExecutionSchedule` level by level. For each level, it dispatches nodes in parallel (if rayon enabled) via `KvStore::dispatch_with_constants`, reading inputs from the `BufferArena` and writing outputs back. Custom ops are dispatched via the `CustomOpRegistry`.

5. **Output extraction**: After execution, `GraphOutputs` contains the final output buffers which callers can retrieve by output node ID.

---

## Integration Points

| Repo/System | Integration |
|-------------|-------------|
| `hologram-ai` | Lowers `AiGraph` to `hologram::Graph` + `ExecutionSchedule`; calls `KvExecutor::execute_with_registry` |
| `hologram-sandbox` | Loads `.holo` archives and executes them inside isolated process/WASM/microVM targets |
| `uor-foundation` | Provides `Primitives` trait for type mappings; `hologram-core` implements `HoloPrimitives` |

---

## Design Constraints

- **O(1) lookup guarantee**: All element-wise operations must resolve to a single array index access. No runtime arithmetic for unary ops.
- **`no_std` core**: `hologram-core` must compile without the standard library for embedded/WASM targets.
- **rkyv only**: All serialization uses rkyv 0.8 with zero-copy deserialization. No serde.
- **Single format version**: No backwards compatibility shims. The `.holo` format has a single version at any time.
- **Max 3 function arguments**: Larger signatures use builder patterns.
- **Functions ≤ 15 lines**: Keep functions small and focused.
- **No TODOs/stubs**: All code paths must be implemented.
- **Feature-gated SIMD/parallel**: `simd` enables AVX2/SSE4.2 paths; `parallel` enables rayon.