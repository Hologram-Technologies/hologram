# hologram: Crate Layout

---

## Dependency Graph

Vertical stack (each layer depends only on the layer above it):

```
uor-foundation (git v3.5.0, traits only, no_std)
       │
   hologram-core       LUT, views, ring, encoding — no_std + alloc
       │
   hologram-graph      expression graph, subgraphs, fusion, scheduling
       │
   hologram-archive    .holo format, rkyv zero-copy, mmap, entrypoints
       │
   hologram-exec       KV executor, buffer arena, parallel levels
```

Horizontal additions (no extra core dependencies):

```
hologram-compiler    graph → optimized .holo (depends on hologram-graph)
hologram-async       Tokio async wrappers (depends on hologram-exec)
hologram-ffi         C ABI + WASM bindings (depends on hologram-compiler + hologram-exec)
hologram-cli         CLI subcommands (depends on all above)
hologram-bench       Criterion benchmarks (12 suites)
```

Root facade:

```
hologram             re-exports all public types; sole consumer dependency
```

---

## Workspace Structure

```
hologram/
├── Cargo.toml                     # workspace root
├── CLAUDE.md                      # agent instructions
├── README.md
├── crates/
│   ├── hologram/                  # root facade (re-exports everything)
│   ├── hologram-core/             # LUT primitives, views, encodings
│   ├── hologram-graph/            # graph IR, scheduling, fusion
│   ├── hologram-archive/          # .holo archive format
│   ├── hologram-exec/             # KvExecutor, BufferArena, LUT-GEMM
│   ├── hologram-compiler/         # compilation pipeline
│   ├── hologram-async/            # async execution wrappers
│   ├── hologram-ffi/              # C ABI + WASM bindings
│   ├── hologram-cli/              # CLI (compile, run, inspect)
│   └── hologram-bench/            # benchmarks
└── tests/
    └── integration/               # cross-crate integration tests
```

---

## Crate Responsibilities

### `hologram` (root facade)

The sole dependency consumers add to their `Cargo.toml`. Re-exports all public
types at the top level as `hologram::TypeName`.

| MUST | MUST NOT |
|------|----------|
| Re-export all public types from subcrates | Expose internal subcrate organization |
| Be the sole crate consumers depend on | Contain domain-specific logic (AI, sandbox) |
| Provide a flat `hologram::TypeName` namespace | Define its own types beyond re-exports |

---

### `hologram-core`

Foundational LUT primitives. `no_std` compatible with `alloc`.

**Defines:**

- `ElementWiseView` — 256-byte lookup table (64-byte aligned)
- `LutOp` — 21 activation functions with precomputed tables
- `PrimOp` — 10 primitive operations (ring arithmetic)
- `Encoding` trait + `AngleEncoding`, `SignedEncoding`, `UnsignedEncoding`, `RawEncoding`

| MUST | MUST NOT |
|------|----------|
| Remain `no_std` compatible | Depend on hologram-graph or hologram-exec |
| Provide SIMD acceleration behind `simd` feature | Contain graph or execution concepts |
| Keep `ElementWiseView` at 256 bytes, 64-byte aligned | Use serde (rkyv only) |

**Depends on:** `uor-foundation` (traits only)

---

### `hologram-graph`

The graph intermediate representation and scheduling.

**Defines:**

- `Graph`, `Node`, `Edge`, `NodeId` — arena-based graph
- `GraphOp` — operation enum (Input, Output, Lut, Prim, FusedView, Constant, MatMulLut4/8, CallSubgraph, Custom)
- `GraphBuilder` — fluent graph construction API
- `ConstantId`, `ConstantStore`, `ConstantData` — immutable data storage
- `SubgraphDef`, `SubgraphId` — hierarchical composition
- `ExecutionSchedule`, `ParallelLevel` — topological execution order
- Fusion passes: `constant_fold`, `view_fusion`, `cse`

| MUST | MUST NOT |
|------|----------|
| Own the graph IR — the single computation representation | Contain execution logic (belongs in hologram-exec) |
| Define scheduling and fusion passes | Contain memory allocation or arena logic |
| Define constant storage types | Define AI-specific operations |

**Depends on:** `hologram-core`

---

### `hologram-archive`

The `.holo` binary archive format.

**Defines:**

- `HoloHeader` — 80-byte fixed header (magic, version, offsets, checksums)
- `HoloWriter` — builder for constructing archives
- `HoloLoader` / `load_from_bytes()` — zero-copy archive loading
- `SectionTable` — custom section index

| MUST | MUST NOT |
|------|----------|
| Support page-aligned sections for mmap | Define graph IR types or execution logic |
| Use rkyv for serialization (zero-copy) | Contain domain-specific logic |
| Validate CRC checksums on load | Use serde |

**Depends on:** `hologram-graph`, `rkyv`, `crc32fast`

---

### `hologram-exec`

The execution engine.

**Defines:**

- `KvExecutor` — stateless graph executor (`execute` takes `&self`)
- `KvStore` — static dispatch table (routes `GraphOp` to O(1) kernels)
- `BufferArena` — scratch memory keyed by `NodeId`
- `CustomOpRegistry`, `CustomOpId`, `CustomHandler` — extension point
- `register_op!` macro — ergonomic custom op registration
- `lut_gemm_4bit()`, `lut_gemm_8bit()` — quantized matmul kernels
- `QuantizedWeights4`, `QuantizedWeights8`, `Psumbook4`, `Psumbook8`

| MUST | MUST NOT |
|------|----------|
| Keep `KvExecutor::execute` as `&self` (stateless, concurrent-safe) | Define graph IR types (belongs in hologram-graph) |
| Support parallel execution behind `parallel` feature | Define archive formats |
| Support custom ops via registry | Contain domain-specific logic (AI, sandbox) |

**Depends on:** `hologram-graph`, `hologram-archive`, `hologram-core`

---

### `hologram-compiler`

The compilation pipeline: graph → optimized `.holo` archive.

**Defines:**

- `compile(graph) -> CompilationOutput` — full pipeline
- Parse stage: validation, arity checks, cycle detection
- Fuse stage: constant folding + view fusion + CSE
- Plan stage: liveness analysis, workspace layout, scheduling
- Emit stage: `HoloWriter` invocation
- `FusionStats` — compilation metrics
- `WorkspaceLayout` — buffer slot assignments

| MUST | MUST NOT |
|------|----------|
| Run all optimization passes in a single topological walk | Perform AI-semantic optimizations (attention fusion, etc.) |
| Minimize peak memory via workspace planning | Depend on hologram-exec |
| Report compilation statistics | Modify the input graph (produces new output) |

**Depends on:** `hologram-graph`, `hologram-archive`

---

### `hologram-async`

Tokio async wrappers for execution.

**Depends on:** `hologram-exec`, `tokio`

---

### `hologram-ffi`

C ABI and WASM bindings.

**Depends on:** `hologram-compiler`, `hologram-exec`

---

### `hologram-cli`

Command-line interface.

**Subcommands:** `compile`, `run`, `inspect`

**Depends on:** all hologram crates, `clap`

---

### `hologram-bench`

Criterion benchmarks. 12 benchmark suites covering LUT operations, graph
construction, compilation, execution, and LUT-GEMM kernels.

**Depends on:** all hologram crates, `criterion`

---

## Dependency Matrix

```
hologram-core        → uor-foundation
hologram-graph       → hologram-core
hologram-archive     → hologram-graph, rkyv, crc32fast
hologram-exec        → hologram-graph, hologram-archive, hologram-core
hologram-compiler    → hologram-graph, hologram-archive
hologram-async       → hologram-exec, tokio
hologram-ffi         → hologram-compiler, hologram-exec
hologram-cli         → hologram-compiler, hologram-exec, hologram-archive, clap
hologram             → all subcrates (re-exports)
```

No crate in the hologram workspace depends on `hologram-ai`, `hologram-sandbox`,
or any other consumer project.

---

## Naming Convention

All crates use the `hologram-` prefix. Never use `holo-` as a shorthand.

Correct: `hologram-core`, `hologram-graph`, `hologram-exec`

Incorrect: `holo-core`, `holo-graph`, `holo-exec`
