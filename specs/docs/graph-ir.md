# hologram: Graph IR

---

## Overview

The graph IR is an arena-based directed acyclic graph of byte-domain operations.
It is the single representation of computation in hologram — all compilation,
optimization, scheduling, and execution operate on this graph.

---

## Core Types

### NodeId

Generational identifier for safe arena access:

```rust
pub struct NodeId {
    index: usize,
    generation: u32,
}
```

Prevents use-after-free when nodes are removed and slots reused.

### Graph

Arena-allocated directed graph:

```rust
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    inputs: Vec<(String, NodeId)>,
    outputs: Vec<(String, NodeId)>,
    constants: ConstantStore,
    subgraphs: Vec<SubgraphDef>,
}
```

### Node

```rust
pub struct Node {
    id: NodeId,
    op: GraphOp,
    predecessors: Vec<NodeId>,
    successors: Vec<NodeId>,
}
```

---

## GraphOp

The operation enum defines all supported operations:

### Boundary Operations

| Op | Arity | Description |
|----|-------|-------------|
| `Input` | 0 | Graph-level input marker |
| `Output` | 1 | Graph-level output marker |

### Primitive Operations (`PrimOp`)

Implemented as ring operations closed under mod 256.

| Op | Arity | Description |
|----|-------|-------------|
| `Neg` | 1 | Arithmetic negation |
| `Abs` | 1 | Absolute value |
| `Copy` | 1 | Identity copy |
| `BitNot` | 1 | Bitwise complement |
| `Add` | 2 | Addition mod 256 |
| `Sub` | 2 | Subtraction mod 256 |
| `Mul` | 2 | Multiplication mod 256 |
| `BitwiseXor` | 2 | Bitwise XOR |
| `BitwiseAnd` | 2 | Bitwise AND |
| `BitwiseOr` | 2 | Bitwise OR |

### LUT Operations (`LutOp`)

Each has a precomputed 256-entry lookup table. All are unary (arity 1).

| Category | Operations |
|----------|-----------|
| Basic | `Relu`, `Sigmoid`, `Tanh`, `Abs` |
| Modern | `Gelu`, `Silu`, `Mish`, `HardSwish`, `HardSigmoid` |
| Trigonometric | `Sin`, `Cos`, `Tan` |
| Exponential | `Exp`, `Exp2`, `Log`, `Log2` |
| Algebraic | `Sqrt` |
| Inverse trig | `Asin`, `Acos`, `Atan` |

Each `LutOp` can be converted to an `ElementWiseView` via `to_view()`.

### Fused View

```rust
FusedView(ElementWiseView)
```

A precomputed 256-byte lookup table — the result of fusing a chain of unary
operations. Applied as a single array lookup per byte.

### Constants

```rust
Constant(ConstantId)
```

Reference to immutable data stored in `ConstantStore`.

### Quantized Matrix Multiplication

```rust
MatMulLut4(ConstantId)   // 4-bit quantized weights (16 centroids)
MatMulLut8(ConstantId)   // 8-bit quantized weights (256 centroids)
```

LUT-GEMM operations that resolve quantized weight matrices from `ConstantStore`.

### Subgraph Invocation

```rust
CallSubgraph(SubgraphId)
```

Template invocation. Subgraphs are flattened (inlined) before scheduling —
they do not exist at execution time.

### Custom Operations

```rust
Custom { id: CustomOpId, arity: u8 }
```

Consumer-defined operations dispatched via `CustomOpRegistry`. The `id` is
looked up in the registry at execution time to find the handler function.

---

## Constants

### ConstantData

```rust
pub enum ConstantData {
    Bytes(Vec<u8>),          // raw byte data
    Scalars(Vec<f64>),       // floating-point scalars
    Matrices(MatrixData),    // structured matrix data
}
```

### ConstantStore

Maps `ConstantId` → `ConstantData`. Constants are preserved through the entire
pipeline — never stripped at import. The store is serialized into the `.holo`
archive's weights section.

For large models, `ConstantData::Deferred` enables lazy loading via mmap
through `HoloLoader`.

### Deferred Constants

`ConstantData::Deferred` is the fourth variant, used for large weight data
that should not be eagerly loaded into memory:

```rust
pub enum ConstantData {
    Bytes(Vec<u8>),
    Scalars(Vec<f64>),
    Matrices(MatrixData),
    Deferred { offset: u64, size: u64 },
}
```

**Resolution lifecycle:**

1. When a `.holo` archive is loaded, large constants are represented as
   `Deferred { offset, size }` — pointers into the backing buffer (mmap or
   byte slice). No data is copied at load time.
2. On first access via `ConstantStore::resolve(id)`, the data is materialized:
   - For mmap-backed archives, the OS pages in the data on demand.
   - For byte-backed archives, the slice `buffer[offset..offset+size]` is
     copied into a `Vec<u8>`.
3. After resolution, the entry is replaced with `ConstantData::Bytes(...)`.
   Subsequent accesses return the materialized data directly
   (**resolve-once semantics**).

**Error handling:**

- If `offset + size` exceeds the backing buffer length, `resolve()` returns
  `Err(ConstantResolutionFailed { id, offset, size })`. Execution aborts — no
  fallback, no silent substitution, no zero-fill.

**Thread safety:**

- `ConstantStore::resolve()` takes `&self` and uses interior mutability
  (`OnceLock<Vec<u8>>` per entry). Concurrent resolution of the same constant
  from multiple threads is safe: one thread resolves, others wait on the
  `OnceLock`.

---

## Subgraphs

### SubgraphDef

```rust
pub struct SubgraphDef {
    id: SubgraphId,
    ports: Vec<Port>,     // input/output interface
    body: Graph,          // the template graph
}
```

Subgraphs enable hierarchical composition. A `CallSubgraph` node invokes a
template, which is flattened into the parent graph before scheduling.

After flattening, no `CallSubgraph` nodes remain — the scheduler sees a flat
graph of concrete operations.

---

## Graph Construction

### GraphBuilder

Fluent builder for constructing graphs:

```rust
let g = GraphBuilder::new()
    .input("x")
    .node_from_graph_input(GraphOp::Input, 0)
    .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
    .node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[1])
    .node_with_inputs(GraphOp::Output, &[2])
    .output("y", 3)
    .build();
```

Builder operations:

- `input(name)` — declare a named graph-level input
- `output(name, node_index)` — declare a named graph-level output
- `node_from_graph_input(op, input_index)` — create node connected to an input
- `node_with_inputs(op, &[predecessor_indices])` — create node with explicit predecessors
- `constant(data)` — add constant data, returns `ConstantId`
- `build()` — validate and produce the final `Graph`

---

## Invariants

1. The graph is a DAG — no cycles allowed (verified at build time)
2. Every node's predecessor count must match the operation's declared arity
3. Every `Output` node has exactly one predecessor
4. Every `Input` node has zero predecessors
5. Every `Constant(id)` must have a corresponding entry in `ConstantStore`
6. Every `CallSubgraph(id)` must reference a defined `SubgraphDef`
7. Every `Custom { id, arity }` must have a handler in `CustomOpRegistry` at execution time
8. `NodeId` generational access prevents stale references

### Degenerate Graph Behavior

The following edge cases are explicitly valid:

- **Empty graph** (zero nodes): `compile()` produces a valid `.holo` archive
  with an empty graph section and zero-level schedule. `execute()` returns an
  empty `GraphOutputs`.
- **No `Input` nodes**: valid (e.g., a pure-constant graph). `execute()`
  ignores `GraphInputs`.
- **No `Output` nodes**: valid but produces no results. `execute()` returns
  an empty `GraphOutputs`. The compiler MAY emit a warning but MUST NOT
  reject the graph.
- **Disconnected components**: valid. All nodes are scheduled; disconnected
  subgraphs land in parallel levels naturally. The compiler does not reject
  disconnected graphs.
- **Dead code** (nodes with no path to any `Output`): the compiler MAY
  eliminate dead nodes during optimization but is not required to. If
  retained, they are scheduled and executed normally — their outputs are
  simply never collected.
- **Single-node graphs** (e.g., `Input → Output`): valid. Compiles to a
  single-level schedule.
