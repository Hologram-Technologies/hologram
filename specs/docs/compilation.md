# hologram: Compilation Pipeline

---

## Overview

The compiler transforms a `Graph` into an optimized `.holo` archive containing
a `SerializedGraph`, `ExecutionSchedule`, and packed weights. The pipeline runs
in three stages.

```
Graph
  │
  ├── Stage 1: Parse
  │   validate structure, check arities, verify acyclicity
  │
  ├── Stage 2: Fuse
  │   constant folding → view fusion → CSE (single topological walk)
  │
  └── Stage 3: Plan & Emit
      liveness analysis → workspace layout → scheduling → .holo archive
```

---

## Entry Point

```rust
pub fn compile(graph: Graph) -> Result<CompilationOutput>

pub struct CompilationOutput {
    pub archive: Vec<u8>,              // serialized .holo archive
    pub schedule: ExecutionSchedule,   // parallel level schedule
    pub stats: CompilationStats,       // fusion + planning metrics
}
```

---

## Stage 1: Parse

Validates the graph before any transformation.

**Checks performed:**

1. **Arity verification** — every node's predecessor count matches its `GraphOp`'s declared arity
2. **Edge consistency** — all edges reference valid `NodeId`s
3. **Acyclicity** — the graph contains no cycles (topological sort must succeed)
4. **Constant resolution** — every `Constant(id)` has a corresponding entry in `ConstantStore`
5. **Subgraph resolution** — every `CallSubgraph(id)` references a defined `SubgraphDef`
6. **Input/output well-formedness** — `Input` nodes have zero predecessors, `Output` nodes have exactly one

Parse failures produce structured errors with the offending `NodeId` and reason.

---

## Stage 2: Fuse

Three optimizations applied in a single topological walk:

### Constant Folding

Detects nodes where **all inputs are constants**. Evaluates the operation at
compile time and replaces the node with a `Constant` node.

```
Before:  Constant(a) → Add ← Constant(b)
After:   Constant(a + b)
```

The folded result is stored as `ConstantData::Bytes(vec![result])` in the
`ConstantStore`.

**Evaluation semantics:**

Constant folding evaluates in the **byte domain** (mod 256 arithmetic),
consistent with the runtime's execution semantics. There is no float-domain
evaluation at compile time.

- `PrimOp` binary: `Add(5, 10) → 15`, `Sub(5, 10) → 251` (wrapping),
  `Mul(20, 20) → 144` (400 mod 256). This matches `KvStore::apply_binary`
  exactly.
- `LutOp` unary: the operation's 256-byte table is consulted.
  `Relu(constant_byte)` produces `table[constant_byte]`.
- `FusedView`: the fused table is consulted identically.
- Multi-element constants are folded element-wise, producing
  `ConstantData::Bytes(result_bytes)`.

**Invariant:** the result of constant folding MUST be identical to executing
the same operation at runtime. The compiler must not introduce any evaluation
path that differs from the executor.

### View Fusion

Detects chains of unary operations (any combination of `Lut`, `FusedView`, and
unary `Prim` ops) and composes their 256-byte tables into a single
`FusedView(ElementWiseView)`.

```
Before:  x → Relu → Sigmoid → Tanh → y
After:   x → FusedView(composed_table) → y
```

The composition works by backward chain walking:

```rust
let relu_view = LutOp::Relu.to_view();
let sigmoid_view = LutOp::Sigmoid.to_view();
let tanh_view = LutOp::Tanh.to_view();
let fused = relu_view.then(&sigmoid_view).then(&tanh_view);
```

Composition cost: O(256) per pair, one-time. Execution cost: O(1) per byte
regardless of original chain length.

### Common Subexpression Elimination (CSE)

Deduplicates nodes with identical `(op, sorted_predecessors)`. Redirects all
uses of the duplicate to the original node.

```
Before:  Node A: Relu(input_0)
         Node B: Relu(input_0)    ← identical to A
After:   Node A: Relu(input_0)
         (B's consumers now point to A)
```

**Determinism guarantees:**

- **Predecessor sorting:** predecessors are sorted by `NodeId` index
  (ascending). Generation is not considered for sorting — two predecessors
  with the same index but different generation are a bug caught by Parse.
- **Survivor selection:** when two or more nodes share identical
  `(op, sorted_predecessors)`, the node with the **smallest `NodeId` index**
  survives. All others are redirected to it.
- **Walk order:** CSE operates during the topological walk (same pass as
  constant folding and view fusion). The first occurrence in topological
  order is always the survivor.
- **Reproducibility:** given the same input graph, CSE produces identical
  output regardless of platform, thread count, or compiler version. The walk
  order is determined solely by graph topology and `NodeId` indices.

### Fusion Statistics

```rust
pub struct FusionStats {
    pub constants_folded: usize,
    pub views_fused: usize,
    pub cse_eliminated: usize,
}
```

Reported to the user via CLI output or `CompilationStats`.

---

## Stage 3: Plan & Emit

### Liveness Analysis

Determines when each node's output buffer is last consumed. A buffer is "dead"
after its last consumer has executed.

```
Node 0: Input
Node 1: Relu(0)     ← last use of node 0's buffer
Node 2: Sigmoid(1)  ← last use of node 1's buffer
Node 3: Output(2)   ← last use of node 2's buffer
```

### Workspace Layout

Allocates workspace slots to minimize peak memory. Dead buffer slots are reused
by later nodes via first-fit-decreasing bin packing.

```rust
pub struct WorkspaceLayout {
    pub slots: Vec<SlotAssignment>,
    pub peak_slots: usize,
}

pub struct SlotAssignment {
    pub node_id: NodeId,
    pub slot: usize,
}
```

### Scheduling

Builds an `ExecutionSchedule` from the optimized graph:

1. **Topological sort** — linearize the DAG
2. **Level assignment** — group nodes into parallel levels (a node enters a level
   when all its predecessors are in earlier levels)
3. **Critical path** — longest dependency chain (determines minimum sequential depth)

```rust
pub struct ExecutionSchedule {
    pub levels: Vec<ParallelLevel>,
    pub critical_path: usize,
}

pub struct ParallelLevel {
    pub node_ids: Vec<NodeId>,
}
```

**Parallelism ratio** = total nodes / critical path length. Higher ratios indicate
more available parallelism.

### Emit

Invokes `HoloWriter` to produce the final `.holo` archive:

1. Serialize the optimized graph via rkyv
2. Pack constants and weights
3. Write page-aligned sections with CRC checksums
4. Produce the 80-byte header

---

## Compiler Scope

The hologram compiler operates on `hologram::Graph` nodes — it has no concept
of AI-specific operations (`AiOp`, attention, norm, etc.). AI-semantic
optimizations (attention fusion, FFN fusion, quant-matmul fusion) must be
performed **before** lowering, while the computation is still in `AiGraph` form.

The two optimization phases are complementary:

| Phase | IR | Optimizations |
|-------|----|--------------|
| AI passes (hologram-ai) | `AiGraph` | Attention fusion, FFN fusion, quant-matmul fusion |
| hologram-compiler | `hologram::Graph` | LUT chain fusion, CSE, constant folding, buffer reuse |

See ADR-0008 for the rationale.
