# hologram: Execution Model

---

## Overview

Hologram's execution model is stateless O(1) dispatch. Every operation in the
graph is executed as a lookup into a precomputed table or a simple byte-level
operation. The executor processes nodes level-by-level, where nodes within a
level can run concurrently.

---

## KvExecutor

The central execution engine. Stateless — `execute` takes `&self`.

```rust
pub struct KvExecutor;

impl KvExecutor {
    pub fn execute(
        sg: &SerializedGraph,
        schedule: &ExecutionSchedule,
        inputs: &GraphInputs,
    ) -> ExecResult<GraphOutputs>
}
```

### Execution Flow

1. **Seed arena** — populate `BufferArena` with graph-level inputs from `GraphInputs`
2. **For each level** in the `ExecutionSchedule`:
   - Gather each node's input buffers from the arena
   - Dispatch each node's operation via `KvStore`
   - Store each node's output buffer back in the arena
3. **Collect outputs** — extract named outputs from the arena into `GraphOutputs`

### Thread Safety

`KvExecutor::execute` takes `&self` (immutable reference). It is stateless —
there is no per-call mutable state. Multiple sessions can call it concurrently
from different threads with no synchronization.

### With Custom Ops

```rust
pub fn execute_with_registry(
    sg: &SerializedGraph,
    schedule: &ExecutionSchedule,
    inputs: &GraphInputs,
    registry: &CustomOpRegistry,
) -> ExecResult<GraphOutputs>
```

Same flow, but `Custom` ops are dispatched through the provided registry.

---

## KvStore

Static dispatch table that routes `GraphOp` to O(1) kernels.

```rust
pub struct KvStore;

impl KvStore {
    pub fn apply_unary(view: &ElementWiseView, input: &[u8]) -> Vec<u8>
    pub fn apply_binary(op: PrimOp, lhs: &[u8], rhs: &[u8]) -> ExecResult<Vec<u8>>
    pub fn dispatch(
        op: &GraphOp,
        inputs: &[&[u8]],
        registry: Option<&CustomOpRegistry>,
    ) -> ExecResult<Vec<u8>>
}
```

### Dispatch Table

| Operation | Dispatch Method |
|-----------|----------------|
| `Lut(op)` | Convert to `ElementWiseView` via `op.to_view()`, then `apply_unary` |
| `FusedView(view)` | Direct `apply_unary` with the precomputed view |
| `Prim(unary)` | Convert to `ElementWiseView`, then `apply_unary` |
| `Prim(binary)` | Element-wise zip via `apply_binary` (e.g. add bytes mod 256) |
| `Constant(id)` | Resolve from `ConstantStore`, return bytes |
| `MatMulLut4(id)` | Resolve quantized weights, run `lut_gemm_4bit()` |
| `MatMulLut8(id)` | Resolve quantized weights, run `lut_gemm_8bit()` |
| `Custom { id, arity }` | Lookup handler in `CustomOpRegistry`, invoke |
| `Input` | Passthrough from `GraphInputs` |
| `Output` | Passthrough to `GraphOutputs` |

---

## BufferArena

HashMap-based scratch memory keyed by `NodeId`.

```rust
pub struct BufferArena {
    buffers: HashMap<NodeId, Vec<u8>>,
}
```

### Operations

- **`insert(node_id, data)`** — store a node's output after execution
- **`get(node_id) -> &[u8]`** — read a node's output (downstream consumers)
- **`take(node_id) -> Vec<u8>`** — consume a node's output (last consumer, enables reuse)

### Lifecycle

1. **Seeded** at execution start with graph-level inputs
2. **Populated** as each level's nodes produce outputs
3. **Drained** as downstream nodes consume buffers
4. **Dead buffers** reclaimed between levels (not during)

The compiler's workspace planning ensures buffers from dead nodes can be
reused by later nodes, minimizing peak memory.

### Ownership

`KvExecutor::execute()` creates a fresh `BufferArena` internally for each
call. The arena is an implementation detail of the execution call — it is not
a parameter and is not visible to callers.

For consumers that need persistent state across calls (e.g., `hologram-ai`'s
KV-cache), the consumer maintains its own `BufferArena` externally. Cached
buffers are injected into `GraphInputs` before calling `execute()`, and
relevant outputs are extracted from `GraphOutputs` afterward. The consumer's
arena and the executor's internal arena are separate objects — they never
alias.

This supersedes ADR-0007's language about "passed as a mutable reference,"
which referred to the consumer's own arena management, not to `KvExecutor`'s
internal state.

---

## Parallel Execution

When the `parallel` feature is enabled, nodes within a `ParallelLevel` are
executed concurrently using Rayon.

### Thread Safety Contract

**Level barrier:** all nodes in level N must complete before any node in
level N+1 begins. Levels are separated by a Rayon `join` barrier. No
speculative execution across levels.

**Two-phase execution per level:**

1. **Dispatch phase** (parallel): each node reads its inputs from the arena
   (immutable access to predecessor outputs), dispatches through `KvStore`,
   and returns `(NodeId, Vec<u8>)`. The arena is not mutated during this
   phase.
2. **Commit phase** (sequential): all outputs from the dispatch phase are
   inserted into the arena. `take()` for dead buffers also happens here.

This two-phase design ensures the arena is never mutated concurrently. No
locks or atomics are required on the arena itself.

```rust
// Sequential (default):
for node_id in &level.node_ids {
    let output = KvStore::dispatch(&node.op, &gathered_inputs, registry)?;
    arena.insert(*node_id, output);
}

// Parallel (behind `parallel` feature):
let outputs: Vec<(NodeId, Vec<u8>)> = level.node_ids.par_iter()
    .map(|&node_id| {
        let inputs = gather_inputs(&arena, &node.predecessors);
        let output = KvStore::dispatch(&node.op, &inputs, registry)?;
        Ok((node_id, output))
    })
    .collect::<ExecResult<_>>()?;

// Commit phase (sequential):
for (node_id, output) in outputs {
    arena.insert(node_id, output);
}
reclaim_dead_buffers(&mut arena, &level);
```

### Parallelism Metrics

Parallelism ratio = total nodes / critical path. A graph with 100 nodes and
a critical path of 10 has a parallelism ratio of 10×.

---

## Custom Ops

Extension point for domain-specific operations (e.g. AI attention, norm, rope).

### CustomOpRegistry

```rust
pub struct CustomOpRegistry {
    handlers: HashMap<CustomOpId, (u8, Arc<CustomHandler>)>,
}

pub type CustomHandler = dyn Fn(&[&[u8]], &ConstantStore) -> ExecResult<Vec<u8>>;
```

### Registration

```rust
register_op!(registry, id = 42, arity = 2, handler = |inputs, constants| {
    let lhs = inputs[0];
    let rhs = inputs[1];
    // domain-specific computation
    Ok(result)
});
```

### Properties

- Registry is **immutable after construction** — built during compilation, shared across sessions
- Handlers receive raw byte slices and the `ConstantStore` for weight access
- Arity is checked at dispatch time — mismatches produce errors, not panics
- Thread-safe: handlers are `Arc<dyn Fn>`, concurrent dispatch is safe

---

## GraphInputs and GraphOutputs

### GraphInputs

Named byte buffers provided by the caller:

```rust
pub struct GraphInputs {
    buffers: HashMap<String, Vec<u8>>,
}
```

Each entry maps an input name (matching a `Graph::inputs` entry) to its byte data.

### GraphOutputs

Named byte buffers produced by execution:

```rust
pub struct GraphOutputs {
    buffers: Vec<(String, Vec<u8>)>,
}
```

Each entry maps an output name (matching a `Graph::outputs` entry) to its computed result.

---

## Error Handling

Execution errors are reported via `ExecResult<T>`:

| Error | Cause |
|-------|-------|
| `MissingInput(name)` | Required graph input not provided |
| `ArityMismatch { node, expected, got }` | Node predecessor count doesn't match op arity |
| `MissingCustomOp(id)` | Custom op not found in registry |
| `ConstantNotFound(id)` | Constant referenced but not in store |
| `BufferNotFound(node_id)` | Arena doesn't contain expected buffer |
| `CustomOpFailed(id, msg)` | Custom handler returned an error |

All errors are descriptive and include the offending node/operation for debugging.
