# Runtime Model — hologram

## Execution Lifecycle

1. **Load**: A `.holo` archive is loaded via `HoloLoader::open()` (mmap) or `load_from_bytes()`. This returns a `LoadedPlan` containing the deserialized graph, execution schedule, weights, and section table.

2. **Initialize**: Create a `BufferArena` to hold intermediate results and a `KvExecutor` instance. Optionally create a `CustomOpRegistry` and register handlers for custom ops.

3. **Inject inputs**: Populate `GraphInputs` with byte buffers for each `Input` node.

4. **Execute**: Call `KvExecutor::execute_with_registry()`. The executor iterates through the `ExecutionSchedule` level by level:
   - Nodes within a level have all dependencies satisfied
   - Level execution is parallel when the `parallel` feature is enabled (via rayon)
   - Each node is dispatched via `KvStore::dispatch_with_constants()`
   - Results are stored in the `BufferArena`

5. **Extract outputs**: Retrieve output buffers from `GraphOutputs` by node ID.

6. **Cleanup**: The `BufferArena` can be cleared for reuse, or dropped.

---

## State Management

| State | Owner | Lifetime |
|-------|-------|----------|
| `LoadedPlan` | Caller | Typically `Arc`-shared for multi-session use |
| `BufferArena` | `KvExecutor` (or caller) | Per-execution; cleared between runs |
| `CustomOpRegistry` | Caller | Shared across executions; immutable after setup |
| `ConstantStore` | `LoadedPlan` | Lifetime of the loaded archive |

No global state. All state is explicitly owned and passed.

---

## Error Handling

Errors propagate via `Result<T, ExecError>`. Error variants include:

| Error | Meaning |
|-------|---------|
| `BufferNotReady(NodeId)` | Input buffer missing for a node |
| `LengthMismatch { expected, actual }` | Binary op operands have different lengths |
| `ShapeMismatch { expected, actual }` | Shape/alignment error (e.g., f32 cast) |
| `ConstantNotFound(u32)` | Referenced constant ID not in store |
| `UnsupportedOp(String)` | Op cannot be executed (e.g., CallSubgraph) |
| `InvalidQuantization(String)` | Quantized weight deserialization failed |

**Panic policy**: Panics only on invariant violations (bugs). All runtime errors are returned via `Result`.

---

## Resource Limits

| Resource | Default | Configuration |
|----------|---------|---------------|
| Thread pool | System cores | Rayon global pool; configure via `RAYON_NUM_THREADS` |
| Memory | Unbounded | Caller manages arena sizing |
| Execution timeout | None | Caller responsibility |

---

## Thread Safety

| Type | Send | Sync | Notes |
|------|------|------|-------|
| `ElementWiseView` | ✓ | ✓ | Immutable 256-byte table |
| `Graph` | ✓ | ✓ | Immutable after construction |
| `LoadedPlan` | ✓ | ✓ | Immutable; safe to share |
| `BufferArena` | ✓ | ✗ | Mutable; one owner per execution |
| `KvExecutor` | ✓ | ✗ | Stateful; one owner per execution |
| `CustomOpRegistry` | ✓ | ✓ | Handlers are `Arc<dyn Fn>` |
| `KvStore` | ✓ | ✓ | Zero-sized; all methods static |