# Runtime Model — hologram

## Execution Lifecycle

### 1. Archive Loading

```
.holo file → HoloLoader::open() → mmap (if std) → LoadedPlan
```

- **mmap path**: Zero-copy; pages faulted on demand
- **bytes path**: `HoloLoader::from_bytes()` for embedded/WASM

### 2. Schedule Construction

```
LoadedPlan.graph() → build_schedule() → ExecutionSchedule
```

- Topological sort of graph nodes
- Grouping into parallel levels (nodes with satisfied dependencies)
- Each level can execute in any order (Rayon parallel if enabled)

### 3. Input Preparation

```
GraphInputs::new().set(index, bytes) → inputs
```

- Caller provides byte slices for each input port
- Inputs are indexed by port number (0, 1, 2, ...)

### 4. Execution

```
KvExecutor::execute(graph, schedule, &inputs) → GraphOutputs
```

For each `ParallelLevel` in schedule:
1. (Optional) Spawn Rayon tasks for level parallelism
2. For each node in level:
   - Read input bytes from `BufferArena`
   - Dispatch via `KvStore` (O(1) table lookup)
   - Write output bytes to `BufferArena`
3. Barrier: wait for level completion before next level

### 5. Output Extraction

```
GraphOutputs.get(name) → Option<&[u8]>
```

- Named outputs as byte slices
- Caller interprets bytes according to expected dtype

---

## State Management

### Stateless Execution

`KvExecutor` is stateless. All execution state lives in:

- **`BufferArena`**: Workspace memory (allocated per execution)
- **`GraphInputs`**: Caller-provided input buffers
- **`GraphOutputs`**: Result buffers (owned by executor, returned to caller)

### Shared State

- **`LoadedPlan`**: Immutable, shareable across threads (`Arc<LoadedPlan>`)
- **`ExecutionSchedule`**: Immutable, shareable
- **`CustomOpRegistry`**: Thread-safe, typically static

### Session State (Consumer Responsibility)

For multi-turn inference (e.g., LLM KV-cache):
- `InferenceSession` in `hologram-ai` owns KV buffers and `present_len`
- hologram receives these as `GraphInputs` on each call
- hologram has no concept of "session" or "context"

---

## Error Handling

### Error Types

| Error | When | Recovery |
|-------|------|----------|
| `GraphValidationError` | Invalid graph structure | Fix graph, recompile |
| `ArchiveLoadError` | Corrupt/invalid `.holo` file | Re-download or recompile |
| `ExecutionError` | Runtime dispatch failure | Log and retry or abort |
| `BufferOverflow` | Arena exhaustion | Increase workspace size |

### Panic Policy

- **No panics in execution path**: All errors return `Result`
- **Panics allowed in**: Debug assertions, test code, provably unreachable paths
- **Bounds checks**: Use `get()` not indexing; bounds violations return errors

### Error Propagation

```rust
pub fn execute(...) -> Result<GraphOutputs, ExecutionError>
```

Errors propagate via `Result`. Callers decide recovery strategy.

---

## Resource Limits

### Memory

| Resource | Default | Configuration |
|----------|---------|---------------|
| BufferArena | Computed from liveness | Set via `ExecutorOptions` |
| Archive mmap | Lazy (page-faulted) | OS-managed |
| Output buffers | Proportional to outputs | Automatic |

### Threads

| Context | Threads | Control |
|---------|---------|---------|
| Level parallelism | Rayon pool | `RAYON_NUM_THREADS` env var |
| Async execution | Tokio runtime | Caller-configured |
| Single-threaded | 1 | Disable `parallel` feature |

### Timeouts

No built-in timeouts. Callers implement timeout via:
- `tokio::time::timeout()` for async execution
- External watchdog for sync execution

---

## Thread Safety

### Send + Sync Types

| Type | Send | Sync | Notes |
|------|------|------|-------|
| `LoadedPlan` | ✓ | ✓ | Immutable after load |
| `ExecutionSchedule` | ✓ | ✓ | Immutable |
| `CustomOpRegistry` | ✓ | ✓ | Interior mutability via RwLock |
| `KvStore` | ✓ | ✓ | Immutable dispatch tables |
| `BufferArena` | ✓ | ✗ | Mutable; one per execution |
| `GraphInputs` | ✓ | ✗ | Mutable; one per execution |
| `GraphOutputs` | ✓ | ✗ | Mutable; one per execution |

### External Locking

- **`CustomOpRegistry`**: Internal RwLock; registration is thread-safe
- **`BufferArena`**: No locking; caller ensures single-writer access
- **Parallel levels**: Each node in a level writes to disjoint arena slots; no locking needed

### WASM

- Single-threaded (no `parallel` feature)
- No mmap (use `from_bytes`)
- All types are `Send` (WASM is single-threaded)