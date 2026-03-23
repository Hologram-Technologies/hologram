# Phase 8.3d: WebGPU Deferred Command Encoder Batching

## Problem

The Phase 8.3 WebGPU backend dispatches each kernel synchronously:
create encoder → encode kernel → submit → poll → map staging → readback.
Each submit+poll cycle costs ~10-50µs of overhead. For a transformer layer
with 13 ops above threshold, that's 130-650µs of pure submit overhead.

Metal (Phase 8.2) solved this with `Mutex<Option<CommandBuffer>>` batching —
all kernels in a level encode into one command buffer, then `flush()` commits
once. But Metal has unified memory: the CPU can read GPU buffers directly
after flush. WebGPU requires explicit staging buffer readback (map_async),
so the same pattern doesn't transfer directly.

## Design: `WgpuDeferred` + `flush_deferred()`

### Core Idea

Encode all GPU dispatches within a level into a **single shared
`CommandEncoder`**, defer staging buffer readback, then **submit once** at
the level boundary. `flush_deferred()` returns all readback data in dispatch
order so the executor can store results into the arena.

### New Types

```rust
/// A single deferred GPU dispatch awaiting readback.
struct DeferredEntry {
    staging_buf: wgpu::Buffer,
    byte_len: usize,
}

/// Pending GPU work: shared command encoder + deferred entries.
struct PendingWork {
    encoder: wgpu::CommandEncoder,
    entries: Vec<DeferredEntry>,
    kept_alive: Vec<wgpu::Buffer>,  // input/param buffers
}
```

### WebGpuBackend Changes

```
WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, ComputePipeline>,
+   pending: Mutex<Option<PendingWork>>,
}
```

Each dispatch method (unary, binary, sgemm, softmax, rms_norm) becomes
`*_deferred`: creates GPU buffers, encodes compute pass into the shared
encoder, appends a `DeferredEntry`, and returns immediately.

### KernelOutput + DispatchResult

```rust
enum KernelOutput {
    Skipped,
    Bytes,
    #[cfg(has_metal)]  MetalBuffer(metal::Buffer),
+   #[cfg(has_webgpu)] WgpuDeferred,
}

enum DispatchResult {
    InOutBuf,
    #[cfg(has_metal)]  MetalBuffer(metal::Buffer),
+   #[cfg(has_webgpu)] WgpuDeferred,
}
```

### ComputeBackend Trait

```rust
trait ComputeBackend {
    // ... existing methods ...
+   fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
+       self.flush();
+       Ok(Vec::new())
+   }
}
```

Default implementation calls `flush()` (for Metal) and returns empty.
Only `WebGpuBackend` overrides with actual staging buffer readback.

### flush_deferred() Implementation

```
1. Take PendingWork from Mutex (replace with None)
2. Submit single CommandEncoder → queue.submit([encoder.finish()])
3. Issue map_async on ALL staging buffers at once
4. Single device.poll(Maintain::Wait) satisfies all pending maps
5. Read back each staging buffer in order → Vec<Vec<u8>>
6. Return results in dispatch order
```

Key insight: one submit + one poll for N dispatches, not N submits + N polls.

### Executor Integration

```rust
let mut deferred_slots: Vec<(u32, u8)> = Vec::new();

for level_idx in 0..self.n_levels() {
    for instr in level_instrs {
        match dispatch_result {
            DispatchResult::InOutBuf => arena.swap_insert(...),
            DispatchResult::MetalBuffer(buf) => arena.insert_metal(...),
            DispatchResult::WgpuDeferred => {
                deferred_slots.push((output_idx, elem_size));
            }
        }
    }

    // Level boundary: flush all deferred work
    let deferred_data = backend.flush_deferred()?;
    for (data, &(out_idx, elem_size)) in
        deferred_data.into_iter().zip(deferred_slots.iter())
    {
        arena.insert_with_elem_size(NodeId::new(out_idx, 0), data, elem_size);
    }
    deferred_slots.clear();
}
```

## Expected Performance Impact

| Scenario | Before (per-dispatch submit) | After (batched) |
|----------|------------------------------|-----------------|
| 1 GPU op per level | Same | Same (no batching benefit) |
| 5 GPU ops per level | 5× submit overhead (~50-250µs) | 1× submit (~10-50µs) |
| 13 ops (transformer) | 13× submit (~130-650µs) | 1× submit (~10-50µs) |

Submit overhead reduction: **(N-1)/N** where N = GPU ops per level.
For transformer layers (N≈13): **~92% submit overhead reduction**.

Compute time is unchanged — the GPU still executes the same kernels.
This only reduces CPU↔GPU synchronization overhead.

## Files Changed

- `crates/hologram-exec/src/backend/webgpu.rs` — `PendingWork`, `DeferredEntry`,
  all dispatch methods → `*_deferred`, `flush_deferred_impl()`
- `crates/hologram-exec/src/backend/mod.rs` — `KernelOutput::WgpuDeferred`,
  `flush_deferred()` trait method, `CachedWebGpuBackend` delegation
- `crates/hologram-exec/src/tape.rs` — `DispatchResult::WgpuDeferred`,
  `deferred_slots` tracking, `flush_deferred()` at level boundaries

## Relation to Metal Batching (Phase 8.2)

Metal batching uses `flush()` + unified memory (GPU buffers readable by CPU).
WebGPU batching uses `flush_deferred()` + explicit staging buffer readback.
Both share the same level-boundary flush pattern in the executor. The trait
provides both methods with sensible defaults so backends only override what
they need:

| Backend | `flush()` | `flush_deferred()` |
|---------|-----------|-------------------|
| CPU     | no-op     | no-op → `[]`      |
| Metal   | commits CB| calls `flush()` → `[]` |
| WebGPU  | no-op     | submit + readback → `[Vec<u8>; N]` |
