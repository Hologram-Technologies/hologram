# Plan: Phase 8.3 — WebGPU/wgpu Compute Shader Path

## Context

The Metal backend provides GPU acceleration on Apple Silicon (17.5x tape speedup, Phase 8.2 batching). Phase 8.3 adds cross-platform GPU via wgpu — works on Vulkan (Linux/Windows), DX12 (Windows), Metal (macOS via wgpu), and browser WebGPU. Feature-gated behind `webgpu` to avoid pulling in the large wgpu dependency tree by default.

## Approach

Mirror the Metal backend architecture (pipeline caching, size thresholds, `flush()` batching) but using WGSL shaders and wgpu API. Returns `KernelOutput::Bytes` (copies result to `out_buf` after GPU readback) rather than a GPU buffer variant — avoids touching `KernelOutput` enum or tape executor.

Split into 4 sub-phases, each independently testable.

## cfg / Feature Gating (CRITICAL)

All WebGPU code MUST be gated behind `#[cfg(has_webgpu)]` or the `webgpu` Cargo feature:

- **webgpu.rs** — entire module gated by `#[cfg(has_webgpu)] pub mod webgpu;` in mod.rs (already exists)
- **wgpu/pollster imports** — only in webgpu.rs (behind module-level cfg)
- **CachedWebGpuBackend** — `#[cfg(has_webgpu)]` on struct + impl
- **BackendSelector::WebGpu** — `#[cfg(has_webgpu)]` on match arm (already exists)
- **Tests** — all WebGPU tests gated with `#[cfg(has_webgpu)] #[test]`
- **build.rs** — emits `has_webgpu` cfg only when `CARGO_FEATURE_WEBGPU` env is set OR `wasm32` target
- **Cargo.toml** — `wgpu` and `pollster` are optional deps, only pulled in by `features = ["webgpu"]`
- **Default compilation** (`cargo test/check` without `--features webgpu`) must NOT pull in wgpu or fail

This matches the Metal pattern: `#[cfg(has_metal)]` gates all Metal code, and `metal` is a target-specific dep.

## Files to Modify

- `Cargo.toml` (workspace) — add `wgpu = "23"`, `pollster = "0.4"` to workspace deps
- `crates/hologram-exec/Cargo.toml` — add optional `wgpu`/`pollster` deps behind `webgpu` feature
- `crates/hologram-exec/build.rs` — expand `has_webgpu` from wasm32-only to feature-gated
- `crates/hologram-exec/src/backend/webgpu.rs` — full implementation (replace stub)
- `crates/hologram-exec/src/backend/mod.rs` — `CachedWebGpuBackend`, OnceLock, resolve updates
- `specs/SPRINT.md` — tick 8.3

---

## Phase 8.3a: Bootstrap (prove the pattern)

### Cargo.toml (workspace root)
```toml
[workspace.dependencies]
wgpu = "23"
pollster = "0.4"
```

### crates/hologram-exec/Cargo.toml
```toml
[dependencies]
wgpu = { workspace = true, optional = true }
pollster = { workspace = true, optional = true }

[features]
webgpu = ["wgpu", "pollster"]
```

### build.rs
Replace wasm32-only detection:
```rust
if std::env::var("CARGO_FEATURE_WEBGPU").is_ok()
    || std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32"
{
    println!("cargo:rustc-cfg=has_webgpu");
}
```

### webgpu.rs — WebGpuBackend struct
```rust
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
}
```
- `new()` → `pollster::block_on(new_async())` for sync init
- `new_async()`: request adapter (HighPerformance) → request device → compile WGSL modules → cache pipelines
- 5 WGSL shader modules: unary, binary, sgemm, softmax, rms_norm (separate because different bind group layouts)

### webgpu.rs — dispatch_unary pattern
1. Create input buffer via `device.create_buffer_init(contents: input_bytes)`
2. Create output buffer via `device.create_buffer(size, STORAGE | COPY_SRC)`
3. Create uniform buffer for count parameter
4. Create bind group from pipeline's auto layout
5. Create staging buffer (MAP_READ | COPY_DST)
6. Encode: begin_compute_pass → set_pipeline → set_bind_group → dispatch_workgroups → end
7. Encode: copy_buffer_to_buffer (output → staging)
8. `queue.submit(encoder.finish())`
9. Map staging → `device.poll(Wait)` → copy to `out_buf`

### mod.rs
- `CachedWebGpuBackend(Arc<WebGpuBackend>)` — delegates all methods
- OnceLock caching in `BackendSelector::WebGpu` and `default_backend()`
- Priority preserved: Metal > WebGPU > CUDA > CPU

### Deliverable
- Single `relu` kernel in WGSL
- End-to-end test: `webgpu_dispatch_relu` (1.5M floats, spot-check correctness)

---

## Phase 8.3b: Complete Elementwise

### WGSL Unary Module (9 kernels)
All share bindings: `input: array<f32>`, `output: array<f32>`, `params: Params { count: u32 }`
- relu, neg, abs_val, sigmoid, silu, tanh_act, exp_act, reciprocal, gelu

### WGSL Binary Module (4 kernels)
Bindings: `a: array<f32>`, `b: array<f32>`, `output: array<f32>`, `params: BinaryParams { count_a, count_b }`
- add_op, mul_op, sub_op, div_op (with modulo broadcasting)

### dispatch_binary method
Same pattern as dispatch_unary but 5 bindings instead of 3.

### kernel_name mapping
`Self::kernel_name(op)` maps `FloatOp` → WGSL entry point name (same names as Metal).

---

## Phase 8.3c: Custom Ops

### WGSL SGEMM Module (tiled 16×16)
Bindings: A, B, C (storage), params: GemmParams { M, K, N } (uniform)
- `var<workgroup> tileA/tileB` for shared memory
- `workgroupBarrier()` for sync
- Workgroup dispatch: `((N+15)/16, (M+15)/16, 1)`

### WGSL Softmax Module
Bindings: input, output (storage), params: { total, row_size } (uniform)
- Per-element: find row max, compute exp(x-max), compute row sum, divide

### WGSL RmsNorm Module
Bindings: input, weight, output (storage), params: { total, row_size, epsilon } (uniform)
- Per-element: compute mean-of-squares for row, inverseSqrt, multiply by weight

### dispatch_sgemm, dispatch_softmax, dispatch_rms_norm methods
Same GPU readback pattern. Size thresholds: 4MB for softmax/rmsnorm, 128×128 for matmul.

### Tests
Mirror Metal tests: `webgpu_dispatch_matmul`, `webgpu_dispatch_softmax`

---

## Phase 8.3d: Command Encoder Batching

Encode multiple compute passes into a single `CommandEncoder`, defer `queue.submit()` to `flush()`, batch all staging readbacks. This mirrors Metal's Phase 8.2 pattern and saves ~10-50µs of submit overhead per GPU dispatch.

### What batching means

**Without batching** (current 8.3a-c):
```
Instr 1: create encoder → encode relu → submit → wait → readback
Instr 2: create encoder → encode add  → submit → wait → readback
Instr 3: create encoder → encode mul  → submit → wait → readback
```
Each `queue.submit()` has ~10-50µs overhead (GPU scheduling + fence sync).

**With batching**:
```
Instr 1: encode relu into shared CommandEncoder
Instr 2: encode add  into shared CommandEncoder
Instr 3: encode mul  into shared CommandEncoder
--- level boundary ---
flush(): submit once → wait once → map all staging buffers → readback all
```
One submit instead of three. GPU processes kernels back-to-back.

### Why Metal batching is simpler

Metal returns `MetalBuffer` — stored directly in the arena via `insert_metal`. On Apple Silicon unified memory, the CPU can read `buf.contents()` after `flush()`. No explicit readback needed.

WebGPU buffers require explicit `map_async` + `device.poll` + `get_mapped_range` to read on CPU. So we can't just store a wgpu buffer in the arena — we need a deferred readback mechanism.

### Key insight: level independence

Instructions within a level are **independent** — they never read each other's outputs. They only read outputs from previous levels (which were flushed). So we can safely defer ALL GPU results within a level and populate them in bulk at flush time.

### Design: `KernelOutput::WgpuDeferred` + `flush_deferred()`

#### New types

```rust
// webgpu.rs
struct DeferredEntry {
    staging_buf: wgpu::Buffer,
    byte_len: usize,
}

struct PendingWork {
    encoder: wgpu::CommandEncoder,
    entries: Vec<DeferredEntry>,
    kept_alive: Vec<wgpu::Buffer>,  // input/output/param buffers referenced by encoder
}
```

#### New KernelOutput variant (cfg-gated)

```rust
// mod.rs
pub enum KernelOutput {
    Skipped,
    Bytes,
    #[cfg(has_metal)]   MetalBuffer(::metal::Buffer),
    #[cfg(has_webgpu)]  WgpuDeferred,  // result available after flush_deferred()
}
```

#### New DispatchResult variant (cfg-gated)

```rust
// tape.rs
enum DispatchResult {
    InOutBuf,
    #[cfg(has_metal)]   MetalBuffer(metal::Buffer),
    #[cfg(has_webgpu)]  WgpuDeferred,  // skip swap_insert; flush will populate arena
}
```

#### New ComputeBackend method

```rust
// mod.rs
pub trait ComputeBackend: Send + Sync {
    // ... existing ...
    fn flush(&self) {}

    /// Flush deferred GPU work. Returns readback data in dispatch order.
    /// Called at level boundaries. Default: calls flush(), returns empty.
    fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
        self.flush();
        Ok(Vec::new())
    }
}
```

Metal and CPU get the default (flush + empty vec). Only WebGPU overrides.

#### Refactored dispatch_* methods (webgpu.rs)

Each dispatch method changes from synchronous to deferred:
1. Get or create shared encoder from `self.pending`
2. Create input/output/params/staging buffers
3. Encode compute pass + `copy_buffer_to_buffer(output → staging)`
4. Push `DeferredEntry { staging_buf, byte_len }` to pending entries
5. Push input/output/params buffers to `kept_alive` (prevent drop before GPU executes)
6. Return — no submit, no poll, no readback

#### `WebGpuBackend::flush_deferred()` implementation

```rust
fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
    let work = match self.pending.lock().unwrap().take() {
        Some(w) => w,
        None => return Ok(Vec::new()),
    };

    // Single submit for ALL dispatches in this level.
    self.queue.submit(std::iter::once(work.encoder.finish()));
    self.device.poll(wgpu::Maintain::Wait);

    // Issue all map_async calls at once.
    let channels: Vec<_> = work.entries.iter().map(|entry| {
        let slice = entry.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        rx
    }).collect();

    // Single poll satisfies all pending maps.
    self.device.poll(wgpu::Maintain::Wait);

    // Read back all staging buffers.
    let mut results = Vec::with_capacity(work.entries.len());
    for (entry, rx) in work.entries.iter().zip(channels) {
        rx.recv().map_err(|_| ExecError::UnsupportedOp("wgpu channel".into()))?
          .map_err(|e| ExecError::UnsupportedOp(format!("wgpu map: {e:?}")))?;
        let data = entry.staging_buf.slice(..).get_mapped_range();
        let mut buf = Vec::with_capacity(entry.byte_len);
        buf.extend_from_slice(&data[..entry.byte_len]);
        results.push(buf);
        drop(data);
        entry.staging_buf.unmap();
    }

    Ok(results)
}
```

#### Execute loop changes (tape.rs)

In both `execute()` and `execute_parallel()`, at level boundary:

```rust
// Before (current):
backend.flush();

// After:
let deferred_data = backend.flush_deferred()?;
for (data, &(out_idx, elem_size)) in deferred_data.into_iter().zip(&deferred_slots) {
    arena.insert_with_elem_size(NodeId::new(out_idx, 0), data, elem_size as usize);
}
deferred_slots.clear();
```

In the dispatch result handling, add:

```rust
#[cfg(has_webgpu)]
DispatchResult::WgpuDeferred => {
    deferred_slots.push((instr.output_idx, instr.output_elem_size));
    // Don't swap_insert — data comes from flush_deferred
}
```

`deferred_slots: Vec<(u32, u8)>` allocated once before the level loop.

### Implementation sequence

1. Add `PendingWork` struct + `pending: Mutex<Option<PendingWork>>` to WebGpuBackend
2. Add `KernelOutput::WgpuDeferred` + `DispatchResult::WgpuDeferred` (cfg-gated)
3. Add `flush_deferred()` to ComputeBackend trait (default: flush + empty vec)
4. Refactor WebGPU dispatch methods to encode-only (no submit)
5. Implement `WebGpuBackend::flush_deferred()` (batch submit + batch readback)
6. Update execute loop: deferred_slots + flush_deferred at level boundaries
7. Update CachedWebGpuBackend to delegate flush_deferred
8. Remove `submit_and_readback` (dead code)

### Files to modify
- `crates/hologram-exec/src/backend/webgpu.rs` — PendingWork, refactored dispatches, flush_deferred
- `crates/hologram-exec/src/backend/mod.rs` — WgpuDeferred variant, flush_deferred trait method, CachedWebGpuBackend
- `crates/hologram-exec/src/tape.rs` — WgpuDeferred dispatch result, deferred_slots, flush_deferred call

### Buffer lifetime safety
All wgpu buffers (input, output, params) must live until after `flush_deferred()` reads from staging. `PendingWork.kept_alive: Vec<wgpu::Buffer>` holds references. wgpu buffers are internally `Arc`-counted, so the encoder's command buffer also retains them — but explicit `kept_alive` ensures correctness regardless of wgpu internals.

---

## Key Design Decisions

1. **Returns `KernelOutput::Bytes`** (not a new GPU buffer variant) — copies result to `out_buf` via staging buffer. Avoids touching the tape executor. Trade-off: one extra copy vs zero changes to the dispatch/arena path.

2. **Separate WGSL modules per kernel category** — WGSL requires consistent `@group/@binding` layouts within a module. Unary (3 bindings), binary (5), sgemm (4), softmax (3), rmsnorm (4) each get their own module.

3. **`pollster::block_on`** for sync device init — wgpu's `request_adapter`/`request_device` are async. `pollster` blocks the current thread. Document that `WebGpuBackend::new()` must not be called from within a tokio runtime.

4. **Feature-gated** behind `webgpu` — not in default features. Users opt-in with `--features webgpu`.

5. **Backend priority unchanged** — Metal (direct) takes precedence over WebGPU (wgpu) on macOS. Both can coexist.

---

## Verification

1. `cargo test --workspace --features webgpu` — all tests pass including new webgpu tests
2. `cargo clippy --workspace --features webgpu -- -D warnings`
3. `cargo test --workspace` (without webgpu feature) — existing tests still pass, wgpu NOT compiled
4. `cargo check --workspace` (without webgpu feature) — no compile errors, no wgpu dependency pulled
5. Conformance: WebGPU relu output matches CPU relu output within f32 tolerance
6. `cargo bench --features webgpu` — verify GPU dispatch fires for large buffers
