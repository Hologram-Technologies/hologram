# Plan: Metal Compute Shader Kernels

## Context

The `ComputeBackend` trait and `MetalBackend` stub are in place (Sprint 16 Phase 1). The stub returns `Ok(false)` for all ops, falling back to CPU. This plan covers implementing actual Metal compute shader kernels for the highest-impact ops.

Metal is auto-detected on macOS via `build.rs` (`has_metal` cfg). Apple Silicon M-series chips have unified memory (shared between CPU and GPU), which eliminates the upload/download overhead that makes GPU dispatch expensive on discrete GPUs.

## Priority Ops (by inference impact)

1. **MatMul** (SGEMM) — dominates transformer inference time (60-80% of total). MPSMatrixMultiplication or custom tiled shader.
2. **Elementwise unary** (Relu, Sigmoid, Silu, Gelu, Tanh) — high parallelism, trivial shader.
3. **Elementwise binary** (Add, Mul, Sub) — residual connections, FFN gates.
4. **Softmax** — attention score normalization. Requires parallel reduction.
5. **RmsNorm** — every transformer layer. Requires parallel sum-of-squares.

## Architecture

### Buffer Management
- Apple Silicon has unified memory — CPU and GPU share the same physical RAM
- `MTLBuffer` with `storageModeShared` avoids copies entirely
- The arena's `Cow::Owned(Vec<u8>)` can be replaced with `MTLBuffer`-backed storage
- Zero-copy path: arena allocates from a Metal buffer pool

### Kernel Pipeline
1. At tape build time: compile `.metal` shaders into `MTLComputePipelineState`
2. At execute time: encode commands into `MTLCommandBuffer`, commit, wait
3. Level-based batching: all ops in a parallel level → single command buffer

### Dependencies
- `metal` crate (Rust bindings for Metal framework): `metal = "0.29"`
- `objc2` for Objective-C runtime interop
- Metal Shading Language (.metal files) compiled at build time

## Files to Create

| File | Purpose |
|------|---------|
| `crates/hologram-exec/src/backend/metal.rs` | Replace stub with real MetalBackend |
| `crates/hologram-exec/src/backend/metal_shaders/` | .metal shader source files |
| `crates/hologram-exec/src/backend/metal_shaders/elementwise.metal` | Relu, Sigmoid, etc. |
| `crates/hologram-exec/src/backend/metal_shaders/matmul.metal` | Tiled SGEMM |
| `crates/hologram-exec/src/backend/metal_shaders/reduce.metal` | Softmax, RmsNorm |

## Incremental Delivery

1. **Sprint 17a**: Metal device init + elementwise unary (simplest, validates pipeline)
2. **Sprint 17b**: Metal matmul (biggest impact, can use MPS as baseline)
3. **Sprint 17c**: Metal softmax + RmsNorm (parallel reductions)
4. **Sprint 17d**: Metal buffer pool (replace arena Vec with MTLBuffer-backed storage)

## hologram-ai Integration

hologram-ai should use `execute_tape` with `BackendSelector::Auto` to automatically get Metal on macOS:

```rust
let tape = hologram::build_tape_from_plan(&plan)?;
let result = hologram::execute_tape(&tape, &plan, &inputs)?;
// ↑ Automatically uses Metal on macOS, CPU elsewhere
```

No hologram-ai code changes needed — the backend selection happens inside hologram.
