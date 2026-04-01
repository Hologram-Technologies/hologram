# Plan 036: CPU Optimization Sweep — Fusion, Prefetch, Parallelism

## Context

After KV cache quantization (Sprint 28) and Conv2d epilogue fusion (Sprint 29),
several CPU-side optimization opportunities remain. These are all platform-agnostic
(work on wasm, native, Apple Silicon alike) and focus on eliminating unnecessary
work: redundant memory passes, missed fusion patterns, underutilized parallelism,
and insufficient prefetch distance.

## Optimizations

### Phase 1: Fusion Gaps (graph compile-time, zero runtime cost)

#### 1.1 AddRmsNorm + Activation
`AddRmsNorm` (residual + normalize) has no fused activation variant, unlike
`RmsNorm`/`LayerNorm`/`GroupNorm` which all have `Fused*Activation` ops.
This pattern appears in every LLM transformer block.

- Add `FusedAddRmsNormActivation { size, epsilon, activation }` to GraphOp
- Extend `try_fuse_norm_activation()` to handle `FloatOp::AddRmsNorm`
- Add `InlineAddRmsNormActivation` TapeKernel + dispatch
- **Files**: graph/mod.rs, float_fusion.rs, tape.rs, tape_builder.rs
- **Expected**: 3-5% latency on LLM inference

#### 1.2 Attention + Residual Add
`Attention -> Add(residual)` is the most common 2-node pattern in transformers.
Fusing eliminates materializing the full attention output buffer.

- Add `FusedAttentionResidualAdd` GraphOp with attention params
- Fusion pattern: Attention with single successor that is Add, where the Add's
  other input is not a descendant of the Attention (the residual stream)
- Dispatch: run attention, then add residual in-place before writeback
- **Files**: graph/mod.rs, float_fusion.rs, tape.rs, tape_builder.rs, attention.rs
- **Expected**: 5-8% latency on transformer blocks

#### 1.3 Transpose Elimination
Adjacent transposes that cancel (involutions) waste memory bandwidth.
Also: Transpose before Attention with matching `heads_first` is redundant.

- Add `try_eliminate_identity_transpose()` pass: detect Transpose(perm) followed
  by Transpose(inverse_perm) and replace with Passthrough
- Detect Transpose([1,0,2]) before Attention(heads_first=true) and absorb
- **Files**: float_fusion.rs or new transpose_elim.rs, fusion/mod.rs
- **Expected**: 2-4% latency (model-dependent)

### Phase 2: Multi-Level Weight Prefetch (runtime, low effort)

Current: prefetch next level's weights via `madvise(MADV_WILLNEED)`.
For large models (7B+ params), one level ahead isn't enough — the OS needs
time to page in 4-8MB of weights before they're needed.

- Change prefetch loop in `execute_inner()` to issue MADV_WILLNEED for
  levels i+1 AND i+2 (2-level lookahead)
- Add MADV_DONTNEED for level i-2 weights (release pages back to OS sooner)
- **Files**: tape.rs (execute_inner, ~lines 3569-3580)
- **Expected**: 5-15% latency on large models, ~20 lines of changes

### Phase 3: Lock-Free LUT-GEMM Parallelism (runtime, medium effort)

Rayon parallelism is blocked for any level containing LUT-GEMM ops because
the `WeightCache` uses `RefCell` (not thread-safe). Since 50%+ of LLM levels
contain quantized matmul, these levels run single-threaded.

- Replace `RefCell<WeightCache>` with `parking_lot::RwLock<WeightCache>` or
  a lock-free concurrent hashmap
- Allow `execute_level_parallel()` to include LUT-GEMM ops
- Add per-thread Psumbook scratch to avoid contention
- **Files**: tape.rs (TapeContext, execute_inner), kv/weight_cache.rs, parallel/mod.rs
- **Expected**: 1.5-2.5x on multi-core systems for LLM inference
- **Risk**: Medium — needs careful testing for correctness under contention

### Phase 4: Additional Fusion Opportunities (lower priority)

#### 4.1 SwiGLU Fusion from Separate Ops
`FusedSwiGLU` exists as a primitive but isn't recognized from the
`Split -> Silu -> Mul` pattern by the fusion pass. Only works if the model
importer explicitly creates it.

- Add `try_fuse_swiglu()`: detect gate/up split followed by Silu(gate) * up
- **Files**: float_fusion.rs
- **Expected**: 3-5% on GLU-family models (LLaMA, Mistral)

#### 4.2 Adaptive sparse_v Threshold
Currently hardcoded at 1e-6. At long contexts (32K+), more positions have
near-zero attention weight, so a higher threshold could skip more V accumulation
without quality loss.

- Make threshold configurable per KvCacheConfig or per Attention op
- Add diagnostic mode that counts skipped positions per-layer
- **Files**: attention.rs, float_op.rs
- **Expected**: 5-20% decode speedup at long contexts

#### 4.3 Activation Checkpointing Validation
The recompute path is wired in `execute_inner()` but the `checkpoint_map` may
not be populated by the compiler. Need to verify the compiler fills it for
SD UNet skip connections.

- Check `tape_builder.rs` checkpoint_map population
- If empty, wire the identification pass from commit 2a5828e into the builder
- **Files**: tape_builder.rs, tape.rs
- **Expected**: O(layer) peak activation memory for SD UNet (huge for 512x512+)

### Phase 5: Wire F16 Activation Compression (memory, medium effort)

F16 compression infrastructure exists in `ArenaBuffer::F16Compressed` (arena.rs)
with `compress_f16()` / `expand_f32()` but is not wired into the execution loop.
This halves activation memory for buffers > 512KB.

- Wire `compress_f16()` into `swap_insert_with_elem_size()` for buffers that won't
  be read for multiple levels (identified by liveness analysis gap > N instructions)
- Auto-expand on read via `get_f32_or_expand()`
- **Files**: buffer/arena.rs, tape.rs (eviction/insert logic)
- **Expected**: ~50% activation memory reduction for large buffers (vision models)
- **Note**: F16 is platform-agnostic (software conversion, no GPU needed)

### Phase 6: Workspace Buffer Reuse (memory, low effort)

The compiler has a `plan_workspace()` (workspace/mod.rs) that assigns buffer
slots based on liveness intervals, allowing non-overlapping tensors to share
memory. Verify this is wired into the tape executor's arena allocation.

- Check if `WorkspaceLayout` assignments drive `BufferArena` slot reuse
- If not wired, map workspace slots to arena pre-allocation at tape start
- **Files**: compiler/workspace/mod.rs, tape.rs (arena init)
- **Expected**: 20-40% peak memory reduction for deep models

### Phase 7: WebGPU Kernel Parity for Wasm (runtime, high effort)

`WebGpuBackend` exists with deferred batching but only dispatches elementwise +
matmul. Since wasm can't use Metal/CUDA, WebGPU is the only GPU acceleration
path for browser deployment.

- Add WebGPU compute shaders for: Conv2d, Softmax, RmsNorm, GroupNorm, Attention
- These are already dispatched to Metal on macOS — port the kernel logic to WGSL
- **Files**: backend/webgpu.rs, new WGSL shader files
- **Expected**: 5-20x speedup for large ops in browser (model-dependent)
- **Note**: This is the long pole for wasm performance

### Phase 8: InstanceNorm + Activation Fusion (low effort)

InstanceNorm exists as a FloatOp but has no fused activation variant, unlike
GroupNorm/LayerNorm/RmsNorm which all have `Fused*Activation` ops. Common in
style transfer and image generation models.

- Add `FusedInstanceNormActivation` to GraphOp
- Extend `try_fuse_norm_activation()` to handle InstanceNorm
- **Files**: graph/mod.rs, float_fusion.rs, tape.rs, tape_builder.rs
- **Expected**: Small but free (same pattern as existing norm fusions)

## Priority Order

| # | Optimization | Effort | Impact | Platform |
|---|-------------|--------|--------|----------|
| 1 | AddRmsNorm+Activation fusion | Low | 3-5% latency | All |
| 2 | Multi-level weight prefetch | Low | 5-15% latency | Native (no-op on wasm) |
| 3 | Attention+Residual fusion | Medium | 5-8% latency | All |
| 4 | Transpose elimination | Low | 2-4% latency | All |
| 5 | Lock-free LUT-GEMM parallel | Medium | 1.5-2.5x multi-core | Native (wasm is single-threaded) |
| 6 | SwiGLU fusion | Medium | 3-5% latency | All |
| 7 | Adaptive sparse_v | Low | 5-20% decode | All |
| 8 | Activation checkpoint validation | Low | Memory | All |
| 9 | F16 activation compression | Medium | 50% activation mem | All |
| 10 | Workspace buffer reuse | Low | 20-40% peak mem | All |
| 11 | WebGPU kernel parity | High | 5-20x (wasm GPU) | Wasm |
| 12 | InstanceNorm+Activation fusion | Low | Small | All |

Items 1-4 and 6-8, 12 are independent and can be parallelized.
Item 5 has highest multi-core impact but needs architectural care.
Item 9-10 are memory optimizations (latency unchanged, capacity increased).
Item 11 is the long-term wasm performance investment.

## Verification

- `cargo test --workspace` after each phase
- `cargo clippy -- -D warnings`
- `cargo bench --bench fusion` (fusion pass timing)
- `cargo bench --bench epilogue_fusion` (fused kernel correctness)
- `cargo bench --bench executor` (tape execution timing)
- `cargo bench --bench matmul` (LUT-GEMM parallel validation)
- `cargo bench --bench kv_cache` (KV cache regression check)
- Peak memory profiling for phases 5, 6, 8
