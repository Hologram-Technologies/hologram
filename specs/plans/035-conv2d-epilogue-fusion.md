# Plan 035: Conv2d Epilogue Fusion — Accelerate Conv2d + Activation Chain

## Context

This chain appears in Stable Diffusion UNet blocks (ResNet + cross-attention). Currently each op is a separate tape instruction with intermediate buffer materialization between them. The question: what fusions and optimizations can eliminate those intermediates and reduce total latency?

## Current State

| Op | Status | Fusion | Key File |
|----|--------|--------|----------|
| Conv2d | im2col + GEMM (BLAS or LUT-GEMM Q4) | **None** — no Conv2d+Activation epilogue | `float_dispatch/conv.rs` |
| Attention | BLAS sgemm / online softmax, GQA, sparse_v | **None** — standalone dispatch | `float_dispatch/attention.rs` |
| GroupNorm + SiLU | Per-group mean/var + scale/bias | **Already fused** via `FusedGroupNormActivation` | `float_dispatch/norm.rs`, `fusion/float_fusion.rs:255-321` |

The GroupNorm+SiLU fusion already exists (from Sprint 23). The gains come from Conv2d epilogue fusion.

## Optimization Opportunities (ranked by impact)

### 1. Conv2d + Activation Epilogue Fusion — HIGH VALUE

**Problem**: Conv2d writes its output to a buffer, then the next op (often SiLU, Relu, or another activation) reads it back. For a 512x512 image with 320 channels, that's 320x512x512x4 = 335MB of unnecessary memory traffic.

**Solution**: Add `FusedConv2dActivation` graph op + `InlineConv2dActivation` tape kernel. Apply the activation in the GEMM epilogue (in-register, before writeback to memory).

**Pattern**: `Conv2d -> unary Activation` (same 2-node pattern as MatMul+Activation)

**Expected gain**: Eliminates one full buffer materialization per Conv2d block. For UNet with ~23 ResNet blocks, that's 23 x 335MB = 7.7GB of saved memory traffic per inference step.

### 2. Conv2d + Bias + Activation Fusion (3-node) — HIGH VALUE

**Problem**: Many Conv2d layers are followed by a bias add then activation: `Conv2d -> Add(bias) -> SiLU`. This is the Conv2d equivalent of the existing MatMul+Bias+Activation fusion.

**Solution**: Add `FusedConv2dBiasActivation` — same 3-node pattern as `FusedMatMulBiasActivation`.

### 3. Attention Output Projection — MEDIUM VALUE (already optimized)

The existing `can_reuse_input` flag on TapeInstruction handles the Attention->MatMul buffer handoff. Just verify it works.

## Implementation Plan

### Phase 1: Conv2d + Activation Epilogue Fusion

**Files to modify**:
- `crates/hologram-graph/src/graph/mod.rs` — add `FusedConv2dActivation` GraphOp variant
- `crates/hologram-graph/src/fusion/float_fusion.rs` — add `try_fuse_conv2d_activation()` pattern
- `crates/hologram-graph/src/fusion/mod.rs` — wire into fusion pass order
- `crates/hologram-exec/src/tape.rs` — add `InlineConv2dActivation` TapeKernel + dispatch
- `crates/hologram-exec/src/tape_builder.rs` — add kernel resolution
- `crates/hologram-exec/src/float_dispatch/conv.rs` — add fused dispatch
- `crates/hologram-exec/src/kv/store.rs` — exhaustive match coverage

### Phase 2: Conv2d + Bias + Activation (3-node)

Same structure as Phase 1 but with the bias constant input. Follow Plan 031 pattern exactly.

### Phase 3: Tests + Benchmarks

1. Graph fusion test: Conv2d -> SiLU detected and fused
2. No-fuse test: Conv2d with fan-out not fused
3. Correctness test: Fused output bit-identical to unfused
4. Conv2d fusion benchmark

## Reusable Infrastructure

- `try_fuse_matmul_activation()` in `float_fusion.rs:76-98` — template for Conv2d fusion
- `try_fuse_matmul_bias_activation()` in `float_fusion.rs:100-175` — template for 3-node
- `apply_activation_to_out_buf()` in `tape.rs` — activation epilogue helper
- `dispatch_matmul_activation_into()` in `matmul.rs` — pattern for fused dispatch

## Verification

1. `cargo test --workspace` — all existing tests pass
2. `cargo clippy -- -D warnings` — zero warnings
3. New fusion tests in `float_fusion.rs` and `tape.rs`
4. Benchmark: Conv2d latency with/without activation fusion
