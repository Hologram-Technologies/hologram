# Plan 031: Bias Fusion — MatMul+Bias+Activation

## Context

Plan 030 added epilogue fusion (MatMul+Activation, Norm+Activation). Benchmarks showed that for **MatMul+Activation alone**, the unfused path matches or beats fused because hologram's `can_reuse_input` optimization already makes the activation step nearly free (in-place overwrite, zero allocation).

The real win is **bias fusion**: `y = activation(xW + b)`. Currently this is 3 tape instructions (MatMul → Add(bias) → Activation) with 2 intermediate buffers. The `can_reuse_input` trick can only eliminate ONE intermediate — the bias Add still needs its own buffer. Fusing all three into a single instruction eliminates both intermediates and runs bias+activation as a single loop on cache-hot matmul output.

This pattern appears in every Linear+Activation layer in every transformer. For a 7B model: ~64 linear layers per transformer block × 32 blocks = ~2048 MatMul+Bias+Activation sequences.

**Branch**: `refactor/bias-fusion`

---

## What exists (from Plan 030)

- `GraphOp::FusedMatMulActivation { m, k, n, activation }` — matmul + activation (no bias)
- `TapeKernel::InlineMatMulActivation { m, k, n, activation }` — tape dispatch
- `try_fuse_matmul_activation()` — fusion pass (2-node: MatMul → Activation)
- `apply_activation_to_out_buf()` — helper for activation post-pass

## Implementation

### Phase 1: Graph Op + TapeKernel

**1.1** Add `GraphOp::FusedMatMulBiasActivation`

File: [graph/mod.rs](crates/hologram-graph/src/graph/mod.rs)

```rust
/// Fused matmul + bias add + activation (full epilogue fusion).
/// Three original inputs become two: [activation_input, weight_constant].
/// Bias is a constant resolved at dispatch time.
FusedMatMulBiasActivation {
    m: u32,
    k: u32,
    n: u32,
    bias_cid: ConstantId,
    activation: FloatOp,
},
```

Arity: 2 (same as MatMul — bias is a constant, not an input edge).

**1.2** Add `TapeKernel::InlineMatMulBiasActivation`

File: [tape.rs](crates/hologram-exec/src/tape.rs)

```rust
InlineMatMulBiasActivation {
    m: u32,
    k: u32,
    n: u32,
    bias_cid: ConstantId,
    activation: FloatOp,
},
```

### Phase 2: Fused Kernel

**2.1** Add `dispatch_matmul_bias_activation_into`

File: [matmul.rs](crates/hologram-exec/src/float_dispatch/matmul.rs)

```rust
pub fn dispatch_matmul_bias_activation_into(
    inputs: &[&[u8]],
    m: usize, k: usize, n: usize,
    bias: &[f32],       // pre-resolved bias vector [N]
    activation: &FloatOp,
    out_buf: &mut Vec<u8>,
) -> ExecResult<()> {
    // Standard matmul.
    dispatch_matmul_into(inputs, m, k, n, out_buf)?;
    // Fused bias + activation in one pass (cache-hot).
    if let Ok(floats) = bytemuck::try_cast_slice_mut::<u8, f32>(out_buf) {
        for row in floats.chunks_mut(n) {
            for (j, v) in row.iter_mut().enumerate() {
                *v = activation.apply_unary(*v + bias[j % bias.len()]);
            }
        }
    }
    Ok(())
}
```

Single loop over output: add bias[col] + apply activation per element. No intermediate buffer.

### Phase 3: Fusion Pass (3-node pattern)

**3.1** Add `try_fuse_matmul_bias_activation`

File: [float_fusion.rs](crates/hologram-graph/src/fusion/float_fusion.rs)

Pattern: `MatMul → Add → Activation` where:
1. MatMul has exactly one successor (the Add)
2. Add's second input is a Constant (the bias vector)
3. Add has exactly one successor (the Activation)
4. Activation is elementwise unary

Replace all three nodes with `FusedMatMulBiasActivation`. The fused op takes MatMul's inputs (activation tensor, weight constant) and carries the bias constant ID + activation type.

**3.2** Also detect: `MatMul → Add` (no activation) — this is also worth fusing as `FusedMatMulBias` to eliminate the intermediate buffer.

**3.3** Wire into `fuse()` — run BEFORE `try_fuse_matmul_activation` (3-node pattern is more valuable than 2-node).

### Phase 4: Tape Dispatch

**4.1** Dispatch arm in `dispatch_kernel`

Resolve bias from constants (via `tape_ctx.constants` + `tape_ctx.weights`), cast to f32, call `dispatch_matmul_bias_activation_into`.

**4.2** Tape builder wiring: `FusedMatMulBiasActivation` → `InlineMatMulBiasActivation`.

**4.3** CLI inspect + kv/store.rs exhaustive match coverage.

### Phase 5: Tests + Benchmark

- Graph fusion test: `Input → MatMul → Add(const) → Relu → Output` fuses to `Input → FusedMatMulBiasActivation → Output`
- No-fuse: Add's second input is NOT a constant → don't fuse
- No-fuse: MatMul has fan-out → don't fuse
- Correctness: fused output bit-identical to unfused
- **Benchmark**: MatMul+Add+Silu (3 ops) vs fused (1 op) — this should show clear improvement

---

## Files to Modify

| File | Change |
|------|--------|
| [graph/mod.rs](crates/hologram-graph/src/graph/mod.rs) | Add `FusedMatMulBiasActivation` |
| [float_fusion.rs](crates/hologram-graph/src/fusion/float_fusion.rs) | Add `try_fuse_matmul_bias_activation()` |
| [fusion/mod.rs](crates/hologram-graph/src/fusion/mod.rs) | Wire into `fuse()` |
| [tape.rs](crates/hologram-exec/src/tape.rs) | Add `InlineMatMulBiasActivation` + dispatch |
| [tape_builder.rs](crates/hologram-exec/src/tape_builder.rs) | Wire mapping |
| [matmul.rs](crates/hologram-exec/src/float_dispatch/matmul.rs) | Add `dispatch_matmul_bias_activation_into` |
| [kv/store.rs](crates/hologram-exec/src/kv/store.rs) | Exhaustive match |
| CLI inspect files | Display support |
| [epilogue_fusion.rs](crates/hologram-bench/benches/epilogue_fusion.rs) | Add bias fusion benchmark |

## Verification

1. `cargo test --workspace` — all pass
2. `cargo clippy --workspace -- -D warnings` — clean
3. `cargo bench -p hologram-bench --bench epilogue_fusion` — fused MatMul+Bias+Activation beats unfused 3-op path
4. Update SPRINT.md with results
