# Plan: MatMul Optimization Investigation

## Context

MatMul dominates transformer inference time (~60-80% of compute). The hologram codebase has a multi-layered MatMul dispatch: tape executor → backend (GPU) → CPU fallback. On macOS with Accelerate, BLAS sgemm is used. Without it, a hand-rolled k-outer loop handles float matmul. LUT-GEMM handles quantized paths.

After the Sprint 21 optimizations (2.1x decode speedup), MatMul dispatch overhead is minimal — but the actual kernel compute has several optimization opportunities.

## Current Architecture

| Path | When | Implementation |
|------|------|----------------|
| InlineMatMul → Metal SGEMM | macOS, output ≥ 128×128 | 16×16 tiled, shared memory |
| InlineMatMul → Accelerate BLAS | macOS, output < 128×128 | cblas_sgemm |
| InlineMatMul → CPU k-outer loop | non-macOS | `for i..m { for p..k { for j..n { o[j] += a*b[j] } } }` |
| dispatch_gemm (alpha/beta/trans) | ONNX Gemm nodes | i,j-outer loop with runtime transpose conditionals |
| LUT-GEMM Q4 | quantized weights | Psumbook4 accumulate + 16-element dot |
| LUT-GEMM Q8 tiled | quantized weights | 4-column tiled, Psumbook8 |

## Optimization Opportunities (ranked by expected impact)

### Phase 1: dispatch_gemm loop restructuring (HIGH — fixes a perf bug)

**File**: `crates/hologram-exec/src/float_dispatch/matmul.rs:405-429`

**Problem**: The non-BLAS `dispatch_gemm` uses an i,j-outer / k-inner loop structure with runtime transpose conditionals inside the innermost loop. This is ~3-5x slower than the k-outer pattern used by `dispatch_matmul`.

**Fix**: Pre-transpose A and/or B once if `trans_a`/`trans_b` are true, then call the standard k-outer matmul + apply alpha/beta scaling. This eliminates the per-element branch and uses the cache-friendly access pattern.

```rust
// Pre-transpose if needed (one-time O(m*k) or O(k*n) cost)
let a_t = if p.trans_a { transpose(a, m, k) } else { a.to_vec() };
let b_t = if p.trans_b { transpose(b, k, n) } else { b.to_vec() };
// Standard k-outer matmul
for i in 0..m {
    for q in 0..k {
        let a_val = a_t[i * k + q];
        for j in 0..n {
            out[i * n + j] += a_val * b_t[q * n + j];
        }
    }
}
// Apply alpha/beta
if p.alpha != 1.0 || p.beta != 0.0 { ... }
```

### Phase 2: dispatch_matmul_into — write directly to out_buf (MEDIUM)

**File**: `crates/hologram-exec/src/float_dispatch/matmul.rs:183`

**Problem**: `dispatch_matmul_into` allocates `vec![0.0f32; out_size]`, computes into it, then does `out_buf.extend_from_slice(bytemuck::cast_slice(&out))`. Same pattern we fixed in norm.

**Fix**: Use `alloc_f32_in(out_buf, out_size)` to write directly to `out_buf`, zero intermediate Vec. Import the helper from norm.rs or make it a shared utility.

### Phase 3: CPU register-blocked matmul (MEDIUM — non-BLAS platforms)

**File**: `crates/hologram-exec/src/float_dispatch/matmul.rs:191-201`

**Problem**: The k-outer loop processes one row of A at a time. On non-BLAS platforms (Linux, WASM), this leaves performance on the table — the compiler can't easily tile across M/N dimensions.

**Fix**: Add a micro-kernel with register blocking. Process MR×NR tiles (e.g., 4×8 or 8×4) in the M×N space, accumulating K iterations into registers before writing back.

```rust
const MR: usize = 4;
const NR: usize = 8;
for i in (0..m).step_by(MR) {
    for j in (0..n).step_by(NR) {
        let mut acc = [[0.0f32; NR]; MR]; // 32 registers
        for p in 0..k {
            for ii in 0..MR {
                let a_val = a[(i + ii) * k + p];
                for jj in 0..NR {
                    acc[ii][jj] += a_val * b[p * n + j + jj];
                }
            }
        }
        // Write tile to output
        for ii in 0..MR {
            for jj in 0..NR {
                out[(i + ii) * n + j + jj] = acc[ii][jj];
            }
        }
    }
}
// + remainder handling for non-multiple-of-MR/NR
```

### Phase 4: Batched matmul GPU dispatch (MEDIUM — GPU utilization)

**File**: `crates/hologram-exec/src/backend/metal.rs`, `webgpu.rs`

**Problem**: Batched matmul (attention Q@K^T, attn@V) falls back to CPU because Metal/WebGPU only have single-matrix kernels. For 32-head attention with seq=512, that's 32 CPU matmuls instead of 1 GPU dispatch.

**Fix**: Add a batched SGEMM kernel using the Z dimension of the GPU grid.

```metal
kernel void batched_sgemm(
    ...,
    constant uint& batch_count [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    ...
) {
    uint batch = gid.z;
    // offset A, B, C by batch * stride
}
```

### Phase 5: Matmul + Bias/Activation fusion (LOW — reduces memory traffic)

**Problem**: Common transformer pattern is MatMul → Add (bias) → ReLU/GeLU. Currently 3 separate tape instructions with intermediate buffer writes.

**Fix**: Add fused TapeKernel variants like `InlineMatMulAdd { m, k, n }` that apply bias in the same loop as matmul output, avoiding a full buffer write-then-read cycle.

This is lower priority because the intermediate buffers are small (arena recycling) and the matmul compute dominates.

---

## Critical Files

| File | Changes |
|------|---------|
| `crates/hologram-exec/src/float_dispatch/matmul.rs` | Phases 1-3: gemm restructure, direct out_buf, register blocking |
| `crates/hologram-exec/src/backend/metal.rs` | Phase 4: batched SGEMM kernel |
| `crates/hologram-exec/src/backend/webgpu.rs` | Phase 4: batched SGEMM kernel |

## Verification

1. `cargo test --workspace` — all existing matmul tests must pass
2. `cargo bench -p hologram-bench --bench lut_gemm` — no regression on quantized path
3. `cargo bench -p hologram-bench --bench executor` — transformer decode step should improve
4. New benchmark: `matmul_sizes` — sweep M/K/N to measure crossover points
