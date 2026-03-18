# 005: Compile-Time-First Acceleration Plan

## Context

Hologram's core thesis: O(1) per-element dispatch via LUT/KV lookups. **Verified and validated** — unary ops, view fusion, binary ops, and dispatch routing are all genuinely O(1). The remaining bottlenecks (LUT-GEMM stride-N cache misses, per-dispatch weight deserialization, unfused linear layers, scalar softmax/attention) should be attacked by pushing as much work as possible into compile time. The compiler knows weight dimensions, quantization format, and graph structure — runtime should execute pre-planned instructions with zero decision-making.

**Scope**: Phases 1-3 implement now. Phases 4-6 roadmap.
**SIMD targets**: x86_64 AVX2, aarch64 NEON, WASM SIMD (all three).

---

## Phase 1: Compile-Time Weight Layout + Runtime Cache

**Goal**: Eliminate the two worst performance problems with zero precision loss.

### Problem A: Weight re-deserialization on every dispatch

In `kv/store.rs:134`, `dispatch_lut_gemm_4` calls `rkyv::from_bytes::<QuantizedWeights4>()` **on every forward pass**. For a 7B model with 64 linear layers, this deserializes all weight matrices every single token. This is likely the single biggest performance bug.

**Fix**: Compile-time weight pinning + runtime cache.

```rust
// New in hologram-exec/src/kv/weight_cache.rs
pub struct WeightCache {
    q4: HashMap<ConstantId, QuantizedWeights4>,
    q8: HashMap<ConstantId, QuantizedWeights8>,
}

impl WeightCache {
    /// Pre-populate during executor initialization (once per session)
    pub fn seed(&mut self, constants: &ConstantStore, weights: &[u8]) { ... }

    /// O(1) lookup — no deserialization
    pub fn get_q4(&self, cid: ConstantId) -> Option<&QuantizedWeights4> { ... }
    pub fn get_q8(&self, cid: ConstantId) -> Option<&QuantizedWeights8> { ... }
}
```

The executor seeds the cache once at load time. All subsequent dispatches are HashMap lookups — **O(1) amortized, zero deserialization**.

**Files**:
- New: `crates/hologram-exec/src/kv/weight_cache.rs`
- Modify: `crates/hologram-exec/src/kv/store.rs` — `dispatch_lut_gemm_4/8` accept `&WeightCache` instead of raw bytes
- Modify: `crates/hologram-exec/src/eval/executor.rs` — seed cache in `KvExecutor::new` or `execute`

**Expected speedup**: 5-10x reduction in per-call overhead for autoregressive decode.

### Problem B: Stride-N cache misses in weight index access

In `matmul.rs:25`: `weights.indices[l * n + col as usize]` — stride-N access. For N=4096, this means one useful byte per 4KB cache line fetch. Same problem in `parallel.rs:69`.

**Fix**: Compile-time column-major transpose of weight indices.

The compiler transposes weight indices from row-major `[l * n + j]` to column-major `[j * k + l]` during the emit stage. The runtime inner loop then reads stride-1 (sequential bytes, perfect cache behavior).

```rust
// New in hologram-compiler/src/layout/mod.rs
pub fn transpose_weight_indices_q8(qw: &mut QuantizedWeights8) {
    let k = qw.rows as usize;
    let n = qw.cols as usize;
    let mut transposed = vec![0u8; k * n];
    for l in 0..k {
        for j in 0..n {
            transposed[j * k + l] = qw.indices[l * n + j];
        }
    }
    qw.indices = transposed;
}
// Similar for Q4 (nibble-packed, more complex bit manipulation)
```

**Alternative — Tiled blocked layout**: Store indices in `[TILE_K × TILE_J]` blocks for even better cache behavior with the tiled kernel from Step 1.3. The compiler selects optimal tile sizes based on matrix dimensions:
- Q4: TILE_K=64, TILE_J=64 (2KB tiles)
- Q8: TILE_K=32, TILE_J=32 (1KB tiles)
- Small matrices (k*n < 1024): column-major fallback

**Implementation**: Add a `layout` field to `QuantizedWeights4/8` indicating the index layout. The compiler sets it during emit; the runtime reads it to select the correct kernel.

```rust
#[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum WeightLayout { RowMajor, ColMajor, TiledBlocked { tile_k: u16, tile_j: u16 } }
```

**Files**:
- Modify: `crates/hologram-exec/src/lut_gemm/quantize.rs` — add `layout` field to `QuantizedWeights4/8`, add transpose functions
- New: `crates/hologram-compiler/src/layout/mod.rs` — layout optimizer stage
- Modify: `crates/hologram-compiler/src/compiler/mod.rs` — add layout stage between fuse and emit
- Modify: `crates/hologram-exec/src/lut_gemm/matmul.rs` — add col-major/tiled kernel variants

**Expected speedup**: 2-4x for large matrices (K >= 2048). Inner loop goes from ~20 cycles/element (L2 miss) to ~2-3 cycles/element (L1 hit).

### Step 1.3: Tiled Multi-Column LUT-GEMM Kernels

With tiled weight layout from Step 1.2, add tiled kernels that process TILE_J columns simultaneously:

```rust
fn lut_gemm_8bit_tiled(a_row: &[f32], weights: &QuantizedWeights8, output: &mut [f32]) {
    let (tile_k, tile_j) = weights.layout.tile_sizes();
    for j_base in (0..n).step_by(tile_j) {
        let mut books = [Psumbook8::new(); TILE_J];  // 2 × 1024B = 2KB in L1
        for l in 0..k {
            let a_val = a_row[l];  // loaded once per tile
            for jj in 0..tile_j {
                let idx = weights.tiled_index(j_base, jj, l);  // stride-1 read
                books[jj].accumulate(idx, a_val);
            }
        }
        for jj in 0..tile_j {
            output[j_base + jj] = books[jj].dot(&weights.centroids);
        }
    }
}
```

**Parallel decomposition** (compile-time planned): The compiler embeds the number of column tiles in the op metadata. The executor parallelizes over tiles via rayon when `num_col_tiles >= 4`.

**Files**:
- Modify: `crates/hologram-exec/src/lut_gemm/matmul.rs` — tiled kernels
- Modify: `crates/hologram-exec/src/lut_gemm/parallel.rs` — tile-parallel instead of per-column parallel

### Step 1.4: SIMD Dot Products (All Architectures)

Add SIMD-accelerated `dot` methods to Psumbook4/Psumbook8:

```rust
// In psumbook.rs
impl Psumbook4 {
    #[cfg(target_arch = "x86_64")]
    pub fn dot_avx2(&self, centroids: &[f32; 16]) -> f32 { ... }

    #[cfg(target_arch = "aarch64")]
    pub fn dot_neon(&self, centroids: &[f32; 16]) -> f32 { ... }

    #[cfg(target_arch = "wasm32")]
    pub fn dot_wasm(&self, centroids: &[f32; 16]) -> f32 { ... }
}
```

Plus ARM NEON `vtbl`-based ElementWiseView in `view/simd.rs`.

**Files**:
- Modify: `crates/hologram-exec/src/lut_gemm/psumbook.rs` — SIMD dot per arch
- New: `crates/hologram-exec/src/lut_gemm/simd.rs` — shared SIMD helpers
- Modify: `crates/hologram-core/src/view/simd.rs` — NEON + WASM paths

**Expected additional speedup**: 1.3-1.5x on top of tiling.

---

## Phase 2: Compile-Time Fusion (Eliminate Intermediate Buffers)

**Goal**: The compiler detects common patterns and emits fused ops. Runtime executes single-pass fused kernels.

### Step 2.1: Compile-Time MatMul+Bias+Activation Fusion

The pattern `MatMulLut → Float(Add) [bias constant] → Float(activation)` appears in every linear layer. The compiler should detect and fuse this.

**New GraphOp variant**:
```rust
GraphOp::FusedLinear {
    weight_cid: ConstantId,
    bias_cid: Option<ConstantId>,
    activation: Option<FloatOp>,  // Relu, Gelu, Silu
    layout: WeightLayout,
    k: u32,
    n: u32,
}
```

**Compiler fusion pass** (new file `hologram-graph/src/fusion/linear_fusion.rs`):
1. Walk topological order. For each `MatMulLut{4,8}` node:
2. Check: sole successor is `Float(Add)`, second Add input is a Constant? → fuse bias
3. Check: sole successor of Add is unary `Float(activation)`? → fuse activation
4. Replace subgraph with `FusedLinear`, remove intermediate nodes

**Runtime kernel**: Bias and activation applied inline during psumbook dot:
```rust
let val = book.dot(&centroids);
let val = val + bias[j];           // fused bias (still in register)
let val = activation.apply(val);    // fused activation (still in register)
output[i * n + j] = val;
```

Eliminates two full M×N buffer passes per linear layer.

**Files**:
- New: `crates/hologram-graph/src/fusion/linear_fusion.rs`
- Modify: `crates/hologram-graph/src/fusion/mod.rs` — integrate linear fusion
- Modify graph op enum (in hologram-graph) — add `FusedLinear` variant
- Modify: `crates/hologram-exec/src/kv/store.rs` — dispatch `FusedLinear`
- New: `crates/hologram-exec/src/lut_gemm/fused.rs` — fused kernel

**Expected speedup**: 15-25% per linear layer.

### Step 2.2: Compile-Time Norm+Activation Fusion

The pattern `RmsNorm → Gelu|Silu|Relu` appears in every transformer layer.

**New FloatOp variant**:
```rust
FloatOp::FusedRmsNormActivation { size: u32, epsilon: u32, activation: u8 }
```

**Compiler fusion pass**: New function `try_fuse_norm_activation()` added to the single-pass fusion engine. Detects `RmsNorm` followed by unary activation (single-successor check).

**Runtime kernel**: Single-pass norm + activate:
```rust
for row in x.chunks_mut(size) {
    let rms = fast_rsqrt(row.iter().map(|v| v * v).sum::<f32>() / size as f32 + eps);
    for (v, w) in row.iter_mut().zip(weight) {
        *v = activation((*v * rms) * w);
    }
}
```

Also includes `fast_rsqrt` (Quake III + Newton-Raphson) for RmsNorm.

**Files**:
- New: `crates/hologram-graph/src/fusion/norm_activation_fusion.rs`
- Modify: `crates/hologram-exec/src/float_dispatch.rs` — fused kernel + fast_rsqrt
- Modify float_op.rs — add variant

**Expected speedup**: 1.3-1.5x for norm+activation blocks.

### Step 2.3: Compile-Time LUT-exp for Softmax

Precompute a 65536-entry f32 exp table covering [-16.0, 0.0] (256KB, fits L2). The softmax kernel uses table lookup instead of `f32::exp()`:

```rust
static SOFTMAX_EXP: LazyLock<Box<[f32; 65536]>> = LazyLock::new(|| { ... });

fn lut_exp(x: f32) -> f32 {
    let idx = (-x.max(-16.0) * (65535.0 / 16.0)) as usize;
    SOFTMAX_EXP[idx.min(65535)]
}
```

Max/sum/normalize stay in full f32. Only exp() uses the LUT.

**Files**:
- Modify: `crates/hologram-exec/src/float_dispatch.rs` — LUT-exp in softmax
- Feature-gated: `#[cfg(feature = "lut-exp")]`

**Expected speedup**: 1.3x for softmax (exp is ~60% of softmax cost).

### Step 2.4: Compile-Time Buffer Alignment

The workspace planner should enforce SIMD alignment:

```rust
pub struct BufferSlot {
    pub slot_id: u32,
    pub occupants: Vec<NodeId>,
    pub alignment: usize,  // 32 for AVX2, 16 for NEON
}
```

Buffer offsets in the flat workspace are aligned: `offset = (offset + align - 1) & !(align - 1)`. The compiler selects alignment based on a target-arch flag.

**Files**:
- Modify: `crates/hologram-compiler/src/workspace/mod.rs`
- Modify executor buffer allocation for aligned `Vec<u8>`

---

## Phase 3: Compile-Time Attention Planning + Tiled Execution

**Goal**: The compiler detects attention patterns and emits a single fused op with pre-planned tile sizes. Runtime executes Flash Attention-style tiled attention with O(1) memory for scores.

### Step 3.1: Compile-Time Attention Op with Baked Tile Sizes

The compiler selects optimal tile sizes based on head_dim and embeds them in the op:

```rust
FloatOp::Attention {
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    scale: u32,
    causal: bool,
    tile_r: u16,  // compiler-selected: e.g., 32 for head_dim=128
    tile_c: u16,  // compiler-selected: e.g., 32
}
```

### Step 3.2: Online-Softmax Tiled Attention Kernel

Replace the current attention which materializes full `[seq_q, seq_k]` scores with online-softmax tiling:

```
for qi in (0..seq_q).step_by(tile_r):
    state = OnlineSoftmax::new()
    for kj in (0..seq_k).step_by(tile_c):
        if causal && fully_masked(qi, kj): continue  // ~50% skip
        scores = matmul_tile(Q[qi..], K[kj..])       // [tile_r, tile_c] only
        online_softmax_update(&mut state, scores, V[kj..])
    output[qi..] = state.finalize()
```

- Scores buffer: `tile_r × tile_c` (constant, e.g. 32×32 = 4KB) vs `seq_q × seq_k` (quadratic)
- Causal tile-skip: ~50% compute reduction for autoregressive models
- Numerically equivalent (online softmax is exact modulo float associativity)

### Step 3.3: Per-Head Parallelism (Compile-Time Planned)

The compiler knows `num_q_heads` and embeds parallelism strategy:
- `num_q_heads >= 4`: parallel over heads via rayon
- `num_q_heads < 4`: sequential (avoid rayon overhead)

Currently the per-head loop in `float_dispatch.rs:1698` is always sequential.

**Files**:
- Modify: `crates/hologram-exec/src/float_dispatch.rs` — new `dispatch_tiled_attention` function
- Modify float_op.rs — add tile_r/tile_c fields to Attention variant
- Modify compiler fusion pass — select tile sizes during emit

**Expected speedup**: 2-4x for long sequences, plus head-level parallelism.

---

## Phases 4-6: Roadmap (Future Work)

### Phase 4: Sliding Window + Quantized K Cache
- Add `window_size` to Attention op → O(seq × window) instead of O(seq²)
- Ring-buffer KV cache storing only `window_size` tokens
- Quantize K cache to Q4 via existing LUT-GEMM infrastructure → 4x memory reduction

### Phase 5: Precomputed Scatter Groups (PSG)
- Compile-time: for each (column, Q-level), store sorted row positions
- Runtime: sequential gather-sum instead of random psumbook scatter
- 3-6x additional speedup for Q4 matmul

### Phase 6: Transformer Block Fusion + DQ-GEMM
- Pattern-match entire transformer blocks → single `TransformerLayer` op
- Eliminate per-op dispatch overhead (~20 matches per layer)
- DQ-GEMM: quantize activations too → integer-only hot loop (research, may not ship)

---

## Implementation Order

```
Phase 1.A  Weight cache (eliminate re-deserialization)     → pure runtime, no format change
Phase 1.B  Column-major weight indices                     → compile-time layout + new kernel
Phase 1.3  Tiled multi-column kernels                      → builds on 1.B layout
Phase 1.4  SIMD dot products (AVX2/NEON/WASM)              → orthogonal, can parallel with 1.3
Phase 2.1  MatMul+Bias+Activation fusion                   → new fusion pass + fused kernel
Phase 2.2  Norm+Activation fusion + fast_rsqrt             → new fusion pass + fused kernel
Phase 2.3  LUT-exp for softmax                             → standalone, feature-gated
Phase 2.4  Buffer alignment                                → workspace planner change
Phase 3.1  Attention op with baked tile sizes               → graph IR change
Phase 3.2  Tiled attention kernel                           → new dispatch function
Phase 3.3  Per-head parallelism                             → rayon in attention dispatch
```

Steps 1.A and 1.4 can be developed in parallel with the rest of Phase 1.
Steps 2.1-2.4 are independent of each other and can be parallelized.
Phase 3 depends on Phase 2.3 (LUT-exp is used inside tiled attention softmax).

---

## Verification Plan

For each step:
1. `cargo test --workspace` — all 918+ tests pass
2. `cargo clippy -- -D warnings` — zero warnings
3. New kernel output matches old kernel output within epsilon
4. Property tests: random matrices, `max_relative_error < threshold`
5. Criterion benchmarks for standard dimensions (k=4096, n=4096, seq=512/2048)
6. Cross-platform: verify scalar fallbacks when SIMD features disabled

**Key invariant**: Old `.holo` archives with `MatMulLut4/8` (RowMajor) must continue to work. New ops are additive to the `GraphOp` enum. The weight cache gracefully handles both old and new layouts.

---

## Composite Speedup Estimate

| Component | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|---|---|---|---|---|
| Weight deser | ~100μs/call | ~0 (cached) | ~0 | ~0 |
| LUT-GEMM per element | ~20 cy (L2 miss) | ~3 cy (L1 + SIMD) | ~2.5 cy (fused bias/act) | ~2.5 cy |
| Softmax exp | ~5 cy/element | ~5 cy | ~1 cy (LUT) | ~1 cy |
| Attention scores | O(seq²×d) dense | same | same | O(seq²×d) tiled, 50% causal skip |
| Norm + activation | 2 passes | 2 passes | 1 pass | 1 pass |
| Buffer traffic | ~10 intermediates/layer | ~10 | ~6 (fused) | ~4 (fused attention) |

**End-to-end estimate**: **3-5x** for a 7B model at seq=2048 after all three phases.
