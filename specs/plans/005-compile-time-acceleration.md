# Hologram Runtime: Compile-Time-First Acceleration Plan (v2)

## Context

Hologram's core thesis: O(1) per-element dispatch via LUT/KV lookups. **Verified and validated** — unary ops, view fusion, binary ops, and dispatch routing are all genuinely O(1). The remaining bottlenecks (LUT-GEMM stride-N cache misses, per-dispatch weight deserialization, unfused linear layers, scalar softmax/attention) should be attacked by pushing as much work as possible into compile time. The compiler knows weight dimensions, quantization format, and graph structure — runtime should execute pre-planned instructions with zero decision-making.

**Unifying principle**: ALL computation — including orchestration, shape resolution, dispatch routing, and buffer management — should reduce to KV/LUT lookups. The compiler converts decisions into static tables; the runtime reads them. No HashMaps, no pattern matching, no shape inference at runtime. The executor becomes a flat instruction-tape reader where every step is a pre-resolved indexed lookup.

**Scope**: Phases 0-3 implement now. Phases 4-6 roadmap. Phases 7-10 strategic.
**SIMD targets**: x86_64 AVX2, aarch64 NEON, WASM SIMD (all three).

---

## Cache Residency Map

All tables are designed to cascade through the CPU cache hierarchy. The hot core fits in L1; warm tables fit in L2; cold storage streams from L3/RAM.

| Table | Size | Cache Tier | Latency | Access Pattern |
|---|---|---|---|---|
| **L1 Hot Core** (~12KB) | | | | |
| Q0 activation LUTs (21) | 5.4KB | L1 | ~1 cy | Every element |
| ElementWiseView (per-op) | 256B | L1 | ~1 cy | Every element, cache-line aligned |
| HLUT page selectors | 256B each | L1 | ~1 cy | Every element (ARE ElementWiseViews) |
| HLUT hot pages (~6-8 of 256) | ~2KB | L1 | ~1 cy | Correlated access → same pages |
| Psumbook4 | 64B (1 cache line) | L1/Register | ~0-1 cy | Per output element in LUT-GEMM |
| Instruction tape | ~1-4KB | L1 | ~1 cy | Sequential, prefetch-friendly |
| **L2 Warm Layer** (~1MB) | | | | |
| Psumbook8 | 1024B | L1-L2 | ~1-4 cy | Per output element, 16 cache lines |
| HLUT all activations (Q2 precision) | ~260KB | L2 | ~5 cy | Smaller than flat Q1 (2.7MB) |
| Q0×Q0 binary tables (6 ops) | 384KB | L2 | ~5 cy | Per binary element-wise op |
| Softmax exp Q1 LUT | 128KB | L2 | ~5 cy | Burst access during softmax |
| Flat arena hot buffers | ~100KB-1MB | L2 | ~5 cy | Current layer's intermediates |
| **L3 Cold Storage** (~16MB) | | | | |
| Q1 activation tables (21) | 2.7MB | L2-L3 | ~5-12 cy | Only if HLUT not available |
| Quantized weight indices (per layer) | 4-32MB | L3 | ~30 cy | Streamed during LUT-GEMM |
| Flat arena full workspace | ~1-16MB | L2-L3 | ~5-30 cy | Buffer reuse via liveness |
| **RAM** (~GB) | | | | |
| KV cache | 0.5-2GB | RAM | ~100+ cy | Per-layer read during attention |
| Weight blob (all layers) | 1-14GB | RAM/mmap | ~100+ cy | Streamed, one layer at a time |

**Design principle**: The per-element hot path (LUT lookup, Psumbook accumulate, instruction fetch) fits entirely in L1 (~12KB). The per-op warm path (broadcast strides, activation tables, binary tables) fits in L2 (~1MB). The per-layer cold path (weights, KV cache) streams from L3/RAM. This layering ensures that the O(1) lookup cost is genuinely ~1 cycle for the common case.

---

## Phase 1: Compile-Time Weight Layout + Runtime Cache

**Goal**: Eliminate the two worst performance problems with zero precision loss.

### Problem A: Weight re-deserialization on every dispatch

In [kv/store.rs:134](crates/hologram-exec/src/kv/store.rs#L134), `dispatch_lut_gemm_4` calls `rkyv::from_bytes::<QuantizedWeights4>()` **on every forward pass**. For a 7B model with 64 linear layers, this deserializes all weight matrices every single token. This is likely the single biggest performance bug.

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
- Modify: [kv/store.rs](crates/hologram-exec/src/kv/store.rs) — `dispatch_lut_gemm_4/8` accept `&WeightCache` instead of raw bytes
- Modify: [eval/executor.rs](crates/hologram-exec/src/eval/executor.rs) — seed cache in `KvExecutor::new` or `execute`

**Expected speedup**: 5-10x reduction in per-call overhead for autoregressive decode.

### Problem B: Stride-N cache misses in weight index access

In [matmul.rs:25](crates/hologram-exec/src/lut_gemm/matmul.rs#L25): `weights.indices[l * n + col as usize]` — stride-N access. For N=4096, this means one useful byte per 4KB cache line fetch. Same problem in [parallel.rs:69](crates/hologram-exec/src/lut_gemm/parallel.rs#L69).

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
- Modify: [lut_gemm/quantize.rs](crates/hologram-exec/src/lut_gemm/quantize.rs) — add `layout` field to `QuantizedWeights4/8`, add transpose functions
- New: `crates/hologram-compiler/src/layout/mod.rs` — layout optimizer stage
- Modify: [compiler/mod.rs](crates/hologram-compiler/src/compiler/mod.rs) — add layout stage between fuse and emit
- Modify: [lut_gemm/matmul.rs](crates/hologram-exec/src/lut_gemm/matmul.rs) — add col-major/tiled kernel variants

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
- Modify: [lut_gemm/matmul.rs](crates/hologram-exec/src/lut_gemm/matmul.rs) — tiled kernels
- Modify: [lut_gemm/parallel.rs](crates/hologram-exec/src/lut_gemm/parallel.rs) — tile-parallel instead of per-column parallel

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

Plus ARM NEON `vtbl`-based ElementWiseView in [view/simd.rs](crates/hologram-core/src/view/simd.rs).

**Files**:
- Modify: [lut_gemm/psumbook.rs](crates/hologram-exec/src/lut_gemm/psumbook.rs) — SIMD dot per arch
- New: `crates/hologram-exec/src/lut_gemm/simd.rs` — shared SIMD helpers
- Modify: [view/simd.rs](crates/hologram-core/src/view/simd.rs) — NEON + WASM paths

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
- Modify: [fusion/mod.rs](crates/hologram-graph/src/fusion/mod.rs) — integrate linear fusion
- Modify graph op enum (in hologram-graph) — add `FusedLinear` variant
- Modify: [kv/store.rs](crates/hologram-exec/src/kv/store.rs) — dispatch `FusedLinear`
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
- Modify: [float_dispatch.rs](crates/hologram-exec/src/float_dispatch.rs) — fused kernel + fast_rsqrt
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
- Modify: [float_dispatch.rs](crates/hologram-exec/src/float_dispatch.rs) — LUT-exp in softmax
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

Currently the per-head loop in [float_dispatch.rs:1698](crates/hologram-exec/src/float_dispatch.rs) is always sequential.

**Files**:
- Modify: [float_dispatch.rs](crates/hologram-exec/src/float_dispatch.rs) — new `dispatch_tiled_attention` function
- Modify float_op.rs — add tile_r/tile_c fields to Attention variant
- Modify compiler fusion pass — select tile sizes during emit

**Expected speedup**: 2-4x for long sequences, plus head-level parallelism.

---

## Phase 0: Execution Orchestration Overhaul (NEW — highest ROI)

**Goal**: Eliminate per-dispatch allocation storm and HashMap overhead. This phase may deliver more speedup than Phases 1-3 combined for graphs with many small ops, because the orchestration overhead currently dominates for anything below ~100μs compute per node.

### Finding: The allocation storm

Every single op dispatch allocates a fresh `Vec<u8>` for output. Every float dispatch allocates strides, broadcast shapes, and intermediate buffers. A single binary broadcast op allocates **5-7 Vecs** (two cast_f32, broadcast_shapes, two compute_broadcast_strides, compute_strides, output). For a transformer layer with ~20 float ops, that's **100+ allocations per layer per token** just for plumbing.

### Finding: HashMap-based arena

`BufferArena` uses `HashMap<NodeId, Cow<[u8]>>` (`buffer/arena.rs:20`). Every input gather does 2-4 HashMap lookups per node input. Shape resolution does 4-8 more. For a 32-node level: **128-256 HashMap operations** before any compute starts. The compiler already plans buffer slots with liveness intervals — the runtime ignores this.

### Finding: Shape resolution repeated per-dispatch

`dispatch_level` (executor.rs:1133-1173) resolves shapes from scratch for every node: 4-8 HashMap lookups + redundant `resolve_compiled_shape()` calls + `resolve_dynamic_sizes()` checking 15+ FloatOp variants. `propagate_level_shapes` runs as a **separate traversal** before dispatch. Two full passes over every level, every execution.

### Step 0.1: Flat Pre-Allocated Buffer Arena

Replace `HashMap<NodeId, Cow<[u8]>>` with a flat `Vec<u8>` workspace where each node's output is at a compile-time-determined offset:

```rust
pub struct FlatArena {
    workspace: Vec<u8>,           // Single contiguous allocation
    offsets: Vec<(u32, u32)>,     // (offset, length) per slot, indexed by slot_id
    node_to_slot: Vec<u16>,       // node_id → slot_id, dense array
}
```

The compiler already computes `BufferSlot` assignments with liveness intervals (`workspace/mod.rs`). Embed slot offsets in the archive. At runtime:
- `arena.get(node_id)` → `&workspace[offsets[node_to_slot[node_id]]]` — **O(1) array index, no hashing**
- `arena.set(node_id, data)` → memcpy into pre-allocated slot — **no allocation**
- Single `vec![0u8; total_workspace_bytes]` at init — **one allocation total**

**Files**:
- New: `crates/hologram-exec/src/buffer/flat_arena.rs`
- Modify: `crates/hologram-exec/src/eval/executor.rs` — use FlatArena
- Modify: `crates/hologram-compiler/src/workspace/mod.rs` — emit slot offsets + sizes
- Modify archive format — embed workspace layout

**Expected speedup**: Eliminates 960-1,920 HashMap ops per execution cycle. For small ops (LUT/elementwise), this overhead is currently **larger than the compute itself**.

### Step 0.2: Output Buffer Pre-allocation / Reuse in Dispatch

Instead of every `dispatch_float_ctx` returning a fresh `Vec<u8>`, pass a mutable output slice from the flat arena:

```rust
// Before: allocates per-op
fn unary_map(inputs: &[&[u8]], f: impl Fn(f32) -> f32) -> ExecResult<Vec<u8>> {
    let out: Vec<f32> = x.iter().map(|&v| f(v)).collect();  // ALLOCATES
    Ok(f32_vec_to_bytes(out))
}

// After: writes into pre-allocated buffer
fn unary_map_into(inputs: &[&[u8]], output: &mut [u8], f: impl Fn(f32) -> f32) -> ExecResult<()> {
    let x = cast_f32(inputs[0])?;
    let out = bytemuck::cast_slice_mut::<u8, f32>(output);
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = f(v);
    }
    Ok(())
}
```

This requires the dispatch API to accept `&mut [u8]` output buffers instead of returning `Vec<u8>`. The flat arena provides the buffer; the dispatch writes into it.

**Eliminates**: 100+ Vec allocations per transformer layer per token.

**Files**:
- Modify: `crates/hologram-exec/src/kv/store.rs` — `dispatch_into` variant accepting output slice
- Modify: `crates/hologram-exec/src/float_dispatch.rs` — `_into` variants for all kernels
- Modify: `crates/hologram-exec/src/eval/executor.rs` — wire dispatch_into with flat arena

### Step 0.3: Compile-Time Shape Resolution

Merge shape propagation into the schedule at compile time. The compiler resolves all shapes that can be determined statically and embeds them per-node:

```rust
pub struct CompiledNode {
    op: GraphOp,
    input_slot_ids: SmallVec<[u16; 4]>,   // direct slot references, no HashMap
    output_slot_id: u16,
    output_shape: SmallVec<[u32; 4]>,      // fully resolved if possible, 0 = dynamic
    output_bytes: u32,                      // pre-computed buffer size
    elem_size: u8,                          // 1, 2, or 4
}
```

At runtime: no shape HashMap lookups, no `resolve_compiled_shape`, no `resolve_dynamic_sizes` for statically-shaped ops. Only dynamic shapes (batch dim, sequence length) need runtime resolution.

**Eliminates**: 5,120+ HashMap ops per execution for shape tracking. Eliminates the separate `propagate_level_shapes` pass.

**Files**:
- New structure in `crates/hologram-graph/src/schedule/compiled_node.rs`
- Modify: `crates/hologram-compiler/src/compiler/mod.rs` — emit CompiledNodes
- Modify: `crates/hologram-exec/src/eval/executor.rs` — use CompiledNode directly

### Step 0.4: Embed Execution Schedule in Archive

Currently `schedule_bridge.rs` re-runs Kahn's algorithm O(V+E) at load time if the schedule isn't embedded. Embed it as a flat instruction tape:

```rust
pub struct EmbeddedSchedule {
    node_ids: Vec<u32>,        // flat list of all node IDs in execution order
    level_starts: Vec<u32>,    // level_starts[i] = first index in node_ids for level i
}
```

At load time: zero schedule computation. Direct iteration.

**Files**:
- Modify: `crates/hologram-archive/src/section/` — new schedule section
- Modify: `crates/hologram-exec/src/eval/schedule_bridge.rs` — load from archive or fallback to Kahn's

### Step 0.5: Stride Memoization for Float Dispatch

`compute_strides` and `compute_broadcast_strides` allocate `Vec<usize>` on every call (`float_dispatch.rs:564-583`). Use stack-allocated `SmallVec<[usize; 6]>` (6 dims covers 99% of tensors) and cache broadcast stride pairs for repeated same-shape ops:

```rust
fn compute_strides_inline(shape: &[usize]) -> SmallVec<[usize; 6]> {
    let mut strides = SmallVec::new();
    strides.resize(shape.len(), 1);
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
```

**Eliminates**: 3-5 heap allocations per binary broadcast op.

**Files**:
- Modify: `crates/hologram-exec/src/float_dispatch.rs` — SmallVec strides, memoized broadcasts

### Step 0.6: Adaptive Parallel Threshold

Current threshold is fixed at 4 nodes (`parallel/mod.rs:13`). For cheap ops (LUT: ~50ns), rayon overhead dominates. For expensive ops (MatMul: ~100μs), 4 is fine.

**Fix**: The compiler annotates each level with estimated compute cost. The executor uses this to decide parallel vs sequential:

```rust
pub struct ParallelLevel {
    pub node_ids: Vec<NodeId>,
    pub estimated_cost_ns: u64,  // compiler estimate
}
// Threshold: parallel if estimated_cost_ns > 10_000 (10μs)
```

**Files**:
- Modify: `crates/hologram-graph/src/schedule/levels.rs` — add cost estimate
- Modify: `crates/hologram-exec/src/parallel/mod.rs` — cost-based threshold

---

## Phases 4-6: Roadmap (Near-Term)

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

## Phases 7-9: Quantize-Into-LUT-Domain (Strategic — Maximum LUT Coverage)

### The Coverage Gap

| Domain | LUT Coverage | Ops Covered |
|---|---|---|
| **Byte (Q0)** | **100%** | 21 activations + 6 arithmetic + bitwise |
| **Word (Q1)** | **100%** | 21 activations (128KB each, 2.7MB total) |
| **Float (f32)** | **0%** | 67 ops — all raw CPU compute |

The architectural opportunity: **push the quantization boundary inward** so more of the graph executes in byte/word domain where everything is already a lookup. The principle: **Quantize Early, Dequantize Late (QEDL)**.

### Phase 7: Float-to-LUT Promotion (Per-Op)

Convert individual float ops to use LUT approximation where the input domain is bounded:

#### 7.1: RoPE Frequency Precomputation
- `freq[i] = 1.0 / base^(2i/dim)` is **static per model** — precompute at compile time
- Store as constant table: `rope_freqs: [f32; max_dim/2]` (~256 entries = 1KB)
- `sin(pos * freq)` and `cos(pos * freq)` can use Q1 sin/cos tables (65536 entries)
- **Eliminates**: powf, division, sin, cos at runtime → all become table lookups
- **Error**: <0.02% (Q1 precision)

#### 7.2: Softmax exp via Q1 LUT (extends Phase 2.3)
- Input range for `exp(x - max)` is bounded: [-16, 0] (values below -16 are negligible)
- Q1 exp table already exists (65536 entries, 128KB)
- Quantize `(x - max)` to 16-bit index → single table lookup → f32 result
- Max-finding and sum normalization stay in f32 (irreducible reductions)
- **Eliminates**: `f32::exp()` (~5 cycles) → table lookup (~1 cycle)
- **Error**: <0.02%

#### 7.3: RmsNorm rsqrt via Piecewise-Linear LUT
- `rsqrt(ms + eps)` input domain is [eps, ~100.0] for typical hidden dims
- Log-scale quantize to 16-bit → Q1 rsqrt table (65536 entries, 128KB)
- Or: use fast_rsqrt (Quake III) + 1 Newton-Raphson iteration (~1e-4 error, no table needed)
- **Eliminates**: sqrt + division → single lookup or 3 integer ops

#### 7.4: Erf via Q1 LUT
- Erf(x) is currently a 7-term polynomial approximation (lines 880-891 in float_op.rs)
- Input domain is bounded (erf saturates at ±3): quantize [-4, 4] to 16-bit
- Q1 erf table: 65536 entries, 128KB
- **Eliminates**: polynomial evaluation (7 multiplies + 4 adds + exp) → single lookup

**Total new table memory**: ~512KB (rope_freqs + rsqrt + erf + softmax_exp), fits comfortably in L2.

### Phase 8: Quantized Intermediate Pipeline (QEDL)

The key architectural shift: instead of operating on f32 tensors between ops, **keep data in quantized form** between consecutive LUT-compatible ops and only dequantize at boundaries that require f32 (reductions, residual additions).

#### Current pipeline (everything in f32):
```
f32 → MatMul(f32) → f32 → RmsNorm(f32) → f32 → Gelu(f32) → f32 → MatMul(f32) → f32
       ↑ expensive      ↑ reduction        ↑ transcendental    ↑ expensive
```

#### QEDL pipeline (quantized intermediates):
```
f32 → LUT-GEMM(Q4) → Q0 → LUT-Gelu(Q0) → Q0 → LUT-GEMM(Q4) → f32
       ↑ already LUT    ↑ FREE (already exists!)  ↑ already LUT
```

The insight: **LUT-GEMM already outputs values that could be quantized to Q0** before the next activation. The activation is already a Q0 LUT. The next LUT-GEMM already accepts quantized inputs. The entire chain can stay in byte domain.

#### Where dequantization is forced:
1. **Residual connections**: `x + sublayer(x)` requires f32 addition for numerical stability
2. **Reductions in norms**: mean/variance computation needs f32 precision
3. **Softmax denominator**: sum must be f32 to avoid overflow
4. **Model outputs**: final logits need f32 for sampling

#### Compiler-driven quantization boundaries:
The compiler analyzes the graph and inserts `Quantize`/`Dequantize` nodes at optimal positions:

```rust
GraphOp::Quantize { scheme: QuantScheme, encoding: Encoding }
GraphOp::Dequantize { scheme: QuantScheme, encoding: Encoding }
```

The fusion pass recognizes chains like `Dequantize → float_op → Quantize` and replaces them with the byte-domain LUT equivalent when the op has a LUT implementation.

**Expected impact**: For a LLaMA-style transformer, ~60% of element-wise ops could stay in byte domain. The remaining 40% (norms, residuals, softmax) require f32 reductions but their per-element parts can still use LUT.

### Phase 9: Binary Arithmetic LUT Tables (Q0×Q0)

Extend byte-domain coverage to f32 binary element-wise ops:

#### 9.1: Quantized Binary Arithmetic
For ops where both operands are already in Q0 domain, use 256×256 precomputed tables:

```rust
// Compile-time: generate table for quantized addition
// table[a][b] = encode(decode(a) + decode(b))
static Q0_FLOAT_ADD: [[u8; 256]; 256] = ...;  // 64KB

fn quantized_add(a: u8, b: u8) -> u8 {
    Q0_FLOAT_ADD[a as usize][b as usize]  // O(1) lookup
}
```

Tables needed (64KB each):
- Q0_FLOAT_ADD, Q0_FLOAT_SUB, Q0_FLOAT_MUL, Q0_FLOAT_DIV
- Q0_FLOAT_MIN, Q0_FLOAT_MAX
- Total: 6 × 64KB = 384KB (fits in L2)

**Key constraint**: These tables are **encoding-dependent**. The mapping `byte → float → compute → float → byte` depends on the encoding (angle, signed, unsigned, raw). The compiler selects the encoding per tensor based on value distribution.

#### 9.2: FusedSwiGLU in Byte Domain
SwiGLU = `silu(gate) * up` is a binary op where:
- `gate` comes from a linear layer (can be Q0)
- `up` comes from a linear layer (can be Q0)
- `silu(gate)` is a Q0 LUT
- The multiply is a Q0×Q0 table lookup

Entire SwiGLU in byte domain: 2 LUT lookups (silu + mul), zero f32 compute.

---

## Theoretical LUT Coverage After All Phases

| Domain | Phase 0-3 | + Phase 7 | + Phase 8 | + Phase 9 | + Phase 10 |
|---|---|---|---|---|---|
| Orchestration | 0% LUT → 100% (tape) | 100% | 100% | 100% | 100% |
| Unary activations | Q0 LUT (byte only) | + Q1 for float domain | + auto-quantize | + auto-quantize | **HLUT Q2 precision** |
| Transcendentals | Raw f32 | LUT (exp, sin, cos, erf, rsqrt) | LUT | LUT | **HLUT (2.5x faster)** |
| Binary arithmetic | Raw f32 | Raw f32 | Partial (in Q chains) | **Q0×Q0 tables** | Q0×Q0 tables |
| MatMul | LUT-GEMM (Q4/Q8) | LUT-GEMM | LUT-GEMM | LUT-GEMM | LUT-GEMM |
| Reductions | Raw f32 | Raw f32 | Raw f32 | Raw f32 (irreducible) | Raw f32 (irreducible) |
| Norms (per-element) | Raw f32 | LUT rsqrt | Q1 per-element | Q1 per-element | **HLUT Q2 rsqrt** |
| Norms (reduction) | Raw f32 | Raw f32 | Raw f32 | Raw f32 (irreducible) | Raw f32 (irreducible) |
| Softmax (exp) | Raw f32 | **Q1 LUT** | Q1 LUT | Q1 LUT | **HLUT (28KB, Q2)** |
| Softmax (reduction) | Raw f32 | Raw f32 | Raw f32 | Raw f32 (irreducible) | Raw f32 (irreducible) |
| RoPE | Raw f32 | **Precomputed tables** | Tables | Tables | **HLUT sin/cos** |
| Attention QK^T | LUT-GEMM or BLAS | LUT-GEMM | LUT-GEMM | LUT-GEMM | LUT-GEMM |
| Residual add | Raw f32 | Raw f32 | Raw f32 | Raw f32 (precision) | Raw f32 (precision) |

**Irreducible f32 (cannot become LUT):**
1. Reductions (sum, mean, max, min, prod) — each output depends on ALL inputs
2. Residual additions — f32 accumulation for numerical stability
3. Softmax denominator — sum over exp values
4. Norm mean/variance — global statistics

**Everything else → LUT.**

After Phase 9, the only raw f32 compute remaining is ~5 reduction operations and residual additions. All per-element computation is LUT-based.

---

## Phase 10: Hierarchical Content-Addressable LUT (HLUT)

### The Scaling Problem

Flat tables hit cache boundaries at higher quantum levels:

| Level | Entries | Table Size | Cache Tier | Latency |
|---|---|---|---|---|
| Q0 (8-bit) | 256 | 256B | **L1** | ~1 cycle |
| Q1 (16-bit) | 65,536 | 128KB | **L2** | ~5 cycles |
| Q2 (24-bit) | 16M | ~50MB | L3/RAM | ~30 cycles |
| Q3 (32-bit) | 4B | ~17GB | Infeasible | N/A |

A flat Q2 table is 1000x slower than Q0 due to cache misses. Q3 doesn't fit in memory at all.

### The Solution: 2-Level Content-Addressable Hierarchy

Instead of a flat table, use a **page-selector** (first level) that routes inputs to **pages** (second level) based on output similarity:

```
input (16-24 bit)
  |
  v
[Page Selector: ElementWiseView, 256B, L1]
  input_hi (high 8 bits) → page_id
  |
  v
[Page Table: 256 pages, each variable-size]
  pages[page_id][input_lo] → output
```

**Key insight**: The page selector IS an `ElementWiseView` — hologram's existing 256-byte LUT infrastructure. It composes with view fusion via `.then()`. The entire hierarchical lookup is native to the existing architecture.

### Content-Addressable Routing

The page selector is NOT a simple bit-range split. The compiler builds it using **k-means clustering on the function's output surface**:

1. At compile time: evaluate `f(x)` for all x in the input domain
2. Cluster the 256 output regions by value similarity (k-means, k=256)
3. Assign each input to the page containing its cluster
4. Build the page selector `ElementWiseView` that maps `input_hi → page_id`

Inputs that produce **similar outputs** land on the same page. This gives:
- **Cache locality**: correlated access patterns (e.g., activations in a row) hit the same 2-3 hot pages
- **Adaptive precision**: flat function regions get tiny pages, curved regions get dense pages

### Adaptive Page Sizes

Not all pages need full 256/65536 entries. The compiler determines per-page granularity based on function curvature in that region:

```rust
enum PageKind {
    Constant(f32),           // Flat region: sigmoid(x) ≈ 1.0 for x > 6
    Linear { a: f32, b: f32 }, // Near-linear: output = a * input_lo + b
    Table256([u8; 256]),      // 8-bit resolution within page
    Table65536(Box<[u16; 65536]>), // 16-bit resolution within page
}
```

For sigmoid:
- ~50 pages are `Constant` (tails where sigmoid ≈ 0 or ≈ 1): **0 bytes**
- ~100 pages are `Linear` (gentle slope regions): **8 bytes each**
- ~106 pages need full `Table256` (the steep transition region): **256 bytes each**
- Total: ~28KB instead of 50MB. **1785x compression vs flat Q2.**

### Performance Analysis

**For Q1 with content-addressable routing (vs flat Q1):**
- Flat: random access across 128KB → L2 latency (~5 cycles)
- Hierarchical: selector (L1, ~1 cycle) + hot page (L1 if recently accessed, ~1 cycle) = **~2 cycles**
- Speedup: **2.5x for correlated access patterns** (typical in transformer activations)

**For Q2 (24-bit precision, currently infeasible):**
- Flat: 50MB, mostly L3/RAM misses → ~30 cycles average
- Hierarchical: selector (L1, 1 cycle) + hot pages in L2 (5 cycles) = **~6 cycles**
- With adaptive pages: total memory 2-5MB (fits in L2), mostly L2 hits
- Speedup: **5x vs flat Q2, and actually feasible** (50MB → 2-5MB)

**For Q3 (32-bit, f32-equivalent precision):**
- Flat: 17GB → impossible
- Hierarchical 2-level: selector (L1) + pages (16MB, L3) = ~30 cycles but **actually possible**
- With adaptive pages: sigmoid needs ~500KB for full f32 precision. **34,000x compression.**

### Integration with Existing Infrastructure

The hierarchical LUT composes naturally with hologram's architecture:

1. **Page selector = ElementWiseView**: Already exists, already SIMD-accelerated (AVX2 `vpshufb`), already serializable via rkyv, already fuses via `.then()`
2. **Pages stored as constants in ConstantStore**: The compiler generates pages at quantization time and stores them in the .holo archive
3. **View fusion extends to hierarchical**: `hlut_sigmoid.then(hlut_relu)` could be compiled into a single hierarchical table at compile time
4. **Instruction tape integration**: The kernel_id maps to `kernel_hlut_q2` which reads the page selector and page table from the constant blob

### New Data Structure

```rust
/// Hierarchical LUT with content-addressable page routing.
pub struct HierarchicalLut {
    /// First level: maps input high byte → page_id. This IS an ElementWiseView.
    page_selector: ElementWiseView,
    /// Second level: variable-size pages indexed by page_id.
    pages: Vec<PageKind>,
    /// Input bit split: how many bits go to selector vs page index.
    selector_bits: u8,  // typically 8
    /// Total input precision (8=Q0, 16=Q1, 24=Q2).
    input_bits: u8,
}

impl HierarchicalLut {
    /// O(1) lookup: selector + page access.
    #[inline]
    pub fn lookup(&self, input: u32) -> f32 {
        let hi = (input >> (self.input_bits - self.selector_bits)) as u8;
        let page_id = self.page_selector.apply(hi);
        let lo = input & ((1 << (self.input_bits - self.selector_bits)) - 1);
        self.pages[page_id as usize].lookup(lo as u16)
    }
}
```

### Compile-Time Construction

```rust
/// Build HLUT for a given function at specified precision.
fn build_hlut<F: Fn(f32) -> f32>(
    f: F,
    input_range: (f32, f32),
    input_bits: u8,
    max_page_error: f32,
) -> HierarchicalLut {
    // 1. Evaluate f(x) over full input domain
    // 2. K-means cluster outputs into 256 groups
    // 3. Assign each input to its cluster's page
    // 4. Build page selector ElementWiseView
    // 5. For each page, determine PageKind:
    //    - If max(f) - min(f) < epsilon → Constant
    //    - If linear_fit_error < threshold → Linear
    //    - Otherwise → Table256 or Table65536
    // 6. Return HierarchicalLut
}
```

### Application to Specific Ops

| Op | Input Range | Flat Q2 Size | HLUT Size | Speedup vs Flat Q1 |
|---|---|---|---|---|
| Sigmoid | [-16, 16] | 50MB | ~28KB | 2.5x |
| Exp (softmax) | [-16, 0] | 25MB | ~15KB | 2.5x |
| Tanh | [-8, 8] | 25MB | ~20KB | 2.5x |
| Erf | [-4, 4] | 12MB | ~30KB | 2x |
| Gelu | [-8, 8] | 25MB | ~40KB | 2x |
| Rsqrt | [0.01, 100] | 50MB | ~50KB | 2.5x |
| Sin/Cos | [0, 2π] | 10MB | ~80KB | 1.5x (less compressible) |

**Total HLUT memory for all ops**: ~260KB — less than flat Q1 (2.7MB), with Q2 precision.

### Files
- New: `crates/hologram-core/src/hlut/mod.rs` — HierarchicalLut, PageKind, lookup
- New: `crates/hologram-core/src/hlut/build.rs` — k-means page construction
- Modify: `crates/hologram-core/src/view/mod.rs` — integrate HLUT as an alternative to flat view
- Modify: `crates/hologram-graph/src/graph/mod.rs` — `GraphOp::HLut(HLutId)` variant
- Modify: `crates/hologram-compiler/src/compiler/mod.rs` — HLUT construction during emit
- Modify: `crates/hologram-exec/src/kv/store.rs` — dispatch HLut ops

---

After Phase 9, the only raw f32 compute remaining is ~5 reduction operations and residual additions. All per-element computation is LUT-based.

---

## Implementation Order

```
Phase 0.1  Flat pre-allocated arena (replace HashMap)       → biggest single win for orchestration
Phase 0.2  Output buffer pre-allocation in dispatch          → eliminate per-op Vec allocations
Phase 0.3  Compile-time shape resolution                     → eliminate shape HashMap passes
Phase 0.4  Embed schedule in archive                         → eliminate O(V+E) at load time
Phase 0.5  SmallVec strides + stride memoization             → eliminate broadcast allocations
Phase 0.6  Adaptive parallel threshold                       → cost-based rayon decisions
Phase 1.A  Weight cache (eliminate re-deserialization)        → pure runtime, no format change
Phase 1.B  Column-major weight indices                       → compile-time layout + new kernel
Phase 1.3  Tiled multi-column kernels                        → builds on 1.B layout
Phase 1.4  SIMD dot products (AVX2/NEON/WASM)                → orthogonal, can parallel with 1.3
Phase 2.1  MatMul+Bias+Activation fusion                     → new fusion pass + fused kernel
Phase 2.2  Norm+Activation fusion + fast_rsqrt               → new fusion pass + fused kernel
Phase 2.3  LUT-exp for softmax                               → standalone, feature-gated
Phase 2.4  Buffer alignment                                  → workspace planner change
Phase 3.1  Attention op with baked tile sizes                 → graph IR change
Phase 3.2  Tiled attention kernel                             → new dispatch function
Phase 3.3  Per-head parallelism                               → rayon in attention dispatch
```

### Step 0.7: Instruction Tape Executor (Everything-Is-A-Lookup)

The ultimate expression of the KV/LUT philosophy applied to orchestration. Replace the current executor loop (match on GraphOp, gather inputs via HashMap, resolve shapes) with a flat instruction tape where every step is pre-resolved:

```rust
/// Compiled instruction — every field is a direct index, no runtime resolution.
pub struct Instruction {
    kernel_id: u16,                         // index into kernel function table
    input_slots: SmallVec<[u16; 4]>,        // direct arena slot indices
    output_slot: u16,                       // direct arena slot index
    output_bytes: u32,                      // pre-computed buffer size
    constant_offset: u32,                   // byte offset into weight blob (0 = no constant)
    constant_len: u32,                      // constant byte length
}

/// The kernel table — a dense array of function pointers, one per op type.
type KernelFn = fn(inputs: &[&[u8]], output: &mut [u8], constant: &[u8]) -> ExecResult<()>;
static KERNEL_TABLE: &[KernelFn] = &[
    kernel_lut_sigmoid,      // 0
    kernel_lut_relu,         // 1
    kernel_prim_add,         // 2
    kernel_matmul_lut4,      // 3
    kernel_float_softmax,    // 4
    kernel_float_attention,  // 5
    // ...
];

/// Execution: pure indexed lookups, zero decisions.
fn execute_tape(tape: &[Instruction], arena: &mut FlatArena, weights: &[u8]) {
    for inst in tape {
        let inputs = inst.input_slots.iter()
            .map(|&s| arena.get(s))       // O(1) array index
            .collect::<SmallVec<_>>();
        let output = arena.get_mut(inst.output_slot);  // O(1) array index
        let constant = &weights[inst.constant_offset..][..inst.constant_len];
        KERNEL_TABLE[inst.kernel_id as usize](&inputs, output, constant);  // O(1) table lookup
    }
}
```

This is the compile-time-first philosophy taken to its conclusion:
- **No `match` on GraphOp** at runtime — the compiler resolves each op to a `kernel_id` index
- **No HashMap lookups** — all buffer references are slot indices into the flat arena
- **No shape resolution** — output sizes pre-computed and baked into the instruction
- **No constant deserialization** — direct byte offset into the weight blob
- **The dispatch loop IS a KV lookup**: `kernel_id → function pointer`

The compiler emits the instruction tape during the emit stage. Each `GraphOp` + its resolved shapes + its constant references collapse into a single `Instruction`. Parallel levels become ranges in the tape: `levels[i] = (tape_start, tape_end)`.

**Files**:
- New: `crates/hologram-exec/src/eval/tape.rs` — Instruction struct, tape executor
- New: `crates/hologram-exec/src/eval/kernel_table.rs` — static kernel function table
- Modify: `crates/hologram-compiler/src/compiler/mod.rs` — emit instruction tape
- Modify archive format — embed tape as a flat section

### Step 0.8: System-Level Optimizations

Quick wins that let the LUT/KV core run at maximum throughput:

**a) Cargo release profile** — add to `.cargo/config.toml`:
```toml
[build]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
lto = "thin"  # faster compile, ~same perf as fat LTO
```
Expected: ~10% free perf from native CPU instruction selection (AVX2 auto-vectorization, BMI2).

**b) KV cache lazy initialization** — replace `vec![0.0f32; cap]` with `Vec::with_capacity(cap)`. Don't zero 2GB of cache memory that will be overwritten during prefill.

**c) Graph metadata dense arrays** — replace 3 HashMaps in `Graph` (`constant_shapes`, `node_shapes`, `node_dtypes`) with `Vec<Option<T>>` indexed by NodeId, matching the existing `slots`/`generations` pattern. Every metadata access becomes an O(1) array index.

**d) Archive decompression into aligned buffer** — decompress directly into `AlignedVec<16>` instead of double-copy (decompress → Vec → AlignedVec).

**e) FFI zero-copy inputs** — accept `&[u8]` slices in FFI instead of copying to `Vec<u8>` per input.

**Files**:
- New: `.cargo/config.toml`
- Modify: `crates/hologram-exec/src/kv_cache.rs` — lazy init
- Modify: `crates/hologram-graph/src/graph/mod.rs` — dense Vec metadata
- Modify: `crates/hologram-archive/src/loader/bytes.rs` — direct decompress
- Modify: `crates/hologram-ffi/src/exec/mod.rs` — borrowed inputs

**Phase 0** should be implemented FIRST — it reduces overhead for ALL ops, not just matmul/attention.
Steps 0.1-0.2 are tightly coupled (arena change enables dispatch_into). Do them together.
Steps 0.3-0.6 are independent and can be parallelized.
Step 0.7 (instruction tape) is the capstone of Phase 0 — it subsumes 0.1-0.3 into a unified design.
  Implement 0.1-0.3 first as incremental steps, then refactor into 0.7 tape format.
Step 0.8 (system-level) can be done anytime — quick wins, no dependencies.
Phase 1 builds on Phase 0 (weight cache uses flat arena; tiled kernels use dispatch_into).
Phases 2-3 are independent of each other but benefit from Phase 0's reduced overhead.

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

| Component | Current | After Phase 0 | After Phase 1 | After Phase 2 | After Phase 3 |
|---|---|---|---|---|---|
| Arena lookup | ~50ns HashMap | ~2ns array index | ~2ns | ~2ns | ~2ns |
| Per-op allocation | 1-7 Vecs/op | 0 (pre-allocated) | 0 | 0 | 0 |
| Shape resolution | 4-8 HashMap/node | 0 (compiled) | 0 | 0 | 0 |
| Schedule build | O(V+E) at load | 0 (embedded) | 0 | 0 | 0 |
| Op dispatch | match 22 arms | fn ptr table[id] | table[id] | table[id] | table[id] |
| Graph metadata | HashMap×3/node | array[id] | array[id] | array[id] | array[id] |
| Weight deser | ~100μs/call | ~100μs | ~0 (cached) | ~0 | ~0 |
| LUT-GEMM per elem | ~20 cy (L2 miss) | ~20 cy | ~3 cy (L1+SIMD) | ~2.5 cy (fused) | ~2.5 cy |
| Softmax exp | ~5 cy/element | ~5 cy | ~5 cy | ~1 cy (LUT) | ~1 cy |
| Attention scores | O(seq²×d) | O(seq²×d) | O(seq²×d) | O(seq²×d) | tiled, 50% skip |
| Norm + activation | 2 passes | 2 passes | 2 passes | 1 pass | 1 pass |

**End-to-end estimate**: Phase 0 alone delivers **2-3x** for graphs with many small ops (the orchestration overhead is currently >50% of runtime for elementwise-heavy graphs). Step 0.7 (instruction tape) + 0.8 (system flags) adds another ~10-15%. Combined with Phases 1-3: **5-10x** for a 7B model at seq=2048.

The key insight: after all phases, the runtime becomes **a flat loop of indexed lookups** — arena reads are array indices, dispatch is a function pointer table, shapes are pre-resolved, constants are byte offsets. The entire execution path is KV/LUT-native, not just the compute kernels.
