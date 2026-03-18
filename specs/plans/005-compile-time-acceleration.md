# Hologram Runtime: Compile-Time-First Acceleration Plan (v2)

## Context

Hologram's core thesis: O(1) per-element dispatch via LUT/KV lookups. **Verified and validated** — unary ops, view fusion, binary ops, and dispatch routing are all genuinely O(1). The remaining bottlenecks (LUT-GEMM stride-N cache misses, per-dispatch weight deserialization, unfused linear layers, scalar softmax/attention) should be attacked by pushing as much work as possible into compile time. The compiler knows weight dimensions, quantization format, and graph structure — runtime should execute pre-planned instructions with zero decision-making.

**Unifying principle**: ALL computation — including orchestration, shape resolution, dispatch routing, and buffer management — should reduce to KV/LUT lookups. The compiler converts decisions into static tables; the runtime reads them. No HashMaps, no pattern matching, no shape inference at runtime. The executor becomes a flat instruction-tape reader where every step is a pre-resolved indexed lookup.

**Scope**: Phases 0-3 implement now. Phases 4-6 roadmap.
**SIMD targets**: x86_64 AVX2, aarch64 NEON, WASM SIMD (all three).

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
