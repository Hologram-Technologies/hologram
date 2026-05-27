# Hologram Operations Reference

This document catalogs the operations available in the hologram runtime and their
semantics. The provenance of an op in v0.5.0 is:

- **`OpKind`** (`crates/hologram-ops/src/kind.rs`, re-exported as
  `hologram_graph::OpKind`) — the closed canonical op catalog. This is the graph IR op
  set. Each canonical op is a marker type in `hologram-ops` plus a const-tagged IRI plus
  an `emit_term` function; the Term tree it emits is the formal specification, and
  per-op reference evaluators verify the backend kernels against it.
- **`KernelCall`** (`crates/hologram-backend/src/kernel_call.rs`) — the lowered,
  pre-resolved dispatch enum. `hologram-compiler` lowers each `OpKind` graph node into
  one or more `KernelCall`s; the CPU backend dispatches them by an exhaustive match
  (`crates/hologram-backend/src/cpu.rs`). `KernelCall` variants carry the shape
  parameters needed for dispatch since the graph IR has no per-edge shape metadata.

Execution runs through the content-addressed `InferenceSession`
(`crates/hologram-exec/src/session.rs`) over a `BufferArena` pool — there is no
`KvExecutor` and no tape.

> **Scope note.** The canonical `OpKind` catalog is closed (see
> `crates/hologram-ops/src/kind.rs`). Some operations described in the tables below are
> *not* present in `hologram_graph::OpKind` in v0.5.0 — notably `Gather`, `GatherND`,
> `Cast`, `Embed`, `Range`, `Shape`, `ScatterND`, `TopK`, `NonZero`, `Compress`,
> `ReverseSequence`, and the KV-cache ops `KvWrite` / `KvRead`. Their semantic
> descriptions are retained here for reference, but they are not part of the current
> canonical op set.

---

## Float-domain tensor operations

### Arithmetic (binary, element-wise with broadcast)

All arithmetic ops take 2 inputs (f32) and broadcast the shorter operand.

| Variant | Semantics |
|---------|-----------|
| `Add` | `out[i] = a[i] + b[i % b.len()]` |
| `Sub` | `out[i] = a[i] - b[i % b.len()]` |
| `Mul` | `out[i] = a[i] * b[i % b.len()]` |
| `Div` | `out[i] = a[i] / b[i % b.len()]` |
| `Pow` | `out[i] = a[i] ^ b[i % b.len()]` |
| `Mod` | `out[i] = a[i] % b[i % b.len()]` |
| `Min` | `out[i] = min(a[i], b[i % b.len()])` |
| `Max` | `out[i] = max(a[i], b[i % b.len()])` |

### Unary Activations

All activation ops take 1 input (f32).

| Variant | Semantics |
|---------|-----------|
| `Neg` | `out[i] = -x[i]` |
| `Relu` | `out[i] = max(0, x[i])` |
| `Gelu` | `out[i] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` (approximate) |
| `Silu` | `out[i] = x * sigmoid(x)` (Swish) |
| `Tanh` | `out[i] = tanh(x[i])` |
| `Sigmoid` | `out[i] = 1 / (1 + exp(-x[i]))` |
| `Exp` | `out[i] = exp(x[i])` |
| `Log` | `out[i] = ln(x[i])` |
| `Sqrt` | `out[i] = sqrt(x[i])` |
| `Abs` | `out[i] = \|x[i]\|` |
| `Reciprocal` | `out[i] = 1 / x[i]` |

### Unary Math

All take 1 input (f32) unless noted.

| Variant | Semantics | Notes |
|---------|-----------|-------|
| `Cos` | `out[i] = cos(x[i])` | |
| `Sin` | `out[i] = sin(x[i])` | |
| `Sign` | `out[i] = -1, 0, or 1` | |
| `Floor` | `out[i] = floor(x[i])` | |
| `Ceil` | `out[i] = ceil(x[i])` | |
| `Round` | `out[i] = round(x[i])` | Nearest |
| `Erf` | `out[i] = erf(x[i])` | Abramowitz & Stegun approximation |
| `Clip { min, max }` | `out[i] = clamp(x[i], min, max)` | `min`/`max` stored as `f32::to_bits()` |
| `IsNaN` | `out[i] = x[i].is_nan() as u8` | Output is u8 (0 or 1) |

### Boolean / Comparison

Boolean ops operate on byte buffers (nonzero = true). Comparison ops interpret inputs as
f32 and produce u8 output.

| Variant | Arity | Semantics |
|---------|-------|-----------|
| `And` | 2 | Logical AND (byte-wise) |
| `Or` | 2 | Logical OR |
| `Xor` | 2 | Logical XOR |
| `Not` | 1 | Logical NOT |
| `Equal` | 2 | `a == b` → u8 |
| `Less` | 2 | `a < b` → u8 |
| `LessOrEqual` | 2 | `a <= b` → u8 |
| `Greater` | 2 | `a > b` → u8 |
| `GreaterOrEqual` | 2 | `a >= b` → u8 |

### Linear Algebra

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `MatMul { m, k, n }` | 2: A (f32), B (f32) | `m`, `k`, `n` — matrix dimensions | `[m, k] × [k, n] → [m, n]`, both row-major |
| `Gemm { m, k, n, alpha, beta, trans_a, trans_b, quant_b }` | 2–3: A (f32), B (f32/quantized), C (f32, optional) | `alpha`/`beta` as f32 bits; `trans_a`/`trans_b` transpose flags; `quant_b`: 0=none, 1=Q4_0, 2=Q8_0 | `out = alpha * op(A) × op(B) + beta * C` |

### Softmax

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `Softmax { size }` | 1 (f32) | `size` — row length | Softmax along last `size` elements of each row |
| `LogSoftmax { size }` | 1 (f32) | `size` — row length | LogSoftmax along last `size` elements of each row |

### Normalization

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `RmsNorm { size, epsilon }` | 2: x (f32), weight (f32) | `size` — norm dimension; `epsilon` as f32 bits | RMS normalization |
| `AddRmsNorm { size, epsilon }` | 3: x (f32), residual (f32), weight (f32) | Same as RmsNorm | Fused `rmsnorm(x + residual, weight, ε)` — eliminates intermediate buffer |
| `LayerNorm { size, epsilon }` | 3: x (f32), weight (f32), bias (f32) | Same as RmsNorm | Layer normalization |
| `InstanceNorm { size, epsilon }` | 2: x (f32), weight (f32) | Same as RmsNorm | Instance normalization (per-channel, spatial) |
| `LRN { size, alpha, beta, bias }` | 1 (f32) | All params as f32 bits | Local response normalization |

### Reductions

All take 1 input (f32). Reduce along the last `size` elements of each row.

| Variant | Semantics |
|---------|-----------|
| `ReduceSum { size }` | Sum reduction |
| `ReduceMean { size }` | Mean reduction |
| `ReduceMax { size }` | Max reduction |
| `ReduceMin { size }` | Min reduction |
| `ReduceProd { size }` | Product reduction |

### Shape Manipulation

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `Gather { dim, dtype }` | 2: data, indices (i64) | `dim` — gather dimension; `dtype` — element type of data | Gather rows by index |
| `GatherND` | 2: data, indices | — | N-D gather (stub: pass-through) |
| `Concat { size_a, size_b, dtype }` | 2: a, b | `size_a`/`size_b` — row sizes; `dtype` — element type | Concatenate along an axis |
| `Reshape` | 1 | — | Pass-through (shape is metadata only, bytes unchanged) |
| `Transpose { perm, ndim }` | 1 | `perm: [u8; 8]` — first `ndim` entries are valid permutation indices | Physical data permutation |
| `Cast { from, to }` | 1 | Source and target `FloatDType` | Type cast |
| `Embed { dim, quant }` | 2: token_ids (u32), table (f32/quantized) | `dim` — embedding dimension; `quant`: 0=none, 1=Q4_0, 2=Q8_0 | Embedding lookup; table is `[vocab, dim]`, output is `[len(ids), dim]` |
| `Where` | 3: cond (u8), x (f32), y (f32) | — | Conditional selection |
| `Range` | 3: start, limit, delta (f32) | — | Generate `[start, limit)` with step |
| `Shape { dtype, start, end }` | 1 | `start`/`end` — dim slice range (opset 15+); negatives count from end | Extract shape as i64 tensor |
| `Slice { axis_from_end, start, end }` | 1 | `axis_from_end` counts backward (1 = last axis); `start`/`end` — element range | Contiguous slice along a single axis |

### Fused Ops

| Variant | Inputs | Semantics |
|---------|--------|-----------|
| `FusedSwiGLU` | 2: gate (f32), up (f32) | `out = silu(gate) * up` — fused SiLU gating |

### Attention & Position Encoding

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `RotaryEmbedding { dim, base, n_heads }` | 1 (f32) | `dim` — embedding dimension; `base` — frequency base (f32 bits); `n_heads` — heads per token | RoPE; position = chunk_index / n_heads |
| `Attention { head_dim, num_q_heads, num_kv_heads, scale, causal, heads_first, qk_norm, rope, rope_base }` | 3: Q, K, V (f32) | `scale` as f32 bits; `causal` — causal mask; `heads_first` — true: `[n_heads, seq, head_dim]` (ONNX), false: `[seq, n_heads, head_dim]` (GGUF); `qk_norm` — RMSNorm on Q/K (Qwen-style); `rope`/`rope_base` — fused RoPE | Scaled dot-product attention (multi-head / grouped-query) |

### Quantization

| Variant | Inputs | Semantics |
|---------|--------|-----------|
| `Dequantize` | 1 (Q4_0) | Dequantize Q4_0 → f32 |

### Vision / Spatial

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `Conv2d { kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, input_h, input_w }` | 2–3: data (f32), weight (f32), bias (f32, optional) | Kernel size, strides, pads, dilations, groups, input spatial dims | 2-D convolution |
| `ConvTranspose { kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, output_pad_h, output_pad_w, input_h, input_w }` | 2–3: same as Conv2d | Same as Conv2d plus `output_pad_h`/`output_pad_w` | 2-D transposed convolution |
| `MaxPool2d { kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w }` | 1 (f32) | Kernel size, strides, pads | 2-D max pooling |
| `AvgPool2d { kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w }` | 1 (f32) | Same as MaxPool2d | 2-D average pooling |
| `GlobalAvgPool` | 1 (f32) | — | Spatial dims → 1 |
| `Resize { mode }` | 1 (f32) | `mode`: u8 encoding of nearest/linear/cubic | Spatial resize |
| `PadOp { mode }` | 1 (f32) | `mode`: 0=constant, 1=reflect, 2=edge | N-D padding |

### Utility

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `TopK { axis, largest }` | 2: data, K (i64) | `axis` — reduction axis; `largest` — descending if true | Top-K along an axis |
| `ScatterND` | 3: data, indices, updates | — | Scatter updates by N-D index |
| `CumSum { axis }` | 1 (f32) | `axis` — cumulation axis | Cumulative sum |
| `NonZero` | 1 | — | Indices of non-zero elements |
| `Compress { axis }` | 2: data, condition | `axis` — compression axis | Compress along an axis |
| `ReverseSequence { batch_axis, time_axis }` | 1 | `batch_axis`, `time_axis` | Reverse along time axis per batch |

### KV Cache

| Variant | Inputs | Parameters | Semantics |
|---------|--------|------------|-----------|
| `KvWrite { layer, n_kv_heads, head_dim, is_key }` | 1: tensor (f32) | `layer` — transformer layer index; `is_key` — true for K, false for V | Write K or V tensor into KV cache; output is pass-through (or full cached tensor in decode) |
| `KvRead { layer, n_kv_heads, head_dim }` | 0 (state-only) | `layer` — transformer layer index | Read full cached K/V from position 0 to current write position |

---

## KernelCall — Lowered dispatch kernels

`hologram-compiler` lowers each graph `OpKind` node into one or more `KernelCall`
variants at compile time (`crates/hologram-backend/src/kernel_call.rs`). The CPU backend
matches on this enum exhaustively (`crates/hologram-backend/src/cpu.rs`) and calls the
appropriate kernel directly, eliminating vtable indirection and HashMap lookups.

The variant names and groupings below describe the v0.5.0 lowering. (The historical
byte-domain `PrimOp` / `LutOp` and the pre-resolved `TapeKernel` tape have been removed;
dispatch is now the single `KernelCall` match.)

### Concrete-op variants

`KernelCall` is largely flat: it has one variant per canonical `OpKind` op, carrying the
baked shape/parameter payload that op needs (e.g. `MatMul { m, k, n }`,
`Softmax { size }`, `RmsNorm { size, epsilon }`, `Conv2d { … }`, `Attention { … }`).
The CPU backend's exhaustive match calls the corresponding kernel directly — elementwise
unary/binary ops route through monomorphic fast paths, and structured ops (Gemm,
Attention, Conv2d, the normalizations, etc.) call their dedicated kernels. There is no
catch-all "generic float" arm and no per-op vtable; the match is closed.

### Fusion variants

Beyond the one-per-op variants, `KernelCall` carries fused kernels that the compiler
emits to elide intermediate buffers. These are content-addressed (κ-labelled) fusions:

| Variant | Semantics |
|---------|-----------|
| `BroadcastBinary` | Expand → elementwise-binary fused into one zero-movement pass (no materialized broadcast) |
| `MatMulActivation` | MatMul immediately followed by an activation, fused — elides the intermediate |
| `MatMulAdd` | MatMul + bias add, fused |
| `MatMulAddActivation` | MatMul + bias add + activation, fused |
| `MatMulDequant` | Dequantize → MatMul fused — elides the dense f32 weight |
| `DequantActivation` | Dequantize → activation fused |

---

## Dispatch architecture

`hologram-compiler` lowers the `OpKind` graph (after its fusion/elision passes) into a
flat sequence of `KernelCall`s. At execution time the `InferenceSession` drives the
backend's `dispatch(&KernelCall, &mut WS)`, which is a single closed match in
`crates/hologram-backend/src/cpu.rs`:

1. Elementwise unary/binary ops dispatch to monomorphic kernels (with broadcast).
2. Comparison/boolean ops dispatch to their comparison kernels.
3. Structured ops (Gemm, Attention, Conv2d, the normalizations, pooling, …) call their
   dedicated kernels.
4. Fusion variants run their single combined kernel.

Because the match is exhaustive over a closed enum, every `KernelCall` is handled at
compile time — there is no fallback arm and no runtime "unhandled op" path. The GPU
backends (`metal`, `wgpu`, gated by Cargo feature) implement the same `dispatch` match.

---

## FloatDType — Element types

Used by dtype-aware ops (`Cast`, `Shape`, `Gather`, `Concat`). Stored in `.holo` archives
and must remain rkyv-serializable with `#[repr(u8)]` encoding.

| Variant | Value | Byte size |
|---------|-------|-----------|
| `F32` | 0 | 4 |
| `F64` | 1 | 8 |
| `I32` | 2 | 4 |
| `I64` | 3 | 8 |
| `F16` | 4 | 2 |
| `BF16` | 5 | 2 |
| `U8` | 6 | 1 |
| `Bool` | 7 | 1 |
| `I8` | 8 | 1 |
