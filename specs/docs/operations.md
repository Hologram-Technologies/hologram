# Hologram Operations Reference

This document catalogs every operation available in the hologram runtime. Operations are
defined at two levels:

- **`FloatOp`** (`hologram-core/src/op/float_op.rs`) — typed tensor operations for AI
  inference, operating on f32 buffers with shape-aware semantics. Each variant carries the
  shape parameters needed for dispatch since the graph IR has no per-edge shape metadata.
- **`TapeKernel`** (`hologram-exec/src/tape.rs`) — pre-resolved dispatch kernels. The tape
  builder maps graph ops to these variants at compile time, eliminating vtable indirection
  and enabling inlining.

Lower-level byte-domain ops (`PrimOp`, `LutOp`) operate on Z/256Z ring arithmetic and
byte-domain activation tables respectively — they are not covered here.

---

## FloatOp — Float-domain tensor operations

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

## TapeKernel — Pre-resolved dispatch kernels

The tape builder resolves each graph operation to a `TapeKernel` variant at compile time.
The executor matches on this enum and calls the appropriate kernel directly, eliminating
vtable indirection and HashMap lookups.

### Generic dispatch

| Variant | Semantics |
|---------|-----------|
| `Float(FloatOp)` | Dispatched via `dispatch_float_into` — the general-purpose path for all `FloatOp` variants |
| `FusedFloatChain(Vec<FloatOp>)` | Fused chain of unary float ops — applies multiple ops in a single pass without intermediate buffers |
| `Output` | Graph output passthrough — marks a tape slot as a final output |

### Byte-domain ops

| Variant | Semantics |
|---------|-----------|
| `LutView(ElementWiseView)` | 256-byte lookup table — byte-domain activation |
| `PrimUnary(ElementWiseView)` | Unary primitive via LUT — byte-domain Z/256Z ring |
| `PrimBinary(PrimOp)` | Binary primitive — byte-domain Z/256Z ring |

### Quantized matmul

| Variant | Parameters | Semantics |
|---------|------------|-----------|
| `MatMulLut4(ConstantId)` | Constant ID of weight table | 4-bit quantized LUT-GEMM matmul |
| `MatMulLut8(ConstantId)` | Constant ID of weight table | 8-bit quantized LUT-GEMM matmul |

### KV cache

| Variant | Parameters | Semantics |
|---------|------------|-----------|
| `KvWrite { layer, n_kv_heads, head_dim, is_key }` | Transformer layer index, head config | Write K/V to autoregressive cache |
| `KvRead { layer, n_kv_heads, head_dim }` | Transformer layer index, head config | Read cached K/V for autoregressive generation |

### Inline hot ops (Phase 9a)

These skip the backend vtable and `dispatch_float_into` entirely. The execute loop calls
the kernel function directly — zero dispatch overhead.

**Inline unary:**

| Variant | Semantics |
|---------|-----------|
| `InlineRelu` | `v.max(0.0)` |
| `InlineNeg` | `-v` |
| `InlineAbs` | `v.abs()` |
| `InlineSigmoid` | `1/(1+exp(-v))` |
| `InlineSilu` | `v * sigmoid(v)` |
| `InlineTanh` | `tanh(v)` |
| `InlineGelu` | GELU (approximate) |
| `InlineExp` | `exp(v)` |
| `InlineReciprocal` | `1.0 / v` |

**Inline binary:**

| Variant | Semantics |
|---------|-----------|
| `InlineAdd` | `a + b` |
| `InlineMul` | `a * b` |
| `InlineSub` | `a - b` |
| `InlineDiv` | `a / b` |

### Inline custom ops (Phase 9a.3–9a.4)

Skip the `dispatch_float_into` → `dispatch_custom_into` indirection. Still try backend
(Metal GPU) first, then fall back to direct CPU kernel call.

| Variant | Parameters | Semantics |
|---------|------------|-----------|
| `InlineMatMul { m, k, n }` | Baked matrix dimensions | MatMul with zero-overhead dispatch |
| `InlineSoftmax { size }` | Baked row size | Softmax with zero-overhead dispatch |
| `InlineRmsNorm { size, epsilon }` | Baked row size and epsilon (f32 bits) | RmsNorm with zero-overhead dispatch |

### Custom extension

| Variant | Semantics |
|---------|-----------|
| `Custom(CustomHandler)` | Registry-based handler baked at tape build time — for user-defined ops registered via `CustomOpRegistry` |

---

## Dispatch Architecture

### FloatOp → TapeKernel resolution

At tape build time, `resolve_float_kernel()` (`tape_builder.rs`) maps each `FloatOp` to
a `TapeKernel` variant. The mapping determines which dispatch tier handles each op at
execution time:

```
FloatOp variant
    │
    ▼
resolve_float_kernel()
    ├──▶ Inline hot ops (13)      ──▶ direct kernel call, no backend
    ├──▶ Inline custom ops (3)    ──▶ try GPU backend, then CPU kernel
    ├──▶ KvWrite / KvRead (2)     ──▶ dedicated KV cache dispatch
    └──▶ Float(op) catch-all (60+)──▶ backend → dispatch_float_into → category dispatch
```

### Tier 1: Inline hot ops (13 ops)

Skip the backend vtable and `dispatch_float_into` entirely. The execute loop calls the
kernel closure directly — zero dispatch overhead.

| FloatOp | TapeKernel | Type |
|---------|-----------|------|
| `Relu` | `InlineRelu` | unary |
| `Neg` | `InlineNeg` | unary |
| `Abs` | `InlineAbs` | unary |
| `Sigmoid` | `InlineSigmoid` | unary |
| `Silu` | `InlineSilu` | unary |
| `Tanh` | `InlineTanh` | unary |
| `Gelu` | `InlineGelu` | unary |
| `Exp` | `InlineExp` | unary |
| `Reciprocal` | `InlineReciprocal` | unary |
| `Add` | `InlineAdd` | binary |
| `Mul` | `InlineMul` | binary |
| `Sub` | `InlineSub` | binary |
| `Div` | `InlineDiv` | binary |

These are the most frequent ops in transformer inference — they appear hundreds of times
per forward pass. Phase 9a benchmarks showed ~36% speedup on Relu (5.1µs → 3.3µs for
64KB buffers).

### Tier 2: Inline custom ops (3 ops)

Bake parameters at build time to skip `dispatch_float_into` → `dispatch_custom_into`
indirection, but still try the GPU backend (Metal/WebGPU) first before falling back to
the CPU kernel.

| FloatOp | TapeKernel |
|---------|-----------|
| `MatMul { m, k, n }` | `InlineMatMul { m, k, n }` |
| `Softmax { size }` | `InlineSoftmax { size }` |
| `RmsNorm { size, epsilon }` | `InlineRmsNorm { size, epsilon }` |

These ops are hot (multiple times per transformer layer) but benefit from GPU
acceleration for large tensors, so the backend check is preserved.

### Tier 3: Generic `Float(op)` (~60+ ops)

All remaining `FloatOp` variants use the catch-all `_ => TapeKernel::Float(*fop)`. At
execution time, this path:

1. Tries the GPU backend (`backend.dispatch_float()`)
2. Falls back to `dispatch_float_into()`, which routes by category:
   - `UnaryElementwise` → `elementwise::unary_map()` with monomorphic fast paths
   - `BinaryElementwise` → `elementwise::binary_elementwise()` with broadcast
   - `BinaryCompare` / `BinaryByteBool` / `UnaryByteBool` → comparison kernels
   - `Custom` → dedicated dispatch functions (Gemm, Attention, Conv2d, etc.)
3. Ultimate fallback: `op.apply_unary(v)` / `op.apply_binary(a, b)`

**Every FloatOp variant is handled** — the catch-all ensures no op is ever unhandled or
panics at dispatch time.

### Why only 16 ops are inlined

Adding inline variants for the remaining ~60 ops would not help and could hurt:

- **Enum bloat**: More `TapeKernel` variants increase match table size, degrading
  instruction cache performance in the hot dispatch loop.
- **Complex ops need GPU**: Conv2d, Attention, and Gemm *must* try the GPU backend —
  skipping it would be a major regression on Metal/WebGPU-capable hardware.
- **Shape ops are near-zero-cost**: Reshape is a no-op (metadata only). Transpose,
  Gather, and Cast are memory moves — dispatch overhead is noise compared to actual
  data movement.
- **Rare ops have negligible total impact**: Cos, Sign, Erf, and similar math ops
  appear at most once per layer. Even saving 2µs per call yields < 2µs total per
  forward pass — not worth the enum bloat.

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
