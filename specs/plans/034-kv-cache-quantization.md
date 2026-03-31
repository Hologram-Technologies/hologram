# Plan 034: KV Cache Quantization — Asymmetric Compression with Boundary Protection

## Context

The hologram runtime's KV cache stores all key/value tensors as f32, making it the
single largest memory bottleneck for long-context inference. Research on asymmetric
KV compression demonstrates that keeping K at high precision while aggressively
compressing V achieves 3.8–6.4× compression with negligible quality loss.

The key insight: K errors propagate exponentially through softmax (attention routing),
while V errors are linear (weighted sum). V is "remarkably robust to quantization."

Additionally, boundary layers (first/last 2) are disproportionately sensitive to
quantization. Protecting them at full precision costs ~12% of cache but prevents the
worst quality degradation cases.

## Current State

- KV cache: `KvCacheState` with `Vec<Vec<f32>>` per layer (`kv_cache.rs`)
- Weight quantization: Q4/Q8/Q16 k-means + orbit maps (LUT-GEMM pipeline)
- F16 activation compression: infrastructure built, not yet wired (`arena.rs`)
- Attention: BLAS + online softmax, GQA, RoPE, causal masking (`attention.rs`)

## Implementation

### Phase 1: KV Cache Compression Config + Boundary Layer Protection

Add a `KvCacheConfig` that controls per-layer compression strategy:

```
KvCacheConfig {
    k_bits: KvBits,         // F32 | Q8
    v_bits: KvBits,         // F32 | Q8 | Q4
    boundary_layers: usize, // Layers at start/end kept at f32 (default 2)
    n_layers: u32,          // Total layers (for boundary calculation)
}

enum KvBits { F32, Q8, Q4 }
```

Boundary layers (0, 1, N-2, N-1) always store f32 regardless of config.

**Files**: `crates/hologram-exec/src/kv_cache.rs`

### Phase 2: Per-Channel Min/Max Quantization for KV Cache

Weight quantization uses k-means (expensive, offline). KV cache quantization must
be online — quantize on every write. Use per-channel min/max affine quantization:

```
quantize(x, bits):
    min, max = channel_min_max(x)
    scale = (max - min) / (2^bits - 1)
    zero_point = round(-min / scale)
    indices = round((x - min) / scale)
    return (indices, scale, zero_point)

dequantize(indices, scale, zero_point):
    return (indices - zero_point) * scale
```

Per-channel = per KV head. Each head gets its own scale/zero_point per token position.

Storage layout per layer:
- Q8: `Vec<u8>` indices + `Vec<(f32, f32)>` (scale, zero_point) per head×position
- Q4: `Vec<u8>` packed indices (2 per byte) + same scales

**Files**: `crates/hologram-exec/src/kv_cache.rs` (new `QuantizedKvBuffer` type)

### Phase 3: Walsh-Hadamard Pre-Rotation

Apply O(d log d) structured rotation before V quantization to Gaussianize the
distribution. This makes min/max quantization more efficient (fewer outliers wasting
dynamic range).

```
walsh_hadamard_rotate(x, head_dim):
    signs = deterministic_signs(head_dim)  // Fixed per model
    x *= signs                              // Random sign flip
    fwht_inplace(x, head_dim)             // Fast Walsh-Hadamard (butterfly)
    x *= signs                              // Second sign flip
    x /= sqrt(head_dim)                    // Normalize
```

Inverse rotation applied on dequantize (WHT is self-inverse up to scaling).

**Files**: `crates/hologram-exec/src/kv_cache.rs` (inline, ~50 lines)

### Phase 4: Integration with Tape Executor

Modify `dispatch_kv_write` and `dispatch_kv_read` in `tape.rs` to:
- On write: quantize V (and optionally K) before storing
- On read: dequantize before returning to attention kernel
- Boundary layers: pass through as f32

The KV cache config is set at `KvCacheState` construction time (model loading).

**Files**: `crates/hologram-exec/src/tape.rs` (KvWrite/KvRead dispatch)

## Memory Impact

| Config | Per-token per-layer (32 heads × 128 dim) | vs f32 |
|--------|------------------------------------------|--------|
| f32/f32 | 32,768 bytes | 1.0× |
| f32/q8 | 20,480 bytes (K f32 + V q8+scales) | 0.63× |
| q8/q4 | 10,240 bytes | 0.31× |
| f32/q4 (recommended) | 18,432 bytes | 0.56× |

With boundary protection (4 layers f32, 28 layers q4): ~0.61× total cache size.

## Verification

- Unit tests: quantize→dequantize round-trip within tolerance
- WHT tests: rotation is self-inverse (WHT(WHT(x)) ≈ x)
- Boundary layer tests: layers 0,1,N-2,N-1 remain f32
- Integration test: KvCacheState write→read with quantization matches f32 within tolerance
- PPL regression: TinyLlama end-to-end (existing test infrastructure)
- Memory profiling: RSS comparison at 2K, 8K, 32K context lengths
