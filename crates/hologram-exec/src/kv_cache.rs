//! KV cache state for autoregressive LLM generation.
//!
//! Holds per-layer K and V buffers that persist across execution calls.
//! During prefill, the full prompt's K/V are written. During decode,
//! each new token's K/V are appended and the full cache is returned
//! for attention computation.
//!
//! Supports optional asymmetric quantization: K and V can independently be
//! stored at f32, q8, or q4 precision. Boundary layers (first/last N) are
//! always kept at f32 regardless of config, as they are disproportionately
//! sensitive to quantization error.

// ── Configuration ────────────────────────────────────────────────────

/// Bit-width for KV cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBits {
    /// Full precision (4 bytes per element).
    F32,
    /// 8-bit quantized (1 byte per element + per-head scales).
    Q8,
    /// 4-bit quantized (0.5 bytes per element + per-head scales).
    Q4,
}

/// Configuration for KV cache quantization.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Bit-width for K (key) cache. Default: F32.
    pub k_bits: KvBits,
    /// Bit-width for V (value) cache. Default: F32.
    pub v_bits: KvBits,
    /// Number of boundary layers (at start and end) kept at f32.
    /// Default: 2 (layers 0, 1, N-2, N-1 stay f32).
    pub boundary_layers: usize,
    /// Whether to apply Walsh-Hadamard rotation before V quantization.
    /// Gaussianizes the distribution for better quantization efficiency.
    pub wht_rotation: bool,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            k_bits: KvBits::F32,
            v_bits: KvBits::F32,
            boundary_layers: 2,
            wht_rotation: false,
        }
    }
}

impl KvCacheConfig {
    /// Asymmetric config: K at f32, V at q4 with WHT rotation and boundary protection.
    /// Best quality at q4; WHT Gaussianizes V distributions for lower quantization error.
    #[must_use]
    pub fn asymmetric_q4() -> Self {
        Self {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q4,
            boundary_layers: 2,
            wht_rotation: true,
        }
    }

    /// Asymmetric config: K at f32, V at q4 without WHT rotation.
    /// ~13× faster read than WHT variant. Recommended when V values are already
    /// well-distributed (most trained LLMs) and max quality isn't critical.
    #[must_use]
    pub fn asymmetric_q4_fast() -> Self {
        Self {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q4,
            boundary_layers: 2,
            wht_rotation: false,
        }
    }

    /// Returns true if the given layer should be stored at f32 regardless of config.
    #[must_use]
    pub fn is_boundary_layer(&self, layer: u32, n_layers: u32) -> bool {
        let b = self.boundary_layers as u32;
        layer < b || layer >= n_layers.saturating_sub(b)
    }

    /// Returns the effective bit-width for K at a given layer.
    #[must_use]
    pub fn effective_k_bits(&self, layer: u32, n_layers: u32) -> KvBits {
        if self.is_boundary_layer(layer, n_layers) {
            KvBits::F32
        } else {
            self.k_bits
        }
    }

    /// Returns the effective bit-width for V at a given layer.
    #[must_use]
    pub fn effective_v_bits(&self, layer: u32, n_layers: u32) -> KvBits {
        if self.is_boundary_layer(layer, n_layers) {
            KvBits::F32
        } else {
            self.v_bits
        }
    }
}

// ── Per-channel affine quantization ──────────────────────────────────

/// Per-channel scale and zero-point for affine quantization.
#[derive(Debug, Clone, Copy)]
struct ChannelParams {
    scale: f32,
    zero_point: f32,
}

/// Quantize a single channel (one head at one position) to q8.
/// Returns params. `out` must have len == `data.len()`.
#[inline]
fn quantize_channel_q8(data: &[f32], out: &mut [u8]) -> ChannelParams {
    // Fused min/max in a single pass (4-wide manual unroll for autovec).
    let n = data.len();
    let (mut min, mut max) = (data[0], data[0]);
    let chunks = n / 4;
    let base = data.as_ptr();
    for c in 0..chunks {
        let off = c * 4;
        unsafe {
            let a = *base.add(off);
            let b = *base.add(off + 1);
            let c = *base.add(off + 2);
            let d = *base.add(off + 3);
            let lo = a.min(b).min(c.min(d));
            let hi = a.max(b).max(c.max(d));
            min = min.min(lo);
            max = max.max(hi);
        }
    }
    for &v in &data[chunks * 4..n] {
        min = min.min(v);
        max = max.max(v);
    }

    // Degenerate: all values identical.
    if (max - min).abs() < f32::EPSILON {
        out.iter_mut().for_each(|b| *b = 0);
        return ChannelParams {
            scale: 1.0,
            zero_point: -min,
        };
    }
    let scale = (max - min) / 255.0;
    let inv_scale = 255.0 / (max - min);
    let bias = -min * inv_scale + 0.5; // +0.5 replaces .round() with truncation

    // Quantize: fused multiply-add + truncate (avoids round() per element).
    for i in 0..n {
        let q = (data[i] * inv_scale + bias) as i32;
        out[i] = q.clamp(0, 255) as u8;
    }
    ChannelParams {
        scale,
        zero_point: -min * (1.0 / scale),
    }
}

/// Quantize a single channel to q4 (16 levels).
/// `out` must have len == `data.len().div_ceil(2)`.
#[inline]
fn quantize_channel_q4(data: &[f32], out: &mut [u8]) -> ChannelParams {
    let n = data.len();
    let (mut min, mut max) = (data[0], data[0]);
    for &v in &data[1..] {
        min = min.min(v);
        max = max.max(v);
    }
    if (max - min).abs() < f32::EPSILON {
        out.iter_mut().for_each(|b| *b = 0);
        return ChannelParams {
            scale: 1.0,
            zero_point: -min,
        };
    }
    let scale = (max - min) / 15.0;
    let inv_scale = 15.0 / (max - min);
    let bias = -min * inv_scale + 0.5;

    // Pack two 4-bit indices per byte.
    let pairs = n / 2;
    for p in 0..pairs {
        let hi = (data[p * 2] * inv_scale + bias) as u8;
        let lo = (data[p * 2 + 1] * inv_scale + bias) as u8;
        out[p] = (hi.min(15) << 4) | lo.min(15);
    }
    if n & 1 != 0 {
        let hi = (data[n - 1] * inv_scale + bias) as u8;
        out[pairs] = hi.min(15) << 4;
    }
    ChannelParams {
        scale,
        zero_point: -min * (1.0 / scale),
    }
}

/// Dequantize q8 indices back to f32.
/// The compiler autovectorizes this to NEON/SSE at opt-level >= 2.
#[inline]
fn dequantize_q8(indices: &[u8], params: &ChannelParams, out: &mut [f32]) {
    let scale = params.scale;
    let zp = params.zero_point;
    let n = indices.len();
    for i in 0..n {
        out[i] = (indices[i] as f32 - zp) * scale;
    }
}

/// Dequantize q8 with fused sign-flip: `out[i] = ((idx - zp) * scale) * signs[i]`.
/// Eliminates a separate `vec_mul_inplace` pass on the WHT read path.
#[inline]
fn dequantize_q8_signed(indices: &[u8], params: &ChannelParams, signs: &[f32], out: &mut [f32]) {
    let scale = params.scale;
    let zp = params.zero_point;
    let n = indices.len();
    for i in 0..n {
        out[i] = (indices[i] as f32 - zp) * scale * signs[i];
    }
}

/// Dequantize q4 packed indices back to f32.
#[inline]
fn dequantize_q4(packed: &[u8], n_elems: usize, params: &ChannelParams, out: &mut [f32]) {
    let scale = params.scale;
    let zp = params.zero_point;
    let pairs = n_elems / 2;
    for p in 0..pairs {
        let byte = packed[p];
        out[p * 2] = ((byte >> 4) as f32 - zp) * scale;
        out[p * 2 + 1] = ((byte & 0x0F) as f32 - zp) * scale;
    }
    if n_elems & 1 != 0 {
        out[n_elems - 1] = ((packed[pairs] >> 4) as f32 - zp) * scale;
    }
}

/// Dequantize q4 with fused sign-flip.
#[inline]
fn dequantize_q4_signed(
    packed: &[u8],
    n_elems: usize,
    params: &ChannelParams,
    signs: &[f32],
    out: &mut [f32],
) {
    let scale = params.scale;
    let zp = params.zero_point;
    let pairs = n_elems / 2;
    for p in 0..pairs {
        let byte = packed[p];
        out[p * 2] = ((byte >> 4) as f32 - zp) * scale * signs[p * 2];
        out[p * 2 + 1] = ((byte & 0x0F) as f32 - zp) * scale * signs[p * 2 + 1];
    }
    if n_elems & 1 != 0 {
        out[n_elems - 1] = ((packed[pairs] >> 4) as f32 - zp) * scale * signs[n_elems - 1];
    }
}

// ── Walsh-Hadamard Transform ─────────────────────────────────────────

/// NEON-accelerated FWHT butterfly for stages where half >= 4.
/// Processes 4 butterfly pairs per iteration using float32x4 SIMD.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fwht_butterfly_neon(data: &mut [f32], half: usize) {
    use core::arch::aarch64::*;
    let n = data.len();
    let step = half * 2;
    let ptr = data.as_mut_ptr();
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Process 4 elements at a time with NEON.
        let end4 = i + (half & !3);
        while j < end4 {
            let a = vld1q_f32(ptr.add(j));
            let b = vld1q_f32(ptr.add(j + half));
            vst1q_f32(ptr.add(j), vaddq_f32(a, b));
            vst1q_f32(ptr.add(j + half), vsubq_f32(a, b));
            j += 4;
        }
        // Scalar remainder (0-3 elements).
        while j < i + half {
            let a = *ptr.add(j);
            let b = *ptr.add(j + half);
            *ptr.add(j) = a + b;
            *ptr.add(j + half) = a - b;
            j += 1;
        }
        i += step;
    }
}

/// In-place Fast Walsh-Hadamard Transform (FWHT) on a slice of length `n`.
///
/// `n` must be a power of 2 (typically `head_dim`). Uses the iterative
/// butterfly algorithm: O(n log n) with O(1) extra memory.
/// The transform is self-inverse up to a factor of `n`: FWHT(FWHT(x)) = n * x.
///
/// On aarch64, stages with half >= 4 use NEON float32x4 intrinsics.
#[inline]
fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");

    // Small stages (half < 4): scalar butterfly.
    let mut half = 1;
    while half < 4.min(n) {
        let step = half * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + half {
                let a = data[j];
                let b = data[j + half];
                data[j] = a + b;
                data[j + half] = a - b;
            }
            i += step;
        }
        half = step;
    }

    // Large stages (half >= 4): SIMD butterfly.
    while half < n {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            fwht_butterfly_neon(data, half);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let step = half * 2;
            let mut i = 0;
            while i < n {
                for j in i..i + half {
                    let a = data[j];
                    let b = data[j + half];
                    data[j] = a + b;
                    data[j + half] = a - b;
                }
                i += step;
            }
        }
        half *= 2;
    }
}

/// Deterministic sign vector for Walsh-Hadamard rotation.
/// Uses a simple PRNG seeded on `dim` for reproducibility across sessions.
fn wht_signs(dim: usize) -> Vec<f32> {
    let mut signs = Vec::with_capacity(dim);
    // Simple LCG seeded on dim for deterministic, architecture-independent signs.
    let mut state: u64 = dim as u64 ^ 0x517c_c1b7_2722_0a95;
    for _ in 0..dim {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        signs.push(if (state >> 63) == 0 { 1.0 } else { -1.0 });
    }
    signs
}

/// Element-wise multiply: data[i] *= factors[i]. NEON-accelerated on aarch64.
#[inline]
fn vec_mul_inplace(data: &mut [f32], factors: &[f32]) {
    let n = data.len();
    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        let dp = data.as_mut_ptr();
        let fp = factors.as_ptr();
        for c in 0..chunks {
            let off = c * 4;
            unsafe {
                use core::arch::aarch64::*;
                let a = vld1q_f32(dp.add(off));
                let b = vld1q_f32(fp.add(off));
                vst1q_f32(dp.add(off), vmulq_f32(a, b));
            }
        }
        for i in chunks * 4..n {
            data[i] *= factors[i];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            data[i] *= factors[i];
        }
    }
}

/// Fused Walsh-Hadamard rotation: signs ⊙ FWHT(signs ⊙ x) / √dim.
///
/// `signs_norm` = `signs[i] / sqrt(dim)`, precomputed at cache construction.
/// Eliminates runtime `1/sqrt(dim)` computation and uses precomputed table
/// for the final sign-flip + normalize pass.
///
/// Write path: first sign-flip → FWHT → multiply by signs_norm.
#[inline]
fn wht_rotate_fused(data: &mut [f32], signs: &[f32], signs_norm: &[f32]) {
    // Pass 1: first sign flip.
    vec_mul_inplace(data, signs);
    // Passes 2..log2(n)+1: FWHT butterfly stages.
    fwht_inplace(data);
    // Final pass: second sign-flip + normalize via precomputed signs_norm.
    vec_mul_inplace(data, signs_norm);
}

/// Fused inverse WHT for read path: data already has first sign-flip applied
/// (fused into dequant), so skip it entirely. Just FWHT + apply signs_norm.
///
/// Saves one full pass over the data vs `wht_rotate_fused`.
/// Caller must ensure `data[i]` was pre-multiplied by `signs[i]` during dequant.
#[inline]
fn wht_unrotate_presigned(data: &mut [f32], signs_norm: &[f32]) {
    // Data already has signs applied — skip first sign-flip pass.
    fwht_inplace(data);
    // Second sign-flip + normalize.
    vec_mul_inplace(data, signs_norm);
}

/// Standard WHT rotation (backward compatibility for tests).
#[cfg(test)]
#[inline]
fn wht_rotate(data: &mut [f32], signs: &[f32]) {
    let dim = data.len();
    let norm = 1.0 / (dim as f32).sqrt();
    let signs_norm: Vec<f32> = signs.iter().map(|&s| s * norm).collect();
    wht_rotate_fused(data, signs, &signs_norm);
}

/// Standard inverse WHT (for tests).
#[cfg(test)]
#[inline]
fn wht_unrotate(data: &mut [f32], signs: &[f32]) {
    wht_rotate(data, signs);
}

// ── Quantized KV buffer ──────────────────────────────────────────────

/// Storage for a single layer's quantized KV data.
#[derive(Debug)]
enum LayerBuffer {
    /// Full-precision storage (boundary layers or F32 config).
    F32(Vec<f32>),
    /// 8-bit quantized: indices + per-head-per-position channel params.
    Q8 {
        indices: Vec<u8>,
        params: Vec<ChannelParams>,
        /// Number of params per token (= n_kv_heads).
        heads: usize,
    },
    /// 4-bit quantized: packed indices + per-head-per-position channel params.
    Q4 {
        packed: Vec<u8>,
        params: Vec<ChannelParams>,
        heads: usize,
        head_dim: usize,
    },
    /// Not yet allocated (lazy init).
    Empty,
}

impl LayerBuffer {
    fn is_empty(&self) -> bool {
        matches!(self, LayerBuffer::Empty)
    }
}

// ── KV Cache State ───────────────────────────────────────────────────

/// Persistent KV cache state held between executor calls.
///
/// Each layer has separate K and V buffers sized for the maximum context length.
/// The `write_pos` advances by `seq_len` per call (prefill writes many positions,
/// decode writes one).
pub struct KvCacheState {
    /// Per-layer K buffers.
    k_buffers: Vec<LayerBuffer>,
    /// Per-layer V buffers.
    v_buffers: Vec<LayerBuffer>,
    /// Current write position (number of tokens written so far).
    write_pos: usize,
    /// Number of KV heads per layer.
    n_kv_heads: u32,
    /// Head dimension.
    head_dim: u32,
    /// Maximum sequence length (context window).
    max_seq: usize,
    /// Sliding window size. When set, reads only return the last `window_size`
    /// tokens instead of the full cache. `None` = full context (no windowing).
    window_size: Option<usize>,
    /// Override for the next advance. When `Some(n)`, the executor advances by
    /// `n` instead of the auto-inferred seq_len. Consumed (reset to `None`)
    /// after each advance. Used for padded prefill where only `actual_len`
    /// tokens are real.
    advance_override: Option<usize>,
    /// Quantization configuration.
    config: KvCacheConfig,
    /// Cached WHT sign vectors (one per head_dim, computed once).
    wht_signs: Option<Vec<f32>>,
    /// Precomputed `signs[i] / sqrt(dim)` — fuses second sign-flip + normalize
    /// into a single multiply, avoiding recomputation per-head.
    wht_signs_norm: Option<Vec<f32>>,
    /// Reusable scratch buffer for WHT rotation (avoids per-head allocation).
    /// Sized to `head_dim` on first use.
    scratch: Vec<f32>,
    /// Per-layer incremental dequant cache for V buffers.
    /// Holds the fully dequantized (+ WHT-unrotated) f32 data for quantized V layers.
    /// Updated incrementally on write: only the newly-written tokens are dequantized
    /// and appended, turning decode reads from O(seq_len) to O(1).
    /// `v_dequant_cache[layer]` has length `write_pos * stride` after advance.
    v_dequant_cache: Vec<Vec<f32>>,
}

impl KvCacheState {
    /// Create a new KV cache for the given model architecture.
    ///
    /// Buffers are lazily allocated on first write — construction is O(n_layers)
    /// with zero data allocation, enabling fast cache creation for models where
    /// not all layers may be used (e.g., early exit, speculative decode).
    #[must_use]
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32, max_seq: usize) -> Self {
        Self::with_config(
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq,
            KvCacheConfig::default(),
        )
    }

    /// Create a new KV cache with explicit quantization config.
    #[must_use]
    pub fn with_config(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: usize,
        config: KvCacheConfig,
    ) -> Self {
        let k_buffers = (0..n_layers).map(|_| LayerBuffer::Empty).collect();
        let v_buffers = (0..n_layers).map(|_| LayerBuffer::Empty).collect();
        let (wht_signs, wht_signs_norm) = if config.wht_rotation && head_dim.is_power_of_two() {
            let signs = wht_signs(head_dim as usize);
            let norm = 1.0 / (head_dim as f32).sqrt();
            let signs_norm: Vec<f32> = signs.iter().map(|&s| s * norm).collect();
            (Some(signs), Some(signs_norm))
        } else {
            (None, None)
        };
        let v_dequant_cache = (0..n_layers).map(|_| Vec::new()).collect();
        Self {
            k_buffers,
            v_buffers,
            write_pos: 0,
            n_kv_heads,
            head_dim,
            max_seq,
            window_size: None,
            advance_override: None,
            config,
            wht_signs,
            wht_signs_norm,
            scratch: Vec::new(),
            v_dequant_cache,
        }
    }

    /// Create a KV cache with a sliding window (bounded context).
    ///
    /// When `window_size` is set, reads return at most the last `window_size`
    /// tokens — older tokens are still in the buffer but not exposed.
    #[must_use]
    pub fn with_window(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq: usize,
        window_size: usize,
    ) -> Self {
        let mut s = Self::new(n_layers, n_kv_heads, head_dim, max_seq);
        s.window_size = Some(window_size);
        s
    }

    /// Current write position (tokens cached so far).
    #[must_use]
    pub fn write_pos(&self) -> usize {
        self.write_pos
    }

    /// Number of layers in the cache.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.k_buffers.len()
    }

    /// The quantization config.
    #[must_use]
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Allocate a layer buffer on first write.
    fn alloc_layer_buffer(
        bits: KvBits,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> LayerBuffer {
        let stride = n_kv_heads * head_dim;
        if stride == 0 || max_seq == 0 {
            return LayerBuffer::Empty;
        }
        match bits {
            KvBits::F32 => LayerBuffer::F32(vec![0.0f32; max_seq * stride]),
            KvBits::Q8 => LayerBuffer::Q8 {
                indices: vec![0u8; max_seq * stride],
                params: vec![
                    ChannelParams {
                        scale: 1.0,
                        zero_point: 0.0
                    };
                    max_seq * n_kv_heads
                ],
                heads: n_kv_heads,
            },
            KvBits::Q4 => {
                let packed_per_token = n_kv_heads * (head_dim.div_ceil(2));
                LayerBuffer::Q4 {
                    packed: vec![0u8; max_seq * packed_per_token],
                    params: vec![
                        ChannelParams {
                            scale: 1.0,
                            zero_point: 0.0
                        };
                        max_seq * n_kv_heads
                    ],
                    heads: n_kv_heads,
                    head_dim,
                }
            }
        }
    }

    /// Write data into a layer buffer at the given position range.
    /// `scratch` is a reusable buffer for WHT rotation (avoids per-head allocation).
    /// When WHT is active, `wht_signs` and `wht_signs_norm` must both be Some.
    #[allow(clippy::too_many_arguments)]
    fn write_to_buffer(
        buf: &mut LayerBuffer,
        data: &[f32],
        start_pos: usize,
        seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wht_signs: Option<&[f32]>,
        wht_signs_norm: Option<&[f32]>,
        scratch: &mut Vec<f32>,
    ) {
        let stride = n_kv_heads * head_dim;
        match buf {
            LayerBuffer::F32(ref mut v) => {
                let start = start_pos * stride;
                let end = start + seq_len * stride;
                v[start..end].copy_from_slice(&data[..seq_len * stride]);
            }
            LayerBuffer::Q8 {
                indices,
                params,
                heads,
            } => {
                let n_heads = *heads;
                if wht_signs.is_some() {
                    scratch.resize(head_dim, 0.0);
                }
                for t in 0..seq_len {
                    let src_off = t * stride;
                    let dst_off = (start_pos + t) * stride;
                    let param_off = (start_pos + t) * n_heads;
                    for h in 0..n_heads {
                        let s = src_off + h * head_dim;
                        let d = dst_off + h * head_dim;
                        let src = if let (Some(signs), Some(sn)) = (wht_signs, wht_signs_norm) {
                            scratch.copy_from_slice(&data[s..s + head_dim]);
                            wht_rotate_fused(scratch, signs, sn);
                            scratch.as_slice()
                        } else {
                            &data[s..s + head_dim]
                        };
                        params[param_off + h] =
                            quantize_channel_q8(src, &mut indices[d..d + head_dim]);
                    }
                }
            }
            LayerBuffer::Q4 {
                packed,
                params,
                heads,
                head_dim: hd,
            } => {
                let n_heads = *heads;
                let dim = *hd;
                let packed_dim = dim.div_ceil(2);
                let packed_stride = n_heads * packed_dim;
                if wht_signs.is_some() {
                    scratch.resize(dim, 0.0);
                }
                for t in 0..seq_len {
                    let src_off = t * stride;
                    let dst_off = (start_pos + t) * packed_stride;
                    let param_off = (start_pos + t) * n_heads;
                    for h in 0..n_heads {
                        let s = src_off + h * dim;
                        let d = dst_off + h * packed_dim;
                        let src = if let (Some(signs), Some(sn)) = (wht_signs, wht_signs_norm) {
                            scratch.copy_from_slice(&data[s..s + dim]);
                            wht_rotate_fused(scratch, signs, sn);
                            scratch.as_slice()
                        } else {
                            &data[s..s + dim]
                        };
                        params[param_off + h] =
                            quantize_channel_q4(src, &mut packed[d..d + packed_dim]);
                    }
                }
            }
            LayerBuffer::Empty => {}
        }
    }

    /// Read data from a layer buffer, dequantizing if needed.
    ///
    /// When WHT is active (`wht_signs` + `wht_signs_norm` both Some), the first
    /// sign-flip is fused into the dequant output, and `wht_unrotate_presigned`
    /// is used — saving one full pass over the data per head.
    fn read_from_buffer(
        buf: &LayerBuffer,
        start_pos: usize,
        n_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wht_signs: Option<&[f32]>,
        wht_signs_norm: Option<&[f32]>,
    ) -> Vec<f32> {
        let stride = n_kv_heads * head_dim;
        match buf {
            LayerBuffer::F32(v) => {
                let start = start_pos * stride;
                let end = start + n_tokens * stride;
                v[start..end].to_vec()
            }
            LayerBuffer::Q8 {
                indices,
                params,
                heads,
            } => {
                let n_heads = *heads;
                let mut out = vec![0.0f32; n_tokens * stride];
                for t in 0..n_tokens {
                    let idx_off = (start_pos + t) * stride;
                    let param_off = (start_pos + t) * n_heads;
                    let out_off = t * stride;
                    for h in 0..n_heads {
                        let si = idx_off + h * head_dim;
                        let so = out_off + h * head_dim;
                        if let Some(signs) = wht_signs {
                            // Fused: dequant + sign-flip in single loop body.
                            dequantize_q8_signed(
                                &indices[si..si + head_dim],
                                &params[param_off + h],
                                signs,
                                &mut out[so..so + head_dim],
                            );
                        } else {
                            dequantize_q8(
                                &indices[si..si + head_dim],
                                &params[param_off + h],
                                &mut out[so..so + head_dim],
                            );
                        }
                    }
                }
                // WHT: skip first sign-flip (fused into dequant), just FWHT + signs_norm.
                if let Some(sn) = wht_signs_norm {
                    for chunk in out.chunks_exact_mut(head_dim) {
                        wht_unrotate_presigned(chunk, sn);
                    }
                }
                out
            }
            LayerBuffer::Q4 {
                packed,
                params,
                heads,
                head_dim: hd,
            } => {
                let n_heads = *heads;
                let dim = *hd;
                let packed_dim = dim.div_ceil(2);
                let packed_stride = n_heads * packed_dim;
                let mut out = vec![0.0f32; n_tokens * stride];
                for t in 0..n_tokens {
                    let p_off = (start_pos + t) * packed_stride;
                    let param_off = (start_pos + t) * n_heads;
                    let out_off = t * stride;
                    for h in 0..n_heads {
                        let sp = p_off + h * packed_dim;
                        let so = out_off + h * dim;
                        if let Some(signs) = wht_signs {
                            // Fused: dequant + sign-flip in single loop body.
                            dequantize_q4_signed(
                                &packed[sp..sp + packed_dim],
                                dim,
                                &params[param_off + h],
                                signs,
                                &mut out[so..so + dim],
                            );
                        } else {
                            dequantize_q4(
                                &packed[sp..sp + packed_dim],
                                dim,
                                &params[param_off + h],
                                &mut out[so..so + dim],
                            );
                        }
                    }
                }
                // WHT: skip first sign-flip (fused into dequant).
                if let Some(sn) = wht_signs_norm {
                    for chunk in out.chunks_exact_mut(dim) {
                        wht_unrotate_presigned(chunk, sn);
                    }
                }
                out
            }
            LayerBuffer::Empty => Vec::new(),
        }
    }

    /// Write K and/or V data for a layer at the current position.
    ///
    /// `k_data` and `v_data` are flat f32 slices of `seq_len * n_kv_heads * head_dim` elements.
    /// Either may be empty (skipped). Caller must call `advance` after writing all layers.
    ///
    /// Returns the number of elements written (from whichever was non-empty).
    pub fn write_layer(&mut self, layer: u32, k_data: &[f32], v_data: &[f32]) -> usize {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return 0;
        }
        let n_heads = self.n_kv_heads as usize;
        let hd = self.head_dim as usize;
        let stride = n_heads * hd;
        if stride == 0 {
            return 0;
        }

        let data = if !k_data.is_empty() { k_data } else { v_data };
        let seq_len = data.len() / stride;
        let end = self.write_pos + seq_len;

        if end > self.max_seq {
            return 0; // would overflow max_seq
        }

        let n_layers = self.k_buffers.len() as u32;

        // Lazy allocation on first write.
        if self.k_buffers[layer_idx].is_empty() {
            let k_bits = self.config.effective_k_bits(layer, n_layers);
            let v_bits = self.config.effective_v_bits(layer, n_layers);
            self.k_buffers[layer_idx] = Self::alloc_layer_buffer(k_bits, self.max_seq, n_heads, hd);
            self.v_buffers[layer_idx] = Self::alloc_layer_buffer(v_bits, self.max_seq, n_heads, hd);
        }

        // Take scratch out to avoid borrow conflict with self.k_buffers/v_buffers.
        let mut scratch = std::mem::take(&mut self.scratch);

        // K never gets WHT rotation (inner-product preservation).
        if !k_data.is_empty() {
            Self::write_to_buffer(
                &mut self.k_buffers[layer_idx],
                k_data,
                self.write_pos,
                seq_len,
                n_heads,
                hd,
                None,
                None,
                &mut scratch,
            );
        }
        // V gets WHT rotation if configured and layer is not boundary.
        if !v_data.is_empty() {
            let is_boundary = self.config.is_boundary_layer(layer, n_layers);
            let v_signs = if is_boundary {
                None
            } else {
                self.wht_signs.as_deref()
            };
            let v_signs_norm = if is_boundary {
                None
            } else {
                self.wht_signs_norm.as_deref()
            };
            Self::write_to_buffer(
                &mut self.v_buffers[layer_idx],
                v_data,
                self.write_pos,
                seq_len,
                n_heads,
                hd,
                v_signs,
                v_signs_norm,
                &mut scratch,
            );

            // Incremental dequant cache: append the original f32 V data directly.
            // This avoids the expensive quantize→dequantize→WHT round-trip.
            // The dequant cache stores the original pre-quantization values,
            // which are actually higher quality than the quantized round-trip.
            if !matches!(self.v_buffers[layer_idx], LayerBuffer::F32(_)) {
                self.v_dequant_cache[layer_idx].extend_from_slice(&v_data[..seq_len * stride]);
            }
        }

        // Return scratch (preserves its allocation for next call).
        self.scratch = scratch;
        seq_len * stride
    }

    /// Advance the write position by `seq_len` tokens.
    /// Call after writing all layers for a step.
    ///
    /// If `set_advance_override` was called, uses that value instead of `seq_len`
    /// (then clears the override).
    pub fn advance(&mut self, seq_len: usize) {
        let n = self.advance_override.take().unwrap_or(seq_len);
        self.write_pos = (self.write_pos + n).min(self.max_seq);
    }

    /// Set an override for the next `advance` call.
    ///
    /// For padded prefill: the model processes `padded_len` tokens but only
    /// `actual_len` are real. Call `set_advance_override(actual_len)` before
    /// execution so the cache only records real token positions.
    pub fn set_advance_override(&mut self, n: usize) {
        self.advance_override = Some(n);
    }

    /// Effective number of visible tokens (respects sliding window).
    fn visible_tokens(&self) -> usize {
        match self.window_size {
            Some(w) => self.write_pos.min(w),
            None => self.write_pos,
        }
    }

    /// Start position for reads (skips tokens outside the window).
    fn read_start(&self) -> usize {
        self.write_pos - self.visible_tokens()
    }

    /// Read cached K data for a layer.
    ///
    /// With sliding window, returns only the last `window_size` tokens.
    /// Without windowing, returns all tokens from position 0 to write_pos.
    ///
    /// For f32 layers this returns borrowed data. For quantized layers it
    /// dequantizes on the fly — use `read_k_owned` for the general case.
    #[must_use]
    pub fn read_k(&self, layer: u32) -> &[f32] {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return &[];
        }
        match &self.k_buffers[layer_idx] {
            LayerBuffer::F32(v) => {
                let stride = self.n_kv_heads as usize * self.head_dim as usize;
                let start = self.read_start() * stride;
                let end = self.write_pos * stride;
                &v[start..end]
            }
            _ => &[], // quantized K: caller should use read_k_owned
        }
    }

    /// Read cached V data for a layer (respects sliding window).
    ///
    /// For f32 layers this returns borrowed data. For quantized layers,
    /// use `read_v_owned`.
    #[must_use]
    pub fn read_v(&self, layer: u32) -> &[f32] {
        let layer_idx = layer as usize;
        if layer_idx >= self.v_buffers.len() {
            return &[];
        }
        match &self.v_buffers[layer_idx] {
            LayerBuffer::F32(v) => {
                let stride = self.n_kv_heads as usize * self.head_dim as usize;
                let start = self.read_start() * stride;
                let end = self.write_pos * stride;
                &v[start..end]
            }
            _ => &[], // quantized V: caller should use read_v_owned
        }
    }

    /// Read K data, dequantizing if necessary. Works for all storage formats.
    #[must_use]
    pub fn read_k_owned(&self, layer: u32) -> Vec<f32> {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return Vec::new();
        }
        let start = self.read_start();
        let n_tokens = self.visible_tokens();
        Self::read_from_buffer(
            &self.k_buffers[layer_idx],
            start,
            n_tokens,
            self.n_kv_heads as usize,
            self.head_dim as usize,
            None,
            None,
        )
    }

    /// Read V data, dequantizing and un-rotating if necessary.
    ///
    /// For quantized layers, uses the incremental dequant cache — O(1) after
    /// the initial prefill since only newly-written tokens are dequantized
    /// on each `write_layer` call.
    #[must_use]
    pub fn read_v_owned(&self, layer: u32) -> Vec<f32> {
        let layer_idx = layer as usize;
        if layer_idx >= self.v_buffers.len() {
            return Vec::new();
        }
        // Use dequant cache if available (quantized layers).
        let cache = &self.v_dequant_cache[layer_idx];
        if !cache.is_empty() {
            let stride = self.n_kv_heads as usize * self.head_dim as usize;
            let start = self.read_start() * stride;
            let end = (self.write_pos * stride).min(cache.len());
            return cache[start..end].to_vec();
        }
        // F32 layers: read directly.
        let n_layers = self.v_buffers.len() as u32;
        let start = self.read_start();
        let n_tokens = self.visible_tokens();
        let is_boundary = self.config.is_boundary_layer(layer, n_layers);
        let v_signs = if is_boundary {
            None
        } else {
            self.wht_signs.as_deref()
        };
        let v_signs_norm = if is_boundary {
            None
        } else {
            self.wht_signs_norm.as_deref()
        };
        Self::read_from_buffer(
            &self.v_buffers[layer_idx],
            start,
            n_tokens,
            self.n_kv_heads as usize,
            self.head_dim as usize,
            v_signs,
            v_signs_norm,
        )
    }

    /// Read cached K including pending (just-written, pre-advance) data.
    ///
    /// Returns `[0..(write_pos + pending_seq) * stride]` — includes both
    /// previously cached tokens and the new tokens written in this step.
    /// This enables a unified code path for prefill and decode.
    #[must_use]
    pub fn read_k_through(&self, layer: u32, pending_seq: usize) -> &[f32] {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return &[];
        }
        match &self.k_buffers[layer_idx] {
            LayerBuffer::F32(v) => {
                let stride = self.n_kv_heads as usize * self.head_dim as usize;
                let total_seq = (self.write_pos + pending_seq).min(self.max_seq);
                let end = (total_seq * stride).min(v.len());
                &v[..end]
            }
            _ => &[], // quantized: caller should use read_k_through_owned
        }
    }

    /// Read cached V including pending (just-written, pre-advance) data.
    #[must_use]
    pub fn read_v_through(&self, layer: u32, pending_seq: usize) -> &[f32] {
        let layer_idx = layer as usize;
        if layer_idx >= self.v_buffers.len() {
            return &[];
        }
        match &self.v_buffers[layer_idx] {
            LayerBuffer::F32(v) => {
                let stride = self.n_kv_heads as usize * self.head_dim as usize;
                let total_seq = (self.write_pos + pending_seq).min(self.max_seq);
                let end = (total_seq * stride).min(v.len());
                &v[..end]
            }
            _ => &[], // quantized: caller should use read_k_through_owned
        }
    }

    /// Read K including pending data, dequantizing if necessary.
    #[must_use]
    pub fn read_k_through_owned(&self, layer: u32, pending_seq: usize) -> Vec<f32> {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return Vec::new();
        }
        let total_seq = (self.write_pos + pending_seq).min(self.max_seq);
        Self::read_from_buffer(
            &self.k_buffers[layer_idx],
            0,
            total_seq,
            self.n_kv_heads as usize,
            self.head_dim as usize,
            None,
            None,
        )
    }

    /// Read V including pending data, dequantizing and un-rotating if necessary.
    ///
    /// Uses incremental dequant cache for quantized layers — the cache includes
    /// pending (pre-advance) tokens since they were dequantized during `write_layer`.
    #[must_use]
    pub fn read_v_through_owned(&self, layer: u32, pending_seq: usize) -> Vec<f32> {
        let layer_idx = layer as usize;
        if layer_idx >= self.v_buffers.len() {
            return Vec::new();
        }
        // Use dequant cache if available.
        let cache = &self.v_dequant_cache[layer_idx];
        if !cache.is_empty() {
            let stride = self.n_kv_heads as usize * self.head_dim as usize;
            let total_seq = (self.write_pos + pending_seq).min(self.max_seq);
            let end = (total_seq * stride).min(cache.len());
            return cache[..end].to_vec();
        }
        // F32 layers: fall through.
        let n_layers = self.v_buffers.len() as u32;
        let total_seq = (self.write_pos + pending_seq).min(self.max_seq);
        let is_boundary = self.config.is_boundary_layer(layer, n_layers);
        let v_signs = if is_boundary {
            None
        } else {
            self.wht_signs.as_deref()
        };
        let v_signs_norm = if is_boundary {
            None
        } else {
            self.wht_signs_norm.as_deref()
        };
        Self::read_from_buffer(
            &self.v_buffers[layer_idx],
            0,
            total_seq,
            self.n_kv_heads as usize,
            self.head_dim as usize,
            v_signs,
            v_signs_norm,
        )
    }

    /// Returns true if the K buffer for this layer is quantized (not f32).
    #[must_use]
    pub fn is_k_quantized(&self, layer: u32) -> bool {
        let layer_idx = layer as usize;
        if layer_idx >= self.k_buffers.len() {
            return false;
        }
        !matches!(
            self.k_buffers[layer_idx],
            LayerBuffer::F32(_) | LayerBuffer::Empty
        )
    }

    /// Returns true if the V buffer for this layer is quantized (not f32).
    #[must_use]
    pub fn is_v_quantized(&self, layer: u32) -> bool {
        let layer_idx = layer as usize;
        if layer_idx >= self.v_buffers.len() {
            return false;
        }
        !matches!(
            self.v_buffers[layer_idx],
            LayerBuffer::F32(_) | LayerBuffer::Empty
        )
    }

    /// Reset the cache for a new sequence.
    pub fn reset(&mut self) {
        self.write_pos = 0;
        // Clear dequant caches (data will be rebuilt incrementally).
        for cache in &mut self.v_dequant_cache {
            cache.clear();
        }
        // Don't need to zero the quantized buffers — they'll be overwritten.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Original API tests (f32 path) ────────────────────────────────

    #[test]
    fn write_and_read() {
        let mut cache = KvCacheState::new(2, 4, 8, 16); // 2 layers, 4 heads, dim=8, max_seq=16
        let stride = 4 * 8; // 32 f32s per token

        // Prefill: write 3 tokens for layer 0.
        let k: Vec<f32> = (0..3 * stride).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..3 * stride).map(|i| (i as f32) * 0.5).collect();
        let written = cache.write_layer(0, &k, &v);
        assert_eq!(written, 3 * stride);

        // Also write layer 1.
        cache.write_layer(1, &k, &v);
        cache.advance(3);
        assert_eq!(cache.write_pos(), 3);

        // Read back.
        let k_read = cache.read_k(0);
        assert_eq!(k_read.len(), 3 * stride);
        assert_eq!(k_read[0], 0.0);
        assert_eq!(k_read[stride], stride as f32);

        // Decode: write 1 more token.
        let k1: Vec<f32> = vec![99.0; stride];
        let v1: Vec<f32> = vec![88.0; stride];
        cache.write_layer(0, &k1, &v1);
        cache.write_layer(1, &k1, &v1);
        cache.advance(1);
        assert_eq!(cache.write_pos(), 4);

        // Read should now include all 4 tokens.
        let k_read = cache.read_k(0);
        assert_eq!(k_read.len(), 4 * stride);
        // First 3 tokens unchanged, 4th is 99.0.
        assert_eq!(k_read[3 * stride], 99.0);
    }

    #[test]
    fn reset_clears_position() {
        let mut cache = KvCacheState::new(1, 2, 4, 8);
        let stride = 2 * 4;
        let k: Vec<f32> = vec![1.0; 2 * stride];
        let v: Vec<f32> = vec![2.0; 2 * stride];
        cache.write_layer(0, &k, &v);
        cache.advance(2);
        assert_eq!(cache.write_pos(), 2);

        cache.reset();
        assert_eq!(cache.write_pos(), 0);
        assert_eq!(cache.read_k(0).len(), 0);
    }

    // ── Boundary layer protection ────────────────────────────────────

    #[test]
    fn boundary_layers_remain_f32() {
        let config = KvCacheConfig {
            k_bits: KvBits::Q8,
            v_bits: KvBits::Q4,
            boundary_layers: 2,
            wht_rotation: false,
        };
        // 6 layers: layers 0,1 and 4,5 should be f32 (boundary).
        let mut cache = KvCacheState::with_config(6, 2, 8, 16, config);
        let stride = 2 * 8;
        let data: Vec<f32> = (0..stride).map(|i| i as f32 * 0.1).collect();

        // Write to all layers to trigger allocation.
        for layer in 0..6 {
            cache.write_layer(layer, &data, &data);
        }
        cache.advance(1);

        // Boundary layers (0, 1, 4, 5) should NOT be quantized.
        assert!(!cache.is_k_quantized(0));
        assert!(!cache.is_k_quantized(1));
        assert!(!cache.is_v_quantized(0));
        assert!(!cache.is_v_quantized(1));
        assert!(!cache.is_k_quantized(4));
        assert!(!cache.is_k_quantized(5));
        assert!(!cache.is_v_quantized(4));
        assert!(!cache.is_v_quantized(5));

        // Middle layers (2, 3) SHOULD be quantized.
        assert!(cache.is_k_quantized(2));
        assert!(cache.is_k_quantized(3));
        assert!(cache.is_v_quantized(2));
        assert!(cache.is_v_quantized(3));
    }

    #[test]
    fn boundary_config_edge_cases() {
        let config = KvCacheConfig {
            k_bits: KvBits::Q8,
            v_bits: KvBits::Q4,
            boundary_layers: 0,
            wht_rotation: false,
        };
        // boundary_layers=0: all layers quantized.
        assert!(!config.is_boundary_layer(0, 6));
        assert!(!config.is_boundary_layer(5, 6));

        let config2 = KvCacheConfig {
            boundary_layers: 10,
            ..config
        };
        // boundary_layers > n_layers: all layers protected.
        assert!(config2.is_boundary_layer(0, 6));
        assert!(config2.is_boundary_layer(3, 6));
        assert!(config2.is_boundary_layer(5, 6));
    }

    // ── Asymmetric K/V quantization ──────────────────────────────────

    #[test]
    fn asymmetric_kv_f32_k_q8_v() {
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q8,
            boundary_layers: 0,
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 8, 16, config);
        let stride = 2 * 8;
        let k: Vec<f32> = (0..3 * stride).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..3 * stride).map(|i| (i as f32) * 0.02 - 0.5).collect();

        cache.write_layer(0, &k, &v);
        cache.advance(3);

        // K should be exact (f32).
        assert!(!cache.is_k_quantized(0));
        let k_read = cache.read_k(0);
        assert_eq!(k_read, &k[..]);

        // V should be quantized — read via owned path.
        assert!(cache.is_v_quantized(0));
        let v_read = cache.read_v_owned(0);
        assert_eq!(v_read.len(), v.len());

        // Q8 round-trip error should be small (within ~0.5% of range).
        let max_err: f32 = v
            .iter()
            .zip(v_read.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let range = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - v.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_err < range * 0.01,
            "Q8 max error {max_err} exceeds 1% of range {range}"
        );
    }

    #[test]
    fn asymmetric_kv_f32_k_q4_v() {
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q4,
            boundary_layers: 0,
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 8, 16, config);
        let stride = 2 * 8;
        let k: Vec<f32> = (0..2 * stride).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..2 * stride).map(|i| (i as f32) * 0.1 - 1.0).collect();

        cache.write_layer(0, &k, &v);
        cache.advance(2);

        // K exact.
        let k_read = cache.read_k(0);
        assert_eq!(k_read, &k[..]);

        // V quantized to 4-bit — higher tolerance.
        let v_read = cache.read_v_owned(0);
        let max_err: f32 = v
            .iter()
            .zip(v_read.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let range = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - v.iter().cloned().fold(f32::INFINITY, f32::min);
        // Q4 has 16 levels: max quantization error ~range/30.
        assert!(
            max_err < range * 0.07,
            "Q4 max error {max_err} exceeds 7% of range {range}"
        );
    }

    #[test]
    fn q8_round_trip_per_channel() {
        // Verify per-channel quantization preserves values within tolerance.
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let mut indices = vec![0u8; 128];
        let params = quantize_channel_q8(&data, &mut indices);
        let mut reconstructed = vec![0.0f32; 128];
        dequantize_q8(&indices, &params, &mut reconstructed);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let err = (orig - recon).abs();
            assert!(err < 0.06, "Q8 error {err} too large for value {orig}");
        }
    }

    #[test]
    fn q4_round_trip_per_channel() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let mut packed = vec![0u8; 64];
        let params = quantize_channel_q4(&data, &mut packed);
        let mut reconstructed = vec![0.0f32; 128];
        dequantize_q4(&packed, 128, &params, &mut reconstructed);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let err = (orig - recon).abs();
            // Q4 = 16 levels over 12.8 range → step ~0.85, max err ~0.43.
            assert!(err < 0.5, "Q4 error {err} too large for value {orig}");
        }
    }

    #[test]
    fn q4_odd_dimension() {
        // Ensure odd head_dim doesn't panic (trailing nibble).
        let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut packed = vec![0u8; 4]; // ceil(7/2)
        let params = quantize_channel_q4(&data, &mut packed);
        let mut reconstructed = vec![0.0f32; 7];
        dequantize_q4(&packed, 7, &params, &mut reconstructed);

        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            let err = (orig - recon).abs();
            assert!(err < 0.5, "Q4 odd-dim error {err} for value {orig}");
        }
    }

    #[test]
    fn constant_channel_quantization() {
        // All-same values should not panic or produce NaN.
        let data = vec![3.14f32; 64];

        let mut q8_idx = vec![0u8; 64];
        let p8 = quantize_channel_q8(&data, &mut q8_idx);
        let mut r8 = vec![0.0f32; 64];
        dequantize_q8(&q8_idx, &p8, &mut r8);
        for &v in &r8 {
            assert!(!v.is_nan());
            assert!((v - 3.14).abs() < 0.01);
        }

        let mut q4_packed = vec![0u8; 32];
        let p4 = quantize_channel_q4(&data, &mut q4_packed);
        let mut r4 = vec![0.0f32; 64];
        dequantize_q4(&q4_packed, 64, &p4, &mut r4);
        for &v in &r4 {
            assert!(!v.is_nan());
            assert!((v - 3.14).abs() < 0.01);
        }
    }

    // ── Walsh-Hadamard Transform ─────────────────────────────────────

    #[test]
    fn fwht_self_inverse() {
        // FWHT(FWHT(x)) = n * x.
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.3 - 2.0).collect();
        let mut data = original.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        let n = original.len() as f32;
        for (orig, val) in original.iter().zip(data.iter()) {
            assert!(
                (val / n - orig).abs() < 1e-5,
                "FWHT not self-inverse: {val}/{n} vs {orig}"
            );
        }
    }

    #[test]
    fn wht_rotate_unrotate_identity() {
        let dim = 32;
        let signs = wht_signs(dim);
        let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.7 - 10.0).collect();
        let mut data = original.clone();

        wht_rotate(&mut data, &signs);
        // After rotation, values should be different.
        assert_ne!(data, original);

        wht_unrotate(&mut data, &signs);
        // After un-rotation, should recover original.
        for (orig, val) in original.iter().zip(data.iter()) {
            assert!(
                (orig - val).abs() < 1e-4,
                "WHT round-trip failed: {orig} vs {val}"
            );
        }
    }

    #[test]
    fn wht_gaussianizes_distribution() {
        // Sparse input (most values near 0, a few outliers) should have
        // smaller max/min ratio after rotation.
        let dim = 64;
        let signs = wht_signs(dim);
        let mut data = vec![0.0f32; dim];
        data[0] = 100.0;
        data[1] = -50.0;
        data[dim - 1] = 75.0;

        let pre_max = data.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));

        wht_rotate(&mut data, &signs);

        let post_max = data.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        // After rotation, the peak should be smaller (energy spread out).
        assert!(
            post_max < pre_max,
            "WHT should reduce peak: pre={pre_max} post={post_max}"
        );
    }

    #[test]
    fn wht_rotation_improves_quantization() {
        // Quantizing after WHT rotation should produce lower error than without.
        let dim = 64;
        let signs = wht_signs(dim);

        // Sparse data with outliers — worst case for uniform quantization.
        let mut data = vec![0.0f32; dim];
        data[0] = 100.0;
        data[1] = -80.0;
        data[dim / 2] = 60.0;

        // Quantize without rotation.
        let mut q_no_rot = vec![0u8; dim];
        let p_no_rot = quantize_channel_q8(&data, &mut q_no_rot);
        let mut recon_no_rot = vec![0.0f32; dim];
        dequantize_q8(&q_no_rot, &p_no_rot, &mut recon_no_rot);
        let mse_no_rot: f32 = data
            .iter()
            .zip(recon_no_rot.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / dim as f32;

        // Quantize with rotation.
        let mut rotated = data.clone();
        wht_rotate(&mut rotated, &signs);
        let mut q_rot = vec![0u8; dim];
        let p_rot = quantize_channel_q8(&rotated, &mut q_rot);
        let mut recon_rot = vec![0.0f32; dim];
        dequantize_q8(&q_rot, &p_rot, &mut recon_rot);
        wht_unrotate(&mut recon_rot, &signs);
        let mse_rot: f32 = data
            .iter()
            .zip(recon_rot.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / dim as f32;

        assert!(
            mse_rot < mse_no_rot,
            "WHT should reduce quantization MSE: with={mse_rot} without={mse_no_rot}"
        );
    }

    // ── Full pipeline: quantized cache with WHT ──────────────────────

    #[test]
    fn full_pipeline_asymmetric_q4_with_wht() {
        let config = KvCacheConfig::asymmetric_q4();
        // 6 layers, boundary_layers=2: layers 0,1,4,5 are f32; layers 2,3 are quantized.
        let mut cache = KvCacheState::with_config(6, 4, 8, 32, config);
        let stride = 4 * 8; // 32

        // Write 5 tokens of data.
        let k: Vec<f32> = (0..5 * stride).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..5 * stride).map(|i| (i as f32) * 0.02 - 1.0).collect();
        for layer in 0..6 {
            cache.write_layer(layer, &k, &v);
        }
        cache.advance(5);

        // Boundary layers: K and V exact.
        for &layer in &[0u32, 1, 4, 5] {
            let k_read = cache.read_k(layer);
            assert_eq!(k_read, &k[..], "boundary layer {layer} K should be exact");
            let v_read = cache.read_v(layer);
            assert_eq!(v_read, &v[..], "boundary layer {layer} V should be exact");
        }

        // Middle layers: K is f32 (config k_bits=F32), V is q4.
        for &layer in &[2u32, 3] {
            let k_read = cache.read_k(layer);
            assert_eq!(k_read, &k[..], "middle layer {layer} K should be f32");

            // V is quantized — read_v returns empty for quantized, use owned.
            assert!(cache.is_v_quantized(layer));
            let v_read = cache.read_v_owned(layer);
            assert_eq!(v_read.len(), v.len());

            // Check tolerance (Q4 + WHT).
            let max_err: f32 = v
                .iter()
                .zip(v_read.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let range = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - v.iter().cloned().fold(f32::INFINITY, f32::min);
            assert!(
                max_err < range * 0.1,
                "layer {layer} Q4+WHT max error {max_err} exceeds 10% of range {range}"
            );
        }
    }

    #[test]
    fn through_reads_work_with_quantization() {
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q8,
            boundary_layers: 0,
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 4, 16, config);
        let stride = 2 * 4;

        // Write 3 tokens.
        let data: Vec<f32> = (0..3 * stride).map(|i| i as f32 * 0.1).collect();
        cache.write_layer(0, &data, &data);
        // Don't advance yet — test read_through with pending=3.

        let k_through = cache.read_k_through(0, 3);
        assert_eq!(k_through.len(), 3 * stride);

        let v_through = cache.read_v_through_owned(0, 3);
        assert_eq!(v_through.len(), 3 * stride);

        // K should be exact (f32).
        assert_eq!(k_through, &data[..]);

        // V should be close (q8 round-trip).
        let max_err: f32 = data
            .iter()
            .zip(v_through.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "through-read Q8 error {max_err} too large");
    }

    #[test]
    fn decode_appends_to_quantized_cache() {
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q8,
            boundary_layers: 0,
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 4, 16, config);
        let stride = 2 * 4;

        // Prefill: 3 tokens.
        let v_prefill: Vec<f32> = (0..3 * stride).map(|i| i as f32 * 0.1).collect();
        let k_prefill: Vec<f32> = (0..3 * stride).map(|i| i as f32).collect();
        cache.write_layer(0, &k_prefill, &v_prefill);
        cache.advance(3);

        // Decode: 1 more token.
        let v_decode: Vec<f32> = vec![99.0; stride];
        let k_decode: Vec<f32> = vec![42.0; stride];
        cache.write_layer(0, &k_decode, &v_decode);
        cache.advance(1);

        assert_eq!(cache.write_pos(), 4);

        // K: exact, 4 tokens.
        let k_read = cache.read_k(0);
        assert_eq!(k_read.len(), 4 * stride);
        assert_eq!(k_read[3 * stride], 42.0);

        // V: quantized, 4 tokens.
        let v_read = cache.read_v_owned(0);
        assert_eq!(v_read.len(), 4 * stride);
        // Last token's first element should be close to 99.0.
        assert!(
            (v_read[3 * stride] - 99.0).abs() < 1.0,
            "decode V value {} not close to 99.0",
            v_read[3 * stride]
        );
    }

    #[test]
    fn default_config_is_f32() {
        let config = KvCacheConfig::default();
        assert_eq!(config.k_bits, KvBits::F32);
        assert_eq!(config.v_bits, KvBits::F32);
        assert_eq!(config.boundary_layers, 2);
        assert!(!config.wht_rotation);

        // Default cache should behave identically to old API.
        let mut cache = KvCacheState::new(1, 2, 4, 8);
        let stride = 2 * 4;
        let data: Vec<f32> = (0..2 * stride).map(|i| i as f32).collect();
        cache.write_layer(0, &data, &data);
        cache.advance(2);

        assert!(!cache.is_k_quantized(0));
        assert!(!cache.is_v_quantized(0));
        assert_eq!(cache.read_k(0), &data[..]);
        assert_eq!(cache.read_v(0), &data[..]);
    }

    // ── Sliding window + quantization ────────────────────────────────

    #[test]
    fn sliding_window_with_quantization() {
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q8,
            boundary_layers: 0,
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 4, 16, config);
        cache.window_size = Some(3);
        let stride = 2 * 4;

        // Write 5 tokens.
        let data: Vec<f32> = (0..5 * stride).map(|i| i as f32 * 0.1).collect();
        cache.write_layer(0, &data, &data);
        cache.advance(5);

        // Window=3: should see only last 3 tokens (positions 2,3,4).
        let k_read = cache.read_k(0);
        assert_eq!(k_read.len(), 3 * stride);

        let v_read = cache.read_v_owned(0);
        assert_eq!(v_read.len(), 3 * stride);
    }

    // ── Plan 038: asymmetric K/V with boundary layers and WHT ────────

    #[test]
    fn asymmetric_q4_v_with_wht_and_boundary() {
        // 4 layers, boundary_layers=1 → layers 0 and 3 stay F32, layers 1-2 use Q4+WHT for V.
        let config = KvCacheConfig {
            k_bits: KvBits::F32,
            v_bits: KvBits::Q4,
            boundary_layers: 1,
            wht_rotation: true,
        };
        let n_layers = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 8u32; // power of 2 for WHT
        let max_seq = 16;
        let stride = n_kv_heads as usize * head_dim as usize;

        let mut cache = KvCacheState::with_config(n_layers, n_kv_heads, head_dim, max_seq, config);

        // Write 3 tokens to all layers.
        let k_data: Vec<f32> = (0..3 * stride).map(|i| (i as f32) * 0.1).collect();
        let v_data: Vec<f32> = (0..3 * stride).map(|i| (i as f32) * 0.05 + 1.0).collect();
        for layer in 0..n_layers {
            cache.write_layer(layer, &k_data, &v_data);
        }
        cache.advance(3);

        // K is always F32 → lossless roundtrip on all layers.
        for layer in 0..n_layers {
            let k_read = cache.read_k(layer);
            assert_eq!(k_read.len(), 3 * stride, "layer {layer} K length");
            for (i, (&got, &expected)) in k_read.iter().zip(k_data.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "layer {layer} K[{i}]: got={got} expected={expected}",
                );
            }
        }

        // Boundary layers (0, 3): V is F32 → lossless.
        for layer in [0u32, 3] {
            let v_read = cache.read_v_owned(layer);
            assert_eq!(v_read.len(), 3 * stride, "boundary layer {layer} V length");
            for (i, (&got, &expected)) in v_read.iter().zip(v_data.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "boundary layer {layer} V[{i}]: got={got} expected={expected}",
                );
            }
        }

        // Inner layers (1, 2): V is Q4+WHT → lossy but close.
        for layer in [1u32, 2] {
            let v_read = cache.read_v_owned(layer);
            assert_eq!(v_read.len(), 3 * stride, "inner layer {layer} V length");
            for (i, (&got, &expected)) in v_read.iter().zip(v_data.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 0.5,
                    "inner layer {layer} V[{i}]: got={got} expected={expected} (Q4 tolerance)",
                );
            }
        }
    }

    #[test]
    fn asymmetric_q8_k_q4_v() {
        // K at Q8, V at Q4 — independent quantization paths.
        let config = KvCacheConfig {
            k_bits: KvBits::Q8,
            v_bits: KvBits::Q4,
            boundary_layers: 0, // no boundary protection
            wht_rotation: false,
        };
        let mut cache = KvCacheState::with_config(1, 2, 4, 16, config);
        let stride = 2 * 4;
        let k_data: Vec<f32> = (0..2 * stride).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let v_data: Vec<f32> = (0..2 * stride).map(|i| (i as f32) * 0.2 + 0.5).collect();
        cache.write_layer(0, &k_data, &v_data);
        cache.advance(2);

        // K: Q8 tolerance (~0.05 per element).
        let k_read = cache.read_k_owned(0);
        for (i, (&got, &expected)) in k_read.iter().zip(k_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.05,
                "K Q8[{i}]: got={got} expected={expected}",
            );
        }

        // V: Q4 tolerance (~0.5 per element).
        let v_read = cache.read_v_owned(0);
        for (i, (&got, &expected)) in v_read.iter().zip(v_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.5,
                "V Q4[{i}]: got={got} expected={expected}",
            );
        }
    }

    #[test]
    fn kv_cache_memory_savings() {
        // Verify Q8 and Q4 buffers use less memory than F32.
        let n_heads = 4usize;
        let head_dim = 8usize;
        let max_seq = 64;
        let stride = n_heads * head_dim;

        let cache_f32 = KvCacheState::new(1, n_heads as u32, head_dim as u32, max_seq);
        let cache_q8 = KvCacheState::with_config(
            1,
            n_heads as u32,
            head_dim as u32,
            max_seq,
            KvCacheConfig {
                k_bits: KvBits::Q8,
                v_bits: KvBits::Q8,
                boundary_layers: 0,
                wht_rotation: false,
            },
        );
        let cache_q4 = KvCacheState::with_config(
            1,
            n_heads as u32,
            head_dim as u32,
            max_seq,
            KvCacheConfig {
                k_bits: KvBits::Q4,
                v_bits: KvBits::Q4,
                boundary_layers: 0,
                wht_rotation: false,
            },
        );

        // Trigger allocation by writing one token.
        let data: Vec<f32> = vec![1.0; stride];
        let mut caches = [cache_f32, cache_q8, cache_q4];
        for c in &mut caches {
            c.write_layer(0, &data, &data);
        }

        // F32: max_seq * stride * 4 bytes per buffer
        let f32_size = max_seq * stride * 4;
        // Q8: max_seq * stride * 1 byte (indices) + max_seq * n_heads * 8 bytes (params)
        let q8_size = max_seq * stride + max_seq * n_heads * std::mem::size_of::<ChannelParams>();
        // Q4: max_seq * n_heads * ceil(head_dim/2) + max_seq * n_heads * 8 bytes (params)
        let q4_size = max_seq * n_heads * head_dim.div_ceil(2)
            + max_seq * n_heads * std::mem::size_of::<ChannelParams>();

        // Q8 should be at most half of F32 (indices + params overhead).
        assert!(
            q8_size <= f32_size / 2,
            "Q8 ({q8_size}) should be <= F32/2 ({})",
            f32_size / 2
        );
        // Q4 should be smaller than Q8.
        assert!(
            q4_size < q8_size,
            "Q4 ({q4_size}) should be < Q8 ({q8_size})"
        );
    }
}
