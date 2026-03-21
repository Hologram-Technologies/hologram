//! KV cache state for autoregressive LLM generation.
//!
//! Holds per-layer K and V buffers that persist across execution calls.
//! During prefill, the full prompt's K/V are written. During decode,
//! each new token's K/V are appended and the full cache is returned
//! for attention computation.

/// Persistent KV cache state held between executor calls.
///
/// Each layer has separate K and V buffers sized for the maximum context length.
/// The `write_pos` advances by `seq_len` per call (prefill writes many positions,
/// decode writes one).
pub struct KvCacheState {
    /// Per-layer K buffers. Each has capacity `max_seq * n_kv_heads * head_dim` f32s.
    k_buffers: Vec<Vec<f32>>,
    /// Per-layer V buffers. Same capacity as K.
    v_buffers: Vec<Vec<f32>>,
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
}

impl KvCacheState {
    /// Create a new KV cache for the given model architecture.
    ///
    /// Buffers are lazily allocated on first write — construction is O(n_layers)
    /// with zero data allocation, enabling fast cache creation for models where
    /// not all layers may be used (e.g., early exit, speculative decode).
    #[must_use]
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32, max_seq: usize) -> Self {
        let k_buffers = (0..n_layers).map(|_| Vec::new()).collect();
        let v_buffers = (0..n_layers).map(|_| Vec::new()).collect();
        Self {
            k_buffers,
            v_buffers,
            write_pos: 0,
            n_kv_heads,
            head_dim,
            max_seq,
            window_size: None,
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

    /// Write K and/or V data for a layer at the current position.
    ///
    /// `k_data` and `v_data` are flat f32 slices of `seq_len * n_kv_heads * head_dim` elements.
    /// Either may be empty (skipped). Caller must call `advance` after writing all layers.
    ///
    /// Returns the number of elements written (from whichever was non-empty).
    pub fn write_layer(&mut self, layer: u32, k_data: &[f32], v_data: &[f32]) -> usize {
        let layer = layer as usize;
        if layer >= self.k_buffers.len() {
            return 0;
        }
        let stride = self.n_kv_heads as usize * self.head_dim as usize;
        if stride == 0 {
            return 0;
        }

        let data = if !k_data.is_empty() { k_data } else { v_data };
        let seq_len = data.len() / stride;
        let start = self.write_pos * stride;
        let end = start + seq_len * stride;

        let cap = self.max_seq * stride;
        // Lazy allocation: allocate on first write to this layer.
        if self.k_buffers[layer].is_empty() && cap > 0 {
            self.k_buffers[layer] = vec![0.0f32; cap];
            self.v_buffers[layer] = vec![0.0f32; cap];
        }

        if end > self.k_buffers[layer].len() {
            return 0; // would overflow max_seq
        }

        if !k_data.is_empty() {
            self.k_buffers[layer][start..end].copy_from_slice(&k_data[..seq_len * stride]);
        }
        if !v_data.is_empty() {
            self.v_buffers[layer][start..end].copy_from_slice(&v_data[..seq_len * stride]);
        }
        seq_len * stride
    }

    /// Advance the write position by `seq_len` tokens.
    /// Call after writing all layers for a step.
    pub fn advance(&mut self, seq_len: usize) {
        self.write_pos = (self.write_pos + seq_len).min(self.max_seq);
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
    #[must_use]
    pub fn read_k(&self, layer: u32) -> &[f32] {
        let layer = layer as usize;
        if layer >= self.k_buffers.len() {
            return &[];
        }
        let stride = self.n_kv_heads as usize * self.head_dim as usize;
        let start = self.read_start() * stride;
        let end = self.write_pos * stride;
        &self.k_buffers[layer][start..end]
    }

    /// Read cached V data for a layer (respects sliding window).
    #[must_use]
    pub fn read_v(&self, layer: u32) -> &[f32] {
        let layer = layer as usize;
        if layer >= self.v_buffers.len() {
            return &[];
        }
        let stride = self.n_kv_heads as usize * self.head_dim as usize;
        let start = self.read_start() * stride;
        let end = self.write_pos * stride;
        &self.v_buffers[layer][start..end]
    }

    /// Read cached K including pending (just-written, pre-advance) data.
    ///
    /// Returns `[0..(write_pos + pending_seq) * stride]` — includes both
    /// previously cached tokens and the new tokens written in this step.
    /// This enables a unified code path for prefill and decode.
    #[must_use]
    pub fn read_k_through(&self, layer: u32, pending_seq: usize) -> &[f32] {
        let layer = layer as usize;
        if layer >= self.k_buffers.len() {
            return &[];
        }
        let stride = self.n_kv_heads as usize * self.head_dim as usize;
        let end = (self.write_pos + pending_seq) * stride;
        &self.k_buffers[layer][..end]
    }

    /// Read cached V including pending (just-written, pre-advance) data.
    #[must_use]
    pub fn read_v_through(&self, layer: u32, pending_seq: usize) -> &[f32] {
        let layer = layer as usize;
        if layer >= self.v_buffers.len() {
            return &[];
        }
        let stride = self.n_kv_heads as usize * self.head_dim as usize;
        let end = (self.write_pos + pending_seq) * stride;
        &self.v_buffers[layer][..end]
    }

    /// Reset the cache for a new sequence.
    pub fn reset(&mut self) {
        self.write_pos = 0;
        // Don't need to zero the buffers — they'll be overwritten.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
