//! Canonical `RotaryEmbedding` op — semantic identity, executable
//! form, and CPU reference kernel.
//!
//! Half-rotation form (used by LLaMA / GPT-NeoX style transformers):
//! for each position `p` and each `k ∈ [0, dim/2)`:
//!
//! ```text
//! θ = p / base^(2k/dim)
//! x'[k]         = x[k] * cos(θ) - x[k + dim/2] * sin(θ)
//! x'[k + dim/2] = x[k] * sin(θ) + x[k + dim/2] * cos(θ)
//! ```
//!
//! Position is the row index along the seq axis — *not* an explicit
//! `position_ids` input. Variants with arbitrary positional inputs
//! need an integer-tensor canonical layer (ADR-048) and stay on
//! `FloatOp::RotaryEmbedding` for now.
//!
//! Input shape: `[..., seq, n_heads, head_dim]` (head_dim must equal
//! the `dim` attribute). 3-D and 4-D inputs are both supported via a
//! single "leading axes flatten to batch" rule in the planner.

use crate::attrs::RotaryEmbeddingAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `rotary_embedding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotaryEmbeddingCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span (same length as input).
    pub output: SlotSpan,
    /// Combined leading-batch dimension (product of all dims before
    /// the seq axis; 1 if the input is 3-D).
    pub batch: u32,
    /// Sequence length.
    pub seq: u32,
    /// Heads per position.
    pub n_heads: u32,
    /// Per-head rotation dimension (must equal head_dim).
    pub dim: u32,
    /// `base` (theta), encoded as `f32::to_bits()`.
    pub base_bits: u32,
}

/// Marker struct for the canonical `rotary_embedding` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RotaryEmbedding(pub RotaryEmbeddingAttrs);

impl Op for RotaryEmbedding {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "rotary_embedding"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Elementwise
    }
}

/// Forward: half-rotation rotary embedding.
pub fn rotary_embedding(storage: &mut [f32], call: &RotaryEmbeddingCall) {
    let batch = call.batch as usize;
    let seq = call.seq as usize;
    let n_heads = call.n_heads as usize;
    let dim = call.dim as usize;
    debug_assert!(dim.is_multiple_of(2), "rotary `dim` must be even");
    let half = dim / 2;
    let base = f32::from_bits(call.base_bits);
    let inv_dim = 1.0_f32 / dim as f32;
    let head_stride = dim;
    let pos_stride = n_heads * dim;
    let batch_stride = seq * pos_stride;

    for b in 0..batch {
        for p in 0..seq {
            let pos_f = p as f32;
            for h in 0..n_heads {
                let row_off = b * batch_stride + p * pos_stride + h * head_stride;
                let in_off = call.input.offset + row_off;
                let out_off = call.output.offset + row_off;
                for k in 0..half {
                    let exponent = 2.0 * k as f32 * inv_dim;
                    let theta = pos_f / libm::powf(base, exponent);
                    let cos_t = libm::cosf(theta);
                    let sin_t = libm::sinf(theta);
                    let lo = storage[in_off + k];
                    let hi = storage[in_off + k + half];
                    storage[out_off + k] = lo * cos_t - hi * sin_t;
                    storage[out_off + k + half] = lo * sin_t + hi * cos_t;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_at_position_zero_is_identity() {
        // pos=0 → all θ=0 → cos=1, sin=0 → output equals input.
        let mut s = vec![0.0_f32; 4 + 4];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let call = RotaryEmbeddingCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 4 },
            batch: 1,
            seq: 1,
            n_heads: 1,
            dim: 4,
            base_bits: 10000.0_f32.to_bits(),
        };
        rotary_embedding(&mut s, &call);
        assert_eq!(&s[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rotary_pos_one_rotates_first_pair_by_unit_angle() {
        // dim=2, two positions [0, 1]; both rows hold x=[1, 0].
        // pos=0 → identity; pos=1 with k=0 has θ = 1 / base^0 = 1, so
        // x' = (cos(1), sin(1)).
        let mut s = [1.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let call = RotaryEmbeddingCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 4 },
            batch: 1,
            seq: 2,
            n_heads: 1,
            dim: 2,
            base_bits: 10000.0_f32.to_bits(),
        };
        rotary_embedding(&mut s, &call);
        let cos1 = libm::cosf(1.0);
        let sin1 = libm::sinf(1.0);
        assert!((s[4] - 1.0).abs() < 1e-5);
        assert!((s[5] - 0.0).abs() < 1e-5);
        assert!((s[6] - cos1).abs() < 1e-5);
        assert!((s[7] - sin1).abs() < 1e-5);
    }
}
