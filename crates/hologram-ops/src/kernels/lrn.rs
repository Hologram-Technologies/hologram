//! Canonical `LRN` op (Local Response Normalization) — semantic
//! identity, executable form, and CPU reference kernel.
//!
//! Cross-channel normalisation:
//! `out[n,c,h,w] = input[n,c,h,w] /
//!     (bias + alpha * sum_{i ∈ window(c)} input[n,i,h,w]²)^beta`,
//! where the window is centred on `c` and spans `size` channels.

use crate::attrs::LrnAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `lrn`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrnCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Channel count.
    pub c: u32,
    /// Spatial height.
    pub h: u32,
    /// Spatial width.
    pub w: u32,
    /// Window size (channels) for the normalisation neighbourhood.
    pub size: u32,
    /// `alpha`, encoded as `f32::to_bits()`.
    pub alpha_bits: u32,
    /// `beta`, encoded as `f32::to_bits()`.
    pub beta_bits: u32,
    /// `bias`, encoded as `f32::to_bits()`.
    pub bias_bits: u32,
}

/// Marker struct for the canonical `lrn` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lrn(pub LrnAttrs);

impl Op for Lrn {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "lrn"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Normalisation
    }
}

/// Forward: cross-channel local response normalisation.
pub fn lrn(storage: &mut [f32], call: &LrnCall) {
    let n = call.n as usize;
    let c = call.c as usize;
    let h = call.h as usize;
    let w = call.w as usize;
    let size = call.size as usize;
    let alpha = f32::from_bits(call.alpha_bits);
    let beta = f32::from_bits(call.beta_bits);
    let bias = f32::from_bits(call.bias_bits);
    let plane = h * w;
    let chw = c * plane;
    let half = size / 2;

    for ni in 0..n {
        for ci in 0..c {
            let lo = ci.saturating_sub(half);
            let hi = (ci + half + 1).min(c);
            for hi_ in 0..h {
                for wi in 0..w {
                    let mut sq = 0.0_f32;
                    for k in lo..hi {
                        let v = storage[call.input.offset + ni * chw + k * plane + hi_ * w + wi];
                        sq += v * v;
                    }
                    let scale = libm::powf(bias + alpha * sq / size as f32, beta);
                    let in_idx = call.input.offset + ni * chw + ci * plane + hi_ * w + wi;
                    let out_idx = call.output.offset + ni * chw + ci * plane + hi_ * w + wi;
                    storage[out_idx] = storage[in_idx] / scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lrn_with_unit_window_normalises_each_channel_independently() {
        // 1×2×1×1 input [3, 4]; size=1 means each channel only sees itself.
        // alpha=1, beta=1, bias=0 → out[c] = x[c] / x[c]² = 1/x[c].
        let mut s = [3.0_f32, 4.0, 0.0, 0.0];
        let call = LrnCall {
            input: SlotSpan { offset: 0, len: 2 },
            output: SlotSpan { offset: 2, len: 2 },
            n: 1,
            c: 2,
            h: 1,
            w: 1,
            size: 1,
            alpha_bits: 1.0_f32.to_bits(),
            beta_bits: 1.0_f32.to_bits(),
            bias_bits: 0_u32,
        };
        lrn(&mut s, &call);
        assert!((s[2] - 1.0 / 3.0).abs() < 1e-5);
        assert!((s[3] - 1.0 / 4.0).abs() < 1e-5);
    }
}
