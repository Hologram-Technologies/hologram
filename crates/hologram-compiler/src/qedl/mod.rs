//! QEDL (Quantize/Encode/Dequantize/Lift) boundary analysis.
//!
//! At compile time, detects domain crossings between byte-domain (Z/256Z) ops
//! and float-domain ops. For each crossing, selects the minimum-error encoding
//! based on the upstream op's curvature profile.
//!
//! UOR grounding: DC_5 (carry decomposition) and CF_3/CF_4 (curvature flux).
//! The curvature profile of a node's output LUT determines whether signed,
//! unsigned, raw, or angle encoding minimizes dequantization error.

pub mod pass;

use hologram_core::op::FloatOp;
use hologram_core::view::ElementWiseView;

/// Algebraic profile of a byte-domain op's output distribution.
///
/// Derived entirely from the 256-byte output LUT — no runtime data needed.
/// Used to select the optimal encoding at QEDL boundaries.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureProfile {
    /// Mean count of trailing 1-bits across all 256 outputs.
    /// Low value (< 1.5): carry-simple → Raw encoding safe.
    pub mean_trailing_ones: f32,
    /// Shannon entropy of output distribution in [0, 1].
    /// 0 = constant output, 1 = uniform distribution.
    pub output_entropy: f32,
    /// True if output spans both sides of 0x80 (signed zero boundary).
    /// Indicates signed encoding reduces dequantization error.
    pub zero_crossing: bool,
    /// True if the output LUT is a bijection (every output appears exactly once).
    pub is_bijective: bool,
}

/// Compute `CurvatureProfile` from an `ElementWiseView` output LUT.
///
/// O(256) — compile-time cost only, never called during inference.
#[must_use]
pub fn compute_profile(view: &ElementWiseView) -> CurvatureProfile {
    let table = view.table();

    let mean_trailing_ones = table.iter().map(|&x| x.trailing_ones() as f32).sum::<f32>() / 256.0;

    // Shannon entropy H = -Σ p(x) log2(p(x)), normalized to [0,1] by dividing by log2(256)=8.
    let mut counts = [0u32; 256];
    for &x in table.iter() {
        counts[x as usize] += 1;
    }
    let output_entropy = -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / 256.0;
            p * p.log2()
        })
        .sum::<f32>()
        / 8.0;

    let zero_crossing = table.iter().any(|&x| x >= 128) && table.iter().any(|&x| x < 128);

    CurvatureProfile {
        mean_trailing_ones,
        output_entropy,
        zero_crossing,
        is_bijective: view.is_bijective(),
    }
}

/// Encoding identifier for a QEDL dequantize boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodingId {
    /// Identity truncation — no scaling.
    Raw = 0,
    /// [0, 1] ↔ [0, 255] unsigned linear.
    Unsigned = 1,
    /// [-1, 1] ↔ [0, 255] signed linear (128 = 0.0).
    Signed = 2,
    /// [0, 2π) ↔ [0, 255] angle encoding.
    Angle = 3,
}

/// Select the minimum-error encoding for a QEDL dequantize boundary.
///
/// Decision tree (exhaustive match, no default fallthrough):
/// 1. zero_crossing AND downstream is additive (Add/Sub) → Signed
/// 2. bijective AND no zero_crossing → Raw (lossless identity lift)
/// 3. mean_trailing_ones < 1.5 → Raw (carry-simple)
/// 4. output_entropy < 0.3 (low-entropy / near-constant) → Unsigned
/// 5. otherwise → Signed
#[must_use]
pub fn select_encoding(profile: &CurvatureProfile, downstream: &FloatOp) -> EncodingId {
    let downstream_is_additive = matches!(downstream, FloatOp::Add | FloatOp::Sub);
    if profile.zero_crossing && downstream_is_additive {
        return EncodingId::Signed;
    }
    if profile.is_bijective && !profile.zero_crossing {
        return EncodingId::Raw;
    }
    if profile.mean_trailing_ones < 1.5 {
        return EncodingId::Raw;
    }
    if profile.output_entropy < 0.3 {
        return EncodingId::Unsigned;
    }
    EncodingId::Signed
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::LutOp;

    #[test]
    fn compute_profile_identity_view() {
        let id = ElementWiseView::identity();
        let p = compute_profile(&id);
        assert!(p.is_bijective);
        // Identity maps all 256 values → uniform distribution → entropy = 1.
        assert!(
            (p.output_entropy - 1.0).abs() < 1e-3,
            "entropy={}",
            p.output_entropy
        );
        // Identity spans both 0..127 and 128..255 → zero_crossing.
        assert!(p.zero_crossing);
    }

    #[test]
    fn compute_profile_constant_zero() {
        // Constant zero output: entropy=0, no zero_crossing (all outputs = 0 < 128).
        let view = ElementWiseView::constant(0);
        let p = compute_profile(&view);
        assert_eq!(p.output_entropy, 0.0);
        assert!(!p.zero_crossing);
        assert!(!p.is_bijective);
    }

    #[test]
    fn select_encoding_bijective_no_crossing_is_raw() {
        // Bijective and no zero_crossing → Raw.
        let profile = CurvatureProfile {
            mean_trailing_ones: 0.5,
            output_entropy: 1.0,
            zero_crossing: false,
            is_bijective: true,
        };
        assert_eq!(select_encoding(&profile, &FloatOp::Mul), EncodingId::Raw);
    }

    #[test]
    fn select_encoding_zero_crossing_additive_is_signed() {
        // zero_crossing + downstream Add → Signed.
        let profile = CurvatureProfile {
            mean_trailing_ones: 1.0,
            output_entropy: 0.8,
            zero_crossing: true,
            is_bijective: false,
        };
        assert_eq!(select_encoding(&profile, &FloatOp::Add), EncodingId::Signed);
    }

    #[test]
    fn select_encoding_low_entropy_is_unsigned() {
        // Low entropy, not zero-crossing, not bijective → Unsigned.
        let profile = CurvatureProfile {
            mean_trailing_ones: 2.0,
            output_entropy: 0.1,
            zero_crossing: false,
            is_bijective: false,
        };
        assert_eq!(
            select_encoding(&profile, &FloatOp::Mul),
            EncodingId::Unsigned
        );
    }

    #[test]
    fn select_encoding_carry_simple_is_raw() {
        // Low trailing_ones, not bijective → Raw.
        let profile = CurvatureProfile {
            mean_trailing_ones: 0.5,
            output_entropy: 0.8,
            zero_crossing: false,
            is_bijective: false,
        };
        assert_eq!(select_encoding(&profile, &FloatOp::Mul), EncodingId::Raw);
    }

    #[test]
    fn select_encoding_high_entropy_is_signed() {
        // High trailing_ones + high entropy → Signed.
        let profile = CurvatureProfile {
            mean_trailing_ones: 3.0,
            output_entropy: 0.9,
            zero_crossing: false,
            is_bijective: false,
        };
        assert_eq!(select_encoding(&profile, &FloatOp::Mul), EncodingId::Signed);
    }

    #[test]
    fn relu_profile_low_entropy() {
        use hologram_graph::graph::GraphOp;
        let relu_view = GraphOp::Lut(LutOp::Relu)
            .to_view()
            .unwrap_or_else(ElementWiseView::identity);
        let p = compute_profile(&relu_view);
        // ReLU clips negative inputs to 0 → output is NOT a bijection.
        assert!(!p.is_bijective);
    }
}
