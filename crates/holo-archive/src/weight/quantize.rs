//! Quantization schemes and parameters for weight storage.

/// Quantization scheme identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[archive(check_bytes)]
pub enum QuantizationScheme {
    /// No quantization.
    None,
    /// Symmetric linear quantization.
    SymmetricLinear,
    /// Asymmetric affine quantization.
    AsymmetricAffine,
    /// Per-group quantization (GPTQ/AWQ style).
    PerGroup,
    /// K-means clustered quantization (LUT-GEMM style).
    KMeansClustered {
        /// Number of quantization bits (4 or 8).
        bits: u8,
    },
}

/// Parameters for quantized weight storage.
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
pub struct QuantizationParams {
    /// Quantization scheme.
    pub scheme: QuantizationScheme,
    /// Scale factor.
    pub scale: f64,
    /// Zero-point offset.
    pub zero_point: f64,
    /// Minimum clamp value.
    pub min_val: f64,
    /// Maximum clamp value.
    pub max_val: f64,
    /// Group size for per-group quantization (0 = per-tensor).
    pub group_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rkyv_round_trip() {
        let p = QuantizationParams {
            scheme: QuantizationScheme::SymmetricLinear,
            scale: 0.125,
            zero_point: 0.0,
            min_val: -1.0,
            max_val: 1.0,
            group_size: 0,
        };
        let bytes = rkyv::to_bytes::<_, 256>(&p).unwrap();
        let archived = rkyv::check_archived_root::<QuantizationParams>(&bytes).unwrap();
        assert_eq!(archived.scale, 0.125);
        assert_eq!(archived.group_size, 0);
    }

    #[test]
    fn scheme_equality() {
        assert_eq!(QuantizationScheme::None, QuantizationScheme::None);
        assert_ne!(QuantizationScheme::None, QuantizationScheme::PerGroup);
    }

    #[test]
    fn kmeans_clustered_rkyv_roundtrip() {
        use rkyv::Deserialize;
        let scheme = QuantizationScheme::KMeansClustered { bits: 4 };
        let bytes = rkyv::to_bytes::<_, 64>(&scheme).unwrap();
        let archived = rkyv::check_archived_root::<QuantizationScheme>(&bytes).unwrap();
        let deserialized: QuantizationScheme = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(deserialized, scheme);
    }

    #[test]
    fn kmeans_clustered_equality() {
        let a = QuantizationScheme::KMeansClustered { bits: 4 };
        let b = QuantizationScheme::KMeansClustered { bits: 8 };
        assert_ne!(a, b);
        assert_eq!(a, QuantizationScheme::KMeansClustered { bits: 4 });
    }
}
