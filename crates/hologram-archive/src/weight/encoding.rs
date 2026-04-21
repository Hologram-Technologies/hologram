//! Tensor encoding descriptors for UOR-based weight quantization.
//!
//! A `TensorEncoding` describes how a tensor's logical f32 values are
//! encoded in the weight blob. This replaces graph-level quantization
//! rewrites with self-describing metadata that travels with the data.
//!
//! UOR interpretation:
//! - `Identity`: native Q3 (32-bit) representation
//! - `BlockQuantized`: Q3 → Q0 projection with per-block scale fiber
//! - `Clustered`: Q3 → sub-Q0 with centroid fiber (LUT-GEMM)

/// Block quantization variant identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum BlockVariant {
    /// Q4_0: 16 half-float scale + 16 bytes packed nibbles per block of 32.
    Q4_0,
    /// Q8_0: half-float scale + 32 signed bytes per block of 32.
    Q8_0,
    /// Q2_K: 2-bit with super-block structure.
    Q2K,
    /// Q4_K: 4-bit with super-block structure.
    Q4K,
    /// Q6_K: 6-bit with super-block structure.
    Q6K,
}

/// How a tensor's logical values are encoded in the weight blob.
///
/// Stored in `TensorMetadata.encoding` and `ConstantData::ContentAddressed`
/// to enable self-describing weight resolution at load time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum TensorEncoding {
    /// Raw IEEE 754 at native dtype. No transformation needed.
    /// UOR: full Q3 (32-bit ring) with all fibers free.
    Identity,

    /// Block-quantized with per-block scale fibers.
    /// UOR: Q3 → Q0 projection; each block of `block_size` elements
    /// shares a scale factor that encodes the lost precision as a fiber.
    BlockQuantized {
        /// Target bit width per element (2, 4, or 8).
        bits: u8,
        /// Number of elements per quantization block (typically 32).
        block_size: u32,
        /// Specific block format variant.
        variant: BlockVariant,
    },

    /// K-means clustered (LUT-GEMM): values mapped to centroid indices.
    /// UOR: Q3 → sub-Q0 with centroid fiber encoding the codebook.
    Clustered {
        /// Bits per index (2, 4, or 8).
        bits: u8,
        /// Number of centroids in the codebook (4, 16, or 256).
        num_centroids: u16,
        /// Matrix rows (for shape recovery during decode).
        rows: u32,
        /// Matrix columns (for shape recovery during decode).
        cols: u32,
    },
}

impl TensorEncoding {
    /// Bits per element for this encoding.
    #[must_use]
    pub const fn bits_per_element(&self) -> u8 {
        match self {
            Self::Identity => 32,
            Self::BlockQuantized { bits, .. } | Self::Clustered { bits, .. } => *bits,
        }
    }

    /// Whether this encoding requires decoding before use.
    #[must_use]
    pub const fn needs_decode(&self) -> bool {
        !matches!(self, Self::Identity)
    }

    /// Whether this is a k-means clustered encoding (LUT-GEMM).
    #[must_use]
    pub const fn is_clustered(&self) -> bool {
        matches!(self, Self::Clustered { .. })
    }

    /// Whether this is a block-quantized encoding.
    #[must_use]
    pub const fn is_block_quantized(&self) -> bool {
        matches!(self, Self::BlockQuantized { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_properties() {
        let enc = TensorEncoding::Identity;
        assert_eq!(enc.bits_per_element(), 32);
        assert!(!enc.needs_decode());
        assert!(!enc.is_clustered());
        assert!(!enc.is_block_quantized());
    }

    #[test]
    fn block_quantized_q4() {
        let enc = TensorEncoding::BlockQuantized {
            bits: 4,
            block_size: 32,
            variant: BlockVariant::Q4_0,
        };
        assert_eq!(enc.bits_per_element(), 4);
        assert!(enc.needs_decode());
        assert!(enc.is_block_quantized());
        assert!(!enc.is_clustered());
    }

    #[test]
    fn clustered_q4() {
        let enc = TensorEncoding::Clustered {
            bits: 4,
            num_centroids: 16,
            rows: 4096,
            cols: 4096,
        };
        assert_eq!(enc.bits_per_element(), 4);
        assert!(enc.needs_decode());
        assert!(enc.is_clustered());
        assert!(!enc.is_block_quantized());
    }

    #[test]
    fn rkyv_round_trip_identity() {
        let enc = TensorEncoding::Identity;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&enc).expect("serialize");
        let deserialized =
            rkyv::from_bytes::<TensorEncoding, rkyv::rancor::Error>(&bytes).expect("deserialize");
        assert_eq!(deserialized, enc);
    }

    #[test]
    fn rkyv_round_trip_block_quantized() {
        let enc = TensorEncoding::BlockQuantized {
            bits: 8,
            block_size: 32,
            variant: BlockVariant::Q8_0,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&enc).expect("serialize");
        let deserialized =
            rkyv::from_bytes::<TensorEncoding, rkyv::rancor::Error>(&bytes).expect("deserialize");
        assert_eq!(deserialized, enc);
    }

    #[test]
    fn rkyv_round_trip_clustered() {
        let enc = TensorEncoding::Clustered {
            bits: 4,
            num_centroids: 16,
            rows: 2048,
            cols: 4096,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&enc).expect("serialize");
        let deserialized =
            rkyv::from_bytes::<TensorEncoding, rkyv::rancor::Error>(&bytes).expect("deserialize");
        assert_eq!(deserialized, enc);
    }

    #[test]
    fn block_variant_rkyv_round_trip() {
        for variant in [
            BlockVariant::Q4_0,
            BlockVariant::Q8_0,
            BlockVariant::Q2K,
            BlockVariant::Q4K,
            BlockVariant::Q6K,
        ] {
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&variant).expect("serialize");
            let deserialized =
                rkyv::from_bytes::<BlockVariant, rkyv::rancor::Error>(&bytes).expect("deserialize");
            assert_eq!(deserialized, variant);
        }
    }
}
