//! Weight tensor metadata and data types.

pub mod dedup;
pub mod quantize;

/// Data type of a stored tensor's elements.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub enum WeightDType {
    /// 32-bit float (4 bytes/element).
    F32,
    /// 64-bit float (8 bytes/element).
    F64,
    /// 16-bit float (2 bytes/element).
    F16,
    /// Brain float 16 (2 bytes/element).
    BF16,
    /// 8-bit signed integer (1 byte/element).
    I8,
    /// 8-bit unsigned integer (1 byte/element).
    U8,
    /// 16-bit signed integer (2 bytes/element).
    I16,
    /// 32-bit signed integer (4 bytes/element).
    I32,
    /// 64-bit signed integer (8 bytes/element).
    I64,
    /// 4-bit signed integer (sub-byte packing).
    I4,
}

impl WeightDType {
    /// Byte size per element (I4 returns 0: sub-byte packing).
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
            Self::I4 => 0,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I4 => "i4",
        }
    }
}

/// Metadata for a single tensor in the weight section.
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct TensorMetadata {
    /// Tensor name (e.g. "layer0.weight").
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Element data type.
    pub dtype: WeightDType,
    /// Byte offset within the weights section.
    pub offset: u64,
    /// Total byte size of this tensor's data.
    pub size: u64,
    /// Optional quantization parameters.
    pub quantization: Option<quantize::QuantizationParams>,
    /// CRC32 of this tensor's raw bytes.
    pub checksum: u32,
    /// Compression scheme: 0 = none, 1 = stratum, 2 = ring_diff, 3 = orbit_torus.
    pub compression_scheme: u8,
}

impl TensorMetadata {
    /// Number of elements (product of shape dimensions).
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_size_common() {
        assert_eq!(WeightDType::F32.byte_size(), 4);
        assert_eq!(WeightDType::F64.byte_size(), 8);
        assert_eq!(WeightDType::U8.byte_size(), 1);
        assert_eq!(WeightDType::I4.byte_size(), 0);
    }

    #[test]
    fn dtype_names() {
        assert_eq!(WeightDType::BF16.name(), "bf16");
        assert_eq!(WeightDType::I32.name(), "i32");
    }

    #[test]
    fn num_elements_scalar() {
        let m = TensorMetadata {
            name: "x".into(),
            shape: vec![],
            dtype: WeightDType::F32,
            offset: 0,
            size: 0,
            quantization: None,
            checksum: 0,
            compression_scheme: 0,
        };
        assert_eq!(m.num_elements(), 1);
    }

    #[test]
    fn num_elements_2d() {
        let m = TensorMetadata {
            name: "w".into(),
            shape: vec![3, 4],
            dtype: WeightDType::F32,
            offset: 0,
            size: 48,
            quantization: None,
            checksum: 0,
            compression_scheme: 0,
        };
        assert_eq!(m.num_elements(), 12);
    }

    #[test]
    fn rkyv_weight_dtype() {
        let dt = WeightDType::BF16;
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&dt).unwrap();
        let deserialized = rkyv::from_bytes::<WeightDType, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(deserialized, WeightDType::BF16);
    }

    #[test]
    fn rkyv_tensor_metadata() {
        let m = TensorMetadata {
            name: "bias".into(),
            shape: vec![128],
            dtype: WeightDType::F32,
            offset: 0,
            size: 512,
            quantization: None,
            checksum: 0xABCD,
            compression_scheme: 0,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&m).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<TensorMetadata>, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.name.as_str(), "bias");
        assert_eq!(archived.checksum, 0xABCD);
    }
}
