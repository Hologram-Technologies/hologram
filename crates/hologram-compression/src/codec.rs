//! Core compression types: traits, blocks, modes, and statistics.

use alloc::vec::Vec;

/// Compression mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompressionMode {
    /// Ring-differential coding + ANS entropy.
    Generic = 0,
    /// Stratum-partitioned entropy coding (SPEC).
    Stratum = 1,
    /// IEEE 754 byte-plane transpose + per-plane coding.
    Float = 2,
    /// Orbit-torus blocked coding for quantized weights.
    Quantized = 3,
}

impl CompressionMode {
    /// Parse from a raw byte tag.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Generic),
            1 => Some(Self::Stratum),
            2 => Some(Self::Float),
            3 => Some(Self::Quantized),
            _ => None,
        }
    }
}

/// A compressed block with metadata for lossless round-trip.
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// The compressed byte payload.
    pub data: Vec<u8>,
    /// Original uncompressed length in bytes.
    pub original_len: usize,
    /// Which compression mode was used.
    pub mode: CompressionMode,
}

/// Statistics about a compression operation.
#[derive(Debug, Clone, Copy)]
pub struct CompressionStats {
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Compression ratio (original / compressed).
    pub ratio: f64,
    /// Stratum histogram: count of bytes with popcount 0..=8.
    pub stratum_histogram: [u32; 9],
}

impl CompressionStats {
    /// Compute stats from original data and compressed output.
    pub fn compute(original: &[u8], compressed: &CompressedBlock) -> Self {
        let mut histogram = [0u32; 9];
        for &byte in original {
            let s = hologram_core::lut::q0::stratum_q0(byte) as usize;
            histogram[s] += 1;
        }
        let original_size = original.len();
        let compressed_size = compressed.data.len();
        Self {
            original_size,
            compressed_size,
            ratio: if compressed_size > 0 {
                original_size as f64 / compressed_size as f64
            } else {
                f64::INFINITY
            },
            stratum_histogram: histogram,
        }
    }
}
