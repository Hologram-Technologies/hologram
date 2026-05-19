//! Weight cache: avoids repeated rkyv deserialization per LUT-GEMM dispatch.
//!
//! The first time a quantized weight constant is accessed, it's deserialized
//! and stored. Subsequent dispatches reuse the cached version.

use std::collections::HashMap;

use hologram_graph::constant::{ConstantData, ConstantId, ConstantStore};

use crate::error::{ExecError, ExecResult};
use crate::lut_gemm::quantize::{QuantizedWeights2, QuantizedWeights4, QuantizedWeights8};
use crate::lut_gemm::quantize_q1::QuantizedWeights16;

/// Cached quantized weight variants.
enum CachedWeight {
    Q4(QuantizedWeights4),
    Q8(Box<QuantizedWeights8>),
    Q16(Box<QuantizedWeights16>),
    Q2(QuantizedWeights2),
}

/// Cache for deserialized quantized weights.
///
/// Keyed by `ConstantId`. Populated lazily on first access.
/// Eliminates repeated `rkyv::from_bytes()` calls in the LUT-GEMM hot path.
pub struct WeightCache {
    entries: HashMap<u32, CachedWeight>,
    /// Cached dequantized f32 weights for BLAS dispatch on platforms with
    /// hardware matrix multiply (AMX). Populated lazily on first access.
    /// Key: ConstantId raw value. Value: dequantized [k, n] f32 row-major.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    dequantized_f32: HashMap<u32, Vec<f32>>,
    /// Q8 dequant cache (same principle as Q4 but simpler — no nibble unpacking).
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    dequantized_f32_q8: HashMap<u32, Vec<f32>>,
}

impl WeightCache {
    /// Create an empty weight cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            dequantized_f32: HashMap::new(),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            dequantized_f32_q8: HashMap::new(),
        }
    }

    /// Number of cached dequantized f32 Q4 entries.
    #[must_use]
    pub fn dequant_f32_count(&self) -> usize {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            self.dequantized_f32.len()
        }
        #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
        {
            0
        }
    }

    /// Pre-warm the dequant cache for all Q4 constants in a tape.
    ///
    /// Scans tape instructions for MatMulLut4 ConstantIds and pre-populates
    /// the dequantized f32 cache. Called once at model load time so decode
    /// steps never pay the dequant overhead. Aligns with "never run an
    /// operation more than once at runtime" principle.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    pub fn prewarm_q4(
        &mut self,
        tape: &crate::tape::EnumTape,
        constants: &ConstantStore,
        weights: &[u8],
    ) {
        use crate::tape::TapeKernel;
        for instr in &tape.instructions {
            let cid = match &instr.kernel {
                TapeKernel::MatMulLut4(c) | TapeKernel::MatMulLut4Activation(c, _) => Some(*c),
                _ => None,
            };
            if let Some(cid) = cid {
                let key = cid.raw();
                if !self.dequantized_f32.contains_key(&key) {
                    let _ = self.get_dequantized_f32(cid, constants, weights);
                }
            }
        }
    }

    /// Pre-warm the cache for all Q8 constants in a tape.
    ///
    /// On macOS with Accelerate: populates dequantized f32 cache.
    /// On other platforms: deserializes rkyv Q8 weights.
    pub fn prewarm_q8(
        &mut self,
        tape: &crate::tape::EnumTape,
        constants: &ConstantStore,
        weights: &[u8],
    ) {
        use crate::tape::TapeKernel;
        for instr in &tape.instructions {
            let cid = match &instr.kernel {
                TapeKernel::MatMulLut8(c) => Some(*c),
                _ => None,
            };
            if let Some(cid) = cid {
                #[cfg(all(feature = "accelerate", target_os = "macos"))]
                {
                    let key = cid.raw();
                    if !self.dequantized_f32_q8.contains_key(&key) {
                        let _ = self.get_dequantized_f32_q8(cid, constants, weights);
                    }
                }
                #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                {
                    let _ = self.get_q8(cid, constants, weights);
                }
            }
        }
    }

    /// Get or create a cached dequantized f32 buffer for a Q4 weight.
    ///
    /// First access deserializes Q4 and dequantizes centroids → f32.
    /// Subsequent accesses return the cached buffer (zero-cost).
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    pub fn get_dequantized_f32(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&[f32]> {
        let key = cid.raw();
        if !self.dequantized_f32.contains_key(&key) {
            let qw = self.get_q4(cid, constants, weights)?;
            let rows = qw.rows as usize;
            let cols = qw.cols as usize;
            let total = rows * cols;
            let has_scales = !qw.row_scales.is_empty();
            let gs = if qw.group_size > 0 {
                qw.group_size as usize
            } else {
                cols // legacy per-row
            };
            let groups_per_row = if gs > 0 { cols.div_ceil(gs) } else { 1 };
            let mut buf = vec![0.0f32; total];
            for (i, o) in buf.iter_mut().enumerate() {
                let byte_idx = i / 2;
                let idx = if i % 2 == 0 {
                    (qw.indices[byte_idx] >> 4) as usize
                } else {
                    (qw.indices[byte_idx] & 0x0F) as usize
                };
                let centroid = qw.centroids[idx];
                if has_scales {
                    let row = i / cols;
                    let col = i % cols;
                    let group = col / gs;
                    *o = centroid * qw.row_scales[row * groups_per_row + group];
                } else {
                    *o = centroid;
                }
            }
            self.dequantized_f32.insert(key, buf);
        }
        Ok(self.dequantized_f32.get(&key).expect("just inserted"))
    }

    /// Get or compute a dequantized f32 matrix from a Q8 weight constant.
    ///
    /// Simpler than Q4: one byte per index (no nibble packing), 256 centroids.
    /// Dequant: `output[i] = centroids[indices[i]]`.
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    pub fn get_dequantized_f32_q8(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&[f32]> {
        let key = cid.raw();
        if !self.dequantized_f32_q8.contains_key(&key) {
            let qw = self.get_q8(cid, constants, weights)?;
            let total = qw.rows as usize * qw.cols as usize;
            let mut buf = vec![0.0f32; total];
            for (i, o) in buf.iter_mut().enumerate() {
                let idx = qw.indices[i] as usize;
                *o = qw.centroids[idx];
            }
            self.dequantized_f32_q8.insert(key, buf);
        }
        Ok(self.dequantized_f32_q8.get(&key).expect("just inserted"))
    }

    /// Get or deserialize a Q4 weight constant.
    ///
    /// Single hash probe per access via the `Entry` API — no double lookup.
    pub fn get_q4(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&QuantizedWeights4> {
        let key = cid.raw();
        let entry = match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let bytes = resolve_constant_bytes(cid, constants, weights)?;
                let qw = rkyv::from_bytes::<QuantizedWeights4, rkyv::rancor::Error>(bytes)
                    .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
                e.insert(CachedWeight::Q4(qw))
            }
        };
        match entry {
            CachedWeight::Q4(qw) => Ok(qw),
            _ => Err(ExecError::InvalidQuantization(
                "weight type mismatch".to_string(),
            )),
        }
    }

    /// Get or deserialize a Q8 weight constant.
    ///
    /// Single hash probe per access via the `Entry` API — no double lookup.
    pub fn get_q8(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&QuantizedWeights8> {
        let key = cid.raw();
        let entry = match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let bytes = resolve_constant_bytes(cid, constants, weights)?;
                let qw = rkyv::from_bytes::<QuantizedWeights8, rkyv::rancor::Error>(bytes)
                    .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
                e.insert(CachedWeight::Q8(Box::new(qw)))
            }
        };
        match entry {
            CachedWeight::Q8(qw) => Ok(qw),
            _ => Err(ExecError::InvalidQuantization(
                "weight type mismatch".to_string(),
            )),
        }
    }

    /// Get or deserialize a Q2 weight constant.
    ///
    /// Single hash probe per access via the `Entry` API — no double lookup.
    pub fn get_q2(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&QuantizedWeights2> {
        let key = cid.raw();
        let entry = match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let bytes = resolve_constant_bytes(cid, constants, weights)?;
                let qw = rkyv::from_bytes::<QuantizedWeights2, rkyv::rancor::Error>(bytes)
                    .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
                e.insert(CachedWeight::Q2(qw))
            }
        };
        match entry {
            CachedWeight::Q2(qw) => Ok(qw),
            _ => Err(ExecError::InvalidQuantization(
                "weight type mismatch".to_string(),
            )),
        }
    }

    /// Get or deserialize a Q16 weight constant.
    ///
    /// Single hash probe per access via the `Entry` API — no double lookup.
    pub fn get_q16(
        &mut self,
        cid: ConstantId,
        constants: &ConstantStore,
        weights: &[u8],
    ) -> ExecResult<&QuantizedWeights16> {
        let key = cid.raw();
        let entry = match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let bytes = resolve_constant_bytes(cid, constants, weights)?;
                let qw = rkyv::from_bytes::<QuantizedWeights16, rkyv::rancor::Error>(bytes)
                    .map_err(|e| ExecError::InvalidQuantization(e.to_string()))?;
                e.insert(CachedWeight::Q16(Box::new(qw)))
            }
        };
        match entry {
            CachedWeight::Q16(qw) => Ok(qw),
            _ => Err(ExecError::InvalidQuantization(
                "weight type mismatch".to_string(),
            )),
        }
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolve a constant ID to its raw bytes.
fn resolve_constant_bytes<'a>(
    cid: ConstantId,
    constants: &'a ConstantStore,
    weights: &'a [u8],
) -> ExecResult<&'a [u8]> {
    let data = constants
        .get(cid)
        .ok_or(ExecError::ConstantNotFound(cid.raw()))?;
    match data {
        ConstantData::Bytes(bytes) => Ok(bytes),
        ConstantData::Deferred {
            byte_size,
            source_id,
        } => {
            let start = *source_id as usize;
            let end = start + *byte_size as usize;
            if end > weights.len() {
                return Err(ExecError::ConstantNotFound(cid.raw()));
            }
            Ok(&weights[start..end])
        }
        ConstantData::ContentAddressed { .. } => {
            // Content-addressed resolution requires a ContentAddressIndex.
            // Full support added via WeightResolver trait in Plan 077 Phase 1.5.
            Err(ExecError::ConstantNotFound(cid.raw()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_default_is_empty() {
        let cache = WeightCache::new();
        assert!(cache.entries.is_empty());
    }
}
