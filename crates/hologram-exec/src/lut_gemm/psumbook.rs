//! Cache-aligned partial sum accumulators for LUT-GEMM.
//!
//! Psumbooks accumulate activation values grouped by weight index,
//! then produce the output element via a centroid dot product.
//!
//! A Psumbook groups activation values by their corresponding quantized
//! weight index. After accumulation, a dot product with centroids produces
//! the matmul output element in O(Q) instead of O(k).

/// Number of quantization levels for 4-bit weights.
pub const Q4_LEVELS: usize = 16;

/// Number of quantization levels for 8-bit weights.
pub const Q8_LEVELS: usize = 256;

/// Partial sum book for 4-bit quantized weights.
///
/// Fits in exactly one 64-byte cache line. Each slot accumulates
/// the sum of activations whose corresponding weight maps to that index.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct Psumbook4 {
    pub(crate) sums: [f32; Q4_LEVELS],
}

impl Psumbook4 {
    /// Create a zeroed psumbook.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            sums: [0.0; Q4_LEVELS],
        }
    }

    /// Add `value` to the bucket at `index`.
    #[inline]
    pub fn accumulate(&mut self, index: u8, value: f32) {
        self.sums[index as usize] += value;
    }

    /// Dot product of accumulated sums with centroids.
    ///
    /// Uses iterator zip+map+sum pattern which LLVM reliably autovectorizes
    /// to SIMD (SSE/AVX2/NEON) at opt-level ≥ 2.
    #[inline]
    #[must_use]
    pub fn dot(&self, centroids: &[f32; Q4_LEVELS]) -> f32 {
        self.sums
            .iter()
            .zip(centroids.iter())
            .map(|(&s, &c)| s * c)
            .sum()
    }

    /// Orbit-compressed dot product.
    ///
    /// Combines each orbit pair's contributions before the centroid multiply,
    /// reducing centroid MACs from Q to `orbits.rep_count`.
    ///
    /// Two-pass O(Q):
    /// 1. Combine sums by representative (each entry contributes once).
    /// 2. Multiply only representative entries by their centroids.
    #[inline]
    #[must_use]
    pub fn dot_orbits(
        &self,
        centroids: &[f32; Q4_LEVELS],
        orbits: &super::orbit::OrbitMap4,
    ) -> f32 {
        let mut combined = [0.0f32; Q4_LEVELS];
        for i in 0..Q4_LEVELS {
            let (rep, sign) = orbits.entries[i];
            combined[rep as usize] += self.sums[i] * (sign as f32);
        }
        let mut result = 0.0f32;
        for i in 0..Q4_LEVELS {
            if orbits.entries[i].0 as usize == i {
                result += centroids[i] * combined[i];
            }
        }
        result
    }

    /// Reset all sums to zero for reuse.
    #[inline]
    pub fn reset(&mut self) {
        self.sums = [0.0; Q4_LEVELS];
    }
}

impl Default for Psumbook4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Partial sum book for 8-bit quantized weights.
///
/// 1024 bytes (256 × 4). Each slot accumulates the sum of activations
/// whose corresponding weight maps to that index.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct Psumbook8 {
    pub(crate) sums: [f32; Q8_LEVELS],
}

impl Psumbook8 {
    /// Create a zeroed psumbook. No heap allocation — 1 KB stack value.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            sums: [0.0; Q8_LEVELS],
        }
    }

    /// Alias for `new()` — stack-allocate a zero-initialized Psumbook8.
    #[inline]
    #[must_use]
    pub fn zeroed() -> Self {
        Self::new()
    }

    /// Add `value` to the bucket at `index`.
    #[inline]
    pub fn accumulate(&mut self, index: u8, value: f32) {
        self.sums[index as usize] += value;
    }

    /// Dot product of accumulated sums with centroids.
    ///
    /// Uses iterator zip+map+sum pattern which LLVM reliably autovectorizes
    /// to SIMD (SSE/AVX2/NEON) at opt-level ≥ 2. For Q8 (256 slots),
    /// this processes 8 f32s per SIMD lane on AVX2.
    #[inline]
    #[must_use]
    pub fn dot(&self, centroids: &[f32; Q8_LEVELS]) -> f32 {
        self.sums
            .iter()
            .zip(centroids.iter())
            .map(|(&s, &c)| s * c)
            .sum()
    }

    /// Orbit-compressed dot product.
    ///
    /// Same semantics as `Psumbook4::dot_orbits` but for 256 centroids.
    /// For fully symmetric Q8 models, `orbits.rep_count ≈ 128`, halving centroid MACs.
    ///
    /// When `rep_count == 256` (no compression), delegates to `dot()` directly —
    /// zero overhead for non-symmetric centroids.
    #[inline]
    #[must_use]
    pub fn dot_orbits(
        &self,
        centroids: &[f32; Q8_LEVELS],
        orbits: &super::orbit::OrbitMap8,
    ) -> f32 {
        // Fast path: no symmetry to exploit → direct SIMD-friendly dot.
        if orbits.rep_count as usize == Q8_LEVELS {
            return self.dot(centroids);
        }
        let mut combined = [0.0f32; Q8_LEVELS];
        for i in 0..Q8_LEVELS {
            let (rep, sign) = orbits.entries[i];
            combined[rep as usize] += self.sums[i] * (sign as f32);
        }
        let mut result = 0.0f32;
        for i in 0..Q8_LEVELS {
            if orbits.entries[i].0 as usize == i {
                result += centroids[i] * combined[i];
            }
        }
        result
    }

    /// Reset all sums to zero for reuse.
    #[inline]
    pub fn reset(&mut self) {
        self.sums = [0.0; Q8_LEVELS];
    }
}

impl Default for Psumbook8 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psumbook4_size() {
        assert_eq!(core::mem::size_of::<Psumbook4>(), 64);
        assert_eq!(core::mem::align_of::<Psumbook4>(), 64);
    }

    #[test]
    fn psumbook8_size() {
        assert_eq!(core::mem::size_of::<Psumbook8>(), 1024);
        assert_eq!(core::mem::align_of::<Psumbook8>(), 64);
    }

    #[test]
    fn psumbook4_accumulate_and_dot() {
        let mut book = Psumbook4::new();
        book.accumulate(0, 1.0);
        book.accumulate(0, 2.0);
        book.accumulate(1, 3.0);
        let mut centroids = [0.0f32; Q4_LEVELS];
        centroids[0] = 10.0;
        centroids[1] = 20.0;
        // (1+2)*10 + 3*20 = 30 + 60 = 90
        assert!((book.dot(&centroids) - 90.0).abs() < 1e-6);
    }

    #[test]
    fn psumbook8_accumulate_and_dot() {
        let mut book = Psumbook8::new();
        book.accumulate(255, 5.0);
        book.accumulate(0, 2.0);
        let mut centroids = [0.0f32; Q8_LEVELS];
        centroids[0] = 1.0;
        centroids[255] = 3.0;
        // 2*1 + 5*3 = 17
        assert!((book.dot(&centroids) - 17.0).abs() < 1e-6);
    }

    #[test]
    fn psumbook4_reset() {
        let mut book = Psumbook4::new();
        book.accumulate(5, 42.0);
        book.reset();
        let centroids = [1.0f32; Q4_LEVELS];
        assert!((book.dot(&centroids)).abs() < 1e-6);
    }

    #[test]
    fn psumbook8_reset() {
        let mut book = Psumbook8::new();
        book.accumulate(100, 42.0);
        book.reset();
        let centroids = [1.0f32; Q8_LEVELS];
        assert!((book.dot(&centroids)).abs() < 1e-6);
    }

    #[test]
    fn psumbook4_default() {
        let book = Psumbook4::default();
        let centroids = [1.0f32; Q4_LEVELS];
        assert!((book.dot(&centroids)).abs() < 1e-6);
    }

    #[test]
    fn psumbook8_default() {
        let book = Psumbook8::default();
        let centroids = [1.0f32; Q8_LEVELS];
        assert!((book.dot(&centroids)).abs() < 1e-6);
    }

    #[test]
    fn psumbook4_all_buckets() {
        let mut book = Psumbook4::new();
        for i in 0..Q4_LEVELS {
            book.accumulate(i as u8, (i + 1) as f32);
        }
        let centroids = [1.0f32; Q4_LEVELS];
        // sum of 1..=16 = 136
        let expected: f32 = (1..=Q4_LEVELS).map(|i| i as f32).sum();
        assert!((book.dot(&centroids) - expected).abs() < 1e-4);
    }

    #[test]
    fn psumbook4_multiple_accumulations() {
        let mut book = Psumbook4::new();
        for _ in 0..1000 {
            book.accumulate(3, 0.001);
        }
        let mut centroids = [0.0f32; Q4_LEVELS];
        centroids[3] = 1.0;
        assert!((book.dot(&centroids) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn psumbook4_negative_values() {
        let mut book = Psumbook4::new();
        book.accumulate(0, 5.0);
        book.accumulate(0, -3.0);
        let mut centroids = [0.0f32; Q4_LEVELS];
        centroids[0] = 2.0;
        // (5-3)*2 = 4
        assert!((book.dot(&centroids) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn psumbook4_dot_orbits_matches_dot() {
        use crate::lut_gemm::orbit::{build_orbit_map_q4, OrbitMap4};
        // Fully symmetric centroids: c[i] = -c[16-i mod 16] exactly.
        let mut centroids = [0.0f32; Q4_LEVELS];
        for i in 1..8usize {
            centroids[i] = i as f32;
            centroids[16 - i] = -(i as f32);
        }
        let orbits = build_orbit_map_q4(&centroids);

        let mut book = Psumbook4::new();
        // Accumulate some values into multiple buckets.
        book.accumulate(1, 2.0);
        book.accumulate(15, 3.0); // maps to neg(1)
        book.accumulate(3, 1.5);
        book.accumulate(0, 0.5);

        let std_dot = book.dot(&centroids);
        let orbit_dot = book.dot_orbits(&centroids, &orbits);
        assert!(
            (std_dot - orbit_dot).abs() < 1e-5,
            "dot_orbits diverges: {std_dot} vs {orbit_dot}"
        );
    }

    #[test]
    fn psumbook8_dot_orbits_matches_dot() {
        use crate::lut_gemm::orbit::build_orbit_map_q8;
        // Fully symmetric Q8 centroids.
        let mut centroids = [0.0f32; Q8_LEVELS];
        for i in 1..128usize {
            centroids[i] = i as f32;
            centroids[256 - i] = -(i as f32);
        }
        let orbits = build_orbit_map_q8(&centroids);

        let mut book = Psumbook8::new();
        book.accumulate(1, 2.0);
        book.accumulate(255, 3.0); // maps to neg(1) = 255
        book.accumulate(64, 1.0);
        book.accumulate(192, 4.0); // maps to neg(64) = 192
        book.accumulate(0, 0.5);

        let std_dot = book.dot(&centroids);
        let orbit_dot = book.dot_orbits(&centroids, &orbits);
        assert!(
            (std_dot - orbit_dot).abs() < 1e-4,
            "dot_orbits Q8 diverges: {std_dot} vs {orbit_dot}"
        );
    }

    #[test]
    fn psumbook8_dot_orbits_asymmetric_matches_dot() {
        use crate::lut_gemm::orbit::build_orbit_map_q8;
        // Asymmetric centroids: no compression, all reps.
        let mut centroids = [0.0f32; Q8_LEVELS];
        for i in 0..Q8_LEVELS {
            centroids[i] = (i as f32 * 1.6180339) % 7.0 + 1.0;
        }
        let orbits = build_orbit_map_q8(&centroids);
        assert_eq!(orbits.rep_count, 256);

        let mut book = Psumbook8::new();
        for i in 0..Q8_LEVELS {
            book.accumulate(i as u8, (i as f32) * 0.01);
        }

        let std_dot = book.dot(&centroids);
        let orbit_dot = book.dot_orbits(&centroids, &orbits);
        assert!(
            (std_dot - orbit_dot).abs() / std_dot.abs() < 1e-5,
            "asymmetric dot_orbits diverges: {std_dot} vs {orbit_dot}"
        );
    }

    #[test]
    fn psumbook8_sparse() {
        let mut book = Psumbook8::new();
        book.accumulate(0, 1.0);
        book.accumulate(128, 2.0);
        book.accumulate(255, 3.0);
        let mut centroids = [0.0f32; Q8_LEVELS];
        centroids[0] = 10.0;
        centroids[128] = 20.0;
        centroids[255] = 30.0;
        // 1*10 + 2*20 + 3*30 = 10+40+90 = 140
        assert!((book.dot(&centroids) - 140.0).abs() < 1e-4);
    }
}
