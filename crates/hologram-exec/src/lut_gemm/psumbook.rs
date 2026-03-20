//! Cache-aligned partial sum accumulators for LUT-GEMM.
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
    sums: [f32; Q4_LEVELS],
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
    sums: [f32; Q8_LEVELS],
}

impl Psumbook8 {
    /// Create a zeroed psumbook.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            sums: [0.0; Q8_LEVELS],
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
