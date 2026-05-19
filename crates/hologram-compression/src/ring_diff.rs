//! Ring-Differential Coding (RDC).
//!
//! Observable: `ByteRing::sub` — reveals sequential correlation by computing
//! residuals in Z/256Z. For correlated data, residuals cluster near 0 mod 256,
//! yielding low entropy.
//!
//! The ring-differential is exact: `add(sub(x, pred), pred) == x` for all x, pred.

use alloc::vec::Vec;
use hologram_core::ring::ByteRing;

/// Prediction order for ring-differential coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PredictionOrder {
    /// Order-0: predict from previous value.
    /// `pred[i] = data[i-1]`, `pred[0] = 0`
    Zero = 0,
    /// Order-1: linear extrapolation in Z/256Z.
    /// `pred[i] = data[i-1] + (data[i-1] - data[i-2])`
    One = 1,
}

/// Compute ring-differential residuals (forward pass).
///
/// `residual[i] = sub(data[i], predictor[i])` in Z/256Z.
pub fn encode(data: &[u8], order: PredictionOrder) -> Vec<u8> {
    let mut residuals = Vec::with_capacity(data.len());
    match order {
        PredictionOrder::Zero => {
            let mut prev = 0u8;
            for &byte in data {
                residuals.push(ByteRing::sub(byte, prev));
                prev = byte;
            }
        }
        PredictionOrder::One => {
            let mut prev2 = 0u8;
            let mut prev1 = 0u8;
            for &byte in data {
                let delta = ByteRing::sub(prev1, prev2);
                let pred = ByteRing::add(prev1, delta);
                residuals.push(ByteRing::sub(byte, pred));
                prev2 = prev1;
                prev1 = byte;
            }
        }
    }
    residuals
}

/// Reconstruct original data from ring-differential residuals (inverse pass).
pub fn decode(residuals: &[u8], order: PredictionOrder) -> Vec<u8> {
    let mut data = Vec::with_capacity(residuals.len());
    match order {
        PredictionOrder::Zero => {
            let mut prev = 0u8;
            for &r in residuals {
                let byte = ByteRing::add(r, prev);
                data.push(byte);
                prev = byte;
            }
        }
        PredictionOrder::One => {
            let mut prev2 = 0u8;
            let mut prev1 = 0u8;
            for &r in residuals {
                let delta = ByteRing::sub(prev1, prev2);
                let pred = ByteRing::add(prev1, delta);
                let byte = ByteRing::add(r, pred);
                data.push(byte);
                prev2 = prev1;
                prev1 = byte;
            }
        }
    }
    data
}

/// Estimate the effectiveness of ring-differential coding by computing
/// the number of zero residuals (a proxy for compressibility).
pub fn zero_residual_fraction(data: &[u8], order: PredictionOrder) -> f64 {
    let residuals = encode(data, order);
    let zeros = residuals.iter().filter(|&&r| r == 0).count();
    zeros as f64 / residuals.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn round_trip_order0_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let residuals = encode(&data, PredictionOrder::Zero);
        let recovered = decode(&residuals, PredictionOrder::Zero);
        assert_eq!(data, recovered);
    }

    #[test]
    fn round_trip_order1_all_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let residuals = encode(&data, PredictionOrder::One);
        let recovered = decode(&residuals, PredictionOrder::One);
        assert_eq!(data, recovered);
    }

    #[test]
    fn round_trip_order0_random_pattern() {
        // A non-trivial pattern
        let data: Vec<u8> = (0..512).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let residuals = encode(&data, PredictionOrder::Zero);
        let recovered = decode(&residuals, PredictionOrder::Zero);
        assert_eq!(data, recovered);
    }

    #[test]
    fn round_trip_order1_random_pattern() {
        let data: Vec<u8> = (0..512).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let residuals = encode(&data, PredictionOrder::One);
        let recovered = decode(&residuals, PredictionOrder::One);
        assert_eq!(data, recovered);
    }

    #[test]
    fn ring_diff_identity() {
        // add(sub(x, pred), pred) == x for all x, pred
        for x in 0..=255u8 {
            for pred in 0..=255u8 {
                let residual = ByteRing::sub(x, pred);
                assert_eq!(ByteRing::add(residual, pred), x);
            }
        }
    }

    #[test]
    fn constant_data_compresses_well() {
        let data = vec![42u8; 100];
        let frac = zero_residual_fraction(&data, PredictionOrder::Zero);
        assert!(frac > 0.98); // All residuals except first should be 0
    }

    #[test]
    fn linear_ramp_order1_compresses_well() {
        // Linear ramp: 0, 1, 2, 3, ... (wrapping)
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let frac = zero_residual_fraction(&data, PredictionOrder::One);
        assert!(frac > 0.95);
    }
}
