//! Sequential LUT-GEMM matrix multiplication kernels.
//!
//! For each output element C[i,j], builds a Psumbook of activation values
//! grouped by quantized weight index, then dots with centroids.

use super::psumbook::{Psumbook4, Psumbook8};
use super::quantize::{
    get_q4_index, QuantizedWeights, QuantizedWeights4, QuantizedWeights8,
};

/// Compute one output element for Q4: build psumbook, dot with centroids.
fn compute_element_q4(
    a_row: &[f32],
    weights: &QuantizedWeights4,
    col: u32,
) -> f32 {
    let mut book = Psumbook4::new();
    for (l, &a_val) in a_row.iter().enumerate() {
        let idx = get_q4_index(&weights.indices, l as u32, col, weights.cols);
        book.accumulate(idx, a_val);
    }
    book.dot(&weights.centroids)
}

/// Compute one output element for Q8: build psumbook, dot with centroids.
fn compute_element_q8(
    a_row: &[f32],
    weights: &QuantizedWeights8,
    col: u32,
) -> f32 {
    let mut book = Psumbook8::new();
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    for (l, &a_val) in a_row.iter().enumerate().take(k) {
        let idx = weights.indices[l * n + col as usize];
        book.accumulate(idx, a_val);
    }
    book.dot(&weights.centroids)
}

/// LUT-GEMM with 4-bit quantized weights.
///
/// `activations`: row-major M×K matrix (f32).
/// `weights`: K×N quantized weight matrix.
/// `output`: row-major M×N output buffer (f32).
pub fn lut_gemm_4bit(
    activations: &[f32],
    weights: &QuantizedWeights4,
    output: &mut [f32],
) {
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = activations.len() / k;
    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        for j in 0..n {
            output[i * n + j] = compute_element_q4(a_row, weights, j as u32);
        }
    }
}

/// LUT-GEMM with 8-bit quantized weights.
///
/// `activations`: row-major M×K matrix (f32).
/// `weights`: K×N quantized weight matrix.
/// `output`: row-major M×N output buffer (f32).
pub fn lut_gemm_8bit(
    activations: &[f32],
    weights: &QuantizedWeights8,
    output: &mut [f32],
) {
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = activations.len() / k;
    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        for j in 0..n {
            output[i * n + j] = compute_element_q8(a_row, weights, j as u32);
        }
    }
}

/// Unified LUT-GEMM dispatching to Q4 or Q8.
pub fn lut_gemm(
    activations: &[f32],
    weights: &QuantizedWeights,
    output: &mut [f32],
) {
    match weights {
        QuantizedWeights::Q4(w) => lut_gemm_4bit(activations, w, output),
        QuantizedWeights::Q8(w) => lut_gemm_8bit(activations, w, output),
    }
}

/// Naive f32 matrix multiply (reference implementation for tests).
///
/// C[i,j] = Σ_l A[i,l] × B[l,j], all row-major.
pub fn naive_matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Maximum relative error between two output buffers.
pub fn max_relative_error(expected: &[f32], actual: &[f32]) -> f32 {
    expected
        .iter()
        .zip(actual.iter())
        .map(|(&e, &a)| {
            if e.abs() < 1e-8 {
                (a - e).abs()
            } else {
                ((a - e) / e).abs()
            }
        })
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut_gemm::quantize::{quantize_4bit, quantize_8bit};

    #[test]
    fn naive_matmul_2x3_times_3x2() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let mut c = [0.0f32; 4]; // 2×2
        naive_matmul(&a, &b, &mut c, 2, 3, 2);
        assert!((c[0] - 58.0).abs() < 1e-4); // 1*7+2*9+3*11
        assert!((c[1] - 64.0).abs() < 1e-4); // 1*8+2*10+3*12
        assert!((c[2] - 139.0).abs() < 1e-4); // 4*7+5*9+6*11
        assert!((c[3] - 154.0).abs() < 1e-4); // 4*8+5*10+6*12
    }

    #[test]
    fn lut_gemm_4bit_identity_like() {
        // Constant weights → all outputs should be sum(row) * centroid
        let k = 4;
        let n = 2;
        let weights = vec![1.0f32; k * n];
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations = vec![1.0, 2.0, 3.0, 4.0]; // 1×4
        let mut output = vec![0.0f32; n];
        lut_gemm_4bit(&activations, &qw, &mut output);
        // Expected: sum(1+2+3+4) * 1.0 = 10.0 for each col
        for &v in &output {
            assert!((v - 10.0).abs() < 0.5, "got {v}, expected ~10.0");
        }
    }

    #[test]
    fn lut_gemm_8bit_identity_like() {
        let k = 4;
        let n = 2;
        let weights = vec![1.0f32; k * n];
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; n];
        lut_gemm_8bit(&activations, &qw, &mut output);
        for &v in &output {
            assert!((v - 10.0).abs() < 0.1, "got {v}, expected ~10.0");
        }
    }

    #[test]
    fn lut_gemm_q4_vs_naive_4x4() {
        let k = 4;
        let n = 4;
        let m = 4;
        let weights: Vec<f32> = (0..k * n)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let mut lut_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        lut_gemm_4bit(&activations, &qw, &mut lut_out);
        naive_matmul(&activations, &weights, &mut naive_out, m, k, n);
        let err = max_relative_error(&naive_out, &lut_out);
        assert!(err < 0.15, "Q4 4×4 error too high: {err}");
    }

    #[test]
    fn lut_gemm_q8_vs_naive_4x4() {
        let k = 4;
        let n = 4;
        let m = 4;
        let weights: Vec<f32> = (0..k * n)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let mut lut_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        lut_gemm_8bit(&activations, &qw, &mut lut_out);
        naive_matmul(&activations, &weights, &mut naive_out, m, k, n);
        let err = max_relative_error(&naive_out, &lut_out);
        assert!(err < 0.02, "Q8 4×4 error too high: {err}");
    }

    #[test]
    fn lut_gemm_q8_vs_naive_8x16() {
        let m = 4;
        let k = 8;
        let n = 16;
        let weights: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32) - 64.0) * 0.01)
            .collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let mut lut_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        lut_gemm_8bit(&activations, &qw, &mut lut_out);
        naive_matmul(&activations, &weights, &mut naive_out, m, k, n);
        let err = max_relative_error(&naive_out, &lut_out);
        assert!(err < 0.05, "Q8 8×16 error too high: {err}");
    }

    #[test]
    fn lut_gemm_unified_dispatch() {
        let k = 4;
        let n = 2;
        let weights = vec![2.0f32; k * n];
        let qw = QuantizedWeights::Q8(Box::new(quantize_8bit(&weights, k as u32, n as u32)));
        let activations = vec![1.0f32; k];
        let mut output = vec![0.0f32; n];
        lut_gemm(&activations, &qw, &mut output);
        for &v in &output {
            assert!((v - 8.0).abs() < 0.5, "got {v}, expected ~8.0");
        }
    }

    #[test]
    fn lut_gemm_q4_vs_naive_32x64() {
        let m = 2;
        let k = 32;
        let n = 64;
        // Positive weights to avoid near-zero expected values
        let weights: Vec<f32> = (0..k * n)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| (i as f32 + 1.0) * 0.05)
            .collect();
        let mut lut_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        lut_gemm_4bit(&activations, &qw, &mut lut_out);
        naive_matmul(&activations, &weights, &mut naive_out, m, k, n);
        let err = max_relative_error(&naive_out, &lut_out);
        assert!(err < 0.20, "Q4 32×64 error too high: {err}");
    }

    #[test]
    fn lut_gemm_single_element() {
        // 1×1 matmul
        let weights = vec![3.0f32];
        let qw = quantize_8bit(&weights, 1, 1);
        let activations = vec![5.0f32];
        let mut output = vec![0.0f32; 1];
        lut_gemm_8bit(&activations, &qw, &mut output);
        assert!((output[0] - 15.0).abs() < 0.5, "got {}", output[0]);
    }

    #[test]
    fn lut_gemm_multiple_rows() {
        let k = 4;
        let n = 2;
        let m = 3;
        let weights = vec![1.0f32; k * n];
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| (i + 1) as f32)
            .collect();
        let mut output = vec![0.0f32; m * n];
        lut_gemm_8bit(&activations, &qw, &mut output);
        // Row 0: sum(1+2+3+4)=10, Row 1: sum(5+6+7+8)=26, Row 2: sum(9+10+11+12)=42
        assert!((output[0] - 10.0).abs() < 0.5);
        assert!((output[2] - 26.0).abs() < 0.5);
        assert!((output[4] - 42.0).abs() < 0.5);
    }

    #[test]
    fn max_relative_error_exact() {
        let a = [1.0, 2.0, 3.0];
        assert!(max_relative_error(&a, &a) < 1e-10);
    }
}
