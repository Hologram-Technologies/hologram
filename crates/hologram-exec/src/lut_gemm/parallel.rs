//! Column-parallel LUT-GEMM via rayon.
//!
//! When the output column count exceeds `PAR_COL_THRESHOLD`, each row's
//! output columns are computed in parallel. Each thread builds its own
//! stack-allocated Psumbook (64B for Q4), avoiding false sharing.

use super::matmul::{lut_gemm_4bit, lut_gemm_8bit};
use super::psumbook::{Psumbook4, Psumbook8};
use super::quantize::{get_q4_index, QuantizedWeights, QuantizedWeights4, QuantizedWeights8};

/// Minimum output columns to justify rayon overhead.
pub const PAR_COL_THRESHOLD: usize = 64;

/// Column-parallel LUT-GEMM with 4-bit quantized weights.
#[cfg(feature = "parallel")]
pub fn lut_gemm_4bit_par(activations: &[f32], weights: &QuantizedWeights4, output: &mut [f32]) {
    let n = weights.cols as usize;
    if n < PAR_COL_THRESHOLD {
        return lut_gemm_4bit(activations, weights, output);
    }
    let k = weights.rows as usize;
    let m = activations.len() / k;
    use rayon::prelude::*;
    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];
        out_row.par_iter_mut().enumerate().for_each(|(j, c)| {
            *c = compute_col_q4(a_row, weights, j as u32);
        });
    }
}

/// Compute single Q4 output column (used by parallel path).
#[cfg(feature = "parallel")]
fn compute_col_q4(a_row: &[f32], w: &QuantizedWeights4, col: u32) -> f32 {
    let mut book = Psumbook4::new();
    for (l, &a_val) in a_row.iter().enumerate() {
        let idx = get_q4_index(&w.indices, l as u32, col, w.cols);
        book.accumulate(idx, a_val);
    }
    book.dot(&w.centroids)
}

/// Column-parallel LUT-GEMM with 8-bit quantized weights.
#[cfg(feature = "parallel")]
pub fn lut_gemm_8bit_par(activations: &[f32], weights: &QuantizedWeights8, output: &mut [f32]) {
    let n = weights.cols as usize;
    if n < PAR_COL_THRESHOLD {
        return lut_gemm_8bit(activations, weights, output);
    }
    let k = weights.rows as usize;
    let m = activations.len() / k;
    use rayon::prelude::*;
    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];
        out_row.par_iter_mut().enumerate().for_each(|(j, c)| {
            *c = compute_col_q8(a_row, weights, j as u32);
        });
    }
}

/// Compute single Q8 output column (used by parallel path).
#[cfg(feature = "parallel")]
fn compute_col_q8(a_row: &[f32], w: &QuantizedWeights8, col: u32) -> f32 {
    let mut book = Psumbook8::new();
    let n = w.cols as usize;
    for (l, &a_val) in a_row.iter().enumerate() {
        let idx = w.indices[l * n + col as usize];
        book.accumulate(idx, a_val);
    }
    book.dot(&w.centroids)
}

/// Unified column-parallel LUT-GEMM dispatching to Q4 or Q8.
#[cfg(feature = "parallel")]
pub fn lut_gemm_par(activations: &[f32], weights: &QuantizedWeights, output: &mut [f32]) {
    match weights {
        QuantizedWeights::Q4(w) => lut_gemm_4bit_par(activations, w, output),
        QuantizedWeights::Q8(w) => lut_gemm_8bit_par(activations, w, output),
    }
}

#[cfg(test)]
#[cfg(feature = "parallel")]
mod tests {
    use super::*;
    use crate::lut_gemm::matmul::{lut_gemm_4bit, lut_gemm_8bit};
    use crate::lut_gemm::quantize::{quantize_4bit, quantize_8bit};

    #[test]
    fn par_q4_matches_sequential() {
        let m = 2;
        let k = 16;
        let n = 128; // > PAR_COL_THRESHOLD
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut seq_out = vec![0.0f32; m * n];
        let mut par_out = vec![0.0f32; m * n];
        lut_gemm_4bit(&activations, &qw, &mut seq_out);
        lut_gemm_4bit_par(&activations, &qw, &mut par_out);
        // Tolerance: int8 activation quantization in the sequential kernel adds
        // ~0.4% error vs the f32 Psumbook path in the parallel kernel.
        for (s, p) in seq_out.iter().zip(par_out.iter()) {
            let tol = s.abs().max(1.0) * 0.01;
            assert!((s - p).abs() < tol, "mismatch: {s} vs {p}");
        }
    }

    #[test]
    fn par_q8_matches_sequential() {
        let m = 2;
        let k = 16;
        let n = 128;
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut seq_out = vec![0.0f32; m * n];
        let mut par_out = vec![0.0f32; m * n];
        lut_gemm_8bit(&activations, &qw, &mut seq_out);
        lut_gemm_8bit_par(&activations, &qw, &mut par_out);
        for (s, p) in seq_out.iter().zip(par_out.iter()) {
            assert!((s - p).abs() < 1e-5, "mismatch: {s} vs {p}");
        }
    }

    #[test]
    fn par_falls_back_below_threshold() {
        let m = 1;
        let k = 4;
        let n = 32; // below PAR_COL_THRESHOLD
        let weights = vec![1.0f32; k * n];
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations = vec![1.0f32; m * k];
        let mut output = vec![0.0f32; m * n];
        lut_gemm_4bit_par(&activations, &qw, &mut output);
        // Should still produce correct results via sequential fallback
        for &v in &output {
            assert!((v - 4.0).abs() < 0.5);
        }
    }

    #[test]
    fn par_unified_dispatch() {
        let m = 1;
        let k = 8;
        let n = 128;
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let qw = QuantizedWeights::Q8(Box::new(quantize_8bit(&weights, k as u32, n as u32)));
        let activations = vec![1.0f32; m * k];
        let mut output = vec![0.0f32; m * n];
        lut_gemm_par(&activations, &qw, &mut output);
        // Just verify no panic and non-zero output
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn par_q4_large_matrix() {
        let m = 4;
        let k = 64;
        let n = 256;
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut seq_out = vec![0.0f32; m * n];
        let mut par_out = vec![0.0f32; m * n];
        lut_gemm_4bit(&activations, &qw, &mut seq_out);
        lut_gemm_4bit_par(&activations, &qw, &mut par_out);
        // Tolerance: int8 activation quantization in sequential kernel.
        for (s, p) in seq_out.iter().zip(par_out.iter()) {
            let tol = s.abs().max(1.0) * 0.01;
            assert!((s - p).abs() < tol, "mismatch: {s} vs {p}");
        }
    }
}
