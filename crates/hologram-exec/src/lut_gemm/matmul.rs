//! Sequential LUT-GEMM matrix multiplication kernels.
//!
//! For each output element C[i,j], builds a Psumbook of activation values
//! grouped by quantized weight index, then dots with centroids.

use super::psumbook::{Psumbook4, Psumbook8};
use super::psumbook_q1::HierarchicalPsumbook16;
use super::quantize::{get_q4_index, QuantizedWeights, QuantizedWeights4, QuantizedWeights8};
use super::quantize_q1::QuantizedWeights16;

/// Compute one output element for Q4: build psumbook, dot with centroids.
#[allow(dead_code)]
fn compute_element_q4(a_row: &[f32], weights: &QuantizedWeights4, col: u32) -> f32 {
    let mut book = Psumbook4::new();
    for (l, &a_val) in a_row.iter().enumerate() {
        let idx = get_q4_index(&weights.indices, l as u32, col, weights.cols);
        book.accumulate(idx, a_val);
    }
    book.dot(&weights.centroids)
}

/// Compute one output element for Q8: build psumbook, dot with orbit-compressed centroids.
fn compute_element_q8(a_row: &[f32], weights: &QuantizedWeights8, col: u32) -> f32 {
    let mut book = Psumbook8::new();
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    for (l, &a_val) in a_row.iter().enumerate().take(k) {
        let idx = weights.indices[l * n + col as usize];
        book.accumulate(idx, a_val);
    }
    book.dot_orbits(&weights.centroids, &weights.orbits)
}

/// LUT-GEMM with 4-bit quantized weights.
///
/// `activations`: row-major M×K matrix (f32).
/// `weights`: K×N quantized weight matrix.
/// `output`: row-major M×N output buffer (f32).
pub fn lut_gemm_4bit(activations: &[f32], weights: &QuantizedWeights4, output: &mut [f32]) {
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = activations.len() / k;
    let centroids = &weights.centroids;

    // hologram LUT approach: row-major streaming with pre-multiplied centroid table.
    // For each K-row: premul[c] = activation[k] * centroids[c], then stream through
    // packed index bytes accumulating premul[idx] into output columns.
    //
    // Data read: 0.5 GB Q4 indices (8x less than f32 BLAS).
    // Centroids (64 bytes) + premul (64 bytes) in L1.
    // Output vector (N×4 bytes) in L1/L2.
    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];
        out_row.fill(0.0);

        for (l, &a_val) in a_row.iter().enumerate() {
            // Pre-multiply all 16 centroids by this activation value.
            let mut premul = [0.0f32; 16];
            for c in 0..16 {
                premul[c] = a_val * centroids[c];
            }

            // Stream through this row's packed index bytes.
            let byte_start = (l * n) / 2;
            let n_bytes = n / 2;
            let idx_row = &weights.indices[byte_start..byte_start + n_bytes];

            // Unrolled inner loop: 4 bytes (8 columns) per iteration.
            // The CPU's OoO execution overlaps load/add across iterations.
            let chunks4 = n_bytes / 4;
            let out_ptr = out_row.as_mut_ptr();
            let premul_ptr = premul.as_ptr();
            for chunk in 0..chunks4 {
                let base = chunk * 4;
                unsafe {
                    let b0 = *idx_row.get_unchecked(base);
                    let b1 = *idx_row.get_unchecked(base + 1);
                    let b2 = *idx_row.get_unchecked(base + 2);
                    let b3 = *idx_row.get_unchecked(base + 3);

                    let c = base * 2;
                    *out_ptr.add(c)     += *premul_ptr.add((b0 >> 4) as usize);
                    *out_ptr.add(c + 1) += *premul_ptr.add((b0 & 0xF) as usize);
                    *out_ptr.add(c + 2) += *premul_ptr.add((b1 >> 4) as usize);
                    *out_ptr.add(c + 3) += *premul_ptr.add((b1 & 0xF) as usize);
                    *out_ptr.add(c + 4) += *premul_ptr.add((b2 >> 4) as usize);
                    *out_ptr.add(c + 5) += *premul_ptr.add((b2 & 0xF) as usize);
                    *out_ptr.add(c + 6) += *premul_ptr.add((b3 >> 4) as usize);
                    *out_ptr.add(c + 7) += *premul_ptr.add((b3 & 0xF) as usize);
                }
            }
            // Remainder.
            for b in (chunks4 * 4)..n_bytes {
                let packed = idx_row[b];
                let col = b * 2;
                out_row[col] += premul[(packed >> 4) as usize];
                out_row[col + 1] += premul[(packed & 0x0F) as usize];
            }
        }
    }
}

/// Minimum k dimension for fiber-ordered accumulation to be beneficial.
///
/// Below this threshold, the 16× pass overhead outweighs the cache-locality gain.
pub const FIBER_THRESHOLD: usize = 512;

/// LUT-GEMM with 8-bit quantized weights.
///
/// `activations`: row-major M×K matrix (f32).
/// `weights`: K×N quantized weight matrix.
/// `output`: row-major M×N output buffer (f32).
///
/// Dispatch:
/// - k ≥ FIBER_THRESHOLD AND n ≥ 4 → fiber-ordered kernel (cache-local accumulation)
/// - n ≥ 4 → tiled kernel (activation-sharing across columns)
/// - otherwise → element-at-a-time fallback
pub fn lut_gemm_8bit(activations: &[f32], weights: &QuantizedWeights8, output: &mut [f32]) {
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    if k >= FIBER_THRESHOLD && n >= 4 {
        lut_gemm_8bit_fiber(activations, weights, output);
    } else if n >= 4 {
        lut_gemm_8bit_tiled(activations, weights, output);
    } else {
        let m = activations.len() / k;
        for i in 0..m {
            let a_row = &activations[i * k..(i + 1) * k];
            for j in 0..n {
                output[i * n + j] = compute_element_q8(a_row, weights, j as u32);
            }
        }
    }
}

/// Fiber-ordered Q8 LUT-GEMM: two-pass radix accumulation by high nibble.
///
/// For each output element, splits the k-dimension into 16 high-nibble groups
/// (each group covers weight indices [hi*16, hi*16+16)). Each group's write target
/// is 64 bytes (16 × f32) — exactly one L1 cache line. This eliminates the
/// cache-line thrashing that occurs when consecutive activations land on
/// different 64-byte slots in the flat 1024-byte Psumbook8.
///
/// Trade-off: reads the weight index array 16× (one pass per nibble group)
/// but all writes within a pass target the same cache line.
/// Zero heap allocation: psumbook is stack-allocated (1 KB).
#[allow(clippy::needless_range_loop)]
#[inline]
pub fn lut_gemm_8bit_fiber(activations: &[f32], weights: &QuantizedWeights8, output: &mut [f32]) {
    use super::psumbook::Psumbook8;
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = activations.len() / k;
    debug_assert_eq!(output.len(), m * n);
    debug_assert_eq!(weights.indices.len(), k * n);

    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        for j in 0..n {
            let mut book = Psumbook8::zeroed();
            // 16 passes, one per high-nibble group.
            // Each pass writes only to sums[hi*16 .. hi*16+16] — a single cache line.
            for hi in 0u8..16 {
                let lo_bound = (hi as usize) * 16;
                let hi_bound = lo_bound + 16;
                for l in 0..k {
                    let idx = weights.indices[l * n + j] as usize;
                    if idx >= lo_bound && idx < hi_bound {
                        book.accumulate(idx as u8, a_row[l]);
                    }
                }
            }
            output[i * n + j] = book.dot_orbits(&weights.centroids, &weights.orbits);
        }
    }
}

/// Tiled LUT-GEMM for Q8: process TILE_N output columns simultaneously.
///
/// For each activation row, reads the activation value once and scatters it
/// into TILE_N Psumbooks simultaneously. This shares the activation cache line
/// across multiple columns, improving L1 hit rate vs. the column-at-a-time kernel.
pub fn lut_gemm_8bit_tiled(activations: &[f32], weights: &QuantizedWeights8, output: &mut [f32]) {
    const TILE_N: usize = 4;
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = activations.len() / k;

    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let o_row = &mut output[i * n..(i + 1) * n];

        // Process columns in tiles of TILE_N.
        let mut j = 0;
        while j + TILE_N <= n {
            let mut books = [
                Psumbook8::new(),
                Psumbook8::new(),
                Psumbook8::new(),
                Psumbook8::new(),
            ];

            // Share each activation value across all TILE_N columns.
            for (l, &a_val) in a_row.iter().enumerate().take(k) {
                let base = l * n + j;
                books[0].accumulate(weights.indices[base], a_val);
                books[1].accumulate(weights.indices[base + 1], a_val);
                books[2].accumulate(weights.indices[base + 2], a_val);
                books[3].accumulate(weights.indices[base + 3], a_val);
            }

            o_row[j] = books[0].dot_orbits(&weights.centroids, &weights.orbits);
            o_row[j + 1] = books[1].dot_orbits(&weights.centroids, &weights.orbits);
            o_row[j + 2] = books[2].dot_orbits(&weights.centroids, &weights.orbits);
            o_row[j + 3] = books[3].dot_orbits(&weights.centroids, &weights.orbits);
            j += TILE_N;
        }

        // Handle remaining columns.
        while j < n {
            o_row[j] = compute_element_q8(a_row, weights, j as u32);
            j += 1;
        }
    }
}

/// Q16 hierarchical LUT-GEMM.
///
/// Allocates one `HierarchicalPsumbook16` outside the column loop and resets it
/// between columns. No allocation in the inner loop.
#[allow(clippy::needless_range_loop)]
pub fn lut_gemm_16bit(activations: &[f32], weights: &QuantizedWeights16, output: &mut [f32]) {
    let k = weights.rows as usize;
    let n = weights.cols as usize;
    let m = if k > 0 { activations.len() / k } else { 0 };
    debug_assert_eq!(output.len(), m * n);
    debug_assert_eq!(weights.indices.len(), k * n);

    // Allocate psumbook once per call; reset between output elements.
    let mut book = HierarchicalPsumbook16::from_tags(weights.page_tags);

    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        for j in 0..n {
            book.reset();
            for l in 0..k {
                let idx = weights.indices[l * n + j];
                book.accumulate(idx, a_row[l]);
            }
            output[i * n + j] = book.dot(&weights.params);
        }
    }
}

/// Unified LUT-GEMM dispatching to Q4 or Q8.
pub fn lut_gemm(activations: &[f32], weights: &QuantizedWeights, output: &mut [f32]) {
    match weights {
        QuantizedWeights::Q4(w) => lut_gemm_4bit(activations, w, output),
        QuantizedWeights::Q8(w) => lut_gemm_8bit(activations, w, output),
    }
}

/// Naive f32 matrix multiply (reference implementation for tests).
///
/// C[i,j] = Σ_l A[i,l] × B[l,j], all row-major.
pub fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
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
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
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
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
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
        let weights: Vec<f32> = (0..k * n).map(|i| ((i as f32) - 64.0) * 0.01).collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
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
        let weights: Vec<f32> = (0..k * n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let qw = quantize_4bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.05).collect();
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
        let activations: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
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

    /// fiber and tiled produce bitwise-identical output.
    ///
    /// Both kernels iterate l=0..k in the same order for each bucket,
    /// so f32 accumulation order is identical despite the 16-pass structure.
    #[test]
    fn fiber_matches_tiled_exhaustive() {
        let m = 4;
        let k = 512;
        let n = 4;
        // Deterministic pseudo-random weights spread across all 256 centroid indices.
        let weights: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 1.6180339) % 1.0) * 2.0 - 1.0)
            .collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * std::f32::consts::E) % 1.0) * 2.0 - 1.0)
            .collect();

        let mut out_fiber = vec![0.0f32; m * n];
        let mut out_tiled = vec![0.0f32; m * n];
        lut_gemm_8bit_fiber(&activations, &qw, &mut out_fiber);
        lut_gemm_8bit_tiled(&activations, &qw, &mut out_tiled);
        for (idx, (a, b)) in out_fiber.iter().zip(out_tiled.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fiber vs tiled mismatch at output[{idx}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn fiber_vs_naive_large() {
        // Fiber output must approximate naive f32 matmul within Q8 quantization error.
        let m = 2;
        let k = 512;
        let n = 4;
        let weights: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * std::f32::consts::SQRT_2) % 1.0) * 0.1)
            .collect();
        let qw = quantize_8bit(&weights, k as u32, n as u32);
        let activations: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 2.236_068) % 1.0) * 0.5)
            .collect();
        let mut lut_out = vec![0.0f32; m * n];
        let mut naive_out = vec![0.0f32; m * n];
        lut_gemm_8bit_fiber(&activations, &qw, &mut lut_out);
        naive_matmul(&activations, &weights, &mut naive_out, m, k, n);
        let err = max_relative_error(&naive_out, &lut_out);
        assert!(err < 0.05, "fiber Q8 512×4 error too high: {err}");
    }
}
