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

    #[cfg(target_arch = "aarch64")]
    {
        lut_gemm_4bit_neon_int8(activations, weights, output, m, k, n, centroids);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        lut_gemm_4bit_scalar_int8(activations, weights, output, m, k, n, centroids);
    }
}

/// Quantize f32 activation row to int8 (per-row symmetric quantization).
/// Returns the dequant scale: `f32_val ≈ int8_val * scale`.
#[inline]
fn quantize_activation_row(a_row: &[f32], a_i8: &mut [i8]) -> f32 {
    let a_max = a_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if a_max < 1e-12 {
        a_i8.iter_mut().for_each(|v| *v = 0);
        return 0.0;
    }
    let inv_scale = 127.0 / a_max;
    for (i, &v) in a_row.iter().enumerate() {
        a_i8[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    a_max / 127.0
}

/// NEON int8 Q4 kernel: both activations and centroids are int8.
///
/// Inner loop: vqtbl1q_s8 (table lookup) → vmull_s8 (int8×int8→int16) →
/// vaddw_s16 (accumulate in int32). f32 conversion happens ONCE per output
/// element instead of per K-row. K-loop unrolled by 4 for ILP.
///
/// Op count: 18 NEON ops per 32 columns per K-row (vs 36 in the old f32 kernel).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn lut_gemm_4bit_neon_int8(
    activations: &[f32],
    weights: &QuantizedWeights4,
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    centroids: &[f32; super::psumbook::Q4_LEVELS],
) {
    use std::arch::aarch64::*;

    let n_bytes = n / 2;

    // Pre-compute fixed int8 centroid table ONCE.
    let centroid_max = centroids.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let centroid_inv_scale = if centroid_max > 1e-12 {
        127.0 / centroid_max
    } else {
        0.0
    };
    let centroid_dequant = centroid_max / 127.0;

    let mut fixed_table_i8 = [0i8; 16];
    for c in 0..16 {
        fixed_table_i8[c] = (centroids[c] * centroid_inv_scale)
            .round()
            .clamp(-127.0, 127.0) as i8;
    }

    // Reusable activation buffer (avoid allocation per row for M>1).
    let mut a_i8_buf = vec![0i8; k];

    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];

        // Quantize activation row to int8.
        let a_scale = quantize_activation_row(a_row, &mut a_i8_buf);
        let combined_scale = a_scale * centroid_dequant;

        let chunks16 = n_bytes / 16; // Each chunk = 16 packed bytes = 32 output columns.

        unsafe {
            let tbl = vld1q_s8(fixed_table_i8.as_ptr());
            let mask_lo = vdupq_n_u8(0x0F);

            for chunk in 0..chunks16 {
                let base = chunk * 16;
                let col_base = base * 2;

                // int32 accumulators: 8 registers for 32 output columns.
                // Layout: [lo_low_0..3, lo_high_0..3, hi_low_0..3, hi_high_0..3]
                // lo = even columns (low nibble), hi = odd columns (high nibble)
                let mut acc_lo_0 = vdupq_n_s32(0);
                let mut acc_lo_1 = vdupq_n_s32(0);
                let mut acc_lo_2 = vdupq_n_s32(0);
                let mut acc_lo_3 = vdupq_n_s32(0);
                let mut acc_hi_0 = vdupq_n_s32(0);
                let mut acc_hi_1 = vdupq_n_s32(0);
                let mut acc_hi_2 = vdupq_n_s32(0);
                let mut acc_hi_3 = vdupq_n_s32(0);

                // Macro for processing one K-row: lookup + widening multiply-accumulate.
                macro_rules! process_k_row {
                    ($ll:expr) => {
                        let a_val = vdup_n_s8(*a_i8_buf.get_unchecked($ll));
                        let idx_ptr = weights.indices.as_ptr().add($ll * n_bytes + base);
                        let packed = vld1q_u8(idx_ptr);

                        let lo_idx = vandq_u8(packed, mask_lo);
                        let hi_idx = vshrq_n_u8(packed, 4);
                        let lo_i8 = vqtbl1q_s8(tbl, lo_idx);
                        let hi_i8 = vqtbl1q_s8(tbl, hi_idx);

                        // vmull_s8: int8 × int8 → int16 (8 products each)
                        let lo_prod_low = vmull_s8(vget_low_s8(lo_i8), a_val);
                        let lo_prod_high = vmull_s8(vget_high_s8(lo_i8), a_val);
                        let hi_prod_low = vmull_s8(vget_low_s8(hi_i8), a_val);
                        let hi_prod_high = vmull_s8(vget_high_s8(hi_i8), a_val);

                        // vaddw_s16: int32 += int16 (widen and accumulate)
                        acc_lo_0 = vaddw_s16(acc_lo_0, vget_low_s16(lo_prod_low));
                        acc_lo_1 = vaddw_s16(acc_lo_1, vget_high_s16(lo_prod_low));
                        acc_lo_2 = vaddw_s16(acc_lo_2, vget_low_s16(lo_prod_high));
                        acc_lo_3 = vaddw_s16(acc_lo_3, vget_high_s16(lo_prod_high));
                        acc_hi_0 = vaddw_s16(acc_hi_0, vget_low_s16(hi_prod_low));
                        acc_hi_1 = vaddw_s16(acc_hi_1, vget_high_s16(hi_prod_low));
                        acc_hi_2 = vaddw_s16(acc_hi_2, vget_low_s16(hi_prod_high));
                        acc_hi_3 = vaddw_s16(acc_hi_3, vget_high_s16(hi_prod_high));
                    };
                }

                // K-loop with unroll-by-4 for instruction-level parallelism.
                let k_main = k - (k % 4);
                let mut l = 0usize;
                while l < k_main {
                    process_k_row!(l);
                    process_k_row!(l + 1);
                    process_k_row!(l + 2);
                    process_k_row!(l + 3);
                    l += 4;
                }

                // K-loop remainder (0-3 rows).
                while l < k {
                    process_k_row!(l);
                    l += 1;
                }

                // Final conversion: int32 → f32, apply combined scale, interleave hi/lo.
                let v_scale = vdupq_n_f32(combined_scale);

                macro_rules! write_group {
                    ($hi_acc:expr, $lo_acc:expr, $col_off:expr) => {
                        let hi_f32 = vmulq_f32(vcvtq_f32_s32($hi_acc), v_scale);
                        let lo_f32 = vmulq_f32(vcvtq_f32_s32($lo_acc), v_scale);
                        let z1 = vzip1q_f32(hi_f32, lo_f32);
                        let z2 = vzip2q_f32(hi_f32, lo_f32);
                        let col = col_base + $col_off;
                        vst1q_f32(out_row.as_mut_ptr().add(col), z1);
                        vst1q_f32(out_row.as_mut_ptr().add(col + 4), z2);
                    };
                }
                write_group!(acc_hi_0, acc_lo_0, 0);
                write_group!(acc_hi_1, acc_lo_1, 8);
                write_group!(acc_hi_2, acc_lo_2, 16);
                write_group!(acc_hi_3, acc_lo_3, 24);
            }

            // Scalar remainder for columns not covered by 32-wide chunks.
            let rem_start = chunks16 * 16;
            for b in rem_start..n_bytes {
                let col = b * 2;
                let mut sum_hi = 0i32;
                let mut sum_lo = 0i32;
                for (l, &a_val_i8) in a_i8_buf.iter().enumerate().take(k) {
                    let a_val = a_val_i8 as i32;
                    let packed = weights.indices[l * n_bytes + b];
                    sum_hi += a_val * fixed_table_i8[(packed >> 4) as usize] as i32;
                    sum_lo += a_val * fixed_table_i8[(packed & 0x0F) as usize] as i32;
                }
                if col < n {
                    out_row[col] = sum_hi as f32 * combined_scale;
                }
                if col + 1 < n {
                    out_row[col + 1] = sum_lo as f32 * combined_scale;
                }
            }
        }
    }
}

/// Scalar fallback with int8 activation quantization.
///
/// Same algorithm as the NEON kernel but without SIMD: quantize activations
/// to int8, multiply int8×int8→int32, convert to f32 once per output element.
#[allow(dead_code, clippy::too_many_arguments)]
fn lut_gemm_4bit_scalar_int8(
    activations: &[f32],
    weights: &QuantizedWeights4,
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    centroids: &[f32; super::psumbook::Q4_LEVELS],
) {
    let n_bytes = n / 2;

    let centroid_max = centroids.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let centroid_inv_scale = if centroid_max > 1e-12 {
        127.0 / centroid_max
    } else {
        0.0
    };
    let centroid_dequant = centroid_max / 127.0;

    let mut fixed_table_i8 = [0i8; 16];
    for c in 0..16 {
        fixed_table_i8[c] = (centroids[c] * centroid_inv_scale)
            .round()
            .clamp(-127.0, 127.0) as i8;
    }

    let mut a_i8_buf = vec![0i8; k];

    for i in 0..m {
        let a_row = &activations[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];

        let a_scale = quantize_activation_row(a_row, &mut a_i8_buf);
        let combined_scale = a_scale * centroid_dequant;

        for b in 0..n_bytes {
            let col = b * 2;
            let mut sum_hi = 0i32;
            let mut sum_lo = 0i32;
            for (l, &a_val_i8) in a_i8_buf.iter().enumerate().take(k) {
                let a_val = a_val_i8 as i32;
                let packed = weights.indices[l * n_bytes + b];
                sum_hi += a_val * fixed_table_i8[(packed >> 4) as usize] as i32;
                sum_lo += a_val * fixed_table_i8[(packed & 0x0F) as usize] as i32;
            }
            out_row[col] = sum_hi as f32 * combined_scale;
            if col + 1 < n {
                out_row[col + 1] = sum_lo as f32 * combined_scale;
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
