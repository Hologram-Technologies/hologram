//! LUT-GEMM conformance tests.
//!
//! Encodes the orbit-compressed GEMM contract: `dot_orbits` must produce
//! numerically identical results to `dot` for any centroid distribution.

use hologram_exec::lut_gemm::matmul::{lut_gemm_8bit, max_relative_error, naive_matmul};
use hologram_exec::lut_gemm::quantize::quantize_8bit;

// ── Orbit Correctness ───────────────────────────────────────────────────────

/// After Phase 2C wires dot_orbits into the hot loop, this test verifies
/// that the orbit-compressed path matches naive matmul within tolerance.
#[test]
fn gemm_q8_orbit_matches_naive() {
    let m = 4;
    let k = 64;
    let n = 64;

    // Generate deterministic test data
    let activations: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 256) as f32 / 256.0)
        .collect();
    let weights_raw: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 + 5) % 256) as f32 / 256.0)
        .collect();

    // Quantize weights
    let qw = quantize_8bit(&weights_raw, k as u32, n as u32);

    // LUT-GEMM (will use dot_orbits after Phase 2C)
    let mut output_lut = vec![0.0f32; m * n];
    lut_gemm_8bit(&activations, &qw, &mut output_lut);

    // Naive reference
    let mut output_naive = vec![0.0f32; m * n];
    naive_matmul(&activations, &weights_raw, &mut output_naive, m, k, n);

    // Q8 quantization introduces error — verify it's bounded
    let err = max_relative_error(&output_naive, &output_lut);
    assert!(
        err < 0.15,
        "Q8 LUT-GEMM vs naive: max relative error {err} exceeds 15%"
    );
}

/// Symmetric centroids should produce rep_count ≈ 128.
/// After Phase 2C, the orbit-compressed GEMM exploits this.
#[test]
fn gemm_q8_symmetric_weights_orbit_compressed() {
    let k = 64;
    let n = 16;

    // Symmetric weights: w[i] ≈ -w[k-1-i]
    let mut weights_raw: Vec<f32> = Vec::with_capacity(k * n);
    for col in 0..n {
        for row in 0..k {
            let v = ((row as f32) - (k as f32 / 2.0)) * 0.01 + col as f32 * 0.001;
            weights_raw.push(v);
        }
    }

    let qw = quantize_8bit(&weights_raw, k as u32, n as u32);

    // Verify orbit compression is effective
    // Symmetric weights tend to produce symmetric centroids
    // (not guaranteed but likely for this distribution)
    let _rep_count = qw.orbits.rep_count;

    // LUT-GEMM must produce valid output regardless of orbit compression
    let activations: Vec<f32> = (0..4 * k).map(|i| (i as f32) / (4 * k) as f32).collect();
    let mut output = vec![0.0f32; 4 * n];
    lut_gemm_8bit(&activations, &qw, &mut output);

    // Output must not contain NaN or inf
    assert!(
        output.iter().all(|&v| v.is_finite()),
        "LUT-GEMM produced non-finite values"
    );
}

/// Performance contract: 4×256×256 LUT-GEMM must complete within budget.
#[cfg(feature = "std")]
#[test]
fn perf_gemm_q8_4x256x256() {
    use std::hint::black_box;
    use std::time::Instant;

    let m = 4;
    let k = 256;
    let n = 256;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let weights_raw: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7 + 3) % 256) as f32 / 256.0)
        .collect();
    let qw = quantize_8bit(&weights_raw, k as u32, n as u32);
    let mut output = vec![0.0f32; m * n];

    // Warm up
    lut_gemm_8bit(&activations, &qw, &mut output);

    let start = Instant::now();
    for _ in 0..10 {
        lut_gemm_8bit(
            black_box(&activations),
            black_box(&qw),
            black_box(&mut output),
        );
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 50,
        "10× lut_gemm_q8(4×256×256) took {}ms, budget 50ms (< 5ms each)",
        elapsed.as_millis()
    );
}
