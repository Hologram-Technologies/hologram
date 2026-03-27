//! Quantization conformance tests.
//!
//! Encodes the contract for uniform Q8 quantization and orbit compression.
//! Tests will FAIL until quantize_8bit_uniform is implemented (Phase 2B).

use hologram_exec::lut_gemm::orbit::build_orbit_map_q8;
use hologram_exec::lut_gemm::quantize::quantize_8bit;

// ── Orbit Compression Invariants (existing infrastructure) ──────────────────

#[test]
fn quantize_orbit_symmetry() {
    // Fully symmetric centroids: c[i] = -c[256-i]
    let mut centroids = [0.0f32; 256];
    for i in 1..128usize {
        centroids[i] = i as f32;
        centroids[256 - i] = -(i as f32);
    }
    let orbit = build_orbit_map_q8(&centroids);
    // 2 self-inverse (0, 128) + 127 pairs = 129 representatives
    assert!(
        orbit.rep_count <= 129,
        "symmetric centroids should have ≤129 reps, got {}",
        orbit.rep_count
    );
}

#[test]
fn quantize_orbit_identity_asymmetric() {
    // Random asymmetric centroids → no compression
    let mut centroids = [0.0f32; 256];
    for i in 0..256usize {
        centroids[i] = (i as f32 * 1.6180339) % 7.0 + 1.0;
    }
    let orbit = build_orbit_map_q8(&centroids);
    assert_eq!(
        orbit.rep_count, 256,
        "asymmetric centroids should have 256 reps"
    );
}

// ── Uniform Quantization (Phase 2B) ────────────────────────────────────────

#[test]
fn quantize_uniform_floor_division() {
    use hologram_exec::lut_gemm::quantize::quantize_8bit_uniform;
    let weights: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
    let qw = quantize_8bit_uniform(&weights, 16, 16);
    // Uniform weights [0..1] with 256 uniform centroids → index should equal input index
    for (i, &idx) in qw.indices.iter().enumerate() {
        assert_eq!(idx, i as u8, "uniform assignment mismatch at {i}");
    }
}

#[test]
fn quantize_uniform_error_bounded() {
    use hologram_exec::lut_gemm::quantize::quantize_8bit_uniform;
    let weights: Vec<f32> = (0..4096).map(|i| (i as f32) / 4096.0).collect();
    let qw = quantize_8bit_uniform(&weights, 64, 64);
    // Reconstruct and measure error
    let mut sq_err = 0.0f32;
    let mut sq_orig = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        let recon = qw.centroids[qw.indices[i] as usize];
        sq_err += (w - recon) * (w - recon);
        sq_orig += w * w;
    }
    let rmse = if sq_orig > 0.0 {
        (sq_err / sq_orig).sqrt()
    } else {
        0.0
    };
    assert!(rmse < 0.05, "uniform Q8 RMSE too high: {rmse}");
}

#[cfg(feature = "std")]
#[test]
fn perf_quantize_q8_uniform() {
    use hologram_exec::lut_gemm::quantize::quantize_8bit_uniform;
    use std::hint::black_box;
    use std::time::Instant;
    let weights: Vec<f32> = (0..4096).map(|i| (i as f32) / 4096.0).collect();
    let start = Instant::now();
    for _ in 0..10 {
        black_box(quantize_8bit_uniform(black_box(&weights), 64, 64));
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() < 50,
        "10× quantize_8bit_uniform(64×64) took {}ms, budget 50ms",
        elapsed.as_millis()
    );
}

// ── Existing Q8 k-means Baseline ────────────────────────────────────────────

#[test]
fn quantize_8bit_kmeans_produces_valid_output() {
    let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let qw = quantize_8bit(&weights, 8, 8);
    assert_eq!(qw.rows, 8);
    assert_eq!(qw.cols, 8);
    assert_eq!(qw.indices.len(), 64);
    // Every index must be < 256
    assert!(qw.indices.iter().all(|&i| (i as usize) < 256));
}
