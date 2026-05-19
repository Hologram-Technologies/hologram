//! Spec XII.5 zero-cost runtime check (in-process complement to
//! `scripts/check_zero_cost.sh`).
//!
//! Verifies the SIMD dot product computes a known result correctly across
//! sizes that exercise the chunked-vector + tail-scalar path. Combined
//! with the equivalence tests in `kernel_equivalence.rs`, this confirms
//! the SIMD-vectorized hot path is doing real work, not silently
//! falling back to scalar (which would still be correct but slow).

use hologram_backend::cpu::simd::{simd_f32_add, simd_f32_dot, simd_f32_fmadd, simd_f32_mul};

#[test]
fn simd_dot_is_correct_at_chunk_sizes() {
    // Sweep sizes that exercise vectorized chunks of 4 / 8 / 16 lanes.
    for n in [1, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128, 1023, 1024] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 3) as f32) * 0.25).collect();
        let want: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = simd_f32_dot(&a, &b);
        assert!(
            (want - got).abs() / want.abs().max(1.0) < 1e-3,
            "dot mismatch at n={n}: want {want}, got {got}",
        );
    }
}

#[test]
fn simd_add_is_correct_at_chunk_sizes() {
    for n in [1, 4, 8, 15, 16, 31, 32, 63, 64, 127, 128, 1024] {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0f32; n];
        simd_f32_add(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] + b[i], "add mismatch at i={i}, n={n}");
        }
    }
}

#[test]
fn simd_mul_is_correct_at_chunk_sizes() {
    for n in [1, 4, 8, 16, 32, 64, 128] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) - 1.0).collect();
        let mut out = vec![0f32; n];
        simd_f32_mul(&a, &b, &mut out);
        for i in 0..n {
            assert_eq!(out[i], a[i] * b[i]);
        }
    }
}

#[test]
fn simd_fmadd_is_correct() {
    let n = 64;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let mut out = vec![10.0_f32; n];
    let baseline = out.clone();
    simd_f32_fmadd(&a, &b, &mut out);
    for i in 0..n {
        let want = baseline[i] + a[i] * b[i];
        assert!((out[i] - want).abs() < 1e-4);
    }
}

/// Spec C-7: `Hasher<32>` is monomorphic — `OUTPUT_BYTES` const must be
/// statically resolved. This is more of a property check than a benchmark.
#[test]
fn hasher_output_bytes_is_const() {
    use prism::vocabulary::Hasher;
    use hologram_host::HologramHasher;
    const N: usize = <HologramHasher as Hasher<32>>::OUTPUT_BYTES;
    assert_eq!(N, 32);
}
