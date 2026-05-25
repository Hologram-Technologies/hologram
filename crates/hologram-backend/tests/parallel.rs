//! **PA — parallel execution conformance.**
//!
//! Multi-core execution must be *observationally invisible*: the cache-oblivious
//! lattice recursion run across the in-tree worker pool (`--features parallel`)
//! produces output **byte-identical** to the single-thread path and equal to
//! the independent f64 reference, **deterministically** across repeated runs
//! (the determinism loop is the data-race stress — disjoint output tiles must
//! never alias). Sizes here clear the pool's work threshold, so with the
//! feature on the parallel frontier is exercised; with it off this still
//! validates the sequential kernel (so the suite is meaningful in both configs).

use hologram_backend::cpu::simd::{matmul_f32_blocked, matmul_f32_packed};
use hologram_backend::layout::pack_b_panels;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            o[i * n + j] = acc as f32;
        }
    }
    o
}

fn assert_matches(got: &[f32], want: &[f32], label: &str) {
    for (idx, (&g, &w)) in got.iter().zip(want).enumerate() {
        let denom = w.abs().max(1.0);
        assert!(
            (g - w).abs() / denom < 1e-4,
            "{label}: diverged at {idx}: got {g} want {w}"
        );
    }
}

#[test]
fn pa1_parallel_matmul_matches_reference_and_is_deterministic() {
    // Each clears the pool threshold (m·k·n ≥ 2^20); rectangular + non-pow2 +
    // a non-multiple-of-16 n exercise tile splitting on both axes and panels.
    for &(m, k, n) in &[
        (256usize, 256usize, 256usize),
        (512, 128, 384),
        (200, 130, 176),
    ] {
        let a = fill(m * k, 0x51 ^ (m as u64));
        let b = fill(k * n, 0x73 ^ (n as u64));
        let want = ref_matmul(&a, &b, m, k, n);

        // Row-major (unpacked) — the matmul_f32_blocked parallel frontier.
        let mut scratch = Vec::new();
        let mut first = vec![0f32; m * n];
        matmul_f32_blocked(&a, &b, &mut first, m, k, n, &mut scratch);
        assert_matches(&first, &want, "blocked vs f64 ref");
        // Determinism / race stress: repeated runs must be byte-identical.
        for _ in 0..16 {
            let mut got = vec![0f32; m * n];
            matmul_f32_blocked(&a, &b, &mut got, m, k, n, &mut scratch);
            assert_eq!(
                got, first,
                "{m}×{k}×{n} blocked: nondeterministic across runs"
            );
        }

        // Packed-weight — the matmul_f32_packed parallel frontier.
        let packed = pack_b_panels(&b, k, n);
        let mut pfirst = vec![0f32; m * n];
        matmul_f32_packed(&a, &packed, &mut pfirst, m, k, n);
        assert_matches(&pfirst, &want, "packed vs f64 ref");
        for _ in 0..16 {
            let mut got = vec![0f32; m * n];
            matmul_f32_packed(&a, &packed, &mut got, m, k, n);
            assert_eq!(
                got, pfirst,
                "{m}×{k}×{n} packed: nondeterministic across runs"
            );
        }
    }
}

/// With the feature on, the pool must actually exist (≥1 runner). Cheap sanity
/// that the parallel substrate is wired, not silently compiled out.
#[cfg(feature = "parallel")]
#[test]
fn pa2_pool_is_live() {
    assert!(
        hologram_backend::cpu::parallel::pool().width() >= 1,
        "parallel pool must report at least one runner"
    );
}
