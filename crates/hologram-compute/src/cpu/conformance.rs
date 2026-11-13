//! Cross-substrate kernel conformance digest.
//!
//! One dependency-free body that exercises the f32 reduction surface —
//! [`simd_f32_dot`](crate::cpu::simd::simd_f32_dot),
//! [`simd_f32_sum`](crate::cpu::simd::simd_f32_sum) and
//! [`matmul_f32_blocked`](crate::cpu::simd::matmul_f32_blocked) — over
//! deterministically generated inputs and folds every result's **bit
//! pattern** into a 64-bit FNV-1a digest.
//!
//! The native test suite and the wasm32 driver
//! (`crates/hologram-compute/conformance-wasm`) both call this one body, so
//! the comparison can never drift from the code it certifies: two substrates
//! agree on a kernel exactly when they return the same digest for it.
//!
//! The two digests are kept apart on purpose. `dot_sum_digest` covers the
//! reductions that carry one fixed association on every target and is pinned
//! to a golden value (`tests/conformance.rs::kc12_*`). `matmul_digest`
//! covers `matmul_f32_blocked`, whose per-architecture register-tiled leaves
//! (fused multiply-add on x86 and NEON, separate multiply and add on wasm
//! SIMD128) round differently; it is measured and reported, never claimed
//! identical. See `docs/numerics/invariance.md`.

use alloc::vec;
use alloc::vec::Vec;

use super::simd;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// 64-bit FNV-1a step — dependency-free and identical on every target.
#[inline]
fn fnv1a(h: u64, byte: u8) -> u64 {
    (h ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

fn fold_bits(h: &mut u64, v: f32) {
    for b in v.to_bits().to_le_bytes() {
        *h = fnv1a(*h, b);
    }
}

/// Deterministic value stream: a pure-integer LCG whose output is mapped to
/// f32 by one integer-to-float conversion and one division by a constant —
/// both correctly rounded under IEEE-754, so bit-identical on every target.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as i32) as f32 / 1.0e9
    }

    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Lengths swept by [`dot_sum_digest`]: every residue around the 4-, 8-, 16-
/// and 32-wide block boundaries the former per-ISA kernels used, plus a long
/// row.
const DOT_SUM_LENS: [usize; 22] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000,
];

/// Digest of `simd_f32_dot` and `simd_f32_sum` over `DOT_SUM_LENS`.
pub fn dot_sum_digest() -> u64 {
    let mut h = FNV_OFFSET;
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for &n in &DOT_SUM_LENS {
        let a = rng.fill(n);
        let b = rng.fill(n);
        fold_bits(&mut h, simd::simd_f32_dot(&a, &b));
        fold_bits(&mut h, simd::simd_f32_sum(&a));
    }
    h
}

/// Digest of `matmul_f32_blocked` over a small odd tile and an
/// attention-like `1×896×64` shape.
pub fn matmul_digest() -> u64 {
    let mut h = FNV_OFFSET;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for &(m, k, n) in &[(3usize, 17usize, 5usize), (1, 896, 64)] {
        let a = rng.fill(m * k);
        let b = rng.fill(k * n);
        let mut out = vec![0.0f32; m * n];
        let mut bt = Vec::new();
        simd::matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt);
        for &v in &out {
            fold_bits(&mut h, v);
        }
    }
    h
}
