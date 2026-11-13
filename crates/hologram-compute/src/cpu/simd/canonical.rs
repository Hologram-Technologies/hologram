//! The canonical f32 dot and sum: **one association on every target**.
//!
//! The order, for `n` terms `t[i]` (`a[i] * b[i]` for dot, `a[i]` for sum):
//!
//! 1. **Sixteen lanes.** Lane `j` (`0 ≤ j < 16`) starts at `+0.0` and adds the
//!    terms at `i ≡ j (mod 16)` over the whole blocks of sixteen, left to
//!    right: `lane[j] = (((0 + t[j]) + t[j+16]) + t[j+32]) + …`.
//! 2. **Halving tree.** For `w = 8, 4, 2, 1`: `lane[j] = lane[j] + lane[j+w]`
//!    for every `j < w`. The result is `lane[0]`.
//! 3. **Tail.** The last `n mod 16` terms are added to that result one by one,
//!    left to right.
//!
//! Every multiply and every add rounds on its own: no fused multiply-add
//! anywhere, and Rust neither reassociates nor contracts float arithmetic. So
//! each implementation below computes the same correctly rounded operations in
//! the same order, and returns the same bits, whatever the vector width:
//! [`dot_scalar`] / [`sum_scalar`] are the definition (and the fallback on
//! targets without a vector twin); the twins hold the sixteen lanes in vector
//! registers (4 × SSE2 / NEON / SIMD128, 2 × AVX2) and walk the same tree.
//!
//! Sixteen was chosen by measurement over thirty-two: equal on native x86-64,
//! and on wasm32 SIMD128 (the deployed target) about 1.7× faster for dot at
//! n = 4096 (240 ns vs 409 ns), see `docs/numerics/invariance.md`.
//!
//! Witnesses: `tests/conformance.rs::kc12_*` (pinned golden on every CI lane;
//! the dispatched kernel equals an independent restatement of this order) and
//! `scripts/wasm-conformance.sh` (native ≡ wasm32 on every tier).

/// Accumulator lanes of the canonical association.
pub(crate) const LANES: usize = 16;

// The definition is the fallback only on targets without a vector twin; elsewhere it is the
// reference the twin tests compare against.
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(dead_code)
)]
/// The definition: canonical dot over `min(a.len(), b.len())` terms.
#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (blocks_a, tail_a) = a[..n].as_chunks::<LANES>();
    let (blocks_b, tail_b) = b[..n].as_chunks::<LANES>();
    let mut lane = [0.0f32; LANES];
    for (x, y) in blocks_a.iter().zip(blocks_b) {
        for ((l, &xa), &yb) in lane.iter_mut().zip(x).zip(y) {
            *l += xa * yb;
        }
    }
    let mut s = tree(lane);
    for (x, y) in tail_a.iter().zip(tail_b) {
        s += x * y;
    }
    s
}

// The definition is the fallback only on targets without a vector twin; elsewhere it is the
// reference the twin tests compare against.
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(dead_code)
)]
/// The definition: canonical sum.
#[inline]
pub(crate) fn sum_scalar(a: &[f32]) -> f32 {
    let (blocks, tail) = a.as_chunks::<LANES>();
    let mut lane = [0.0f32; LANES];
    for x in blocks {
        for (l, &xa) in lane.iter_mut().zip(x) {
            *l += xa;
        }
    }
    let mut s = tree(lane);
    for &v in tail {
        s += v;
    }
    s
}

// The definition is the fallback only on targets without a vector twin; elsewhere it is the
// reference the twin tests compare against.
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(dead_code)
)]
/// Step 2: `lane[j] += lane[j + w]` for `w = 8, 4, 2, 1`.
#[inline]
fn tree(mut lane: [f32; LANES]) -> f32 {
    let mut w = LANES;
    while w > 1 {
        w /= 2;
        let (lo, hi) = lane.split_at_mut(w);
        for (l, &h) in lo.iter_mut().zip(&hi[..w]) {
            *l += h;
        }
    }
    lane[0]
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    //! SSE2 (x86-64 baseline: lanes `4v..4v+4` in register `v`) and AVX2
    //! (lanes `8v..8v+8` in register `v`) twins.
    use core::arch::x86_64::*;

    /// Tree steps `w = 2, 1` inside one 4-lane register.
    #[inline(always)]
    unsafe fn finish4(v: __m128) -> f32 {
        let h = _mm_add_ps(v, _mm_movehl_ps(v, v)); // l0 + l2, l1 + l3
        _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps::<0b01>(h, h))) // + (l1 + l3)
    }

    /// Tree steps `w = 8, 4, 2, 1` over four 4-lane registers.
    #[inline(always)]
    unsafe fn tree_sse(r: [__m128; 4]) -> f32 {
        let lo = _mm_add_ps(r[0], r[2]); // w = 8
        let hi = _mm_add_ps(r[1], r[3]);
        finish4(_mm_add_ps(lo, hi)) // w = 4, then 2, 1
    }

    /// Tree steps `w = 8, 4, 2, 1` over two 8-lane registers.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn tree_avx(r: [__m256; 2]) -> f32 {
        let v = _mm256_add_ps(r[0], r[1]); // w = 8
        finish4(_mm_add_ps(
            _mm256_castps256_ps128(v),
            _mm256_extractf128_ps::<1>(v),
        )) // w = 4, then 2, 1
    }

    /// # Safety
    /// None beyond x86-64 (SSE2 is baseline); slice bounds are respected.
    pub(crate) unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let whole = n - n % super::LANES;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut r = [_mm_setzero_ps(); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                let p = _mm_mul_ps(
                    _mm_loadu_ps(pa.add(o + 4 * v)),
                    _mm_loadu_ps(pb.add(o + 4 * v)),
                );
                *acc = _mm_add_ps(*acc, p);
            }
            o += super::LANES;
        }
        let mut s = tree_sse(r);
        for (x, y) in a[whole..n].iter().zip(&b[whole..n]) {
            s += x * y;
        }
        s
    }

    /// # Safety
    /// None beyond x86-64 (SSE2 is baseline); slice bounds are respected.
    pub(crate) unsafe fn sum_sse2(a: &[f32]) -> f32 {
        let n = a.len();
        let whole = n - n % super::LANES;
        let pa = a.as_ptr();
        let mut r = [_mm_setzero_ps(); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                *acc = _mm_add_ps(*acc, _mm_loadu_ps(pa.add(o + 4 * v)));
            }
            o += super::LANES;
        }
        let mut s = tree_sse(r);
        for &x in &a[whole..] {
            s += x;
        }
        s
    }

    /// # Safety
    /// AVX2 must be available; slice bounds are respected.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let whole = n - n % super::LANES;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut r = [_mm256_setzero_ps(); 2];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                let p = _mm256_mul_ps(
                    _mm256_loadu_ps(pa.add(o + 8 * v)),
                    _mm256_loadu_ps(pb.add(o + 8 * v)),
                );
                *acc = _mm256_add_ps(*acc, p);
            }
            o += super::LANES;
        }
        let mut s = tree_avx(r);
        for (x, y) in a[whole..n].iter().zip(&b[whole..n]) {
            s += x * y;
        }
        s
    }

    /// # Safety
    /// AVX2 must be available; slice bounds are respected.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sum_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        let whole = n - n % super::LANES;
        let pa = a.as_ptr();
        let mut r = [_mm256_setzero_ps(); 2];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                *acc = _mm256_add_ps(*acc, _mm256_loadu_ps(pa.add(o + 8 * v)));
            }
            o += super::LANES;
        }
        let mut s = tree_avx(r);
        for &x in &a[whole..] {
            s += x;
        }
        s
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    //! NEON twin (baseline on aarch64): lanes `4v..4v+4` in register `v`.
    use core::arch::aarch64::*;

    /// Tree steps `w = 8, 4, 2, 1` over four 4-lane registers.
    #[inline(always)]
    unsafe fn tree(r: [float32x4_t; 4]) -> f32 {
        let v = vaddq_f32(vaddq_f32(r[0], r[2]), vaddq_f32(r[1], r[3])); // w = 8, 4
        let h = vadd_f32(vget_low_f32(v), vget_high_f32(v)); // w = 2
        vget_lane_f32::<0>(h) + vget_lane_f32::<1>(h) // w = 1
    }

    /// # Safety
    /// NEON (baseline on aarch64); slice bounds are respected.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let whole = n - n % super::LANES;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut r = [vdupq_n_f32(0.0); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                // vmulq then vaddq: two roundings, never vfmaq.
                let p = vmulq_f32(vld1q_f32(pa.add(o + 4 * v)), vld1q_f32(pb.add(o + 4 * v)));
                *acc = vaddq_f32(*acc, p);
            }
            o += super::LANES;
        }
        let mut s = tree(r);
        for (x, y) in a[whole..n].iter().zip(&b[whole..n]) {
            s += x * y;
        }
        s
    }

    /// # Safety
    /// NEON (baseline on aarch64); slice bounds are respected.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn sum(a: &[f32]) -> f32 {
        let n = a.len();
        let whole = n - n % super::LANES;
        let pa = a.as_ptr();
        let mut r = [vdupq_n_f32(0.0); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                *acc = vaddq_f32(*acc, vld1q_f32(pa.add(o + 4 * v)));
            }
            o += super::LANES;
        }
        let mut s = tree(r);
        for &x in &a[whole..] {
            s += x;
        }
        s
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm {
    //! wasm SIMD128 twin: lanes `4v..4v+4` in register `v`.
    use core::arch::wasm32::*;

    /// Tree steps `w = 8, 4, 2, 1` over four 4-lane registers.
    #[inline(always)]
    fn tree(r: [v128; 4]) -> f32 {
        let v = f32x4_add(f32x4_add(r[0], r[2]), f32x4_add(r[1], r[3])); // w = 8, 4
        let h = f32x4_add(v, i32x4_shuffle::<2, 3, 2, 3>(v, v)); // w = 2
        f32x4_extract_lane::<0>(h) + f32x4_extract_lane::<1>(h) // w = 1
    }

    /// # Safety
    /// SIMD128 enabled (cfg-gated); `v128_load` is unaligned; bounds respected.
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let whole = n - n % super::LANES;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut r = [f32x4_splat(0.0); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                let p = f32x4_mul(
                    v128_load(pa.add(o + 4 * v) as *const v128),
                    v128_load(pb.add(o + 4 * v) as *const v128),
                );
                *acc = f32x4_add(*acc, p);
            }
            o += super::LANES;
        }
        let mut s = tree(r);
        for (x, y) in a[whole..n].iter().zip(&b[whole..n]) {
            s += x * y;
        }
        s
    }

    /// # Safety
    /// SIMD128 enabled (cfg-gated); `v128_load` is unaligned; bounds respected.
    pub(crate) unsafe fn sum(a: &[f32]) -> f32 {
        let n = a.len();
        let whole = n - n % super::LANES;
        let pa = a.as_ptr();
        let mut r = [f32x4_splat(0.0); 4];
        let mut o = 0;
        while o < whole {
            for (v, acc) in r.iter_mut().enumerate() {
                *acc = f32x4_add(*acc, v128_load(pa.add(o + 4 * v) as *const v128));
            }
            o += super::LANES;
        }
        let mut s = tree(r);
        for &x in &a[whole..] {
            s += x;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn mixed(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mant = ((s >> 40) as u32 & 0xFFFF) as f32 / 65536.0 + 0.5;
                let e = ((s >> 33) & 0xF) as i32 - 8;
                let sign = if (s >> 32) & 1 == 0 { 1.0 } else { -1.0 };
                sign * mant * libm::powf(2.0, e as f32)
            })
            .collect()
    }

    /// Every vector twin compiled for this target equals the scalar
    /// definition bit for bit — including twins the dispatcher would not pick
    /// on this host (SSE2 on an AVX2 machine).
    #[test]
    fn every_twin_equals_the_scalar_definition() {
        for n in (0usize..=300).chain([1000, 4099]) {
            let a = mixed(0xA11CE ^ n as u64, n);
            let b = mixed(0xB0B ^ n as u64, n);
            let (d, s) = (dot_scalar(&a, &b).to_bits(), sum_scalar(&a).to_bits());
            #[cfg(target_arch = "x86_64")]
            unsafe {
                assert_eq!(x86::dot_sse2(&a, &b).to_bits(), d, "sse2 dot n={n}");
                assert_eq!(x86::sum_sse2(&a).to_bits(), s, "sse2 sum n={n}");
                #[cfg(feature = "std")]
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(x86::dot_avx2(&a, &b).to_bits(), d, "avx2 dot n={n}");
                    assert_eq!(x86::sum_avx2(&a).to_bits(), s, "avx2 sum n={n}");
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                assert_eq!(neon::dot(&a, &b).to_bits(), d, "neon dot n={n}");
                assert_eq!(neon::sum(&a).to_bits(), s, "neon sum n={n}");
            }
            let _ = (d, s);
        }
    }
}
