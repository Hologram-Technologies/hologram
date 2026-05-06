//! SIMD vectorized paths for hot-loop float kernels.
//!
//! Each `simd_*` function picks the widest SIMD path the build target enables
//! (AVX-512 ▸ AVX2 ▸ NEON ▸ scalar). Per spec III.2 / I-6, the active CPU
//! backend pins `WITT_LEVEL_MAX_BITS` to its register width — the SIMD
//! kernels lay out work to match that bit width.

#![allow(unsafe_op_in_unsafe_fn)]

/// SIMD-vectorized f32 add: `out[i] = a[i] + b[i]` for `i in 0..n`.
#[inline]
pub fn simd_f32_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe { return avx512::add_f32(a, b, out); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return avx2::add_f32(a, b, out); }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe { return neon::add_f32(a, b, out); }
    scalar::add_f32(a, b, out)
}

/// SIMD-vectorized f32 multiply: `out[i] = a[i] * b[i]`.
#[inline]
pub fn simd_f32_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe { return avx512::mul_f32(a, b, out); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return avx2::mul_f32(a, b, out); }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe { return neon::mul_f32(a, b, out); }
    scalar::mul_f32(a, b, out)
}

/// SIMD-vectorized f32 fused multiply-add: `out[i] += a[i] * b[i]`.
#[inline]
pub fn simd_f32_fmadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe { return avx512::fmadd_f32(a, b, out); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return avx2::fmadd_f32(a, b, out); }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe { return neon::fmadd_f32(a, b, out); }
    scalar::fmadd_f32(a, b, out)
}

/// SIMD-vectorized f32 dot product.
#[inline]
pub fn simd_f32_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe { return avx512::dot_f32(a, b); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return avx2::dot_f32(a, b); }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe { return neon::dot_f32(a, b); }
    scalar::dot_f32(a, b)
}

/// Scalar reference implementations (always available).
mod scalar {
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n { out[i] = a[i] + b[i]; }
    }
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n { out[i] = a[i] * b[i]; }
    }
    pub fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n { out[i] = a[i].mul_add(b[i], out[i]); }
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = 0f32;
        for i in 0..n { acc += a[i] * b[i]; }
        acc
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use core::arch::x86_64::*;
    pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            let vo = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), vo);
        }
        for i in chunks * 8..n { out[i] = a[i] + b[i]; }
    }
    pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            let vo = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), vo);
        }
        for i in chunks * 8..n { out[i] = a[i] * b[i]; }
    }
    pub unsafe fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            let vc = _mm256_loadu_ps(out.as_ptr().add(k * 8));
            #[cfg(target_feature = "fma")]
            let vo = _mm256_fmadd_ps(va, vb, vc);
            #[cfg(not(target_feature = "fma"))]
            let vo = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), vo);
        }
        for i in chunks * 8..n { out[i] = a[i].mul_add(b[i], out[i]); }
    }
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            #[cfg(target_feature = "fma")]
            { acc = _mm256_fmadd_ps(va, vb, acc); }
            #[cfg(not(target_feature = "fma"))]
            { acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb)); }
        }
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let mut total: f32 = buf.iter().sum();
        for i in chunks * 8..n { total += a[i] * b[i]; }
        total
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[allow(dead_code)]
mod avx2 {
    pub unsafe fn add_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn mul_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn fmadd_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn dot_f32(_a: &[f32], _b: &[f32]) -> f32 { unreachable!() }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512 {
    use core::arch::x86_64::*;
    pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            let vo = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), vo);
        }
        for i in chunks * 16..n { out[i] = a[i] + b[i]; }
    }
    pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            let vo = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), vo);
        }
        for i in chunks * 16..n { out[i] = a[i] * b[i]; }
    }
    pub unsafe fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            let vc = _mm512_loadu_ps(out.as_ptr().add(k * 16));
            let vo = _mm512_fmadd_ps(va, vb, vc);
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), vo);
        }
        for i in chunks * 16..n { out[i] = a[i].mul_add(b[i], out[i]); }
    }
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 16;
        let mut acc = _mm512_setzero_ps();
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        let total = _mm512_reduce_add_ps(acc);
        let mut tail = 0f32;
        for i in chunks * 16..n { tail += a[i] * b[i]; }
        total + tail
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[allow(dead_code)]
mod avx512 {
    pub unsafe fn add_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn mul_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn fmadd_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn dot_f32(_a: &[f32], _b: &[f32]) -> f32 { unreachable!() }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use core::arch::aarch64::*;
    pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            let vo = vaddq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(k * 4), vo);
        }
        for i in chunks * 4..n { out[i] = a[i] + b[i]; }
    }
    pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            let vo = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(k * 4), vo);
        }
        for i in chunks * 4..n { out[i] = a[i] * b[i]; }
    }
    pub unsafe fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            let vc = vld1q_f32(out.as_ptr().add(k * 4));
            let vo = vfmaq_f32(vc, va, vb);
            vst1q_f32(out.as_mut_ptr().add(k * 4), vo);
        }
        for i in chunks * 4..n { out[i] = a[i].mul_add(b[i], out[i]); }
    }
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let lanes = [vgetq_lane_f32(acc, 0), vgetq_lane_f32(acc, 1),
                     vgetq_lane_f32(acc, 2), vgetq_lane_f32(acc, 3)];
        let mut total: f32 = lanes.iter().sum();
        for i in chunks * 4..n { total += a[i] * b[i]; }
        total
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
#[allow(dead_code)]
mod neon {
    pub unsafe fn add_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn mul_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn fmadd_f32(_a: &[f32], _b: &[f32], _o: &mut [f32]) { unreachable!() }
    pub unsafe fn dot_f32(_a: &[f32], _b: &[f32]) -> f32 { unreachable!() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_matches_scalar() {
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0f32; 32];
        simd_f32_add(&a, &b, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], a[i] + b[i]);
        }
    }

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=20).map(|i| (i * 3) as f32).collect();
        let want: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let got = simd_f32_dot(&a, &b);
        assert!((want - got).abs() < 1e-3, "want {want}, got {got}");
    }

    #[test]
    fn fmadd_matches_scalar() {
        let a: Vec<f32> = vec![1.0; 16];
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out: Vec<f32> = vec![10.0; 16];
        simd_f32_fmadd(&a, &b, &mut out);
        for i in 0..16 {
            let want = 10.0 + a[i] * b[i];
            assert!((out[i] - want).abs() < 1e-5);
        }
    }
}
