//! Walsh-Hadamard Transform (WHT) for KV cache value rotation.
//!
//! Applying WHT rotation before quantizing V tensors Gaussianizes their
//! distribution, reducing quantization error at q4/q8 precision. The
//! transform is its own inverse (up to scaling), so the same butterfly
//! is used for both rotation and unrotation.
//!
//! On aarch64, NEON float32x4 intrinsics accelerate the butterfly stages.

// ── Walsh-Hadamard Transform ─────────────────────────────────────────

/// NEON-accelerated FWHT butterfly for stages where half >= 4.
/// Processes 4 butterfly pairs per iteration using float32x4 SIMD.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fwht_butterfly_neon(data: &mut [f32], half: usize) {
    use core::arch::aarch64::*;
    let n = data.len();
    let step = half * 2;
    let ptr = data.as_mut_ptr();
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Process 4 elements at a time with NEON.
        let end4 = i + (half & !3);
        while j < end4 {
            let a = vld1q_f32(ptr.add(j));
            let b = vld1q_f32(ptr.add(j + half));
            vst1q_f32(ptr.add(j), vaddq_f32(a, b));
            vst1q_f32(ptr.add(j + half), vsubq_f32(a, b));
            j += 4;
        }
        // Scalar remainder (0-3 elements).
        while j < i + half {
            let a = *ptr.add(j);
            let b = *ptr.add(j + half);
            *ptr.add(j) = a + b;
            *ptr.add(j + half) = a - b;
            j += 1;
        }
        i += step;
    }
}

/// In-place Fast Walsh-Hadamard Transform (FWHT) on a slice of length `n`.
///
/// `n` must be a power of 2 (typically `head_dim`). Uses the iterative
/// butterfly algorithm: O(n log n) with O(1) extra memory.
/// The transform is self-inverse up to a factor of `n`: FWHT(FWHT(x)) = n * x.
///
/// On aarch64, stages with half >= 4 use NEON float32x4 intrinsics.
#[inline]
pub(crate) fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");

    // Small stages (half < 4): scalar butterfly.
    let mut half = 1;
    while half < 4.min(n) {
        let step = half * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + half {
                let a = data[j];
                let b = data[j + half];
                data[j] = a + b;
                data[j + half] = a - b;
            }
            i += step;
        }
        half = step;
    }

    // Large stages (half >= 4): SIMD butterfly.
    while half < n {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            fwht_butterfly_neon(data, half);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let step = half * 2;
            let mut i = 0;
            while i < n {
                for j in i..i + half {
                    let a = data[j];
                    let b = data[j + half];
                    data[j] = a + b;
                    data[j + half] = a - b;
                }
                i += step;
            }
        }
        half *= 2;
    }
}

/// Deterministic sign vector for Walsh-Hadamard rotation.
/// Uses a simple PRNG seeded on `dim` for reproducibility across sessions.
pub(crate) fn wht_signs(dim: usize) -> Vec<f32> {
    let mut signs = Vec::with_capacity(dim);
    // Simple LCG seeded on dim for deterministic, architecture-independent signs.
    let mut state: u64 = dim as u64 ^ 0x517c_c1b7_2722_0a95;
    for _ in 0..dim {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        signs.push(if (state >> 63) == 0 { 1.0 } else { -1.0 });
    }
    signs
}

/// Element-wise multiply: data[i] *= factors[i]. NEON-accelerated on aarch64.
#[inline]
pub(crate) fn vec_mul_inplace(data: &mut [f32], factors: &[f32]) {
    let n = data.len();
    #[cfg(target_arch = "aarch64")]
    {
        let chunks = n / 4;
        let dp = data.as_mut_ptr();
        let fp = factors.as_ptr();
        for c in 0..chunks {
            let off = c * 4;
            unsafe {
                use core::arch::aarch64::*;
                let a = vld1q_f32(dp.add(off));
                let b = vld1q_f32(fp.add(off));
                vst1q_f32(dp.add(off), vmulq_f32(a, b));
            }
        }
        for i in chunks * 4..n {
            data[i] *= factors[i];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            data[i] *= factors[i];
        }
    }
}

/// Fused Walsh-Hadamard rotation: signs ⊙ FWHT(signs ⊙ x) / √dim.
///
/// `signs_norm` = `signs[i] / sqrt(dim)`, precomputed at cache construction.
/// Eliminates runtime `1/sqrt(dim)` computation and uses precomputed table
/// for the final sign-flip + normalize pass.
///
/// Write path: first sign-flip → FWHT → multiply by signs_norm.
#[inline]
pub(crate) fn wht_rotate_fused(data: &mut [f32], signs: &[f32], signs_norm: &[f32]) {
    // Pass 1: first sign flip.
    vec_mul_inplace(data, signs);
    // Passes 2..log2(n)+1: FWHT butterfly stages.
    fwht_inplace(data);
    // Final pass: second sign-flip + normalize via precomputed signs_norm.
    vec_mul_inplace(data, signs_norm);
}

/// Fused inverse WHT for read path: data already has first sign-flip applied
/// (fused into dequant), so skip it entirely. Just FWHT + apply signs_norm.
///
/// Saves one full pass over the data vs `wht_rotate_fused`.
/// Caller must ensure `data[i]` was pre-multiplied by `signs[i]` during dequant.
#[inline]
pub(crate) fn wht_unrotate_presigned(data: &mut [f32], signs_norm: &[f32]) {
    // Data already has signs applied — skip first sign-flip pass.
    fwht_inplace(data);
    // Second sign-flip + normalize.
    vec_mul_inplace(data, signs_norm);
}

/// Standard WHT rotation (backward compatibility for tests).
#[cfg(test)]
#[inline]
pub(crate) fn wht_rotate(data: &mut [f32], signs: &[f32]) {
    let dim = data.len();
    let norm = 1.0 / (dim as f32).sqrt();
    let signs_norm: Vec<f32> = signs.iter().map(|&s| s * norm).collect();
    wht_rotate_fused(data, signs, &signs_norm);
}

/// Standard inverse WHT (for tests).
#[cfg(test)]
#[inline]
pub(crate) fn wht_unrotate(data: &mut [f32], signs: &[f32]) {
    wht_rotate(data, signs);
}
