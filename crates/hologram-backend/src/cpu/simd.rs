//! Runtime-dispatched SIMD for hot-loop f32 kernels.
//!
//! The previous design was strictly cfg-gated — `target_feature =
//! "avx512f"` had to be active at build time for the AVX-512 path to
//! be reachable. That made a stock `cargo build --release` binary use
//! the scalar fallback on machines that fully support AVX-512.
//!
//! This module compiles **all** SIMD paths unconditionally on x86-64
//! (each one tagged with `#[target_feature(enable = "...")]`) and
//! picks the widest available at first call via `std::is_x86_feature_detected!`,
//! caching the choice in a `OnceLock`-resolved function pointer. The
//! first call costs one CPUID; subsequent calls dispatch through a
//! single indirect jump — about one cycle of overhead on modern x86.
//!
//! Per spec III.2 / I-6 each backend's `HostBounds::WITT_LEVEL_MAX_BITS`
//! still names its natural register width — the runtime path just
//! also handles the case where the **build target's** floor is below
//! the **host machine's** ceiling.

#![allow(unsafe_op_in_unsafe_fn)]

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, Ordering};

/// SIMD path the runtime dispatcher selected. Cached after first
/// detection. Values: 0 = unresolved, 1 = scalar, 2 = AVX2+FMA,
/// 3 = AVX-512F, 4 = NEON.
static SIMD_PATH: AtomicU8 = AtomicU8::new(0);

#[inline]
fn resolve_path() -> u8 {
    let cached = SIMD_PATH.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let chosen = detect_path();
    SIMD_PATH.store(chosen, Ordering::Relaxed);
    chosen
}

fn detect_path() -> u8 {
    // Runtime CPU-feature detection requires `std` (`is_x86_feature_detected!`
    // reads CPUID via the OS). On `no_std` targets (wasm / embedded) there is
    // no runtime probe, so we fall back to the build target's compile-time
    // feature floor (`cfg!(target_feature = …)`); a `RUSTFLAGS=-C
    // target-feature=+avx2` build then still reaches the vector path.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return 3;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return 2;
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx512f"
    ))]
    {
        return 3;
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(feature = "std"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        return 2;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return 4;
    }
    1
}

/// SIMD-vectorized f32 add: `out[i] = a[i] + b[i]`.
#[inline]
pub fn simd_f32_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::add_f32_avx512(a, b, out) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::add_f32_avx2(a, b, out) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::add_f32_neon(a, b, out) },
        _ => scalar::add_f32(a, b, out),
    }
}

/// SIMD-vectorized f32 multiply.
#[inline]
pub fn simd_f32_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::mul_f32_avx512(a, b, out) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::mul_f32_avx2(a, b, out) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::mul_f32_neon(a, b, out) },
        _ => scalar::mul_f32(a, b, out),
    }
}

/// SIMD-vectorized f32 fused multiply-add: `out[i] += a[i] * b[i]`.
#[inline]
pub fn simd_f32_fmadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::fmadd_f32_avx512(a, b, out) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::fmadd_f32_avx2(a, b, out) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::fmadd_f32_neon(a, b, out) },
        _ => scalar::fmadd_f32(a, b, out),
    }
}

/// SIMD-vectorized f32 dot product.
#[inline]
pub fn simd_f32_dot(a: &[f32], b: &[f32]) -> f32 {
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::dot_f32_avx512(a, b) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::dot_f32_avx2(a, b) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::dot_f32_neon(a, b) },
        _ => scalar::dot_f32(a, b),
    }
}

mod scalar {
    #[cfg(not(feature = "std"))]
    use crate::cpu::mathf::FloatExt;

    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n {
            out[i] = a[i] + b[i];
        }
    }
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n {
            out[i] = a[i] * b[i];
        }
    }
    pub fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n {
            out[i] = a[i].mul_add(b[i], out[i]);
        }
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = 0f32;
        for i in 0..n {
            acc += a[i] * b[i];
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    #[cfg(not(feature = "std"))]
    use crate::cpu::mathf::FloatExt;
    use core::arch::x86_64::*;

    // ─── AVX2 + FMA path ──────────────────────────────────────────

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), _mm256_add_ps(va, vb));
        }
        for i in chunks * 8..n {
            out[i] = a[i] + b[i];
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn mul_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), _mm256_mul_ps(va, vb));
        }
        for i in chunks * 8..n {
            out[i] = a[i] * b[i];
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn fmadd_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 8;
        for k in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(k * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(k * 8));
            let vc = _mm256_loadu_ps(out.as_ptr().add(k * 8));
            _mm256_storeu_ps(out.as_mut_ptr().add(k * 8), _mm256_fmadd_ps(va, vb, vc));
        }
        for i in chunks * 8..n {
            out[i] = a[i].mul_add(b[i], out[i]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        // Unroll across four independent accumulators so the OOO core can
        // keep the FMA units saturated (latency 4-5 cycles, throughput 1
        // per cycle on Zen 4 / Sapphire Rapids).
        let chunks = n / 32;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for k in 0..chunks {
            let off = k * 32;
            let a0 = _mm256_loadu_ps(a.as_ptr().add(off));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(off + 8));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(off + 16));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(off + 24));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(off));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(off + 8));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(off + 16));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(off + 24));
            acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            acc2 = _mm256_fmadd_ps(a2, b2, acc2);
            acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let mut total: f32 = buf.iter().sum();
        // Tail: 8-wide chunks.
        let tail_chunks = (n - chunks * 32) / 8;
        for k in 0..tail_chunks {
            let off = chunks * 32 + k * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(off));
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            let v = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(buf.as_mut_ptr(), v);
            total += buf.iter().sum::<f32>();
        }
        // Final scalar tail.
        let done = chunks * 32 + tail_chunks * 8;
        for i in done..n {
            total += a[i] * b[i];
        }
        total
    }

    // ─── AVX-512 path ─────────────────────────────────────────────

    #[target_feature(enable = "avx512f")]
    pub unsafe fn add_f32_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), _mm512_add_ps(va, vb));
        }
        for i in chunks * 16..n {
            out[i] = a[i] + b[i];
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul_f32_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), _mm512_mul_ps(va, vb));
        }
        for i in chunks * 16..n {
            out[i] = a[i] * b[i];
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn fmadd_f32_avx512(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 16;
        for k in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(k * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(k * 16));
            let vc = _mm512_loadu_ps(out.as_ptr().add(k * 16));
            _mm512_storeu_ps(out.as_mut_ptr().add(k * 16), _mm512_fmadd_ps(va, vb, vc));
        }
        for i in chunks * 16..n {
            out[i] = a[i].mul_add(b[i], out[i]);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 64;
        // Four 512-bit accumulators (1024 bits / 64 lanes per iter).
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        for k in 0..chunks {
            let off = k * 64;
            let a0 = _mm512_loadu_ps(a.as_ptr().add(off));
            let a1 = _mm512_loadu_ps(a.as_ptr().add(off + 16));
            let a2 = _mm512_loadu_ps(a.as_ptr().add(off + 32));
            let a3 = _mm512_loadu_ps(a.as_ptr().add(off + 48));
            let b0 = _mm512_loadu_ps(b.as_ptr().add(off));
            let b1 = _mm512_loadu_ps(b.as_ptr().add(off + 16));
            let b2 = _mm512_loadu_ps(b.as_ptr().add(off + 32));
            let b3 = _mm512_loadu_ps(b.as_ptr().add(off + 48));
            acc0 = _mm512_fmadd_ps(a0, b0, acc0);
            acc1 = _mm512_fmadd_ps(a1, b1, acc1);
            acc2 = _mm512_fmadd_ps(a2, b2, acc2);
            acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        }
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let mut total = _mm512_reduce_add_ps(acc);
        // 16-wide tail.
        let tail_chunks = (n - chunks * 64) / 16;
        for k in 0..tail_chunks {
            let off = chunks * 64 + k * 16;
            let va = _mm512_loadu_ps(a.as_ptr().add(off));
            let vb = _mm512_loadu_ps(b.as_ptr().add(off));
            total += _mm512_reduce_add_ps(_mm512_mul_ps(va, vb));
        }
        let done = chunks * 64 + tail_chunks * 16;
        for i in done..n {
            total += a[i] * b[i];
        }
        total
    }

    /// Register-blocked f32 GEMM micro-kernel: `out = A·B`, row-major, with
    /// a 4×16 output tile held in eight YMM accumulators across the `k`
    /// loop. Each `k` step loads two contiguous 8-wide B vectors and issues
    /// `MR` broadcast-FMAs per vector — peak FMA throughput with one stream
    /// of B and no horizontal reductions. Row/column remainders fall to a
    /// scalar tail.
    ///
    /// # Safety
    /// Requires AVX2 + FMA (the caller gates on `resolve_path`). The slices
    /// must satisfy `a.len() >= m*k`, `b.len() >= k*n`, `out.len() >= m*n`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matmul_f32_fma(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        const MR: usize = 4;
        const NR: usize = 16;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr();

        let mut i = 0;
        while i + MR <= m {
            let mut j = 0;
            while j + NR <= n {
                let mut c = [[_mm256_setzero_ps(); 2]; MR];
                for kk in 0..k {
                    let brow = bp.add(kk * n + j);
                    let b0 = _mm256_loadu_ps(brow);
                    let b1 = _mm256_loadu_ps(brow.add(8));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = _mm256_set1_ps(*ap.add((i + r) * k + kk));
                        cr[0] = _mm256_fmadd_ps(av, b0, cr[0]);
                        cr[1] = _mm256_fmadd_ps(av, b1, cr[1]);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    let orow = op.add((i + r) * n + j);
                    _mm256_storeu_ps(orow, cr[0]);
                    _mm256_storeu_ps(orow.add(8), cr[1]);
                }
                j += NR;
            }
            // Column remainder for this row block.
            while j < n {
                for r in 0..MR {
                    let mut s = 0.0f32;
                    for kk in 0..k {
                        s += *ap.add((i + r) * k + kk) * *bp.add(kk * n + j);
                    }
                    *op.add((i + r) * n + j) = s;
                }
                j += 1;
            }
            i += MR;
        }
        // Row remainder.
        while i < m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += *ap.add(i * k + kk) * *bp.add(kk * n + j);
                }
                *op.add(i * n + j) = s;
            }
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch {
    #[cfg(not(feature = "std"))]
    use crate::cpu::mathf::FloatExt;
    use core::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub unsafe fn add_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            vst1q_f32(out.as_mut_ptr().add(k * 4), vaddq_f32(va, vb));
        }
        for i in chunks * 4..n {
            out[i] = a[i] + b[i];
        }
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn mul_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            vst1q_f32(out.as_mut_ptr().add(k * 4), vmulq_f32(va, vb));
        }
        for i in chunks * 4..n {
            out[i] = a[i] * b[i];
        }
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn fmadd_f32_neon(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(k * 4));
            let vb = vld1q_f32(b.as_ptr().add(k * 4));
            let vc = vld1q_f32(out.as_ptr().add(k * 4));
            vst1q_f32(out.as_mut_ptr().add(k * 4), vfmaq_f32(vc, va, vb));
        }
        for i in chunks * 4..n {
            out[i] = a[i].mul_add(b[i], out[i]);
        }
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 16;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for k in 0..chunks {
            let off = k * 16;
            let a0 = vld1q_f32(a.as_ptr().add(off));
            let a1 = vld1q_f32(a.as_ptr().add(off + 4));
            let a2 = vld1q_f32(a.as_ptr().add(off + 8));
            let a3 = vld1q_f32(a.as_ptr().add(off + 12));
            let b0 = vld1q_f32(b.as_ptr().add(off));
            let b1 = vld1q_f32(b.as_ptr().add(off + 4));
            let b2 = vld1q_f32(b.as_ptr().add(off + 8));
            let b3 = vld1q_f32(b.as_ptr().add(off + 12));
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            acc2 = vfmaq_f32(acc2, a2, b2);
            acc3 = vfmaq_f32(acc3, a3, b3);
        }
        let acc01 = vaddq_f32(acc0, acc1);
        let acc23 = vaddq_f32(acc2, acc3);
        let acc = vaddq_f32(acc01, acc23);
        let lanes = [
            vgetq_lane_f32(acc, 0),
            vgetq_lane_f32(acc, 1),
            vgetq_lane_f32(acc, 2),
            vgetq_lane_f32(acc, 3),
        ];
        let mut total: f32 = lanes.iter().sum();
        for i in chunks * 16..n {
            total += a[i] * b[i];
        }
        total
    }
}

// ─── Blocked-tile f32 matmul (cache-aware) ─────────────────────────

/// Cache-aware blocked f32 matmul: `out = A * B`.
///
/// `A` is row-major `M × K`; `B` is row-major `K × N`; output is
/// row-major `M × N`. Operates over zero-copy `&[f32]` / `&mut [f32]`
/// views (caller supplies these from `bytemuck::cast_slice` over the
/// arena's aligned byte buffers).
///
/// **Cost model**: blocks are sized so that one row-strip of A
/// (`BM × BK`), one column-strip of B (`BK × BN`), and one tile of
/// the output (`BM × BN`) all fit in L1 cache. The inner kernel
/// streams through `simd_f32_dot` per output cell so the SIMD path
/// already chosen by `resolve_path()` carries through.
///
/// For square `N × N × N` matmul the asymptotic cost is `N³`
/// multiply-adds; the blocked layout reduces L1 misses from `Θ(N³)`
/// (naïve) to `Θ(N³ / B)` where `B` is the block dimension — a
/// constant-factor win that compounds with SIMD lane width to give
/// near-peak GFLOPS on the host's natural register width.
pub fn matmul_f32_blocked(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bt_scratch: &mut Vec<f32>,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Register-blocked FMA micro-kernel on x86: AVX2 and AVX-512 both carry
    // FMA, so a 4×16 output tile is accumulated entirely in YMM registers
    // while B rows stream contiguously — no per-element horizontal reduction
    // and no B transpose, which is where the dot-product form bled cycles.
    #[cfg(target_arch = "x86_64")]
    {
        let p = resolve_path();
        if p == 2 || p == 3 {
            let _ = &bt_scratch; // unused on this path
                                 // SAFETY: `resolve_path` confirmed AVX2 + FMA are present.
            unsafe {
                x86::matmul_f32_fma(a, b, out, m, k, n);
            }
            return;
        }
    }

    // Portable fallback (NEON dot product / scalar): pre-transpose B into a
    // column-major scratch so each output element's dot product reads
    // contiguous memory, then reduce per element.
    const BS: usize = 64;
    bt_scratch.clear();
    bt_scratch.resize(k * n, 0.0);
    for kk in 0..k {
        for j in 0..n {
            bt_scratch[j * k + kk] = b[kk * n + j];
        }
    }

    let mut ii = 0;
    while ii < m {
        let i_end = (ii + BS).min(m);
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + BS).min(n);
            for i in ii..i_end {
                let row = &a[i * k..i * k + k];
                let out_row = &mut out[i * n..i * n + n];
                for j in jj..j_end {
                    let col = &bt_scratch[j * k..j * k + k];
                    out_row[j] = simd_f32_dot(row, col);
                }
            }
            jj += BS;
        }
        ii += BS;
    }
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

    #[test]
    fn blocked_matmul_matches_naive() {
        // 17 × 19 × 23 — odd sizes exercise the tail handling.
        let m = 17usize;
        let k = 19usize;
        let n = 23usize;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001 + 1.0).collect();
        let mut bt_scratch = Vec::new();
        let mut got = vec![0f32; m * n];
        matmul_f32_blocked(&a, &b, &mut got, m, k, n, &mut bt_scratch);

        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                want[i * n + j] = s;
            }
        }
        for i in 0..m * n {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "diff at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn blocked_matmul_handles_large_dims() {
        // 128 × 128 × 128 — exercises the inter-block stride.
        let n = 128usize;
        let a: Vec<f32> = (0..n * n).map(|i| ((i % 31) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32) * 0.01).collect();
        let mut bt_scratch = Vec::new();
        let mut got = vec![0f32; n * n];
        matmul_f32_blocked(&a, &b, &mut got, n, n, n, &mut bt_scratch);
        // Sanity: corner element matches a manual dot.
        let row0 = &a[0..n];
        let col0: Vec<f32> = (0..n).map(|kk| b[kk * n]).collect();
        let want00: f32 = row0.iter().zip(col0.iter()).map(|(x, y)| x * y).sum();
        assert!((got[0] - want00).abs() < 1e-2);
    }
}
