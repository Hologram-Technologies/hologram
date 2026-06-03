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

/// Below this `m·k·n`, a matmul runs single-threaded — keeping the small-op
/// path single-core-optimal. The pool's wake/barrier + per-tile dispatch costs
/// ~tens of µs, which only pays once the per-core slice is large enough: 128³
/// (≈2.1M) is *faster* sequential (measured 46µs vs 59µs split), while 256³
/// (≈16.8M) and production weight matmuls (e.g. 64·256·1024 ≈ 16.8M) win ~1.8×.
/// The crossover sits between, so the grain is set at 8M.
#[cfg(feature = "parallel")]
const PAR_THRESHOLD: u64 = 1 << 23;

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
    #[allow(unreachable_code)]
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

    /// Register-tiled FMA **leaf** of the cache-oblivious recursion: a 4×16
    /// output tile (`MR×NR`) held entirely in YMM accumulators while B rows
    /// stream contiguously. Operates on **strided** sub-matrix views (`lda`,
    /// `ldb`, `ldc` = the parent matrices' row strides), so a sub-block of a
    /// larger matrix is multiplied in place with no packing/copy. `accumulate`
    /// selects `C += A·B` (the `k`-split combine) vs `C = A·B` (a fresh tile).
    ///
    /// # Safety
    /// Requires AVX2 + FMA (the caller gates on `resolve_path`); pointers must
    /// address `m×k`, `k×n`, `m×n` sub-matrices with the given row strides.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_fma_strided(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        const NR: usize = 16;
        let mut i = 0;
        while i + MR <= m {
            let mut j = 0;
            while j + NR <= n {
                let mut c = [[_mm256_setzero_ps(); 2]; MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        let orow = out.add((i + r) * ldc + j);
                        cr[0] = _mm256_loadu_ps(orow);
                        cr[1] = _mm256_loadu_ps(orow.add(8));
                    }
                }
                for kk in 0..k {
                    let brow = b.add(kk * ldb + j);
                    let b0 = _mm256_loadu_ps(brow);
                    let b1 = _mm256_loadu_ps(brow.add(8));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = _mm256_set1_ps(*a.add((i + r) * lda + kk));
                        cr[0] = _mm256_fmadd_ps(av, b0, cr[0]);
                        cr[1] = _mm256_fmadd_ps(av, b1, cr[1]);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    let orow = out.add((i + r) * ldc + j);
                    _mm256_storeu_ps(orow, cr[0]);
                    _mm256_storeu_ps(orow.add(8), cr[1]);
                }
                j += NR;
            }
            // Column remainder for this row block.
            while j < n {
                for r in 0..MR {
                    let mut s = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                    for kk in 0..k {
                        s += *a.add((i + r) * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add((i + r) * ldc + j) = s;
                }
                j += 1;
            }
            i += MR;
        }
        // Row remainder.
        while i < m {
            for j in 0..n {
                let mut s = if accumulate {
                    *out.add(i * ldc + j)
                } else {
                    0.0
                };
                for kk in 0..k {
                    s += *a.add(i * lda + kk) * *b.add(kk * ldb + j);
                }
                *out.add(i * ldc + j) = s;
            }
            i += 1;
        }
    }

    /// **Cache-oblivious recursive matmul** — hologram's "lattice recursion"
    /// at the kernel level. `C = A·B` (row-major sub-matrices addressed by
    /// strides) is computed by recursively halving the **largest** of `m, n,
    /// k` until a sub-problem fits the register-tile leaf, which runs the FMA
    /// microkernel. Splitting the largest dimension keeps every sub-problem's
    /// working set ⊆ the next cache tier *without any per-cache block
    /// constant* — blocking for L1/L2/L3 emerges from the subdivision, so the
    /// miss count stays compulsory-only (each datum read once, reused before
    /// eviction) and efficiency holds at arbitrary size: no contrived ceiling.
    /// The single threshold `LEAF` is the register-tier base case (the analog
    /// of a term-recursion base case), not a cache-size knob.
    ///
    /// # Safety
    /// AVX2+FMA must be present (the caller checks `resolve_path`); pointers
    /// must address `m×k`, `k×n`, `m×n` sub-matrices with the given strides.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_recursive(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        // Register-tier leaf: a 64³ tile's three operands (~48 KiB) sit in
        // L1/L2 and the 4×16 microkernel saturates the FMA units. Above this,
        // recurse — never a per-cache block constant.
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_fma_strided(a, b, out, m, k, n, lda, ldb, ldc, accumulate);
            return;
        }
        if m >= n && m >= k {
            let h = m / 2;
            matmul_f32_recursive(a, b, out, h, k, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h * lda),
                b,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else if n >= m && n >= k {
            let h = n / 2;
            matmul_f32_recursive(a, b, out, m, k, h, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a,
                b.add(h),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else {
            // k-split: the two half-products accumulate into the same C tile.
            let h = k / 2;
            matmul_f32_recursive(a, b, out, m, h, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h),
                b.add(h * ldb),
                out,
                m,
                k - h,
                n,
                lda,
                ldb,
                ldc,
                true,
            );
        }
    }

    /// Matmul with **panel-packed B** (`C = A·Bᵖ`). `bpacked` holds B in
    /// `NR`-wide column panels, each `k`-contiguous (see
    /// [`crate::layout::pack_b_panels`]): panel `p`, row `kk` occupies the 16 floats at
    /// `bpacked[(p·k + kk)·16 ..]`. The leaf therefore streams B **fully
    /// contiguously** across `kk` — no strided row gather, so once a panel is
    /// resident its reuse across the `m`-tiles costs no further misses. This is
    /// the kernel half of the compile-time weight-layout monomorphism: the
    /// constant weight is packed once at compile time (zero runtime copy) into
    /// exactly the order this loop consumes. `a` (the activation) is read
    /// row-contiguous; `out` is written in 4×16 tiles.
    ///
    /// # Safety
    /// AVX2+FMA required; `a` is `m×k` (row stride `lda`), `out` is `m×n` (row
    /// stride `ldc`), `bpacked` is `⌈n/16⌉·k·16` floats.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_b(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        let n_panels = n.div_ceil(16);
        let mut i = 0;
        while i + MR <= m {
            for p in 0..n_panels {
                let cols = core::cmp::min(16, n - p * 16);
                if cols == 16 {
                    // Full 4×16 register tile. `k_stride` is the panel stride
                    // (the weight's full k), so a k-subrange of the recursion
                    // still indexes the correct panel rows.
                    let base = p * k_stride * 16;
                    let mut c = [[_mm256_setzero_ps(); 2]; MR];
                    if accumulate {
                        for (r, cr) in c.iter_mut().enumerate() {
                            let orow = out.add((i + r) * ldc + p * 16);
                            cr[0] = _mm256_loadu_ps(orow);
                            cr[1] = _mm256_loadu_ps(orow.add(8));
                        }
                    }
                    for kk in 0..k {
                        let bp = bpacked.add(base + kk * 16);
                        let b0 = _mm256_loadu_ps(bp);
                        let b1 = _mm256_loadu_ps(bp.add(8));
                        for (r, cr) in c.iter_mut().enumerate() {
                            let av = _mm256_set1_ps(*a.add((i + r) * lda + kk));
                            cr[0] = _mm256_fmadd_ps(av, b0, cr[0]);
                            cr[1] = _mm256_fmadd_ps(av, b1, cr[1]);
                        }
                    }
                    for (r, cr) in c.iter().enumerate() {
                        let orow = out.add((i + r) * ldc + p * 16);
                        _mm256_storeu_ps(orow, cr[0]);
                        _mm256_storeu_ps(orow.add(8), cr[1]);
                    }
                } else {
                    // Partial trailing panel (n not a multiple of 16): scalar.
                    for r in 0..MR {
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *a.add((i + r) * lda + kk)
                                    * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
            i += MR;
        }
        // Row remainder (m not a multiple of MR): scalar over the packed panels.
        while i < m {
            for j in 0..n {
                let p = j / 16;
                let c = j % 16;
                let mut s = if accumulate {
                    *out.add(i * ldc + j)
                } else {
                    0.0
                };
                for kk in 0..k {
                    s += *a.add(i * lda + kk) * *bpacked.add((p * k_stride + kk) * 16 + c);
                }
                *out.add(i * ldc + j) = s;
            }
            i += 1;
        }
    }

    /// **Cache-oblivious recursive matmul with panel-packed B** — the packed
    /// twin of [`matmul_f32_recursive`]. Halve the largest of m, n, k (N at a
    /// 16-column panel boundary, K accumulating) down to the packed
    /// register-tile leaf, so a packed weight enjoys the *same* compulsory-only
    /// miss behaviour at arbitrary M as the unpacked path — the per-M-tile
    /// re-streaming of B a flat packed loop would suffer is gone, with no
    /// per-cache block constant and still zero copy. `k_stride` is the weight's
    /// full k (the panel stride), invariant under the recursion.
    ///
    /// # Safety
    /// AVX2+FMA required; pointers address the relevant sub-matrices and
    /// `bpacked` is laid out by [`super::pack_b_panels`] with panel stride
    /// `k_stride`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_recursive(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_packed_b(a, bpacked, out, m, k, n, lda, ldc, k_stride, accumulate);
            return;
        }
        if m >= n && m >= k && m > LEAF {
            let h = m / 2;
            matmul_f32_packed_recursive(a, bpacked, out, h, k, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h * lda),
                bpacked,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else if n >= m && n >= k && n > LEAF {
            // Split N on a 16-column panel boundary so packed panels stay whole.
            let mut h = (n / 2) & !15;
            if h < 16 {
                h = 16;
            }
            matmul_f32_packed_recursive(a, bpacked, out, m, k, h, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a,
                bpacked.add((h / 16) * k_stride * 16),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else {
            // k-split: the second half accumulates into the same C tile;
            // `bpacked` shifts by `h` rows within every panel (stride
            // `k_stride` unchanged), `a` shifts by `h` columns.
            let h = k / 2;
            matmul_f32_packed_recursive(a, bpacked, out, m, h, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h),
                bpacked.add(h * 16),
                out,
                m,
                k - h,
                n,
                lda,
                ldc,
                k_stride,
                true,
            );
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

    // ─── NEON register-tiled f32 matmul ────────────────────────────
    // The aarch64 analog of the x86 `matmul_f32_fma_strided` /
    // `matmul_f32_recursive` pair. A 4×16 output tile (MR×NR) is held in
    // sixteen 128-bit NEON accumulators (4 q-registers per row) while B rows
    // stream contiguously — no per-cell horizontal reduction, no per-call B
    // transpose. The cache-oblivious recursion is structurally identical to
    // the x86 path; only the leaf differs, so the same compulsory-only miss
    // behaviour and LEAF=64 register-tier base case carry over. aarch64 has 32
    // q-registers, so the 16 C accumulators + 4 B regs + 1 broadcast fit.

    /// Register-tiled NEON leaf: `C [±]= A·B` over strided sub-matrix views.
    ///
    /// # Safety
    /// NEON (baseline on aarch64); pointers must address `m×k`, `k×n`, `m×n`
    /// sub-matrices with the given row strides (`lda`, `ldb`, `ldc`).
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_fma_strided(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        const NR: usize = 16;
        let mut i = 0;
        while i + MR <= m {
            let mut j = 0;
            while j + NR <= n {
                let mut c = [[vdupq_n_f32(0.0); 4]; MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        let orow = out.add((i + r) * ldc + j);
                        cr[0] = vld1q_f32(orow);
                        cr[1] = vld1q_f32(orow.add(4));
                        cr[2] = vld1q_f32(orow.add(8));
                        cr[3] = vld1q_f32(orow.add(12));
                    }
                }
                for kk in 0..k {
                    let brow = b.add(kk * ldb + j);
                    let b0 = vld1q_f32(brow);
                    let b1 = vld1q_f32(brow.add(4));
                    let b2 = vld1q_f32(brow.add(8));
                    let b3 = vld1q_f32(brow.add(12));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = vdupq_n_f32(*a.add((i + r) * lda + kk));
                        cr[0] = vfmaq_f32(cr[0], av, b0);
                        cr[1] = vfmaq_f32(cr[1], av, b1);
                        cr[2] = vfmaq_f32(cr[2], av, b2);
                        cr[3] = vfmaq_f32(cr[3], av, b3);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    let orow = out.add((i + r) * ldc + j);
                    vst1q_f32(orow, cr[0]);
                    vst1q_f32(orow.add(4), cr[1]);
                    vst1q_f32(orow.add(8), cr[2]);
                    vst1q_f32(orow.add(12), cr[3]);
                }
                j += NR;
            }
            // Column remainder for this row block (n not a multiple of NR).
            while j < n {
                for r in 0..MR {
                    let mut s = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                    for kk in 0..k {
                        s += *a.add((i + r) * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add((i + r) * ldc + j) = s;
                }
                j += 1;
            }
            i += MR;
        }
        // Row remainder (m not a multiple of MR) — vectorized GEMV per row.
        // Each remaining row streams B across `k`, accumulating output columns
        // 16-wide (then 4-wide, then a scalar tail). This is the single-token
        // decode path (M=1), which would otherwise run fully scalar.
        while i < m {
            let arow = a.add(i * lda);
            let orow = out.add(i * ldc);
            let mut j = 0;
            while j + 16 <= n {
                let (mut c0, mut c1, mut c2, mut c3) = if accumulate {
                    (
                        vld1q_f32(orow.add(j)),
                        vld1q_f32(orow.add(j + 4)),
                        vld1q_f32(orow.add(j + 8)),
                        vld1q_f32(orow.add(j + 12)),
                    )
                } else {
                    let z = vdupq_n_f32(0.0);
                    (z, z, z, z)
                };
                for kk in 0..k {
                    let av = vdupq_n_f32(*arow.add(kk));
                    let brow = b.add(kk * ldb + j);
                    c0 = vfmaq_f32(c0, av, vld1q_f32(brow));
                    c1 = vfmaq_f32(c1, av, vld1q_f32(brow.add(4)));
                    c2 = vfmaq_f32(c2, av, vld1q_f32(brow.add(8)));
                    c3 = vfmaq_f32(c3, av, vld1q_f32(brow.add(12)));
                }
                vst1q_f32(orow.add(j), c0);
                vst1q_f32(orow.add(j + 4), c1);
                vst1q_f32(orow.add(j + 8), c2);
                vst1q_f32(orow.add(j + 12), c3);
                j += 16;
            }
            while j + 4 <= n {
                let mut c0 = if accumulate {
                    vld1q_f32(orow.add(j))
                } else {
                    vdupq_n_f32(0.0)
                };
                for kk in 0..k {
                    let av = vdupq_n_f32(*arow.add(kk));
                    c0 = vfmaq_f32(c0, av, vld1q_f32(b.add(kk * ldb + j)));
                }
                vst1q_f32(orow.add(j), c0);
                j += 4;
            }
            while j < n {
                let mut s = if accumulate { *orow.add(j) } else { 0.0 };
                for kk in 0..k {
                    s += *arow.add(kk) * *b.add(kk * ldb + j);
                }
                *orow.add(j) = s;
                j += 1;
            }
            i += 1;
        }
    }

    /// Cache-oblivious recursive matmul over the NEON leaf (the aarch64 twin of
    /// x86's `matmul_f32_recursive`). Halves the largest of m, n, k until the
    /// register-tier leaf, so blocking for each cache tier emerges from the
    /// subdivision with no per-cache block constant.
    ///
    /// # Safety
    /// NEON (baseline on aarch64); pointers must address `m×k`, `k×n`, `m×n`
    /// sub-matrices with the given strides.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_recursive(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_fma_strided(a, b, out, m, k, n, lda, ldb, ldc, accumulate);
            return;
        }
        if m >= n && m >= k {
            let h = m / 2;
            matmul_f32_recursive(a, b, out, h, k, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h * lda),
                b,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else if n >= m && n >= k {
            let h = n / 2;
            matmul_f32_recursive(a, b, out, m, k, h, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a,
                b.add(h),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else {
            // k-split: the two half-products accumulate into the same C tile.
            let h = k / 2;
            matmul_f32_recursive(a, b, out, m, h, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h),
                b.add(h * ldb),
                out,
                m,
                k - h,
                n,
                lda,
                ldb,
                ldc,
                true,
            );
        }
    }

    /// NEON packed-B leaf (`C [±]= A·Bᵖ`). `bpacked` is the same 16-wide
    /// panel-packed layout the x86 packed kernel consumes (see
    /// [`crate::layout::pack_b_panels`]): panel `p`, row `kk` occupies the 16
    /// floats at `bpacked[(p·k_stride + kk)·16 ..]`, streamed contiguously into
    /// four NEON registers. So aarch64 reuses the compile-time weight layout
    /// unchanged — zero runtime copy.
    ///
    /// # Safety
    /// NEON (baseline on aarch64); `a` is `m×k` (stride `lda`), `out` is `m×n`
    /// (stride `ldc`), `bpacked` is panel-packed with panel stride `k_stride`.
    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_b(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        let n_panels = n.div_ceil(16);
        let mut i = 0;
        while i + MR <= m {
            for p in 0..n_panels {
                let cols = core::cmp::min(16, n - p * 16);
                if cols == 16 {
                    let base = p * k_stride * 16;
                    let mut c = [[vdupq_n_f32(0.0); 4]; MR];
                    if accumulate {
                        for (r, cr) in c.iter_mut().enumerate() {
                            let orow = out.add((i + r) * ldc + p * 16);
                            cr[0] = vld1q_f32(orow);
                            cr[1] = vld1q_f32(orow.add(4));
                            cr[2] = vld1q_f32(orow.add(8));
                            cr[3] = vld1q_f32(orow.add(12));
                        }
                    }
                    for kk in 0..k {
                        let bp = bpacked.add(base + kk * 16);
                        let b0 = vld1q_f32(bp);
                        let b1 = vld1q_f32(bp.add(4));
                        let b2 = vld1q_f32(bp.add(8));
                        let b3 = vld1q_f32(bp.add(12));
                        for (r, cr) in c.iter_mut().enumerate() {
                            let av = vdupq_n_f32(*a.add((i + r) * lda + kk));
                            cr[0] = vfmaq_f32(cr[0], av, b0);
                            cr[1] = vfmaq_f32(cr[1], av, b1);
                            cr[2] = vfmaq_f32(cr[2], av, b2);
                            cr[3] = vfmaq_f32(cr[3], av, b3);
                        }
                    }
                    for (r, cr) in c.iter().enumerate() {
                        let orow = out.add((i + r) * ldc + p * 16);
                        vst1q_f32(orow, cr[0]);
                        vst1q_f32(orow.add(4), cr[1]);
                        vst1q_f32(orow.add(8), cr[2]);
                        vst1q_f32(orow.add(12), cr[3]);
                    }
                } else {
                    // Partial trailing panel (n not a multiple of 16): scalar.
                    for r in 0..MR {
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *a.add((i + r) * lda + kk)
                                    * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
            i += MR;
        }
        // Row remainder (m not a multiple of MR) — vectorized GEMV over packed
        // panels. Each remaining row accumulates each full 16-wide panel in four
        // NEON registers across `k`; the trailing partial panel falls to scalar.
        // This is the single-token decode path (M=1), packed-weight form.
        while i < m {
            let arow = a.add(i * lda);
            let orow = out.add(i * ldc);
            let n_full = n / 16;
            for p in 0..n_full {
                let base = p * k_stride * 16;
                let (mut c0, mut c1, mut c2, mut c3) = if accumulate {
                    (
                        vld1q_f32(orow.add(p * 16)),
                        vld1q_f32(orow.add(p * 16 + 4)),
                        vld1q_f32(orow.add(p * 16 + 8)),
                        vld1q_f32(orow.add(p * 16 + 12)),
                    )
                } else {
                    let z = vdupq_n_f32(0.0);
                    (z, z, z, z)
                };
                for kk in 0..k {
                    let av = vdupq_n_f32(*arow.add(kk));
                    let bp = bpacked.add(base + kk * 16);
                    c0 = vfmaq_f32(c0, av, vld1q_f32(bp));
                    c1 = vfmaq_f32(c1, av, vld1q_f32(bp.add(4)));
                    c2 = vfmaq_f32(c2, av, vld1q_f32(bp.add(8)));
                    c3 = vfmaq_f32(c3, av, vld1q_f32(bp.add(12)));
                }
                vst1q_f32(orow.add(p * 16), c0);
                vst1q_f32(orow.add(p * 16 + 4), c1);
                vst1q_f32(orow.add(p * 16 + 8), c2);
                vst1q_f32(orow.add(p * 16 + 12), c3);
            }
            for j in n_full * 16..n {
                let p = j / 16;
                let c = j % 16;
                let mut s = if accumulate { *orow.add(j) } else { 0.0 };
                for kk in 0..k {
                    s += *arow.add(kk) * *bpacked.add((p * k_stride + kk) * 16 + c);
                }
                *orow.add(j) = s;
            }
            i += 1;
        }
    }

    /// Cache-oblivious recursive matmul with panel-packed B over the NEON leaf
    /// (the aarch64 twin of x86's `matmul_f32_packed_recursive`). N is split on
    /// 16-column panel boundaries so packed panels stay whole; K accumulates.
    ///
    /// # Safety
    /// NEON (baseline on aarch64); pointers address the relevant sub-matrices
    /// and `bpacked` is panel-packed with panel stride `k_stride`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_recursive(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_packed_b(a, bpacked, out, m, k, n, lda, ldc, k_stride, accumulate);
            return;
        }
        if m >= n && m >= k && m > LEAF {
            let h = m / 2;
            matmul_f32_packed_recursive(a, bpacked, out, h, k, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h * lda),
                bpacked,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else if n >= m && n >= k && n > LEAF {
            // Split N on a 16-column panel boundary so packed panels stay whole.
            let mut h = (n / 2) & !15;
            if h < 16 {
                h = 16;
            }
            matmul_f32_packed_recursive(a, bpacked, out, m, k, h, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a,
                bpacked.add((h / 16) * k_stride * 16),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else {
            // k-split: the second half accumulates into the same C tile.
            let h = k / 2;
            matmul_f32_packed_recursive(a, bpacked, out, m, h, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h),
                bpacked.add(h * 16),
                out,
                m,
                k - h,
                n,
                lda,
                ldc,
                k_stride,
                true,
            );
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use core::arch::wasm32::*;

    // ─── wasm SIMD128 register-tiled f32 matmul ────────────────────
    // The wasm32 analog of the NEON `aarch` matmul kernels. wasm SIMD128 is
    // 128-bit / 4×f32 — the same width as NEON — so this mirrors the 4×16 tile
    // structure exactly, with `f32x4_*` lanes. SIMD128 has no fused
    // multiply-add (that's relaxed-SIMD), so the inner step is a separate
    // `f32x4_add(acc, f32x4_mul(av, b))`; still 4-wide, a large win over the
    // scalar fallback. The cache-oblivious recursion is identical to the other
    // arches. No multi-core (wasm is single-threaded here); the sequential
    // recursion carries the whole matmul. This satisfies the wasm portability
    // mandate from a 128-bit design shared in shape with NEON.

    /// Register-tiled SIMD128 leaf: `C [±]= A·B` over strided sub-matrix views.
    ///
    /// # Safety
    /// simd128 must be enabled; pointers must address `m×k`, `k×n`, `m×n`
    /// sub-matrices with the given strides. `v128_load`/`v128_store` perform
    /// unaligned wasm loads (no alignment fault), matching the NEON path.
    #[target_feature(enable = "simd128")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_fma_strided(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        const NR: usize = 16;
        let mut i = 0;
        while i + MR <= m {
            let mut j = 0;
            while j + NR <= n {
                let mut c = [[f32x4_splat(0.0); 4]; MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        let orow = out.add((i + r) * ldc + j);
                        cr[0] = v128_load(orow as *const v128);
                        cr[1] = v128_load(orow.add(4) as *const v128);
                        cr[2] = v128_load(orow.add(8) as *const v128);
                        cr[3] = v128_load(orow.add(12) as *const v128);
                    }
                }
                for kk in 0..k {
                    let brow = b.add(kk * ldb + j);
                    let b0 = v128_load(brow as *const v128);
                    let b1 = v128_load(brow.add(4) as *const v128);
                    let b2 = v128_load(brow.add(8) as *const v128);
                    let b3 = v128_load(brow.add(12) as *const v128);
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = f32x4_splat(*a.add((i + r) * lda + kk));
                        cr[0] = f32x4_add(cr[0], f32x4_mul(av, b0));
                        cr[1] = f32x4_add(cr[1], f32x4_mul(av, b1));
                        cr[2] = f32x4_add(cr[2], f32x4_mul(av, b2));
                        cr[3] = f32x4_add(cr[3], f32x4_mul(av, b3));
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    let orow = out.add((i + r) * ldc + j);
                    v128_store(orow as *mut v128, cr[0]);
                    v128_store(orow.add(4) as *mut v128, cr[1]);
                    v128_store(orow.add(8) as *mut v128, cr[2]);
                    v128_store(orow.add(12) as *mut v128, cr[3]);
                }
                j += NR;
            }
            while j < n {
                for r in 0..MR {
                    let mut s = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                    for kk in 0..k {
                        s += *a.add((i + r) * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add((i + r) * ldc + j) = s;
                }
                j += 1;
            }
            i += MR;
        }
        // Row remainder (m not a multiple of MR) — vectorized GEMV per row
        // (single-token decode path, M=1), SIMD128 form.
        while i < m {
            let arow = a.add(i * lda);
            let orow = out.add(i * ldc);
            let mut j = 0;
            while j + 16 <= n {
                let (mut c0, mut c1, mut c2, mut c3) = if accumulate {
                    (
                        v128_load(orow.add(j) as *const v128),
                        v128_load(orow.add(j + 4) as *const v128),
                        v128_load(orow.add(j + 8) as *const v128),
                        v128_load(orow.add(j + 12) as *const v128),
                    )
                } else {
                    let z = f32x4_splat(0.0);
                    (z, z, z, z)
                };
                for kk in 0..k {
                    let av = f32x4_splat(*arow.add(kk));
                    let brow = b.add(kk * ldb + j);
                    c0 = f32x4_add(c0, f32x4_mul(av, v128_load(brow as *const v128)));
                    c1 = f32x4_add(c1, f32x4_mul(av, v128_load(brow.add(4) as *const v128)));
                    c2 = f32x4_add(c2, f32x4_mul(av, v128_load(brow.add(8) as *const v128)));
                    c3 = f32x4_add(c3, f32x4_mul(av, v128_load(brow.add(12) as *const v128)));
                }
                v128_store(orow.add(j) as *mut v128, c0);
                v128_store(orow.add(j + 4) as *mut v128, c1);
                v128_store(orow.add(j + 8) as *mut v128, c2);
                v128_store(orow.add(j + 12) as *mut v128, c3);
                j += 16;
            }
            while j + 4 <= n {
                let mut c0 = if accumulate {
                    v128_load(orow.add(j) as *const v128)
                } else {
                    f32x4_splat(0.0)
                };
                for kk in 0..k {
                    let av = f32x4_splat(*arow.add(kk));
                    c0 = f32x4_add(c0, f32x4_mul(av, v128_load(b.add(kk * ldb + j) as *const v128)));
                }
                v128_store(orow.add(j) as *mut v128, c0);
                j += 4;
            }
            while j < n {
                let mut s = if accumulate { *orow.add(j) } else { 0.0 };
                for kk in 0..k {
                    s += *arow.add(kk) * *b.add(kk * ldb + j);
                }
                *orow.add(j) = s;
                j += 1;
            }
            i += 1;
        }
    }

    /// Cache-oblivious recursive matmul over the SIMD128 leaf (wasm twin of the
    /// NEON `matmul_f32_recursive`).
    ///
    /// # Safety
    /// simd128 enabled; pointers address `m×k`, `k×n`, `m×n` sub-matrices with
    /// the given strides.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_recursive(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        accumulate: bool,
    ) {
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_fma_strided(a, b, out, m, k, n, lda, ldb, ldc, accumulate);
            return;
        }
        if m >= n && m >= k {
            let h = m / 2;
            matmul_f32_recursive(a, b, out, h, k, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h * lda),
                b,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else if n >= m && n >= k {
            let h = n / 2;
            matmul_f32_recursive(a, b, out, m, k, h, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a,
                b.add(h),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldb,
                ldc,
                accumulate,
            );
        } else {
            let h = k / 2;
            matmul_f32_recursive(a, b, out, m, h, n, lda, ldb, ldc, accumulate);
            matmul_f32_recursive(
                a.add(h),
                b.add(h * ldb),
                out,
                m,
                k - h,
                n,
                lda,
                ldb,
                ldc,
                true,
            );
        }
    }

    /// SIMD128 packed-B leaf (`C [±]= A·Bᵖ`) over the 16-wide panel layout.
    ///
    /// # Safety
    /// simd128 enabled; `a` is `m×k` (stride `lda`), `out` is `m×n` (stride
    /// `ldc`), `bpacked` is panel-packed with panel stride `k_stride`.
    #[target_feature(enable = "simd128")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_b(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const MR: usize = 4;
        let n_panels = n.div_ceil(16);
        let mut i = 0;
        while i + MR <= m {
            for p in 0..n_panels {
                let cols = core::cmp::min(16, n - p * 16);
                if cols == 16 {
                    let base = p * k_stride * 16;
                    let mut c = [[f32x4_splat(0.0); 4]; MR];
                    if accumulate {
                        for (r, cr) in c.iter_mut().enumerate() {
                            let orow = out.add((i + r) * ldc + p * 16);
                            cr[0] = v128_load(orow as *const v128);
                            cr[1] = v128_load(orow.add(4) as *const v128);
                            cr[2] = v128_load(orow.add(8) as *const v128);
                            cr[3] = v128_load(orow.add(12) as *const v128);
                        }
                    }
                    for kk in 0..k {
                        let bp = bpacked.add(base + kk * 16);
                        let b0 = v128_load(bp as *const v128);
                        let b1 = v128_load(bp.add(4) as *const v128);
                        let b2 = v128_load(bp.add(8) as *const v128);
                        let b3 = v128_load(bp.add(12) as *const v128);
                        for (r, cr) in c.iter_mut().enumerate() {
                            let av = f32x4_splat(*a.add((i + r) * lda + kk));
                            cr[0] = f32x4_add(cr[0], f32x4_mul(av, b0));
                            cr[1] = f32x4_add(cr[1], f32x4_mul(av, b1));
                            cr[2] = f32x4_add(cr[2], f32x4_mul(av, b2));
                            cr[3] = f32x4_add(cr[3], f32x4_mul(av, b3));
                        }
                    }
                    for (r, cr) in c.iter().enumerate() {
                        let orow = out.add((i + r) * ldc + p * 16);
                        v128_store(orow as *mut v128, cr[0]);
                        v128_store(orow.add(4) as *mut v128, cr[1]);
                        v128_store(orow.add(8) as *mut v128, cr[2]);
                        v128_store(orow.add(12) as *mut v128, cr[3]);
                    }
                } else {
                    for r in 0..MR {
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *a.add((i + r) * lda + kk)
                                    * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
            i += MR;
        }
        // Row remainder (m not a multiple of MR) — vectorized GEMV over packed
        // panels (single-token decode path, M=1), SIMD128 form.
        while i < m {
            let arow = a.add(i * lda);
            let orow = out.add(i * ldc);
            let n_full = n / 16;
            for p in 0..n_full {
                let base = p * k_stride * 16;
                let (mut c0, mut c1, mut c2, mut c3) = if accumulate {
                    (
                        v128_load(orow.add(p * 16) as *const v128),
                        v128_load(orow.add(p * 16 + 4) as *const v128),
                        v128_load(orow.add(p * 16 + 8) as *const v128),
                        v128_load(orow.add(p * 16 + 12) as *const v128),
                    )
                } else {
                    let z = f32x4_splat(0.0);
                    (z, z, z, z)
                };
                for kk in 0..k {
                    let av = f32x4_splat(*arow.add(kk));
                    let bp = bpacked.add(base + kk * 16);
                    c0 = f32x4_add(c0, f32x4_mul(av, v128_load(bp as *const v128)));
                    c1 = f32x4_add(c1, f32x4_mul(av, v128_load(bp.add(4) as *const v128)));
                    c2 = f32x4_add(c2, f32x4_mul(av, v128_load(bp.add(8) as *const v128)));
                    c3 = f32x4_add(c3, f32x4_mul(av, v128_load(bp.add(12) as *const v128)));
                }
                v128_store(orow.add(p * 16) as *mut v128, c0);
                v128_store(orow.add(p * 16 + 4) as *mut v128, c1);
                v128_store(orow.add(p * 16 + 8) as *mut v128, c2);
                v128_store(orow.add(p * 16 + 12) as *mut v128, c3);
            }
            for j in n_full * 16..n {
                let p = j / 16;
                let c = j % 16;
                let mut s = if accumulate { *orow.add(j) } else { 0.0 };
                for kk in 0..k {
                    s += *arow.add(kk) * *bpacked.add((p * k_stride + kk) * 16 + c);
                }
                *orow.add(j) = s;
            }
            i += 1;
        }
    }

    /// Cache-oblivious recursive packed matmul over the SIMD128 leaf (wasm twin
    /// of the NEON `matmul_f32_packed_recursive`).
    ///
    /// # Safety
    /// simd128 enabled; pointers address the relevant sub-matrices and
    /// `bpacked` is panel-packed with panel stride `k_stride`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn matmul_f32_packed_recursive(
        a: *const f32,
        bpacked: *const f32,
        out: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldc: usize,
        k_stride: usize,
        accumulate: bool,
    ) {
        const LEAF: usize = 64;
        if m <= LEAF && k <= LEAF && n <= LEAF {
            matmul_f32_packed_b(a, bpacked, out, m, k, n, lda, ldc, k_stride, accumulate);
            return;
        }
        if m >= n && m >= k && m > LEAF {
            let h = m / 2;
            matmul_f32_packed_recursive(a, bpacked, out, h, k, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h * lda),
                bpacked,
                out.add(h * ldc),
                m - h,
                k,
                n,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else if n >= m && n >= k && n > LEAF {
            let mut h = (n / 2) & !15;
            if h < 16 {
                h = 16;
            }
            matmul_f32_packed_recursive(a, bpacked, out, m, k, h, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a,
                bpacked.add((h / 16) * k_stride * 16),
                out.add(h),
                m,
                k,
                n - h,
                lda,
                ldc,
                k_stride,
                accumulate,
            );
        } else {
            let h = k / 2;
            matmul_f32_packed_recursive(a, bpacked, out, m, h, n, lda, ldc, k_stride, accumulate);
            matmul_f32_packed_recursive(
                a.add(h),
                bpacked.add(h * 16),
                out,
                m,
                k - h,
                n,
                lda,
                ldc,
                k_stride,
                true,
            );
        }
    }
}

// ─── Compile-time weight-layout monomorphism ───────────────────────
// The packing transform that produces `Bᵖ` lives in `crate::layout`
// (not CPU-gated, so the compiler can run it); this kernel consumes it.

/// `C = A·Bᵖ` where `Bᵖ` is [`crate::layout::pack_b_panels`]-packed (the compile-time
/// weight layout). Runtime-dispatched: the AVX2+FMA leaf streams packed
/// panels contiguously; the portable fallback reads the same layout scalarly.
/// Zero-copy — `bpacked` is the constant weight's stored representation.
pub fn matmul_f32_packed(
    a: &[f32],
    bpacked: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let p = resolve_path();
        if p == 2 || p == 3 {
            // UOR-native multi-core: cut the lattice recursion at the parallel
            // grain — bisect the output into ≈one disjoint tile per core — and
            // run the frontier across the pool, each tile executing the
            // sequential cache-oblivious recursion (panel-aligned column cuts).
            #[cfg(feature = "parallel")]
            {
                use crate::cpu::parallel::{self, SendConst, SendMut};
                let w = parallel::pool().width();
                if w > 1 && (m as u64) * (k as u64) * (n as u64) >= PAR_THRESHOLD {
                    let tiles = parallel::output_tiles(m, n, w, 16);
                    if tiles.len() > 1 {
                        let (ap, bp, op) = (
                            SendConst(a.as_ptr()),
                            SendConst(bpacked.as_ptr()),
                            SendMut(out.as_mut_ptr()),
                        );
                        let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                            .into_iter()
                            .map(|(r0, rows, c0, cols)| {
                                Box::new(move || {
                                    // Capture the Send wrappers whole (Rust 2021
                                    // disjoint capture would otherwise grab the
                                    // raw `*mut`/`*const` fields, not Send).
                                    let (ap, bp, op) = (ap, bp, op);
                                    // SAFETY: tiles are disjoint output regions;
                                    // a/bpacked are shared read-only; AVX2+FMA.
                                    unsafe {
                                        x86::matmul_f32_packed_recursive(
                                            ap.0.add(r0 * k),
                                            bp.0.add((c0 / 16) * k * 16),
                                            op.0.add(r0 * n + c0),
                                            rows,
                                            k,
                                            cols,
                                            k,
                                            n,
                                            k,
                                            false,
                                        );
                                    }
                                }) as Box<dyn FnOnce() + Send>
                            })
                            .collect();
                        parallel::pool().run(tasks);
                        return;
                    }
                }
            }
            // SAFETY: `resolve_path` confirmed AVX2 + FMA. lda=k, ldc=n,
            // k_stride=k (panel stride), fresh output (accumulate=false).
            unsafe {
                x86::matmul_f32_packed_recursive(
                    a.as_ptr(),
                    bpacked.as_ptr(),
                    out.as_mut_ptr(),
                    m,
                    k,
                    n,
                    k,
                    n,
                    k,
                    false,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64; the 16-wide packed layout is consumed
        // directly by the register-tiled kernel. Multi-core: bisect the output
        // into ≈one disjoint tile per core (panel-aligned column cuts), each
        // running the sequential cache-oblivious recursion.
        #[cfg(feature = "parallel")]
        {
            use crate::cpu::parallel::{self, SendConst, SendMut};
            let w = parallel::pool().width();
            if w > 1 && (m as u64) * (k as u64) * (n as u64) >= PAR_THRESHOLD {
                let tiles = parallel::output_tiles(m, n, w, 16);
                if tiles.len() > 1 {
                    let (ap, bp, op) = (
                        SendConst(a.as_ptr()),
                        SendConst(bpacked.as_ptr()),
                        SendMut(out.as_mut_ptr()),
                    );
                    let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                        .into_iter()
                        .map(|(r0, rows, c0, cols)| {
                            Box::new(move || {
                                let (ap, bp, op) = (ap, bp, op);
                                // SAFETY: tiles are disjoint output regions;
                                // a/bpacked are shared read-only; NEON baseline.
                                unsafe {
                                    aarch::matmul_f32_packed_recursive(
                                        ap.0.add(r0 * k),
                                        bp.0.add((c0 / 16) * k * 16),
                                        op.0.add(r0 * n + c0),
                                        rows,
                                        k,
                                        cols,
                                        k,
                                        n,
                                        k,
                                        false,
                                    );
                                }
                            }) as Box<dyn FnOnce() + Send>
                        })
                        .collect();
                    parallel::pool().run(tasks);
                    return;
                }
            }
        }
        // SAFETY: NEON baseline. lda=k, ldc=n, k_stride=k, fresh output.
        unsafe {
            aarch::matmul_f32_packed_recursive(
                a.as_ptr(),
                bpacked.as_ptr(),
                out.as_mut_ptr(),
                m,
                k,
                n,
                k,
                n,
                k,
                false,
            );
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // wasm SIMD128 (128-bit / 4×f32, same shape as NEON). Single-threaded;
        // the 16-wide packed layout is consumed directly by the register-tiled
        // SIMD128 kernel.
        // SAFETY: simd128 gate. lda=k, ldc=n, k_stride=k, fresh output.
        unsafe {
            wasm_simd::matmul_f32_packed_recursive(
                a.as_ptr(),
                bpacked.as_ptr(),
                out.as_mut_ptr(),
                m,
                k,
                n,
                k,
                n,
                k,
                false,
            );
        }
        return;
    }
    // Portable scalar fallback over the packed layout (wasm-without-simd128 and
    // other no-SIMD targets; aarch64 NEON, wasm SIMD128, and AVX x86 return above).
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    for i in 0..m {
        for j in 0..n {
            let (p, c) = (j / 16, j % 16);
            let mut s = 0f32;
            for kk in 0..k {
                s += a[i * k + kk] * bpacked[(p * k + kk) * 16 + c];
            }
            out[i * n + j] = s;
        }
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
                                 // UOR-native multi-core: the lattice recursion's
                                 // frontier as disjoint output tiles, one per core.
            #[cfg(feature = "parallel")]
            {
                use crate::cpu::parallel::{self, SendConst, SendMut};
                let w = parallel::pool().width();
                if w > 1 && (m as u64) * (k as u64) * (n as u64) >= PAR_THRESHOLD {
                    let tiles = parallel::output_tiles(m, n, w, 1);
                    if tiles.len() > 1 {
                        let (ap, bp, op) = (
                            SendConst(a.as_ptr()),
                            SendConst(b.as_ptr()),
                            SendMut(out.as_mut_ptr()),
                        );
                        let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                            .into_iter()
                            .map(|(r0, rows, c0, cols)| {
                                Box::new(move || {
                                    // Capture the Send wrappers whole (see above).
                                    let (ap, bp, op) = (ap, bp, op);
                                    // SAFETY: tiles are disjoint output regions;
                                    // a/b are shared read-only; AVX2+FMA present.
                                    unsafe {
                                        x86::matmul_f32_recursive(
                                            ap.0.add(r0 * k),
                                            bp.0.add(c0),
                                            op.0.add(r0 * n + c0),
                                            rows,
                                            k,
                                            cols,
                                            k,
                                            n,
                                            n,
                                            false,
                                        );
                                    }
                                }) as Box<dyn FnOnce() + Send>
                            })
                            .collect();
                        parallel::pool().run(tasks);
                        return;
                    }
                }
            }
            // SAFETY: `resolve_path` confirmed AVX2 + FMA are present.
            // Strides are the contiguous row-major strides
            // (lda=k, ldb=n, ldc=n); `accumulate=false` is a fresh output.
            unsafe {
                x86::matmul_f32_recursive(
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    m,
                    k,
                    n,
                    k,
                    n,
                    n,
                    false,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = &bt_scratch; // unused on the NEON register-tiled path
                             // NEON is baseline on aarch64. Multi-core: bisect
                             // the output into ≈one disjoint tile per core, each
                             // running the sequential cache-oblivious recursion.
        #[cfg(feature = "parallel")]
        {
            use crate::cpu::parallel::{self, SendConst, SendMut};
            let w = parallel::pool().width();
            if w > 1 && (m as u64) * (k as u64) * (n as u64) >= PAR_THRESHOLD {
                let tiles = parallel::output_tiles(m, n, w, 1);
                if tiles.len() > 1 {
                    let (ap, bp, op) = (
                        SendConst(a.as_ptr()),
                        SendConst(b.as_ptr()),
                        SendMut(out.as_mut_ptr()),
                    );
                    let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                        .into_iter()
                        .map(|(r0, rows, c0, cols)| {
                            Box::new(move || {
                                let (ap, bp, op) = (ap, bp, op);
                                // SAFETY: tiles are disjoint output regions;
                                // a/b are shared read-only; NEON baseline.
                                unsafe {
                                    aarch::matmul_f32_recursive(
                                        ap.0.add(r0 * k),
                                        bp.0.add(c0),
                                        op.0.add(r0 * n + c0),
                                        rows,
                                        k,
                                        cols,
                                        k,
                                        n,
                                        n,
                                        false,
                                    );
                                }
                            }) as Box<dyn FnOnce() + Send>
                        })
                        .collect();
                    parallel::pool().run(tasks);
                    return;
                }
            }
        }
        // SAFETY: NEON baseline. Contiguous row-major strides (lda=k, ldb=n,
        // ldc=n); fresh output (accumulate=false).
        unsafe {
            aarch::matmul_f32_recursive(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                m,
                k,
                n,
                k,
                n,
                n,
                false,
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // wasm SIMD128 (128-bit / 4×f32, same shape as NEON). Single-threaded.
        // SAFETY: simd128 gate. Contiguous row-major strides (lda=k, ldb=n,
        // ldc=n); fresh output (accumulate=false).
        let _ = &bt_scratch; // unused on the SIMD128 register-tiled path
        unsafe {
            wasm_simd::matmul_f32_recursive(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                m,
                k,
                n,
                k,
                n,
                n,
                false,
            );
        }
        return;
    }

    // Portable fallback (scalar dot product) for wasm-without-simd128 and other
    // no-SIMD targets; aarch64 NEON, wasm SIMD128, and AVX x86 return above.
    // Pre-transpose B into a column-major scratch so each output element's dot
    // product reads contiguous memory, then reduce per element.
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
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
    fn packed_b_matmul_matches_naive() {
        // Odd sizes (non-multiple-of-16 n) exercise panel padding + the
        // partial-column / m-remainder tails; the large size drives the
        // cache-oblivious packed recursion (M/N/K splits incl. accumulation).
        #[cfg(target_arch = "x86_64")]
        for &(m, k, n) in &[
            (4usize, 8usize, 16usize),
            (5, 7, 19),
            (13, 11, 37),
            (64, 64, 64),
            (200, 130, 176),
        ] {
            if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
                return;
            }
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.3).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.002 + 0.5).collect();
            let packed = crate::layout::pack_b_panels(&b, k, n);
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
            // Both the leaf directly and the full recursion must match.
            for recursive in [false, true] {
                let mut got = vec![0f32; m * n];
                unsafe {
                    if recursive {
                        x86::matmul_f32_packed_recursive(
                            a.as_ptr(),
                            packed.as_ptr(),
                            got.as_mut_ptr(),
                            m,
                            k,
                            n,
                            k,
                            n,
                            k,
                            false,
                        );
                    } else {
                        x86::matmul_f32_packed_b(
                            a.as_ptr(),
                            packed.as_ptr(),
                            got.as_mut_ptr(),
                            m,
                            k,
                            n,
                            k,
                            n,
                            k,
                            false,
                        );
                    }
                }
                for idx in 0..m * n {
                    // Relative tolerance: tile/recursion summation reorders the
                    // f32 reduction vs the naïve reference, so large magnitudes
                    // differ in the last f32 ulps (≈2.5e-7 rel), well within f32.
                    let denom = want[idx].abs().max(1.0);
                    assert!(
                        (got[idx] - want[idx]).abs() / denom < 1e-4,
                        "{m}×{k}×{n} recursive={recursive} diff at {idx}: got {} want {}",
                        got[idx],
                        want[idx]
                    );
                }
            }
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

    /// Arch-agnostic packed-B matmul correctness through the **public**
    /// `matmul_f32_packed` dispatcher — so the aarch64 NEON packed kernel
    /// (which the x86-gated `packed_b_matmul_matches_naive` never exercises) and
    /// the wasm/portable fallback are both covered, not just the x86 leaf.
    #[test]
    fn packed_matmul_dispatch_matches_naive() {
        for &(m, k, n) in &[
            (1usize, 2048usize, 64usize), // GEMV: single-token decode shape (M=1)
            (1, 11, 37),                  // GEMV with partial trailing panel
            (2, 13, 48),                  // M=2 remainder, 3 full panels
            (3, 9, 50),                   // M=3 remainder, full + partial panel
            (4, 8, 16),
            (5, 7, 19), // non-multiple-of-16 n → partial-panel + m-remainder tails
            (13, 11, 37),
            (64, 64, 64),
            (130, 96, 176), // > LEAF → drives the cache-oblivious packed recursion
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.3).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.002 + 0.5).collect();
            let packed = crate::layout::pack_b_panels(&b, k, n);
            let mut got = vec![0f32; m * n];
            matmul_f32_packed(&a, &packed, &mut got, m, k, n);

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
            for idx in 0..m * n {
                let denom = want[idx].abs().max(1.0);
                assert!(
                    (got[idx] - want[idx]).abs() / denom < 1e-4,
                    "{m}×{k}×{n} diff at {idx}: got {} want {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }

    /// Unpacked GEMV / small-M correctness through `matmul_f32_blocked` — the
    /// vectorized M<4 row-remainder (single-token decode path, M=1) exercising
    /// the 16-wide, 4-wide, and scalar-tail column sub-paths.
    #[test]
    fn blocked_gemv_small_m_matches_naive() {
        for &(m, k, n) in &[
            (1usize, 2048usize, 64usize), // decode GEMV (16-wide path)
            (1, 17, 22),                  // 16-wide + 4-wide + scalar tail
            (2, 9, 35),
            (3, 31, 17), // 16-wide + scalar tail
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.0007 - 0.2).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0011 + 0.3).collect();
            let mut bt = Vec::new();
            let mut got = vec![0f32; m * n];
            matmul_f32_blocked(&a, &b, &mut got, m, k, n, &mut bt);

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
            for idx in 0..m * n {
                let denom = want[idx].abs().max(1.0);
                assert!(
                    (got[idx] - want[idx]).abs() / denom < 1e-4,
                    "{m}×{k}×{n} diff at {idx}: got {} want {}",
                    got[idx],
                    want[idx]
                );
            }
        }
    }

    /// Full-matrix equivalence at a size whose `m·k·n` exceeds `PAR_THRESHOLD`,
    /// so under `--features parallel` the multi-core output-tiling path runs
    /// (one disjoint tile per worker) for both the unpacked and packed kernels.
    /// Verifies every element (not just a corner) against the naïve reference.
    #[test]
    fn large_matmul_matches_naive_full() {
        let (m, k, n) = (256usize, 256usize, 256usize); // 16.7M MACs ≥ PAR_THRESHOLD
        let a: Vec<f32> = (0..m * k).map(|i| (((i % 53) as f32) - 26.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (((i % 37) as f32) - 18.0) * 0.01).collect();

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

        let mut bt_scratch = Vec::new();
        let mut got_blocked = vec![0f32; m * n];
        matmul_f32_blocked(&a, &b, &mut got_blocked, m, k, n, &mut bt_scratch);

        let packed = crate::layout::pack_b_panels(&b, k, n);
        let mut got_packed = vec![0f32; m * n];
        matmul_f32_packed(&a, &packed, &mut got_packed, m, k, n);

        for idx in 0..m * n {
            let denom = want[idx].abs().max(1.0);
            assert!(
                (got_blocked[idx] - want[idx]).abs() / denom < 1e-4,
                "blocked diff at {idx}: got {} want {}",
                got_blocked[idx],
                want[idx]
            );
            assert!(
                (got_packed[idx] - want[idx]).abs() / denom < 1e-4,
                "packed diff at {idx}: got {} want {}",
                got_packed[idx],
                want[idx]
            );
        }
    }
}
