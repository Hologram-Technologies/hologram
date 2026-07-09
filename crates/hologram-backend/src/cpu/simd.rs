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

// The four kernel constants below (`EXP_F32_LO/HI`, `I8_DOT_K_MAX`,
// `I4_VALUES`) and `E8_CODEBOOK` are `#[deprecated]` for one release before
// they are removed or demoted to `pub(crate)` in 0.9.0 — the deprecate-then-
// version-out lifecycle in `specs/docs/quality-gates.md`. rustc lints
// intra-crate uses of deprecated items, and the kernels here still read them,
// so the lint is suppressed for this module only.
#![allow(deprecated)]
#![allow(unsafe_op_in_unsafe_fn)]

use alloc::vec::Vec;
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
use core::sync::atomic::{AtomicU8, Ordering};
use hologram_types::DTypeId;

// The runtime dispatcher and the portable scalar kernels below exist for the
// targets that *choose* a path at run time (x86 feature detection) or have no
// vector unit. On wasm+simd128 every `simd_f32_*` wrapper calls its SIMD128
// twin unconditionally, so nothing here is reachable — gate them off rather
// than ship dead code to the deployed target.
/// SIMD path the runtime dispatcher selected. Cached after first
/// detection. Values: 0 = unresolved, 1 = scalar, 2 = AVX2+FMA,
/// 3 = AVX-512F, 4 = NEON.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
static SIMD_PATH: AtomicU8 = AtomicU8::new(0);

/// Below this `m·k·n`, a matmul runs single-threaded — keeping the small-op
/// path single-core-optimal. The pool's wake/barrier + per-tile dispatch costs
/// ~tens of µs, which only pays once the per-core slice is large enough: 128³
/// (≈2.1M) is *faster* sequential (measured 46µs vs 59µs split), while 256³
/// (≈16.8M) and production weight matmuls (e.g. 64·256·1024 ≈ 16.8M) win ~1.8×.
/// The crossover sits between, so the grain is set at 8M.
#[cfg(feature = "parallel")]
const PAR_THRESHOLD: u64 = 1 << 23;

/// Decode-GEMV parallel grain. A W8A8 omajor GEMV streams `k·n` weight bytes
/// through a compute-bound (not memory-bound) integer dot, so it scales near
/// the pool width — worth splitting at a much smaller work size than the f32
/// register-tiled matmul above. A `k·n` of 4.2M (a d=2048 decode projection)
/// is ~95 µs single-thread, well above the pool's fork/join cost; the 1M grain
/// keeps every real decode matmul parallel while leaving trivially small ones
/// (and the packing tails) serial.
#[cfg(all(
    feature = "parallel",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
const GEMV_PAR_THRESHOLD: u64 = 1 << 20;

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

/// SIMD-vectorized f32 add: `out[i] = a[i] + b[i]`. wasm SIMD128 (the deployed
/// target — residual adds, bias) has a dedicated lane; add is exact, so it is
/// bit-identical to the scalar loop.
#[inline]
pub fn simd_f32_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::add_f32(a, b, out) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

/// SIMD-vectorized f32 multiply. wasm SIMD128 (the deployed target — gating
/// muls, scale) has a dedicated lane; multiply is exact, so bit-identical to
/// the scalar loop.
#[inline]
pub fn simd_f32_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::mul_f32(a, b, out) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

/// SIMD-vectorized f32 fused multiply-add: `out[i] += a[i] * b[i]`. The x86 and
/// NEON lanes fuse (FMA); the wasm SIMD128 lane multiplies-then-adds (SIMD128
/// has no FMA, and the relaxed-SIMD fused op both changes the bits and regresses
/// latency-bound chains on the wasmtime host — see the matmul note), so the
/// wasm result may differ in the last ULP from the fused arches, as the f32 dot
/// path already does.
#[inline]
pub fn simd_f32_fmadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::fmadd_f32(a, b, out) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

/// SIMD-vectorized f32 dot product. wasm SIMD128 (the deployed target) has a
/// dedicated lane; x86 (AVX2/AVX-512) and aarch64 (NEON) select at runtime;
/// everything else is scalar.
#[inline]
pub fn simd_f32_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::dot_f32(a, b) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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

/// SIMD-vectorized horizontal sum (`Σ a[i]`). Same dispatch shape as
/// [`simd_f32_dot`]: wasm SIMD128 has a dedicated lane, x86/aarch select at
/// runtime, else scalar. Feeds the LayerNorm mean. Floating-point reduction
/// order is arch-dependent exactly as the dot/matmul reductions already are —
/// content-addresses are derivation-based, not a re-hash of the f32 bytes, so
/// this introduces no divergence the f32 path did not already carry.
#[inline]
pub fn simd_f32_sum(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::sum_f32(a) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::sum_f32_avx512(a) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::sum_f32_avx2(a) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::sum_f32_neon(a) },
        _ => scalar::sum_f32(a),
    }
}

/// SIMD-vectorized horizontal maximum (`max a[i]`, `−∞` for an empty slice).
/// Feeds the softmax / attention max-subtraction pass. `max` is exact and
/// order-independent for non-NaN inputs, so this is bit-identical to a scalar
/// left-fold there; a NaN element makes the result NaN under either path (the
/// SIMD `max` intrinsics and `f32::max` differ only in *which* NaN/operand
/// survives, and softmax over a NaN row is NaN regardless).
#[inline]
pub fn simd_f32_max(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::max_f32(a) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::max_f32_avx512(a) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::max_f32_avx2(a) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::max_f32_neon(a) },
        _ => scalar::max_f32(a),
    }
}

/// SIMD-vectorized horizontal minimum (`min a[i]`, `+∞` for an empty slice).
/// The mirror of [`simd_f32_max`] — exact and order-independent for non-NaN
/// inputs, so bit-identical to a scalar left-fold there.
#[inline]
pub fn simd_f32_min(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::min_f32(a) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::min_f32_avx512(a) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::min_f32_avx2(a) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::min_f32_neon(a) },
        _ => scalar::min_f32(a),
    }
}

/// Broadcast-scalar AXPY: `out[i] += s * b[i]` (one scalar `s` against a whole
/// vector `b`). The attention P·V accumulation — distinct from
/// [`simd_f32_fmadd`], which is a vector×vector FMA. Computed **non-fused**
/// (multiply then add) on every arch, so each lane replays the scalar
/// `*o += s * b[i]` bit-for-bit — there is no cross-lane reduction, so the
/// result is bit-identical to the scalar loop, not merely close. wasm SIMD128
/// (the deployed target) has a dedicated lane.
#[inline]
pub fn simd_f32_axpy(out: &mut [f32], s: f32, b: &[f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds handled inside.
    return unsafe { wasm_simd::axpy_f32(out, s, b) };
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    match resolve_path() {
        #[cfg(target_arch = "x86_64")]
        3 => unsafe { x86::axpy_f32_avx512(out, s, b) },
        #[cfg(target_arch = "x86_64")]
        2 => unsafe { x86::axpy_f32_avx2(out, s, b) },
        #[cfg(target_arch = "aarch64")]
        4 => unsafe { aarch::axpy_f32_neon(out, s, b) },
        _ => scalar::axpy_f32(out, s, b),
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
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
    pub fn sum_f32(a: &[f32]) -> f32 {
        let mut acc = 0f32;
        for &v in a {
            acc += v;
        }
        acc
    }
    pub fn max_f32(a: &[f32]) -> f32 {
        let mut m = f32::NEG_INFINITY;
        for &v in a {
            m = m.max(v);
        }
        m
    }
    pub fn min_f32(a: &[f32]) -> f32 {
        let mut m = f32::INFINITY;
        for &v in a {
            m = m.min(v);
        }
        m
    }
    pub fn axpy_f32(out: &mut [f32], s: f32, b: &[f32]) {
        let n = out.len().min(b.len());
        for i in 0..n {
            out[i] += s * b[i];
        }
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

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sum_f32_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 32;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for k in 0..chunks {
            let off = k * 32;
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(a.as_ptr().add(off)));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(a.as_ptr().add(off + 8)));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(a.as_ptr().add(off + 16)));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(a.as_ptr().add(off + 24)));
        }
        let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut buf = [0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let mut total: f32 = buf.iter().sum();
        for &v in &a[chunks * 32..n] {
            total += v;
        }
        total
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn max_f32_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let chunks = n / 8;
        let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
        for k in 0..chunks {
            m = _mm256_max_ps(m, _mm256_loadu_ps(a.as_ptr().add(k * 8)));
        }
        let mut buf = [f32::NEG_INFINITY; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), m);
        let mut total = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &v in &a[chunks * 8..n] {
            total = total.max(v);
        }
        total
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn min_f32_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::INFINITY;
        }
        let chunks = n / 8;
        let mut m = _mm256_set1_ps(f32::INFINITY);
        for k in 0..chunks {
            m = _mm256_min_ps(m, _mm256_loadu_ps(a.as_ptr().add(k * 8)));
        }
        let mut buf = [f32::INFINITY; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), m);
        let mut total = buf.iter().copied().fold(f32::INFINITY, f32::min);
        for &v in &a[chunks * 8..n] {
            total = total.min(v);
        }
        total
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_f32_avx2(out: &mut [f32], s: f32, b: &[f32]) {
        let n = out.len().min(b.len());
        let chunks = n / 8;
        let sv = _mm256_set1_ps(s);
        for k in 0..chunks {
            let off = k * 8;
            let vb = _mm256_loadu_ps(b.as_ptr().add(off));
            let vo = _mm256_loadu_ps(out.as_ptr().add(off));
            // Non-fused (mul then add) so each lane matches the scalar `out += s*b`.
            let r = _mm256_add_ps(vo, _mm256_mul_ps(sv, vb));
            _mm256_storeu_ps(out.as_mut_ptr().add(off), r);
        }
        for i in chunks * 8..n {
            out[i] += s * b[i];
        }
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

    #[target_feature(enable = "avx512f")]
    pub unsafe fn sum_f32_avx512(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 64;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        for k in 0..chunks {
            let off = k * 64;
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(a.as_ptr().add(off)));
            acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(a.as_ptr().add(off + 16)));
            acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(a.as_ptr().add(off + 32)));
            acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(a.as_ptr().add(off + 48)));
        }
        let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        let mut total = _mm512_reduce_add_ps(acc);
        for &v in &a[chunks * 64..n] {
            total += v;
        }
        total
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn max_f32_avx512(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let chunks = n / 16;
        let mut m = _mm512_set1_ps(f32::NEG_INFINITY);
        for k in 0..chunks {
            m = _mm512_max_ps(m, _mm512_loadu_ps(a.as_ptr().add(k * 16)));
        }
        let mut total = _mm512_reduce_max_ps(m);
        for &v in &a[chunks * 16..n] {
            total = total.max(v);
        }
        total
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn min_f32_avx512(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::INFINITY;
        }
        let chunks = n / 16;
        let mut m = _mm512_set1_ps(f32::INFINITY);
        for k in 0..chunks {
            m = _mm512_min_ps(m, _mm512_loadu_ps(a.as_ptr().add(k * 16)));
        }
        let mut total = _mm512_reduce_min_ps(m);
        for &v in &a[chunks * 16..n] {
            total = total.min(v);
        }
        total
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn axpy_f32_avx512(out: &mut [f32], s: f32, b: &[f32]) {
        let n = out.len().min(b.len());
        let chunks = n / 16;
        let sv = _mm512_set1_ps(s);
        for k in 0..chunks {
            let off = k * 16;
            let vb = _mm512_loadu_ps(b.as_ptr().add(off));
            let vo = _mm512_loadu_ps(out.as_ptr().add(off));
            let r = _mm512_add_ps(vo, _mm512_mul_ps(sv, vb));
            _mm512_storeu_ps(out.as_mut_ptr().add(off), r);
        }
        for i in chunks * 16..n {
            out[i] += s * b[i];
        }
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
            // Column remainder for this row block. These tiers (8-wide FMA,
            // then scalar) must stay **identical** to the row-remainder block's
            // tiers below. An output cell's arithmetic may depend on its column
            // — never on its row index. `_mm256_fmadd_ps` rounds once where the
            // scalar `s += a*b` rounds twice, so a tile row and a remainder row
            // that disagreed on the tier for column `j` would compute different
            // bytes from identical inputs. f32 result bytes are
            // content-addressed, so that is a κ hazard, not a rounding nicety.
            // Pinned by `matmul_row_bytes_are_independent_of_row_index`.
            while j + 8 <= n {
                let mut c = [_mm256_setzero_ps(); MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = _mm256_loadu_ps(out.add((i + r) * ldc + j));
                    }
                }
                for kk in 0..k {
                    let bv = _mm256_loadu_ps(b.add(kk * ldb + j));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = _mm256_set1_ps(*a.add((i + r) * lda + kk));
                        *cr = _mm256_fmadd_ps(av, bv, *cr);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    _mm256_storeu_ps(out.add((i + r) * ldc + j), *cr);
                }
                j += 8;
            }
            while j < n {
                let mut s = [0f32; MR];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
            i += MR;
        }
        #[target_feature(enable = "avx2,fma")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
            accumulate: bool,
        ) {
            let mut j = 0;
            // Wide-column tiers for the low-`R` (decode) monomorphizations. The
            // 16-column loop below keeps only `2·R` accumulators in flight, so
            // at `R = 1` it is **FMA-latency** bound, not bandwidth bound —
            // measured 0.433 ms vs 0.238 ms for the 8-accumulator `MR` tile at
            // k = n = 1024, monotonic in accumulator count. Widening the column
            // block adds independent chains.
            //
            // This cannot change any byte: cells are independent across `j`, and
            // every tier here is `fmadd`, so the fused-vs-scalar column set is
            // still exactly `j < 8·⌊n/8⌋` — which is what row-index
            // independence requires (see the MR tile's matching tiers).
            if R == 1 {
                while j + 64 <= n {
                    let mut c = [_mm256_setzero_ps(); 8];
                    if accumulate {
                        for (g, cg) in c.iter_mut().enumerate() {
                            *cg = _mm256_loadu_ps(out.add(i * ldc + j + g * 8));
                        }
                    }
                    for kk in 0..k {
                        let av = _mm256_set1_ps(*a.add(i * lda + kk));
                        let brow = b.add(kk * ldb + j);
                        for (g, cg) in c.iter_mut().enumerate() {
                            *cg = _mm256_fmadd_ps(av, _mm256_loadu_ps(brow.add(g * 8)), *cg);
                        }
                    }
                    for (g, cg) in c.iter().enumerate() {
                        _mm256_storeu_ps(out.add(i * ldc + j + g * 8), *cg);
                    }
                    j += 64;
                }
            }
            if R == 2 {
                while j + 32 <= n {
                    let mut c = [[_mm256_setzero_ps(); 4]; R];
                    if accumulate {
                        for (r, cr) in c.iter_mut().enumerate() {
                            let orow = out.add((i + r) * ldc + j);
                            for (g, cg) in cr.iter_mut().enumerate() {
                                *cg = _mm256_loadu_ps(orow.add(g * 8));
                            }
                        }
                    }
                    for kk in 0..k {
                        let brow = b.add(kk * ldb + j);
                        let bv = [
                            _mm256_loadu_ps(brow),
                            _mm256_loadu_ps(brow.add(8)),
                            _mm256_loadu_ps(brow.add(16)),
                            _mm256_loadu_ps(brow.add(24)),
                        ];
                        for (r, cr) in c.iter_mut().enumerate() {
                            let av = _mm256_set1_ps(*a.add((i + r) * lda + kk));
                            for (g, cg) in cr.iter_mut().enumerate() {
                                *cg = _mm256_fmadd_ps(av, bv[g], *cg);
                            }
                        }
                    }
                    for (r, cr) in c.iter().enumerate() {
                        let orow = out.add((i + r) * ldc + j);
                        for (g, cg) in cr.iter().enumerate() {
                            _mm256_storeu_ps(orow.add(g * 8), *cg);
                        }
                    }
                    j += 32;
                }
            }
            while j + 16 <= n {
                let mut c = [[_mm256_setzero_ps(); 2]; R];
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
                j += 16;
            }
            while j + 8 <= n {
                let mut c = [_mm256_setzero_ps(); R];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = _mm256_loadu_ps(out.add((i + r) * ldc + j));
                    }
                }
                for kk in 0..k {
                    let bv = _mm256_loadu_ps(b.add(kk * ldb + j));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = _mm256_set1_ps(*a.add((i + r) * lda + kk));
                        *cr = _mm256_fmadd_ps(av, bv, *cr);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    _mm256_storeu_ps(out.add((i + r) * ldc + j), *cr);
                }
                j += 8;
            }
            while j < n {
                let mut s = [0f32; R];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the M=1 /
        // small-M **decode (GEMV)** shape.
        //
        // The leftover rows share **one** pass over B. The MR tile above
        // amortizes each B load across 4 rows; doing the remainder a row at a
        // time re-streams the whole `k×n` weight per row, and B — not the
        // arithmetic — sets the time here.
        //
        // `R` is a **const** so the row loops unroll and `R = 1` monomorphizes
        // back into exactly the specialized single-row GEMV. A runtime `rem`
        // bound (`.take(rem)`) leaves the row loop rolled, which costs the
        // single-threaded wasm decode path ~40% at `m = 1` — the one shape that
        // matters most — even while it speeds `m = 2..3`.
        //
        // Every output cell still accumulates over `kk` ascending through the
        // same chain as the MR tile, so results are **bit-identical** to the
        // per-row form. That is required, not incidental: f32 result bytes are
        // content-addressed, so reassociating the reduction would re-key every κ
        // that depends on it.
        match m - i {
            1 => rem_rows::<1>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            2 => rem_rows::<2>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            3 => rem_rows::<3>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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
        #[target_feature(enable = "avx2,fma")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            bpacked: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldc: usize,
            k_stride: usize,
            accumulate: bool,
        ) {
            // Wide-panel tier for the `R = 1` (decode) monomorphization: one
            // panel leaves only 2 accumulators in flight and the loop
            // becomes multiply-add-latency bound rather than bandwidth bound.
            // Running four panels together puts 8 independent chains in flight.
            //
            // Byte-neutral: cells are independent across panels and this tier
            // uses the same multiply-add as the single-panel tier below, so no
            // cell's `kk` chain changes.
            let n_full = n / 16;
            let mut p0 = 0;
            if R == 1 {
                while p0 + 4 <= n_full {
                    let mut c = [_mm256_setzero_ps(); 8];
                    if accumulate {
                        for (g, cg) in c.iter_mut().enumerate() {
                            *cg = _mm256_loadu_ps(out.add(i * ldc + p0 * 16 + g * 8));
                        }
                    }
                    for kk in 0..k {
                        let av = _mm256_set1_ps(*a.add(i * lda + kk));
                        for q in 0..4 {
                            let bp = bpacked.add((p0 + q) * k_stride * 16 + kk * 16);
                            c[2 * q] = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c[2 * q]);
                            c[2 * q + 1] =
                                _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c[2 * q + 1]);
                        }
                    }
                    for (g, cg) in c.iter().enumerate() {
                        _mm256_storeu_ps(out.add(i * ldc + p0 * 16 + g * 8), *cg);
                    }
                    p0 += 4;
                }
            }
            for p in p0..n.div_ceil(16) {
                let cols = core::cmp::min(16, n - p * 16);
                let base = p * k_stride * 16;
                if cols == 16 {
                    let mut c = [[_mm256_setzero_ps(); 2]; R];
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
                    // Partial trailing panel (< 16 cols): scalar, same k order.
                    for r in 0..R {
                        let arow = a.add((i + r) * lda);
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *arow.add(kk) * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the decode
        // (GEMV) shape, packed-panel form. The leftover rows share **one** pass
        // over the packed weight rather than re-streaming it per row, and `R` is
        // a const so `R = 1` monomorphizes back to the specialized single-row
        // GEMV. Each cell keeps its `kk`-ascending chain, so the result is
        // bit-identical to the per-row form.
        match m - i {
            1 => rem_rows::<1>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            2 => rem_rows::<2>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            3 => rem_rows::<3>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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
    #[target_feature(enable = "neon")]
    pub unsafe fn sum_f32_neon(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 16;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        for k in 0..chunks {
            let off = k * 16;
            acc0 = vaddq_f32(acc0, vld1q_f32(a.as_ptr().add(off)));
            acc1 = vaddq_f32(acc1, vld1q_f32(a.as_ptr().add(off + 4)));
            acc2 = vaddq_f32(acc2, vld1q_f32(a.as_ptr().add(off + 8)));
            acc3 = vaddq_f32(acc3, vld1q_f32(a.as_ptr().add(off + 12)));
        }
        let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let lanes = [
            vgetq_lane_f32(acc, 0),
            vgetq_lane_f32(acc, 1),
            vgetq_lane_f32(acc, 2),
            vgetq_lane_f32(acc, 3),
        ];
        let mut total: f32 = lanes.iter().sum();
        for &v in &a[chunks * 16..n] {
            total += v;
        }
        total
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn max_f32_neon(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let chunks = n / 4;
        let mut m = vdupq_n_f32(f32::NEG_INFINITY);
        for k in 0..chunks {
            m = vmaxq_f32(m, vld1q_f32(a.as_ptr().add(k * 4)));
        }
        let mut total = vmaxvq_f32(m);
        for &v in &a[chunks * 4..n] {
            total = total.max(v);
        }
        total
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn min_f32_neon(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::INFINITY;
        }
        let chunks = n / 4;
        let mut m = vdupq_n_f32(f32::INFINITY);
        for k in 0..chunks {
            m = vminq_f32(m, vld1q_f32(a.as_ptr().add(k * 4)));
        }
        let mut total = vminvq_f32(m);
        for &v in &a[chunks * 4..n] {
            total = total.min(v);
        }
        total
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn axpy_f32_neon(out: &mut [f32], s: f32, b: &[f32]) {
        let n = out.len().min(b.len());
        let chunks = n / 4;
        let sv = vdupq_n_f32(s);
        for k in 0..chunks {
            let off = k * 4;
            let vb = vld1q_f32(b.as_ptr().add(off));
            let vo = vld1q_f32(out.as_ptr().add(off));
            // Non-fused (mul then add) to match the scalar `out += s*b` bitwise.
            let r = vaddq_f32(vo, vmulq_f32(sv, vb));
            vst1q_f32(out.as_mut_ptr().add(off), r);
        }
        for i in chunks * 4..n {
            out[i] += s * b[i];
        }
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
            // These tiers (4-wide FMA, then scalar) must stay **identical** to
            // the row-remainder block's tiers below: an output cell's
            // arithmetic may depend on its column, never on its row index.
            // `vfmaq_f32` rounds once where the scalar `s += a*b` rounds twice,
            // so a mismatch here makes two identical input rows produce
            // different bytes depending on whether they land in the MR tile or
            // the remainder — a κ hazard, since f32 result bytes are
            // content-addressed. Pinned by
            // `matmul_row_bytes_are_independent_of_row_index`.
            while j + 4 <= n {
                let mut c = [vdupq_n_f32(0.0); MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = vld1q_f32(out.add((i + r) * ldc + j));
                    }
                }
                for kk in 0..k {
                    let bv = vld1q_f32(b.add(kk * ldb + j));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = vdupq_n_f32(*a.add((i + r) * lda + kk));
                        *cr = vfmaq_f32(*cr, av, bv);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    vst1q_f32(out.add((i + r) * ldc + j), *cr);
                }
                j += 4;
            }
            while j < n {
                let mut s = [0f32; MR];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
            i += MR;
        }
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
            accumulate: bool,
        ) {
            let mut j = 0;
            // Wide-column tier for the `R = 1` (decode) monomorphization. The
            // 16-column loop below keeps only `4·R` accumulators in flight, so
            // at `R = 1` it is multiply-add-latency bound rather than bandwidth
            // bound. Widening to 32 columns puts 8 independent chains in flight.
            //
            // This cannot change any byte: cells are independent across `j`, and
            // this tier uses the same `vfmaq_f32` as the 16- and 4-column tiers, so
            // the vector-vs-scalar column set is still exactly `j < 4·⌊n/4⌋` —
            // which is what row-index independence requires.
            if R == 1 {
                while j + 32 <= n {
                    let mut c = [vdupq_n_f32(0.0); 8];
                    if accumulate {
                        for (g, cg) in c.iter_mut().enumerate() {
                            *cg = vld1q_f32(out.add(i * ldc + j + g * 4));
                        }
                    }
                    for kk in 0..k {
                        let av = vdupq_n_f32(*a.add(i * lda + kk));
                        let brow = b.add(kk * ldb + j);
                        for (g, cg) in c.iter_mut().enumerate() {
                            let bv = vld1q_f32(brow.add(g * 4));
                            *cg = vfmaq_f32(*cg, av, bv);
                        }
                    }
                    for (g, cg) in c.iter().enumerate() {
                        vst1q_f32(out.add(i * ldc + j + g * 4), *cg);
                    }
                    j += 32;
                }
            }
            while j + 16 <= n {
                let mut c = [[vdupq_n_f32(0.0); 4]; R];
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
                j += 16;
            }
            while j + 4 <= n {
                let mut c = [vdupq_n_f32(0.0); R];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = vld1q_f32(out.add((i + r) * ldc + j));
                    }
                }
                for kk in 0..k {
                    let bv = vld1q_f32(b.add(kk * ldb + j));
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = vdupq_n_f32(*a.add((i + r) * lda + kk));
                        *cr = vfmaq_f32(*cr, av, bv);
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    vst1q_f32(out.add((i + r) * ldc + j), *cr);
                }
                j += 4;
            }
            while j < n {
                let mut s = [0f32; R];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the M=1 /
        // small-M **decode (GEMV)** shape.
        //
        // The leftover rows share **one** pass over B. The MR tile above
        // amortizes each B load across 4 rows; doing the remainder a row at a
        // time re-streams the whole `k×n` weight per row, and B — not the
        // arithmetic — sets the time here.
        //
        // `R` is a **const** so the row loops unroll and `R = 1` monomorphizes
        // back into exactly the specialized single-row GEMV. A runtime `rem`
        // bound (`.take(rem)`) leaves the row loop rolled, which measured ~40%
        // slower at `m = 1` on single-threaded wasm — the decode shape that
        // matters most — even while it sped up `m = 2..3`.
        //
        // Every output cell still accumulates over `kk` ascending through the
        // same chain as the MR tile, so results are **bit-identical** to the
        // per-row form. Required, not incidental: f32 result bytes are
        // content-addressed, so reassociating the reduction re-keys every κ.
        match m - i {
            1 => rem_rows::<1>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            2 => rem_rows::<2>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            3 => rem_rows::<3>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            bpacked: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldc: usize,
            k_stride: usize,
            accumulate: bool,
        ) {
            for p in 0..n.div_ceil(16) {
                let cols = core::cmp::min(16, n - p * 16);
                let base = p * k_stride * 16;
                if cols == 16 {
                    let mut c = [[vdupq_n_f32(0.0); 4]; R];
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
                    // Partial trailing panel (< 16 cols): scalar, same k order.
                    for r in 0..R {
                        let arow = a.add((i + r) * lda);
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *arow.add(kk) * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the decode
        // (GEMV) shape, packed-panel form. The leftover rows share **one** pass
        // over the packed weight rather than re-streaming it per row, and `R` is
        // a const so `R = 1` monomorphizes back to the specialized single-row
        // GEMV. Each cell keeps its `kk`-ascending chain, so the result is
        // bit-identical to the per-row form.
        match m - i {
            1 => rem_rows::<1>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            2 => rem_rows::<2>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            3 => rem_rows::<3>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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

    /// wasm SIMD128 f32 dot product — the deployed-target twin of
    /// `x86::dot_f32_avx2` / `aarch::dot_f32_neon`. Four independent 4-lane
    /// accumulators hide the add latency; the horizontal reduction and scalar
    /// tail keep it correct for any length. (Floating-point reduction order is
    /// arch-dependent here as it already is for every f32 dot/matmul — the
    /// content-address of a value is derivation-based, not a re-hash of its
    /// bytes, so this introduces no cross-target divergence the f32 path did
    /// not already have.)
    ///
    /// # Safety
    /// simd128 must be enabled; `v128_load` is an unaligned wasm load.
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / 16;
        let (mut c0, mut c1, mut c2, mut c3) = (
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
        );
        for k in 0..chunks {
            let off = k * 16;
            let ap = a.as_ptr().add(off);
            let bp = b.as_ptr().add(off);
            c0 = f32x4_add(
                c0,
                f32x4_mul(v128_load(ap as *const v128), v128_load(bp as *const v128)),
            );
            c1 = f32x4_add(
                c1,
                f32x4_mul(
                    v128_load(ap.add(4) as *const v128),
                    v128_load(bp.add(4) as *const v128),
                ),
            );
            c2 = f32x4_add(
                c2,
                f32x4_mul(
                    v128_load(ap.add(8) as *const v128),
                    v128_load(bp.add(8) as *const v128),
                ),
            );
            c3 = f32x4_add(
                c3,
                f32x4_mul(
                    v128_load(ap.add(12) as *const v128),
                    v128_load(bp.add(12) as *const v128),
                ),
            );
        }
        let acc = f32x4_add(f32x4_add(c0, c1), f32x4_add(c2, c3));
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        for i in chunks * 16..n {
            sum += *a.as_ptr().add(i) * *b.as_ptr().add(i);
        }
        sum
    }

    /// wasm SIMD128 horizontal sum — the deployed-target twin of
    /// `x86::sum_f32_avx2` / `aarch::sum_f32_neon`. Four 4-lane accumulators,
    /// horizontal reduce, scalar tail.
    ///
    /// # Safety
    /// simd128 enabled; `v128_load` is an unaligned wasm load.
    pub unsafe fn sum_f32(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 16;
        let (mut c0, mut c1, mut c2, mut c3) = (
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
            f32x4_splat(0.0),
        );
        for k in 0..chunks {
            let off = k * 16;
            let ap = a.as_ptr().add(off);
            c0 = f32x4_add(c0, v128_load(ap as *const v128));
            c1 = f32x4_add(c1, v128_load(ap.add(4) as *const v128));
            c2 = f32x4_add(c2, v128_load(ap.add(8) as *const v128));
            c3 = f32x4_add(c3, v128_load(ap.add(12) as *const v128));
        }
        let acc = f32x4_add(f32x4_add(c0, c1), f32x4_add(c2, c3));
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        for i in chunks * 16..n {
            sum += *a.as_ptr().add(i);
        }
        sum
    }

    /// wasm SIMD128 horizontal maximum (`−∞` for empty). `f32x4_pmax` picks the
    /// second operand on ties/NaN — immaterial for softmax (a NaN row is NaN
    /// either way), and exact for the ordinary finite case.
    ///
    /// # Safety
    /// simd128 enabled; `v128_load` is an unaligned wasm load.
    pub unsafe fn max_f32(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::NEG_INFINITY;
        }
        let chunks = n / 4;
        let mut m = f32x4_splat(f32::NEG_INFINITY);
        for k in 0..chunks {
            m = f32x4_pmax(m, v128_load(a.as_ptr().add(k * 4) as *const v128));
        }
        let mut total = f32x4_extract_lane::<0>(m)
            .max(f32x4_extract_lane::<1>(m))
            .max(f32x4_extract_lane::<2>(m))
            .max(f32x4_extract_lane::<3>(m));
        for i in chunks * 4..n {
            total = total.max(*a.as_ptr().add(i));
        }
        total
    }

    /// wasm SIMD128 horizontal minimum (`+∞` for empty). Mirror of [`max_f32`].
    ///
    /// # Safety
    /// simd128 enabled; `v128_load` is an unaligned wasm load.
    pub unsafe fn min_f32(a: &[f32]) -> f32 {
        let n = a.len();
        if n == 0 {
            return f32::INFINITY;
        }
        let chunks = n / 4;
        let mut m = f32x4_splat(f32::INFINITY);
        for k in 0..chunks {
            m = f32x4_pmin(m, v128_load(a.as_ptr().add(k * 4) as *const v128));
        }
        let mut total = f32x4_extract_lane::<0>(m)
            .min(f32x4_extract_lane::<1>(m))
            .min(f32x4_extract_lane::<2>(m))
            .min(f32x4_extract_lane::<3>(m));
        for i in chunks * 4..n {
            total = total.min(*a.as_ptr().add(i));
        }
        total
    }

    /// wasm SIMD128 broadcast-scalar AXPY (`out[i] += s * b[i]`), non-fused so
    /// each lane matches the scalar loop bit-for-bit. Deliberately mul-then-add
    /// rather than `f32x4_relaxed_madd` — the fused relaxed op both changes the
    /// bits and, as measured elsewhere in this module, regresses latency-bound
    /// chains on the wasmtime host.
    ///
    /// # Safety
    /// simd128 enabled; `v128_load`/`v128_store` are unaligned wasm accesses.
    pub unsafe fn axpy_f32(out: &mut [f32], s: f32, b: &[f32]) {
        let n = out.len().min(b.len());
        let chunks = n / 4;
        let sv = f32x4_splat(s);
        for k in 0..chunks {
            let off = k * 4;
            let vb = v128_load(b.as_ptr().add(off) as *const v128);
            let vo = v128_load(out.as_ptr().add(off) as *const v128);
            let r = f32x4_add(vo, f32x4_mul(sv, vb));
            v128_store(out.as_mut_ptr().add(off) as *mut v128, r);
        }
        for i in chunks * 4..n {
            out[i] += s * b[i];
        }
    }

    /// wasm SIMD128 elementwise `out = a + b` (exact — bit-identical to scalar).
    ///
    /// # Safety
    /// simd128 enabled; `v128_load`/`v128_store` are unaligned wasm accesses.
    pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let off = k * 4;
            let va = v128_load(a.as_ptr().add(off) as *const v128);
            let vb = v128_load(b.as_ptr().add(off) as *const v128);
            v128_store(out.as_mut_ptr().add(off) as *mut v128, f32x4_add(va, vb));
        }
        for i in chunks * 4..n {
            out[i] = a[i] + b[i];
        }
    }

    /// wasm SIMD128 elementwise `out = a * b` (exact — bit-identical to scalar).
    ///
    /// # Safety
    /// simd128 enabled; `v128_load`/`v128_store` are unaligned wasm accesses.
    pub unsafe fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let off = k * 4;
            let va = v128_load(a.as_ptr().add(off) as *const v128);
            let vb = v128_load(b.as_ptr().add(off) as *const v128);
            v128_store(out.as_mut_ptr().add(off) as *mut v128, f32x4_mul(va, vb));
        }
        for i in chunks * 4..n {
            out[i] = a[i] * b[i];
        }
    }

    /// wasm SIMD128 `out[i] += a[i] * b[i]`, non-fused (mul then add) — SIMD128
    /// has no FMA. The scalar tail uses the same non-fused form so the whole
    /// call is internally consistent.
    ///
    /// # Safety
    /// simd128 enabled; `v128_load`/`v128_store` are unaligned wasm accesses.
    pub unsafe fn fmadd_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        let n = a.len().min(b.len()).min(out.len());
        let chunks = n / 4;
        for k in 0..chunks {
            let off = k * 4;
            let va = v128_load(a.as_ptr().add(off) as *const v128);
            let vb = v128_load(b.as_ptr().add(off) as *const v128);
            let vo = v128_load(out.as_ptr().add(off) as *const v128);
            let r = f32x4_add(vo, f32x4_mul(va, vb));
            v128_store(out.as_mut_ptr().add(off) as *mut v128, r);
        }
        for i in chunks * 4..n {
            out[i] += a[i] * b[i];
        }
    }

    // ─── wasm SIMD128 register-tiled f32 matmul ────────────────────
    // The wasm32 analog of the NEON `aarch` matmul kernels. wasm SIMD128 is
    // 128-bit / 4×f32 — the same width as NEON — so this mirrors the 4×16 tile
    // structure exactly, with `f32x4_*` lanes. SIMD128 has no fused
    // multiply-add (that's relaxed-SIMD), so the inner step is a separate
    // `f32x4_add(acc, f32x4_mul(av, b))`; still 4-wide, a large win over the
    // scalar fallback. `f32x4_relaxed_madd` was measured here (wasmtime,
    // x86-64 FMA host) and REGRESSED these kernels ~30% — the accumulator
    // chains are latency-bound and the fused op lengthens the chain — so the
    // relaxed-SIMD tier deliberately covers only the integer i8 dot (see
    // `gemv_i8_omajor_wasm_relaxed`), where it is exact and measured faster. The cache-oblivious recursion is identical to the other
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
            // Column remainder for this row block, tiered to match the
            // row-remainder block below (4-wide, then scalar) so an output
            // cell's arithmetic depends on its column and never on its row
            // index. On SIMD128 this tier is numerically inert — there is no
            // fused multiply-add, so `f32x4_add(c, f32x4_mul(av, b))` and the
            // scalar `s += a*b` round identically — but keeping the three
            // architectures structurally identical is what makes the shared
            // `matmul_row_bytes_are_independent_of_row_index` pin meaningful,
            // and it vectorizes the tail.
            while j + 4 <= n {
                let mut c = [f32x4_splat(0.0); MR];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = v128_load(out.add((i + r) * ldc + j) as *const v128);
                    }
                }
                for kk in 0..k {
                    let bv = v128_load(b.add(kk * ldb + j) as *const v128);
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = f32x4_splat(*a.add((i + r) * lda + kk));
                        *cr = f32x4_add(*cr, f32x4_mul(av, bv));
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    v128_store(out.add((i + r) * ldc + j) as *mut v128, *cr);
                }
                j += 4;
            }
            while j < n {
                let mut s = [0f32; MR];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
            i += MR;
        }
        #[target_feature(enable = "simd128")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            b: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
            accumulate: bool,
        ) {
            let mut j = 0;
            // Wide-column tier for the `R = 1` (decode) monomorphization. The
            // 16-column loop below keeps only `4·R` accumulators in flight, so
            // at `R = 1` it is multiply-add-latency bound rather than bandwidth
            // bound. Widening to 32 columns puts 8 independent chains in flight.
            //
            // This cannot change any byte: cells are independent across `j`, and
            // this tier uses the same `add(mul(..))` as the 16- and 4-column tiers, so
            // the vector-vs-scalar column set is still exactly `j < 4·⌊n/4⌋` —
            // which is what row-index independence requires.
            if R == 1 {
                while j + 32 <= n {
                    let mut c = [f32x4_splat(0.0); 8];
                    if accumulate {
                        for (g, cg) in c.iter_mut().enumerate() {
                            *cg = v128_load(out.add(i * ldc + j + g * 4) as *const v128);
                        }
                    }
                    for kk in 0..k {
                        let av = f32x4_splat(*a.add(i * lda + kk));
                        let brow = b.add(kk * ldb + j);
                        for (g, cg) in c.iter_mut().enumerate() {
                            let bv = v128_load(brow.add(g * 4) as *const v128);
                            *cg = f32x4_add(*cg, f32x4_mul(av, bv));
                        }
                    }
                    for (g, cg) in c.iter().enumerate() {
                        v128_store(out.add(i * ldc + j + g * 4) as *mut v128, *cg);
                    }
                    j += 32;
                }
            }
            while j + 16 <= n {
                let mut c = [[f32x4_splat(0.0); 4]; R];
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
                j += 16;
            }
            while j + 4 <= n {
                let mut c = [f32x4_splat(0.0); R];
                if accumulate {
                    for (r, cr) in c.iter_mut().enumerate() {
                        *cr = v128_load(out.add((i + r) * ldc + j) as *const v128);
                    }
                }
                for kk in 0..k {
                    let bv = v128_load(b.add(kk * ldb + j) as *const v128);
                    for (r, cr) in c.iter_mut().enumerate() {
                        let av = f32x4_splat(*a.add((i + r) * lda + kk));
                        *cr = f32x4_add(*cr, f32x4_mul(av, bv));
                    }
                }
                for (r, cr) in c.iter().enumerate() {
                    v128_store(out.add((i + r) * ldc + j) as *mut v128, *cr);
                }
                j += 4;
            }
            while j < n {
                let mut s = [0f32; R];
                for (r, sr) in s.iter_mut().enumerate() {
                    *sr = if accumulate {
                        *out.add((i + r) * ldc + j)
                    } else {
                        0.0
                    };
                }
                for kk in 0..k {
                    let bv = *b.add(kk * ldb + j);
                    for (r, sr) in s.iter_mut().enumerate() {
                        *sr += *a.add((i + r) * lda + kk) * bv;
                    }
                }
                for (r, sr) in s.iter().enumerate() {
                    *out.add((i + r) * ldc + j) = *sr;
                }
                j += 1;
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the M=1 /
        // small-M **decode (GEMV)** shape.
        //
        // The leftover rows share **one** pass over B. The MR tile above
        // amortizes each B load across 4 rows; doing the remainder a row at a
        // time re-streams the whole `k×n` weight per row, and B — not the
        // arithmetic — sets the time here.
        //
        // `R` is a **const** so the row loops unroll and `R = 1` monomorphizes
        // back into exactly the specialized single-row GEMV. A runtime `rem`
        // bound (`.take(rem)`) leaves the row loop rolled, which measured ~40%
        // slower at `m = 1` on single-threaded wasm — the decode shape that
        // matters most — even while it sped up `m = 2..3`.
        //
        // Every output cell still accumulates over `kk` ascending through the
        // same chain as the MR tile, so results are **bit-identical** to the
        // per-row form. Required, not incidental: f32 result bytes are
        // content-addressed, so reassociating the reduction re-keys every κ.
        match m - i {
            1 => rem_rows::<1>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            2 => rem_rows::<2>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            3 => rem_rows::<3>(a, b, out, i, k, n, lda, ldb, ldc, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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
        #[target_feature(enable = "simd128")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn rem_rows<const R: usize>(
            a: *const f32,
            bpacked: *const f32,
            out: *mut f32,
            i: usize,
            k: usize,
            n: usize,
            lda: usize,
            ldc: usize,
            k_stride: usize,
            accumulate: bool,
        ) {
            for p in 0..n.div_ceil(16) {
                let cols = core::cmp::min(16, n - p * 16);
                let base = p * k_stride * 16;
                if cols == 16 {
                    let mut c = [[f32x4_splat(0.0); 4]; R];
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
                    // Partial trailing panel (< 16 cols): scalar, same k order.
                    for r in 0..R {
                        let arow = a.add((i + r) * lda);
                        for cc in 0..cols {
                            let j = p * 16 + cc;
                            let mut s = if accumulate {
                                *out.add((i + r) * ldc + j)
                            } else {
                                0.0
                            };
                            for kk in 0..k {
                                s += *arow.add(kk) * *bpacked.add((p * k_stride + kk) * 16 + cc);
                            }
                            *out.add((i + r) * ldc + j) = s;
                        }
                    }
                }
            }
        }

        // Row remainder (`m` not a multiple of MR), 1..=3 rows — the decode
        // (GEMV) shape, packed-panel form. The leftover rows share **one** pass
        // over the packed weight rather than re-streaming it per row, and `R` is
        // a const so `R = 1` monomorphizes back to the specialized single-row
        // GEMV. Each cell keeps its `kk`-ascending chain, so the result is
        // bit-identical to the per-row form.
        match m - i {
            1 => rem_rows::<1>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            2 => rem_rows::<2>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            3 => rem_rows::<3>(a, bpacked, out, i, k, n, lda, ldc, k_stride, accumulate),
            r => debug_assert_eq!(r, 0, "MR = 4 leaves at most 3 remainder rows"),
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
#[allow(clippy::needless_return)] // cfg-gated arch dispatch
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
#[allow(clippy::needless_return)] // cfg-gated arch dispatch
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
        // `resize` alone zero-fills only the *growth*; a preceding `clear()`
        // would force `k·n` zero stores on every call — 4 MB for a 1024²
        // panel — that the transpose below immediately overwrites in full.
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

// ─── Fused per-channel symmetric int8 matmul (SPIKE) ───────────────
// `out[i][j] = scale[j] · Σ_k a[i][k] · (f32)bq[k][j]` (zero-point 0).
// Reads the i8 weight directly and dequantizes each 16-wide column tile in
// registers; the per-column scale factors OUT of the k-loop to the writeback,
// so the dense f32 weight is never materialized (unlike `matmul_dequant`'s
// dequant-to-f32-scratch path). aarch64 NEON + portable scalar; this is a spike
// to measure whether the fused int path beats dequant-then-matmul.

/// NEON inner: GEMV-style over output columns, 16 wide. `a` `[m,k]` row-major,
/// `bq` `[k,n]` row-major i8, `scales` `[n]`, `out` `[m,n]`.
///
/// # Safety
/// NEON (baseline aarch64); slices sized `m*k`, `k*n`, `n`, `m*n`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matmul_i8_pc_neon(
    a: *const f32,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use core::arch::aarch64::*;
    for i in 0..m {
        let arow = a.add(i * k);
        let orow = out.add(i * n);
        let mut j = 0;
        while j + 16 <= n {
            let (mut c0, mut c1, mut c2, mut c3) = (
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
                vdupq_n_f32(0.0),
            );
            for kk in 0..k {
                let av = vdupq_n_f32(*arow.add(kk));
                // 16 i8 weights for this k-row, this column panel.
                let q = vld1q_s8(bq.add(kk * n + j));
                let lo = vmovl_s8(vget_low_s8(q)); // i16x8 (cols 0..8)
                let hi = vmovl_s8(vget_high_s8(q)); // i16x8 (cols 8..16)
                let b0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
                let b1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
                let b2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
                let b3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
                c0 = vfmaq_f32(c0, av, b0);
                c1 = vfmaq_f32(c1, av, b1);
                c2 = vfmaq_f32(c2, av, b2);
                c3 = vfmaq_f32(c3, av, b3);
            }
            // Apply the per-column scale once, at writeback.
            vst1q_f32(orow.add(j), vmulq_f32(c0, vld1q_f32(scales.add(j))));
            vst1q_f32(orow.add(j + 4), vmulq_f32(c1, vld1q_f32(scales.add(j + 4))));
            vst1q_f32(orow.add(j + 8), vmulq_f32(c2, vld1q_f32(scales.add(j + 8))));
            vst1q_f32(
                orow.add(j + 12),
                vmulq_f32(c3, vld1q_f32(scales.add(j + 12))),
            );
            j += 16;
        }
        while j < n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += *arow.add(kk) * (*bq.add(kk * n + j) as f32);
            }
            *orow.add(j) = acc * *scales.add(j);
            j += 1;
        }
    }
}

/// wasm SIMD128 inner for the fused per-channel int8 matmul — the wasm twin of
/// `matmul_i8_pc_neon`. SIMD128 has no FMA (mul+add), and widens i8→f32 via the
/// extend ladder. This is the primary-target (browser) decode kernel.
///
/// # Safety
/// simd128 enabled; slices sized `m*k`, `k*n`, `n`, `m*n`. `v128_load`/`store`
/// are unaligned wasm loads (no alignment fault).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn matmul_i8_pc_wasm(
    a: *const f32,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use core::arch::wasm32::*;
    for i in 0..m {
        let arow = a.add(i * k);
        let orow = out.add(i * n);
        let mut j = 0;
        while j + 16 <= n {
            let (mut c0, mut c1, mut c2, mut c3) = (
                f32x4_splat(0.0),
                f32x4_splat(0.0),
                f32x4_splat(0.0),
                f32x4_splat(0.0),
            );
            for kk in 0..k {
                let av = f32x4_splat(*arow.add(kk));
                let q = v128_load(bq.add(kk * n + j) as *const v128);
                let lo = i16x8_extend_low_i8x16(q);
                let hi = i16x8_extend_high_i8x16(q);
                let b0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo));
                let b1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo));
                let b2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi));
                let b3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi));
                c0 = f32x4_add(c0, f32x4_mul(av, b0));
                c1 = f32x4_add(c1, f32x4_mul(av, b1));
                c2 = f32x4_add(c2, f32x4_mul(av, b2));
                c3 = f32x4_add(c3, f32x4_mul(av, b3));
            }
            v128_store(
                orow.add(j) as *mut v128,
                f32x4_mul(c0, v128_load(scales.add(j) as *const v128)),
            );
            v128_store(
                orow.add(j + 4) as *mut v128,
                f32x4_mul(c1, v128_load(scales.add(j + 4) as *const v128)),
            );
            v128_store(
                orow.add(j + 8) as *mut v128,
                f32x4_mul(c2, v128_load(scales.add(j + 8) as *const v128)),
            );
            v128_store(
                orow.add(j + 12) as *mut v128,
                f32x4_mul(c3, v128_load(scales.add(j + 12) as *const v128)),
            );
            j += 16;
        }
        while j < n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += *arow.add(kk) * (*bq.add(kk * n + j) as f32);
            }
            *orow.add(j) = acc * *scales.add(j);
            j += 1;
        }
    }
}

/// x86-64 AVX2 inner for the fused per-channel int8 matmul — the x86 twin of
/// `matmul_i8_pc_neon` (a first-class target must not fall to the scalar triple
/// loop). GEMV over 16-column panels: two YMM accumulators, the i8 weights
/// sign-extended to f32 in-register (`_mm256_cvtepi8_epi32`), the per-column
/// scale factored out to the writeback. FMA — bit-close (rel < 1e-4) to the
/// naive reference, as the NEON lane already is.
///
/// # Safety
/// AVX2 + FMA (the caller gates on `x86_has_avx2`); slices sized `m*k`, `k*n`,
/// `n`, `m*n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_i8_pc_avx2(
    a: *const f32,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use core::arch::x86_64::*;
    for i in 0..m {
        let arow = a.add(i * k);
        let orow = out.add(i * n);
        let mut j = 0;
        // 16-column panels: two 8-wide YMM accumulators.
        while j + 16 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            for kk in 0..k {
                let av = _mm256_set1_ps(*arow.add(kk));
                // 16 i8 weights for this k-row / column panel.
                let q = _mm_loadu_si128(bq.add(kk * n + j) as *const __m128i);
                let b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)); // cols j..j+8
                let qhi = _mm_srli_si128(q, 8); // high 8 i8 → low
                let b1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qhi)); // cols j+8..j+16
                c0 = _mm256_fmadd_ps(av, b0, c0);
                c1 = _mm256_fmadd_ps(av, b1, c1);
            }
            _mm256_storeu_ps(
                orow.add(j),
                _mm256_mul_ps(c0, _mm256_loadu_ps(scales.add(j))),
            );
            _mm256_storeu_ps(
                orow.add(j + 8),
                _mm256_mul_ps(c1, _mm256_loadu_ps(scales.add(j + 8))),
            );
            j += 16;
        }
        // 8-column tail.
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            for kk in 0..k {
                let av = _mm256_set1_ps(*arow.add(kk));
                let q = _mm_loadl_epi64(bq.add(kk * n + j) as *const __m128i); // 8 i8
                let b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
                c0 = _mm256_fmadd_ps(av, b0, c0);
            }
            _mm256_storeu_ps(
                orow.add(j),
                _mm256_mul_ps(c0, _mm256_loadu_ps(scales.add(j))),
            );
            j += 8;
        }
        // Scalar remainder.
        while j < n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += *arow.add(kk) * (*bq.add(kk * n + j) as f32);
            }
            *orow.add(j) = acc * *scales.add(j);
            j += 1;
        }
    }
}

/// Fused per-channel symmetric int8 matmul (zero-point 0). See module comment.
pub fn matmul_i8_per_channel(
    a: &[f32],
    bq: &[i8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bq.len(), k * n);
    debug_assert_eq!(scales.len(), n);
    debug_assert!(out.len() >= m * n);

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; sizes checked above.
    unsafe {
        matmul_i8_pc_neon(
            a.as_ptr(),
            bq.as_ptr(),
            scales.as_ptr(),
            out.as_mut_ptr(),
            m,
            k,
            n,
        );
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; sizes checked above.
    unsafe {
        matmul_i8_pc_wasm(
            a.as_ptr(),
            bq.as_ptr(),
            scales.as_ptr(),
            out.as_mut_ptr(),
            m,
            k,
            n,
        );
    }
    // x86-64 (AVX2 runtime-detected) + portable scalar fallback
    // (wasm-without-simd128 / other); aarch64 NEON and wasm SIMD128 ran above
    // and this block is compiled out for them.
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        let done = if x86_has_avx2() {
            // SAFETY: AVX2 detected; sizes checked above.
            unsafe {
                matmul_i8_pc_avx2(
                    a.as_ptr(),
                    bq.as_ptr(),
                    scales.as_ptr(),
                    out.as_mut_ptr(),
                    m,
                    k,
                    n,
                );
            }
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = false;
        if !done {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for kk in 0..k {
                        acc += a[i * k + kk] * (bq[kk * n + j] as f32);
                    }
                    out[i * n + j] = acc * scales[j];
                }
            }
        }
    }
}

// ─── Low-precision (bf16/f16) weight: widen + streamed GEMV ─────────
// A bf16/f16 matmul used to widen the WHOLE weight to an f32 scratch on every
// call (a scalar per-element `read_float` loop) and then run the f32 kernel.
// For an M=1 decode that re-materializes the entire constant weight each token
// — the dominant per-token cost. These two entry points remove it: a vectorized
// bulk widen for the large-M (prefill) materialize path, and a **streamed**
// small-M GEMV that reads the low-precision weight directly (widening 8/16
// elements in-register) and never materializes the f32 weight — the bf16/f16
// analog of the int8 `matmul_i8_per_channel` fused decode kernel.

/// Widen `bf16` (`is_f16=false`) or `f16` (`true`) elements from `src` (2 bytes
/// each, little-endian) into `out`, `out.len()` elements. Bit-identical to
/// element-wise `read_bf16`/`read_f16` (bf16 = `bits << 16`; f16 = IEEE half).
#[allow(clippy::needless_return)] // cfg-gated arch dispatch
pub fn widen_lowp_to_f32(src: &[u8], out: &mut [f32], is_f16: bool) {
    let n = out.len().min(src.len() / 2);
    #[cfg(target_arch = "x86_64")]
    if x86_has_avx2() {
        // SAFETY: AVX2 detected; bounds are `n` for both slices.
        unsafe {
            x86_widen_lowp(src.as_ptr(), out.as_mut_ptr(), n, is_f16);
        }
        return;
    }
    // NEON is baseline on aarch64. This is the *prefill* path — it widens the
    // whole k×n weight before the blocked f32 matmul — so a scalar loop here
    // throttles a SIMD matmul on a first-class target.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON baseline; bounds are `n` for both slices.
        unsafe {
            neon_widen_lowp(src.as_ptr(), out.as_mut_ptr(), n, is_f16);
        }
        return;
    }
    // wasm SIMD128 is the deployed target — same reasoning.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: simd128 gate; bounds are `n` for both slices.
        unsafe {
            wasm_widen_lowp(src.as_ptr(), out.as_mut_ptr(), n, is_f16);
        }
        return;
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    for (i, o) in out[..n].iter_mut().enumerate() {
        *o = lowp_scalar(src, i, is_f16);
    }
}

#[inline(always)]
fn lowp_scalar(src: &[u8], i: usize, is_f16: bool) -> f32 {
    let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
    if is_f16 {
        crate::cpu::dtype::f16_to_f32(bits)
    } else {
        f32::from_bits((bits as u32) << 16)
    }
}

/// AVX2 bulk widen (8-wide). bf16 zero-extends `u16→u32` and shifts left 16;
/// f16 uses the F16C `cvtph_ps` (present on every AVX2 CPU).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn x86_widen_lowp(src: *const u8, out: *mut f32, n: usize, is_f16: bool) {
    use core::arch::x86_64::*;
    let chunks = n / 8;
    for c in 0..chunks {
        let p = src.add(c * 16) as *const __m128i; // 8 × u16
        let h = _mm_loadu_si128(p);
        let f = if is_f16 {
            _mm256_cvtph_ps(h)
        } else {
            let u32s = _mm256_cvtepu16_epi32(h);
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(u32s))
        };
        _mm256_storeu_ps(out.add(c * 8), f);
    }
    let done = chunks * 8;
    for i in done..n {
        let bits = u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]);
        *out.add(i) = if is_f16 {
            crate::cpu::dtype::f16_to_f32(bits)
        } else {
            f32::from_bits((bits as u32) << 16)
        };
    }
}

/// NEON bulk widen (4-wide). bf16 zero-extends `u16→u32` and shifts left 16;
/// f16 uses the architectural half-precision convert (`vcvt_f32_f16`). Both are
/// bit-identical to `lowp_scalar` — bf16 is exact by construction, and NEON's
/// f16 convert is the IEEE-754 half→single conversion the scalar `f16_to_f32`
/// implements.
///
/// # Safety
/// NEON baseline on aarch64; `src` holds `2n` bytes, `out` holds `n` f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_widen_lowp(src: *const u8, out: *mut f32, n: usize, is_f16: bool) {
    use core::arch::aarch64::*;
    let chunks = n / 4;
    for c in 0..chunks {
        let p = src.add(c * 8) as *const u16;
        let h = vld1_u16(p); // 4 × u16
        let f = if is_f16 {
            vcvt_f32_f16(vreinterpret_f16_u16(h))
        } else {
            vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(h)))
        };
        vst1q_f32(out.add(c * 4), f);
    }
    for i in chunks * 4..n {
        let bits = u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]);
        *out.add(i) = if is_f16 {
            crate::cpu::dtype::f16_to_f32(bits)
        } else {
            f32::from_bits((bits as u32) << 16)
        };
    }
}

/// wasm SIMD128 bulk widen (4-wide) — the deployed target's twin. bf16 shifts
/// the zero-extended `u16` left 16; f16 has no SIMD128 convert instruction, so
/// it falls to the exact scalar `f16_to_f32` per lane (still bit-identical, and
/// bf16 — the common low-precision weight — takes the vector path).
///
/// # Safety
/// simd128 enabled; `src` holds `2n` bytes, `out` holds `n` f32.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn wasm_widen_lowp(src: *const u8, out: *mut f32, n: usize, is_f16: bool) {
    use core::arch::wasm32::*;
    if !is_f16 {
        let chunks = n / 4;
        for c in 0..chunks {
            // 4 × u16 → zero-extend to u32 → << 16 → reinterpret as f32.
            let h = v128_load64_zero(src.add(c * 8) as *const u64);
            let u32s = u32x4_extend_low_u16x8(h);
            v128_store(out.add(c * 4) as *mut v128, i32x4_shl(u32s, 16));
        }
        for i in chunks * 4..n {
            let bits = u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]);
            *out.add(i) = f32::from_bits((bits as u32) << 16);
        }
        return;
    }
    for i in 0..n {
        let bits = u16::from_le_bytes([*src.add(i * 2), *src.add(i * 2 + 1)]);
        *out.add(i) = crate::cpu::dtype::f16_to_f32(bits);
    }
}

/// Streamed small-M matmul `out[m,n] = a[m,k] · widen(B[k,n])` where `B` is a
/// low-precision (bf16/f16) row-major weight read **directly** — widened in
/// registers, never materialized to an f32 panel. `a` is already f32. For the
/// M=1 decode shape this is a single streaming pass over `B` (2 bytes/elem),
/// the bf16/f16 analog of the int8 decode kernel. Result f32-accumulated,
/// bit-identical to widen-then-`matmul_f32` (both sum `a·widen(b)` in the same
/// order per output).
#[allow(clippy::needless_return)] // cfg-gated arch dispatch
pub fn matmul_lowp_gemv(
    a: &[f32],
    b: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    is_f16: bool,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    debug_assert_eq!(a.len(), m * k);
    debug_assert!(b.len() >= k * n * 2);
    debug_assert!(out.len() >= m * n);

    #[cfg(target_arch = "x86_64")]
    {
        if x86_has_avx2() {
            // SAFETY: AVX2 (+F16C) detected; sizes checked above.
            unsafe {
                x86_matmul_lowp_gemv(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n, is_f16);
            }
        } else {
            matmul_lowp_gemv_scalar(a, b, out, m, k, n, is_f16);
        }
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: simd128 gate; sizes checked above.
        unsafe {
            wasm_matmul_lowp_gemv(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n, is_f16);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON baseline; sizes checked above.
        unsafe {
            neon_matmul_lowp_gemv(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n, is_f16);
        }
        return;
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128"),
        target_arch = "aarch64"
    )))]
    matmul_lowp_gemv_scalar(a, b, out, m, k, n, is_f16);
}

/// Portable scalar streamed low-precision GEMV — still reads `B` directly (no
/// f32 materialization); the SIMD lanes above supersede it per arch.
fn matmul_lowp_gemv_scalar(
    a: &[f32],
    b: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    is_f16: bool,
) {
    for (i, orow) in out[..m * n].chunks_exact_mut(n).enumerate() {
        let arow = &a[i * k..i * k + k];
        orow.fill(0.0);
        for (kk, &av) in arow.iter().enumerate() {
            let brow = kk * n;
            for (j, o) in orow.iter_mut().enumerate() {
                *o += av * lowp_scalar(b, brow + j, is_f16);
            }
        }
    }
}

/// wasm SIMD128 streamed low-precision GEMV — the deployed-target twin of the
/// x86 kernel. SIMD128 is 4-wide and has no FMA (mul + add). The M=1 decode path
/// uses 8 independent `f32x4` accumulators (32 columns) to hide the add-chain
/// latency; bf16 widens with the extend+shift ladder, f16 is widened per lane
/// (no SIMD128 half-conversion). Larger M falls to a straightforward streamed
/// row loop.
///
/// # Safety
/// simd128 enabled; `a` is `m*k` f32, `b` is `k*n*2` bytes, `out` is `m*n` f32.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn wasm_matmul_lowp_gemv(
    a: *const f32,
    b: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    is_f16: bool,
) {
    use core::arch::wasm32::*;
    #[inline(always)]
    unsafe fn widen4(p: *const u8, is_f16: bool) -> v128 {
        if is_f16 {
            f32x4(
                crate::cpu::dtype::f16_to_f32(u16::from_le_bytes([*p, *p.add(1)])),
                crate::cpu::dtype::f16_to_f32(u16::from_le_bytes([*p.add(2), *p.add(3)])),
                crate::cpu::dtype::f16_to_f32(u16::from_le_bytes([*p.add(4), *p.add(5)])),
                crate::cpu::dtype::f16_to_f32(u16::from_le_bytes([*p.add(6), *p.add(7)])),
            )
        } else {
            // bf16: zero-extend low 4 u16 → u32, shift left 16, reinterpret f32.
            let v = v128_load64_zero(p as *const u64); // 4 × u16 in the low 64 bits
            i32x4_shl(u32x4_extend_low_u16x8(v), 16)
        }
    }
    if m == 1 {
        let mut j = 0;
        while j + 32 <= n {
            let mut c = [f32x4_splat(0.0); 8];
            for kk in 0..k {
                let av = f32x4_splat(*a.add(kk));
                let bp = b.add((kk * n + j) * 2);
                for (t, cc) in c.iter_mut().enumerate() {
                    *cc = f32x4_add(*cc, f32x4_mul(av, widen4(bp.add(t * 8), is_f16)));
                }
            }
            for (t, &cc) in c.iter().enumerate() {
                v128_store(out.add(j + t * 4) as *mut v128, cc);
            }
            j += 32;
        }
        while j + 4 <= n {
            let mut c0 = f32x4_splat(0.0);
            for kk in 0..k {
                let av = f32x4_splat(*a.add(kk));
                c0 = f32x4_add(c0, f32x4_mul(av, widen4(b.add((kk * n + j) * 2), is_f16)));
            }
            v128_store(out.add(j) as *mut v128, c0);
            j += 4;
        }
        while j < n {
            let mut acc = 0f32;
            let bs = core::slice::from_raw_parts(b, k * n * 2);
            for kk in 0..k {
                acc += *a.add(kk) * lowp_scalar(bs, kk * n + j, is_f16);
            }
            *out.add(j) = acc;
            j += 1;
        }
        return;
    }
    // General small-M: hand off to the portable streamed scalar path.
    matmul_lowp_gemv_scalar(
        core::slice::from_raw_parts(a, m * k),
        core::slice::from_raw_parts(b, k * n * 2),
        core::slice::from_raw_parts_mut(out, m * n),
        m,
        k,
        n,
        is_f16,
    );
}

/// NEON streamed low-precision GEMV — the aarch64 twin. M=1 uses 8 `float32x4`
/// accumulators (32 columns); bf16 widens via `vshll` (u16→u32 << 16), f16 via
/// `vcvt_f32_f16`.
///
/// # Safety
/// NEON baseline; `a` is `m*k` f32, `b` is `k*n*2` bytes, `out` is `m*n` f32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_lowp_gemv(
    a: *const f32,
    b: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    is_f16: bool,
) {
    use core::arch::aarch64::*;
    #[inline(always)]
    unsafe fn widen4(p: *const u8, is_f16: bool) -> float32x4_t {
        if is_f16 {
            let h = vld1_u16(p as *const u16); // 4 × f16
            vcvt_f32_f16(vreinterpret_f16_u16(h))
        } else {
            let u = vld1_u16(p as *const u16); // 4 × u16 (bf16)
                                               // Widen u16 → u32, shift left 16, reinterpret as f32.
            vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(u)))
        }
    }
    if m == 1 {
        let mut j = 0;
        while j + 32 <= n {
            let mut c = [vdupq_n_f32(0.0); 8];
            for kk in 0..k {
                let av = vdupq_n_f32(*a.add(kk));
                let bp = b.add((kk * n + j) * 2);
                for (t, cc) in c.iter_mut().enumerate() {
                    *cc = vfmaq_f32(*cc, av, widen4(bp.add(t * 8), is_f16));
                }
            }
            for (t, &cc) in c.iter().enumerate() {
                vst1q_f32(out.add(j + t * 4), cc);
            }
            j += 32;
        }
        while j + 4 <= n {
            let mut c0 = vdupq_n_f32(0.0);
            for kk in 0..k {
                let av = vdupq_n_f32(*a.add(kk));
                c0 = vfmaq_f32(c0, av, widen4(b.add((kk * n + j) * 2), is_f16));
            }
            vst1q_f32(out.add(j), c0);
            j += 4;
        }
        while j < n {
            let mut acc = 0f32;
            let bs = core::slice::from_raw_parts(b, k * n * 2);
            for kk in 0..k {
                acc += *a.add(kk) * lowp_scalar(bs, kk * n + j, is_f16);
            }
            *out.add(j) = acc;
            j += 1;
        }
        return;
    }
    // General small-M: hand off to the portable streamed scalar path.
    matmul_lowp_gemv_scalar(
        core::slice::from_raw_parts(a, m * k),
        core::slice::from_raw_parts(b, k * n * 2),
        core::slice::from_raw_parts_mut(out, m * n),
        m,
        k,
        n,
        is_f16,
    );
}

/// x86 AVX2 streamed low-precision GEMV: 16-column panels, two YMM
/// accumulators per M-row, the 16 bf16/f16 weights widened in-register each
/// k-step. Mirrors `matmul_i8_pc_avx2` (no per-channel scale).
///
/// # Safety
/// AVX2 + FMA + F16C (the caller gates on `x86_has_avx2`); `a` is `m*k` f32,
/// `b` is `k*n*2` bytes, `out` is `m*n` f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn x86_matmul_lowp_gemv(
    a: *const f32,
    b: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    is_f16: bool,
) {
    use core::arch::x86_64::*;
    #[inline(always)]
    unsafe fn widen8(p: *const u8, is_f16: bool) -> __m256 {
        let h = _mm_loadu_si128(p as *const __m128i); // 8 × u16
        if is_f16 {
            _mm256_cvtph_ps(h)
        } else {
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
        }
    }
    #[inline(always)]
    unsafe fn widen16(p: *const u8, is_f16: bool) -> (__m256, __m256) {
        (widen8(p, is_f16), widen8(p.add(16), is_f16))
    }
    // M=1 decode fast path: 8 YMM accumulators (64 columns) to hide the FMA
    // latency — the loop-carried accumulate chain is otherwise the limiter (only
    // 2 in-flight FMAs with the general path's 16-column panel).
    if m == 1 {
        let mut j = 0;
        while j + 64 <= n {
            let mut c = [_mm256_setzero_ps(); 8];
            for kk in 0..k {
                let av = _mm256_set1_ps(*a.add(kk));
                let bp = b.add((kk * n + j) * 2);
                c[0] = _mm256_fmadd_ps(av, widen8(bp, is_f16), c[0]);
                c[1] = _mm256_fmadd_ps(av, widen8(bp.add(16), is_f16), c[1]);
                c[2] = _mm256_fmadd_ps(av, widen8(bp.add(32), is_f16), c[2]);
                c[3] = _mm256_fmadd_ps(av, widen8(bp.add(48), is_f16), c[3]);
                c[4] = _mm256_fmadd_ps(av, widen8(bp.add(64), is_f16), c[4]);
                c[5] = _mm256_fmadd_ps(av, widen8(bp.add(80), is_f16), c[5]);
                c[6] = _mm256_fmadd_ps(av, widen8(bp.add(96), is_f16), c[6]);
                c[7] = _mm256_fmadd_ps(av, widen8(bp.add(112), is_f16), c[7]);
            }
            for (t, &cc) in c.iter().enumerate() {
                _mm256_storeu_ps(out.add(j + t * 8), cc);
            }
            j += 64;
        }
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            for kk in 0..k {
                let av = _mm256_set1_ps(*a.add(kk));
                c0 = _mm256_fmadd_ps(av, widen8(b.add((kk * n + j) * 2), is_f16), c0);
            }
            _mm256_storeu_ps(out.add(j), c0);
            j += 8;
        }
        while j < n {
            let mut acc = 0f32;
            let bs = core::slice::from_raw_parts(b, k * n * 2);
            for kk in 0..k {
                acc += *a.add(kk) * lowp_scalar(bs, kk * n + j, is_f16);
            }
            *out.add(j) = acc;
            j += 1;
        }
        return;
    }
    for i in 0..m {
        let arow = a.add(i * k);
        let orow = out.add(i * n);
        let mut j = 0;
        while j + 16 <= n {
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            for kk in 0..k {
                let av = _mm256_set1_ps(*arow.add(kk));
                let (b0, b1) = widen16(b.add((kk * n + j) * 2), is_f16);
                c0 = _mm256_fmadd_ps(av, b0, c0);
                c1 = _mm256_fmadd_ps(av, b1, c1);
            }
            _mm256_storeu_ps(orow.add(j), c0);
            _mm256_storeu_ps(orow.add(j + 8), c1);
            j += 16;
        }
        while j + 8 <= n {
            let mut c0 = _mm256_setzero_ps();
            for kk in 0..k {
                let av = _mm256_set1_ps(*arow.add(kk));
                let p = b.add((kk * n + j) * 2) as *const __m128i;
                let bw = if is_f16 {
                    _mm256_cvtph_ps(_mm_loadu_si128(p))
                } else {
                    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                        _mm_loadu_si128(p),
                    )))
                };
                c0 = _mm256_fmadd_ps(av, bw, c0);
            }
            _mm256_storeu_ps(orow.add(j), c0);
            j += 8;
        }
        while j < n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += *arow.add(kk)
                    * lowp_scalar(
                        core::slice::from_raw_parts(b, k * n * 2),
                        kk * n + j,
                        is_f16,
                    );
            }
            *orow.add(j) = acc;
            j += 1;
        }
    }
}

// ─── Deterministic vectorized f32 exp (decode softmax path) ────────
// The decode path runs softmax per head per step over the attention bucket;
// its exp was a scalar `libm::expf` per element. This is the vectorized
// replacement: one fixed algorithm — range reduction `x = k·ln2 + r` with a
// trunc-cast round-half-away (the same rounding discipline as the W8A8
// quantizer), a degree-6 Taylor polynomial in Horner form over
// |r| ≤ ln2/2, and a 2^k exponent-bit scale — evaluated with plain IEEE
// mul/add (deliberately no FMA) so scalar, NEON, and wasm SIMD128 lanes
// produce **bit-identical** results. That is stronger determinism than the
// path it replaces: `libm::expf` (no_std) and a platform libm (std) did not
// agree bit-for-bit across builds. Inputs below `EXP_F32_LO` (including the
// causal mask's −∞ scores) map to exactly 0.0, so masked attention
// positions keep zero probability; NaN also maps to 0.0 (out-of-domain,
// documented), and inputs above `EXP_F32_HI` clamp.

/// Underflow cutoff: `exp(x) = 0.0` exactly for `x < EXP_F32_LO` (−∞ and
/// NaN included). e^−87.3 ≈ 1.2e−38 is the last normal-range value.
#[deprecated(
    since = "0.8.0",
    note = "kernel-internal constant; it was never part of a supported surface. Removed in 0.9.0."
)]
pub const EXP_F32_LO: f32 = -87.336_54;
/// Clamp ceiling, chosen so the scale exponent `k ≤ 127` stays a normal
/// f32 (e^88 ≈ 1.65e38 < f32::MAX).
#[deprecated(
    since = "0.8.0",
    note = "kernel-internal constant; it was never part of a supported surface. Removed in 0.9.0."
)]
pub const EXP_F32_HI: f32 = 88.0;

const EXP_LOG2E: f32 = core::f32::consts::LOG2_E;
// Cody–Waite split of ln2: HI has zeroed low mantissa bits so `kf·HI` is
// exact for |k| ≤ 2^15; HI + LO = ln2 to ~1e-11. HI is spelled as its exact
// bit pattern (0.693359375) so the zeroed low bits are explicit.
const EXP_LN2_HI: f32 = f32::from_bits(0x3F31_8000);
const EXP_LN2_LO: f32 = -2.121_944_4e-4;
// Degree-6 Taylor coefficients 1/6!, …, 1/2! (Horner from C6 down to r+1).
const EXP_C6: f32 = 1.0 / 720.0;
const EXP_C5: f32 = 1.0 / 120.0;
const EXP_C4: f32 = 1.0 / 24.0;
const EXP_C3: f32 = 1.0 / 6.0;
const EXP_C2: f32 = 0.5;

/// The scalar specification of the deterministic exp. Every SIMD lane
/// computes exactly this sequence of IEEE operations; the bit-identity
/// tests compare lanes against it.
#[inline]
pub fn exp_f32_det(x: f32) -> f32 {
    if x.is_nan() || x < EXP_F32_LO {
        return 0.0; // underflow, −∞, NaN
    }
    let x = if x > EXP_F32_HI { EXP_F32_HI } else { x };
    let t = x * EXP_LOG2E;
    let k = if t >= 0.0 {
        (t + 0.5) as i32
    } else {
        (t - 0.5) as i32
    };
    let kf = k as f32;
    let r = (x - kf * EXP_LN2_HI) - kf * EXP_LN2_LO;
    let mut p = EXP_C6;
    p = p * r + EXP_C5;
    p = p * r + EXP_C4;
    p = p * r + EXP_C3;
    p = p * r + EXP_C2;
    p = p * r + 1.0;
    p = p * r + 1.0;
    let scale = f32::from_bits((((k + 127) as u32) << 23).min(0x7f00_0000));
    p * scale
}

/// NEON lanes of [`exp_f32_det`] — the identical operation sequence, 4 wide.
///
/// # Safety
/// NEON (baseline aarch64); `xs` valid for `len` reads/writes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_f32_neon_inplace(xs: *mut f32, len: usize) {
    use core::arch::aarch64::*;
    let lo = vdupq_n_f32(EXP_F32_LO);
    let hi = vdupq_n_f32(EXP_F32_HI);
    let log2e = vdupq_n_f32(EXP_LOG2E);
    let ln2_hi = vdupq_n_f32(EXP_LN2_HI);
    let ln2_lo = vdupq_n_f32(EXP_LN2_LO);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= len {
        let x0 = vld1q_f32(xs.add(i));
        // Keep-mask on the ORIGINAL input: false for underflow/−∞/NaN.
        let keep = vcgeq_f32(x0, lo);
        let x = vminq_f32(x0, hi);
        let t = vmulq_f32(x, log2e);
        // round half away from zero: trunc(t ± 0.5) by sign of t.
        let neg = vcltq_f32(t, vdupq_n_f32(0.0));
        let adj = vbslq_f32(neg, vnegq_f32(half), half);
        let k = vcvtq_s32_f32(vaddq_f32(t, adj)); // trunc toward zero
        let kf = vcvtq_f32_s32(k);
        let r = vsubq_f32(vsubq_f32(x, vmulq_f32(kf, ln2_hi)), vmulq_f32(kf, ln2_lo));
        let mut p = vdupq_n_f32(EXP_C6);
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EXP_C5));
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EXP_C4));
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EXP_C3));
        p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(EXP_C2));
        p = vaddq_f32(vmulq_f32(p, r), one);
        p = vaddq_f32(vmulq_f32(p, r), one);
        let bits = vshlq_n_s32::<23>(vaddq_s32(k, vdupq_n_s32(127)));
        let scale = vreinterpretq_f32_s32(bits);
        let e = vmulq_f32(p, scale);
        let z = vdupq_n_f32(0.0);
        vst1q_f32(xs.add(i), vbslq_f32(keep, e, z));
        i += 4;
    }
    while i < len {
        *xs.add(i) = exp_f32_det(*xs.add(i));
        i += 1;
    }
}

/// wasm SIMD128 lanes of [`exp_f32_det`] — the identical operation
/// sequence, 4 wide.
///
/// # Safety
/// simd128 enabled; `xs` valid for `len` reads/writes.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn exp_f32_wasm_inplace(xs: *mut f32, len: usize) {
    use core::arch::wasm32::*;
    let lo = f32x4_splat(EXP_F32_LO);
    let hi = f32x4_splat(EXP_F32_HI);
    let log2e = f32x4_splat(EXP_LOG2E);
    let ln2_hi = f32x4_splat(EXP_LN2_HI);
    let ln2_lo = f32x4_splat(EXP_LN2_LO);
    let half = f32x4_splat(0.5);
    let one = f32x4_splat(1.0);
    let mut i = 0;
    while i + 4 <= len {
        let x0 = v128_load(xs.add(i) as *const v128);
        let keep = f32x4_ge(x0, lo);
        // pmin semantics match the scalar `if x > HI { HI }` for non-NaN x;
        // NaN lanes are discarded by `keep`.
        let x = f32x4_pmin(x0, hi);
        let t = f32x4_mul(x, log2e);
        let neg = f32x4_lt(t, f32x4_splat(0.0));
        let adj = v128_bitselect(f32x4_neg(half), half, neg);
        let k = i32x4_trunc_sat_f32x4(f32x4_add(t, adj));
        let kf = f32x4_convert_i32x4(k);
        let r = f32x4_sub(f32x4_sub(x, f32x4_mul(kf, ln2_hi)), f32x4_mul(kf, ln2_lo));
        let mut p = f32x4_splat(EXP_C6);
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C5));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C4));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C3));
        p = f32x4_add(f32x4_mul(p, r), f32x4_splat(EXP_C2));
        p = f32x4_add(f32x4_mul(p, r), one);
        p = f32x4_add(f32x4_mul(p, r), one);
        let bits = i32x4_shl(i32x4_add(k, i32x4_splat(127)), 23);
        let e = f32x4_mul(p, bits);
        let z = f32x4_splat(0.0);
        v128_store(xs.add(i) as *mut v128, v128_bitselect(e, z, keep));
        i += 4;
    }
    while i < len {
        *xs.add(i) = exp_f32_det(*xs.add(i));
        i += 1;
    }
}

/// Elementwise deterministic `exp` in place — the decode softmax's exp
/// pass. Bit-identical across scalar / x86 AVX2 / NEON / wasm SIMD128 (see
/// [`exp_f32_det`], the scalar specification every lane replays).
#[allow(clippy::needless_return)] // cfg-gated arch dispatch
pub fn simd_f32_exp_inplace(xs: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; slice bounds by construction.
    unsafe {
        exp_f32_neon_inplace(xs.as_mut_ptr(), xs.len());
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: simd128 gate; slice bounds by construction.
    unsafe {
        exp_f32_wasm_inplace(xs.as_mut_ptr(), xs.len());
        return;
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if x86_has_avx2() {
            // SAFETY: AVX2 detected; slice bounds by construction.
            unsafe {
                exp_f32_avx2_inplace(xs.as_mut_ptr(), xs.len());
            }
            return;
        }
        for x in xs.iter_mut() {
            *x = exp_f32_det(*x);
        }
    }
}

// ─── x86_64 AVX2 lanes for the decode kernels ──────────────────────
// The integer GEMV and deterministic-exp kernels below carry NEON and wasm
// SIMD128 inners; on x86_64 (a first-class deployment target, and the native
// bench/CI lane) they must not fall to scalar — per this module's contract, a
// stock build picks the widest ISA at runtime. These AVX2 inners replay the
// EXACT scalar specifications: the integer dot uses `_mm256_madd_epi16` over
// i8→i16 widenings (an exact, associative i32 reduction — bit-identical to the
// scalar/NEON/wasm total), and the exp reuses the same no-FMA op sequence as
// `exp_f32_det`. Selected by runtime CPUID (`is_x86_feature_detected!`) under
// `std`, or the build's compile-time feature floor on no_std x86.

/// `true` if the AVX2 decode lanes are usable — runtime CPUID under `std`,
/// the build target's compile-time floor on no_std.
#[cfg(target_arch = "x86_64")]
#[inline]
fn x86_has_avx2() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2")
    }
}

/// Exact horizontal sum of the eight i32 lanes of a `__m256i`.
///
/// # Safety
/// AVX2 must be enabled at the call site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_i32(v: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0100_1110>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0000_0001>(s));
    _mm_cvtsi128_si32(s)
}

/// AVX2 inner for the output-major W8A8 int8 GEMV — the x86 twin of
/// `gemv_i8_omajor_neon`/`_wasm`: i8→i16 widen (`_mm256_cvtepi8_epi16`) then
/// `_mm256_madd_epi16` exact-i32 pairwise accumulation, 4 outputs in flight.
///
/// # Safety
/// AVX2 enabled; `q` len `k`, `bq` len `n*k`, `scales` len `n`, `out` len
/// `n`; `k ≤ I8_DOT_K_MAX`. Loads are unaligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemv_i8_omajor_avx2(
    q: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::x86_64::*;
    let kv = k & !15;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * k),
            bq.add((j + 1) * k),
            bq.add((j + 2) * k),
            bq.add((j + 3) * k),
        ];
        let mut c = [_mm256_setzero_si256(); 4];
        let mut kk = 0;
        while kk < kv {
            let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.add(kk) as *const __m128i));
            for (cr, row) in c.iter_mut().zip(rows.iter()) {
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.add(kk) as *const __m128i));
                *cr = _mm256_add_epi32(*cr, _mm256_madd_epi16(av, w));
            }
            kk += 16;
        }
        let mut sums = [0i32; 4];
        for (sr, cr) in sums.iter_mut().zip(c.iter()) {
            *sr = hsum256_i32(*cr);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in sums.iter_mut().zip(rows.iter()) {
                *sr += qa * (*row.add(kk) as i32);
            }
            kk += 1;
        }
        for (r, &sr) in sums.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = bq.add(j * k);
        let mut s = 0i32;
        for kk in 0..k {
            s += (*q.add(kk) as i32) * (*row.add(kk) as i32);
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// Native (x86_64 AVX2 / aarch64 NEON) serial inner for one quantized
/// activation row against a contiguous output-column range of the omajor W8A8
/// weight — the unit both the serial call and each pool task run. Every output
/// column is a whole dot computed by one participant in the same reduction
/// order, so a partitioned decode GEMV is bit-identical to the serial run.
///
/// # Safety
/// `q` len `k`; `bq`/`scales`/`out` are the sub-range bases (len `k` per
/// column, `n` columns); `k ≤ I8_DOT_K_MAX`. Unaligned loads.
#[cfg(all(
    feature = "parallel",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[inline]
unsafe fn gemv_i8_omajor_native(
    q: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        gemv_i8_omajor_neon(q, bq, scales, out, k, n, scale_a);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86_has_avx2() {
            gemv_i8_omajor_avx2(q, bq, scales, out, k, n, scale_a);
        } else {
            for j in 0..n {
                let row = bq.add(j * k);
                let mut s = 0i32;
                for kk in 0..k {
                    s += (*q.add(kk) as i32) * (*row.add(kk) as i32);
                }
                *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
            }
        }
    }
}

/// AVX2 inner for the output-major W4A8 packed-i4 GEMV — the x86 twin of the
/// de-interleaved wasm i4 kernel. The activation arrives split
/// (`de = [q_even | q_odd]`, each `k/2`, built once per token) so the packed
/// nibbles need no reorder: one 16-byte weight load yields 32 values, the low
/// nibbles pair with the even activations and the high with the odd, through
/// two `_mm_shuffle_epi8` LUT hits into the same exact madd pipeline.
///
/// # Safety
/// AVX2 enabled; `q` len `k` (scalar tail), `de` len `k`, `bq` len `n*k/2`,
/// `scales` len `n`, `out` len `n`; `k` even, `k ≤ I8_DOT_K_MAX`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i4_omajor_avx2(
    q: *const i8,
    de: *const i8,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::x86_64::*;
    let kb = k / 2;
    let (qe, qo) = (de, de.add(kb));
    let table = _mm_loadu_si128(I4_VALUES.as_ptr() as *const __m128i);
    let low_mask = _mm_set1_epi8(0x0F);
    let kv = k & !31;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * kb),
            bq.add((j + 1) * kb),
            bq.add((j + 2) * kb),
            bq.add((j + 3) * kb),
        ];
        let mut c = [_mm256_setzero_si256(); 4];
        let mut kk = 0;
        while kk < kv {
            let h = kk / 2; // 16 packed bytes = weights kk..kk+32
            let ae = _mm256_cvtepi8_epi16(_mm_loadu_si128(qe.add(h) as *const __m128i));
            let ao = _mm256_cvtepi8_epi16(_mm_loadu_si128(qo.add(h) as *const __m128i));
            for (cr, row) in c.iter_mut().zip(rows.iter()) {
                let b = _mm_loadu_si128(row.add(h) as *const __m128i);
                let we8 = _mm_shuffle_epi8(table, _mm_and_si128(b, low_mask));
                let wo8 = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(b), low_mask));
                let we = _mm256_cvtepi8_epi16(we8);
                let wo = _mm256_cvtepi8_epi16(wo8);
                *cr = _mm256_add_epi32(*cr, _mm256_madd_epi16(ae, we));
                *cr = _mm256_add_epi32(*cr, _mm256_madd_epi16(ao, wo));
            }
            kk += 32;
        }
        let mut sums = [0i32; 4];
        for (sr, cr) in sums.iter_mut().zip(c.iter()) {
            *sr = hsum256_i32(*cr);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in sums.iter_mut().zip(rows.iter()) {
                *sr += qa * i4_at(core::slice::from_raw_parts(*row, kb), kk) as i32;
            }
            kk += 1;
        }
        for (r, &sr) in sums.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = core::slice::from_raw_parts(bq.add(j * kb), kb);
        let mut sv = 0i32;
        for kk in 0..k {
            sv += (*q.add(kk) as i32) * i4_at(row, kk) as i32;
        }
        *out.add(j) = (sv as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// AVX2 lanes of [`exp_f32_det`] — the identical no-FMA operation sequence,
/// 8 wide.
///
/// # Safety
/// AVX2 enabled; `xs` valid for `len` reads/writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_f32_avx2_inplace(xs: *mut f32, len: usize) {
    use core::arch::x86_64::*;
    let lo = _mm256_set1_ps(EXP_F32_LO);
    let hi = _mm256_set1_ps(EXP_F32_HI);
    let log2e = _mm256_set1_ps(EXP_LOG2E);
    let ln2_hi = _mm256_set1_ps(EXP_LN2_HI);
    let ln2_lo = _mm256_set1_ps(EXP_LN2_LO);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= len {
        let x0 = _mm256_loadu_ps(xs.add(i));
        // keep = x >= LO; a NaN lane compares false (→ zeroed), matching the
        // scalar `is_nan() || x < LO → 0.0`.
        let keep = _mm256_cmp_ps::<_CMP_GE_OQ>(x0, lo);
        let x = _mm256_min_ps(x0, hi);
        let t = _mm256_mul_ps(x, log2e);
        let neg = _mm256_cmp_ps::<_CMP_LT_OQ>(t, zero);
        let adj = _mm256_blendv_ps(half, _mm256_sub_ps(zero, half), neg);
        let ki = _mm256_cvttps_epi32(_mm256_add_ps(t, adj));
        let kf = _mm256_cvtepi32_ps(ki);
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(kf, ln2_hi)),
            _mm256_mul_ps(kf, ln2_lo),
        );
        let mut p = _mm256_set1_ps(EXP_C6);
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_C5));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_C4));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_C3));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_C2));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
        p = _mm256_add_ps(_mm256_mul_ps(p, r), one);
        let bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(ki, _mm256_set1_epi32(127)));
        let bits = _mm256_min_epi32(bits, _mm256_set1_epi32(0x7f00_0000));
        let scale = _mm256_castsi256_ps(bits);
        let e = _mm256_mul_ps(p, scale);
        _mm256_storeu_ps(xs.add(i), _mm256_blendv_ps(zero, e, keep));
        i += 8;
    }
    while i < len {
        *xs.add(i) = exp_f32_det(*xs.add(i));
        i += 1;
    }
}

// ─── Output-major W8A8 int8 GEMV (decode) ──────────────────────────
// `matmul_i8_per_channel` above reads the `[k,n]` weight k-inner: at decode
// (m = 1) that walk touches 16 bytes of every 64-byte line with no line reuse
// between column tiles (reuse distance ≈ the whole matrix), and every product
// is float (W8A32). This kernel is the decode-shaped replacement: the weight
// is **output-major** `[n,k]` — each output's k-vector contiguous; the layout
// is compile-time derived content under its own κ (see the compiler's
// weight-layout monomorphism pass) — the activation row is quantized once per
// token to symmetric i8 (W8A8), and the dot products accumulate in **exact
// integer** arithmetic (wasm `i32x4_dot_i16x8`, NEON `vmull_s8` +
// `vpadalq_s16`). Integer sums are associative, so scalar / NEON / wasm
// produce bit-identical output — a single fused `(Σ q·w) · (scale_a ·
// scale_w[j])` writeback per output is the only float rounding, and the
// reduction order cannot perturb CE derivation keys.

/// Upper bound on `k` for exact i32 accumulation: `k · 127²` must stay below
/// `i32::MAX`. Rejected loudly; real decode shapes sit three orders of
/// magnitude below (~2k–19k). One definition, shared with the compiler's
/// emission gate.
#[deprecated(
    since = "0.8.0",
    note = "kernel-internal constant; it was never part of a supported surface. Removed in 0.9.0."
)]
pub const I8_DOT_K_MAX: usize = crate::kernel_call::mm_act_quant::K_MAX;

/// Reused per-token quantized-activation row (zero alloc per call after
/// warm-up under `std`; a transient alloc on `no_std`, matching the other
/// kernel scratches).
#[cfg(feature = "std")]
fn with_q8_scratch<R>(f: impl FnOnce(&mut Vec<i8>) -> R) -> R {
    std::thread_local! {
        static Q8: core::cell::RefCell<Vec<i8>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    Q8.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(not(feature = "std"))]
fn with_q8_scratch<R>(f: impl FnOnce(&mut Vec<i8>) -> R) -> R {
    let mut v = Vec::new();
    f(&mut v)
}

/// Two reusable i8 scratch buffers for the vectorized quantized-GEMV
/// activation re-layouts (zero alloc per call after warm-up under `std`;
/// transient on no_std, like the other kernel scratches). Callers repurpose
/// them per path: the wasm i8 relaxed tier holds the `q⁺ / q⁻` i7 split
/// (both `k`); the i4 paths pack the de-interleaved activation into the
/// first buffer (`k` baseline, `2k` wasm-relaxed) and leave the second idle.
/// The buffers carry no fixed size — each caller `resize`s to its own
/// `k`-derived length — so the scratch is shape-agnostic.
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
))]
fn with_gemv_scratch<R>(f: impl FnOnce(&mut Vec<i8>, &mut Vec<i8>) -> R) -> R {
    #[cfg(feature = "std")]
    {
        std::thread_local! {
            static SCRATCH: core::cell::RefCell<(Vec<i8>, Vec<i8>)> =
                const { core::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        SCRATCH.with(|cell| {
            let mut g = cell.borrow_mut();
            let (p, n) = &mut *g;
            f(p, n)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        let (mut p, mut n) = (Vec::new(), Vec::new());
        f(&mut p, &mut n)
    }
}

/// `q = q⁺ − q⁻` elementwise: `q⁺ = max(q, 0)`, `q⁻ = max(−q, 0)`, both in
/// `[0, 127]`. O(k) against the GEMV's O(k·n).
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
fn split_q7(q: &[i8], qp: &mut [i8], qn: &mut [i8]) {
    for ((&v, p), n) in q.iter().zip(qp.iter_mut()).zip(qn.iter_mut()) {
        *p = v.max(0);
        *n = (-v).max(0);
    }
}

/// Quantize one activation row to symmetric i8: `scale = max|a| / 127`,
/// `q = clamp(round_half_away_from_zero(a · (127 / max|a|)), -127, 127)`.
/// Returns the scale (`0.0` for an all-zero row — the caller writes a zero
/// output row). A deterministic pure function of the row bytes: IEEE f32
/// mul/add plus Rust's saturating trunc-cast, no libm, no fenv, no
/// data-dependent order.
fn quantize_row_i8(a: &[f32], q: &mut [i8]) -> f32 {
    let mut amax = 0f32;
    for &v in a {
        let av = if v < 0.0 { -v } else { v };
        if av > amax {
            amax = av;
        }
    }
    if amax == 0.0 {
        return 0.0;
    }
    let inv = 127.0 / amax;
    for (dst, &v) in q.iter_mut().zip(a) {
        let t = v * inv;
        // Round half away from zero via trunc-cast; |t| ≤ 127 + ulps by
        // construction, clamp guards the boundary ulp.
        let r = if t >= 0.0 {
            (t + 0.5) as i32
        } else {
            (t - 0.5) as i32
        };
        *dst = r.clamp(-127, 127) as i8;
    }
    amax / 127.0
}

/// NEON inner: one quantized activation row (`q`, len `k`) against the
/// output-major weight (`bq`, `[n,k]`), 4 outputs in flight, exact i32
/// accumulation (`vmull_s8` products pairwise-accumulated by `vpadalq_s16`).
///
/// # Safety
/// NEON (baseline aarch64); `q` len `k`, `bq` len `n*k`, `scales` len `n`,
/// `out` len `n`; `k ≤ I8_DOT_K_MAX`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemv_i8_omajor_neon(
    q: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::aarch64::*;
    let kv = k & !15;
    let mut j = 0;
    while j + 4 <= n {
        let r0 = bq.add(j * k);
        let r1 = bq.add((j + 1) * k);
        let r2 = bq.add((j + 2) * k);
        let r3 = bq.add((j + 3) * k);
        let (mut c0, mut c1, mut c2, mut c3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        let mut kk = 0;
        while kk < kv {
            let av = vld1q_s8(q.add(kk));
            let alo = vget_low_s8(av);
            let ahi = vget_high_s8(av);
            let w0 = vld1q_s8(r0.add(kk));
            c0 = vpadalq_s16(c0, vmull_s8(alo, vget_low_s8(w0)));
            c0 = vpadalq_s16(c0, vmull_s8(ahi, vget_high_s8(w0)));
            let w1 = vld1q_s8(r1.add(kk));
            c1 = vpadalq_s16(c1, vmull_s8(alo, vget_low_s8(w1)));
            c1 = vpadalq_s16(c1, vmull_s8(ahi, vget_high_s8(w1)));
            let w2 = vld1q_s8(r2.add(kk));
            c2 = vpadalq_s16(c2, vmull_s8(alo, vget_low_s8(w2)));
            c2 = vpadalq_s16(c2, vmull_s8(ahi, vget_high_s8(w2)));
            let w3 = vld1q_s8(r3.add(kk));
            c3 = vpadalq_s16(c3, vmull_s8(alo, vget_low_s8(w3)));
            c3 = vpadalq_s16(c3, vmull_s8(ahi, vget_high_s8(w3)));
            kk += 16;
        }
        let mut s0 = vaddvq_s32(c0);
        let mut s1 = vaddvq_s32(c1);
        let mut s2 = vaddvq_s32(c2);
        let mut s3 = vaddvq_s32(c3);
        while kk < k {
            let qa = *q.add(kk) as i32;
            s0 += qa * (*r0.add(kk) as i32);
            s1 += qa * (*r1.add(kk) as i32);
            s2 += qa * (*r2.add(kk) as i32);
            s3 += qa * (*r3.add(kk) as i32);
            kk += 1;
        }
        *out.add(j) = (s0 as f32) * (scale_a * *scales.add(j));
        *out.add(j + 1) = (s1 as f32) * (scale_a * *scales.add(j + 1));
        *out.add(j + 2) = (s2 as f32) * (scale_a * *scales.add(j + 2));
        *out.add(j + 3) = (s3 as f32) * (scale_a * *scales.add(j + 3));
        j += 4;
    }
    while j < n {
        let row = bq.add(j * k);
        let mut s = 0i32;
        for kk in 0..k {
            s += (*q.add(kk) as i32) * (*row.add(kk) as i32);
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// wasm SIMD128 inner — the wasm twin of `gemv_i8_omajor_neon`: sequential
/// k-inner loads over contiguous weight rows (full-line use, prefetchable),
/// `i16x8_extend` + `i32x4_dot_i16x8` exact integer accumulation (the
/// widening ladder to f32 and the per-step float rounding are gone). The
/// activation extends amortize over the 4 output rows in flight.
///
/// # Safety
/// simd128 enabled; `q` len `k`, `bq` len `n*k`, `scales` len `n`, `out`
/// len `n`; `k ≤ I8_DOT_K_MAX`. `v128_load` is an unaligned wasm load.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
#[target_feature(enable = "simd128")]
unsafe fn gemv_i8_omajor_wasm(
    q: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::wasm32::*;
    let kv = k & !15;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * k),
            bq.add((j + 1) * k),
            bq.add((j + 2) * k),
            bq.add((j + 3) * k),
        ];
        let mut c = [i32x4_splat(0); 4];
        let mut kk = 0;
        while kk < kv {
            let av = v128_load(q.add(kk) as *const v128);
            let alo = i16x8_extend_low_i8x16(av);
            let ahi = i16x8_extend_high_i8x16(av);
            for (cr, row) in c.iter_mut().zip(rows.iter()) {
                let w = v128_load(row.add(kk) as *const v128);
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(alo, i16x8_extend_low_i8x16(w)));
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(ahi, i16x8_extend_high_i8x16(w)));
            }
            kk += 16;
        }
        let mut s = [0i32; 4];
        for (sr, cr) in s.iter_mut().zip(c.iter()) {
            *sr = i32x4_extract_lane::<0>(*cr)
                + i32x4_extract_lane::<1>(*cr)
                + i32x4_extract_lane::<2>(*cr)
                + i32x4_extract_lane::<3>(*cr);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in s.iter_mut().zip(rows.iter()) {
                *sr += qa * (*row.add(kk) as i32);
            }
            kk += 1;
        }
        for (r, &sr) in s.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = bq.add(j * k);
        let mut s = 0i32;
        for kk in 0..k {
            s += (*q.add(kk) as i32) * (*row.add(kk) as i32);
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// wasm relaxed-SIMD inner — the **same W8A8 function** as
/// `gemv_i8_omajor_wasm`, executed with `i32x4_relaxed_dot_i8x16_i7x16_add`.
/// The signed activation row is split `q = q⁺ − q⁻` with both halves in the
/// i7 range `[0, 127]`, where the relaxed dot is exact and deterministic on
/// every engine; each 16-wide step is then two relaxed dots per row instead
/// of two extends + two dots + two adds, with no activation extends at all.
/// Products stay ≤ 127², so the instruction's internal pairwise i16 sums
/// (≤ 32258) cannot saturate. Exact integer throughout: output remains
/// bit-identical to the baseline and scalar paths — the relaxed tier is an
/// execution speedup of the identical function, not a numeric variant.
///
/// # Safety
/// simd128 + relaxed-simd enabled; `q`/`qp`/`qn` len `k`, `bq` len `n*k`,
/// `scales` len `n`, `out` len `n`; `k ≤ I8_DOT_K_MAX`.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
#[target_feature(enable = "simd128", enable = "relaxed-simd")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i8_omajor_wasm_relaxed(
    q: *const i8,
    qp: *const i8,
    qn: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::wasm32::*;
    #[inline(always)]
    unsafe fn hsum(v: v128) -> i32 {
        i32x4_extract_lane::<0>(v)
            + i32x4_extract_lane::<1>(v)
            + i32x4_extract_lane::<2>(v)
            + i32x4_extract_lane::<3>(v)
    }
    let kv = k & !15;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * k),
            bq.add((j + 1) * k),
            bq.add((j + 2) * k),
            bq.add((j + 3) * k),
        ];
        let mut cp = [i32x4_splat(0); 4];
        let mut cn = [i32x4_splat(0); 4];
        let mut kk = 0;
        while kk < kv {
            let vp = v128_load(qp.add(kk) as *const v128);
            let vn = v128_load(qn.add(kk) as *const v128);
            for r in 0..4 {
                let w = v128_load(rows[r].add(kk) as *const v128);
                cp[r] = i32x4_relaxed_dot_i8x16_i7x16_add(w, vp, cp[r]);
                cn[r] = i32x4_relaxed_dot_i8x16_i7x16_add(w, vn, cn[r]);
            }
            kk += 16;
        }
        let mut s = [0i32; 4];
        for r in 0..4 {
            s[r] = hsum(cp[r]) - hsum(cn[r]);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in s.iter_mut().zip(rows.iter()) {
                *sr += qa * (*row.add(kk) as i32);
            }
            kk += 1;
        }
        for (r, &sr) in s.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = bq.add(j * k);
        let mut acc = 0i32;
        for kk in 0..k {
            acc += (*q.add(kk) as i32) * (*row.add(kk) as i32);
        }
        *out.add(j) = (acc as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// Pool executor: one participant's contiguous output-row range of the
/// omajor W8A8 GEMV, running the identical single-threaded inner this build
/// dispatches — so a partitioned run is bit-identical to the serial run
/// (every output row is computed whole, by one participant, in the same
/// reduction order).
///
/// # Safety
/// Called only from `wasm_pool` fork-join: `args` point into buffers the
/// publisher keeps alive until every participant is done, and participant
/// ranges are disjoint.
#[cfg(all(
    target_arch = "wasm32",
    feature = "wasm-threads",
    target_feature = "simd128"
))]
pub(crate) unsafe fn pool_exec_gemv(
    job: &crate::cpu::wasm_pool::GemvJob,
    part: usize,
    parts: usize,
) {
    use crate::cpu::wasm_pool::GemvOperands;
    let (k, n) = (job.k, job.n);
    let start = part * n / parts;
    let end = (part + 1) * n / parts;
    if start >= end {
        return;
    }
    let rows = end - start;
    let scales = job.scales.add(start);
    let out = job.out.add(start);
    match job.operands {
        // Packed i4: k/2 bytes per output row.
        GemvOperands::I4 { bq, de } => {
            let bq_part = bq.add(start * (k / 2));
            #[cfg(not(target_feature = "relaxed-simd"))]
            gemv_i4_omajor_wasm(job.q, de, bq_part, scales, out, k, rows, job.scale_a);
            #[cfg(target_feature = "relaxed-simd")]
            gemv_i4_omajor_wasm_relaxed(job.q, de, bq_part, scales, out, k, rows, job.scale_a);
        }
        // E8 codebook: k/8 index bytes per output row. One exact-i32 inner on
        // both SIMD tiers (`i32x4_dot_i16x8`).
        GemvOperands::E8cb { bq, codebook } => {
            let bq_part = bq.add(start * (k / 8));
            gemv_e8cb_omajor_wasm(job.q, bq_part, codebook, scales, out, k, rows, job.scale_a);
        }
        // int8: k bytes per output row.
        GemvOperands::I8 { bq, qp, qn } => {
            let bq_part = bq.add(start * k);
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                let _ = (qp, qn); // the plain SIMD128 inner reads `q` directly
                gemv_i8_omajor_wasm(job.q, bq_part, scales, out, k, rows, job.scale_a);
            }
            #[cfg(target_feature = "relaxed-simd")]
            gemv_i8_omajor_wasm_relaxed(job.q, qp, qn, bq_part, scales, out, k, rows, job.scale_a);
        }
    }
}

/// Per-row f32 scratch for the output-major GEMVs' per-token activation scales
/// (zero alloc per call after warm-up under `std`; transient on no_std, like the
/// other kernel scratches).
#[cfg(feature = "std")]
fn with_scale_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    std::thread_local! {
        static SA: core::cell::RefCell<Vec<f32>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    SA.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(not(feature = "std"))]
fn with_scale_scratch<R>(f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    let mut v = Vec::new();
    f(&mut v)
}

/// Output columns per block on the `m > 1` output-major GEMV path.
///
/// A block's weight slab is `cols · k` bytes and is re-read by every one of the
/// `m` rows, so it is sized to stay resident between rows. Without the blocking,
/// a per-row loop re-streams the whole `[n,k]` weight once **per row** — `m`
/// passes instead of one — and the per-row cost never falls below the weight's
/// memory bandwidth, however large `m` grows.
///
/// This only bites once the weight exceeds cache; at 4 MB the GEMV is
/// compute-bound. Measured at `m = 16`: 1.13× at a 4 MB weight, **2.37×** at
/// 32 MB, **2.03×** at 64 MB (`i8_m_scaling`), the blocked path reaching the
/// same ~48 GMAC/s compute ceiling at every weight size.
///
/// Structural (a cache-residency ratio), not model-derived.
const OMAJOR_BLOCK_BYTES: usize = 192 * 1024;

#[inline]
fn omajor_col_block(k: usize, n: usize) -> usize {
    (OMAJOR_BLOCK_BYTES / k.max(1)).clamp(1, n)
}

/// The serial output-major i8 GEMV inner this build dispatches, over a
/// **contiguous column range**: `bq`/`scales`/`out` already point at the first
/// column. No pool — the caller owns the partitioning.
///
/// # Safety
/// `bq` addresses `cols · k` i8, `scales`/`out` address `cols` elements, and `q`
/// (plus `qp`/`qn` on relaxed-SIMD builds) addresses `k` i8.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn gemv_i8_omajor_serial(
    q: *const i8,
    qp: *const i8,
    qn: *const i8,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    cols: usize,
    scale_a: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (qp, qn);
        gemv_i8_omajor_neon(q, bq, scales, out, k, cols, scale_a);
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(target_feature = "relaxed-simd")
    ))]
    {
        let _ = (qp, qn);
        gemv_i8_omajor_wasm(q, bq, scales, out, k, cols, scale_a);
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ))]
    gemv_i8_omajor_wasm_relaxed(q, qp, qn, bq, scales, out, k, cols, scale_a);
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let _ = (qp, qn);
        #[cfg(target_arch = "x86_64")]
        let done = if x86_has_avx2() {
            gemv_i8_omajor_avx2(q, bq, scales, out, k, cols, scale_a);
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = false;
        if !done {
            // Portable scalar: the same exact-i32 dot, one whole column at a
            // time. Not a numerically different tier.
            for j in 0..cols {
                let row = bq.add(j * k);
                let mut acc = 0i32;
                for kk in 0..k {
                    acc += (*q.add(kk) as i32) * (*row.add(kk) as i32);
                }
                *out.add(j) = (acc as f32) * (scale_a * *scales.add(j));
            }
        }
    }
}

/// One column tile of the `m > 1` path: sub-block the tile so its weight slab
/// stays resident, and run **every row** against a block before moving on.
///
/// Each output cell is still one whole dot, computed by the same serial inner
/// over the same `k`-vector in the same order — only the order in which cells
/// are *visited* changes. The accumulation is an exact i32 sum and integer
/// addition is associative, so this cannot move a single bit, let alone a κ.
///
/// # Safety
/// Disjoint output columns per caller; `q_all`/`sa` address `m·k` / `m`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i8_omajor_tile_rows(
    q_all: *const i8,
    qp_all: *const i8,
    qn_all: *const i8,
    sa: *const f32,
    bq: *const i8,
    scales: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    c0: usize,
    tile_cols: usize,
) {
    let cb = omajor_col_block(k, tile_cols);
    let mut c = 0usize;
    while c < tile_cols {
        let w = cb.min(tile_cols - c);
        let col = c0 + c;
        for i in 0..m {
            let scale_a = *sa.add(i);
            if scale_a == 0.0 {
                continue; // the row was zero-filled up front
            }
            gemv_i8_omajor_serial(
                q_all.add(i * k),
                qp_all.add(i * k),
                qn_all.add(i * k),
                bq.add(col * k),
                scales.add(col),
                out.add(i * n + col),
                k,
                w,
                scale_a,
            );
        }
        c += w;
    }
}

/// Fused per-channel symmetric int8 matmul over an **output-major** weight
/// with per-token dynamic activation quantization (W8A8). `a` is `[m,k]`
/// row-major f32, `bq` is `[n,k]` i8 (each output's k-vector contiguous),
/// `scales` `[n]`, `out` `[m,n]`.
///
/// `m = 1` is the decode GEMV this kernel is shaped for and keeps the pooled
/// dispatch (native column-tiles / wasm fork-join). For `m > 1` the output
/// columns are **blocked**, so a block's weight slab is read once and reused by
/// every row, instead of the whole `[n,k]` weight being re-streamed per row.
///
/// Output is **bit-identical** across scalar / x86 AVX2 / NEON / wasm SIMD128,
/// serial or parallel, at every `m`: the accumulation is exact integer, every
/// target shares the same quantization and writeback expressions, and every
/// output cell is one whole dot by one participant. Integer addition is
/// associative and commutative, so no tiling, blocking, or completion order can
/// perturb a bit — unlike the f32 matmul, whose summation order is part of its
/// contract.
pub fn matmul_i8_pc_omajor(
    a: &[f32],
    bq: &[i8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    assert!(
        k <= I8_DOT_K_MAX,
        "matmul_i8_pc_omajor: k {k} exceeds exact-i32 bound {I8_DOT_K_MAX}"
    );
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bq.len(), k * n);
    debug_assert_eq!(scales.len(), n);
    debug_assert!(out.len() >= m * n);

    with_q8_scratch(|q| {
        if m == 1 {
            // Decode. One row, so there is nothing to amortize a weight pass
            // over; keep the pooled dispatch, which partitions the *columns*.
            q.resize(k, 0);
            let scale_a = quantize_row_i8(&a[..k], q);
            let orow = &mut out[..n];
            if scale_a == 0.0 {
                orow.fill(0.0);
                return;
            }
            // Native multi-core: split the output columns across the pool, each
            // participant running the identical serial inner over a disjoint
            // contiguous column range — bit-identical to serial.
            #[cfg(all(
                feature = "parallel",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            {
                use crate::cpu::parallel::{self, SendConst, SendMut};
                let w = parallel::pool().width();
                if w > 1 && (k as u64) * (n as u64) >= GEMV_PAR_THRESHOLD {
                    let tiles = parallel::output_tiles(1, n, w, 4);
                    if tiles.len() > 1 {
                        let (qp, bp, sp, op) = (
                            SendConst(q.as_ptr()),
                            SendConst(bq.as_ptr()),
                            SendConst(scales.as_ptr()),
                            SendMut(orow.as_mut_ptr()),
                        );
                        let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                            .into_iter()
                            .map(|(_, _, c0, cols)| {
                                Box::new(move || {
                                    let (qp, bp, sp, op) = (qp, bp, sp, op);
                                    // SAFETY: disjoint column ranges; q/bq/scales
                                    // shared read-only; sizes checked by caller.
                                    unsafe {
                                        gemv_i8_omajor_native(
                                            qp.0,
                                            bp.0.add(c0 * k),
                                            sp.0.add(c0),
                                            op.0.add(c0),
                                            k,
                                            cols,
                                            scale_a,
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
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                // The pool partitions the GEMV's output columns into disjoint
                // contiguous ranges; every column is a whole dot by one
                // participant, so the result is bit-identical to serial.
                #[cfg(all(feature = "wasm-threads", not(target_feature = "relaxed-simd")))]
                let pooled =
                    crate::cpu::wasm_pool::fork_join_gemv(crate::cpu::wasm_pool::GemvJob {
                        q: q.as_ptr(),
                        scales: scales.as_ptr(),
                        out: orow.as_mut_ptr(),
                        k,
                        n,
                        scale_a,
                        operands: crate::cpu::wasm_pool::GemvOperands::I8 {
                            bq: bq.as_ptr(),
                            qp: core::ptr::null(),
                            qn: core::ptr::null(),
                        },
                    });
                #[cfg(not(all(feature = "wasm-threads", not(target_feature = "relaxed-simd"))))]
                let pooled = false;
                if pooled {
                    return;
                }
            }
            // Relaxed-SIMD build: the inner reads the q⁺/q⁻ i7 split.
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                target_feature = "relaxed-simd"
            ))]
            {
                with_gemv_scratch(|qp, qn| {
                    qp.resize(k, 0);
                    qn.resize(k, 0);
                    split_q7(q, qp, qn);
                    #[cfg(feature = "wasm-threads")]
                    let pooled =
                        crate::cpu::wasm_pool::fork_join_gemv(crate::cpu::wasm_pool::GemvJob {
                            q: q.as_ptr(),
                            scales: scales.as_ptr(),
                            out: orow.as_mut_ptr(),
                            k,
                            n,
                            scale_a,
                            operands: crate::cpu::wasm_pool::GemvOperands::I8 {
                                bq: bq.as_ptr(),
                                qp: qp.as_ptr(),
                                qn: qn.as_ptr(),
                            },
                        });
                    #[cfg(not(feature = "wasm-threads"))]
                    let pooled = false;
                    if !pooled {
                        // SAFETY: simd128 + relaxed-simd gates; sizes checked.
                        unsafe {
                            gemv_i8_omajor_serial(
                                q.as_ptr(),
                                qp.as_ptr(),
                                qn.as_ptr(),
                                bq.as_ptr(),
                                scales.as_ptr(),
                                orow.as_mut_ptr(),
                                k,
                                n,
                                scale_a,
                            );
                        }
                    }
                });
            }
            #[cfg(not(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                target_feature = "relaxed-simd"
            )))]
            // SAFETY: sizes checked above; the serial inner is this build's.
            unsafe {
                gemv_i8_omajor_serial(
                    q.as_ptr(),
                    core::ptr::null(),
                    core::ptr::null(),
                    bq.as_ptr(),
                    scales.as_ptr(),
                    orow.as_mut_ptr(),
                    k,
                    n,
                    scale_a,
                );
            }
            return;
        }

        // ── m > 1 ────────────────────────────────────────────────────────
        // Quantize every row up front, then walk *column blocks* on the
        // outside and rows on the inside, so a block's weight slab is read
        // once and reused by all `m` rows.
        q.resize(m * k, 0);
        with_scale_scratch(|sa| {
            sa.resize(m, 0.0);
            for i in 0..m {
                sa[i] = quantize_row_i8(&a[i * k..(i + 1) * k], &mut q[i * k..(i + 1) * k]);
                if sa[i] == 0.0 {
                    out[i * n..i * n + n].fill(0.0);
                }
            }

            // Relaxed-SIMD reads a q⁺/q⁻ split of every row; build them all
            // once rather than per (block, row).
            #[cfg(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                target_feature = "relaxed-simd"
            ))]
            {
                with_gemv_scratch(|qp, qn| {
                    qp.resize(m * k, 0);
                    qn.resize(m * k, 0);
                    split_q7(q, qp, qn);
                    // SAFETY: sizes checked; single participant.
                    unsafe {
                        gemv_i8_omajor_tile_rows(
                            q.as_ptr(),
                            qp.as_ptr(),
                            qn.as_ptr(),
                            sa.as_ptr(),
                            bq.as_ptr(),
                            scales.as_ptr(),
                            out.as_mut_ptr(),
                            m,
                            k,
                            n,
                            0,
                            n,
                        );
                    }
                });
            }

            #[cfg(not(all(
                target_arch = "wasm32",
                target_feature = "simd128",
                target_feature = "relaxed-simd"
            )))]
            {
                // Native multi-core: the parallel frontier is the **column
                // tile**, and each participant loops all `m` rows inside its
                // tile. One fork/join for the whole call (the per-row dispatch
                // paid `m` of them), and each tile's weight slab is reused
                // across rows. Disjoint output columns ⇒ bit-identical.
                #[cfg(all(
                    feature = "parallel",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                ))]
                {
                    use crate::cpu::parallel::{self, SendConst, SendMut};
                    let w = parallel::pool().width();
                    if w > 1 && (k as u64) * (n as u64) >= GEMV_PAR_THRESHOLD {
                        let tiles = parallel::output_tiles(1, n, w, 4);
                        if tiles.len() > 1 {
                            let (qp, ap, bp, sp, op) = (
                                SendConst(q.as_ptr()),
                                SendConst(sa.as_ptr()),
                                SendConst(bq.as_ptr()),
                                SendConst(scales.as_ptr()),
                                SendMut(out.as_mut_ptr()),
                            );
                            let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                                .into_iter()
                                .map(|(_, _, c0, cols)| {
                                    Box::new(move || {
                                        let (qp, ap, bp, sp, op) = (qp, ap, bp, sp, op);
                                        // SAFETY: disjoint column ranges.
                                        unsafe {
                                            gemv_i8_omajor_tile_rows(
                                                qp.0,
                                                core::ptr::null(),
                                                core::ptr::null(),
                                                ap.0,
                                                bp.0,
                                                sp.0,
                                                op.0,
                                                m,
                                                k,
                                                n,
                                                c0,
                                                cols,
                                            );
                                        }
                                    })
                                        as Box<dyn FnOnce() + Send>
                                })
                                .collect();
                            parallel::pool().run(tasks);
                            return;
                        }
                    }
                }
                // SAFETY: sizes checked above; single participant.
                unsafe {
                    gemv_i8_omajor_tile_rows(
                        q.as_ptr(),
                        core::ptr::null(),
                        core::ptr::null(),
                        sa.as_ptr(),
                        bq.as_ptr(),
                        scales.as_ptr(),
                        out.as_mut_ptr(),
                        m,
                        k,
                        n,
                        0,
                        n,
                    );
                }
            }
        });
    })
}

// ─── Output-major W4A8 int4 GEMV (decode, LUT tier) ────────────────
// The Q0/LUT tier's decode-critical core (plan 077 item 6): the multiply
// against a stored weight is replaced by an in-register 16-entry table
// lookup (`i8x16_swizzle` / `vqtbl1q_s8` — the LUT lives in one SIMD
// register), and the streamed weight bytes HALVE (4-bit nibble indices,
// k/2 bytes per output column). The looked-up i8 values then flow through
// the exact same integer W8A8 dot pipeline as the i8 kernel — extends +
// `i32x4_dot_i16x8` on the baseline tier, `q⁺/q⁻` relaxed dots on the
// relaxed tier, plain integer MACs on scalar — so the output stays
// **bit-identical** across scalar / NEON / wasm on both SIMD tiers. At the
// bandwidth-bound decode regime items 1–5 reached, halving the bytes is the
// step-time lever. Linear i4 today (the table is the fixed i4 value grid);
// the same shape generalizes to per-channel codebooks (non-uniform Q4)
// without touching the dot pipeline.

/// The i4 value grid as a swizzle table: nibble `0..=7 → 0..=7`,
/// `8..=15 → −8..=−1` (two's complement), matching the archive's packed-i4
/// convention (element `l` = nibble `l`, low nibble first).
#[deprecated(
    since = "0.8.0",
    note = "kernel-internal constant; it was never part of a supported surface. Removed in 0.9.0."
)]
pub const I4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

/// Nibble `l` of a packed span (low nibble first — the archive convention).
#[inline]
fn i4_at(packed: &[u8], l: usize) -> i8 {
    let byte = packed[l >> 1];
    let nib = if l & 1 == 0 { byte & 0x0F } else { byte >> 4 };
    I4_VALUES[nib as usize]
}

/// NEON inner: one quantized activation row against the output-major packed
/// i4 weight (`[n, k/2]` bytes), 4 outputs in flight; `vqtbl1q_s8` performs
/// the 16-entry LUT, then the exact-i32 dot pipeline of the i8 kernel.
///
/// # Safety
/// NEON (baseline aarch64); `q` len `k`, `bq` len `n*k/2`, `scales` len
/// `n`, `out` len `n`; `k` even, `k ≤ I8_DOT_K_MAX`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemv_i4_omajor_neon(
    q: *const i8,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::aarch64::*;
    let kb = k / 2; // bytes per column
    let table = vld1q_s8(I4_VALUES.as_ptr());
    let low_mask = vdup_n_u8(0x0F);
    let kv = k & !15;
    let mut j = 0;
    while j + 4 <= n {
        let r0 = bq.add(j * kb);
        let r1 = bq.add((j + 1) * kb);
        let r2 = bq.add((j + 2) * kb);
        let r3 = bq.add((j + 3) * kb);
        let (mut c0, mut c1, mut c2, mut c3) = (
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
            vdupq_n_s32(0),
        );
        let mut kk = 0;
        while kk < kv {
            let av = vld1q_s8(q.add(kk));
            let alo = vget_low_s8(av);
            let ahi = vget_high_s8(av);
            let unpack = |r: *const u8| -> int8x16_t {
                let b = vld1_u8(r.add(kk / 2));
                let lo = vand_u8(b, low_mask);
                let hi = vshr_n_u8::<4>(b);
                let z = vzip_u8(lo, hi);
                vqtbl1q_s8(table, vcombine_u8(z.0, z.1))
            };
            let w0 = unpack(r0);
            c0 = vpadalq_s16(c0, vmull_s8(alo, vget_low_s8(w0)));
            c0 = vpadalq_s16(c0, vmull_s8(ahi, vget_high_s8(w0)));
            let w1 = unpack(r1);
            c1 = vpadalq_s16(c1, vmull_s8(alo, vget_low_s8(w1)));
            c1 = vpadalq_s16(c1, vmull_s8(ahi, vget_high_s8(w1)));
            let w2 = unpack(r2);
            c2 = vpadalq_s16(c2, vmull_s8(alo, vget_low_s8(w2)));
            c2 = vpadalq_s16(c2, vmull_s8(ahi, vget_high_s8(w2)));
            let w3 = unpack(r3);
            c3 = vpadalq_s16(c3, vmull_s8(alo, vget_low_s8(w3)));
            c3 = vpadalq_s16(c3, vmull_s8(ahi, vget_high_s8(w3)));
            kk += 16;
        }
        let mut sums = [
            vaddvq_s32(c0),
            vaddvq_s32(c1),
            vaddvq_s32(c2),
            vaddvq_s32(c3),
        ];
        while kk < k {
            let qa = *q.add(kk) as i32;
            sums[0] += qa * i4_at(core::slice::from_raw_parts(r0, kb), kk) as i32;
            sums[1] += qa * i4_at(core::slice::from_raw_parts(r1, kb), kk) as i32;
            sums[2] += qa * i4_at(core::slice::from_raw_parts(r2, kb), kk) as i32;
            sums[3] += qa * i4_at(core::slice::from_raw_parts(r3, kb), kk) as i32;
            kk += 1;
        }
        for (r, &sv) in sums.iter().enumerate() {
            *out.add(j + r) = (sv as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = core::slice::from_raw_parts(bq.add(j * kb), kb);
        let mut sv = 0i32;
        for kk in 0..k {
            sv += (*q.add(kk) as i32) * i4_at(row, kk) as i32;
        }
        *out.add(j) = (sv as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// wasm SIMD128 inner (baseline tier). The activation row arrives
/// **de-interleaved** (`de = [q_even(k/2) | q_odd(k/2)]`, built once per
/// token and amortized over all `n` rows), so the packed weight needs no
/// lane shuffle: one 16-byte load yields 32 weights — the low nibbles pair
/// with the even activations, the high nibbles with the odd — through two
/// `i8x16_swizzle` LUT hits into the exact `i32x4_dot_i16x8` pipeline.
/// Integer sums are associative, so the pairing order leaves the output
/// bit-identical to the sequential scalar specification.
///
/// # Safety
/// simd128 enabled; `q` len `k` (scalar tail), `de` len `k`, `bq` len
/// `n*k/2`, `scales` len `n`, `out` len `n`; `k` even, `k ≤ I8_DOT_K_MAX`.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i4_omajor_wasm(
    q: *const i8,
    de: *const i8,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::wasm32::*;
    let kb = k / 2;
    let (qe, qo) = (de, de.add(kb));
    let table = v128_load(I4_VALUES.as_ptr() as *const v128);
    let kv = k & !31;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * kb),
            bq.add((j + 1) * kb),
            bq.add((j + 2) * kb),
            bq.add((j + 3) * kb),
        ];
        let mut c = [i32x4_splat(0); 4];
        let mut kk = 0;
        while kk < kv {
            let h = kk / 2; // 16 packed bytes = weights kk..kk+32
            let ae = v128_load(qe.add(h) as *const v128);
            let ao = v128_load(qo.add(h) as *const v128);
            let ae_lo = i16x8_extend_low_i8x16(ae);
            let ae_hi = i16x8_extend_high_i8x16(ae);
            let ao_lo = i16x8_extend_low_i8x16(ao);
            let ao_hi = i16x8_extend_high_i8x16(ao);
            for (cr, row) in c.iter_mut().zip(rows.iter()) {
                let b = v128_load(row.add(h) as *const v128);
                let we = i8x16_swizzle(table, v128_and(b, u8x16_splat(0x0F)));
                let wo = i8x16_swizzle(table, u8x16_shr(b, 4));
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(ae_lo, i16x8_extend_low_i8x16(we)));
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(ae_hi, i16x8_extend_high_i8x16(we)));
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(ao_lo, i16x8_extend_low_i8x16(wo)));
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(ao_hi, i16x8_extend_high_i8x16(wo)));
            }
            kk += 32;
        }
        let mut sums = [0i32; 4];
        for (sr, cr) in sums.iter_mut().zip(c.iter()) {
            *sr = i32x4_extract_lane::<0>(*cr)
                + i32x4_extract_lane::<1>(*cr)
                + i32x4_extract_lane::<2>(*cr)
                + i32x4_extract_lane::<3>(*cr);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in sums.iter_mut().zip(rows.iter()) {
                *sr += qa * i4_at(core::slice::from_raw_parts(*row, kb), kk) as i32;
            }
            kk += 1;
        }
        for (r, &sr) in sums.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = core::slice::from_raw_parts(bq.add(j * kb), kb);
        let mut sv = 0i32;
        for kk in 0..k {
            sv += (*q.add(kk) as i32) * i4_at(row, kk) as i32;
        }
        *out.add(j) = (sv as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// wasm relaxed-SIMD inner — the same function via `q⁺/q⁻` i7-split relaxed
/// dots over the de-interleaved layout
/// (`de = [qe⁺ | qo⁺ | qe⁻ | qo⁻]`, each `k/2`).
///
/// # Safety
/// As `gemv_i4_omajor_wasm`, with `de` len `2k`.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
#[target_feature(enable = "simd128", enable = "relaxed-simd")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i4_omajor_wasm_relaxed(
    q: *const i8,
    de: *const i8,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::wasm32::*;
    #[inline(always)]
    unsafe fn hsum(v: v128) -> i32 {
        i32x4_extract_lane::<0>(v)
            + i32x4_extract_lane::<1>(v)
            + i32x4_extract_lane::<2>(v)
            + i32x4_extract_lane::<3>(v)
    }
    let kb = k / 2;
    let (qep, qop, qen, qon) = (de, de.add(kb), de.add(2 * kb), de.add(3 * kb));
    let table = v128_load(I4_VALUES.as_ptr() as *const v128);
    let kv = k & !31;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * kb),
            bq.add((j + 1) * kb),
            bq.add((j + 2) * kb),
            bq.add((j + 3) * kb),
        ];
        let mut cp = [i32x4_splat(0); 4];
        let mut cn = [i32x4_splat(0); 4];
        let mut kk = 0;
        while kk < kv {
            let h = kk / 2;
            let vep = v128_load(qep.add(h) as *const v128);
            let vop = v128_load(qop.add(h) as *const v128);
            let ven = v128_load(qen.add(h) as *const v128);
            let von = v128_load(qon.add(h) as *const v128);
            for r in 0..4 {
                let b = v128_load(rows[r].add(h) as *const v128);
                let we = i8x16_swizzle(table, v128_and(b, u8x16_splat(0x0F)));
                let wo = i8x16_swizzle(table, u8x16_shr(b, 4));
                cp[r] = i32x4_relaxed_dot_i8x16_i7x16_add(we, vep, cp[r]);
                cp[r] = i32x4_relaxed_dot_i8x16_i7x16_add(wo, vop, cp[r]);
                cn[r] = i32x4_relaxed_dot_i8x16_i7x16_add(we, ven, cn[r]);
                cn[r] = i32x4_relaxed_dot_i8x16_i7x16_add(wo, von, cn[r]);
            }
            kk += 32;
        }
        let mut sums = [0i32; 4];
        for r in 0..4 {
            sums[r] = hsum(cp[r]) - hsum(cn[r]);
        }
        while kk < k {
            let qa = *q.add(kk) as i32;
            for (sr, row) in sums.iter_mut().zip(rows.iter()) {
                *sr += qa * i4_at(core::slice::from_raw_parts(*row, kb), kk) as i32;
            }
            kk += 1;
        }
        for (r, &sr) in sums.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    while j < n {
        let row = core::slice::from_raw_parts(bq.add(j * kb), kb);
        let mut sv = 0i32;
        for kk in 0..k {
            sv += (*q.add(kk) as i32) * i4_at(row, kk) as i32;
        }
        *out.add(j) = (sv as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// Column-blocked `m > 1` driver for the packed-i4 GEMV: a block's packed weight
/// slab is read once and reused by every row, instead of the whole `[n, k/2]`
/// weight being re-streamed per row.
///
/// Byte-neutral: every output cell is one whole dot over the same `k`-vector,
/// visited in a different order. The accumulation is an exact i32 sum and
/// integer addition is associative.
///
/// # Safety
/// `q_all` addresses `m·k` i8; `de_all` addresses `m·ds` i8 (or is null when
/// this build's inner ignores it); `sa` addresses `m` f32; rows with
/// `sa[i] == 0` are pre-zeroed by the caller.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_i4_omajor_blocked(
    q_all: *const i8,
    de_all: *const i8,
    ds: usize,
    sa: *const f32,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let kb = k / 2;
    let cb = omajor_col_block(kb, n);
    let mut c = 0usize;
    while c < n {
        let w = cb.min(n - c);
        for i in 0..m {
            let scale_a = *sa.add(i);
            if scale_a == 0.0 {
                continue; // the row was zero-filled up front
            }
            gemv_i4_omajor_serial(
                q_all.add(i * k),
                if de_all.is_null() {
                    de_all
                } else {
                    de_all.add(i * ds)
                },
                bq.add(c * kb),
                scales.add(c),
                out.add(i * n + c),
                k,
                w,
                scale_a,
            );
        }
        c += w;
    }
}

/// De-interleaved activation stride for the packed-i4 inner, in i8 elements.
/// The baseline tiers read `[q_even | q_odd]` (`k`); the relaxed-SIMD tier reads
/// the non-negative split `[qe⁺ | qo⁺ | qe⁻ | qo⁻]` (`2k`).
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
))]
#[inline]
fn i4_de_stride(k: usize) -> usize {
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ))]
    {
        2 * k
    }
    #[cfg(not(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    )))]
    {
        k
    }
}

/// Build one row's de-interleaved activation. Amortized over all `n` output
/// columns (and, on the `m > 1` path, over every column block), so the packed
/// nibbles need no lane shuffle in the inner loop.
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
))]
#[inline]
fn i4_deinterleave_row(q: &[i8], de: &mut [i8], k: usize) {
    let kb = k / 2;
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ))]
    for (t, pair) in q.chunks_exact(2).enumerate() {
        de[t] = pair[0].max(0);
        de[kb + t] = pair[1].max(0);
        de[2 * kb + t] = (-pair[0]).max(0);
        de[3 * kb + t] = (-pair[1]).max(0);
    }
    #[cfg(not(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    )))]
    for (t, pair) in q.chunks_exact(2).enumerate() {
        de[t] = pair[0];
        de[kb + t] = pair[1];
    }
}

/// Run `f` with the de-interleaved activation for the single row in `q`.
///
/// Only the x86 and wasm inners read it; the NEON inner shuffles nibbles
/// in-register and the scalar fallback indexes them directly, so on those
/// targets there is nothing to build.
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
))]
#[inline]
fn with_i4_de_scratch<R>(q: &[i8], k: usize, f: impl FnOnce(&[i8]) -> R) -> R {
    with_gemv_scratch(|de, _unused| {
        de.resize(i4_de_stride(k), 0);
        i4_deinterleave_row(q, de, k);
        f(de)
    })
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
)))]
#[inline]
fn with_i4_de_scratch<R>(q: &[i8], k: usize, f: impl FnOnce(&[i8]) -> R) -> R {
    let _ = (q, k);
    f(&[])
}

/// The serial output-major packed-i4 GEMV inner this build dispatches, over a
/// **contiguous column range**: `bq` points at the first column's `k/2` packed
/// bytes, `scales`/`out` at that column.
///
/// # Safety
/// `bq` addresses `cols · k/2` bytes; `scales`/`out` address `cols` elements;
/// `q` addresses `k` i8 and `de` `i4_de_stride(k)` i8.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn gemv_i4_omajor_serial(
    q: *const i8,
    de: *const i8,
    bq: *const u8,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    cols: usize,
    scale_a: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = de; // the NEON inner shuffles nibbles in-register
        gemv_i4_omajor_neon(q, bq, scales, out, k, cols, scale_a);
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(target_feature = "relaxed-simd")
    ))]
    gemv_i4_omajor_wasm(q, de, bq, scales, out, k, cols, scale_a);
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ))]
    gemv_i4_omajor_wasm_relaxed(q, de, bq, scales, out, k, cols, scale_a);
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        let done = if x86_has_avx2() {
            gemv_i4_omajor_avx2(q, de, bq, scales, out, k, cols, scale_a);
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = false;
        if !done {
            let _ = de;
            let kb = k / 2;
            for j in 0..cols {
                let row = core::slice::from_raw_parts(bq.add(j * kb), kb);
                let mut sv = 0i32;
                for kk in 0..k {
                    sv += (*q.add(kk) as i32) * i4_at(row, kk) as i32;
                }
                *out.add(j) = (sv as f32) * (scale_a * *scales.add(j));
            }
        }
    }
}

/// Fused per-channel symmetric **int4** matmul over an output-major packed
/// weight with per-token dynamic activation quantization (W4A8). `a` is
/// `[m,k]` row-major f32, `bq` is `[n, k/2]` packed nibbles (element `l` of
/// column `j` = nibble `l` of its `k/2`-byte span, low nibble first),
/// `scales` `[n]`, `out` `[m,n]`. Streams **half** the weight bytes of the
/// i8 kernel; all-integer accumulation keeps it bit-identical across
/// scalar / x86 AVX2 / NEON / wasm on both SIMD tiers. `k` must be even
/// (loud).
pub fn matmul_i4_pc_omajor(
    a: &[f32],
    bq: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    assert!(
        k.is_multiple_of(2),
        "matmul_i4_pc_omajor: k must be even (packed nibbles)"
    );
    assert!(
        k <= I8_DOT_K_MAX,
        "matmul_i4_pc_omajor: k {k} exceeds exact-i32 bound {I8_DOT_K_MAX}"
    );
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bq.len(), k * n / 2);
    debug_assert_eq!(scales.len(), n);
    debug_assert!(out.len() >= m * n);

    with_q8_scratch(|q| {
        if m == 1 {
            // Decode: nothing to amortize a weight pass over. Keep the pooled
            // dispatch, which partitions the output columns.
            q.resize(k, 0);
            let scale_a = quantize_row_i8(&a[..k], q);
            let orow = &mut out[..n];
            if scale_a == 0.0 {
                orow.fill(0.0);
                return;
            }
            with_i4_de_scratch(q, k, |de| {
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    // The pool partitions output columns disjointly.
                    #[cfg(feature = "wasm-threads")]
                    let pooled =
                        crate::cpu::wasm_pool::fork_join_gemv(crate::cpu::wasm_pool::GemvJob {
                            q: q.as_ptr(),
                            scales: scales.as_ptr(),
                            out: orow.as_mut_ptr(),
                            k,
                            n,
                            scale_a,
                            operands: crate::cpu::wasm_pool::GemvOperands::I4 {
                                bq: bq.as_ptr(),
                                de: de.as_ptr(),
                            },
                        });
                    #[cfg(not(feature = "wasm-threads"))]
                    let pooled = false;
                    if pooled {
                        return;
                    }
                }
                // SAFETY: sizes checked above; this build's serial inner.
                unsafe {
                    gemv_i4_omajor_serial(
                        q.as_ptr(),
                        de.as_ptr(),
                        bq.as_ptr(),
                        scales.as_ptr(),
                        orow.as_mut_ptr(),
                        k,
                        n,
                        scale_a,
                    );
                }
            });
            return;
        }

        // ── m > 1 ────────────────────────────────────────────────────────
        // Quantize (and de-interleave) every row up front, then walk column
        // blocks outside and rows inside, so a block's packed weight slab is
        // read once and reused by all `m` rows instead of being re-streamed per
        // row. Byte-neutral: each output cell is still one whole dot over the
        // same k-vector, and the accumulation is an exact i32 sum.
        q.resize(m * k, 0);
        with_scale_scratch(|sa| {
            sa.resize(m, 0.0);
            for i in 0..m {
                sa[i] = quantize_row_i8(&a[i * k..(i + 1) * k], &mut q[i * k..(i + 1) * k]);
                if sa[i] == 0.0 {
                    out[i * n..i * n + n].fill(0.0);
                }
            }
            #[cfg(any(
                all(target_arch = "wasm32", target_feature = "simd128"),
                target_arch = "x86_64"
            ))]
            with_gemv_scratch(|de_all, _unused| {
                let ds = i4_de_stride(k);
                de_all.resize(m * ds, 0);
                for i in 0..m {
                    i4_deinterleave_row(
                        &q[i * k..(i + 1) * k],
                        &mut de_all[i * ds..(i + 1) * ds],
                        k,
                    );
                }
                // SAFETY: disjoint output columns; sizes checked above.
                unsafe {
                    gemv_i4_omajor_blocked(
                        q.as_ptr(),
                        de_all.as_ptr(),
                        ds,
                        sa.as_ptr(),
                        bq.as_ptr(),
                        scales.as_ptr(),
                        out.as_mut_ptr(),
                        m,
                        k,
                        n,
                    );
                }
            });
            #[cfg(not(any(
                all(target_arch = "wasm32", target_feature = "simd128"),
                target_arch = "x86_64"
            )))]
            // SAFETY: disjoint output columns; this build's inner ignores `de`.
            unsafe {
                gemv_i4_omajor_blocked(
                    q.as_ptr(),
                    core::ptr::null(),
                    0,
                    sa.as_ptr(),
                    bq.as_ptr(),
                    scales.as_ptr(),
                    out.as_mut_ptr(),
                    m,
                    k,
                    n,
                );
            }
        });
    })
}

/// Fixed prototype E8 codebook: 256 entries × 8 i8 lattice coordinates.
///
/// **Deprecated.** The codebook is *model* data, not engine data: a deployment
/// flows a per-model learned E8 codebook (QuIP#-style) as a constant operand —
/// the `Dequantize` node's 4th input — and every kernel now takes it as a
/// parameter. Nothing in the engine reads this table. It survives one release
/// so a consumer that referenced it can migrate, per the deprecation lifecycle
/// in `specs/docs/quality-gates.md`; it is removed in 0.9.0.
#[deprecated(
    since = "0.8.0",
    note = "the codebook is per-model operand data: pass it to `matmul_e8cb_omajor`, \
            or as the Dequantize node's 4th input. Removed in 0.9.0."
)]
pub const E8_CODEBOOK: [i8; 256 * 8] = build_e8_codebook();

const fn build_e8_codebook() -> [i8; 256 * 8] {
    let mut cb = [0i8; 256 * 8];
    let mut i = 0;
    while i < 256 * 8 {
        cb[i] = (((i * 37 + 11) % 255) as i32 - 127) as i8;
        i += 1;
    }
    cb
}

// ─── Output-major E8 lattice-codebook GEMV (decode, VQ tier) ───────────
// Vector-quantized weights: each 8-element subvector of a column's k-vector is a
// single codebook index (an E8-lattice point, QuIP#-style). An 8-bit index over
// a 256-entry codebook is **1 bit per logical weight** — 8× fewer streamed bytes
// than i8 — with the codebook (256×8 i8 = 2 KB) staying L1-resident. Indices
// expand through the codebook LUT into i8 weights that flow into the SAME exact
// integer W8A8 dot pipeline (widen → madd, per-column scale writeback), so the
// result is bit-identical to the scalar reference on every target.
//
// The codebook is **the model's**, delivered as a constant operand — which E8
// points a model quantized against is model data, not engine data, so two models
// with different codebooks coexist and each addresses distinctly. The kernel is
// agnostic to its contents; it requires only the full 256-entry index space, so
// any `u8` index dereferences in range without a per-call bounds scan.
//
// `k` must be a multiple of 8 — the group dimension is the E8 lattice's own
// dimension, not a tuning choice. Lanes: x86 AVX2, aarch64 NEON, wasm SIMD128
// (the deployed target), and a portable scalar reference. All four are pinned to
// the same bit-exact conformance sweep, so no first-class target falls to
// scalar.

/// Reused i16 pre-widened codebook scratch (256×8). Widening the i8 codebook
/// to i16 **once per call** lifts the per-column `cvtepi8_epi16` out of the hot
/// loop — the codebook then loads straight as `madd_epi16` operands. O(2048)
/// against the GEMV's O(k·n); the scratch keeps it zero-alloc after warm-up.
#[cfg(feature = "std")]
fn with_cb16_scratch<R>(f: impl FnOnce(&mut Vec<i16>) -> R) -> R {
    std::thread_local! {
        static CB16: core::cell::RefCell<Vec<i16>> = const { core::cell::RefCell::new(Vec::new()) };
    }
    CB16.with(|cell| f(&mut cell.borrow_mut()))
}
#[cfg(not(feature = "std"))]
fn with_cb16_scratch<R>(f: impl FnOnce(&mut Vec<i16>) -> R) -> R {
    let mut v = Vec::new();
    f(&mut v)
}

/// Native serial inner for one quantized activation row against a contiguous
/// output-column range of the E8-codebook weight — the unit both the serial
/// call and each pool task run (bit-identical partition). x86 AVX2 when
/// present, exact scalar otherwise. `cb16` is the codebook pre-widened to i16.
///
/// # Safety
/// `q` len `k`; `bq` is `[n, k/8]` u8 indices (sub-range base); `cb16` len
/// `256*8` i16; `scales`/`out` sub-range bases (`n` cols); `k` multiple of 8,
/// `k ≤ I8_DOT_K_MAX`. Unaligned loads.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_e8cb_omajor_native(
    q: *const i8,
    bq: *const u8,
    cb16: *const i16,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    // wasm SIMD128 is the deployed target — its own inner, not the scalar path.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        gemv_e8cb_omajor_wasm(q, bq, cb16, scales, out, k, n, scale_a);
    }
    // Everything else: x86 AVX2 when detected, aarch64 NEON (baseline), exact
    // scalar otherwise. Guarded off the wasm-SIMD128 build so that branch has no
    // dead scalar tail.
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        #[cfg(target_arch = "x86_64")]
        if x86_has_avx2() {
            gemv_e8cb_omajor_avx2(q, bq, cb16, scales, out, k, n, scale_a);
            return;
        }
        // NEON is baseline on aarch64 — a first-class target must never take the
        // scalar inner.
        #[cfg(target_arch = "aarch64")]
        gemv_e8cb_omajor_neon(q, bq, cb16, scales, out, k, n, scale_a);
        // Portable scalar reference inner (x86-without-AVX2,
        // wasm-without-simd128, other).
        #[cfg(not(target_arch = "aarch64"))]
        {
            let g = k / 8;
            for j in 0..n {
                let row = bq.add(j * g);
                let mut s = 0i32;
                for gg in 0..g {
                    let e = cb16.add(*row.add(gg) as usize * 8);
                    let qb = q.add(gg * 8);
                    for t in 0..8 {
                        s += (*qb.add(t) as i32) * (*e.add(t) as i32);
                    }
                }
                *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
            }
        }
    }
}

/// NEON inner for the E8-codebook GEMV — the aarch64 twin of
/// `gemv_e8cb_omajor_avx2`. One 8-element group per step, 4 output columns in
/// flight: the group's activations widen once (`vmovl_s8`) and each column's
/// codebook entry loads straight as an `int16x8_t` (the codebook arrives
/// pre-widened to i16). `vmlal_s16` accumulates the exact i32 partials — i16×i16
/// products are exact in i32 and integer sums are associative, so the result is
/// **bit-identical** to the scalar / AVX2 / wasm paths.
///
/// # Safety
/// NEON is baseline on aarch64; layouts per `gemv_e8cb_omajor_native`; `k` a
/// multiple of 8, `k ≤ I8_DOT_K_MAX`. Unaligned loads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_e8cb_omajor_neon(
    q: *const i8,
    bq: *const u8,
    cb16: *const i16,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::aarch64::*;
    let g = k / 8;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * g),
            bq.add((j + 1) * g),
            bq.add((j + 2) * g),
            bq.add((j + 3) * g),
        ];
        // Two i32x4 accumulators per column: the low and high halves of the
        // 8-wide product, summed once at the end.
        let mut lo = [vdupq_n_s32(0); 4];
        let mut hi = [vdupq_n_s32(0); 4];
        for gg in 0..g {
            // Widen the group's 8 activations once, shared across the columns.
            let qv = vmovl_s8(vld1_s8(q.add(gg * 8)));
            for r in 0..4 {
                let e = vld1q_s16(cb16.add(*rows[r].add(gg) as usize * 8));
                lo[r] = vmlal_s16(lo[r], vget_low_s16(qv), vget_low_s16(e));
                hi[r] = vmlal_s16(hi[r], vget_high_s16(qv), vget_high_s16(e));
            }
        }
        for r in 0..4 {
            let s = vaddvq_s32(vaddq_s32(lo[r], hi[r]));
            *out.add(j + r) = (s as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    // Column tail (< 4): scalar exact dot over the i16 codebook.
    while j < n {
        let row = bq.add(j * g);
        let mut s = 0i32;
        for gg in 0..g {
            let e = cb16.add(*row.add(gg) as usize * 8);
            let qb = q.add(gg * 8);
            for t in 0..8 {
                s += (*qb.add(t) as i32) * (*e.add(t) as i32);
            }
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// wasm SIMD128 inner for the E8-codebook GEMV — the deployed-target twin of
/// `gemv_e8cb_omajor_avx2`. One 8-D group (8 k) per step, 4 output columns in
/// flight: the group's activations widen once (`v128_load64_zero` +
/// `i16x8_extend_low_i8x16`) and each column's codebook entry loads straight as
/// an `i16x8`; `i32x4_dot_i16x8` is the exact pairwise i32 partial (products
/// ≤ 127² so the instruction's internal i16 sums never saturate), accumulated
/// and horizontally summed — **bit-identical** to the scalar / AVX2 path
/// (integer sums are associative). The codebook arrives pre-widened to i16.
///
/// # Safety
/// simd128 enabled; layouts per `gemv_e8cb_omajor_native`; `k` a multiple of 8,
/// `k ≤ I8_DOT_K_MAX`. `v128_load`/`v128_load64_zero` are unaligned wasm loads.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_e8cb_omajor_wasm(
    q: *const i8,
    bq: *const u8,
    cb16: *const i16,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::wasm32::*;
    let g = k / 8;
    let mut j = 0;
    while j + 4 <= n {
        let rows = [
            bq.add(j * g),
            bq.add((j + 1) * g),
            bq.add((j + 2) * g),
            bq.add((j + 3) * g),
        ];
        let mut c = [i32x4_splat(0); 4];
        for gg in 0..g {
            // Widen the group's 8 activations once, shared across the 4 columns.
            let qv = i16x8_extend_low_i8x16(v128_load64_zero(q.add(gg * 8) as *const u64));
            for (cr, &row) in c.iter_mut().zip(rows.iter()) {
                let e = v128_load(cb16.add(*row.add(gg) as usize * 8) as *const v128);
                *cr = i32x4_add(*cr, i32x4_dot_i16x8(qv, e));
            }
        }
        for (r, &cr) in c.iter().enumerate() {
            let s = i32x4_extract_lane::<0>(cr)
                + i32x4_extract_lane::<1>(cr)
                + i32x4_extract_lane::<2>(cr)
                + i32x4_extract_lane::<3>(cr);
            *out.add(j + r) = (s as f32) * (scale_a * *scales.add(j + r));
        }
        j += 4;
    }
    // Column tail (< 4): scalar exact dot over the i16 codebook.
    while j < n {
        let row = bq.add(j * g);
        let mut s = 0i32;
        for gg in 0..g {
            let e = cb16.add(*row.add(gg) as usize * 8);
            let qb = q.add(gg * 8);
            for t in 0..8 {
                s += (*qb.add(t) as i32) * (*e.add(t) as i32);
            }
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// AVX2 inner for the E8-codebook GEMV: **8 output columns in flight** (to hide
/// the dependent codebook-load latency), two 8-D groups (16 k) per step. The
/// codebook arrives pre-widened to i16, so each index is a direct
/// `_mm_loadu_si128` of 8 i16; two entries compose one `__m256i`
/// (`set_m128i`) and `madd_epi16` against the widened activations — the exact
/// i32 accumulation of the i8 kernel, no per-column widen.
///
/// # Safety
/// AVX2 enabled; layouts per `gemv_e8cb_omajor_native`; `k` multiple of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_e8cb_omajor_avx2(
    q: *const i8,
    bq: *const u8,
    cb16: *const i16,
    scales: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
    scale_a: f32,
) {
    use core::arch::x86_64::*;
    #[inline(always)]
    unsafe fn entry(cb16: *const i16, idx: usize) -> core::arch::x86_64::__m128i {
        _mm_loadu_si128(cb16.add(idx * 8) as *const __m128i) // 8 i16
    }
    let g = k / 8; // 8-D groups per column
    let gv = g & !1; // even group count (16 k per SIMD step)
    let mut j = 0;
    while j + 8 <= n {
        let mut rows = [core::ptr::null::<u8>(); 8];
        for (r, slot) in rows.iter_mut().enumerate() {
            *slot = bq.add((j + r) * g);
        }
        let mut c = [_mm256_setzero_si256(); 8];
        let mut gg = 0;
        while gg < gv {
            let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.add(gg * 8) as *const __m128i));
            for (cr, &row) in c.iter_mut().zip(rows.iter()) {
                // Both group indices in one u16 load (little-endian: low byte =
                // group gg, high byte = group gg+1) — halves the index-load
                // pressure on the gather's critical path.
                let pair = (row.add(gg) as *const u16).read_unaligned();
                let wv = _mm256_set_m128i(
                    entry(cb16, (pair >> 8) as usize),
                    entry(cb16, (pair & 0xff) as usize),
                );
                *cr = _mm256_add_epi32(*cr, _mm256_madd_epi16(av, wv));
            }
            gg += 2;
        }
        let mut s = [0i32; 8];
        for (sr, cr) in s.iter_mut().zip(c.iter()) {
            *sr = hsum256_i32(*cr);
        }
        if gg < g {
            for (sr, &row) in s.iter_mut().zip(rows.iter()) {
                let e = cb16.add(*row.add(gg) as usize * 8);
                let qb = q.add(gg * 8);
                for t in 0..8 {
                    *sr += (*qb.add(t) as i32) * (*e.add(t) as i32);
                }
            }
        }
        for (r, &sr) in s.iter().enumerate() {
            *out.add(j + r) = (sr as f32) * (scale_a * *scales.add(j + r));
        }
        j += 8;
    }
    // Column tail (< 8): scalar exact dot over the i16 codebook.
    while j < n {
        let row = bq.add(j * g);
        let mut s = 0i32;
        for gg in 0..g {
            let e = cb16.add(*row.add(gg) as usize * 8);
            let qb = q.add(gg * 8);
            for t in 0..8 {
                s += (*qb.add(t) as i32) * (*e.add(t) as i32);
            }
        }
        *out.add(j) = (s as f32) * (scale_a * *scales.add(j));
        j += 1;
    }
}

/// Fused E8 lattice-codebook GEMV over an **output-major** weight.
///
/// # This tier is W1A8, not W1A32
///
/// The `a: &[f32]` signature is misleading on its own: this entry point
/// **quantizes the activation internally**, per token, to symmetric i8
/// (`quantize_row_i8`), exactly as the W8A8 int8 decode GEMV does. So the tier
/// is *W1A8* — 1 bit per weight **and** 8-bit activations — and it inherits the
/// full W8A8 activation-rounding error on top of whatever error the model's
/// codebook already carries. See `docs/numerics/w8a8.md`; a caller that needs
/// f32 activations against VQ weights must dequantize and use the f32 matmul.
///
/// # Layout
///
/// `a` is `[m,k]` f32; `bq` is `[n, k/8]` `u8` codebook indices (each output's
/// index-vector contiguous); `codebook` is the **model's** `256×8` i8 codebook
/// (a constant operand, not an engine table); `scales` is `[n]`; `out` is
/// `[m,n]`. `k` must be a multiple of 8 (whole E8 groups).
///
/// The integer dot is exact, so the result is bit-identical across x86 AVX2,
/// aarch64 NEON, wasm SIMD128 and the scalar reference, serial or
/// `--features parallel`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_e8cb_omajor(
    a: &[f32],
    bq: &[u8],
    codebook: &[i8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    const G: usize = DTypeId::E8CB_GROUP_DIM as usize;
    assert!(
        k.is_multiple_of(G),
        "matmul_e8cb_omajor: k must be a multiple of {G} (whole E8 groups)"
    );
    assert!(
        k <= I8_DOT_K_MAX,
        "matmul_e8cb_omajor: k {k} exceeds exact-i32 bound {I8_DOT_K_MAX}"
    );
    // The codebook spans the full `u8` index space, so any stored index
    // dereferences in range without a per-call bounds scan. The caller (the
    // backend's dequant dispatch) enforces this on operands it did not produce.
    assert!(
        codebook.len() == DTypeId::E8CB_MAX_ENTRIES * G,
        "matmul_e8cb_omajor: codebook must be {} entries × {G} coords",
        DTypeId::E8CB_MAX_ENTRIES
    );
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bq.len(), (k / G) * n);
    debug_assert_eq!(scales.len(), n);
    debug_assert!(out.len() >= m * n);

    with_cb16_scratch(|cb16| {
        cb16.clear();
        cb16.extend(codebook.iter().map(|&v| v as i16));
        with_q8_scratch(|q| {
            let g = k / 8;
            if m == 1 {
                // Decode: one row, nothing to amortize a weight pass over.
                q.resize(k, 0);
                let scale_a = quantize_row_i8(&a[..k], q);
                let orow = &mut out[..n];
                if scale_a == 0.0 {
                    orow.fill(0.0);
                    return;
                }
                // wasm multi-core: fork-join the output columns across the
                // embedder's workers, each running the identical serial inner —
                // bit-identical to serial. Falls through when no workers are
                // registered or the job is below the pool's latency floor.
                #[cfg(all(
                    target_arch = "wasm32",
                    target_feature = "simd128",
                    feature = "wasm-threads"
                ))]
                {
                    let pooled =
                        crate::cpu::wasm_pool::fork_join_gemv(crate::cpu::wasm_pool::GemvJob {
                            q: q.as_ptr(),
                            scales: scales.as_ptr(),
                            out: orow.as_mut_ptr(),
                            k,
                            n,
                            scale_a,
                            operands: crate::cpu::wasm_pool::GemvOperands::E8cb {
                                bq: bq.as_ptr(),
                                codebook: cb16.as_ptr(),
                            },
                        });
                    if pooled {
                        return;
                    }
                }
                // Native multi-core: disjoint output-column ranges across the
                // pool, each running the identical serial inner (bit-identical).
                #[cfg(all(
                    feature = "parallel",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                ))]
                {
                    use crate::cpu::parallel::{self, SendConst, SendMut};
                    let w = parallel::pool().width();
                    if w > 1 && (k as u64) * (n as u64) >= GEMV_PAR_THRESHOLD {
                        let tiles = parallel::output_tiles(1, n, w, 8);
                        if tiles.len() > 1 {
                            let (qp, bp, cbp, sp, op) = (
                                SendConst(q.as_ptr()),
                                SendConst(bq.as_ptr()),
                                SendConst(cb16.as_ptr()),
                                SendConst(scales.as_ptr()),
                                SendMut(orow.as_mut_ptr()),
                            );
                            let tasks: Vec<Box<dyn FnOnce() + Send>> = tiles
                                .into_iter()
                                .map(|(_, _, c0, cols)| {
                                    Box::new(move || {
                                        let (qp, bp, cbp, sp, op) = (qp, bp, cbp, sp, op);
                                        // SAFETY: disjoint column ranges; q/bq/
                                        // cb16/scales shared read-only; sizes ok.
                                        unsafe {
                                            gemv_e8cb_omajor_native(
                                                qp.0,
                                                bp.0.add(c0 * g),
                                                cbp.0,
                                                sp.0.add(c0),
                                                op.0.add(c0),
                                                k,
                                                cols,
                                                scale_a,
                                            );
                                        }
                                    })
                                        as Box<dyn FnOnce() + Send>
                                })
                                .collect();
                            parallel::pool().run(tasks);
                            return;
                        }
                    }
                }
                // SAFETY: sizes checked above; per-arch inner selected within.
                unsafe {
                    gemv_e8cb_omajor_native(
                        q.as_ptr(),
                        bq.as_ptr(),
                        cb16.as_ptr(),
                        scales.as_ptr(),
                        orow.as_mut_ptr(),
                        k,
                        n,
                        scale_a,
                    );
                }
                return;
            }

            // ── m > 1 ────────────────────────────────────────────────────
            // Quantize every row up front, then walk column blocks outside and
            // rows inside, so a block's index slab (and the codebook) stay
            // resident across all `m` rows instead of the whole `[n, k/8]`
            // index stream being re-read per row. Byte-neutral: each output
            // cell is still one whole dot over the same k-vector, and the
            // accumulation is an exact i32 sum.
            q.resize(m * k, 0);
            with_scale_scratch(|sa| {
                sa.resize(m, 0.0);
                for i in 0..m {
                    sa[i] = quantize_row_i8(&a[i * k..(i + 1) * k], &mut q[i * k..(i + 1) * k]);
                    if sa[i] == 0.0 {
                        out[i * n..i * n + n].fill(0.0);
                    }
                }
                let cb = omajor_col_block(g, n);
                let mut c = 0usize;
                while c < n {
                    let w = cb.min(n - c);
                    for (i, &scale_a) in sa.iter().enumerate() {
                        if scale_a == 0.0 {
                            continue; // the row was zero-filled up front
                        }
                        // SAFETY: disjoint output columns; sizes checked above.
                        unsafe {
                            gemv_e8cb_omajor_native(
                                q.as_ptr().add(i * k),
                                bq.as_ptr().add(c * g),
                                cb16.as_ptr(),
                                scales.as_ptr().add(c),
                                out.as_mut_ptr().add(i * n + c),
                                k,
                                w,
                                scale_a,
                            );
                        }
                    }
                    c += w;
                }
            });
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // `vec!` via alloc so the suite also builds/runs on no_std targets
    // (wasm32-wasip1 under wasmtime — the deployed-kernel test lane).
    use alloc::vec;

    #[test]
    fn matmul_lowp_gemv_matches_reference() {
        use crate::cpu::dtype::{read_bf16, read_f16, write_bf16, write_f16};
        // Cover both dtypes, the M=1 fast path with its 64/8/scalar column
        // tiers (n crossing 64 and non-multiples), and small M>1.
        for is_f16 in [false, true] {
            for &(m, k, n) in &[
                (1usize, 64usize, 64usize),
                (1, 100, 130), // n = 130 → 64 + 64 + 8 - 6 tail
                (1, 2048, 71), // decode-ish k, odd n
                (1, 7, 5),
                (1, 1, 1),
                (2, 96, 17),
                (4, 34, 80),
            ] {
                let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
                let mut b = vec![0u8; k * n * 2];
                for i in 0..k * n {
                    let v = ((i % 29) as f32 - 14.0) * 0.03;
                    if is_f16 {
                        write_f16(&mut b, i, v);
                    } else {
                        write_bf16(&mut b, i, v);
                    }
                }
                let mut got = vec![0f32; m * n];
                matmul_lowp_gemv(&a, &b, &mut got, m, k, n, is_f16);
                // f64 reference (bf16/f16 → f32 widening is exact).
                for i in 0..m {
                    for j in 0..n {
                        let mut want = 0f64;
                        for kk in 0..k {
                            let bw = if is_f16 {
                                read_f16(&b, kk * n + j)
                            } else {
                                read_bf16(&b, kk * n + j)
                            };
                            want += a[i * k + kk] as f64 * bw as f64;
                        }
                        let denom = want.abs().max(1.0);
                        assert!(
                            (got[i * n + j] as f64 - want).abs() / denom < 1e-4,
                            "is_f16={is_f16} shape=({m},{k},{n}) [{i},{j}] got {} want {want}",
                            got[i * n + j]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn matmul_i8_per_channel_matches_naive() {
        // GEMV (decode) + small + odd + decode-shaped cases.
        for &(m, k, n) in &[
            (1usize, 64usize, 48usize),
            (3, 17, 33),
            (5, 9, 16),
            (1, 2048, 64),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
            let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 0.001).collect();
            let mut got = vec![0f32; m * n];
            matmul_i8_per_channel(&a, &bq, &scales, &mut got, m, k, n);
            for i in 0..m {
                for j in 0..n {
                    let mut want = 0f32;
                    for kk in 0..k {
                        want += a[i * k + kk] * (bq[kk * n + j] as f32);
                    }
                    want *= scales[j];
                    let denom = want.abs().max(1.0);
                    assert!(
                        (got[i * n + j] - want).abs() / denom < 1e-4,
                        "{m}x{k}x{n} ({i},{j}): {} vs {want}",
                        got[i * n + j]
                    );
                }
            }
        }
    }

    /// Multi-threaded fork-join vs serial, bit for bit — run on
    /// `wasm32-wasip1-threads` under wasmtime (`-S threads`), where std
    /// threads drive the exact atomics job queue the browser's web workers
    /// will. Proves the static row partition preserves the per-output
    /// reduction order (structural determinism), not just approximate
    /// results.
    #[test]
    #[cfg(all(
        feature = "std",
        feature = "wasm-threads",
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    fn parallel_gemv_matches_serial_bitwise() {
        use crate::cpu::wasm_pool;
        // Above the pool's latency floor (k·n = 512 KiB ≥ 256 KiB), with
        // n not a multiple of the participant count.
        let (m, k, n) = (1usize, 512usize, 1027usize);
        let a: Vec<f32> = (0..k).map(|i| ((i % 31) as f32 - 15.0) * 0.041).collect();
        let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 253) - 126) as i8).collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.004 + (j as f32) * 1e-6).collect();

        // Serial baselines: no workers registered yet.
        let mut serial = vec![0f32; n];
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut serial, m, k, n);
        let bq4: Vec<u8> = (0..k * n / 2).map(|i| (i % 247) as u8).collect();
        let mut serial4 = vec![0f32; n];
        matmul_i4_pc_omajor(&a, &bq4, &scales, &mut serial4, m, k, n);

        let handles: Vec<_> = (0..3u32)
            .map(|i| std::thread::spawn(move || wasm_pool::hologram_worker_run(i)))
            .collect();
        while wasm_pool::hologram_pool_workers() < 3 {
            std::thread::yield_now();
        }

        let mut par = vec![f32::NAN; n];
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut par, m, k, n);
        for (j, (p, sv)) in par.iter().zip(&serial).enumerate() {
            assert_eq!(
                p.to_bits(),
                sv.to_bits(),
                "i8 output {j}: parallel {p} vs serial {sv}"
            );
        }
        let mut par4 = vec![f32::NAN; n];
        matmul_i4_pc_omajor(&a, &bq4, &scales, &mut par4, m, k, n);
        for (j, (p, sv)) in par4.iter().zip(&serial4).enumerate() {
            assert_eq!(
                p.to_bits(),
                sv.to_bits(),
                "i4 output {j}: parallel {p} vs serial {sv}"
            );
        }

        wasm_pool::hologram_pool_shutdown();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn exp_det_simd_matches_scalar_spec_bitwise() {
        // Dense grid + edge cases; the in-place (SIMD) form must equal the
        // scalar specification bit-for-bit on every target this compiles
        // for — the determinism contract of the decode softmax's exp.
        let mut xs: Vec<f32> = Vec::new();
        let mut v = -90.0f32;
        while v <= 90.0 {
            xs.push(v);
            v += 0.037;
        }
        xs.extend_from_slice(&[
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            0.0,
            -0.0,
            EXP_F32_LO,
            EXP_F32_HI,
            -87.3,
            88.9,
            1e-8,
            -1e-8,
        ]);
        let want: Vec<f32> = xs.iter().map(|&x| exp_f32_det(x)).collect();
        let mut got = xs.clone();
        simd_f32_exp_inplace(&mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "x = {} (idx {i}): {g} vs {w}",
                xs[i]
            );
        }
    }

    #[test]
    fn exp_det_accuracy_and_edges() {
        // Accuracy against libm over the working range, plus the exact
        // edge semantics: exp(0) = 1 exactly, underflow/−∞/NaN → 0.0.
        let mut v = -87.0f32;
        let mut max_rel = 0f64;
        while v <= 88.0 {
            let e = f64::from(exp_f32_det(v));
            let r = f64::from(libm::expf(v));
            if r > 0.0 {
                max_rel = max_rel.max(((e - r) / r).abs());
            }
            v += 0.0173;
        }
        assert!(max_rel < 2e-6, "max rel err {max_rel:.3e}");
        assert_eq!(exp_f32_det(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp_f32_det(f32::NEG_INFINITY).to_bits(), 0.0f32.to_bits());
        assert_eq!(exp_f32_det(f32::NAN).to_bits(), 0.0f32.to_bits());
        assert_eq!(exp_f32_det(-1000.0).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn matmul_i4_pc_omajor_matches_integer_reference() {
        // The W4A8 LUT-tier kernel against an independent integer
        // reference (nibble → value grid → exact i32 dot → one fused
        // writeback): bit equality on every target this compiles for,
        // including the k-tail (k ≡ 2 mod 16) and multi-row m.
        for &(m, k, n) in &[
            (1usize, 64usize, 48usize),
            (1, 2048, 64),
            (1, 130, 33),
            (1, 18, 3),
            (2, 96, 17),
            (4, 34, 5),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.37)
                .collect();
            let bq: Vec<u8> = (0..k * n / 2).map(|i| (i % 251) as u8).collect();
            let scales: Vec<f32> = (0..n).map(|j| 0.02 + (j as f32) * 0.001).collect();
            let mut got = vec![0f32; m * n];
            matmul_i4_pc_omajor(&a, &bq, &scales, &mut got, m, k, n);
            let kb = k / 2;
            for i in 0..m {
                let row = &a[i * k..(i + 1) * k];
                let mut amax = 0f32;
                for &v in row {
                    amax = amax.max(v.abs());
                }
                let inv = 127.0 / amax;
                let scale_a = amax / 127.0;
                let q: Vec<i32> = row
                    .iter()
                    .map(|&v| {
                        let t = v * inv;
                        let r = if t >= 0.0 {
                            (t + 0.5) as i32
                        } else {
                            (t - 0.5) as i32
                        };
                        r.clamp(-127, 127)
                    })
                    .collect();
                for j in 0..n {
                    let col = &bq[j * kb..(j + 1) * kb];
                    let mut sv = 0i32;
                    for (kk, &qa) in q.iter().enumerate() {
                        let byte = col[kk >> 1];
                        let nib = if kk & 1 == 0 { byte & 0x0F } else { byte >> 4 };
                        let w = if nib < 8 { nib as i32 } else { nib as i32 - 16 };
                        sv += qa * w;
                    }
                    let want = (sv as f32) * (scale_a * scales[j]);
                    assert_eq!(
                        got[i * n + j].to_bits(),
                        want.to_bits(),
                        "{m}x{k}x{n} ({i},{j}): {} vs {want}",
                        got[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_i8_pc_omajor_matches_integer_reference() {
        // Decode GEMV (m = 1, various k/n incl. non-multiples of 16/4),
        // small-m, and tiny edge shapes. The reference restates the W8A8
        // spec independently (amax → inv → trunc-round → exact i32 dot →
        // one fused writeback) and the comparison is **bit equality** —
        // the kernel's integer accumulation must make SIMD order
        // invisible on every target this test compiles for.
        for &(m, k, n) in &[
            (1usize, 64usize, 48usize),
            (1, 2048, 64),
            (1, 129, 33),
            (1, 16, 3),
            (1, 7, 1),
            (2, 100, 17),
            (4, 31, 5),
            // Above GEMV_PAR_THRESHOLD (k·n ≥ 1<<20): exercises the native
            // column-partitioned pool path when built `--features parallel`.
            // `640` splits evenly on the 4-wide panel; `653` forces a ragged
            // final tile so the partition boundary is tested off-panel.
            (1, 2048, 640),
            (1, 2048, 653),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.37)
                .collect();
            let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
            let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 0.001).collect();
            let mut got = vec![0f32; m * n];
            matmul_i8_pc_omajor(&a, &bq, &scales, &mut got, m, k, n);
            for i in 0..m {
                let row = &a[i * k..(i + 1) * k];
                let mut amax = 0f32;
                for &v in row {
                    amax = amax.max(v.abs());
                }
                if amax == 0.0 {
                    for j in 0..n {
                        assert_eq!(got[i * n + j].to_bits(), 0f32.to_bits());
                    }
                    continue;
                }
                let inv = 127.0 / amax;
                let scale_a = amax / 127.0;
                let q: Vec<i32> = row
                    .iter()
                    .map(|&v| {
                        let t = v * inv;
                        let r = if t >= 0.0 {
                            (t + 0.5) as i32
                        } else {
                            (t - 0.5) as i32
                        };
                        r.clamp(-127, 127)
                    })
                    .collect();
                for j in 0..n {
                    let mut s = 0i32;
                    for kk in 0..k {
                        s += q[kk] * bq[j * k + kk] as i32;
                    }
                    let want = (s as f32) * (scale_a * scales[j]);
                    assert_eq!(
                        got[i * n + j].to_bits(),
                        want.to_bits(),
                        "{m}x{k}x{n} ({i},{j}): {} vs {want}",
                        got[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_i8_pc_omajor_zero_row_is_zero() {
        let a = vec![0f32; 2 * 40];
        let bq: Vec<i8> = (0..40 * 6).map(|i| (i % 100) as i8).collect();
        let scales = vec![0.5f32; 6];
        let mut out = vec![f32::NAN; 2 * 6];
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, 2, 40, 6);
        for v in out {
            assert_eq!(v.to_bits(), 0f32.to_bits());
        }
    }

    #[test]
    fn matmul_e8cb_omajor_matches_integer_reference() {
        // E8-codebook GEMV: the AVX2 / scalar inner (and, under `--features
        // parallel`, the column-partitioned pool path) must equal an
        // independent scalar restatement of the spec — amax→inv→trunc-round
        // quant, index→codebook LUT, exact i32 dot, one fused writeback — with
        // **bit** equality. Shapes cover the 2-group SIMD body, an odd-group
        // tail (k/8 odd), the n<4 scalar tail, small m>1, and one shape above
        // GEMV_PAR_THRESHOLD (k·n ≥ 1<<20) incl. a ragged non-panel n.
        let codebook: Vec<i8> = (0..256 * 8)
            .map(|i| ((i * 37 + 11) % 255 - 127) as i8)
            .collect();
        for &(m, k, n) in &[
            (1usize, 64usize, 48usize),
            (1, 2048, 64),
            (1, 8, 3),   // one group, n<4 tail
            (1, 24, 5),  // odd group count (3), n tail
            (2, 96, 17), // small m>1
            (1, 2048, 640),
            (1, 2048, 653), // > threshold, ragged final tile
        ] {
            let g = k / 8;
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.31)
                .collect();
            let bq: Vec<u8> = (0..g * n).map(|i| ((i * 53 + 7) % 256) as u8).collect();
            let scales: Vec<f32> = (0..n).map(|j| 0.02 + (j as f32) * 0.0007).collect();
            let mut got = vec![0f32; m * n];
            matmul_e8cb_omajor(&a, &bq, &codebook, &scales, &mut got, m, k, n);
            for i in 0..m {
                let row = &a[i * k..(i + 1) * k];
                let mut amax = 0f32;
                for &v in row {
                    amax = amax.max(v.abs());
                }
                if amax == 0.0 {
                    for j in 0..n {
                        assert_eq!(got[i * n + j].to_bits(), 0f32.to_bits());
                    }
                    continue;
                }
                let inv = 127.0 / amax;
                let scale_a = amax / 127.0;
                let q: Vec<i32> = row
                    .iter()
                    .map(|&v| {
                        let t = v * inv;
                        let r = if t >= 0.0 {
                            (t + 0.5) as i32
                        } else {
                            (t - 0.5) as i32
                        };
                        r.clamp(-127, 127)
                    })
                    .collect();
                for j in 0..n {
                    let mut s = 0i32;
                    for gg in 0..g {
                        let idx = bq[j * g + gg] as usize;
                        for t in 0..8 {
                            s += q[gg * 8 + t] * codebook[idx * 8 + t] as i32;
                        }
                    }
                    let want = (s as f32) * (scale_a * scales[j]);
                    assert_eq!(
                        got[i * n + j].to_bits(),
                        want.to_bits(),
                        "{m}x{k}x{n} ({i},{j})"
                    );
                }
            }
        }
    }

    /// Independent scalar restatement of the E8-codebook GEMV spec, the
    /// bit-exact target for the kernel (any arch / `--features parallel`).
    fn e8cb_ref(
        a: &[f32],
        bq: &[u8],
        codebook: &[i8],
        scales: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let g = k / 8;
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            let row = &a[i * k..(i + 1) * k];
            let amax = row.iter().fold(0f32, |mx, &v| mx.max(v.abs()));
            if amax == 0.0 {
                continue; // zero row → zero output
            }
            let inv = 127.0 / amax;
            let scale_a = amax / 127.0;
            let q: Vec<i32> = row
                .iter()
                .map(|&v| {
                    let t = v * inv;
                    let r = if t >= 0.0 {
                        (t + 0.5) as i32
                    } else {
                        (t - 0.5) as i32
                    };
                    r.clamp(-127, 127)
                })
                .collect();
            for j in 0..n {
                let mut s = 0i32;
                for gg in 0..g {
                    let idx = bq[j * g + gg] as usize;
                    for t in 0..8 {
                        s += q[gg * 8 + t] * codebook[idx * 8 + t] as i32;
                    }
                }
                out[i * n + j] = (s as f32) * (scale_a * scales[j]);
            }
        }
        out
    }

    /// `widen_lowp_to_f32` must be **bit-identical** to element-wise
    /// `read_bf16`/`read_f16` on every arch. It is the prefill path (it widens
    /// the whole k×n weight before the blocked f32 matmul), so it has a SIMD
    /// lane on x86 AVX2, aarch64 NEON and wasm SIMD128; all must agree with the
    /// scalar spec, including the sub-chunk tail.
    #[test]
    fn widen_lowp_matches_elementwise_reference() {
        use crate::cpu::dtype::{read_bf16, read_f16, write_bf16, write_f16};
        // Lengths spanning the 4-wide (NEON/wasm) and 8-wide (AVX2) chunk
        // widths and every tail remainder.
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 64, 129] {
            for is_f16 in [false, true] {
                let mut src = vec![0u8; len * 2];
                for i in 0..len {
                    // A spread of magnitudes, signs, and exact halves.
                    let v = (i as f32 - (len as f32) / 2.0) * 0.375;
                    if is_f16 {
                        write_f16(&mut src, i, v);
                    } else {
                        write_bf16(&mut src, i, v);
                    }
                }
                let mut got = vec![0f32; len];
                widen_lowp_to_f32(&src, &mut got, is_f16);
                for (i, &g) in got.iter().enumerate() {
                    let want = if is_f16 {
                        read_f16(&src, i)
                    } else {
                        read_bf16(&src, i)
                    };
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "len={len} is_f16={is_f16} elem {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_e8cb_omajor_random_sweep() {
        // Deterministic xorshift64 — any failure reproduces from the seed.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let codebook: Vec<i8> = (0..256 * 8).map(|_| next() as i8).collect();
        for _ in 0..96 {
            let m = 1 + (next() % 4) as usize; // 1..=4 (decode + small prefill)
            let g = 1 + (next() % 48) as usize; // 1..=48 groups
            let k = g * 8;
            let n = 1 + (next() % 96) as usize; // spans the 8-col body + <8 tail
            let a: Vec<f32> = (0..m * k)
                .map(|_| (next() % 4001) as f32 * 5e-4 - 1.0)
                .collect();
            let bq: Vec<u8> = (0..g * n).map(|_| next() as u8).collect();
            let scales: Vec<f32> = (0..n)
                .map(|_| 1e-3 + (next() % 1000) as f32 * 1e-5)
                .collect();
            let mut got = vec![0f32; m * n];
            matmul_e8cb_omajor(&a, &bq, &codebook, &scales, &mut got, m, k, n);
            let want = e8cb_ref(&a, &bq, &codebook, &scales, m, k, n);
            for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    gv.to_bits(),
                    wv.to_bits(),
                    "m={m} k={k} n={n} lane {idx}: {gv} vs {wv}"
                );
            }
        }
    }

    #[test]
    fn matmul_e8cb_omajor_zero_row_is_zero() {
        let codebook: Vec<i8> = (0..256 * 8).map(|i| (i % 200 - 100) as i8).collect();
        let (m, k, n) = (2usize, 32usize, 20usize);
        let a = vec![0f32; m * k];
        let bq: Vec<u8> = (0..(k / 8) * n).map(|i| (i * 7) as u8).collect();
        let scales = vec![0.5f32; n];
        let mut out = vec![f32::NAN; m * n];
        matmul_e8cb_omajor(&a, &bq, &codebook, &scales, &mut out, m, k, n);
        for v in out {
            assert_eq!(v.to_bits(), 0f32.to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "multiple of 8")]
    fn matmul_e8cb_omajor_rejects_k_not_multiple_of_8() {
        let codebook = vec![0i8; 256 * 8];
        let (m, k, n) = (1usize, 12usize, 4usize); // k=12: not a whole E8 group
        let a = vec![1.0f32; m * k];
        let bq = vec![0u8; n];
        let scales = vec![1.0f32; n];
        let mut out = vec![0f32; m * n];
        matmul_e8cb_omajor(&a, &bq, &codebook, &scales, &mut out, m, k, n);
    }

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
    fn sum_matches_scalar() {
        // Cover lengths spanning the SIMD block widths (8/16/64) and their
        // tails, so both the vector body and the scalar remainder run.
        for len in [0usize, 1, 3, 7, 8, 15, 16, 17, 63, 64, 65, 100, 257] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 3.0).collect();
            let want: f32 = a.iter().sum();
            let got = simd_f32_sum(&a);
            assert!(
                (want - got).abs() <= 1e-3 * want.abs().max(1.0),
                "len {len}: want {want}, got {got}"
            );
        }
    }

    #[test]
    fn max_matches_scalar() {
        assert_eq!(simd_f32_max(&[]), f32::NEG_INFINITY);
        for len in [1usize, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 100, 257] {
            // Interleave sign so the running max is not monotone.
            let a: Vec<f32> = (0..len)
                .map(|i| if i % 3 == 0 { -(i as f32) } else { i as f32 })
                .collect();
            let want = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got = simd_f32_max(&a);
            assert_eq!(want, got, "len {len}");
        }
    }

    #[test]
    fn min_matches_scalar() {
        assert_eq!(simd_f32_min(&[]), f32::INFINITY);
        for len in [1usize, 3, 4, 7, 8, 15, 16, 17, 63, 64, 65, 100, 257] {
            let a: Vec<f32> = (0..len)
                .map(|i| if i % 3 == 0 { -(i as f32) } else { i as f32 })
                .collect();
            let want = a.iter().copied().fold(f32::INFINITY, f32::min);
            let got = simd_f32_min(&a);
            assert_eq!(want, got, "len {len}");
        }
    }

    #[test]
    fn axpy_matches_scalar_bit_identical() {
        // Non-fused SIMD axpy has no cross-lane reduction, so it must match the
        // scalar `out += s*b` exactly (bit-for-bit), not merely within epsilon.
        for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 64, 100] {
            let s = 0.37f32;
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.011 - 0.4).collect();
            let base: Vec<f32> = (0..len).map(|i| (i as f32) * 0.003 + 1.0).collect();
            let mut got = base.clone();
            simd_f32_axpy(&mut got, s, &b);
            let mut want = base.clone();
            for i in 0..len {
                want[i] += s * b[i];
            }
            assert_eq!(got, want, "len {len}");
        }
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

    /// An output row's **bytes** must not depend on where the row sits in the
    /// call: two identical input rows must produce byte-identical output rows,
    /// whether one lands in the `MR` register tile and the other in the row
    /// remainder.
    ///
    /// This is a content-addressing invariant, not a numerical nicety. f32
    /// result bytes are hashed into κ, so a cell whose arithmetic depends on
    /// its row index makes the same logical matmul yield two different
    /// addresses. It regressed for a decade of shapes because the tile's column
    /// tail was scalar (`s += a*b`, two roundings) while the row remainder's
    /// was a fused-multiply-add tier (one rounding) — the two disagreed for
    /// every column in `n mod 16 >= 8` on AVX2, and `>= 4` on NEON.
    ///
    /// `n` values below straddle both tiers; `m = 5..7` puts rows 0..3 in the
    /// tile and rows 4.. in the remainder. Compares row 0 against row 4, which
    /// are seeded identically.
    /// **Schedule-independence of the integer GEMVs.** The `m > 1` path of every
    /// tier (i8 / packed-i4 / E8CB) blocks
    /// the output columns so a weight slab is read once and reused by every row,
    /// instead of the whole `[n,k]` weight being re-streamed per row. Visiting
    /// the output cells in a different order must not change one bit: each cell
    /// is still one whole dot over the same `k`-vector, the accumulation is an
    /// exact i32 sum, and integer addition is associative.
    ///
    /// So a batched call must equal `m` independent single-row calls, byte for
    /// byte. Shapes straddle the column-block boundary and include a
    /// zero-activation row (`scale_a == 0`), which takes the early-out.
    #[test]
    fn batched_integer_gemv_equals_row_by_row_bit_for_bit() {
        for &(k, n) in &[(16usize, 5usize), (64, 129), (8, 1), (128, 64)] {
            for &m in &[1usize, 2, 3, 5, 8] {
                let bq: Vec<i8> = (0..k * n)
                    .map(|i| (((i * 31 + 7) % 255) as i32 - 127) as i8)
                    .collect();
                let scales: Vec<f32> = (0..n).map(|j| 0.007 + j as f32 * 0.0003).collect();
                let a: Vec<f32> = (0..m * k)
                    .map(|i| {
                        // Row 2 (when present) is all-zero: exercises scale_a == 0.
                        if m > 2 && i / k == 2 {
                            0.0
                        } else {
                            ((i % 37) as f32 - 18.0) * 0.031
                        }
                    })
                    .collect();

                let mut batched = vec![0f32; m * n];
                matmul_i8_pc_omajor(&a, &bq, &scales, &mut batched, m, k, n);

                for i in 0..m {
                    let mut single = vec![0f32; n];
                    matmul_i8_pc_omajor(&a[i * k..(i + 1) * k], &bq, &scales, &mut single, 1, k, n);
                    for j in 0..n {
                        assert_eq!(
                            batched[i * n + j].to_bits(),
                            single[j].to_bits(),
                            "i8 m={m} k={k} n={n} row {i} col {j}: batched {} vs single-row {}",
                            batched[i * n + j],
                            single[j]
                        );
                    }
                }

                // Packed i4 (W4A8): `k/2` bytes per output column.
                let bq4: Vec<u8> = (0..k * n / 2).map(|i| ((i * 53 + 9) % 251) as u8).collect();
                let mut b4 = vec![0f32; m * n];
                matmul_i4_pc_omajor(&a, &bq4, &scales, &mut b4, m, k, n);
                for i in 0..m {
                    let mut single = vec![0f32; n];
                    matmul_i4_pc_omajor(
                        &a[i * k..(i + 1) * k],
                        &bq4,
                        &scales,
                        &mut single,
                        1,
                        k,
                        n,
                    );
                    for j in 0..n {
                        assert_eq!(
                            b4[i * n + j].to_bits(),
                            single[j].to_bits(),
                            "i4 m={m} k={k} n={n} row {i} col {j}"
                        );
                    }
                }

                // E8CB (W1A8): `k/8` index bytes per column; `k` must be a
                // whole number of 8-D groups.
                if k.is_multiple_of(8) {
                    let cb: Vec<i8> = (0..256 * 8)
                        .map(|i| ((i * 37 + 11) % 255 - 127) as i8)
                        .collect();
                    let bq8: Vec<u8> = (0..n * (k / 8))
                        .map(|i| ((i * 17 + 3) % 256) as u8)
                        .collect();
                    let mut b8 = vec![0f32; m * n];
                    matmul_e8cb_omajor(&a, &bq8, &cb, &scales, &mut b8, m, k, n);
                    for i in 0..m {
                        let mut single = vec![0f32; n];
                        matmul_e8cb_omajor(
                            &a[i * k..(i + 1) * k],
                            &bq8,
                            &cb,
                            &scales,
                            &mut single,
                            1,
                            k,
                            n,
                        );
                        for j in 0..n {
                            assert_eq!(
                                b8[i * n + j].to_bits(),
                                single[j].to_bits(),
                                "e8cb m={m} k={k} n={n} row {i} col {j}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// **Codec-invariance.** A weight tier is a *codec*: a decode `d : Code → 𝔽`
    /// from a stored alphabet into the working alphabet `{-127..=127}`. MatMul is
    /// the exact integer accumulation `Σ aᵢ · d(cᵢ)`. So if two codecs decode to
    /// the *same* weight sequence, the accumulation must be the same integer —
    /// the result is a function of the decoded operands alone, never of the codec
    /// that produced them, nor of the tier's bit-width.
    ///
    /// The tier is therefore a residency/bandwidth choice; the arithmetic
    /// identity is fixed. This test is the runtime witness: i8 (identity codec),
    /// packed-i4 (nibble → 16-entry grid) and E8CB (index → 8-D codebook block)
    /// are given three factorizations of one weight matrix, and must return
    /// **byte-identical** f32.
    ///
    /// All three quantize the activation identically (`quantize_row_i8`), so the
    /// only thing varying is the weight codec. Nothing weaker than bit-equality
    /// is asserted: the accumulation is exact i32, so there is no rounding step
    /// at which two codecs could legitimately diverge.
    #[test]
    fn tiers_that_decode_to_the_same_weights_agree_bit_for_bit() {
        // i4 spans `{-8..=7}`; pick weights in that range so one matrix is
        // exactly representable in all three codecs.
        let (m, k, n) = (1usize, 16usize, 5usize); // k = 2 whole E8 groups
        let w = |kk: usize, j: usize| -> i8 { (((kk * 7 + j * 3) % 16) as i32 - 8) as i8 };
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.043)
            .collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.011 + j as f32 * 0.0007).collect();

        // Codec 1 — i8: the identity codec. `[n, k]`, each output's k-vector
        // contiguous.
        let bq_i8: Vec<i8> = (0..n)
            .flat_map(|j| (0..k).map(move |kk| w(kk, j)))
            .collect();

        // Codec 2 — packed i4: a nibble indexes the 16-entry grid `I4_VALUES`.
        // `[n, k/2]`, low nibble first (see `i4_at`).
        let nib = |v: i8| -> u8 { I4_VALUES.iter().position(|&x| x == v).unwrap() as u8 };
        let bq_i4: Vec<u8> = (0..n)
            .flat_map(|j| (0..k / 2).map(move |b| nib(w(2 * b, j)) | (nib(w(2 * b + 1, j)) << 4)))
            .collect();

        // Codec 3 — E8CB: one index decodes an 8-weight block. `[n, k/8]`
        // indices into a `256×8` codebook. Give each (column, group) its own
        // entry, which is exactly what a per-model learned codebook does.
        let groups = k / 8;
        let mut codebook = vec![0i8; 256 * 8];
        let mut bq_e8 = vec![0u8; n * groups];
        for j in 0..n {
            for g in 0..groups {
                let idx = j * groups + g;
                assert!(idx < 256, "test shape must fit the 256-entry codebook");
                bq_e8[j * groups + g] = idx as u8;
                for t in 0..8 {
                    codebook[idx * 8 + t] = w(g * 8 + t, j);
                }
            }
        }

        let mut out_i8 = vec![0f32; m * n];
        let mut out_i4 = vec![0f32; m * n];
        let mut out_e8 = vec![0f32; m * n];
        matmul_i8_pc_omajor(&a, &bq_i8, &scales, &mut out_i8, m, k, n);
        matmul_i4_pc_omajor(&a, &bq_i4, &scales, &mut out_i4, m, k, n);
        matmul_e8cb_omajor(&a, &bq_e8, &codebook, &scales, &mut out_e8, m, k, n);

        for j in 0..n {
            assert_eq!(
                out_i8[j].to_bits(),
                out_i4[j].to_bits(),
                "col {j}: i8 and packed-i4 decode to the same weights but disagree \
                 ({} vs {})",
                out_i8[j],
                out_i4[j]
            );
            assert_eq!(
                out_i8[j].to_bits(),
                out_e8[j].to_bits(),
                "col {j}: i8 and E8CB decode to the same weights but disagree \
                 ({} vs {})",
                out_i8[j],
                out_e8[j]
            );
        }
    }

    #[test]
    fn matmul_row_bytes_are_independent_of_row_index() {
        for &n in &[16usize, 20, 23, 24, 28, 31, 32, 40, 64] {
            for &k in &[1usize, 3, 7, 33, 65] {
                for &m in &[5usize, 6, 7] {
                    let mut a = vec![0f32; m * k];
                    for kk in 0..k {
                        // Row 0 (tiled) and row 4 (remainder) are identical.
                        let v = ((kk * 37 + 11) % 97) as f32 * 0.0317 - 1.3;
                        a[kk] = v;
                        a[4 * k + kk] = v;
                    }
                    for r in (1..4).chain(5..m) {
                        for kk in 0..k {
                            a[r * k + kk] = ((r * 13 + kk * 7) % 53) as f32 * 0.011;
                        }
                    }
                    let b: Vec<f32> = (0..k * n)
                        .map(|i| ((i * 29 + 5) % 101) as f32 * 0.0213 - 1.07)
                        .collect();

                    let mut bt = Vec::new();
                    let mut out = vec![0f32; m * n];
                    matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt);
                    for j in 0..n {
                        assert_eq!(
                            out[j].to_bits(),
                            out[4 * n + j].to_bits(),
                            "matmul_f32_blocked m={m} k={k} n={n}: tiled row 0 and \
                             remainder row 4 have identical inputs but column {j} \
                             differs ({:#010x} vs {:#010x})",
                            out[j].to_bits(),
                            out[4 * n + j].to_bits()
                        );
                    }

                    // Same invariant on the panel-packed leaf.
                    let mut outp = vec![0f32; m * n];
                    let bp = crate::layout::pack_b_panels(&b, k, n);
                    matmul_f32_packed(&a, &bp, &mut outp, m, k, n);
                    for j in 0..n {
                        assert_eq!(
                            outp[j].to_bits(),
                            outp[4 * n + j].to_bits(),
                            "matmul_f32_packed m={m} k={k} n={n}: tiled row 0 and \
                             remainder row 4 have identical inputs but column {j} \
                             differs"
                        );
                    }
                }
            }
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
            (1usize, 2048usize, 64usize), // decode GEMV, k-split accumulate
            (1, 17, 22),                  // 16-wide + scalar tail
            (1, 100, 24),                 // 16-wide + 8-wide, no scalar tail
            (1, 130, 264),                // recurse on n & k (m=1 remainder + accumulate)
            (1, 3, 7),                    // scalar-tail-only (n < 8)
            (2, 9, 35),
            (3, 31, 17),   // 16-wide + scalar tail
            (3, 200, 152), // recursing multi-row remainder
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
        let a: Vec<f32> = (0..m * k)
            .map(|i| (((i % 53) as f32) - 26.0) * 0.01)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| (((i % 37) as f32) - 18.0) * 0.01)
            .collect();

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
