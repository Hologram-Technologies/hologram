//! Standalone matmul timing harness — runs under wasmtime (no criterion).
//!
//! Times the public f32 matmul kernels. Build with and without
//! `-Ctarget-feature=+simd128` to compare the wasm SIMD128 path against the
//! scalar fallback (and on native aarch64, NEON vs the same kernels):
//!
//!   RUSTFLAGS="-Ctarget-feature=+simd128" cargo run --release \
//!     --example wasm_matmul_timing --target wasm32-wasip1
//!   cargo run --release --example wasm_matmul_timing --target wasm32-wasip1
//!
//! (single-threaded; the `parallel` feature is off for wasm)

use hologram_compute::cpu::simd::{
    matmul_f32_blocked, matmul_f32_packed, matmul_i4_pc_omajor, matmul_i8_pc_omajor,
    matmul_i8_per_channel,
};
use std::time::Instant;

fn gflops(m: usize, k: usize, n: usize, per: f64) -> f64 {
    (2.0 * m as f64 * k as f64 * n as f64) / per / 1e9
}

fn bench_blocked(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut out = vec![0f32; m * n];
    let mut bt = Vec::new();
    matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  {label:<24} {:>9.1} us/iter  {:>7.2} GFLOP/s",
        per * 1e6,
        gflops(m, k, n, per)
    );
}

fn bench_packed(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let packed = hologram_compute::layout::pack_b_panels(&b, k, n);
    let mut out = vec![0f32; m * n];
    matmul_f32_packed(&a, &packed, &mut out, m, k, n); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_f32_packed(&a, &packed, &mut out, m, k, n);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  {label:<24} {:>9.1} us/iter  {:>7.2} GFLOP/s",
        per * 1e6,
        gflops(m, k, n, per)
    );
}

/// Decode int8 GEMV: reports **GB/s of int8 weight bytes streamed** — the
/// numerator of hologram-ai's bandwidth-ratio witness. `omajor_w8a8` is the
/// decode kernel (output-major weight, per-token W8A8 integer accumulation);
/// `kn_w8a32` is the prior fused path at the same shape.
fn bench_i8_gemv(k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.037).collect();
    let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 1e-5).collect();
    let mut out = vec![0f32; n];
    let gbs = |per: f64| (k * n) as f64 / per / 1e9;

    matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, 1, k, n); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, 1, k, n);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  omajor_w8a8 1x{k}x{n:<6} {:>9.1} us/iter  {:>7.2} GB/s int8",
        per * 1e6,
        gbs(per)
    );

    matmul_i8_per_channel(&a, &bq, &scales, &mut out, 1, k, n); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_i8_per_channel(&a, &bq, &scales, &mut out, 1, k, n);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  kn_w8a32    1x{k}x{n:<6} {:>9.1} us/iter  {:>7.2} GB/s int8",
        per * 1e6,
        gbs(per)
    );
}

/// Decode softmax exp: the deterministic vectorized exp vs the scalar
/// libm loop it replaced, over an attention-bucket-sized row.
fn bench_exp(len: usize, iters: usize) {
    use hologram_compute::cpu::simd::simd_f32_exp_inplace;
    let base: Vec<f32> = (0..len).map(|i| -((i % 89) as f32) * 0.93).collect();
    let mut buf = base.clone();
    simd_f32_exp_inplace(&mut buf); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        buf.copy_from_slice(&base);
        simd_f32_exp_inplace(&mut buf);
        std::hint::black_box(&buf);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  exp_det simd  {len:>6}   {:>9.2} us/iter  {:>8.1} Melem/s",
        per * 1e6,
        len as f64 / per / 1e6
    );
    let t0 = Instant::now();
    for _ in 0..iters {
        buf.copy_from_slice(&base);
        for x in buf.iter_mut() {
            *x = libm::expf(*x);
        }
        std::hint::black_box(&buf);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  libm expf     {len:>6}   {:>9.2} us/iter  {:>8.1} Melem/s",
        per * 1e6,
        len as f64 / per / 1e6
    );
}

/// LUT-tier W4A8 GEMV: half the streamed bytes of the i8 kernel; GB/s is
/// over the ACTUAL packed-i4 bytes (k·n/2), so compare step TIME against
/// the i8 line at the same shape for the tier's win.
fn bench_i4_gemv(k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.037).collect();
    let bq: Vec<u8> = (0..k * n / 2).map(|i| (i % 251) as u8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 1e-5).collect();
    let mut out = vec![0f32; n];
    matmul_i4_pc_omajor(&a, &bq, &scales, &mut out, 1, k, n); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_i4_pc_omajor(&a, &bq, &scales, &mut out, 1, k, n);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  omajor_w4a8 1x{k}x{n:<6} {:>9.1} us/iter  {:>7.2} GB/s int4",
        per * 1e6,
        (k * n / 2) as f64 / per / 1e9
    );
}

fn main() {
    println!("matmul_f32_blocked (unpacked, single-thread):");
    bench_blocked("64x64x64", 64, 64, 64, 3000);
    bench_blocked("128x128x128", 128, 128, 128, 800);
    bench_blocked("256x256x256", 256, 256, 256, 150);
    bench_blocked("gemv 1x512x512", 1, 512, 512, 4000);
    bench_blocked("gemv 1x2048x2048", 1, 2048, 2048, 400);
    // Small-`m` sweep on the lane that actually ships: single-threaded wasm
    // SIMD128. `m = 1..3` take the const-generic row-remainder path, `m = 4`
    // the MR register tile. **Absolute time must not rise as `m` falls** — that
    // would mean a leftover row is re-streaming the whole `k×n` weight instead
    // of sharing the tile's single pass over B. (`matmul_small_m` in
    // hologram-bench pins the same signal on the host; it cannot run here
    // because criterion does not build for wasm.)
    println!("small-m sweep, k = n = 1024 (time must not rise as m falls):");
    for m in [1usize, 2, 3, 4, 6, 8] {
        bench_blocked(&format!("m={m:<2} 1024x1024"), m, 1024, 1024, 40);
    }
    println!("matmul_f32_packed (packed weights — real decode path):");
    bench_packed("256x256x256", 256, 256, 256, 150);
    bench_packed("gemv 1x2048x2048", 1, 2048, 2048, 400);
    println!("int8 decode GEMV (deployed browser shapes, m = 1):");
    bench_i8_gemv(896, 896, 2000);
    bench_i8_gemv(896, 4864, 400);
    bench_i8_gemv(4864, 896, 400);
    bench_i8_gemv(1536, 8960, 100);
    bench_i8_gemv(3584, 18944, 20);
    println!("int4 decode GEMV (LUT tier, half the streamed bytes):");
    bench_i4_gemv(896, 4864, 400);
    bench_i4_gemv(1536, 8960, 100);
    bench_i4_gemv(3584, 18944, 20);
    println!("decode softmax exp (deterministic vectorized vs scalar libm):");
    bench_exp(4096, 4000);
}
