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

use hologram_backend::cpu::simd::{matmul_f32_blocked, matmul_f32_packed};
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
    let packed = hologram_backend::layout::pack_b_panels(&b, k, n);
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

fn main() {
    println!("matmul_f32_blocked (unpacked, single-thread):");
    bench_blocked("64x64x64", 64, 64, 64, 3000);
    bench_blocked("128x128x128", 128, 128, 128, 800);
    bench_blocked("256x256x256", 256, 256, 256, 150);
    bench_blocked("gemv 1x512x512", 1, 512, 512, 4000);
    bench_blocked("gemv 1x2048x2048", 1, 2048, 2048, 400);
    println!("matmul_f32_packed (packed weights — real decode path):");
    bench_packed("256x256x256", 256, 256, 256, 150);
    bench_packed("gemv 1x2048x2048", 1, 2048, 2048, 400);
}
