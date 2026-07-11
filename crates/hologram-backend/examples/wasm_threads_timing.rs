//! Decode GEMV thread-scaling harness — wasm32-wasip1-threads under
//! wasmtime (std threads drive the same atomics job queue the browser's
//! web workers will):
//!
//!   RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --release \
//!     --example wasm_threads_timing --target wasm32-wasip1-threads \
//!     -p hologram-backend --no-default-features --features cpu,std,wasm-threads
//!   wasmtime run -W threads=y -S threads target/.../wasm_threads_timing.wasm
//!
//! Iteration signal only; the browser witness stays downstream.

use hologram_backend::cpu::simd::{matmul_i4_pc_omajor, matmul_i8_pc_omajor};
use hologram_backend::cpu::wasm_pool;
use std::time::Instant;

fn bench_i4(label: &str, k: usize, n: usize, iters: usize) {
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
        "  {label:<22} 1x{k}x{n:<6} {:>9.1} us/iter  {:>7.2} GB/s int4",
        per * 1e6,
        (k * n / 2) as f64 / per / 1e9
    );
}

fn bench_m(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
        .collect();
    let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 1e-5).collect();
    let mut out = vec![0f32; m * n];
    matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
        std::hint::black_box(&out);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  {label:<22} {m}x{k}x{n:<6} {:>9.1} us/iter  {:>8.2} GMAC/s",
        per * 1e6,
        (m * k * n) as f64 / per / 1e9
    );
}

fn bench(label: &str, k: usize, n: usize, iters: usize) {
    bench_m(label, 1, k, n, iters);
}

fn main() {
    println!("serial (0 workers):");
    bench("gemv_w8a8", 896, 4864, 300);
    bench("gemv_w8a8", 1536, 8960, 100);
    bench("gemv_w8a8", 3584, 18944, 20);
    bench_i4("gemv_w4a8", 3584, 18944, 20);
    // Prefill (m > 1): the batched GEMM, whose serial form dominates TTFT.
    bench_m("gemm_w8a8 prefill", 32, 896, 4864, 40);
    bench_m("gemm_w8a8 prefill", 128, 896, 4864, 10);
    bench_m("gemm_w8a8 prefill", 128, 1536, 8960, 4);

    let workers = 3u32;
    let _handles: Vec<_> = (0..workers)
        .map(|i| std::thread::spawn(move || wasm_pool::hologram_worker_run(i)))
        .collect();
    while wasm_pool::hologram_pool_workers() < workers {
        std::thread::yield_now();
    }
    println!("pooled ({} workers + main):", workers);
    bench("gemv_w8a8", 896, 4864, 300);
    bench("gemv_w8a8", 1536, 8960, 100);
    bench("gemv_w8a8", 3584, 18944, 20);
    bench_i4("gemv_w4a8", 3584, 18944, 20);
    // Pooled prefill: same output-column partition, every participant runs all
    // m rows against its weight tile — TTFT's serial gap, closed.
    bench_m("gemm_w8a8 prefill", 32, 896, 4864, 40);
    bench_m("gemm_w8a8 prefill", 128, 896, 4864, 10);
    bench_m("gemm_w8a8 prefill", 128, 1536, 8960, 4);
    wasm_pool::hologram_pool_shutdown();
}
