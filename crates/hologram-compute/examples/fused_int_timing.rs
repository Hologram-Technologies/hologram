//! Spike timing: fused per-channel int8 matmul vs f32 matmul (single-thread).
//!
//! Run native (no parallel, so both are single-threaded — apples-to-apples):
//!   cargo run --release --example fused_int_timing
//!
//! Measures whether reading the i8 weight directly (and dequantizing in
//! registers) beats the f32 matmul (and, by extension, the dequant-then-matmul
//! path, which is f32 matmul PLUS a full dequant pass).

use hologram_compute::cpu::simd::{matmul_f32_blocked, matmul_i8_per_channel};
use std::time::Instant;

fn gflops(m: usize, k: usize, n: usize, per: f64) -> f64 {
    (2.0 * m as f64 * k as f64 * n as f64) / per / 1e9
}

fn row(label: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let bf: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j % 31) as f32 * 0.0003).collect();
    let mut out = vec![0f32; m * n];
    let mut bt = Vec::new();

    // f32
    matmul_f32_blocked(&a, &bf, &mut out, m, k, n, &mut bt);
    let t = Instant::now();
    for _ in 0..iters {
        matmul_f32_blocked(&a, &bf, &mut out, m, k, n, &mut bt);
        std::hint::black_box(&out);
    }
    let f32_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    // fused int8
    matmul_i8_per_channel(&a, &bq, &scales, &mut out, m, k, n);
    let t = Instant::now();
    for _ in 0..iters {
        matmul_i8_per_channel(&a, &bq, &scales, &mut out, m, k, n);
        std::hint::black_box(&out);
    }
    let i8_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

    println!(
        "  {label:<22} f32 {:>8.1}us ({:>6.1} GF)   int8 {:>8.1}us ({:>6.1} GF)   int8/f32 {:.2}x",
        f32_us,
        gflops(m, k, n, f32_us / 1e6),
        i8_us,
        gflops(m, k, n, i8_us / 1e6),
        i8_us / f32_us,
    );
}

fn main() {
    println!("fused int8 vs f32 matmul (single-thread). int8/f32 < 1.0 means int8 faster.");
    println!("compute-bound:");
    row("256x256x256", 256, 256, 256, 300);
    row("512x512x512", 512, 512, 512, 50);
    println!("decode-shaped (GEMV, M=1):");
    row("1x2048x2048", 1, 2048, 2048, 2000);
    row("1x4096x4096", 1, 4096, 4096, 800);
    row("1x2048x8192", 1, 2048, 8192, 500);
    println!("M-crossover sweep at K=N=2048 (find where int8/f32 crosses 1.0):");
    for m in [1usize, 2, 3, 4, 6, 8, 16] {
        row(&format!("{m}x2048x2048"), m, 2048, 2048, 800);
    }
}
