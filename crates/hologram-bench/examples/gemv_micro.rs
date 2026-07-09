//! Decode GEMV kernel comparison: i8 (W8A8) vs i4 (W4A8) vs E8-codebook
//! (1 bit/weight) omajor kernels. Two regimes:
//!   A. L3-resident weight reused across reps  -> the compute ceiling.
//!   B. Weight larger than L3, streamed once   -> the real decode (DRAM) regime.
//! GMAC/s is the precision-invariant metric (all three do the same k*n MACs);
//! GB/s shows the streamed-weight-byte perspective.
//!
//!   cargo run --release --example gemv_micro -p hologram-bench                 # 1 thread
//!   cargo run --release --example gemv_micro -p hologram-bench --features parallel  # pool

use hologram_backend::cpu::simd::{matmul_e8cb_omajor, matmul_i4_pc_omajor, matmul_i8_pc_omajor};
use std::hint::black_box;
use std::time::Instant;

struct R {
    ms: f64,
    gmacs: f64,
    gbps: f64,
}

fn bench(macs: u64, weight_bytes: u64, iters: u64, mut f: impl FnMut()) -> R {
    for _ in 0..2 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let s = t.elapsed().as_secs_f64();
    R {
        ms: s / iters as f64 * 1e3,
        gmacs: macs as f64 * iters as f64 / s / 1e9,
        gbps: weight_bytes as f64 * iters as f64 / s / 1e9,
    }
}

fn i8_weight(k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    let bq = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let sc = (0..n).map(|j| 0.01 + j as f32 * 1e-6).collect();
    (bq, sc)
}
fn i4_weight(k: usize, n: usize) -> Vec<u8> {
    (0..n * k / 2).map(|i| (i % 251) as u8).collect()
}
fn e8_weight(k: usize, n: usize) -> (Vec<u8>, Vec<i8>) {
    let idx = (0..(k / 8) * n)
        .map(|i| ((i * 53 + 7) % 256) as u8)
        .collect();
    let cb = (0..256 * 8)
        .map(|i| ((i * 37 + 11) % 255 - 127) as i8)
        .collect();
    (idx, cb)
}

fn run(tag: &str, k: usize, n: usize, iters: u64) {
    let macs = (k as u64) * (n as u64);
    let a = vec![0.35f32; k];
    let mut out = vec![0f32; n];
    let (wi8, sc) = i8_weight(k, n);
    let wi4 = i4_weight(k, n);
    let (widx, cb) = e8_weight(k, n);

    let r8 = bench(macs, (k * n) as u64, iters, || {
        matmul_i8_pc_omajor(&a, &wi8, &sc, &mut out, 1, k, n);
        black_box(&out);
    });
    let r4 = bench(macs, (k * n / 2) as u64, iters, || {
        matmul_i4_pc_omajor(&a, &wi4, &sc, &mut out, 1, k, n);
        black_box(&out);
    });
    let re = bench(macs, (k * n / 8) as u64, iters, || {
        matmul_e8cb_omajor(&a, &widx, &cb, &sc, &mut out, 1, k, n);
        black_box(&out);
    });

    println!(
        "\n[{tag}]  k={k} n={n}  (weight: i8={} MB  i4={} MB  e8cb={} MB)",
        k * n / 1_000_000,
        k * n / 2 / 1_000_000,
        k * n / 8 / 1_000_000
    );
    println!(
        "  {:14} {:>9}  {:>10}  {:>10}",
        "kernel", "ms/call", "GMAC/s", "GB/s(wt)"
    );
    for (name, r) in [("i8 W8A8", &r8), ("i4 W4A8", &r4), ("e8cb 1-bit", &re)] {
        println!(
            "  {name:14} {:9.3}  {:10.1}  {:10.1}",
            r.ms, r.gmacs, r.gbps
        );
    }
    println!(
        "  -> e8cb vs i8: {:.2}x MAC-rate, {:.2}x wall-time",
        re.gmacs / r8.gmacs,
        r8.ms / re.ms
    );
}

fn main() {
    #[cfg(feature = "parallel")]
    println!("threads: pool enabled");
    #[cfg(not(feature = "parallel"))]
    println!("threads: single (build --features parallel for the pool)");

    // Regime A: L3-resident 4 MB i8 weight, reused -> compute ceiling.
    run("A: L3-resident ceiling", 2048, 2048, 60);
    // Regime B: weight >> L3, streamed once -> the real decode (DRAM) regime.
    // e8cb weight = 67 MB (>2x L3); i4 = 268 MB; i8 = 537 MB.
    run("B: DRAM-streaming decode", 2048, 262_144, 6);
}
