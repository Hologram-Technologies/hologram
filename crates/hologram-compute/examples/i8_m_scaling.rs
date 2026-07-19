//! `m`-scaling probe for the output-major integer GEMV — a **regression pin**.
//!
//! For `m > 1` the kernel blocks the output columns, so a block's weight slab is
//! read once and reused by every row. A per-row loop instead re-streams the
//! whole `[n,k]` weight for each row, and then the weight is never amortized.
//!
//! **The signal is `ms per row`, not GFLOP/s and not GB/s.** Total time growing
//! with `m` is not a defect — the work grows with `m`. The defect is *per-row*
//! time staying flat as `m` rises: that means each row paid for its own pass
//! over the weight. When the weight is amortized, per-row time falls toward the
//! compute bound.
//!
//! It only shows up once the weight exceeds cache. At 4 MB the GEMV is
//! compute-bound and per-row time is nearly flat either way; that is physics, not
//! a bug, and an earlier version of this file mistook it for one.
//!
//! ```text
//! x86-64 AVX2, ms per row              m=1     m=4     m=16     GMAC/s @ m=16
//!   4 MB weight (L3)   per-row loop   0.120   0.097   0.095       44.3
//!                      blocked        0.115   0.087   0.084       50.1
//!  32 MB weight (LLC)  per-row loop   1.713   1.570   1.698       19.8
//!                      blocked        1.783   0.937   0.718       46.8   (2.37x)
//!  64 MB weight (DRAM) per-row loop   2.900   2.744   2.798       24.0
//!                      blocked        2.688   1.617   1.377       48.8   (2.03x)
//! ```
//!
//! Blocked prefill reaches the same ~48 GMAC/s ceiling the cache-resident case
//! hits; the per-row loop is pinned at the weight's DRAM bandwidth forever.
//! Decode (`m = 1`) is untouched — one row has nothing to amortize over.
//!
//! **Why reordering is free.** Every output cell is one whole dot over the same
//! `k`-vector; only the order in which cells are visited changes. The
//! accumulation is an exact i32 sum and integer addition is associative, so no
//! bit can move — unlike the f32 matmul, whose summation order is part of its
//! contract. Pinned by `batched_integer_gemv_equals_row_by_row_bit_for_bit`.
//!
//!   cargo run --release --example i8_m_scaling -p hologram-compute

use hologram_compute::cpu::simd::matmul_i8_pc_omajor;
use std::hint::black_box;
use std::time::Instant;

fn bench(m: usize, k: usize, n: usize, bq: &[i8], scales: &[f32]) -> f64 {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
        .collect();
    let mut out = vec![0f32; m * n];
    for _ in 0..2 {
        matmul_i8_pc_omajor(&a, bq, scales, &mut out, m, k, n);
    }
    let iters: u64 = if k * n > 30_000_000 { 3 } else { 15 };
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..iters {
            matmul_i8_pc_omajor(&a, bq, scales, &mut out, m, k, n);
            black_box(&out);
        }
        best = best.min(t.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

fn main() {
    for &(k, n) in &[(2048usize, 2048usize), (2048, 16384), (4096, 16384)] {
        let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.01 + j as f32 * 1e-6).collect();
        println!("\nk={k} n={n} — {} MB i8 weight", k * n / 1_048_576);
        println!(
            "{:>4} {:>10} {:>13} {:>11}  note",
            "m", "ms/call", "ms per row", "GMAC/s"
        );

        let mut per_row_at_1 = 0f64;
        for &m in &[1usize, 2, 4, 8, 16] {
            let s = bench(m, k, n, &bq, &scales);
            let per_row = s / m as f64;
            if m == 1 {
                per_row_at_1 = per_row;
            }
            // Per-row time must FALL as `m` rises once the weight is amortized.
            // Staying at the m=1 cost means every row paid for its own pass.
            let note = if m >= 8 && per_row > per_row_at_1 * 0.9 {
                "<-- per-row cost flat: the weight is re-streamed per row"
            } else {
                ""
            };
            println!(
                "{m:>4} {:>10.3} {:>13.4} {:>11.1}  {note}",
                s * 1e3,
                per_row * 1e3,
                (m * k * n) as f64 / s / 1e9,
            );
        }
    }
    println!(
        "\nRead `ms per row`, not the totals: total time grows with `m` because the\n\
         work does. A flat per-row cost is the defect. At 4 MB the kernel is\n\
         compute-bound and per-row time is nearly flat regardless — that is physics."
    );
}
