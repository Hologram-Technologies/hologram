//! Small-`m` efficiency probe for the f32 matmul.
//!
//! `matmul_f32_blocked`'s micro-kernel works on a register tile of `MR = 4`
//! output rows. This sweep exists because throughput collapses when `m` is not a
//! multiple of that tile: the remainder rows take a much slower path, so `m = 3`
//! can take **longer in absolute time than `m = 4`** while doing less work.
//!
//! It is a *regression pin*, not a target — it records the shape of the cliff so
//! that a fix (or a regression) is visible. Decode (`m = 1`) and short prefill
//! (`m = 2..3`) sit squarely inside it.
//!
//!   cargo run --release --example matmul_small_m -p hologram-bench

use hologram_backend::cpu::simd::matmul_f32_blocked;
use std::hint::black_box;
use std::time::Instant;

fn run(m: usize, k: usize, n: usize, iters: u64) -> (f64, f64) {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut out = vec![0f32; m * n];
    let mut bt: Vec<f32> = Vec::new();
    for _ in 0..3 {
        matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt);
    }
    let t = Instant::now();
    for _ in 0..iters {
        matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut bt);
        black_box(&out);
    }
    let s = t.elapsed().as_secs_f64();
    let ms = s / iters as f64 * 1e3;
    let gflops = 2.0 * (m * k * n) as f64 * iters as f64 / s / 1e9;
    (ms, gflops)
}

fn main() {
    let (k, n) = (1024usize, 1024usize);
    println!("matmul_f32_blocked, k={k} n={n}  (MR = 4 register tile)\n");
    println!("{:>4}  {:>10}  {:>12}  note", "m", "ms/call", "GFLOP/s");
    let mut best = 0f64;
    let mut rows = Vec::new();
    for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16] {
        let (ms, g) = run(m, k, n, if m <= 8 { 40 } else { 20 });
        best = best.max(g);
        rows.push((m, ms, g));
    }
    // Low GFLOP/s at small `m` is physics, not a defect: B (`k·n`) dominates the
    // traffic, so time floors at one pass over it regardless of `m`. The real
    // pathology is *absolute time* rising as `m` falls — that means extra passes
    // over B. Flag only that.
    let t4 = rows
        .iter()
        .find(|r| r.0 == 4)
        .map(|r| r.1)
        .unwrap_or(f64::MAX);
    for (m, ms, g) in rows {
        // One B pass is the floor, so m < 4 should cost about what m = 4 costs.
        // Materially more means extra passes over B — the defect this pins.
        let note = if m < 4 && ms > t4 * 1.5 {
            "<-- extra pass over B: slower than m=4 while doing less work"
        } else {
            ""
        };
        println!("{m:>4}  {ms:>10.3}  {g:>12.2}  {note}");
    }
    println!(
        "\nPeak observed: {best:.2} GFLOP/s at multiple-of-MR `m`. Small `m` is\n\
         bound by one pass over B (`k·n`), so its GFLOP/s is naturally lower —\n\
         but its *time* must not exceed m=4's. The remainder rows share a single\n\
         B pass; any change here must preserve the summation order, because f32\n\
         result bytes are content-addressed and reassociating the reduction\n\
         re-keys every κ that depends on it."
    );
}
