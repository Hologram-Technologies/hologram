//! `m`-scaling probe for the output-major integer GEMV — records a **known
//! defect**, not a target.
//!
//! `matmul_i8_pc_omajor` handles `m > 1` as `for i in 0..m { gemv(row_i) }`.
//! Each row streams the entire `[n,k]` weight, so a call at `m` rows moves `m`
//! full passes over it. Time is linear in `m` and the effective weight
//! bandwidth collapses:
//!
//! ```text
//! k = n = 2048 (4 MB i8 weight), x86-64 AVX2
//!    m   ms/call   ms per row   weight GB/s
//!    1     0.127        0.127          32.9
//!    2     0.261        0.130          16.1
//!    4     0.379        0.095          11.1
//!    8     0.764        0.095           5.5
//!   16     1.820        0.114           2.3
//! ```
//!
//! If the weight were streamed once per call, `ms/call` would be roughly flat.
//!
//! **Who this hits.** Decode (`m = 1`) is unaffected. A *constant* weight is
//! capped at `decode_gate::OMAJOR_W8A8_MAX_M`, so it pays at most ~3×. A weight
//! bound at load time and declared `weight_layout = OUTPUT_MAJOR` uses this
//! kernel at **every** `m` — its bytes are `[n,k]` and nothing else can read
//! them — so a graph that declares it and then prefills at large `m` pays the
//! full factor. Declare `OUTPUT_MAJOR` on decode-stage weight slots.
//! See `docs/numerics/w8a8.md`.
//!
//! **The fix, and why it is byte-free.** Block the output columns so a weight
//! block is loaded once and reused across all `m` rows: quantize every row up
//! front, then loop column blocks outside and rows inside. Every output cell is
//! still one whole dot computed by the same inner over the same `k`-vector —
//! only the order in which cells are visited changes. The accumulation is an
//! exact i32 sum and integer addition is associative, so unlike the f32 matmul
//! there is no summation-order constraint at all: no κ can move. The work is
//! confined to the `matmul_*_omajor` wrappers — the SIMD inners already take a
//! contiguous column sub-range, which is how the thread pool partitions them —
//! and the native `parallel` / `wasm-threads` dispatches would move from
//! per-row-over-columns to per-column-block-over-rows, which also cuts the
//! fork/join count from `m` to one.
//!
//!   cargo run --release --example i8_m_scaling -p hologram-backend

use hologram_backend::cpu::simd::matmul_i8_pc_omajor;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let (k, n) = (2048usize, 2048usize); // 4 MB i8 weight — larger than L2
    let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + j as f32 * 1e-6).collect();

    println!(
        "matmul_i8_pc_omajor, k={k} n={n} ({} MB i8 weight)\n",
        k * n / 1_048_576
    );
    println!(
        "{:>4} {:>10} {:>12} {:>13}  note",
        "m", "ms/call", "ms per row", "weight GB/s"
    );

    let mut base = 0f64;
    for m in [1usize, 2, 4, 8, 16] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = vec![0f32; m * n];
        for _ in 0..3 {
            matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
        }
        let iters = 20u64;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            for _ in 0..iters {
                matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
                black_box(&out);
            }
            let s = t.elapsed().as_secs_f64() / iters as f64;
            best = best.min(s);
        }
        if m == 1 {
            base = best;
        }
        // One pass over the weight is the floor, so `ms/call` should be near
        // flat. Time growing with `m` means the weight is re-streamed per row.
        let passes = best / base;
        let note = if m > 1 && passes > (m as f64) * 0.6 {
            "<-- re-streams the weight per row"
        } else {
            ""
        };
        println!(
            "{m:>4} {:>10.3} {:>12.3} {:>13.1}  {note}",
            best * 1e3,
            best * 1e3 / m as f64,
            (k * n) as f64 / best / 1e9,
        );
    }
    println!(
        "\nFlat `ms/call` would mean one weight pass per call. See this file's\n\
         module doc for the fix; it cannot change any output byte, because the\n\
         accumulation is an exact i32 sum and integer addition is associative."
    );
}
