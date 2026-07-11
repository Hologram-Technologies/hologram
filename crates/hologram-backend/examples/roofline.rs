//! Roofline verification: are the decode/prefill kernels **at the machine's
//! ceilings**, or merely faster than yesterday?
//!
//! Incremental tuning cannot answer "are we done?". A roofline can: a
//! bandwidth-bound kernel is optimal when it moves bytes at the machine's
//! measured streaming rate, and a compute-bound kernel when it sustains the
//! same MAC rate cache-resident and DRAM-streamed alike. This harness measures
//! the ceilings *on the same machine, in the same process* and places each
//! kernel against them — so the verdict travels with the run instead of being
//! a stale claim in a doc.
//!
//! - **Read ceiling**: bytes/s of a pure streaming reduction over a
//!   DRAM-resident buffer (the shape of a decode weight scan — read-only,
//!   sequential, no writeback traffic to speak of).
//! - **Copy ceiling**: bytes/s of `copy_from_slice` (read + write), the
//!   conventional memcpy number, for context.
//! - **Decode (`m = 1`)** streams the whole weight once per token and is
//!   bandwidth-bound at DRAM sizes: its weight-GB/s over the read ceiling is
//!   the optimality figure.
//! - **Prefill (`m ≫ 1`)** amortizes the weight over rows and is
//!   compute-bound: its GMAC/s against the cache-resident ceiling (the same
//!   kernel at a weight that fits in cache) is the figure.
//!
//! Host numbers; the wasm lane's equivalents come from `wasm_threads_timing`
//! under wasmtime. Per the project's measurement discipline, neither is quoted
//! for the other.
//!
//!   cargo run --release --example roofline -p hologram-backend

use hologram_backend::cpu::simd::{matmul_e8cb_omajor, matmul_i4_pc_omajor, matmul_i8_pc_omajor};
use std::hint::black_box;
use std::time::Instant;

fn best_of<F: FnMut() -> f64>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        best = best.min(f());
    }
    best
}

/// Pure streaming read: XOR-fold the buffer as u64 words through four
/// independent accumulators, so the loop is load-bound, not add-latency-bound.
/// (A naive per-byte sum measures the accumulator dependency chain, not the
/// memory system — it reported 3.8 GB/s on a machine whose kernels stream at
/// 18+, which is how a broken ceiling makes every verdict nonsense.)
fn read_ceiling_gbs(buf: &[i8]) -> f64 {
    let words: &[u64] = bytemuck::cast_slice(&buf[..buf.len() & !7]);
    let t = best_of(7, || {
        let t0 = Instant::now();
        let (mut a0, mut a1, mut a2, mut a3) = (0u64, 0u64, 0u64, 0u64);
        for w in words.chunks_exact(4) {
            a0 ^= w[0];
            a1 ^= w[1];
            a2 ^= w[2];
            a3 ^= w[3];
        }
        black_box(a0 ^ a1 ^ a2 ^ a3);
        t0.elapsed().as_secs_f64()
    });
    buf.len() as f64 / t / 1e9
}

fn copy_ceiling_gbs(src: &[i8], dst: &mut [i8]) -> f64 {
    let t = best_of(5, || {
        let t0 = Instant::now();
        dst.copy_from_slice(src);
        black_box(&dst[0]);
        t0.elapsed().as_secs_f64()
    });
    // Bytes moved = read + write.
    (2 * src.len()) as f64 / t / 1e9
}

struct Gemm<'a> {
    bq8: &'a [i8],
    bq4: &'a [u8],
    bqe: &'a [u8],
    cb: &'a [i8],
    scales: &'a [f32],
    k: usize,
    n: usize,
}

impl Gemm<'_> {
    fn time_i8(&self, m: usize, iters: usize) -> f64 {
        let a: Vec<f32> = (0..m * self.k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = vec![0f32; m * self.n];
        matmul_i8_pc_omajor(&a, self.bq8, self.scales, &mut out, m, self.k, self.n);
        best_of(3, || {
            let t0 = Instant::now();
            for _ in 0..iters {
                matmul_i8_pc_omajor(&a, self.bq8, self.scales, &mut out, m, self.k, self.n);
                black_box(&out);
            }
            t0.elapsed().as_secs_f64() / iters as f64
        })
    }
    fn time_i4(&self, iters: usize) -> f64 {
        let a: Vec<f32> = (0..self.k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = vec![0f32; self.n];
        matmul_i4_pc_omajor(&a, self.bq4, self.scales, &mut out, 1, self.k, self.n);
        best_of(3, || {
            let t0 = Instant::now();
            for _ in 0..iters {
                matmul_i4_pc_omajor(&a, self.bq4, self.scales, &mut out, 1, self.k, self.n);
                black_box(&out);
            }
            t0.elapsed().as_secs_f64() / iters as f64
        })
    }
    fn time_e8(&self, iters: usize) -> f64 {
        let a: Vec<f32> = (0..self.k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let mut out = vec![0f32; self.n];
        matmul_e8cb_omajor(
            &a,
            self.bqe,
            self.cb,
            self.scales,
            &mut out,
            1,
            self.k,
            self.n,
        );
        best_of(3, || {
            let t0 = Instant::now();
            for _ in 0..iters {
                matmul_e8cb_omajor(
                    &a,
                    self.bqe,
                    self.cb,
                    self.scales,
                    &mut out,
                    1,
                    self.k,
                    self.n,
                );
                black_box(&out);
            }
            t0.elapsed().as_secs_f64() / iters as f64
        })
    }
}

fn main() {
    // ── Machine ceilings, measured now, here ───────────────────────────────
    let big = vec![7i8; 256 << 20]; // 256 MB — far beyond LLC
    let mut dst = vec![0i8; 256 << 20];
    let read = read_ceiling_gbs(&big);
    let copy = copy_ceiling_gbs(&big, &mut dst);
    drop(dst);
    println!("machine ceilings (measured):");
    println!("  streaming read : {read:7.1} GB/s");
    println!("  memcpy (r+w)   : {copy:7.1} GB/s\n");

    // ── DRAM-streamed decode: bandwidth-bound, judged vs the read ceiling ──
    let (k, n) = (4096usize, 16384usize); // 64 MB i8 weight
    let bq8: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let bq4: Vec<u8> = (0..k * n / 2).map(|i| ((i * 53 + 9) % 251) as u8).collect();
    let cb: Vec<i8> = (0..256 * 8)
        .map(|i| ((i * 37 + 11) % 255 - 127) as i8)
        .collect();
    let bqe: Vec<u8> = (0..n * (k / 8))
        .map(|i| ((i * 17 + 3) % 256) as u8)
        .collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.01 + j as f32 * 1e-6).collect();
    let g = Gemm {
        bq8: &bq8,
        bq4: &bq4,
        bqe: &bqe,
        cb: &cb,
        scales: &scales,
        k,
        n,
    };

    println!("decode (m = 1), 64 MB-class weight — bandwidth-bound, vs read ceiling:");
    let t8 = g.time_i8(1, 12);
    let w8 = (k * n) as f64 / t8 / 1e9;
    println!(
        "  i8   W8A8 : {w8:7.1} GB/s weight   = {:5.1}% of read ceiling",
        100.0 * w8 / read
    );
    let t4 = g.time_i4(12);
    let w4 = (k * n / 2) as f64 / t4 / 1e9;
    println!(
        "  i4   W4A8 : {w4:7.1} GB/s weight   = {:5.1}% of read ceiling ({}x fewer bytes than i8)",
        100.0 * w4 / read,
        2
    );
    let te = g.time_e8(12);
    let we = (n * (k / 8)) as f64 / te / 1e9;
    println!(
        "  e8cb W1A8 : {we:7.1} GB/s indices  = {:5.1}% of read ceiling ({}x fewer bytes than i8)",
        100.0 * we / read,
        8
    );
    println!(
        "  (>100% means the probe under-reads the true ceiling — the kernel, with its\n\
         \u{20}  multiple streams and prefetch, is the better bandwidth probe; the verdict\n\
         \u{20}  'no kernel headroom' holds a fortiori. i4/e8cb sit BELOW the byte ceiling\n\
         \u{20}  because their bottleneck is decode compute (unpack / gather), not bytes:\n\
         \u{20}  i8 {:.1}, i4 {:.1}, e8cb {:.1} GMAC/s — the MAC rates, not the byte rates,\n\
         \u{20}  are their binding constraint.)\n",
        (k * n) as f64 / t8 / 1e9,
        (k * n) as f64 / t4 / 1e9,
        (k * n) as f64 / te / 1e9
    );

    // ── Prefill: compute-bound, judged vs the cache-resident ceiling ───────
    // Cache-resident ceiling: same kernel, 4 MB weight, large m.
    let (ck, cn) = (2048usize, 2048usize);
    let cbq: Vec<i8> = (0..ck * cn)
        .map(|i| ((i as i64 % 255) - 127) as i8)
        .collect();
    let cs: Vec<f32> = (0..cn).map(|j| 0.01 + j as f32 * 1e-6).collect();
    let cg = Gemm {
        bq8: &cbq,
        bq4: &bq4,
        bqe: &bqe,
        cb: &cb,
        scales: &cs,
        k: ck,
        n: cn,
    };
    let t_ceil = cg.time_i8(64, 30);
    let ceil = (64 * ck * cn) as f64 / t_ceil / 1e9;
    println!("prefill (i8 W8A8) — compute-bound, vs cache-resident ceiling {ceil:.1} GMAC/s:");
    for m in [16usize, 64, 128] {
        let t = g.time_i8(m, 5);
        let gm = (m * k * n) as f64 / t / 1e9;
        println!(
            "  m = {m:<4} 64 MB weight: {gm:7.1} GMAC/s  = {:5.1}% of ceiling",
            100.0 * gm / ceil
        );
    }
    println!(
        "\nVerdict semantics: a bandwidth-bound kernel near the read ceiling and a\n\
         compute-bound kernel near the cache-resident ceiling have no headroom left\n\
         in the kernel itself — remaining levers are fewer bytes (deeper weight\n\
         codecs), more participants (pooling), or **not recomputing** (κ reuse)."
    );
}
