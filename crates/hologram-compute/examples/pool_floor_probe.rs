//! Pool-admission pin: the fork-join floor must admit by **work** and decline
//! by **width** — both measured here, on the lane that ships.
//!
//! Two failure modes this pins, each found by measurement:
//!
//! - A **byte-keyed** floor (`k·n`) was blind to the batch: a 112–224 KiB
//!   per-head projection at `m = 64..128` is 15–29 MMAC of embarrassingly
//!   parallel work that ran serial (~0.9–2.0 ms). The work-based floor
//!   (`m·k·n`) admits it: measured 1.7–3.9× pooled on 4 participants.
//! - A **work-only** floor over-admits narrow jobs: at `n = 8`, four
//!   participants get 2 columns each, every row runs the SIMD column tail, and
//!   pooling *lost* (473 µs vs 342 µs serial). The width gate
//!   (`n ≥ 8 · participants`) declines it; the shape must print at its serial
//!   speed in both phases below.
//!
//! The `n = 8` line also doubles as the **publisher-overhead probe**: with the
//! GEMM ~free, it times the serial per-call work (quantize `m` rows +
//! dispatch) the pool cannot help with — measured ~1.6% of the full-width
//! pooled call (342 µs vs 21.7 ms at `128×1536×8960`), which is why the
//! publisher's quantize is not worth parallelising (Amdahl: ≤ ~1.2%).
//!
//! Run on wasm32-wasip1-threads under wasmtime (`-W threads=y -S threads=y`).
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    feature = "wasm-threads"
))]
mod probe {
    use hologram_compute::cpu::simd::matmul_i8_pc_omajor;
    use hologram_compute::cpu::wasm_pool;
    use std::time::Instant;

    fn time_m(m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
            .collect();
        let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 1e-5).collect();
        let mut out = vec![0f32; m * n];
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
            std::hint::black_box(&out);
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn run() {
        // Below the 256 KiB byte floor, big m: head/expert-projection shapes.
        let shapes = [
            (128usize, 896usize, 128usize),
            (128, 256, 896),
            (64, 896, 256),
            // Publisher-overhead probe: n=8 makes the GEMM ~free, so this times the
            // SERIAL per-call work (quantize m rows + dispatch) the pool cannot
            // help with; compare against the full-width shape below.
            (128, 1536, 8),
            (128, 1536, 8960),
        ];
        println!("serial (0 workers):");
        for &(m, k, n) in &shapes {
            let t = time_m(m, k, n, 30);
            println!(
                "  {m}x{k}x{n}  kn={:>6} KiB  {:>9.1} us  {:>7.2} GMAC/s",
                (k * n) >> 10,
                t * 1e6,
                (m * k * n) as f64 / t / 1e9
            );
        }
        let _h: Vec<_> = (0..3u32)
            .map(|i| std::thread::spawn(move || wasm_pool::hologram_worker_run(i)))
            .collect();
        while wasm_pool::hologram_pool_workers() < 3 {
            std::thread::yield_now();
        }
        println!("with pool (3 workers + main):");
        for &(m, k, n) in &shapes {
            let t = time_m(m, k, n, 30);
            println!(
                "  {m}x{k}x{n}  kn={:>6} KiB  {:>9.1} us  {:>7.2} GMAC/s",
                (k * n) >> 10,
                t * 1e6,
                (m * k * n) as f64 / t / 1e9
            );
        }
        wasm_pool::hologram_pool_shutdown();
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    feature = "wasm-threads"
))]
fn main() {
    probe::run();
}

#[cfg(not(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    feature = "wasm-threads"
)))]
fn main() {
    eprintln!(
        "pool_floor_probe measures the wasm fork-join pool; build for \
         wasm32-wasip1-threads with --features cpu,std,wasm-threads and run \
         under wasmtime (-W threads=y -S threads=y)."
    );
}
