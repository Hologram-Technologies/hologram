//! Criterion benchmarks for float matmul across sizes.
//!
//! Sweeps M×K×N to measure crossover points between micro-kernel,
//! BLAS (Accelerate), and GPU (Metal) paths.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hologram_fused_component::float_dispatch::dispatch_matmul;

fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sweep");

    // Common transformer sizes: (M, K, N)
    // M = batch*seq (small for decode, large for prefill)
    // K = hidden_dim, N = hidden_dim or vocab_size
    let sizes: &[(usize, usize, usize)] = &[
        (1, 64, 64),        // tiny
        (1, 256, 256),      // small projection
        (1, 2048, 2048),    // LLaMA decode step (single token)
        (4, 2048, 2048),    // small batch decode
        (32, 128, 128),     // attention Q@K^T (32 heads, seq=128)
        (128, 2048, 2048),  // prefill projection
        (1, 2048, 8192),    // FFN up-projection (decode)
        (1, 4096, 4096),    // large decode (L2 stress)
        (128, 4096, 11008), // LLaMA-2 7B FFN prefill (L2 stress)
    ];

    for &(m, k, n) in sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        group.bench_with_input(
            BenchmarkId::new("dispatch_matmul", format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, &(m, k, n)| {
                bench.iter(|| {
                    dispatch_matmul(black_box(&[&a_bytes[..], &b_bytes[..]]), m, k, n).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_sizes);
criterion_main!(benches);
