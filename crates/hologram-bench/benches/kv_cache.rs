//! KV cache benchmarks: write/read throughput and memory for f32 vs quantized.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hologram_exec::kv_cache::{KvBits, KvCacheConfig, KvCacheState};

const N_LAYERS: u32 = 32;
const N_KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const STRIDE: usize = (N_KV_HEADS as usize) * (HEAD_DIM as usize); // 1024

fn make_token_data(n_tokens: usize) -> Vec<f32> {
    (0..n_tokens * STRIDE)
        .map(|i| (i as f32) * 0.001 - 0.5)
        .collect()
}

// ── Write throughput ─────────────────────────────────────────────────

fn bench_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_write");
    let seq_lens = [1usize, 16, 128];

    let configs: Vec<(&str, KvCacheConfig)> = vec![
        ("f32", KvCacheConfig::default()),
        (
            "f32k_q8v",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q8,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        (
            "f32k_q4v",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q4,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        (
            "f32k_q8v_wht",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q8,
                boundary_layers: 2,
                wht_rotation: true,
            },
        ),
        ("f32k_q4v_wht", KvCacheConfig::asymmetric_q4()),
    ];

    for &seq_len in &seq_lens {
        let data = make_token_data(seq_len);

        for (name, config) in &configs {
            group.bench_with_input(BenchmarkId::new(*name, seq_len), &seq_len, |b, _| {
                let mut cache =
                    KvCacheState::with_config(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048, config.clone());
                b.iter(|| {
                    cache.reset();
                    for layer in 0..N_LAYERS {
                        cache.write_layer(layer, black_box(&data), black_box(&data));
                    }
                    cache.advance(seq_len);
                });
            });
        }
    }
    group.finish();
}

// ── Read throughput ──────────────────────────────────────────────────

fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_read");
    let seq_len = 512usize;
    let data = make_token_data(seq_len);

    let configs: Vec<(&str, KvCacheConfig)> = vec![
        ("f32", KvCacheConfig::default()),
        (
            "q8",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q8,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        (
            "q4",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q4,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        (
            "q8_wht",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q8,
                boundary_layers: 2,
                wht_rotation: true,
            },
        ),
        ("q4_wht", KvCacheConfig::asymmetric_q4()),
    ];

    let mut caches: Vec<(&str, KvCacheState)> = configs
        .iter()
        .map(|(name, config)| {
            let mut cache =
                KvCacheState::with_config(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048, config.clone());
            for layer in 0..N_LAYERS {
                cache.write_layer(layer, &data, &data);
            }
            cache.advance(seq_len);
            (*name, cache)
        })
        .collect();

    for (name, cache) in &mut caches {
        group.bench_function(format!("{name}/read_v"), |b| {
            b.iter(|| {
                for layer in 0..N_LAYERS {
                    let borrowed = cache.read_v(layer);
                    if !borrowed.is_empty() {
                        black_box(borrowed);
                    } else {
                        black_box(cache.read_v_owned(layer));
                    }
                }
            });
        });
    }

    group.finish();
}

// ── Single-layer micro-benchmarks ────────────────────────────────────

fn bench_single_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_single_layer");
    // Isolate per-layer cost without the loop-over-32-layers noise.
    let seq_len = 512usize;
    let data = make_token_data(seq_len);
    let layer = 16u32; // middle layer (not boundary)

    let configs: Vec<(&str, KvCacheConfig)> = vec![
        (
            "q8",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q8,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        (
            "q4",
            KvCacheConfig {
                k_bits: KvBits::F32,
                v_bits: KvBits::Q4,
                boundary_layers: 2,
                wht_rotation: false,
            },
        ),
        ("q4_wht", KvCacheConfig::asymmetric_q4()),
    ];

    for (name, config) in &configs {
        // Write benchmark (single layer, 512 tokens).
        group.bench_function(format!("{name}/write_512tok"), |b| {
            let mut cache =
                KvCacheState::with_config(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048, config.clone());
            b.iter(|| {
                cache.reset();
                cache.write_layer(layer, black_box(&data), black_box(&data));
                cache.advance(seq_len);
            });
        });

        // Read benchmark (single layer, 512 tokens pre-filled).
        group.bench_function(format!("{name}/read_512tok"), |b| {
            let mut cache =
                KvCacheState::with_config(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048, config.clone());
            cache.write_layer(layer, &data, &data);
            cache.advance(seq_len);
            b.iter(|| {
                black_box(cache.read_v_owned(layer));
            });
        });
    }

    group.finish();
}

// ── Memory usage ─────────────────────────────────────────────────────

fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_memory");
    let seq_len = 2048usize;
    let data = make_token_data(seq_len);

    group.bench_function("f32/2K_ctx_alloc_write", |b| {
        b.iter(|| {
            let mut cache = KvCacheState::new(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048);
            for layer in 0..N_LAYERS {
                cache.write_layer(layer, black_box(&data), black_box(&data));
            }
            cache.advance(seq_len);
            black_box(&cache);
        });
    });

    group.bench_function("q4_wht/2K_ctx_alloc_write", |b| {
        b.iter(|| {
            let config = KvCacheConfig::asymmetric_q4();
            let mut cache = KvCacheState::with_config(N_LAYERS, N_KV_HEADS, HEAD_DIM, 2048, config);
            for layer in 0..N_LAYERS {
                cache.write_layer(layer, black_box(&data), black_box(&data));
            }
            cache.advance(seq_len);
            black_box(&cache);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_write,
    bench_read,
    bench_single_layer,
    bench_memory_footprint,
);
criterion_main!(benches);
