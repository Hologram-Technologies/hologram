//! Tiered execution path overhead benchmark (PM_7).
//!
//! Compiles a small `relu -> sigmoid` graph and benchmarks session
//! execution. When the `tiered-exec` feature is enabled on `hologram-exec`,
//! the session uses the tiered dispatch path (coherence checks, tier
//! routing, migration bookkeeping). The baseline measures the same session
//! execute, establishing the overhead floor.
//!
//! Run with:
//!   cargo bench --bench tiered_executor

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

const MODEL_SOURCE: &str = r"
input x
op relu x as=y
op sigmoid y as=z
output z
";

/// Larger model with more ops to amplify any per-call overhead.
const CHAIN_SOURCE: &str = r"
input x
op relu x as=a
op sigmoid a as=b
op tanh b as=c
op relu c as=d
op sigmoid d as=e
op tanh e as=f
output f
";

fn bench_session_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered_executor");

    // --- Small model (2 ops): relu -> sigmoid ---
    {
        let out = compile_from_source(MODEL_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&out.archive, backend).unwrap();
        let input_count = session.input_count();
        let zeros = vec![0u8; 256];
        let inputs: Vec<InputBuffer> = (0..input_count)
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();

        group.bench_function("small_2op_execute", |b| {
            b.iter(|| {
                let outputs = session.execute(black_box(&inputs)).unwrap();
                black_box(outputs);
            });
        });
    }

    // --- Chained model (6 ops): relu -> sigmoid -> tanh -> relu -> sigmoid -> tanh ---
    {
        let out = compile_from_source(CHAIN_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&out.archive, backend).unwrap();
        let input_count = session.input_count();
        let zeros = vec![0u8; 256];
        let inputs: Vec<InputBuffer> = (0..input_count)
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();

        group.bench_function("chain_6op_execute", |b| {
            b.iter(|| {
                let outputs = session.execute(black_box(&inputs)).unwrap();
                black_box(outputs);
            });
        });
    }

    // --- Repeated execution (same inputs, measures content-addressing reuse) ---
    {
        let out = compile_from_source(CHAIN_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&out.archive, backend).unwrap();
        let input_count = session.input_count();
        let zeros = vec![0u8; 256];
        let inputs: Vec<InputBuffer> = (0..input_count)
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();

        // Warm up: first execution populates the content-address cache.
        let _ = session.execute(&inputs).unwrap();

        group.bench_function("chain_6op_cached_reexecute", |b| {
            b.iter(|| {
                let outputs = session.execute(black_box(&inputs)).unwrap();
                black_box(outputs);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_session_execute);
criterion_main!(benches);
