//! Epilogue fusion benchmark: MatMul + Silu (fused vs unfused).
//!
//! Compiles and executes a `matmul → silu` graph through the full
//! pipeline. With fusion enabled, the two ops collapse into a single
//! `FusedMatMulActivation` kernel that eliminates the intermediate
//! buffer write/read. The benchmark measures compile + execute time
//! across several matrix sizes.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_backend::CpuBackend;
use hologram_exec::{InferenceSession, BufferArena, InputBuffer};
use uor_foundation::WittLevel;

const MATMUL_SILU_SOURCE: &str = r"
input a
input b
op matmul a b as=c
op silu c as=d
output d
";

fn bench_matmul_silu_compile(c: &mut Criterion) {
    c.bench_function("epilogue_fusion::compile", |b| {
        b.iter(|| {
            let out = compile_from_source(
                black_box(MATMUL_SILU_SOURCE),
                WittLevel::W32,
                BackendKind::Cpu,
            ).unwrap();
            black_box(out);
        });
    });
}

fn bench_matmul_silu_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("epilogue_fusion::execute");
    for &size in &[64usize, 256, 1024] {
        let bytes = size * size; // W8 byte-domain: 1 byte per element
        let out = compile_from_source(MATMUL_SILU_SOURCE, WittLevel::W32, BackendKind::Cpu)
            .unwrap();
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&out.archive, backend).unwrap();
        let zeros = vec![0u8; bytes];
        let inputs: Vec<InputBuffer> = (0..session.input_count())
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    let outputs = session.execute(black_box(&inputs)).unwrap();
                    black_box(outputs);
                });
            },
        );
    }
    group.finish();
}

/// Standalone matmul (no activation) for comparison.
fn bench_matmul_only_execute(c: &mut Criterion) {
    let src = r"
        input a
        input b
        op matmul a b as=c
        output c
    ";
    let mut group = c.benchmark_group("matmul_only::execute");
    for &size in &[64usize, 256, 1024] {
        let bytes = size * size;
        let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(&out.archive, backend).unwrap();
        let zeros = vec![0u8; bytes];
        let inputs: Vec<InputBuffer> = (0..session.input_count())
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    let outputs = session.execute(black_box(&inputs)).unwrap();
                    black_box(outputs);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_silu_compile, bench_matmul_silu_execute, bench_matmul_only_execute);
criterion_main!(benches);
