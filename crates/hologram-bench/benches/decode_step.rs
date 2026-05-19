//! Transformer-block decode-step benchmark (spec XII.4).
//!
//! Constructs a representative attention-block subgraph (matmul → softmax →
//! matmul → layer_norm → relu) and times one decode step end-to-end through
//! the full pipeline (compile → archive → load → execute).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

const DECODE_STEP_SOURCE: &str = r"
input q
input k
input v
op matmul q k as=qk
op softmax qk as=attn
op matmul attn v as=ctx
op layer_norm ctx as=normed
op relu normed as=out
output out
";

fn bench_compile(c: &mut Criterion) {
    c.bench_function("decode_step::compile", |b| {
        b.iter(|| {
            let out = compile_from_source(
                black_box(DECODE_STEP_SOURCE),
                WittLevel::W32,
                BackendKind::Cpu,
            )
            .unwrap();
            black_box(out);
        });
    });
}

fn bench_session_load(c: &mut Criterion) {
    let out = compile_from_source(DECODE_STEP_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
    c.bench_function("decode_step::session_load", |b| {
        b.iter(|| {
            let backend: CpuBackend<BufferArena> = CpuBackend::new();
            let session = InferenceSession::load(black_box(&out.archive), backend).unwrap();
            black_box(session);
        });
    });
}

fn bench_execute(c: &mut Criterion) {
    let out = compile_from_source(DECODE_STEP_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    let zeros = vec![0u8; 4096];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    c.bench_function("decode_step::execute", |b| {
        b.iter(|| {
            let outputs = session.execute(black_box(&inputs)).unwrap();
            black_box(outputs);
        });
    });
}

criterion_group!(benches, bench_compile, bench_session_load, bench_execute);
criterion_main!(benches);
