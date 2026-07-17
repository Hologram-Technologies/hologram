//! Transformer-block decode-step benchmark (spec XII.4).
//!
//! Constructs a representative attention-block subgraph (matmul → softmax →
//! matmul → layer_norm → relu) and times one decode step end-to-end through
//! the full pipeline (compile → archive → load → execute).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

// Representative decode-step shapes. `q :32x32 · k :32x32 → qk :32x32`,
// `attn :32x32 · v :32x32 → ctx :32x32`. Inputs are 32×32 f32 = 4 KiB each,
// matching the zero buffer used in `bench_execute`. Shape annotations are
// **required** for MatMul/Gemm operands — the kernel is strictly 2-D and
// `ShapeArgs::from_graph` refuses missing dims (no silent m=k=n=0 no-op).
const DECODE_STEP_SOURCE: &str = r"
input q :32x32
input k :32x32
input v :32x32
op matmul q k :32x32 as=qk
op softmax qk as=attn
op matmul attn v :32x32 as=ctx
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
