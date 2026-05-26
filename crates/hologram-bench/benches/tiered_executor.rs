//! Tiered execution path overhead benchmark.
//!
//! Compiles a small `relu -> sigmoid` graph and benchmarks two execution
//! paths side-by-side:
//!
//! 1. **Baseline** — `Executor::run_levels` (raw schedule walk, no tier logic).
//! 2. **Tiered session** — `InferenceSession::execute` (includes tier dispatch,
//!    coherence checks, and migration bookkeeping when `tiered-exec` is enabled).
//!
//! The delta between the two isolates the overhead introduced by the tiered
//! execution machinery.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_archive::{decode_exec_plan, decode_ports, decoder, format::SectionKind, HoloLoader};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_exec::{BufferArena, Executor, InferenceSession, InputBuffer, SlotSpan};
use uor_foundation::WittLevel;

const MODEL_SOURCE: &str = r"
input x
op relu x as=y
op sigmoid y as=z
output z
";

/// Slot size used for every slot in the baseline workspace. 1024 bytes is
/// more than enough for the tiny benchmark model and avoids replicating
/// the private per-slot sizing logic from `session.rs`.
const SLOT_BYTES: u32 = 1024;

/// Build the raw executor state from an archive using only public APIs.
fn build_baseline(
    archive: &[u8],
) -> (
    Vec<hologram_backend::KernelCall>,
    Vec<Vec<u32>>,
    BufferArena,
    CpuBackend<BufferArena>,
) {
    let plan = HoloLoader::from_bytes(archive)
        .unwrap()
        .into_plan()
        .unwrap();

    let calls_section = plan.section(SectionKind::KernelCalls).unwrap();
    let calls = decoder::decode_calls(calls_section).unwrap();

    let levels: Vec<Vec<u32>> = plan
        .section(SectionKind::ExecPlan)
        .ok()
        .map(decode_exec_plan)
        .transpose()
        .unwrap()
        .unwrap_or_else(|| vec![(0..calls.len() as u32).collect()]);

    let inputs = plan
        .section(SectionKind::Inputs)
        .ok()
        .map(decode_ports)
        .transpose()
        .unwrap()
        .unwrap_or_default();
    let outputs = plan
        .section(SectionKind::Outputs)
        .ok()
        .map(decode_ports)
        .transpose()
        .unwrap()
        .unwrap_or_default();

    // Determine slot count from port descriptors. The kernel calls also
    // reference slots but their BufferRef fields are buried inside variant
    // payloads (private `buffers()` helper in session.rs). For this small
    // model the ports cover all relevant slots; over-provisioning a few
    // extra slots is harmless for a benchmark.
    let mut slot_count: usize = 0;
    for p in inputs.iter().chain(outputs.iter()) {
        let need = (p.slot as usize).saturating_add(1);
        if need > slot_count {
            slot_count = need;
        }
    }
    // Pad up: kernels may reference intermediate slots beyond ports.
    // 16 slots is generous for a 2-op graph.
    if slot_count < 16 {
        slot_count = 16;
    }

    let mut slots = Vec::with_capacity(slot_count);
    let mut total: usize = 0;
    for _ in 0..slot_count {
        slots.push(SlotSpan {
            offset: total as u32,
            length: SLOT_BYTES,
        });
        total += SLOT_BYTES as usize;
    }
    let workspace = BufferArena::with_capacity(total, slots);
    let backend: CpuBackend<BufferArena> = CpuBackend::new();

    (calls, levels, workspace, backend)
}

fn bench_tiered_vs_baseline(c: &mut Criterion) {
    let out = compile_from_source(MODEL_SOURCE, WittLevel::W32, BackendKind::Cpu).unwrap();
    let archive = &out.archive;

    let mut group = c.benchmark_group("tiered_executor");

    // --- Baseline: raw Executor::run_levels ---
    {
        let (calls, levels, mut workspace, mut backend) = build_baseline(archive);

        // Zero-fill the input slot(s).
        let input_ports = HoloLoader::from_bytes(archive)
            .unwrap()
            .into_plan()
            .unwrap()
            .section(SectionKind::Inputs)
            .ok()
            .map(decode_ports)
            .transpose()
            .unwrap()
            .unwrap_or_default();
        for p in &input_ports {
            if let Some(dst) = workspace.write_slot(p.slot as usize) {
                for b in dst.iter_mut() {
                    *b = 0;
                }
            }
        }

        group.bench_function("baseline_run_levels", |b| {
            b.iter(|| {
                Executor::run_levels(
                    black_box(&mut backend),
                    black_box(&calls),
                    black_box(&levels),
                    black_box(&mut workspace),
                )
                .unwrap();
            });
        });
    }

    // --- Tiered: InferenceSession::execute ---
    {
        let backend: CpuBackend<BufferArena> = CpuBackend::new();
        let mut session = InferenceSession::load(archive, backend).unwrap();
        let input_count = session.input_count();
        let zeros = vec![0u8; 256]; // plenty for the small model
        let inputs: Vec<InputBuffer> = (0..input_count)
            .map(|_| InputBuffer { bytes: &zeros })
            .collect();

        group.bench_function("session_execute_tiered", |b| {
            b.iter(|| {
                let outputs = session.execute(black_box(&inputs)).unwrap();
                black_box(outputs);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tiered_vs_baseline);
criterion_main!(benches);
