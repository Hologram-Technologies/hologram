//! End-to-end: compile a graph, load it as a session, execute it.

use hologram_compiler::{compile_from_source, BackendKind};
use hologram_backend::CpuBackend;
use hologram_exec::{InferenceSession, BufferArena, InputBuffer};
use uor_foundation::WittLevel;

#[test]
fn round_trip_compile_load_execute() {
    let src = "
        input x
        op relu x as=y
        output y
    ";
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();

    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    assert!(session.kernel_count() >= 1);

    let zeros = vec![0u8; 64];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).unwrap();
    assert_eq!(outputs.len(), session.output_count());
}

#[test]
fn empty_archive_loads_executes() {
    let out = hologram_compiler::compile(
        hologram_graph::Graph::new(),
        BackendKind::Cpu,
        WittLevel::W32,
    ).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    let _ = session.execute(&[]).unwrap();
}
