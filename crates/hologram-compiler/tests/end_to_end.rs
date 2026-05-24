//! Spec XII.3: a representative `Graph` containing one of every `OpKind`
//! compiles to a `.holo` archive without error. Empty-graph baseline below;
//! exhaustive op coverage layers on as kernels mature.

use hologram_compiler::{Compiler, BackendKind};
use hologram_graph::Graph;
use uor_foundation::WittLevel;

#[test]
fn empty_graph_compiles() {
    let g = Graph::new();
    let out = Compiler::new(g, BackendKind::Cpu, WittLevel::W32).compile().unwrap();
    assert!(out.archive.len() >= 4 + 2 + 2 + 2 + 32);
    assert_eq!(&out.archive[..4], b"HOLO");
}

#[test]
fn empty_graph_compile_then_load() {
    let g = Graph::new();
    let out = Compiler::new(g, BackendKind::Cpu, WittLevel::W32).compile().unwrap();
    let plan = hologram_archive::HoloLoader::from_bytes(&out.archive).unwrap()
        .into_plan().unwrap();
    assert!(!plan.sections().is_empty());
}

/// Verify that a fused MatMul+Relu graph compiles, loads, and executes
/// without error. The fusion pass replaces MatMul→Relu with a single
/// FusedMatMulActivation node.
#[test]
fn fused_matmul_relu_executes() {
    use hologram_compiler::compile_from_source;
    use hologram_backend::CpuBackend;
    use hologram_exec::{InferenceSession, BufferArena, InputBuffer};

    let src = r"
        input a
        input b
        op matmul a b as=c
        op relu c as=d
        output d
    ";
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    let zeros = vec![0u8; 4096];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).unwrap();
    // MatMul on all-zeros → all-zeros, Relu(0) = 0.
    assert!(!outputs.is_empty());
    assert!(outputs[0].bytes.iter().all(|&b| b == 0));
}

/// Verify that a Silu→Mul (SwiGLU) pattern compiles and executes after
/// fusion collapses it into FusedSwiGlu.
#[test]
fn fused_swiglu_executes() {
    use hologram_compiler::compile_from_source;
    use hologram_backend::CpuBackend;
    use hologram_exec::{InferenceSession, BufferArena, InputBuffer};

    let src = r"
        input gate
        input up
        op silu gate as=s
        op mul s up as=out
        output out
    ";
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&out.archive, backend).unwrap();
    let zeros = vec![0u8; 256];
    let inputs: Vec<InputBuffer> = (0..session.input_count())
        .map(|_| InputBuffer { bytes: &zeros })
        .collect();
    let outputs = session.execute(&inputs).unwrap();
    assert!(!outputs.is_empty());
}
