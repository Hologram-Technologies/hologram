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
