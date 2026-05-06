//! Source parser smoke test.

use hologram_compiler::compile_from_source;
use hologram_compiler::BackendKind;
use uor_foundation::WittLevel;

#[test]
fn parses_minimal_graph() {
    let src = r#"
        # comment
        input x
        op relu x as=y
        output y
    "#;
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    assert!(out.archive.len() >= 4 + 2 + 2 + 2 + 32);
    assert!(out.stats.total_nodes >= 3);
}

#[test]
fn parses_matmul_pipeline() {
    let src = "
        input a
        input b
        op matmul a b as=c
        op relu c as=d
        output d
    ";
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    assert!(out.stats.validated_units >= 2);
}

#[test]
fn rejects_unknown_op() {
    let src = "op nonsense\n";
    assert!(compile_from_source(src, WittLevel::W32, BackendKind::Cpu).is_err());
}

#[test]
fn rejects_unresolved_input() {
    let src = "op relu missing\n";
    assert!(compile_from_source(src, WittLevel::W32, BackendKind::Cpu).is_err());
}
