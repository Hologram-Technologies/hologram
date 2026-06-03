//! Source parser smoke test.

use hologram_compiler::compile_from_source;
use hologram_compiler::BackendKind;
use prism::vocabulary::WittLevel;

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
    // MatMul kernels are strictly 2-D; the source frontend must carry
    // shape annotations or `ShapeArgs::from_graph` rejects them at
    // compile time (refuse-not-fabricate on rank!=2 / unknown dims).
    let src = "
        input a :2x4
        input b :4x3
        op matmul a b :2x3 as=c
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

#[test]
fn parser_accepts_every_op_in_catalog() {
    // Every `OpKind::name()` should round-trip through the source parser.
    // This is the inverse of `dispatch_coverage::every_op_kind_dispatches_*`
    // — the source parser is the user-facing entry point and must accept
    // every canonical op name exposed by the closed `OpKind` catalog.
    use hologram_graph::OpKind;
    for &kind in OpKind::ALL {
        // Slice = ProjectField requires its starts/ends as *index-constant*
        // operands (to compute the sub-region byte offset); the bare text
        // frontend can't yet express constants, so a generic-input Slice is
        // malformed and correctly rejected. Its end-to-end behavior is covered
        // by `hologram-exec/tests/desugar.rs::slice_is_zero_movement_projectfield`.
        // Slice (ProjectField) and Pad require *index/pad constant* operands to
        // compute their byte regions; the bare text frontend can't express
        // constants yet, so a generic-input form is malformed and correctly
        // rejected. Both are covered end-to-end by hologram-exec/tests/desugar.rs.
        // MatMul + Gemm require shape-annotated rank-2 operands (the kernel is
        // strictly 2-D, and `ShapeArgs::from_graph` refuses missing dims to
        // prevent silent m=k=n=0 no-op kernels). The bare-input form here has
        // no shape; covered by `parses_matmul_pipeline` below.
        if matches!(
            kind,
            hologram_graph::OpKind::Slice
                | hologram_graph::OpKind::Pad
                | hologram_graph::OpKind::Transpose
                | hologram_graph::OpKind::Expand
                | hologram_graph::OpKind::RotaryEmbedding
                | hologram_graph::OpKind::Lrn
                | hologram_graph::OpKind::Resize
                | hologram_graph::OpKind::MatMul
                | hologram_graph::OpKind::Gemm
                | hologram_graph::OpKind::Im2Col
                | hologram_graph::OpKind::Col2Im
                | hologram_graph::OpKind::Gather
                | hologram_graph::OpKind::Cast
        ) {
            continue;
        }
        let arity = kind.primary_arity() as usize;
        let mut src = String::new();
        for i in 0..arity {
            src.push_str(&format!("input v{}\n", i));
        }
        src.push_str(&format!("op {} ", kind.name()));
        for i in 0..arity {
            src.push_str(&format!("v{} ", i));
        }
        src.push_str("as=y\n");
        src.push_str("output y\n");
        let out = compile_from_source(&src, WittLevel::W32, BackendKind::Cpu);
        assert!(
            out.is_ok(),
            "compile failed for {:?}: {:?}",
            kind,
            out.err()
        );
    }
}

// ── MatMul/Gemm rank validation regression tests ────────────────────────────
// Witness for the UOR-native refuse-not-fabricate change in
// `ShapeArgs::from_graph`: hologram's matmul kernel is strictly 2-D, but the
// previous inference silently treated a rank-3 A as `m=A[0], k=A[1]`,
// reading only the first `m*k` floats of a tens-of-thousands-of-elements
// activation. These tests pin the loud behaviour at the compile boundary.

#[test]
fn matmul_rank3_is_rejected_loud() {
    // A rank-3 [batch, seq, hidden] activation — the canonical bug pattern.
    // The kernel cannot consume it; the compiler must refuse, not collapse.
    let src = "
        input a :1x2x4
        input b :4x3
        op matmul a b :1x2x3 as=c
        output c
    ";
    let err = compile_from_source(src, WittLevel::W32, BackendKind::Cpu)
        .err()
        .expect("rank-3 MatMul A must be rejected, not silently collapsed");
    assert!(
        format!("{err:?}").contains("matmul-rank-must-be-2"),
        "expected ShapeViolation matmul-rank-must-be-2, got: {err:?}"
    );
}

#[test]
fn gemm_rank3_is_rejected_loud() {
    // Same shape pathology for Gemm (`Y = αAB + βC`) — the kernel is the
    // same strictly-2-D contraction. (`op gemm` takes a/b/c; we drop c to
    // keep the test compact — the shape check fires before any C-arity
    // requirement.)
    let src = "
        input a :1x2x4
        input b :4x3
        op gemm a b :1x2x3 as=c
        output c
    ";
    let err = compile_from_source(src, WittLevel::W32, BackendKind::Cpu)
        .err()
        .expect("rank-3 Gemm A must be rejected, not silently collapsed");
    assert!(
        format!("{err:?}").contains("matmul-rank-must-be-2"),
        "expected ShapeViolation matmul-rank-must-be-2, got: {err:?}"
    );
}

#[test]
fn matmul_k_dim_mismatch_is_rejected_loud() {
    // The contraction dim must agree: A[..,K]·B[K,..] = OUT[..,..].
    // Previously the inference picked m/k from A independently of B, so a
    // mismatch silently produced a kernel that read past one operand.
    let src = "
        input a :2x4
        input b :7x3
        op matmul a b :2x3 as=c
        output c
    ";
    let err = compile_from_source(src, WittLevel::W32, BackendKind::Cpu)
        .err()
        .expect("contraction-dim mismatch must be rejected loud");
    assert!(
        format!("{err:?}").contains("matmul-k-mismatch"),
        "expected ShapeViolation matmul-k-mismatch, got: {err:?}"
    );
}

#[test]
fn matmul_rank2_with_concrete_dims_compiles() {
    // Positive control: a canonical rank-2 matmul with concrete shapes
    // compiles cleanly. Guards against accidentally widening the refusal.
    let src = "
        input a :2x4
        input b :4x3
        op matmul a b :2x3 as=c
        output c
    ";
    let out = compile_from_source(src, WittLevel::W32, BackendKind::Cpu)
        .expect("canonical rank-2 MatMul must still compile");
    assert!(out.stats.validated_units >= 1);
}
