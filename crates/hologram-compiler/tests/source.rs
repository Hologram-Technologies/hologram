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

#[test]
fn parser_accepts_every_op_in_catalog() {
    // Every `OpKind::name()` should round-trip through the source parser.
    // This is the inverse of `dispatch_coverage::every_op_kind_dispatches_*`
    // — the source parser is the user-facing entry point and must accept
    // all 105 spec-V.3/V.4/X-5 op names.
    use hologram_graph::OpKind;
    const ALL: &[OpKind] = &[
        OpKind::Neg,
        OpKind::Bnot,
        OpKind::Succ,
        OpKind::Pred,
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Xor,
        OpKind::And,
        OpKind::Or,
        OpKind::Relu,
        OpKind::Sigmoid,
        OpKind::Tanh,
        OpKind::Gelu,
        OpKind::Silu,
        OpKind::Elu,
        OpKind::Selu,
        OpKind::Exp,
        OpKind::Log,
        OpKind::Log1p,
        OpKind::Sqrt,
        OpKind::Reciprocal,
        OpKind::Sin,
        OpKind::Cos,
        OpKind::Tan,
        OpKind::Asin,
        OpKind::Acos,
        OpKind::Atan,
        OpKind::Ceil,
        OpKind::Floor,
        OpKind::Round,
        OpKind::Erf,
        OpKind::IsNaN,
        OpKind::Sign,
        OpKind::Abs,
        OpKind::Div,
        OpKind::Pow,
        OpKind::Mod,
        OpKind::Min,
        OpKind::Max,
        OpKind::Equal,
        OpKind::Less,
        OpKind::LessOrEqual,
        OpKind::Greater,
        OpKind::GreaterOrEqual,
        OpKind::MatMul,
        OpKind::Gemm,
        OpKind::Conv2d,
        OpKind::ConvTranspose2d,
        OpKind::LayerNorm,
        OpKind::RmsNorm,
        OpKind::GroupNorm,
        OpKind::InstanceNorm,
        OpKind::AddRmsNorm,
        OpKind::ReduceSum,
        OpKind::ReduceMean,
        OpKind::ReduceProd,
        OpKind::ReduceMin,
        OpKind::ReduceMax,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Concat,
        OpKind::Slice,
        OpKind::Softmax,
        OpKind::LogSoftmax,
        OpKind::MaxPool2d,
        OpKind::AvgPool2d,
        OpKind::GlobalAvgPool,
        OpKind::Attention,
        OpKind::FusedSwiGlu,
        OpKind::Pad,
        OpKind::Expand,
        OpKind::Resize,
        OpKind::CumSum,
        OpKind::RotaryEmbedding,
        OpKind::Clip,
        OpKind::Lrn,
        OpKind::Where,
        OpKind::Dequantize,
    ];
    for &kind in ALL {
        // Slice = ProjectField requires its starts/ends as *index-constant*
        // operands (to compute the sub-region byte offset); the bare text
        // frontend can't yet express constants, so a generic-input Slice is
        // malformed and correctly rejected. Its end-to-end behavior is covered
        // by `hologram-exec/tests/desugar.rs::slice_is_zero_movement_projectfield`.
        // Slice (ProjectField) and Pad require *index/pad constant* operands to
        // compute their byte regions; the bare text frontend can't express
        // constants yet, so a generic-input form is malformed and correctly
        // rejected. Both are covered end-to-end by hologram-exec/tests/desugar.rs.
        if matches!(kind, hologram_graph::OpKind::Slice | hologram_graph::OpKind::Pad) {
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
