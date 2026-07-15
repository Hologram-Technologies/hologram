//! Source IR lowering tests.

use hologram_compiler::source::{
    lower_ir, parse_document, parse_ir, parse_ir_diagnostic, parse_ir_with_options,
    resolve_source_language, source_language_from_extension, source_language_from_name,
    HologramFrontend, SourceBinding, SourceConst, SourceDocument, SourceExternalConst,
    SourceExternalTensor, SourceFrontend, SourceGraph, SourceInput, SourceItem, SourceLanguage,
    SourceOpCall, SourceOutput, SourceParseOptions, SourceProgram, SourceTensorLiteral, SourceType,
};
use hologram_compiler::{compile_from_source_language, BackendKind, Compiler};
use hologram_graph::registry::ShapeDescriptor;
use hologram_graph::{NodeId, OpKind, ReduceAttrs};
use hologram_types::HologramHasher;
use prism::vocabulary::Hasher;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use uor_foundation::WittLevel;

#[test]
fn lowers_ir_without_text_parser() {
    let mut program = SourceProgram::new();
    let x = program.intern("x");
    let y = program.intern("y");
    program.push(SourceItem::Input(SourceInput::new(
        x,
        SourceType::f32(None),
    )));
    program.push(SourceItem::Binding(SourceBinding::op(
        Some(y),
        SourceOpCall::new(OpKind::Relu, vec![x], None),
    )));
    program.push(SourceItem::Output(SourceOutput::new(y)));

    let graph = lower_ir(&program).expect("source IR lowers");
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
}

#[test]
fn rejects_constant_value_count_mismatch() {
    let mut program = SourceProgram::new();
    let w = program.intern("w");
    let literal = SourceTensorLiteral::new(vec![0; 3 * 4], 3);
    let constant = SourceConst::new(w, SourceType::f32(Some(shape_2x2())), literal);
    program.push(SourceItem::Const(constant));

    let err = lower_ir(&program).expect_err("value-count mismatch must fail");
    assert!(format!("{err:?}").contains("value count mismatch"));
}

#[test]
fn rejects_duplicate_source_names() {
    let mut program = SourceProgram::new();
    let x = program.intern("x");
    program.push(SourceItem::Input(SourceInput::new(
        x,
        SourceType::f32(None),
    )));
    program.push(SourceItem::Input(SourceInput::new(
        x,
        SourceType::f32(None),
    )));

    let err = lower_ir(&program).expect_err("duplicate source name must fail");
    assert!(format!("{err:?}").contains("duplicate name"));
}

#[test]
fn rejects_unresolved_ir_input() {
    let mut program = SourceProgram::new();
    let missing = program.intern("missing");
    let y = program.intern("y");
    program.push(SourceItem::Binding(SourceBinding::op(
        Some(y),
        SourceOpCall::new(OpKind::Relu, vec![missing], None),
    )));

    let err = lower_ir(&program).expect_err("unresolved IR input must fail");
    assert!(format!("{err:?}").contains("unresolved input"));
}

#[test]
fn rejects_output_from_constant_source() {
    let mut program = SourceProgram::new();
    let w = program.intern("w");
    let literal = SourceTensorLiteral::new(vec![0; 4 * 4], 4);
    let constant = SourceConst::new(w, SourceType::f32(Some(shape_2x2())), literal);
    program.push(SourceItem::Const(constant));
    program.push(SourceItem::Output(SourceOutput::new(w)));

    let err = lower_ir(&program).expect_err("constant output source must fail");
    assert!(format!("{err:?}").contains("unknown/!node source"));
}

#[test]
fn lowers_external_const_to_constant_store() {
    let _env = external_env_guard();
    let bytes = 1.0f32.to_le_bytes();
    let path = write_temp_tensor(&bytes);
    let mut program = SourceProgram::new();
    let w = program.intern("w");
    let reference = SourceExternalTensor::file(
        path.to_string_lossy(),
        0,
        bytes.len() as u64,
        digest(&bytes),
    );
    let constant = SourceExternalConst::new(w, SourceType::f32(Some(shape_1())), reference);
    program.push(SourceItem::ExternalConst(constant));

    let graph = lower_ir(&program).expect("external const lowers");
    assert_eq!(graph.constants().len(), 1);
    std::fs::remove_file(path).expect("remove temp tensor");
}

#[test]
fn rejects_external_const_hash_mismatch() {
    let _env = external_env_guard();
    let bytes = 1.0f32.to_le_bytes();
    let path = write_temp_tensor(&bytes);
    let mut program = SourceProgram::new();
    let w = program.intern("w");
    let bad_hash = [0u8; 32];
    let reference =
        SourceExternalTensor::file(path.to_string_lossy(), 0, bytes.len() as u64, bad_hash);
    let constant = SourceExternalConst::new(w, SourceType::f32(Some(shape_1())), reference);
    program.push(SourceItem::ExternalConst(constant));

    let err = lower_ir(&program).expect_err("hash mismatch must fail");
    assert!(format!("{err:?}").contains("content hash mismatch"));
    std::fs::remove_file(path).expect("remove temp tensor");
}

#[test]
fn external_const_program_matches_inline_native_graph_and_archive() {
    let _env = external_env_guard();
    let bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let path = write_temp_tensor(&bytes);
    let external = external_matmul_program(&path, &bytes);
    let native_graph = graph_from_source(golden_native_matmul_source(), SourceLanguage::Hologram);
    let external_graph = lower_ir(&external).expect("external source lowers");

    assert_eq!(native_graph.node_count(), external_graph.node_count());
    assert_eq!(native_graph.inputs(), external_graph.inputs());
    assert_eq!(native_graph.outputs(), external_graph.outputs());
    assert_eq!(
        constant_bytes(&native_graph, 0),
        constant_bytes(&external_graph, 0)
    );
    assert_eq!(
        archive_from_source(golden_native_matmul_source(), SourceLanguage::Hologram),
        archive_from_program(&external)
    );
    std::fs::remove_file(path).expect("remove temp tensor");
}

#[test]
fn external_const_root_env_rejects_outside_files() {
    let _env = external_env_guard();
    let bytes = 1.0f32.to_le_bytes();
    let root = unique_temp_path("hologram-external-root");
    let path = write_temp_tensor(&bytes);
    std::fs::create_dir_all(&root).expect("create external root");
    std::env::set_var("HOLOGRAM_EXTERNAL_TENSOR_ROOT", &root);

    let mut program = SourceProgram::new();
    let w = program.intern("w");
    let reference = SourceExternalTensor::file(
        path.to_string_lossy(),
        0,
        bytes.len() as u64,
        digest(&bytes),
    );
    program.push(SourceItem::ExternalConst(SourceExternalConst::new(
        w,
        SourceType::f32(Some(shape_1())),
        reference,
    )));

    let err = lower_ir(&program).expect_err("outside root must fail");
    assert!(format!("{err:?}").contains("outside root"));
    std::fs::remove_file(path).expect("remove temp tensor");
    std::fs::remove_dir(root).expect("remove external root");
}

#[test]
fn lowers_source_attrs_to_graph_sparse_attrs() {
    let mut program = SourceProgram::new();
    let x = program.intern("x");
    let y = program.intern("y");
    let mut call = SourceOpCall::new(OpKind::ReduceSum, vec![x], None);
    call.attrs.reduce = Some(ReduceAttrs {
        axes_mask: 0b10,
        keepdims: true,
    });
    program.push(SourceItem::Input(SourceInput::new(
        x,
        SourceType::f32(Some(shape_2x2())),
    )));
    program.push(SourceItem::Binding(SourceBinding::op(Some(y), call)));

    let graph = lower_ir(&program).expect("attr source IR lowers");
    let attrs = graph
        .reduce_attrs(NodeId(1))
        .expect("reduce attrs attached");
    assert_eq!(attrs.axes_mask, 0b10);
    assert!(attrs.keepdims);
}

#[test]
fn interns_symbols_once() {
    let mut program = SourceProgram::new();
    let first = program.intern("x");
    let second = program.intern("x");
    assert_eq!(first, second);
    assert_eq!(program.symbol_name(first), Some("x"));
}

#[test]
fn parses_legacy_source_to_ir() {
    let source = "input x\nop relu x as=y\noutput y\n";
    let program = parse_ir(source, SourceLanguage::Hologram).expect("legacy source parses");
    assert_eq!(program.items().len(), 3);
}

#[test]
fn native_frontend_adapter_parses_ir() {
    let source = "input x\nop relu x as=y\noutput y\n";
    let program = HologramFrontend
        .parse_ir(source)
        .expect("native adapter parses");
    assert_eq!(program.items().len(), 3);
}

#[test]
fn frontend_metadata_resolves_names_and_extensions() {
    assert_eq!(
        source_language_from_name("py"),
        Some(SourceLanguage::Python)
    );
    assert_eq!(
        source_language_from_name("TypeScript"),
        Some(SourceLanguage::TypeScript)
    );
    assert_eq!(
        source_language_from_extension(".tsx"),
        Some(SourceLanguage::TypeScript)
    );
    assert_eq!(
        source_language_from_extension(".txt"),
        Some(SourceLanguage::Hologram)
    );
    assert_eq!(source_language_from_extension(".holo"), None);
    assert_eq!(
        resolve_source_language(None, Some("unknown")).unwrap(),
        SourceLanguage::Hologram
    );
}

#[cfg(not(feature = "frontend-python"))]
#[test]
fn planned_host_frontends_are_registered_but_unsupported() {
    let err = parse_ir("", SourceLanguage::Python).expect_err("python is not implemented yet");
    assert!(format!("{err:?}").contains("source language unsupported"));
}

#[cfg(not(feature = "frontend-typescript"))]
#[test]
fn typescript_frontend_is_registered_but_feature_gated() {
    let err = parse_ir("", SourceLanguage::TypeScript).expect_err("typescript feature is disabled");
    assert!(format!("{err:?}").contains("source language unsupported"));
}

#[cfg(not(feature = "frontend-rust"))]
#[test]
fn rust_frontend_is_registered_but_feature_gated() {
    let err = parse_ir("", SourceLanguage::Rust).expect_err("rust feature is disabled");
    assert!(format!("{err:?}").contains("source language unsupported"));
}

#[test]
fn parses_native_source_as_one_implicit_document_graph() {
    let source = "input x\nop relu x as=y\noutput y\n";
    let document =
        parse_document(source, SourceLanguage::Hologram).expect("source document parses");
    assert_eq!(document.graphs().len(), 1);
    assert_eq!(document.graphs()[0].name.as_deref(), None);
}

#[test]
fn selects_named_graph_from_source_document() {
    let mut document = SourceDocument::new();
    document.push(SourceGraph::named("encoder", simple_program("x")));
    document.push(SourceGraph::named("decoder", simple_program("z")));

    let options = SourceParseOptions::new().graph("decoder");
    let program = document.select(&options).expect("named graph selected");
    assert_eq!(program.symbol_name(program.items()[0].name()), Some("z"));
}

#[test]
fn rejects_ambiguous_unselected_source_document() {
    let mut document = SourceDocument::new();
    document.push(SourceGraph::named("encoder", simple_program("x")));
    document.push(SourceGraph::named("decoder", simple_program("z")));

    let err = document
        .select(&SourceParseOptions::new())
        .expect_err("multi-graph document requires selection");
    assert!(format!("{err:?}").contains("source graph ambiguous"));
}

#[test]
fn rejects_missing_named_source_graph() {
    let options = SourceParseOptions::new().graph("missing");
    let err = parse_ir_with_options("input x\n", SourceLanguage::Hologram, &options)
        .expect_err("implicit graph does not match named selection");
    assert!(format!("{err:?}").contains("source graph not found"));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_extracts_graph_and_ignores_unrelated_code() {
    let source = r#"
def ordinary_app_code():
    return 42

def encoder(h):
    "graph doc"
    x = h.input("x", dtype="f32", shape=[2, 3])
    w = h.const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    y = h.ops.matmul(x, w, shape=[2, 2])
    h.output("y", y)
"#;
    let document = parse_document(source, SourceLanguage::Python).expect("python document parses");
    assert_eq!(document.graphs().len(), 1);
    assert_eq!(document.graphs()[0].name.as_deref(), Some("encoder"));

    let program = parse_ir(source, SourceLanguage::Python).expect("python graph selected");
    assert_eq!(program.items().len(), 4);
    let graph = lower_ir(&program).expect("python source lowers");
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_requires_selection_for_multiple_graphs() {
    let source = r#"
def encoder(h):
    x = h.input("x")
    h.output(x)

def decoder(h):
    z = h.input("z")
    h.output(z)
"#;
    let err = parse_ir(source, SourceLanguage::Python)
        .expect_err("multiple inferred graphs require selection");
    assert!(format!("{err:?}").contains("source graph ambiguous"));

    let options = SourceParseOptions::new().graph("decoder");
    let program = parse_ir_with_options(source, SourceLanguage::Python, &options)
        .expect("selected graph parses");
    assert_eq!(program.symbol_name(program.items()[0].name()), Some("z"));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_reports_missing_graph_when_no_builder_pattern_exists() {
    let source = r#"
def ordinary_app_code(x):
    return x
"#;
    let document = parse_document(source, SourceLanguage::Python).expect("python parses");
    assert_eq!(document.graphs().len(), 0);
    let err = parse_ir(source, SourceLanguage::Python).expect_err("no graph candidates");
    assert!(format!("{err:?}").contains("source graph missing"));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_rejects_unsupported_statement_inside_graph() {
    let source = r#"
def graph(h):
    x = h.input("x")
    if True:
        pass
    h.output(x)
"#;
    let err = parse_document(source, SourceLanguage::Python)
        .expect_err("graph functions reject unsupported statements");
    assert!(format!("{err:?}").contains("unsupported graph statement"));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_reports_rejected_statement_position() {
    let source =
        "def graph(h):\n    x = h.input(\"x\")\n    if True:\n        pass\n    h.output(x)\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::Python)
        .expect_err("unsupported graph statement has a span");
    assert_eq!(err.kind, "python: unsupported graph statement");
    assert_eq!(err.line, 3);
    assert_eq!(err.column, 5);
    assert_eq!(err.rejected, "if True:");
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_reports_rejected_expr_position() {
    let source = "def graph(h):\n    x = h.input(\"x\", dtype=\"f64\")\n    h.output(x)\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::Python)
        .expect_err("unsupported dtype has a span");
    let column = source.lines().nth(1).unwrap().find("\"f64\"").unwrap() + 1;
    assert_eq!(err.kind, "python: unsupported dtype");
    assert_eq!(err.line, 2);
    assert_eq!(err.column, column);
    assert_eq!(err.rejected, "\"f64\"");
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_parses_reduce_attrs_to_graph_attrs() {
    let source = r#"
def graph(h):
    x = h.input("x", shape=[2, 3])
    y = h.ops.reduce_sum(x, shape=[2], axes=[1], keepdims=True)
    h.output("y", y)
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Python).unwrap()).unwrap();
    let attrs = graph.reduce_attrs(NodeId(1)).unwrap();
    assert_eq!(attrs.axes_mask, 0b10);
    assert!(attrs.keepdims);
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_parses_conv_attrs_to_graph_attrs() {
    let source = r#"
def graph(h):
    x = h.input("x", shape=[1, 1, 4, 4])
    w = h.input("w", shape=[1, 1, 3, 3])
    y = h.ops.conv2d(x, w, shape=[1, 1, 2, 2], stride=[2, 3], pads=[1, 2, 1, 2], kernel=[3, 3])
    h.output("y", y)
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Python).unwrap()).unwrap();
    let attrs = graph.conv_attrs(NodeId(2)).unwrap();
    assert_eq!((attrs.stride_h, attrs.stride_w), (2, 3));
    assert_eq!((attrs.pad_h, attrs.pad_w), (1, 2));
    assert_eq!((attrs.k_h, attrs.k_w), (3, 3));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_parses_scalar_attrs_to_graph_attrs() {
    let source = r#"
def graph(h):
    a = h.input("a", shape=[2, 3])
    b = h.input("b", shape=[3, 2])
    c = h.input("c", shape=[2, 2])
    y = h.ops.gemm(a, b, c, shape=[2, 2], alpha=0.5, beta=0.25)
    h.output("y", y)
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Python).unwrap()).unwrap();
    let attrs = graph.gemm_attrs(NodeId(3)).unwrap();
    assert_eq!(attrs.alpha_bits, 0.5f32.to_bits());
    assert_eq!(attrs.beta_bits, 0.25f32.to_bits());
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_frontend_rejects_attrs_on_wrong_op() {
    let source = r#"
def graph(h):
    x = h.input("x", shape=[2, 3])
    y = h.ops.relu(x, shape=[2, 3], keepdims=True)
    h.output("y", y)
"#;
    let err =
        parse_ir_diagnostic(source, SourceLanguage::Python).expect_err("attr must be op scoped");
    assert_eq!(err.kind, "op: attr not valid");
    assert_eq!(err.line, 4);
    assert!(err.rejected.contains("keepdims=True"));
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_source_matches_native_matmul_archive() {
    let native = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let python = r#"
def graph(h):
    x = h.input("x", shape=[2, 3])
    w = h.const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    y = h.ops.matmul(x, w, shape=[2, 2])
    h.output(y)
"#;
    assert_language_sources_equivalent(native, python, SourceLanguage::Python);
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_source_matches_native_reduce_attrs_archive() {
    let native = "
        input x: f32[2, 3]
        let y: f32[2] = reduce_sum(x, axes=[1], keepdims=true)
        output y
    ";
    let python = r#"
def graph(h):
    x = h.input("x", shape=[2, 3])
    y = h.ops.reduce_sum(x, shape=[2], axes=[1], keepdims=True)
    h.output(y)
"#;
    assert_language_sources_equivalent(native, python, SourceLanguage::Python);
}

#[cfg(feature = "frontend-python")]
#[test]
fn python_source_matches_native_gemm_attrs_archive() {
    let native = "
        input a: f32[2, 3]
        input b: f32[3, 2]
        input c: f32[2, 2]
        let y: f32[2, 2] = gemm(a, b, c, alpha=0.5, beta=0.25)
        output y
    ";
    let python = r#"
def graph(h):
    a = h.input("a", shape=[2, 3])
    b = h.input("b", shape=[3, 2])
    c = h.input("c", shape=[2, 2])
    y = h.ops.gemm(a, b, c, shape=[2, 2], alpha=0.5, beta=0.25)
    h.output(y)
"#;
    assert_language_sources_equivalent(native, python, SourceLanguage::Python);
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_extracts_graph_and_ignores_unrelated_code() {
    let source = r#"
import { app } from "./app";

function ordinaryAppCode() {
    return 42;
}

export function encoder(h: HologramBuilder) {
    const x = h.input("x", { dtype: "f32", shape: [2, 3] });
    const w = h.constant("w", { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] });
    const y = h.ops.matmul(x, w, { shape: [2, 2] });
    h.output("y", y);
}
"#;
    let document =
        parse_document(source, SourceLanguage::TypeScript).expect("typescript document parses");
    assert_eq!(document.graphs().len(), 1);
    assert_eq!(document.graphs()[0].name.as_deref(), Some("encoder"));

    let program = parse_ir(source, SourceLanguage::TypeScript).expect("typescript graph selected");
    assert_eq!(program.items().len(), 4);
    let graph = lower_ir(&program).expect("typescript source lowers");
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_requires_selection_for_multiple_graphs() {
    let source = r#"
function encoder(h) {
    const x = h.input("x");
    h.output(x);
}

function decoder(h) {
    const z = h.input("z");
    h.output(z);
}
"#;
    let err = parse_ir(source, SourceLanguage::TypeScript)
        .expect_err("multiple inferred graphs require selection");
    assert!(format!("{err:?}").contains("source graph ambiguous"));

    let options = SourceParseOptions::new().graph("decoder");
    let program = parse_ir_with_options(source, SourceLanguage::TypeScript, &options)
        .expect("selected graph parses");
    assert_eq!(program.symbol_name(program.items()[0].name()), Some("z"));
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_reports_missing_graph_when_no_builder_pattern_exists() {
    let source = r#"
function ordinaryAppCode(x: number) {
    return x;
}
"#;
    let document = parse_document(source, SourceLanguage::TypeScript).expect("typescript parses");
    assert_eq!(document.graphs().len(), 0);
    let err = parse_ir(source, SourceLanguage::TypeScript).expect_err("no graph candidates");
    assert!(format!("{err:?}").contains("source graph missing"));
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_reports_rejected_statement_position() {
    let source =
        "function graph(h) {\n    const x = h.input(\"x\");\n    if (true) {}\n    h.output(x);\n}\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::TypeScript)
        .expect_err("unsupported graph statement has a span");
    assert_eq!(err.kind, "typescript: unsupported graph statement");
    assert_eq!(err.line, 3);
    assert_eq!(err.column, 5);
    assert_eq!(err.rejected, "if (true) {}");
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_reports_rejected_expr_position() {
    let source = "function graph(h) {\n    const x = h.input(\"x\", { dtype: \"f64\" });\n    h.output(x);\n}\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::TypeScript)
        .expect_err("unsupported dtype has a span");
    let column = source.lines().nth(1).unwrap().find("\"f64\"").unwrap() + 1;
    assert_eq!(err.kind, "typescript: unsupported dtype");
    assert_eq!(err.line, 2);
    assert_eq!(err.column, column);
    assert_eq!(err.rejected, "\"f64\"");
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_parses_reduce_attrs_to_graph_attrs() {
    let source = r#"
function graph(h) {
    const x = h.input("x", { shape: [2, 3] });
    const y = h.ops.reduce_sum(x, { shape: [2], axes: [1], keepdims: true });
    h.output("y", y);
}
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::TypeScript).unwrap()).unwrap();
    let attrs = graph.reduce_attrs(NodeId(1)).unwrap();
    assert_eq!(attrs.axes_mask, 0b10);
    assert!(attrs.keepdims);
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_parses_scalar_attrs_to_graph_attrs() {
    let source = r#"
function graph(h) {
    const a = h.input("a", { shape: [2, 3] });
    const b = h.input("b", { shape: [3, 2] });
    const c = h.input("c", { shape: [2, 2] });
    const y = h.ops.gemm(a, b, c, { shape: [2, 2], alpha: 0.5, beta: 0.25 });
    h.output("y", y);
}
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::TypeScript).unwrap()).unwrap();
    let attrs = graph.gemm_attrs(NodeId(3)).unwrap();
    assert_eq!(attrs.alpha_bits, 0.5f32.to_bits());
    assert_eq!(attrs.beta_bits, 0.25f32.to_bits());
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_frontend_rejects_attrs_on_wrong_op() {
    let source = r#"
function graph(h) {
    const x = h.input("x", { shape: [2, 3] });
    const y = h.ops.relu(x, { shape: [2, 3], keepdims: true });
    h.output("y", y);
}
"#;
    let err = parse_ir_diagnostic(source, SourceLanguage::TypeScript)
        .expect_err("attr must be op scoped");
    assert_eq!(err.kind, "op: attr not valid");
    assert_eq!(err.line, 4);
    assert!(err.rejected.contains("keepdims: true"));
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_source_matches_native_matmul_archive() {
    let native = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let typescript = r#"
function graph(h) {
    const x = h.input("x", { shape: [2, 3] });
    const w = h.const("w", { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] });
    const y = h.ops.matmul(x, w, { shape: [2, 2] });
    h.output(y);
}
"#;
    assert_sources_equivalent(native, typescript);
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_source_matches_native_reduce_attrs_archive() {
    let native = "
        input x: f32[2, 3]
        let y: f32[2] = reduce_sum(x, axes=[1], keepdims=true)
        output y
    ";
    let typescript = r#"
function graph(h) {
    const x = h.input("x", { shape: [2, 3] });
    const y = h.ops.reduce_sum(x, { shape: [2], axes: [1], keepdims: true });
    h.output(y);
}
"#;
    assert_sources_equivalent(native, typescript);
}

#[cfg(feature = "frontend-typescript")]
#[test]
fn typescript_source_matches_native_gemm_attrs_archive() {
    let native = "
        input a: f32[2, 3]
        input b: f32[3, 2]
        input c: f32[2, 2]
        let y: f32[2, 2] = gemm(a, b, c, alpha=0.5, beta=0.25)
        output y
    ";
    let typescript = r#"
function graph(h) {
    const a = h.input("a", { shape: [2, 3] });
    const b = h.input("b", { shape: [3, 2] });
    const c = h.input("c", { shape: [2, 2] });
    const y = h.ops.gemm(a, b, c, { shape: [2, 2], alpha: 0.5, beta: 0.25 });
    h.output(y);
}
"#;
    assert_sources_equivalent(native, typescript);
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_extracts_graph_and_ignores_unrelated_code() {
    let source = r#"
fn ordinary_app_code() -> i32 {
    42
}

pub fn encoder(h: &mut HologramBuilder) {
    let x = h.input("x", dtype("f32"), shape([2, 3]));
    let w = h.constant("w", shape([3, 2]), values([1, 2, 3, 4, 5, 6]));
    let y = h.ops().matmul(x, w, shape([2, 2]));
    h.output("y", y);
}
"#;
    let document = parse_document(source, SourceLanguage::Rust).expect("rust document parses");
    assert_eq!(document.graphs().len(), 1);
    assert_eq!(document.graphs()[0].name.as_deref(), Some("encoder"));

    let program = parse_ir(source, SourceLanguage::Rust).expect("rust graph selected");
    assert_eq!(program.items().len(), 4);
    let graph = lower_ir(&program).expect("rust source lowers");
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_requires_selection_for_multiple_graphs() {
    let source = r#"
fn encoder(h: &mut HologramBuilder) {
    let x = h.input("x");
    h.output(x);
}

fn decoder(h: &mut HologramBuilder) {
    let z = h.input("z");
    h.output(z);
}
"#;
    let err = parse_ir(source, SourceLanguage::Rust)
        .expect_err("multiple inferred graphs require selection");
    assert!(format!("{err:?}").contains("source graph ambiguous"));

    let options = SourceParseOptions::new().graph("decoder");
    let program = parse_ir_with_options(source, SourceLanguage::Rust, &options)
        .expect("selected graph parses");
    assert_eq!(program.symbol_name(program.items()[0].name()), Some("z"));
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_reports_missing_graph_when_no_builder_pattern_exists() {
    let source = r#"
fn ordinary_app_code(x: i32) -> i32 {
    x
}
"#;
    let document = parse_document(source, SourceLanguage::Rust).expect("rust parses");
    assert_eq!(document.graphs().len(), 0);
    let err = parse_ir(source, SourceLanguage::Rust).expect_err("no graph candidates");
    assert!(format!("{err:?}").contains("source graph missing"));
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_reports_rejected_statement_position() {
    let source =
        "fn graph(h: &mut HologramBuilder) {\n    let x = h.input(\"x\");\n    if true {}\n    h.output(x);\n}\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::Rust)
        .expect_err("unsupported graph statement has a span");
    assert_eq!(err.kind, "rust: unsupported graph statement");
    assert_eq!(err.line, 3);
    assert_eq!(err.column, 5);
    assert_eq!(err.rejected, "if true {}");
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_reports_rejected_expr_position() {
    let source =
        "fn graph(h: &mut HologramBuilder) {\n    let x = h.input(\"x\", dtype(\"f64\"));\n    h.output(x);\n}\n";
    let err = parse_ir_diagnostic(source, SourceLanguage::Rust)
        .expect_err("unsupported dtype has a span");
    let column = source.lines().nth(1).unwrap().find("\"f64\"").unwrap() + 1;
    assert_eq!(err.kind, "rust: unsupported dtype");
    assert_eq!(err.line, 2);
    assert_eq!(err.column, column);
    assert_eq!(err.rejected, "\"f64\"");
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_parses_reduce_attrs_to_graph_attrs() {
    let source = r#"
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", shape([2, 3]));
    let y = h.ops().reduce_sum(x, shape([2]), axes([1]), keepdims(true));
    h.output("y", y);
}
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Rust).unwrap()).unwrap();
    let attrs = graph.reduce_attrs(NodeId(1)).unwrap();
    assert_eq!(attrs.axes_mask, 0b10);
    assert!(attrs.keepdims);
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_parses_scalar_attrs_to_graph_attrs() {
    let source = r#"
fn graph(h: &mut HologramBuilder) {
    let a = h.input("a", shape([2, 3]));
    let b = h.input("b", shape([3, 2]));
    let c = h.input("c", shape([2, 2]));
    let y = h.ops().gemm(a, b, c, shape([2, 2]), alpha(0.5), beta(0.25));
    h.output("y", y);
}
"#;
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Rust).unwrap()).unwrap();
    let attrs = graph.gemm_attrs(NodeId(3)).unwrap();
    assert_eq!(attrs.alpha_bits, 0.5f32.to_bits());
    assert_eq!(attrs.beta_bits, 0.25f32.to_bits());
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_frontend_rejects_attrs_on_wrong_op() {
    let source = r#"
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", shape([2, 3]));
    let y = h.ops().relu(x, shape([2, 3]), keepdims(true));
    h.output("y", y);
}
"#;
    let err =
        parse_ir_diagnostic(source, SourceLanguage::Rust).expect_err("attr must be op scoped");
    assert_eq!(err.kind, "op: attr not valid");
    assert_eq!(err.line, 4);
    assert!(err.rejected.contains("keepdims(true)"));
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_source_matches_native_matmul_archive() {
    let native = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let rust = r#"
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", shape([2, 3]));
    let w = h.constant("w", shape([3, 2]), values([1, 2, 3, 4, 5, 6]));
    let y = h.ops().matmul(x, w, shape([2, 2]));
    h.output(y);
}
"#;
    assert_language_sources_equivalent(native, rust, SourceLanguage::Rust);
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_source_matches_native_reduce_attrs_archive() {
    let native = "
        input x: f32[2, 3]
        let y: f32[2] = reduce_sum(x, axes=[1], keepdims=true)
        output y
    ";
    let rust = r#"
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", shape([2, 3]));
    let y = h.ops().reduce_sum(x, shape([2]), axes([1]), keepdims(true));
    h.output(y);
}
"#;
    assert_language_sources_equivalent(native, rust, SourceLanguage::Rust);
}

#[cfg(feature = "frontend-rust")]
#[test]
fn rust_source_matches_native_gemm_attrs_archive() {
    let native = "
        input a: f32[2, 3]
        input b: f32[3, 2]
        input c: f32[2, 2]
        let y: f32[2, 2] = gemm(a, b, c, alpha=0.5, beta=0.25)
        output y
    ";
    let rust = r#"
fn graph(h: &mut HologramBuilder) {
    let a = h.input("a", shape([2, 3]));
    let b = h.input("b", shape([3, 2]));
    let c = h.input("c", shape([2, 2]));
    let y = h.ops().gemm(a, b, c, shape([2, 2]), alpha(0.5), beta(0.25));
    h.output(y);
}
"#;
    assert_language_sources_equivalent(native, rust, SourceLanguage::Rust);
}

#[cfg(all(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
#[test]
fn all_frontends_match_native_matmul_archive() {
    let native = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let python = r#"
def graph(h):
    x = h.input("x", shape=[2, 3])
    w = h.const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    y = h.ops.matmul(x, w, shape=[2, 2])
    h.output(y)
"#;
    let typescript = r#"
function graph(h) {
    const x = h.input("x", { shape: [2, 3] });
    const w = h.const("w", { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] });
    const y = h.ops.matmul(x, w, { shape: [2, 2] });
    h.output(y);
}
"#;
    let rust = r#"
fn graph(h: &mut HologramBuilder) {
    let x = h.input("x", shape([2, 3]));
    let w = h.constant("w", shape([3, 2]), values([1, 2, 3, 4, 5, 6]));
    let y = h.ops().matmul(x, w, shape([2, 2]));
    h.output(y);
}
"#;
    assert_language_sources_equivalent(native, python, SourceLanguage::Python);
    assert_language_sources_equivalent(native, typescript, SourceLanguage::TypeScript);
    assert_language_sources_equivalent(native, rust, SourceLanguage::Rust);
}

#[test]
fn compile_from_source_language_keeps_native_path() {
    let source = "input x\nop relu x as=y\noutput y\n";
    let output = compile_from_source_language(
        source,
        SourceLanguage::Hologram,
        WittLevel::W32,
        BackendKind::Cpu,
    )
    .expect("native source compiles through language API");
    assert!(output.stats.total_nodes >= 3);
}

#[test]
fn parses_native_v2_source_to_ir() {
    let source = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let program = parse_ir(source, SourceLanguage::Hologram).expect("v2 source parses");
    assert_eq!(program.items().len(), 4);
}

#[test]
fn parses_native_v2_reduce_attrs_to_graph_attrs() {
    let source = "
        input x: f32[2, 3]
        let y: f32[2] = reduce_sum(x, axes=[1], keepdims=true)
        output y
    ";
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Hologram).unwrap()).unwrap();
    let attrs = graph.reduce_attrs(NodeId(1)).unwrap();
    assert_eq!(attrs.axes_mask, 0b10);
    assert!(attrs.keepdims);
}

#[test]
fn parses_native_v2_conv_attrs_to_graph_attrs() {
    let source = "
        input x: f32[1, 1, 4, 4]
        input w: f32[1, 1, 3, 3]
        let y: f32[1, 1, 2, 2] = conv2d(x, w, stride=[2, 3], pads=[1, 2, 1, 2], kernel=[3, 3])
        output y
    ";
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Hologram).unwrap()).unwrap();
    let attrs = graph.conv_attrs(NodeId(2)).unwrap();
    assert_eq!((attrs.stride_h, attrs.stride_w), (2, 3));
    assert_eq!((attrs.pad_h, attrs.pad_w), (1, 2));
    assert_eq!((attrs.k_h, attrs.k_w), (3, 3));
}

#[test]
fn parses_native_v2_scalar_attrs_to_graph_attrs() {
    let source = "
        input a: f32[2, 3]
        input b: f32[3, 2]
        input c: f32[2, 2]
        let y: f32[2, 2] = gemm(a, b, c, alpha=0.5, beta=0.25)
        output y
    ";
    let graph = lower_ir(&parse_ir(source, SourceLanguage::Hologram).unwrap()).unwrap();
    let attrs = graph.gemm_attrs(NodeId(3)).unwrap();
    assert_eq!(attrs.alpha_bits, 0.5f32.to_bits());
    assert_eq!(attrs.beta_bits, 0.25f32.to_bits());
}

#[test]
fn rejects_native_v2_attrs_on_wrong_op() {
    let source = "
        input x: f32[2, 3]
        let y: f32[2, 3] = relu(x, keepdims=true)
    ";
    let err = parse_ir(source, SourceLanguage::Hologram).expect_err("attr must be op scoped");
    assert!(format!("{err:?}").contains("attr not valid"));
}

#[test]
fn native_v2_diagnostics_report_line_column_and_token() {
    let source = "
        input x: f32[2, 3]
        output x extra
    ";
    let err = parse_ir_diagnostic(source, SourceLanguage::Hologram).unwrap_err();
    let line = source.lines().nth(2).unwrap();
    assert_eq!(err.line, 3);
    assert_eq!(err.column, line.find("extra").unwrap() + 1);
    assert_eq!(err.kind, "source: bad v2 syntax");
    assert_eq!(err.rejected, "extra");
}

#[test]
fn legacy_diagnostics_report_line_column_and_token() {
    let source = "
        input x
        wat x
    ";
    let err = parse_ir_diagnostic(source, SourceLanguage::Hologram).unwrap_err();
    let line = source.lines().nth(2).unwrap();
    assert_eq!(err.line, 3);
    assert_eq!(err.column, line.find("wat").unwrap() + 1);
    assert_eq!(err.kind, "unknown directive");
    assert_eq!(err.rejected, "wat");
}

#[test]
fn compiles_native_v2_source() {
    let source = "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    ";
    let output = compile_from_source_language(
        source,
        SourceLanguage::Hologram,
        WittLevel::W32,
        BackendKind::Cpu,
    )
    .expect("v2 source compiles");
    assert!(output.stats.validated_units >= 1);
}

#[test]
fn source_types_do_not_leak_into_runtime_crates() {
    let root = workspace_root();
    for runtime_crate in [
        "crates/hologram-exec/src",
        "crates/hologram-backend/src",
        "crates/hologram-archive/src",
    ] {
        assert_no_source_type_refs(&root.join(runtime_crate));
    }
}

fn shape_2x2() -> ShapeDescriptor {
    ShapeDescriptor::rank2(2, 2)
}

fn shape_2x3() -> ShapeDescriptor {
    ShapeDescriptor::rank2(2, 3)
}

fn shape_3x2() -> ShapeDescriptor {
    ShapeDescriptor::rank2(3, 2)
}

fn shape_1() -> ShapeDescriptor {
    ShapeDescriptor::rank1(1)
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    HologramHasher::initial().fold_bytes(bytes).finalize()
}

fn write_temp_tensor(bytes: &[u8]) -> std::path::PathBuf {
    let path = unique_temp_path("hologram-source-external").with_extension("bin");
    std::fs::write(&path, bytes).expect("write temp tensor");
    path
}

fn unique_temp_path(prefix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
}

struct ExternalEnvGuard {
    previous: Option<std::ffi::OsString>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl Drop for ExternalEnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("HOLOGRAM_EXTERNAL_TENSOR_ROOT", value),
            None => std::env::remove_var("HOLOGRAM_EXTERNAL_TENSOR_ROOT"),
        }
    }
}

fn external_env_guard() -> ExternalEnvGuard {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    let previous = std::env::var_os("HOLOGRAM_EXTERNAL_TENSOR_ROOT");
    std::env::remove_var("HOLOGRAM_EXTERNAL_TENSOR_ROOT");
    ExternalEnvGuard {
        previous,
        _guard: guard,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn golden_native_matmul_source() -> &'static str {
    "
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    "
}

fn external_matmul_program(path: &Path, bytes: &[u8]) -> SourceProgram {
    let mut program = SourceProgram::new();
    let (x, w, y) = matmul_symbols(&mut program);
    program.push(SourceItem::Input(SourceInput::new(
        x,
        SourceType::f32(Some(shape_2x3())),
    )));
    program.push(external_weight(w, path, bytes));
    program.push(matmul_binding(x, w, y));
    program.push(SourceItem::Output(SourceOutput::new(y)));
    program
}

fn matmul_symbols(
    program: &mut SourceProgram,
) -> (
    hologram_compiler::source::SourceSymbol,
    hologram_compiler::source::SourceSymbol,
    hologram_compiler::source::SourceSymbol,
) {
    (
        program.intern("x"),
        program.intern("w"),
        program.intern("y"),
    )
}

fn external_weight(
    name: hologram_compiler::source::SourceSymbol,
    path: &Path,
    bytes: &[u8],
) -> SourceItem {
    let reference = SourceExternalTensor::file(
        path.to_string_lossy().to_string(),
        0,
        bytes.len() as u64,
        digest(bytes),
    );
    SourceItem::ExternalConst(SourceExternalConst::new(
        name,
        SourceType::f32(Some(shape_3x2())),
        reference,
    ))
}

fn matmul_binding(
    x: hologram_compiler::source::SourceSymbol,
    w: hologram_compiler::source::SourceSymbol,
    y: hologram_compiler::source::SourceSymbol,
) -> SourceItem {
    let ty = Some(SourceType::f32(Some(shape_2x2())));
    let call = SourceOpCall::new(OpKind::MatMul, vec![x, w], ty);
    SourceItem::Binding(SourceBinding::op(Some(y), call))
}

trait ItemName {
    fn name(&self) -> hologram_compiler::source::SourceSymbol;
}

impl ItemName for SourceItem {
    fn name(&self) -> hologram_compiler::source::SourceSymbol {
        match self {
            SourceItem::Input(input) => input.name,
            SourceItem::Const(constant) => constant.name,
            SourceItem::ExternalConst(constant) => constant.name,
            SourceItem::Binding(binding) => binding.name.expect("named binding"),
            SourceItem::Output(output) => output.name,
        }
    }
}

fn simple_program(input_name: &str) -> SourceProgram {
    let mut program = SourceProgram::new();
    let symbol = program.intern(input_name);
    program.push(SourceItem::Input(SourceInput::new(
        symbol,
        SourceType::f32(None),
    )));
    program
}

#[cfg(feature = "frontend-typescript")]
fn assert_sources_equivalent(native: &str, typescript: &str) {
    assert_language_sources_equivalent(native, typescript, SourceLanguage::TypeScript);
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_language_sources_equivalent(native: &str, source: &str, language: SourceLanguage) {
    let native_graph = graph_from_source(native, SourceLanguage::Hologram);
    let host_graph = graph_from_source(source, language);
    assert_graphs_equivalent(&native_graph, &host_graph);
    assert_archives_equivalent(native, source, language);
}

fn graph_from_source(source: &str, language: SourceLanguage) -> hologram_graph::Graph {
    let program = parse_ir(source, language).expect("source parses");
    lower_ir(&program).expect("source lowers")
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_archives_equivalent(native: &str, source: &str, language: SourceLanguage) {
    let native = archive_from_source(native, SourceLanguage::Hologram);
    let host = archive_from_source(source, language);
    assert_eq!(native, host);
}

fn archive_from_source(source: &str, language: SourceLanguage) -> Vec<u8> {
    compile_from_source_language(source, language, WittLevel::W32, BackendKind::Cpu)
        .expect("source compiles")
        .archive
}

fn archive_from_program(program: &SourceProgram) -> Vec<u8> {
    let graph = lower_ir(program).expect("program lowers");
    Compiler::new(graph, BackendKind::Cpu, WittLevel::W32)
        .compile()
        .expect("program compiles")
        .archive
}

fn constant_bytes(graph: &hologram_graph::Graph, id: u32) -> &[u8] {
    let id = hologram_graph::node::ConstantId(id);
    &graph.constants().get(id).expect("constant exists").bytes
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_graphs_equivalent(left: &hologram_graph::Graph, right: &hologram_graph::Graph) {
    assert_eq!(left.node_count(), right.node_count());
    assert_eq!(left.inputs(), right.inputs());
    assert_eq!(left.outputs(), right.outputs());
    assert_constants_equivalent(left, right);
    for index in 0..left.node_count() {
        assert_node_equivalent(left, right, NodeId(index as u32));
    }
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_node_equivalent(left: &hologram_graph::Graph, right: &hologram_graph::Graph, id: NodeId) {
    let left_node = left.get(id).expect("left node exists");
    let right_node = right.get(id).expect("right node exists");
    assert_eq!(left_node.op, right_node.op);
    assert_eq!(left_node.inputs, right_node.inputs);
    assert_eq!(left_node.output_dtype, right_node.output_dtype);
    assert_eq!(left_node.output_shape, right_node.output_shape);
    assert_node_attrs_equivalent(left, right, id);
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_node_attrs_equivalent(
    left: &hologram_graph::Graph,
    right: &hologram_graph::Graph,
    id: NodeId,
) {
    assert_eq!(left.quant_attrs(id), right.quant_attrs(id));
    assert_eq!(left.conv_attrs(id), right.conv_attrs(id));
    assert_eq!(left.lrn_attrs(id), right.lrn_attrs(id));
    assert_eq!(left.gemm_attrs(id), right.gemm_attrs(id));
    assert_eq!(left.norm_attrs(id), right.norm_attrs(id));
    assert_eq!(left.reduce_attrs(id), right.reduce_attrs(id));
    assert_eq!(left.gather_attrs(id), right.gather_attrs(id));
    assert_eq!(left.attention_attrs(id), right.attention_attrs(id));
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_constants_equivalent(left: &hologram_graph::Graph, right: &hologram_graph::Graph) {
    assert_eq!(left.constants().len(), right.constants().len());
    for index in 0..left.constants().len() {
        assert_constant_equivalent(left, right, index as u32);
    }
}

#[cfg(any(
    feature = "frontend-python",
    feature = "frontend-typescript",
    feature = "frontend-rust"
))]
fn assert_constant_equivalent(
    left: &hologram_graph::Graph,
    right: &hologram_graph::Graph,
    id: u32,
) {
    let id = hologram_graph::node::ConstantId(id);
    let left = left.constants().get(id).expect("left constant exists");
    let right = right.constants().get(id).expect("right constant exists");
    assert_eq!(left.bytes, right.bytes);
    assert_eq!(left.dtype, right.dtype);
    assert_eq!(left.shape, right.shape);
}

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn assert_no_source_type_refs(path: &Path) {
    for entry in std::fs::read_dir(path).expect("read runtime crate") {
        let entry = entry.expect("runtime crate entry");
        let path = entry.path();
        if path.is_dir() {
            assert_no_source_type_refs(&path);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            assert_file_has_no_source_type_refs(&path);
        }
    }
}

fn assert_file_has_no_source_type_refs(path: &Path) {
    let text = std::fs::read_to_string(path).expect("read runtime source");
    for forbidden in FORBIDDEN_RUNTIME_SOURCE_REFS {
        assert!(
            !text.contains(forbidden),
            "{} must not reference compiler source type {forbidden}",
            path.display()
        );
    }
}

const FORBIDDEN_RUNTIME_SOURCE_REFS: &[&str] = &[
    "hologram_compiler::source",
    "SourceProgram",
    "SourceDocument",
    "SourceGraph",
    "SourceParseOptions",
    "SourceLanguage",
    "SourceItem",
    "SourceExpr",
    "SourceType",
    "SourceDiagnostic",
];
