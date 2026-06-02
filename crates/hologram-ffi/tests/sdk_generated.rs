use hologram_ffi::sdk;
use hologram_graph::OpKind;

const PYTHON: &str = include_str!("../../../sdk/python/hologram/_generated.py");
const TYPESCRIPT: &str = include_str!("../../../sdk/typescript/src/generated.ts");

#[test]
fn generated_python_is_current() {
    assert_eq!(PYTHON, sdk::generate_python());
}

#[test]
fn generated_typescript_is_current() {
    assert_eq!(TYPESCRIPT, sdk::generate_typescript());
}

#[test]
fn sdk_metadata_covers_all_ops() {
    let ops: Vec<_> = sdk::ops().collect();
    assert_eq!(ops.len(), OpKind::ALL.len());
    for (meta, kind) in ops.iter().zip(OpKind::ALL.iter().copied()) {
        assert_eq!(meta.kind, kind);
        assert_eq!(meta.name, kind.name());
        assert_eq!(meta.arity, kind.primary_arity());
        assert!(!meta.dtype_policy.is_empty());
        assert!(!meta.shape_policy.is_empty());
        assert!(!meta.doc.is_empty());
    }
}

#[test]
fn generated_files_include_attr_metadata() {
    assert!(PYTHON.contains(
        "\"reduce_sum\": OpSpec(\"reduce_sum\", 1, (\"axes\", \"axes_mask\", \"keepdims\")"
    ));
    assert!(TYPESCRIPT.contains(
        "reduce_sum: { name: \"reduce_sum\", arity: 1, attrs: [\"axes\", \"axes_mask\", \"keepdims\"] as const"
    ));
    assert!(PYTHON.contains(
        "\"dequantize\": OpSpec(\"dequantize\", 1, (\"axis\", \"scale\", \"scale_bits\", \"quant_dtype\", \"zero_point\")"
    ));
    assert!(TYPESCRIPT.contains(
        "dequantize: { name: \"dequantize\", arity: 1, attrs: [\"axis\", \"scale\", \"scale_bits\", \"quant_dtype\", \"zero_point\"] as const"
    ));
}

#[test]
fn generated_files_include_rich_metadata_and_methods() {
    assert!(PYTHON.contains(
        "\"matmul\": OpSpec(\"matmul\", 2, (), \"f32-source-builder\", \"rank-2-concrete\""
    ));
    assert!(PYTHON.contains("def matmul(tensor, *inputs, **attrs):"));
    assert!(TYPESCRIPT.contains("export type OpArity<Name extends OpName>"));
    assert!(TYPESCRIPT.contains("export interface GeneratedTensorMethods"));
    assert!(TYPESCRIPT
        .contains("matmul(input0: TInput, attrs?: TAttrs & OpOptionsFor<\"matmul\">): TTensor;"));
}
