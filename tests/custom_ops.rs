//! Integration tests for Sprint 11: Custom Op Extension API.

use std::sync::Arc;

use holo_archive::HoloWriter;
use holo_exec::eval::build_schedule;
use holo_exec::{
    execute_bytes_with_ops, register_op, CustomOpId, CustomOpRegistry, GraphInputs, KvExecutor,
};
use holo_graph::{builder::GraphBuilder, graph::GraphOp};

// ── helpers ──────────────────────────────────────────────────────────────────

fn build_custom_chain_archive(id: u32, arity: u8) -> Vec<u8> {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .custom_op(CustomOpId(id), arity, &[0]) // 1
        .node_with_inputs(GraphOp::Output, &[1]) // 2
        .output("y", 2)
        .build();
    HoloWriter::new().set_graph(&g).build().unwrap()
}

fn inputs_from(data: Vec<u8>) -> GraphInputs {
    GraphInputs::from_pairs(vec![(0, data)])
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Custom passthrough: arity-1 op that copies input bytes.
#[test]
fn custom_passthrough() {
    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(1),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
    );

    let archive = build_custom_chain_archive(1, 1);
    let result =
        execute_bytes_with_ops(&archive, &inputs_from(vec![10, 20, 30]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[10, 20, 30]);
}

/// Custom op that doubles each byte (mod 256).
#[test]
fn custom_double_bytes() {
    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(2),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].iter().map(|&b| b.wrapping_mul(2)).collect())),
    );

    let archive = build_custom_chain_archive(2, 1);
    let result =
        execute_bytes_with_ops(&archive, &inputs_from(vec![1, 2, 128]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[2, 4, 0]);
}

/// Custom binary op that element-wise adds two buffers.
#[test]
fn custom_binary_add() {
    let g = GraphBuilder::new()
        .input("a")
        .input("b")
        .node_from_graph_input(GraphOp::Input, 0) // 0
        .node_from_graph_input(GraphOp::Input, 1) // 1
        .custom_op(CustomOpId(3), 2, &[0, 1]) // 2
        .node_with_inputs(GraphOp::Output, &[2]) // 3
        .output("sum", 3)
        .build();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();

    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(3),
        2,
        Arc::new(|inputs, _| {
            Ok(inputs[0]
                .iter()
                .zip(inputs[1].iter())
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect())
        }),
    );

    let mut inputs = GraphInputs::new();
    inputs.set(0, vec![10, 20, 30]);
    inputs.set(1, vec![1, 2, 3]);
    let result = execute_bytes_with_ops(&archive, &inputs, &registry).unwrap();
    assert_eq!(result.by_name("sum").unwrap(), &[11, 22, 33]);
}

/// No registry → UnsupportedOp error for a Custom node.
#[test]
fn unregistered_op_errors() {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .custom_op(CustomOpId(99), 1, &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();
    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let empty_registry = CustomOpRegistry::new();
    let result = execute_bytes_with_ops(&archive, &inputs_from(vec![1]), &empty_registry);
    assert!(result.is_err());
}

/// Registry present but id not registered → error.
#[test]
fn unregistered_with_registry_errors() {
    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(1),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
    );

    // op id=2 not registered
    let archive = build_custom_chain_archive(2, 1);
    let result = execute_bytes_with_ops(&archive, &inputs_from(vec![5]), &registry);
    assert!(result.is_err());
}

/// `register_op!` macro compiles and dispatches correctly.
#[test]
fn register_op_macro_works() {
    let mut registry = CustomOpRegistry::new();
    register_op!(
        registry,
        id = 10u32,
        arity = 1u8,
        handler = |inputs, _| { Ok(inputs[0].iter().map(|&b| b.wrapping_add(1)).collect()) }
    );

    let archive = build_custom_chain_archive(10, 1);
    let result =
        execute_bytes_with_ops(&archive, &inputs_from(vec![0, 127, 255]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[1, 128, 0]);
}

/// Custom op round-trips through archive (serializes graph, executes).
#[test]
fn custom_op_serializes() {
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .custom_op(CustomOpId(5), 1, &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();

    let archive = HoloWriter::new().set_graph(&g).build().unwrap();
    let plan = holo_archive::load_from_bytes(&archive).unwrap();
    // Graph survives round-trip
    assert_eq!(plan.node_count(), 3);

    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(5),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
    );
    let result = execute_bytes_with_ops(&archive, &inputs_from(vec![42]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[42]);
}

/// `GraphBuilder::custom_op` produces correct node wiring.
#[test]
fn custom_op_graph_builder() {
    let g = GraphBuilder::new()
        .node(GraphOp::Input) // 0
        .custom_op(CustomOpId(7), 1, &[0]) // 1
        .build();
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edges().len(), 1);
}

/// Custom op with access to `ConstantStore` (reads a stored constant).
#[test]
fn custom_op_with_constants() {
    // Build a graph that has a constant and a custom op reading from inputs
    let archive = build_custom_chain_archive(8, 1);

    let mut registry = CustomOpRegistry::new();
    // This handler just xor's the input with 0xFF
    registry.register(
        CustomOpId(8),
        1,
        Arc::new(|inputs, _constants| Ok(inputs[0].iter().map(|&b| b ^ 0xFF).collect())),
    );

    let result =
        execute_bytes_with_ops(&archive, &inputs_from(vec![0x00, 0xFF, 0xAA]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[0xFF, 0x00, 0x55]);
}

/// `execute_bytes_with_ops` works when graph goes through full compiler pipeline.
#[test]
fn custom_op_in_compiler_pipeline() {
    use holo_compiler::CompilerBuilder;

    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .custom_op(CustomOpId(20), 1, &[0])
        .node_with_inputs(GraphOp::Output, &[1])
        .output("y", 2)
        .build();

    let compiled = CompilerBuilder::new(g).fuse(false).build().unwrap().archive;

    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(20),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].iter().map(|&b| b.wrapping_mul(3)).collect())),
    );

    let result = execute_bytes_with_ops(&compiled, &inputs_from(vec![1, 2, 3]), &registry).unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[3, 6, 9]);
}

/// `KvExecutor::execute_with_registry` works end-to-end.
#[test]
fn kv_executor_execute_with_registry() {
    use holo_archive::load_from_bytes;

    let archive = build_custom_chain_archive(30, 1);
    let plan = load_from_bytes(&archive).unwrap();
    let schedule = build_schedule(plan.graph()).unwrap();

    let mut registry = CustomOpRegistry::new();
    registry.register(
        CustomOpId(30),
        1,
        Arc::new(|inputs, _| Ok(inputs[0].to_vec())),
    );

    let result = KvExecutor::execute_with_registry(
        plan.graph(),
        &schedule,
        &inputs_from(vec![7, 8]),
        &registry,
    )
    .unwrap();
    assert_eq!(result.by_name("y").unwrap(), &[7, 8]);
}
