//! Named + shaped I/O ports survive compile → archive → session load, and are
//! queryable by name (so a caller maps `input_ids`/… to the right `execute`
//! position) — the multi-input model requirement.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

#[test]
fn named_ports_round_trip_through_session() {
    // out = relu(x) ; name the input "features" and output "activations".
    let mut graph = Graph::new();
    let sh = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(2, 3));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_named_input(x, "features");
    let relu = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_named_output(out, "activations");

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let session = InferenceSession::load(&compiled.archive, backend).unwrap();

    let (idx, port) = session
        .input_port_by_name("features")
        .expect("input bound by name");
    assert_eq!(idx, 0);
    assert_eq!(port.shape, vec![2, 3]);
    assert_eq!(port.element_count, 6);

    let (_, oport) = session
        .output_port_by_name("activations")
        .expect("output bound by name");
    assert_eq!(oport.shape, vec![2, 3]);

    assert!(session.input_port_by_name("nonexistent").is_none());
}

#[test]
fn graph_extensions_reach_the_session() {
    // A producer attaches opaque metadata on the graph; it must survive
    // compile → archive → load and be fetchable by key at runtime.
    let mut graph = Graph::new();
    let sh = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(2));
    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_input(x);
    let relu = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    graph.add_output(out);
    graph.add_extension("tokenizer.json", b"{\"model\":\"bpe\"}".to_vec());
    graph.add_extension("generation_config", vec![7, 8, 9]);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let session = InferenceSession::load(&compiled.archive, backend).unwrap();

    assert_eq!(
        session.extension("tokenizer.json").unwrap(),
        b"{\"model\":\"bpe\"}"
    );
    assert_eq!(session.extension("generation_config").unwrap(), &[7, 8, 9]);
    assert!(session.extension("absent").is_none());
    let keys: Vec<&str> = session.extension_keys().collect();
    assert_eq!(keys, vec!["tokenizer.json", "generation_config"]);
}
