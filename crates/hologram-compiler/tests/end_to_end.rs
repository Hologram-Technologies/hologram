//! Spec XII.3: a representative `Graph` containing one of every `OpKind`
//! compiles to a `.holo` archive without error. Empty-graph baseline below;
//! exhaustive op coverage layers on as kernels mature.

use hologram_compiler::{BackendKind, Compiler};
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{ConvAttrs, Graph, GraphOp, InputSource, Node, NodeId, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

#[test]
fn empty_graph_compiles() {
    let g = Graph::new();
    let out = Compiler::new(g, BackendKind::Cpu, WittLevel::W32)
        .compile()
        .unwrap();
    assert!(out.archive.len() >= 4 + 2 + 2 + 2 + 32);
    assert_eq!(&out.archive[..4], b"HOLO");
}

#[test]
fn empty_graph_compile_then_load() {
    let g = Graph::new();
    let out = Compiler::new(g, BackendKind::Cpu, WittLevel::W32)
        .compile()
        .unwrap();
    let plan = hologram_archive::HoloLoader::from_bytes(&out.archive)
        .unwrap()
        .into_plan()
        .unwrap();
    assert!(!plan.sections().is_empty());
}

#[test]
fn conv_attrs_thread_through_compile() {
    // Conv2d with `ConvAttrs { stride = (2, 2), pad = (1, 1) }` must
    // reach the lowered KernelCall — previously the compiler hardcoded
    // (1, 0). Smoke-test by compiling and confirming the archive parses.
    let mut g = Graph::new();
    let shape_x = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 1, 4, 4));
    let shape_w = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 1, 3, 3));
    let shape_y = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(1, 1, 2, 2));
    let dtype = DTypeId(8); // F32
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: dtype,
        output_shape: shape_x,
    });
    g.add_input(x);
    let w = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: dtype,
        output_shape: shape_w,
    });
    g.add_input(w);
    let conv = g.add_node(Node {
        op: GraphOp::Op(OpKind::Conv2d),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Node(w)]),
        output_dtype: dtype,
        output_shape: shape_y,
    });
    g.set_conv_attrs(
        NodeId(conv.0),
        ConvAttrs {
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
        },
    );
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(conv)]),
        output_dtype: dtype,
        output_shape: shape_y,
    });
    g.add_output(out);

    let archive = Compiler::new(g, BackendKind::Cpu, WittLevel::W32)
        .compile()
        .expect("conv graph compiles");
    let plan =
        hologram_archive::HoloLoader::from_bytes(&archive.archive).expect("archive verifies");
    let _ = plan.into_plan().expect("plan parses");
}
