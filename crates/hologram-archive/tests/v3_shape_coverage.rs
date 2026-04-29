//! ADR-053 step 6: round-trip shape coverage tests.
//!
//! Verifies that v3 archives carry a `node_shapes` entry for every
//! dispatch-producing node and a `constant_shapes` entry for every
//! referenced constant — both at write time (writer rejects missing
//! coverage) and after a full rkyv round-trip (load preserves what was
//! written).

use hologram_archive::error::ArchiveError;
use hologram_archive::format::graph::SerializedGraph;
use hologram_archive::format::header::HoloHeader;
use hologram_archive::format::FORMAT_VERSION;
use hologram_archive::writer::holo_writer::HoloWriter;
use hologram_archive::{entrypoint::schedule::LayerHeader, load_from_bytes};
use hologram_core::op::{LutOp, PrimOp};
use hologram_graph::builder::GraphBuilder;
use hologram_graph::constant::ConstantData;
use hologram_graph::graph::GraphOp;

/// A representative graph: input → relu → constant fold-with → add → output.
/// Exercises Lut, Prim, and Constant variants — the three op families that
/// participate in shape coverage checks.
fn representative_graph() -> hologram_graph::Graph {
    GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .set_node_shape(1, vec![4])
        .constant_with_shape(ConstantData::Bytes(vec![1, 2, 3, 4]), vec![4])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        .set_node_shape(3, vec![4])
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build()
}

#[test]
fn v3_archive_carries_full_shape_coverage() {
    let g = representative_graph();
    let archive = HoloWriter::new()
        .set_graph(&g)
        .add_section(&LayerHeader::new())
        .build()
        .expect("v3 writer accepts fully-populated graph");

    // Inspect the header — must report v3.
    let header =
        HoloHeader::from_bytes(&archive[..std::mem::size_of::<HoloHeader>()]).expect("header");
    assert_eq!(
        header.version, FORMAT_VERSION,
        "writer should emit current FORMAT_VERSION"
    );
    assert!(header.is_supported_version());
    assert!(!header.needs_v2_shape_compat());

    // Round-trip: load and re-validate coverage on the decoded SerializedGraph.
    let plan = load_from_bytes(&archive).expect("load v3 archive");
    let sg: &SerializedGraph = plan.graph();
    sg.validate_shape_coverage()
        .expect("decoded archive preserves shape coverage");

    // Every non-Input/non-Output/non-Constant node has a shape entry.
    let nodes_with_shape: std::collections::HashSet<_> =
        sg.node_shapes.iter().map(|(id, _)| *id).collect();
    let constants_with_shape: std::collections::HashSet<_> =
        sg.constant_shapes.iter().map(|(id, _)| *id).collect();

    for node in &sg.nodes {
        match &node.op {
            GraphOp::Input | GraphOp::Output => {}
            GraphOp::Constant(cid) => {
                assert!(
                    constants_with_shape.contains(cid),
                    "constant {cid:?} missing shape after round-trip"
                );
            }
            other => {
                assert!(
                    nodes_with_shape.contains(&node.id),
                    "node {:?} (op {other:?}) missing shape after round-trip",
                    node.id
                );
            }
        }
    }
}

#[test]
fn writer_rejects_graph_missing_node_shape() {
    // Same shape as `representative_graph` but the Add node (idx 3) has no
    // shape set. The writer should reject this rather than emit silently.
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .node_with_inputs(GraphOp::Lut(LutOp::Relu), &[0])
        .set_node_shape(1, vec![4])
        .constant_with_shape(ConstantData::Bytes(vec![1, 2, 3, 4]), vec![4])
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[1, 2])
        // Intentionally omitting `.set_node_shape(3, ...)`.
        .node_with_inputs(GraphOp::Output, &[3])
        .output("y", 4)
        .build();

    let err = HoloWriter::new().set_graph(&g).build().unwrap_err();
    match err {
        ArchiveError::MissingNodeShape { .. } => {}
        other => panic!("expected MissingNodeShape, got {other:?}"),
    }
}

#[test]
fn writer_rejects_graph_missing_constant_shape() {
    // Constant added without a shape — writer should reject.
    let g = GraphBuilder::new()
        .input("x")
        .node_from_graph_input(GraphOp::Input, 0)
        .constant(ConstantData::Bytes(vec![1, 2, 3, 4]))
        .node_with_inputs(GraphOp::Prim(PrimOp::Add), &[0, 1])
        .set_node_shape(2, vec![4])
        .node_with_inputs(GraphOp::Output, &[2])
        .output("y", 3)
        .build();

    let err = HoloWriter::new().set_graph(&g).build().unwrap_err();
    match err {
        ArchiveError::MissingConstantShape { .. } => {}
        other => panic!("expected MissingConstantShape, got {other:?}"),
    }
}
