//! Workspace size scaling (spec VIII.3 + X-7 trillion-param claim).
//!
//! Asserts the workspace memory footprint scales with the SUM of slot
//! sizes, not with `slot_count * max_size`. A regression here would
//! make trillion-parameter / UHD-streaming workloads infeasible.

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession};
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

#[test]
fn workspace_total_is_sum_not_product() {
    // Mixed-size graph: one "large" tensor (4096 elements = 16 KiB at f32)
    // alongside three small tensors (4 elements each). Old layout would
    // allocate 4 × 16 KiB = 64 KiB; new layout allocates ~16 KiB + 3·256
    // (the 64-byte floor on small slots).
    let mut graph = Graph::new();
    let big_shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(4096));
    let small_shape = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));

    let big_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big_shape,
    });
    graph.add_input(big_in);

    let big_relu = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(big_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big_shape,
    });
    let big_out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(big_relu)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big_shape,
    });
    graph.add_output(big_out);

    // Three small ops sharing the small shape; runtime will allocate
    // 64-byte slots for them (the floor) — far less than the 16 KiB
    // big slot.
    let small_in = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small_shape,
    });
    graph.add_input(small_in);
    let small_a = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Sigmoid),
        inputs: SmallVec::from_iter([InputSource::Node(small_in)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small_shape,
    });
    let small_b = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Tanh),
        inputs: SmallVec::from_iter([InputSource::Node(small_a)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small_shape,
    });
    let small_out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(small_b)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small_shape,
    });
    graph.add_output(small_out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let cap = session.workspace().capacity();
    // Two big slots (input + relu output, each 16 KiB) plus four small
    // slots × 64 B floor ≈ 33 KiB. Old layout would have been
    // 7 × 16 KiB = 112 KiB. Threshold is generous (64 KiB) so tighter
    // schedule-aware liveness compaction stays within it without a flake.
    assert!(
        cap < 64 * 1024,
        "workspace capacity {} bytes too large — per-slot sizing regressed",
        cap
    );
}

#[test]
fn one_giant_tensor_doesnt_explode_workspace() {
    // A graph whose only big tensor is a single 1 MiB input (not
    // realistic for tests, but demonstrates the layout principle).
    // The workspace should be approximately 1 MiB, NOT n_slots * 1 MiB.
    let mut graph = Graph::new();
    let big_n: u64 = 256 * 1024; // 1 MiB at f32
    let big = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(big_n));
    let small = graph.shape_registry_mut().intern(ShapeDescriptor::rank1(4));

    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big,
    });
    graph.add_input(x);
    let y = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(x)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(y)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: big,
    });
    graph.add_output(out);

    // Pad with five small intermediate ops sharing the small shape.
    let mut prev = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small,
    });
    graph.add_input(prev);
    for _ in 0..5 {
        prev = graph.add_node(Node {
            op: GraphOp::Op(OpKind::Sigmoid),
            inputs: SmallVec::from_iter([InputSource::Node(prev)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: small,
        });
    }
    let small_out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(prev)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: small,
    });
    graph.add_output(small_out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let cap = session.workspace().capacity();
    // Big-tensor pathway holds 2 slots × 1 MiB = 2 MiB. Small ops add
    // ~7 × 64 B ≈ 448 B. Old layout would have been ~9 × 1 MiB = 9 MiB.
    let big_bytes = (big_n as usize) * 4;
    assert!(
        cap < (big_bytes * 4),
        "workspace ({} bytes) exceeds 4× the big-tensor footprint ({}); \
         per-slot scaling regressed",
        cap,
        big_bytes
    );
}
