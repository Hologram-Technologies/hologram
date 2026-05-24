//! Fusion pass throughput benchmark.
//!
//! Measures the time to run the fusion engine on graphs of varying
//! sizes (10, 100, 1000 nodes). The fusion pass itself should be
//! negligible compared to execution savings.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use hologram_graph::node::Node;
use hologram_graph::registry::DTypeId;
use hologram_graph::fusion;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn build_chain_graph(n: usize) -> Graph {
    let mut g = Graph::new();
    let shape = g.shape_registry_mut()
        .intern(hologram_graph::registry::ShapeDescriptor::rank1(64));

    let input = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    g.add_input(input);

    // Alternate fusable activations to exercise the chain fusion pass.
    let ops = [OpKind::Relu, OpKind::Sigmoid, OpKind::Tanh, OpKind::Gelu, OpKind::Silu];
    let mut prev = input;
    for i in 0..n {
        let kind = ops[i % ops.len()];
        let node = g.add_node(Node {
            op: GraphOp::Op(kind),
            inputs: SmallVec::from_iter([InputSource::Node(prev)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: shape,
        });
        prev = node;
    }

    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(prev)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    g.add_output(out);
    g
}

fn bench_fusion_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion_pass");
    for &n in &[10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut g = build_chain_graph(n);
                    let stats = fusion::fuse(&mut g);
                    black_box(stats);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fusion_pass);
criterion_main!(benches);
