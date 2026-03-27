//! LUT table lookup benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_core::lut::activation;
use hologram_core::lut::arith;
use hologram_core::lut::q0;
use hologram_core::op::LutOp;
use hologram_core::op::PrimOp;
use hologram_core::q2::arith as arith_q2;
use hologram_core::q2::ring::TripleRing;
use hologram_core::ring::ByteRing;
use hologram_core::view::ElementWiseView;

fn bench_q0_stratum(c: &mut Criterion) {
    c.bench_function("q0::stratum_q0(127)", |b| {
        b.iter(|| q0::stratum_q0(black_box(127)))
    });
}

fn bench_arith_add(c: &mut Criterion) {
    c.bench_function("arith::add_q0(100, 200)", |b| {
        b.iter(|| arith::add_q0(black_box(100), black_box(200)))
    });
}

fn bench_arith_mul(c: &mut Criterion) {
    c.bench_function("arith::mul_q0(13, 17)", |b| {
        b.iter(|| arith::mul_q0(black_box(13), black_box(17)))
    });
}

fn bench_sigmoid_lut(c: &mut Criterion) {
    c.bench_function("activation::sigmoid_lut(128)", |b| {
        b.iter(|| activation::sigmoid_lut(black_box(128)))
    });
}

fn bench_sigmoid_vs_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid_comparison");
    group.bench_function("lut_lookup", |b| {
        b.iter(|| activation::sigmoid_lut(black_box(128)))
    });
    group.bench_function("f64_compute", |b| {
        b.iter(|| {
            let x: f64 = black_box(0.0);
            1.0 / (1.0 + (-x).exp())
        })
    });
    group.finish();
}

fn bench_all_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_lookup");
    for op in &LutOp::ALL {
        group.bench_function(op.name(), |b| b.iter(|| op.apply(black_box(128))));
    }
    group.finish();
}

fn bench_activation_batch(c: &mut Criterion) {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    c.bench_function("sigmoid_1024_scalars", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for &byte in black_box(&data) {
                sum += activation::sigmoid_lut(byte) as u64;
            }
            sum
        })
    });
}

fn bench_group_order_byte(c: &mut Criterion) {
    use uor_foundation::kernel::op::Group;
    let ring = ByteRing;
    c.bench_function("group::order(ByteRing)", |b| {
        b.iter(|| black_box(&ring).order())
    });
}

fn bench_group_generators_byte(c: &mut Criterion) {
    use uor_foundation::kernel::op::Group;
    let ring = ByteRing;
    c.bench_function("group::generated_by(ByteRing)", |b| {
        b.iter(|| black_box(&ring).generated_by().len())
    });
}

fn bench_binary_op_commutative(c: &mut Criterion) {
    use uor_foundation::kernel::op::BinaryOp;
    c.bench_function("prim::commutative(Add)", |b| {
        b.iter(|| black_box(PrimOp::Add).commutative())
    });
}

fn bench_normed_division_dimension(c: &mut Criterion) {
    use uor_foundation::kernel::division::NormedDivisionAlgebra;
    let ring = ByteRing;
    c.bench_function("nda::algebra_dimension(ByteRing)", |b| {
        b.iter(|| black_box(&ring).algebra_dimension())
    });
}

fn bench_view_identity_check(c: &mut Criterion) {
    let identity = ElementWiseView::identity();
    c.bench_function("view::identity_eq_check", |b| {
        b.iter(|| black_box(&identity) == &ElementWiseView::identity())
    });
}

fn bench_triple_ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("triple_ring");
    group.bench_function("add_q2", |b| {
        b.iter(|| arith_q2::add_q2(black_box(0x00AB_CDEF), black_box(0x00123456)))
    });
    group.bench_function("mul_q2", |b| {
        b.iter(|| arith_q2::mul_q2(black_box(0x00AB_CDEF), black_box(0x00123456)))
    });
    group.bench_function("ring_quantum", |b| {
        use uor_foundation::kernel::schema::Ring;
        b.iter(|| black_box(TripleRing).ring_quantum())
    });
    group.bench_function("neg_q2", |b| {
        b.iter(|| arith_q2::neg_q2(black_box(0x00AB_CDEF)))
    });
    group.finish();
}

fn bench_precision_pass(c: &mut Criterion) {
    use hologram_compiler::precision::promote_prim_ring_levels;
    use hologram_graph::builder::GraphBuilder;
    use hologram_graph::graph::GraphOp;

    // Build a 101-node graph: Input + 50 × (Sigmoid → Prim(Neg)).
    let mut g = GraphBuilder::new().node(GraphOp::Input);
    for i in 0..50 {
        g = g.node_with_inputs(GraphOp::Lut(LutOp::Sigmoid), &[i]);
        g = g.node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[i * 2 + 1]);
    }
    let graph_template = g.build();

    c.bench_function("precision_pass_101_nodes", |b| {
        b.iter(|| {
            let mut g = graph_template.clone();
            promote_prim_ring_levels(black_box(&mut g))
        })
    });
}

criterion_group!(
    benches,
    bench_q0_stratum,
    bench_arith_add,
    bench_arith_mul,
    bench_sigmoid_lut,
    bench_sigmoid_vs_f64,
    bench_all_activations,
    bench_activation_batch,
    bench_group_order_byte,
    bench_group_generators_byte,
    bench_binary_op_commutative,
    bench_normed_division_dimension,
    bench_view_identity_check,
    bench_triple_ring,
    bench_precision_pass,
);
criterion_main!(benches);
