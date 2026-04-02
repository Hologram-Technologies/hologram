//! Benchmarks for the cascade state machine and certificate store.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_cascade::{run_cascade, run_cascade_with_graph, CertificateStore};
use hologram_core::op::{PrimOp, RingLevel};
use hologram_core::term::{HoloAddress, HoloCompileUnit, TermArena, TermKind};
use hologram_graph::{GraphBuilder, GraphOp};
use uor_foundation::enums::VerificationDomain;

fn make_unit(budget: f64) -> HoloCompileUnit {
    let mut arena = TermArena::new();
    let root = arena.alloc(TermKind::IntLit(42));
    let mut unit = HoloCompileUnit::new(
        arena,
        root,
        RingLevel::Q0,
        budget,
        &[VerificationDomain::Algebraic],
    );
    let hash = *blake3::hash(b"bench").as_bytes();
    unit.unit_address = hash;
    unit.address = HoloAddress::from_hash(hash);
    unit
}

fn bench_cascade_cold(c: &mut Criterion) {
    c.bench_function("cascade/cold", |b| {
        b.iter_batched(
            || (make_unit(1000.0), CertificateStore::new(16)),
            |(unit, mut store)| run_cascade(black_box(&unit), &mut store).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_cascade_warm(c: &mut Criterion) {
    let unit = make_unit(1000.0);
    let mut store = CertificateStore::new(16);
    run_cascade(&unit, &mut store).unwrap(); // warm the cache

    c.bench_function("cascade/warm_cache_hit", |b| {
        b.iter(|| run_cascade(black_box(&unit), &mut store).unwrap())
    });
}

fn bench_certificate_insert(c: &mut Criterion) {
    c.bench_function("cascade/cert_insert", |b| {
        b.iter_batched(
            || CertificateStore::new(1024),
            |mut store| {
                for i in 0..100u8 {
                    let mut addr = [0u8; 32];
                    addr[0] = i;
                    store.insert(hologram_cascade::Certificate {
                        unit_address: addr,
                        quantum_level: RingLevel::Q0,
                        budget_consumed: 5.0,
                        converged: true,
                    });
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_certificate_lookup(c: &mut Criterion) {
    let mut store = CertificateStore::new(1024);
    for i in 0..200u8 {
        let mut addr = [0u8; 32];
        addr[0] = i;
        store.insert(hologram_cascade::Certificate {
            unit_address: addr,
            quantum_level: RingLevel::Q0,
            budget_consumed: 5.0,
            converged: true,
        });
    }

    c.bench_function("cascade/cert_lookup", |b| {
        b.iter(|| {
            let mut addr = [0u8; 32];
            addr[0] = 42;
            store.get(black_box(&addr), RingLevel::Q0)
        })
    });
}

fn bench_cascade_with_graph(c: &mut Criterion) {
    c.bench_function("cascade/with_graph", |b| {
        b.iter_batched(
            || {
                let unit = make_unit(1000.0);
                let graph = GraphBuilder::new()
                    .node(GraphOp::Input)
                    .node_with_inputs(GraphOp::Prim(PrimOp::Neg), &[0])
                    .node_with_inputs(GraphOp::Output, &[1])
                    .output("result", 2)
                    .input("x")
                    .build();
                let store = CertificateStore::new(16);
                (unit, graph, store)
            },
            |(unit, graph, mut store)| {
                run_cascade_with_graph(black_box(&unit), graph, &mut store).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_cascade_cold,
    bench_cascade_warm,
    bench_certificate_insert,
    bench_certificate_lookup,
    bench_cascade_with_graph,
);
criterion_main!(benches);
