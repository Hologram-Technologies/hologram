//! Compiler pipeline benchmark.

use criterion::{criterion_group, criterion_main, Criterion};
use hologram_compiler::{Compiler, BackendKind};
use hologram_graph::Graph;
use uor_foundation::WittLevel;

fn bench_compile(c: &mut Criterion) {
    c.bench_function("compile_empty_graph", |b| {
        b.iter(|| {
            let g = Graph::new();
            let _ = Compiler::new(g, BackendKind::Cpu, WittLevel::W16).compile().unwrap();
        });
    });
}

criterion_group!(benches, bench_compile);
criterion_main!(benches);
