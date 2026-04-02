//! Benchmarks for the preflight pipeline (shape validation, CS_6, CS_7).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compiler::preflight;
use hologram_core::op::{PrimOp, RingLevel};
use hologram_core::term::{HoloCompileUnit, TermArena, TermKind};
use uor_foundation::enums::VerificationDomain;

/// Build a CompileUnit with `n` nodes (chain of neg(neg(neg(...42...)))).
fn make_unit(n: usize) -> HoloCompileUnit {
    let mut arena = TermArena::new();
    let mut id = arena.alloc(TermKind::IntLit(42));
    for _ in 1..n {
        id = arena.alloc(TermKind::UnaryApp {
            op: PrimOp::Neg,
            arg: id,
        });
    }
    HoloCompileUnit::new(
        arena,
        id,
        RingLevel::Q0,
        100.0,
        &[VerificationDomain::Algebraic],
    )
}

fn bench_budget_solvency(c: &mut Criterion) {
    let unit = make_unit(1);
    c.bench_function("preflight/budget_solvency", |b| {
        b.iter(|| preflight::check_budget_solvency(black_box(&unit)))
    });
}

fn bench_shape_validation_100(c: &mut Criterion) {
    let unit = make_unit(100);
    c.bench_function("preflight/shape_validation/100", |b| {
        b.iter(|| preflight::validate_shape(black_box(&unit)).unwrap())
    });
}

fn bench_unit_address_100(c: &mut Criterion) {
    let unit = make_unit(100);
    c.bench_function("preflight/unit_address/100", |b| {
        b.iter(|| preflight::compute_unit_address(black_box(&unit.arena), unit.root_term))
    });
}

fn bench_unit_address_1000(c: &mut Criterion) {
    let unit = make_unit(1000);
    c.bench_function("preflight/unit_address/1000", |b| {
        b.iter(|| preflight::compute_unit_address(black_box(&unit.arena), unit.root_term))
    });
}

fn bench_full_preflight_100(c: &mut Criterion) {
    c.bench_function("preflight/full/100", |b| {
        b.iter_batched(
            || make_unit(100),
            |mut unit| preflight::run_preflight(&mut unit).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_budget_solvency,
    bench_shape_validation_100,
    bench_unit_address_100,
    bench_unit_address_1000,
    bench_full_preflight_100,
);
criterion_main!(benches);
