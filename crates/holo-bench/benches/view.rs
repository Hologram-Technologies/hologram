//! ElementWiseView benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holo_core::view::ElementWiseView;

fn bench_apply_single(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    c.bench_function("view::apply(byte)", |b| {
        b.iter(|| view.apply(black_box(42)))
    });
}

fn bench_compose(c: &mut Criterion) {
    let a = ElementWiseView::new(|x| x.wrapping_add(1));
    let b = ElementWiseView::new(|x| x.wrapping_mul(3));
    c.bench_function("view::then(compose)", |b_iter| {
        b_iter.iter(|| black_box(&a).then(black_box(&b)))
    });
}

fn bench_apply_slice_small(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let mut data = [0u8; 64];
    c.bench_function("view::apply_slice(64B)", |b| {
        b.iter(|| {
            data = black_box(data);
            view.apply_slice(&mut data);
        })
    });
}

fn bench_apply_slice_1k(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let mut data = vec![0u8; 1024];
    c.bench_function("view::apply_slice(1KB)", |b| {
        b.iter(|| {
            view.apply_slice(black_box(&mut data));
        })
    });
}

fn bench_apply_slice_64k(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let mut data = vec![0u8; 65536];
    c.bench_function("view::apply_slice(64KB)", |b| {
        b.iter(|| {
            view.apply_slice(black_box(&mut data));
        })
    });
}

fn bench_apply_to(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let input = vec![0u8; 1024];
    let mut output = vec![0u8; 1024];
    c.bench_function("view::apply_to(1KB)", |b| {
        b.iter(|| {
            view.apply_to(black_box(&input), &mut output);
        })
    });
}

fn bench_is_bijective(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    c.bench_function("view::is_bijective()", |b| {
        b.iter(|| black_box(&view).is_bijective())
    });
}

fn bench_inverse(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    c.bench_function("view::inverse()", |b| b.iter(|| black_box(&view).inverse()));
}

fn bench_composition_chain(c: &mut Criterion) {
    let inc = ElementWiseView::new(|x| x.wrapping_add(1));
    c.bench_function("view::chain_10_compositions", |b| {
        b.iter(|| {
            let mut v = ElementWiseView::identity();
            for _ in 0..10 {
                v = v.then(black_box(&inc));
            }
            v
        })
    });
}

fn bench_rkyv_serialize(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    c.bench_function("view::rkyv_serialize", |b| {
        b.iter(|| rkyv::to_bytes::<_, 512>(black_box(&view)).unwrap())
    });
}

fn bench_rkyv_deserialize(c: &mut Criterion) {
    let view = ElementWiseView::new(|x| x.wrapping_add(1));
    let bytes = rkyv::to_bytes::<_, 512>(&view).unwrap();
    c.bench_function("view::rkyv_check_archived", |b| {
        b.iter(|| rkyv::check_archived_root::<ElementWiseView>(black_box(&bytes)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_apply_single,
    bench_compose,
    bench_apply_slice_small,
    bench_apply_slice_1k,
    bench_apply_slice_64k,
    bench_apply_to,
    bench_is_bijective,
    bench_inverse,
    bench_composition_chain,
    bench_rkyv_serialize,
    bench_rkyv_deserialize,
);
criterion_main!(benches);
