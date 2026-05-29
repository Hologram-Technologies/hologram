//! SP class criterion floors for the in-memory reference backend — architecture §4.
//! Runs from `just perf`. These are the *reference* floors; backend-specific stores (native,
//! bare-metal, opfs) inherit the SP discipline and add their own floors on top.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::KappaStore;

fn sp_mem_idempotent_put_no_rewrite(c: &mut Criterion) {
    let store = MemKappaStore::new();
    let payload: Vec<u8> = (0..16 * 1024).map(|i| i as u8).collect();
    let _ = store.put("blake3", &payload).unwrap();

    let mut g = c.benchmark_group("sp_mem");
    g.measurement_time(Duration::from_secs(2));
    g.bench_function("idempotent_put_hit_16kb", |b| {
        b.iter(|| {
            let k = store.put("blake3", black_box(&payload)).unwrap();
            black_box(k);
        })
    });
    g.finish();
}

fn sp_mem_get_is_zero_copy_arc_clone(c: &mut Criterion) {
    let store = MemKappaStore::new();
    let payload: Vec<u8> = (0..16 * 1024).map(|i| (i * 7) as u8).collect();
    let k = store.put("blake3", &payload).unwrap();

    let mut g = c.benchmark_group("sp_mem");
    g.measurement_time(Duration::from_secs(2));
    g.bench_function("get_arc_clone_16kb", |b| {
        b.iter(|| {
            let v = store.get(black_box(&k)).unwrap().unwrap();
            black_box(v);
        })
    });
    g.finish();
}

criterion_group!(
    sp_floors_mem,
    sp_mem_idempotent_put_no_rewrite,
    sp_mem_get_is_zero_copy_arc_clone,
);
criterion_main!(sp_floors_mem);
