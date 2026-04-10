//! FFI overhead benchmarks — measuring call overhead through the C ABI layer.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_ffi::encoding::{hologram_encoding_embed, hologram_encoding_lift, hologram_lut_apply};
use hologram_ffi::graph::*;
use std::ffi::CString;

/// Benchmark: build a 10-node graph through FFI.
fn bench_ffi_graph_build(c: &mut Criterion) {
    c.bench_function("ffi/graph_build_10", |b| {
        b.iter(|| {
            let builder = hologram_graph_builder_new();
            let name = CString::new("x").unwrap();
            hologram_graph_builder_input(builder, name.as_ptr());
            hologram_graph_builder_node_from_input(builder, 0, 0, 0);
            for i in 0..8 {
                let inputs = [i as usize];
                hologram_graph_builder_node_with_inputs(builder, 3, i % 21, inputs.as_ptr(), 1);
            }
            let inputs = [8usize];
            hologram_graph_builder_node_with_inputs(builder, 1, 0, inputs.as_ptr(), 1);
            let name_y = CString::new("y").unwrap();
            hologram_graph_builder_output(builder, name_y.as_ptr(), 9);
            let g = hologram_graph_builder_build(builder);
            black_box(hologram_graph_node_count(g));
            hologram_graph_free(g);
        });
    });
}

/// Benchmark: LUT apply through FFI (single byte).
fn bench_ffi_lut_apply(c: &mut Criterion) {
    c.bench_function("ffi/lut_apply", |b| {
        b.iter(|| {
            black_box(hologram_lut_apply(black_box(0), black_box(128)));
        });
    });
}

/// Benchmark: encoding embed through FFI.
fn bench_ffi_encoding_embed(c: &mut Criterion) {
    c.bench_function("ffi/encoding_embed", |b| {
        b.iter(|| {
            black_box(hologram_encoding_embed(black_box(1), black_box(0.5)));
        });
    });
}

/// Benchmark: encoding lift through FFI.
fn bench_ffi_encoding_lift(c: &mut Criterion) {
    c.bench_function("ffi/encoding_lift", |b| {
        b.iter(|| {
            black_box(hologram_encoding_lift(black_box(1), black_box(128)));
        });
    });
}

/// Benchmark: full pipeline through FFI (build → compile → execute).
fn bench_ffi_full_pipeline(c: &mut Criterion) {
    c.bench_function("ffi/full_pipeline", |b| {
        b.iter(|| {
            let builder = hologram_graph_builder_new();
            let name = CString::new("x").unwrap();
            hologram_graph_builder_input(builder, name.as_ptr());
            hologram_graph_builder_node_from_input(builder, 0, 0, 0);
            let inputs = [0usize];
            hologram_graph_builder_node_with_inputs(builder, 3, 0, inputs.as_ptr(), 1);
            let inputs2 = [1usize];
            hologram_graph_builder_node_with_inputs(builder, 1, 0, inputs2.as_ptr(), 1);
            let name_y = CString::new("y").unwrap();
            hologram_graph_builder_output(builder, name_y.as_ptr(), 2);
            let g = hologram_graph_builder_build(builder);

            let out = hologram_ffi::compiler::hologram_compile(g);
            let archive_ptr = hologram_ffi::compiler::hologram_compilation_archive_ptr(out);
            let archive_len = hologram_ffi::compiler::hologram_compilation_archive_len(out);

            let inp = hologram_ffi::exec::hologram_inputs_new();
            hologram_ffi::exec::hologram_inputs_set(inp, 0, [42u8].as_ptr(), 1);

            let outputs = hologram_ffi::exec::hologram_fused_componentute_bytes(
                archive_ptr,
                archive_len,
                inp,
            );
            black_box(hologram_ffi::exec::hologram_outputs_len(outputs));
            hologram_ffi::exec::hologram_outputs_free(outputs);
            hologram_ffi::exec::hologram_inputs_free(inp);
            hologram_ffi::compiler::hologram_compilation_free(out);
        });
    });
}

criterion_group!(
    benches,
    bench_ffi_graph_build,
    bench_ffi_lut_apply,
    bench_ffi_encoding_embed,
    bench_ffi_encoding_lift,
    bench_ffi_full_pipeline,
);
criterion_main!(benches);
