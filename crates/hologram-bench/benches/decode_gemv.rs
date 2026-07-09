//! Decode-shaped int8 GEMV benchmarks (m = 1, k/n of deployed 0.5B–7B
//! projections) — the upstream regression mirror of hologram-ai's
//! bandwidth-ratio witness.
//!
//! Throughput is reported as **bytes of int8 weight streamed per second**
//! (`Throughput::Bytes(k·n)`), so the ratio against calibrated stream
//! bandwidth is directly visible. Numbers here (native, and the wasmtime lane
//! via `hologram-backend/examples/wasm_matmul_timing.rs`) are iteration
//! signals only: acceptance for the browser kernel is witnessed downstream by
//! hologram-ai's performance contract, which exercises the actual deployed
//! wasm SIMD128 build. These benches exist to catch shape-level regressions
//! before they reach it.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hologram_backend::cpu::simd::{
    matmul_i4_pc_omajor, matmul_i8_pc_omajor, matmul_i8_per_channel,
};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind, QuantAttrs,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I8: u8 = 2;

/// (k, n) of representative single-token decode projections, sampled across
/// model scales so the k-inner, n-inner, and cache-exceeding regimes are all
/// covered (0.5B-class: 896×896 / 896×4864 / 4864×896; 1.5B-class:
/// 1536×8960; 7B-class: 3584×18944). Samples only — the kernel and the
/// compile-time fusion are shape-generic; nothing in the runtime is fitted
/// to these dimensions.
const SHAPES: &[(usize, usize)] = &[
    (896, 896),
    (896, 4864),
    (4864, 896),
    (1536, 8960),
    (3584, 18944),
];

fn bench_kernels(c: &mut Criterion) {
    let mut g = c.benchmark_group("decode_gemv");
    // Weight matrices up to 68 MB / iter: keep sampling small so the large
    // shapes stay in seconds, not minutes.
    g.sample_size(10);
    for &(k, n) in SHAPES {
        let a: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.037).collect();
        let bq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
        let scales: Vec<f32> = (0..n).map(|j| 0.01 + (j as f32) * 1e-5).collect();
        let mut out = vec![0f32; n];
        g.throughput(Throughput::Bytes((k * n) as u64));
        // The decode kernel this work order targets: output-major weight,
        // per-token W8A8, exact integer accumulation.
        g.bench_function(format!("i8_omajor_w8a8_1x{k}x{n}"), |b| {
            b.iter(|| {
                matmul_i8_pc_omajor(black_box(&a), black_box(&bq), &scales, &mut out, 1, k, n);
                black_box(&out);
            })
        });
        // LUT tier: packed i4, HALF the streamed bytes (throughput axis is
        // over the actual k·n/2 bytes — compare step TIME against the i8
        // line for the tier's win, which materializes when bandwidth-bound).
        let bq4: Vec<u8> = (0..k * n / 2).map(|i| (i % 251) as u8).collect();
        g.throughput(Throughput::Bytes((k * n / 2) as u64));
        g.bench_function(format!("i4_omajor_w8a8_1x{k}x{n}"), |b| {
            b.iter(|| {
                matmul_i4_pc_omajor(black_box(&a), black_box(&bq4), &scales, &mut out, 1, k, n);
                black_box(&out);
            })
        });
        g.throughput(Throughput::Bytes((k * n) as u64));
        // The prior fused path ([k,n] stride-n walk, W8A32 float
        // accumulation) at the same shapes — the ratio between the two is
        // the layout + integer-accumulation win.
        g.bench_function(format!("i8_kn_w8a32_1x{k}x{n}"), |b| {
            b.iter(|| {
                matmul_i8_per_channel(black_box(&a), black_box(&bq), &scales, &mut out, 1, k, n);
                black_box(&out);
            })
        });
    }
    g.finish();
}

/// Full-pipeline decode step over the compile-time-fused omajor W8A8 call:
/// compile → load once, then per-step `execute` with a **novel input each
/// iteration** (as decode has), so the content-addressed memo cannot elide
/// the kernel — this measures dispatch + binding + kernel, the per-op
/// overhead surface of the seq-1 residual.
fn bench_session(c: &mut Criterion) {
    let (m, k, n) = (1usize, 896usize, 4864usize);
    let wq: Vec<i8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as i8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.005 + (j as f32) * 1e-6).collect();

    let mut graph = Graph::new();
    let sa = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let sw = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let so = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));
    let sv = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let wc = graph.constants_mut().insert(ConstantEntry {
        bytes: wq.iter().map(|&x| x as u8).collect(),
        dtype: DTypeId(DTYPE_I8),
        shape: sw,
    });
    let sc = graph.constants_mut().insert(ConstantEntry {
        bytes: scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: sv,
    });
    let zc = graph.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4],
        dtype: DTypeId(DTYPE_I8),
        shape: sv,
    });
    let ai = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    graph.add_input(ai);
    let dq = graph.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Constant(wc),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sw,
    });
    graph.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0,
            zero_point: 0,
            axis: 1,
            ..Default::default()
        },
    );
    let mm = graph.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let mut session = InferenceSession::load(&compiled.archive, backend).unwrap();
    let mut a_bytes: Vec<u8> = (0..k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.037)
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut g = c.benchmark_group("decode_gemv");
    g.sample_size(10);
    g.throughput(Throughput::Bytes((k * n) as u64));
    g.bench_function(format!("session_step_novel_1x{k}x{n}_i8"), |b| {
        let mut step = 0u32;
        b.iter(|| {
            step = step.wrapping_add(1);
            a_bytes[..4].copy_from_slice(&(step as f32 * 1e-3).to_le_bytes());
            let outputs = session
                .execute(black_box(&[InputBuffer { bytes: &a_bytes }]))
                .unwrap();
            black_box(outputs);
        })
    });
    g.finish();
}

criterion_group!(benches, bench_kernels, bench_session);
criterion_main!(benches);
