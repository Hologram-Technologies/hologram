//! Realistic multi-layer int8 decode (M=1) profiling harness.
//!
//! Builds an L-layer LLaMA-style decode graph (RmsNorm + q/k/v/o projections +
//! RmsNorm + gate/up/silu/mul/down MLP + residuals), all weights int8-quantized
//! constants, and runs `session.execute` with a **novel input each step** (so
//! the graph memo never elides the walk — the real per-token cost). Reports the
//! per-token latency, the effective int8-weight bandwidth, and the dispatch/
//! reuse counts so we can see whether the fast fused decode path is engaged.
//!
//! Tune via env: DECODE_D, DECODE_HIDDEN, DECODE_LAYERS, DECODE_STEPS.
//! Profile:  cargo build --release --example decode_profile -p hologram-bench
//!   then    perf record -g target/release/examples/decode_profile ; perf report

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::{ConstantId, Node, NodeId},
    registry::{DTypeId, ShapeDescriptor, ShapeId},
    Graph, GraphOp, InputSource, NormAttrs, OpKind, QuantAttrs,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;
use std::time::Instant;

const F32: u8 = 8;
const I8: u8 = 2;
const BF16: u8 = 7;
const E8CB: u8 = 11;
fn f32t() -> DTypeId {
    DTypeId(F32)
}
fn bf16_bytes(vals: impl Iterator<Item = f32>) -> Vec<u8> {
    vals.flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
/// Whether to build a plain bf16 model (DECODE_DTYPE=bf16) vs int8-quantized.
fn is_bf16() -> bool {
    std::env::var("DECODE_DTYPE").ok().as_deref() == Some("bf16")
}
/// Whether to build an E8-codebook (1 bit/weight) model (DECODE_DTYPE=e8cb).
fn is_e8cb() -> bool {
    std::env::var("DECODE_DTYPE").ok().as_deref() == Some("e8cb")
}
/// The activation/compute dtype of the model.
fn act_dtype() -> DTypeId {
    if is_bf16() {
        DTypeId(BF16)
    } else {
        DTypeId(F32)
    }
}

fn env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn shape2(g: &mut Graph, a: u64, b: u64) -> ShapeId {
    g.shape_registry_mut().intern(ShapeDescriptor::rank2(a, b))
}
fn shape1(g: &mut Graph, n: u64) -> ShapeId {
    g.shape_registry_mut().intern(ShapeDescriptor::rank1(n))
}

/// (weight, scale, zero-point, weight-shape) int8 constants.
fn int8_weight(g: &mut Graph, k: usize, n: usize) -> (ConstantId, ConstantId, ConstantId, ShapeId) {
    let sw = shape2(g, k as u64, n as u64);
    let sv = shape1(g, n as u64);
    let wq: Vec<u8> = (0..k * n)
        .map(|i| (((i as i64 % 255) - 127) as i8) as u8)
        .collect();
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: wq,
        dtype: DTypeId(I8),
        shape: sw,
    });
    let scales: Vec<u8> = (0..n)
        .flat_map(|j| (0.01f32 + j as f32 * 1e-6).to_le_bytes())
        .collect();
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: scales,
        dtype: f32t(),
        shape: sv,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4],
        dtype: DTypeId(I8),
        shape: sv,
    });
    (wc, sc, zc, sw)
}

/// (index, scale, zero-point, weight-shape) E8-codebook constants. The weight
/// is a `[k/8, n]` grid of `u8` codebook indices (one per 8-D group), declared
/// with logical shape `[k,n]` + dtype E8CB so storage is `k*n/8` bytes.
fn e8cb_weight(g: &mut Graph, k: usize, n: usize) -> (ConstantId, ConstantId, ConstantId, ShapeId) {
    let sw = shape2(g, k as u64, n as u64);
    let sv = shape1(g, n as u64);
    // [k/8, n] row-major indices (element gk*n + j), matching the compiler's
    // input-major → omajor transpose.
    let idx: Vec<u8> = (0..(k / 8) * n)
        .map(|i| ((i * 53 + 7) % 256) as u8)
        .collect();
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: idx,
        dtype: DTypeId(E8CB),
        shape: sw,
    });
    let scales: Vec<u8> = (0..n)
        .flat_map(|j| (0.01f32 + j as f32 * 1e-6).to_le_bytes())
        .collect();
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: scales,
        dtype: f32t(),
        shape: sv,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4],
        dtype: DTypeId(I8),
        shape: sv,
    });
    (wc, sc, zc, sw)
}

/// A projection `x[1,k] · W[k,n] → [1,n]`. int8/e8cb: dequant(weight)→matmul;
/// bf16: a plain bf16 weight constant fed straight to matmul.
fn proj(g: &mut Graph, x: NodeId, k: usize, n: usize) -> NodeId {
    let so = shape2(g, 1, n as u64);
    if is_bf16() {
        let sw = shape2(g, k as u64, n as u64);
        let wbytes = bf16_bytes((0..k * n).map(|i| ((i as i64 % 255) - 127) as f32 * 0.01));
        let wc = g.constants_mut().insert(ConstantEntry {
            bytes: wbytes,
            dtype: DTypeId(BF16),
            shape: sw,
        });
        return g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(wc)]),
            output_dtype: act_dtype(),
            output_shape: so,
        });
    }
    let (quant_dtype, (wc, sc, zc, sw)) = if is_e8cb() {
        (E8CB, e8cb_weight(g, k, n))
    } else {
        (I8, int8_weight(g, k, n))
    };
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Constant(wc),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: f32t(),
        output_shape: sw,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype,
            scale_bits: 0,
            zero_point: 0,
            axis: 1,
        },
    );
    g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Node(dq)]),
        output_dtype: f32t(),
        output_shape: so,
    })
}

fn gamma(g: &mut Graph, n: usize) -> ConstantId {
    let sv = shape1(g, n as u64);
    let bytes = if is_bf16() {
        bf16_bytes((0..n).map(|_| 1.0f32))
    } else {
        (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect()
    };
    g.constants_mut().insert(ConstantEntry {
        bytes,
        dtype: act_dtype(),
        shape: sv,
    })
}

fn rmsnorm(g: &mut Graph, x: NodeId, n: usize) -> NodeId {
    let gc = gamma(g, n);
    let sh = shape2(g, 1, n as u64);
    let nn = g.add_node(Node {
        op: GraphOp::Op(OpKind::RmsNorm),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(gc)]),
        output_dtype: act_dtype(),
        output_shape: sh,
    });
    g.set_norm_attrs(nn, NormAttrs { num_groups: 0 });
    nn
}

fn elem(g: &mut Graph, op: OpKind, a: NodeId, b: NodeId, n: usize) -> NodeId {
    let sh = shape2(g, 1, n as u64);
    g.add_node(Node {
        op: GraphOp::Op(op),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Node(b)]),
        output_shape: sh,
        output_dtype: act_dtype(),
    })
}
fn unary(g: &mut Graph, op: OpKind, a: NodeId, n: usize) -> NodeId {
    let sh = shape2(g, 1, n as u64);
    g.add_node(Node {
        op: GraphOp::Op(op),
        inputs: SmallVec::from_iter([InputSource::Node(a)]),
        output_shape: sh,
        output_dtype: act_dtype(),
    })
}

fn main() {
    let d = env("DECODE_D", 2048);
    let hidden = env("DECODE_HIDDEN", 5632);
    let layers = env("DECODE_LAYERS", 24);
    let steps = env("DECODE_STEPS", 20);

    let mut g = Graph::new();
    let s_x = shape2(&mut g, 1, d as u64);
    let inp = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: act_dtype(),
        output_shape: s_x,
    });
    g.add_input(inp);
    let mut x = inp;

    let mut int8_bytes: usize = 0;
    for _ in 0..layers {
        // Attention block (projection weight volume; simplified residual fold).
        let a_in = rmsnorm(&mut g, x, d);
        let _q = proj(&mut g, a_in, d, d);
        let _k = proj(&mut g, a_in, d, d);
        let _v = proj(&mut g, a_in, d, d);
        let o = proj(&mut g, a_in, d, d);
        int8_bytes += 4 * d * d;
        x = elem(&mut g, OpKind::Add, x, o, d);
        // MLP block (SwiGLU).
        let m_in = rmsnorm(&mut g, x, d);
        let gate = proj(&mut g, m_in, d, hidden);
        let up = proj(&mut g, m_in, d, hidden);
        int8_bytes += 2 * d * hidden;
        let act = unary(&mut g, OpKind::Silu, gate, hidden);
        let mul = elem(&mut g, OpKind::Mul, act, up, hidden);
        let down = proj(&mut g, mul, hidden, d);
        int8_bytes += hidden * d;
        x = elem(&mut g, OpKind::Add, x, down, d);
    }
    let gc = gamma(&mut g, d);
    let out = g.add_node(Node {
        op: GraphOp::Op(OpKind::RmsNorm),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(gc)]),
        output_dtype: act_dtype(),
        output_shape: s_x,
    });
    g.set_norm_attrs(out, NormAttrs { num_groups: 0 });
    let outn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Reshape),
        inputs: SmallVec::from_iter([InputSource::Node(out)]),
        output_dtype: act_dtype(),
        output_shape: s_x,
    });
    // Mark the final norm as the output.
    let out_node = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(outn)]),
        output_dtype: act_dtype(),
        output_shape: s_x,
    });
    g.add_output(out_node);

    // Streamed weight bytes per logical weight: bf16 = 2, i8 = 1, e8cb = 1/8.
    let bytes_per_weight = if is_bf16() {
        2.0
    } else if is_e8cb() {
        0.125
    } else {
        1.0
    };
    let dtype_label = if is_bf16() {
        "bf16"
    } else if is_e8cb() {
        "e8cb"
    } else {
        "i8"
    };
    let weight_bytes = int8_bytes as f64 * bytes_per_weight;

    let nodes = g.node_count();
    eprintln!(
        "building: d={d} hidden={hidden} layers={layers} nodes={nodes} weight={:.1} MB dtype={dtype_label}",
        weight_bytes / 1e6,
    );

    let t0 = Instant::now();
    let witt = if is_bf16() {
        WittLevel::W16
    } else {
        WittLevel::W32
    };
    let compiled = compile(g, BackendKind::Cpu, witt).unwrap();
    eprintln!("compile: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    let t1 = Instant::now();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    eprintln!(
        "load: {:.1} ms  kernels={}",
        t1.elapsed().as_secs_f64() * 1e3,
        sess.kernel_count()
    );

    let mut xb: Vec<u8> = if is_bf16() {
        bf16_bytes((0..d).map(|i| i as f32 * 1e-3))
    } else {
        (0..d)
            .flat_map(|i| (i as f32 * 1e-3).to_le_bytes())
            .collect()
    };
    let _ = sess.execute(&[InputBuffer { bytes: &xb }]).unwrap();

    let mut best = f64::INFINITY;
    let mut total = 0.0;
    for s in 0..steps {
        xb[..4].copy_from_slice(&((s as f32 + 1.0) * 1e-3).to_le_bytes());
        let t = Instant::now();
        let out = sess.execute(&[InputBuffer { bytes: &xb }]).unwrap();
        std::hint::black_box(&out);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total += ms;
        best = best.min(ms);
    }
    let avg = total / steps as f64;
    let gbps = weight_bytes / (best * 1e-3) / 1e9;
    let gmacs = int8_bytes as f64 / (best * 1e-3) / 1e9;
    eprintln!("per-token: avg={avg:.2} ms  best={best:.2} ms  |  weight BW at best = {gbps:.1} GB/s  |  {gmacs:.1} GMAC/s");
    eprintln!(
        "dispatched={} skipped(reuse)={}",
        sess.last_dispatched(),
        sess.last_skipped()
    );
}
