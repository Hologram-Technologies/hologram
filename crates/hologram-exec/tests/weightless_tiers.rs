//! Every quantization tier, through the binding a real consumer ships.
//!
//! `weightless_omajor.rs` witnesses the **i8** tier end-to-end via a weightless
//! κ constant (`ConstantStore::insert_external`) resolved by a `WeightProvider`.
//! The other tiers — packed-i4, E8CB, u8 — had executing numeric tests too, but
//! every one bound the weight as an *inline compile-time constant* or a *graph
//! input*. Neither is a shape a weight-paging consumer ships, which is exactly
//! the trap `OUTPUT_MAJOR` fell into: the executing configuration was not the
//! shipped configuration.
//!
//! Each test here compiles weightless, loads through a provider, executes, and
//! checks the result against an **exact integer oracle** restated independently
//! of the kernels. For the tiers with no output-major GEMV, it checks the value
//! the generic dequant path must produce and pins the compiler's refusal to
//! promise a layout no kernel can honour.

use std::borrow::Cow;
use std::sync::Arc;

use hologram_archive::{WeightFingerprint, WeightProvider};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind, QuantAttrs,
};
use hologram_types::{act_quant, weight_layout};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_U8: u8 = 1;
const DTYPE_I4: u8 = 10;
const DTYPE_E8CB: u8 = 11;

/// The 16-entry grid a packed-i4 nibble indexes (mirrors the kernel's table).
const I4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Serves exactly one weight body, by its κ. The archive carries no weight bytes.
struct KappaProvider {
    kappa: WeightFingerprint,
    body: Vec<u8>,
}
impl WeightProvider for KappaProvider {
    fn size(&self, fp: WeightFingerprint) -> Option<usize> {
        (fp == self.kappa).then_some(self.body.len())
    }
    fn get_range(&self, fp: WeightFingerprint, offset: usize, len: usize) -> Option<Cow<'_, [u8]>> {
        if fp != self.kappa || offset + len > self.body.len() {
            return None;
        }
        Some(Cow::Borrowed(&self.body[offset..offset + len]))
    }
}

fn scale(j: usize) -> f32 {
    0.011 + j as f32 * 0.0007
}
fn act(i: usize) -> f32 {
    ((i * 7 % 41) as f32 - 20.0) * 0.037
}

/// `Σ q·w` per output column, exactly: quantize the activation per token,
/// accumulate in i32 (no rounding in the reduction), one f32 writeback.
fn w8a8_oracle(a: &[f32], n: usize, w: impl Fn(usize, usize) -> i8) -> Vec<f32> {
    let k = a.len();
    let amax = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return vec![0.0; n];
    }
    let inv = 127.0 / amax;
    let q: Vec<i32> = a
        .iter()
        .map(|&v| {
            let t = v * inv;
            let r = if t >= 0.0 { t + 0.5 } else { t - 0.5 } as i32;
            r.clamp(-127, 127)
        })
        .collect();
    let sa = amax / 127.0;
    (0..n)
        .map(|j| {
            let acc: i32 = (0..k).map(|i| q[i] * w(i, j) as i32).sum();
            (acc as f32) * (sa * scale(j))
        })
        .collect()
}

/// Build `A[1,k] · dequant(W)` where `W` is a weightless κ constant of `dtype`,
/// bound from `body` at materialization. `extra` is an optional 4th Dequantize
/// operand (the E8CB codebook), carried as an ordinary constant — it is model
/// data, small, and lives in the archive.
#[allow(clippy::too_many_arguments)]
fn weightless_tier_graph(
    k: usize,
    n: usize,
    dtype: u8,
    body: Vec<u8>,
    extra: Option<(Vec<u8>, u8, u64)>,
    layout: u8,
    aq: u8,
    zp_value: i32,
    axis: i32,
) -> Result<(Vec<u8>, Arc<KappaProvider>), hologram_compiler::CompileError> {
    let kappa = WeightFingerprint::of(&body);
    let mut g = Graph::new();
    let a_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, k as u64));
    let w_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let v_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let o_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, n as u64));

    let a_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    g.add_input(a_in);

    let wc = g
        .constants_mut()
        .insert_external(DTypeId(dtype), w_sh, kappa.0);
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: (0..n).flat_map(|j| scale(j).to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: v_sh,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: (0..n).flat_map(|_| zp_value.to_le_bytes()).collect(),
        dtype: DTypeId(dtype),
        shape: v_sh,
    });
    let mut inputs = SmallVec::from_iter([
        InputSource::Constant(wc),
        InputSource::Constant(sc),
        InputSource::Constant(zc),
    ]);
    if let Some((bytes, edt, elen)) = extra {
        let e_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(elen));
        let ec = g.constants_mut().insert(ConstantEntry {
            bytes,
            dtype: DTypeId(edt),
            shape: e_sh,
        });
        inputs.push(InputSource::Constant(ec));
    }
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs,
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: dtype,
            scale_bits: 0,
            zero_point: 0,
            axis,
            weight_layout: layout,
            act_quant: aq,
        },
    );
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a_in), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: o_sh,
    });
    g.add_output(out);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32)?;
    Ok((compiled.archive, Arc::new(KappaProvider { kappa, body })))
}

fn run(archive: &[u8], provider: Arc<KappaProvider>, a: &[f32]) -> Vec<f32> {
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(archive, CpuBackend::new(), provider, usize::MAX).unwrap();
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = sess.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
    le_to_f32(&out[0].bytes)
}

/// **Packed i4 (W4A8), weightless.** A nibble indexes the 16-entry grid; the
/// bound body is output-major `[n, k/2]`. Must reproduce the exact integer
/// oracle bit for bit.
#[test]
fn weightless_i4_output_major_matches_the_exact_integer_oracle() {
    let (k, n) = (64usize, 8usize);
    // Weights in the i4 range.
    let w = |i: usize, j: usize| -> i8 { (((i * 5 + j * 3) % 16) as i32 - 8) as i8 };
    let nib = |v: i8| -> u8 { I4_VALUES.iter().position(|&x| x == v).unwrap() as u8 };
    // Output-major, packed: each column's k-vector is k/2 bytes, low nibble first.
    let body: Vec<u8> = (0..n)
        .flat_map(|j| (0..k / 2).map(move |b| nib(w(2 * b, j)) | (nib(w(2 * b + 1, j)) << 4)))
        .collect();

    let (archive, provider) = weightless_tier_graph(
        k,
        n,
        DTYPE_I4,
        body,
        None,
        weight_layout::OUTPUT_MAJOR,
        act_quant::W8A8_TOKEN_SYM,
        0,
        1,
    )
    .expect("a weightless i4 weight may declare OUTPUT_MAJOR");

    let a: Vec<f32> = (0..k).map(act).collect();
    let got = run(&archive, provider, &a);
    let want = w8a8_oracle(&a, n, w);
    for j in 0..n {
        assert_eq!(got[j].to_bits(), want[j].to_bits(), "i4 col {j}");
    }
}

/// **E8CB (W1A8), weightless.** One `u8` index decodes an 8-weight block through
/// the model's own codebook — a constant operand, carried in the archive because
/// it is model data, while the index stream arrives at materialization.
#[test]
fn weightless_e8cb_output_major_matches_the_exact_integer_oracle() {
    let (k, n) = (64usize, 8usize); // k = 8 whole E8 groups
    let groups = k / 8;
    // A per-(column, group) entry, which is what a learned codebook gives.
    let wv = |i: usize, j: usize| -> i8 { (((i * 31 + j * 17) % 255) as i32 - 127) as i8 };
    let mut codebook = vec![0i8; 256 * 8];
    let mut body = vec![0u8; n * groups];
    for j in 0..n {
        for gi in 0..groups {
            let idx = j * groups + gi;
            assert!(idx < 256);
            body[j * groups + gi] = idx as u8;
            for t in 0..8 {
                codebook[idx * 8 + t] = wv(gi * 8 + t, j);
            }
        }
    }
    let cb_bytes: Vec<u8> = codebook.iter().map(|&v| v as u8).collect();

    let (archive, provider) = weightless_tier_graph(
        k,
        n,
        DTYPE_E8CB,
        body,
        Some((cb_bytes, DTYPE_E8CB, 256 * 8)),
        weight_layout::OUTPUT_MAJOR,
        act_quant::W8A8_TOKEN_SYM,
        0,
        1,
    )
    .expect("a weightless e8cb weight may declare OUTPUT_MAJOR");

    let a: Vec<f32> = (0..k).map(act).collect();
    let got = run(&archive, provider, &a);
    let want = w8a8_oracle(&a, n, wv);
    for j in 0..n {
        assert_eq!(got[j].to_bits(), want[j].to_bits(), "e8cb col {j}");
    }
}

/// **u8, weightless.** The tier has no output-major GEMV
/// (`QuantTier::omajor_fusable == false`), so it takes the generic dequant path
/// and computes `Σ a·((q − zp)·s)` in f32.
///
/// Two halves. It must *execute* correctly through the paged binding — a
/// quantized GEMM on the u8 tier had no end-to-end test in **any** binding. And
/// it must not be able to *promise* a layout no kernel can honour: declaring
/// `OUTPUT_MAJOR` on it is refused at compile time, naming the tier, rather than
/// deferred to a runtime that would read `[k,n]` bytes as `[n,k]`.
#[test]
fn weightless_u8_executes_on_the_generic_path_and_cannot_declare_output_major() {
    let (k, n) = (32usize, 6usize);
    let zp = 128i32;
    let q = |i: usize, j: usize| -> u8 { ((i * 11 + j * 7) % 256) as u8 };
    // The generic path reads the graph's own `[k,n]` layout.
    let body: Vec<u8> = (0..k).flat_map(|i| (0..n).map(move |j| q(i, j))).collect();

    let (archive, provider) = weightless_tier_graph(
        k,
        n,
        DTYPE_U8,
        body,
        None,
        weight_layout::ROW_MAJOR,
        act_quant::W8A32,
        zp,
        1,
    )
    .expect("a weightless u8 weight compiles on the generic path");

    let a: Vec<f32> = (0..k).map(act).collect();
    let got = run(&archive, provider, &a);

    // W8A32 reference: dequantize, then multiply. f32, so compare with a
    // tolerance — reassociation between this loop and the blocked matmul is
    // expected and is *not* what distinguishes the paths (see
    // `only_the_integer_oracle_separates_w8a8_from_w8a32`).
    for (j, &g) in got.iter().enumerate().take(n) {
        let want: f32 = (0..k)
            .map(|i| a[i] * ((q(i, j) as i32 - zp) as f32 * scale(j)))
            .sum();
        assert!((g - want).abs() < 1e-3, "u8 col {j}: got {g} want {want}");
    }

    // And the refusal, for the RIGHT reason. Use a symmetric zero-point so the
    // zero-point predicate cannot fire first and mask the tier check — both
    // errors name OUTPUT_MAJOR, so asserting only on that would prove nothing.
    let body2: Vec<u8> = (0..k).flat_map(|i| (0..n).map(move |j| q(i, j))).collect();
    let err = weightless_tier_graph(
        k,
        n,
        DTYPE_U8,
        body2,
        None,
        weight_layout::OUTPUT_MAJOR,
        act_quant::W8A8_TOKEN_SYM,
        0, // symmetric
        1,
    )
    .err()
    .expect("u8 has no output-major kernel; the declaration must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("OUTPUT_MAJOR") && msg.contains("no output-major GEMV"),
        "the refusal must be the tier's missing kernel, not some earlier \
         predicate; got: {msg}"
    );
}
