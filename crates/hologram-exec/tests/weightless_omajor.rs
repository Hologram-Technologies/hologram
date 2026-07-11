//! The **positive** end-to-end witness for `weight_layout = OUTPUT_MAJOR`.
//!
//! Everything else in the substrate that declares `OUTPUT_MAJOR` is a fail-loud
//! rejection test. `docs/numerics/w8a8.md` asserts in prose that the fused
//! output-major kernel "runs at every `m`, decode and prefill alike", and that
//! sentence is load-bearing for a consumer doing chunked prefill — but until
//! this file, nothing executed it.
//!
//! The weight is bound the way a real consumer binds it: a **weightless
//! constant** (`ConstantStore::insert_external`) naming the κ of a body that
//! arrives at materialization through a `WeightProvider`. Not a graph input,
//! which is a shape nobody ships.
//!
//! ## Why the oracle is exact, and why that matters
//!
//! "W8A8 differs from an f32 reference" is a **false test**: it passes under
//! W8A32 too, because a naive `Σ a·(w·s)` reference loop already disagrees with
//! `matmul_f32_blocked` by reassociation noise. Nothing about W8A8 is being
//! measured.
//!
//! What separates the two is that W8A8's accumulation is an *exact* i32 sum, so
//! it has a closed-form integer oracle that it reproduces **bit for bit** — and
//! W8A32, computing a genuinely different function in f32, cannot. Exactness is
//! what makes the invariance checkable, not merely true.

use std::borrow::Cow;
use std::sync::Arc;

use hologram_archive::{
    decoder, format::SectionKind, HoloLoader, WeightFingerprint, WeightProvider,
};
use hologram_backend::{CpuBackend, KernelCall};
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
const DTYPE_I8: u8 = 2;

fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Serves exactly one weight body, by its κ. The weightless archive carries no
/// weight bytes at all — this is the entire byte source.
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

/// `w[j][i]` — the logical weight, indexed output-column-major (the layout the
/// binder materializes).
fn w(i: usize, j: usize) -> i8 {
    (((i * 31 + j * 17) % 255) as i32 - 127) as i8
}
fn scale(j: usize) -> f32 {
    0.011 + j as f32 * 0.0007
}
fn act(r: usize, i: usize) -> f32 {
    (((r * 13 + i * 7) % 41) as f32 - 20.0) * 0.037
}

/// The exact integer oracle for W8A8, restated independently of the kernels:
/// quantize the activation row per token, accumulate `Σ q·w` in `i32` (exact —
/// no rounding anywhere in the reduction), then one f32 writeback.
fn w8a8_oracle(a: &[f32], k: usize, n: usize) -> Vec<f32> {
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

/// `[m, k] · dequant(W)` where `W` is a **weightless** i8 constant: the graph
/// carries its κ, the bytes arrive from the provider. `declare` opts the weight
/// slot into the output-major W8A8 decode form.
///
/// Returns `(archive, provider)`. The provider holds the weight as `[n, k]` —
/// the layout `OUTPUT_MAJOR` promises the binder will materialize.
fn weightless_graph(m: usize, k: usize, n: usize, declare: bool) -> (Vec<u8>, Arc<KappaProvider>) {
    // The bound body. `OUTPUT_MAJOR` promises `[n, k]` — each output column's
    // k-vector contiguous; `ROW_MAJOR` binds the graph's own `[k, n]`. The κ the
    // graph names is the κ of whichever body the binder will materialize, which
    // is the whole content of the declaration.
    let body: Vec<u8> = if declare {
        (0..n)
            .flat_map(|j| (0..k).map(move |i| w(i, j) as u8))
            .collect()
    } else {
        (0..k)
            .flat_map(|i| (0..n).map(move |j| w(i, j) as u8))
            .collect()
    };
    let kappa = WeightFingerprint::of(&body);

    let mut g = Graph::new();
    let a_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, k as u64));
    let w_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let v_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));
    let o_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m as u64, n as u64));

    let a_in = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: a_sh,
    });
    g.add_input(a_in);

    // The weightless weight: κ now, bytes at materialization.
    let wc = g
        .constants_mut()
        .insert_external(DTypeId(DTYPE_I8), w_sh, kappa.0);
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: (0..n).flat_map(|j| scale(j).to_le_bytes()).collect(),
        dtype: DTypeId(DTYPE_F32),
        shape: v_sh,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: vec![0u8; n * 4], // symmetric
        dtype: DTypeId(DTYPE_I8),
        shape: v_sh,
    });
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Constant(wc),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: w_sh,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0,
            zero_point: 0,
            axis: 1, // per output column
            weight_layout: if declare {
                weight_layout::OUTPUT_MAJOR
            } else {
                weight_layout::ROW_MAJOR
            },
            act_quant: if declare {
                act_quant::W8A8_TOKEN_SYM
            } else {
                act_quant::W8A32
            },
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

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    (compiled.archive, Arc::new(KappaProvider { kappa, body }))
}

fn run(archive: &[u8], provider: Arc<KappaProvider>, a: &[f32]) -> Vec<f32> {
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(archive, CpuBackend::new(), provider, usize::MAX).unwrap();
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = sess.execute(&[InputBuffer { bytes: &bytes }]).unwrap();
    le_to_f32(&out[0].bytes)
}

/// A weightless κ-bound weight declaring `OUTPUT_MAJOR` compiles, loads through
/// a provider, reaches the fused output-major integer GEMV, and reproduces the
/// exact integer oracle **bit for bit**.
///
/// This is the configuration a paging consumer actually ships, and it was
/// previously rejected at compile time by a predicate that asked "is this a
/// constant?" instead of "does this constant have bytes?".
#[test]
fn weightless_output_major_w8a8_executes_and_matches_the_exact_integer_oracle() {
    let (k, n) = (64usize, 8usize);
    let (archive, provider) = weightless_graph(1, k, n, true);

    // The archive carries the fused output-major W8A8 call, and no weight bytes.
    let plan = HoloLoader::from_bytes(&archive)
        .unwrap()
        .into_plan()
        .unwrap();
    let calls = decoder::decode_calls(plan.section(SectionKind::KernelCalls).unwrap()).unwrap();
    // Compile-time fusion cannot fire: there are no bytes to transpose. The
    // archive carries the unfused pair, and no weight body.
    assert!(
        calls.iter().any(|c| matches!(c, KernelCall::Dequantize(_)))
            && calls.iter().any(|c| matches!(c, KernelCall::MatMul(_))),
        "a weightless weight has no bytes to transpose at compile time"
    );

    // The **load-time** fusion is what must honour the declaration.
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider, usize::MAX).unwrap();
    assert_eq!(
        sess.dequant_fused_count(),
        1,
        "the load-time fusion must absorb the Dequantize into a fused MatMulDequant"
    );

    let a: Vec<f32> = (0..k).map(|i| act(0, i)).collect();
    let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = le_to_f32(&sess.execute(&[InputBuffer { bytes: &bytes }]).unwrap()[0].bytes);
    let want = w8a8_oracle(&a, k, n);

    for j in 0..n {
        assert_eq!(
            got[j].to_bits(),
            want[j].to_bits(),
            "col {j}: session {} vs exact integer oracle {}",
            got[j],
            want[j]
        );
    }
}

/// The doc says the output-major kernel "runs at every `m`, decode and prefill
/// alike". Execute that: a prefill batch of `m` rows must equal `m` independent
/// decode steps, **bit for bit**, through the session — not just through the
/// kernel.
///
/// The activation is quantized per token, so each row carries its own scale;
/// batching must not let one row's `amax` leak into another's.
#[test]
fn prefill_batch_equals_decode_steps_bit_for_bit() {
    let (k, n, m) = (64usize, 8usize, 5usize); // m straddles the MR = 4 tile

    let a_batch: Vec<f32> = (0..m)
        .flat_map(|r| (0..k).map(move |i| act(r, i)))
        .collect();
    let (arch_b, prov_b) = weightless_graph(m, k, n, true);
    let batched = run(&arch_b, prov_b, &a_batch);

    for r in 0..m {
        let a_row: Vec<f32> = (0..k).map(|i| act(r, i)).collect();
        let (arch_1, prov_1) = weightless_graph(1, k, n, true);
        let single = run(&arch_1, prov_1, &a_row);
        for j in 0..n {
            assert_eq!(
                batched[r * n + j].to_bits(),
                single[j].to_bits(),
                "row {r} col {j}: prefill {} vs decode step {}",
                batched[r * n + j],
                single[j]
            );
        }
        // ...and each row is itself the exact oracle.
        let want = w8a8_oracle(&a_row, k, n);
        for j in 0..n {
            assert_eq!(single[j].to_bits(), want[j].to_bits(), "row {r} col {j}");
        }
    }
}

/// The discriminator, stated as a test rather than as prose.
///
/// A naive `Σ a·(w·s)` f32 reference already disagrees with the *undeclared*
/// (W8A32) result — reassociation noise — so "differs from an f32 reference"
/// distinguishes nothing. Only the exact integer oracle does: W8A8 reproduces it
/// bit for bit, and W8A32 cannot, because it computes a different function.
#[test]
fn only_the_integer_oracle_separates_w8a8_from_w8a32() {
    let (k, n) = (64usize, 8usize);
    let a: Vec<f32> = (0..k).map(|i| act(3, i)).collect();

    let (arch_on, prov_on) = weightless_graph(1, k, n, true);
    let w8a8 = run(&arch_on, prov_on, &a);
    let oracle = w8a8_oracle(&a, k, n);

    // W8A8 *is* the oracle, exactly.
    for j in 0..n {
        assert_eq!(w8a8[j].to_bits(), oracle[j].to_bits(), "col {j}");
    }

    // The undeclared weight keeps W8A32 semantics over the same logical weight,
    // bound row-major. It is a different function, so it cannot reproduce the
    // oracle.
    let (arch_off, prov_off) = weightless_graph(1, k, n, false);
    let w8a32 = run(&arch_off, prov_off, &a);

    let differs = (0..n).any(|j| w8a32[j].to_bits() != oracle[j].to_bits());
    assert!(
        differs,
        "W8A32 reproduced the exact integer oracle — then the two paths are not \
         distinct functions and the opt-in means nothing"
    );

    // And the trap: a naive f32 reference ALSO differs from W8A32, so that
    // comparison would have "passed" for the wrong reason.
    let naive: Vec<f32> = (0..n)
        .map(|j| (0..k).map(|i| a[i] * (w(i, j) as f32 * scale(j))).sum())
        .collect();
    let naive_differs_from_w8a32 = (0..n).any(|j| naive[j].to_bits() != w8a32[j].to_bits());
    assert!(
        naive_differs_from_w8a32,
        "if a naive f32 loop matched W8A32 bit-for-bit, `differs from an f32 \
         reference` would be a valid discriminator — it is not, and this test \
         exists to keep that documented"
    );
}

/// A weightless archive carries no weight bytes. Loading it **fully resident**,
/// with no provider to bind them, must fail loud.
///
/// It used to pin an empty body: the slot took the κ-label of *empty content*
/// and every kernel downstream read zeros and derived addresses for a weight
/// that was never bound. A plausible answer is the worst answer.
#[test]
fn weightless_archive_without_a_provider_fails_loud() {
    let (k, n) = (64usize, 8usize);
    let (archive, _provider) = weightless_graph(1, k, n, true);

    let err = InferenceSession::<CpuBackend<BufferArena>>::load(&archive, CpuBackend::new())
        .err()
        .expect("a weightless archive must not load without a weight provider");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("weights") || msg.to_lowercase().contains("section"),
        "error should name the missing weight bodies, got: {msg}"
    );
}

/// **The reuse lever, witnessed through the shipped binding.**
///
/// Pooling scales prefill linearly in participants; the roofline shows the
/// kernels are already at the machine's ceilings. The lever that is *not*
/// linear is content addressing: a re-executed prefill whose inputs κ-match a
/// prior run is a **graph memo hit** — no kernel dispatches, no weight bytes
/// page, the output is returned by address. Its cost is hashing the inputs,
/// independent of model size. A shared system prompt re-executing across
/// requests rides this for free.
///
/// This is asserted through a weightless + paged session — the binding a real
/// consumer ships — not a toy inline graph: zero dispatches AND zero
/// additional provider bytes on the repeat, byte-identical output, and a
/// *different* prompt still recomputes (the memo is keyed on input content,
/// not stuck).
#[test]
fn repeated_prefill_through_the_shipped_binding_is_a_memo_hit() {
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct CountingKappaProvider {
        kappa: WeightFingerprint,
        body: Vec<u8>,
        served: AtomicUsize,
    }
    impl WeightProvider for CountingKappaProvider {
        fn size(&self, fp: WeightFingerprint) -> Option<usize> {
            (fp == self.kappa).then_some(self.body.len())
        }
        fn get_range(
            &self,
            fp: WeightFingerprint,
            offset: usize,
            len: usize,
        ) -> Option<Cow<'_, [u8]>> {
            if fp != self.kappa || offset + len > self.body.len() {
                return None;
            }
            self.served.fetch_add(len, Ordering::Relaxed);
            Some(Cow::Borrowed(&self.body[offset..offset + len]))
        }
    }

    let (m, k, n) = (4usize, 64usize, 8usize);
    let (archive, plain) = weightless_graph(m, k, n, true);
    let provider = Arc::new(CountingKappaProvider {
        kappa: plain.kappa,
        body: plain.body.clone(),
        served: AtomicUsize::new(0),
    });
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider.clone(), usize::MAX)
            .unwrap();

    let prompt_a: Vec<u8> = (0..m * k)
        .flat_map(|i| act(i / k, i % k).to_le_bytes())
        .collect();
    let first = sess.execute(&[InputBuffer { bytes: &prompt_a }]).unwrap()[0]
        .bytes
        .clone();
    let dispatched_cold = sess.last_dispatched();
    let served_cold = provider.served.load(Ordering::Relaxed);
    assert!(dispatched_cold > 0, "cold prefill must execute kernels");
    assert!(served_cold > 0, "cold prefill must page the weight in");

    // The repeat: same prompt, same κ — a memo hit end to end.
    let second = sess.execute(&[InputBuffer { bytes: &prompt_a }]).unwrap()[0]
        .bytes
        .clone();
    assert_eq!(second, first, "memoized output must be byte-identical");
    assert_eq!(
        sess.last_dispatched(),
        0,
        "a κ-matched prefill must dispatch no kernels"
    );
    assert_eq!(
        provider.served.load(Ordering::Relaxed),
        served_cold,
        "a κ-matched prefill must page no additional weight bytes"
    );

    // A different prompt is not served from the memo.
    let prompt_b: Vec<u8> = (0..m * k)
        .flat_map(|i| (act(i / k, i % k) + 0.25).to_le_bytes())
        .collect();
    let third = sess.execute(&[InputBuffer { bytes: &prompt_b }]).unwrap()[0]
        .bytes
        .clone();
    assert!(
        sess.last_dispatched() > 0,
        "a different prompt must recompute — the memo is keyed on input content"
    );
    assert_ne!(third, first, "different prompt, different output");
}
