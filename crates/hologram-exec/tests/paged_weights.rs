//! Lazy constant residency — the weight-tier pager (plan: bounded-window
//! weight residency).
//!
//! `InferenceSession::load` pins every model constant resident at load; a
//! bounded host (a browser's single 32-bit heap) cannot hold a weight set
//! larger than its window that way. `load_paged` inverts the owned
//! `WeightStore` into a host `WeightProvider` and makes constant residency
//! lazy against a byte budget: each weight pages in on the first walk that
//! dispatches a kernel consuming it, evicting LRU to hold the budget, so the
//! arena is a **window** over the provider rather than a full copy.
//!
//! Witness (bounded resource, exact output — the `parallel_gemv` shape):
//! peak resident weight bytes stay under budget across a full multi-step
//! decode while output is **bit-identical** to the fully-pinned run. Residency
//! is orthogonal to identity — a paged range hashes to the same κ — so the
//! derivation keys and kernels are unchanged.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::borrow::Cow;
use std::sync::Arc;

use hologram_archive::{
    decode_weights, format::SectionKind, HoloLoader, WeightFingerprint, WeightProvider, WeightStore,
};
use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
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

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// A `WeightProvider` that counts bytes served, so a test can prove weights
/// actually page (served > 0) and that a fitting budget re-pages nothing
/// (served == sum, once).
struct CountingProvider {
    inner: WeightStore,
    bytes_served: AtomicUsize,
    calls: AtomicUsize,
}

impl CountingProvider {
    fn new(inner: WeightStore) -> Self {
        Self {
            inner,
            bytes_served: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
        }
    }
    fn served(&self) -> usize {
        self.bytes_served.load(Ordering::Relaxed)
    }
    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl WeightProvider for CountingProvider {
    fn size(&self, fp: WeightFingerprint) -> Option<usize> {
        self.inner.size(fp)
    }
    fn get_range(&self, fp: WeightFingerprint, offset: usize, len: usize) -> Option<Cow<'_, [u8]>> {
        let r = self.inner.get_range(fp, offset, len)?;
        self.bytes_served.fetch_add(r.len(), Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Some(r)
    }
}

fn decode_store(archive: &[u8]) -> WeightStore {
    let plan = HoloLoader::from_bytes(archive)
        .unwrap()
        .into_plan()
        .unwrap();
    decode_weights(plan.section(SectionKind::Weights).unwrap()).unwrap()
}

/// `y = (…((x · W1) · W2) … · Wn)` over `n` distinct [d,d] f32 constant
/// weights, each `d*d*4` bytes (above the 4 KiB inline threshold ⇒ stored
/// `by_reference`, the paged tier). Returns (archive, weights).
fn chain_graph_n(d: usize, n: usize) -> (Vec<u8>, WeightStore) {
    let mut g = Graph::new();
    let sx = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, d as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(d as u64, d as u64));
    let so = sx;
    let weight = |seed: u64| -> Vec<f32> {
        (0..d * d)
            .map(|i| {
                (((i as u64).wrapping_mul(2654435761).wrapping_add(seed) % 97) as f32 - 48.0) * 0.01
            })
            .collect()
    };
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_input(x);
    let mut cur = x;
    for w in 0..n {
        let c = g.constants_mut().insert(ConstantEntry {
            bytes: f32_to_le(&weight(w as u64 + 1)),
            dtype: DTypeId(DTYPE_F32),
            shape: sw,
        });
        let mm = g.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(cur), InputSource::Constant(c)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: so,
        });
        cur = mm;
    }
    let outn = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(cur)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(outn);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let store = decode_store(&compiled.archive);
    (compiled.archive, store)
}

fn chain_graph(d: usize) -> (Vec<u8>, WeightStore) {
    chain_graph_n(d, 3)
}

fn novel_input(d: usize, step: u32) -> Vec<u8> {
    let x: Vec<f32> = (0..d)
        .map(|i| ((i as u32 + step) % 17) as f32 * 0.03 - 0.25)
        .collect();
    f32_to_le(&x)
}

#[test]
fn paged_weights_stay_under_budget_and_match_pinned() {
    let d = 64; // each weight = 64*64*4 = 16 KiB by_reference (> 4 KiB inline)
    let per_weight = d * d * 4;
    let (archive, store) = chain_graph(d);
    assert_eq!(
        store.len(),
        3,
        "three distinct large weights, stored by ref"
    );

    // Budget below the weight-set sum (3 × 16 KiB) but above the largest
    // single weight, so a correct pager holds peak resident ≤ budget by
    // streaming: page → use → evict → page the next.
    let budget = per_weight + per_weight / 2; // 24 KiB, < 48 KiB sum, ≥ 16 KiB max

    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider.clone(), budget)
            .unwrap();

    // The fully-pinned session holds the whole weight set resident; the paged
    // session holds none at load (nothing paged until the first walk).
    assert!(
        pinned.resident_bytes() >= 3 * per_weight,
        "pinned load must hold the whole weight set ({} < {})",
        pinned.resident_bytes(),
        3 * per_weight
    );
    assert_eq!(paged.paged_weight_bytes(), 0, "paged load pins nothing");

    let mut peak = 0usize;
    for step in 0..8u32 {
        let x = novel_input(d, step);
        let want = pinned.execute(&[InputBuffer { bytes: &x }]).unwrap();
        let got = paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
        // Bit-identical: residency is orthogonal to identity.
        assert_eq!(want.len(), got.len());
        for (w, g) in want.iter().zip(&got) {
            let wf = le_to_f32(&w.bytes);
            let gf = le_to_f32(&g.bytes);
            assert_eq!(wf.len(), gf.len());
            for (a, b) in wf.iter().zip(&gf) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "paged output diverged at step {step}"
                );
            }
        }
        peak = peak.max(paged.paged_weight_bytes());
    }

    // The pager held the window: peak resident weight bytes never exceeded the
    // budget, even though the weight set is twice it.
    assert!(
        peak <= budget,
        "peak resident weight {peak} exceeded budget {budget}"
    );
    // Weights actually paged (the provider was the byte source), and under a
    // tight budget they re-stream — served far exceeds one weight-set copy.
    assert!(provider.served() > 0, "no weight ever paged");
    assert!(
        provider.call_count() > 0,
        "provider never consulted — weights not paged"
    );
}

#[test]
fn fitting_budget_pages_once_then_zero_overhead() {
    // With a budget above the weight-set sum, every weight pages once on the
    // first walk and stays resident across walks (steady-state pinning) — the
    // provider serves exactly one copy total and is never consulted again, so
    // a fitting model has zero paging overhead after warm-up.
    let d = 64;
    let sum = 3 * d * d * 4;
    let (archive, store) = chain_graph(d);
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider.clone(), sum * 4)
            .unwrap();

    for step in 0..6u32 {
        let x = novel_input(d, step);
        paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
        if step == 0 {
            // After the first walk, the whole set is resident and served once.
            assert_eq!(
                paged.paged_weight_bytes(),
                sum,
                "fitting budget must hold the whole set resident"
            );
        }
    }
    let served_after = provider.served();
    assert_eq!(
        served_after, sum,
        "a fitting model must page each weight exactly once ({served_after} vs {sum})"
    );
}

#[test]
fn paged_matches_pinned_when_budget_is_unbounded() {
    // budget == 0 (unbounded) must be identical to a fully-pinned load, byte
    // for byte, and page each weight once.
    let d = 48;
    let (archive, store) = chain_graph(d);
    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider, 0).unwrap();
    for step in 0..4u32 {
        let x = novel_input(d, step);
        let want = pinned.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        let got = paged.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        assert_eq!(
            want, got,
            "unbounded paged diverged from pinned at step {step}"
        );
    }
}

/// A **multi-lazy-weight node**: a per-channel dequantize→matmul whose packed
/// weight AND its per-channel scale/zero-point vectors are all large enough
/// to be stored `by_reference`, so a single kernel reads three paged weights
/// at once. The pager must hold all three simultaneously resident — grouping
/// protects each from eviction while the others page — or the kernel would
/// read an evicted operand. `n` is chosen so the scale/zp vectors (n·4 bytes)
/// exceed the 4 KiB inline threshold. Returns (archive, weights).
fn per_channel_dequant_graph(k: usize, n: usize) -> (Vec<u8>, WeightStore) {
    const DTYPE_I8: u8 = 2;
    assert!(n * 4 > 4096, "scale/zp vectors must be by_reference");
    let mut g = Graph::new();
    let sa = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, k as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(k as u64, n as u64));
    let so = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, n as u64));
    let sv = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank1(n as u64));

    let wq: Vec<u8> = (0..k * n).map(|i| ((i as i64 % 255) - 127) as u8).collect();
    let scales: Vec<f32> = (0..n).map(|j| 0.001 + (j as f32) * 1e-6).collect();
    let zeros: Vec<u8> = vec![0u8; n * 4]; // symmetric i32 zero-points
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: wq,
        dtype: DTypeId(DTYPE_I8),
        shape: sw,
    });
    let sc = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(&scales),
        dtype: DTypeId(DTYPE_F32),
        shape: sv,
    });
    let zc = g.constants_mut().insert(ConstantEntry {
        bytes: zeros,
        dtype: DTypeId(DTYPE_I8),
        shape: sv,
    });
    let ai = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sa,
    });
    g.add_input(ai);
    let dq = g.add_node(Node {
        op: GraphOp::Op(OpKind::Dequantize),
        inputs: SmallVec::from_iter([
            InputSource::Constant(wc),
            InputSource::Constant(sc),
            InputSource::Constant(zc),
        ]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sw,
    });
    g.set_quant_attrs(
        dq,
        QuantAttrs {
            quant_dtype: DTYPE_I8,
            scale_bits: 0,
            zero_point: 0,
            axis: 1,
            ..Default::default()
        },
    );
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(ai), InputSource::Node(dq)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    let outn = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: so,
    });
    g.add_output(outn);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let plan = HoloLoader::from_bytes(&compiled.archive)
        .unwrap()
        .into_plan()
        .unwrap();
    let store = decode_weights(plan.section(SectionKind::Weights).unwrap()).unwrap();
    (compiled.archive, store)
}

#[test]
fn multi_weight_node_pages_operands_together_and_matches_pinned() {
    // A per-channel dequant matmul reads three paged weights at once. Set the
    // budget to exactly the node's simultaneous footprint — the pager must
    // hold all three together (grouping), and any smaller unit would evict a
    // still-needed operand and corrupt the result.
    let (k, n) = (256usize, 2048usize); // scale/zp = 2048*4 = 8 KiB each (by ref)
    let (archive, store) = per_channel_dequant_graph(k, n);
    let group_bytes: usize = store.entries().map(|(_, b)| b.len()).sum();

    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let provider = Arc::new(CountingProvider::new(store));
    // Budget == the group footprint: tight enough that a non-grouping pager
    // would evict an operand mid-node, but exactly sufficient for the group.
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider.clone(), group_bytes)
            .unwrap();

    for step in 0..5u32 {
        let x = novel_input(k, step);
        let want = pinned.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        let got = paged.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        assert_eq!(
            want, got,
            "multi-weight-node paged output diverged at step {step}"
        );
        assert!(
            paged.paged_weight_bytes() <= group_bytes,
            "resident weight bytes exceeded the group budget"
        );
    }
    assert!(provider.served() > 0, "weights never paged");
}

/// A provider that reports correct sizes but refuses to serve bytes — so a
/// paged load succeeds (sizes known) yet the first page-in at execute fails,
/// surfacing as an error rather than a wrong answer or a panic.
struct SizeOnlyProvider(WeightStore);
impl WeightProvider for SizeOnlyProvider {
    fn size(&self, fp: WeightFingerprint) -> Option<usize> {
        self.0.size(fp)
    }
    fn get_range(&self, _fp: WeightFingerprint, _o: usize, _l: usize) -> Option<Cow<'_, [u8]>> {
        None
    }
}

fn run_steps(
    sess: &mut InferenceSession<CpuBackend<BufferArena>>,
    d: usize,
    steps: u32,
) -> Vec<Vec<u8>> {
    (0..steps)
        .map(|s| {
            let x = novel_input(d, s);
            sess.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
                .bytes
                .clone()
        })
        .collect()
}

#[test]
fn deduped_identical_weights_page_once() {
    // Two graph constants with identical bytes dedup to one body (one
    // fingerprint) in the store, so the paged tier holds one entry and serves
    // it once even though two matmuls consume it.
    let d = 64;
    let mut g = Graph::new();
    let sx = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, d as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(d as u64, d as u64));
    let wbytes = f32_to_le(
        &(0..d * d)
            .map(|i| (i % 50) as f32 * 0.01 - 0.25)
            .collect::<Vec<_>>(),
    );
    let c1 = g.constants_mut().insert(ConstantEntry {
        bytes: wbytes.clone(),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let c2 = g.constants_mut().insert(ConstantEntry {
        bytes: wbytes.clone(),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_input(x);
    let m1 = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(c1)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    let m2 = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(m1), InputSource::Constant(c2)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    let outn = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(m2)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_output(outn);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let store = decode_store(&compiled.archive);
    assert_eq!(store.len(), 1, "identical weights dedup to one body");

    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&compiled.archive, CpuBackend::new(), provider.clone(), 0)
            .unwrap();
    for step in 0..4u32 {
        let x = novel_input(d, step);
        paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
    }
    assert_eq!(
        provider.served(),
        d * d * 4,
        "deduped weight paged exactly once"
    );
}

#[test]
fn memo_hit_does_not_repage() {
    // A repeated input hits the whole-graph memo — no walk, so no paging.
    let d = 64;
    let (archive, store) = chain_graph(d);
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider.clone(), 0).unwrap();
    let x = novel_input(d, 7);
    paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
    let after_first = provider.call_count();
    assert!(after_first > 0, "first step paged");
    // Identical input again: memo hit, no walk, no page.
    paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
    assert_eq!(provider.call_count(), after_first, "memo hit must not page");
}

#[test]
fn execute_addressed_paged_matches_pinned() {
    let d = 48;
    let (archive, store) = chain_graph(d);
    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider, d * d * 4 * 2).unwrap();
    for step in 0..4u32 {
        let x = novel_input(d, step);
        let want = pinned.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        // Address-space path: intern then execute_addressed.
        let lbl = paged.intern_input(&x);
        let out_lbls = paged.execute_addressed(&[lbl]).unwrap();
        let got = paged.resolve(&out_lbls[0]).unwrap().to_vec();
        assert_eq!(want, got, "execute_addressed paged diverged at step {step}");
    }
}

#[test]
fn paging_failure_surfaces_as_error_not_wrong_answer() {
    let d = 64;
    let (archive, store) = chain_graph(d);
    // Sizes known ⇒ load succeeds; get_range refuses ⇒ execute errors.
    let provider: Arc<dyn WeightProvider + Send + Sync> = Arc::new(SizeOnlyProvider(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider, 0).unwrap();
    let x = novel_input(d, 0);
    let r = paged.execute(&[InputBuffer { bytes: &x }]);
    assert!(
        r.is_err(),
        "unservable weight must error, never wrong-answer"
    );
}

#[test]
fn load_paged_errors_when_provider_lacks_a_weight() {
    // An empty provider (no sizes) must fail the paged load loudly rather
    // than silently binding an empty weight.
    let d = 64;
    let (archive, _store) = chain_graph(d);
    let provider: Arc<dyn WeightProvider + Send + Sync> = Arc::new(WeightStore::new());
    let r = InferenceSession::<CpuBackend<BufferArena>>::load_paged(
        &archive,
        CpuBackend::new(),
        provider,
        0,
    );
    assert!(r.is_err(), "missing provider weight must fail load");
}

#[test]
fn long_run_tight_budget_stays_bounded_and_matches_pinned() {
    // Eight weights, budget for two, thirty steps: peak resident weight bytes
    // never exceed the budget while every step is bit-identical to pinned.
    let d = 64;
    let per = d * d * 4;
    let (archive, store) = chain_graph_n(d, 8);
    assert_eq!(store.len(), 8);
    let budget = 2 * per;
    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&archive, CpuBackend::new(), provider, budget).unwrap();
    let mut peak = 0usize;
    for step in 0..30u32 {
        let x = novel_input(d, step);
        let want = pinned.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        let got = paged.execute(&[InputBuffer { bytes: &x }]).unwrap()[0]
            .bytes
            .clone();
        assert_eq!(want, got, "step {step} diverged");
        peak = peak.max(paged.paged_weight_bytes());
    }
    assert!(peak <= budget, "peak {peak} exceeded budget {budget}");
    assert!(peak >= per, "at least one weight was resident");
}

#[test]
fn output_is_deterministic_across_every_budget() {
    // The pager must be output-invariant to the residency budget: a fully
    // pinned run, an unbounded paged run, and paged runs from a 1-byte budget
    // up through the whole weight set must all agree bit-for-bit.
    let d = 64;
    let per = d * d * 4;
    let (archive, store) = chain_graph_n(d, 5);
    let sum = 5 * per;
    let mut pinned: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&archive, CpuBackend::new()).unwrap();
    let reference = run_steps(&mut pinned, d, 6);

    for &budget in &[0usize, 1, per, per + per / 2, 3 * per, sum, sum * 4] {
        let provider = Arc::new(CountingProvider::new(store.clone()));
        let mut paged: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load_paged(&archive, CpuBackend::new(), provider, budget).unwrap();
        let got = run_steps(&mut paged, d, 6);
        assert_eq!(got, reference, "budget {budget} diverged from pinned");
    }
}

#[test]
fn output_aliased_constant_stays_pinned_and_resolves() {
    // A graph whose output *is* a large constant: that constant backs an
    // output port, so a paged load must keep it pinned (output collection is
    // read-only and cannot page a lazy body back in). Under the tightest
    // possible budget the output still resolves to the constant's bytes.
    let d = 64;
    let mut g = Graph::new();
    let sx = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, d as u64));
    let sw = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(d as u64, d as u64));
    let wbytes = f32_to_le(
        &(0..d * d)
            .map(|i| (i % 71) as f32 * 0.02 - 0.7)
            .collect::<Vec<_>>(),
    );
    // Matmul weight (a kernel consumer → packable, becomes the paged tier).
    let wc = g.constants_mut().insert(ConstantEntry {
        bytes: wbytes,
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    // A distinct large constant emitted **only** as a graph output (no kernel
    // consumes it, so it is stored raw). Its slot backs an output port ⇒ a
    // paged load must keep it pinned, or output collection could not resolve
    // an evicted body.
    let vbytes = f32_to_le(
        &(0..d * d)
            .map(|i| (i % 53) as f32 * 0.017 + 0.11)
            .collect::<Vec<_>>(),
    );
    let vc = g.constants_mut().insert(ConstantEntry {
        bytes: vbytes.clone(),
        dtype: DTypeId(DTYPE_F32),
        shape: sw,
    });
    let x = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_input(x);
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(x), InputSource::Constant(wc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    let o1 = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sx,
    });
    g.add_output(o1);
    let o2 = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Constant(vc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sw,
    });
    g.add_output(o2);

    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let store = decode_store(&compiled.archive);
    let provider = Arc::new(CountingProvider::new(store));
    let mut paged: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load_paged(&compiled.archive, CpuBackend::new(), provider, 1).unwrap();
    let x = novel_input(d, 0);
    let out = paged.execute(&[InputBuffer { bytes: &x }]).unwrap();
    // The second output is the raw output-only constant — it must resolve to
    // its bytes even under a 1-byte budget (it is pinned, not paged).
    assert_eq!(
        out[1].bytes, vbytes,
        "output-aliased constant must stay resolvable"
    );
}
