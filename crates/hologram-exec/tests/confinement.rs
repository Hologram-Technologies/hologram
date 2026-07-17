//! Confinement witnesses — the resource analogue of "the canonical orbit
//! never leaves the phase box." A steady-state addressed decode loop must be
//! **O(1) in pool memory per step**: the caches move in place, the small
//! per-step operands recycle through the two transient generations and the
//! free list, and nothing accumulates. Drift in the memory dimension is as
//! much a defect as drift in the values; this pins both total allocation and
//! the resident-value count after warmup.
//!
//! Also pinned: `intern_input` is idempotent and allocation-free on repeated
//! bytes, including the promotion case (bytes whose label has aged into the
//! previous generation re-intern with no copy and are guaranteed bindable).

use hologram_compiler::{compile, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession};
use hologram_graph::{
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DTYPE_I32: u8 = 4;

fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 53) as f32 - 26.0) * 0.027)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The decode-step graph: `DecodeAttention(q, k, v, k_new, v_new, mask)` plus
/// two `KvCacheWrite`s — the same shape the timing example and hologram-ai's
/// decode drive.
fn step_graph(b: u64, h: u64, hkv: u64, bucket: u64, d: u64) -> Graph {
    let mut g = Graph::new();
    let q_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, h, 1, d));
    let kc_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let kn_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, 1, d));
    let mask_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(1, bucket + 1));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let mut input = |sh, dt: u8| {
        let n = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(dt),
            output_shape: sh,
        });
        g.add_input(n);
        InputSource::Node(n)
    };
    let q = input(q_sh, DTYPE_F32);
    let kc = input(kc_sh, DTYPE_F32);
    let vc = input(kc_sh, DTYPE_F32);
    let kn = input(kn_sh, DTYPE_F32);
    let vn = input(kn_sh, DTYPE_F32);
    let mask = input(mask_sh, DTYPE_F32);
    let pos = input(pos_sh, DTYPE_I32);
    let attn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Attention),
        inputs: SmallVec::from_iter([q, kc, vc, kn, vn, mask]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: q_sh,
    });
    let wk = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([kc, kn, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: kc_sh,
    });
    let wv = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([vc, vn, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: kc_sh,
    });
    for (src, sh) in [(attn, q_sh), (wk, kc_sh), (wv, kc_sh)] {
        let o = g.add_node(Node {
            op: GraphOp::Output,
            inputs: SmallVec::from_iter([InputSource::Node(src)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: sh,
        });
        g.add_output(o);
    }
    g
}

/// **Exact resource confinement.** After warmup, every further decode step
/// leaves total pool allocation AND the distinct-resident count exactly
/// where they were: the loop is O(1) memory per step, with no drift.
#[test]
fn addressed_decode_loop_is_confined_in_pool_memory() {
    let (b, h, hkv, bucket, d) = (1u64, 4u64, 2u64, 16u64, 8u64);
    let compiled = compile(
        step_graph(b, h, hkv, bucket, d),
        BackendKind::Cpu,
        WittLevel::W32,
    )
    .unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let kvn = (b * hkv * bucket * d) as usize;
    let mut lk = sess.intern_input(&to_le(&f32s(kvn, 1)));
    let mut lv = sess.intern_input(&to_le(&f32s(kvn, 2)));
    let l = (bucket + 1) as usize;

    let mut warm: Option<(usize, usize)> = None;
    for step in 0..10usize {
        let lq = sess.intern_input(&to_le(&f32s((b * h * d) as usize, 10 + step)));
        let lkn = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 20 + step)));
        let lvn = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 30 + step)));
        let mask: Vec<f32> = (0..l)
            .map(|j| {
                if j < 3 + step || j == bucket as usize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let lm = sess.intern_input(&to_le(&mask));
        let lp = sess.intern_input(&((3 + step) as u32).to_le_bytes());
        let out = sess
            .execute_addressed(&[lq, lk, lv, lkn, lvn, lm, lp])
            .unwrap();
        lk = out[1];
        lv = out[2];

        // Warmup: two walks fill both transient generations and the free
        // list; from step 3 on, the loop must be exactly steady.
        if step == 3 {
            warm = Some((sess.pool_allocated_bytes(), sess.resident_count()));
        }
        if step > 3 {
            let (wb, wc) = warm.unwrap();
            assert_eq!(
                sess.pool_allocated_bytes(),
                wb,
                "step {step}: pool allocation drifted from the steady state"
            );
            assert_eq!(
                sess.resident_count(),
                wc,
                "step {step}: resident-value count drifted from the steady state"
            );
        }
    }
}

/// `intern_input` is idempotent and allocation-free on repeat — including
/// after the label ages into the previous generation, where re-interning
/// **promotes** it (a liveness statement) instead of leaving the caller with
/// a label the next walk would refuse.
#[test]
fn intern_is_idempotent_allocation_free_and_promoting() {
    let (b, h, hkv, bucket, d) = (1u64, 2u64, 1u64, 4u64, 4u64);
    let compiled = compile(
        step_graph(b, h, hkv, bucket, d),
        BackendKind::Cpu,
        WittLevel::W32,
    )
    .unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let kvn = (b * hkv * bucket * d) as usize;
    let cache = to_le(&f32s(kvn, 5));
    let l1 = sess.intern_input(&cache);
    let bytes_after_first = sess.pool_allocated_bytes();
    let l2 = sess.intern_input(&cache);
    assert_eq!(l1, l2, "identical bytes must intern to one label");
    assert_eq!(
        sess.pool_allocated_bytes(),
        bytes_after_first,
        "re-interning resident bytes must allocate nothing"
    );

    // Age the label into the previous generation with one unrelated walk.
    let lk = sess.intern_input(&to_le(&f32s(kvn, 6)));
    let lv = sess.intern_input(&to_le(&f32s(kvn, 7)));
    let lq = sess.intern_input(&to_le(&f32s((b * h * d) as usize, 8)));
    let lkn = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 9)));
    let lvn = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 10)));
    let mask = vec![0.0f32; (bucket + 1) as usize];
    let lm = sess.intern_input(&to_le(&mask));
    let lp = sess.intern_input(&1u32.to_le_bytes());
    let out = sess
        .execute_addressed(&[lq, lk, lv, lkn, lvn, lm, lp])
        .unwrap();

    // Re-intern: same label, no copy, and the label is bindable again.
    let before = sess.pool_allocated_bytes();
    let l3 = sess.intern_input(&cache);
    assert_eq!(l1, l3);
    assert_eq!(
        sess.pool_allocated_bytes(),
        before,
        "promotion must move the existing buffer, not copy it"
    );
    let lq2 = sess.intern_input(&to_le(&f32s((b * h * d) as usize, 11)));
    let lkn2 = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 12)));
    let lvn2 = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 13)));
    let lm2 = sess.intern_input(&to_le(&mask));
    let lp2 = sess.intern_input(&2u32.to_le_bytes());
    sess.execute_addressed(&[lq2, l3, out[2], lkn2, lvn2, lm2, lp2])
        .expect("a just-interned label must always be bindable");
}
