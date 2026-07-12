//! Witnesses for κ-leases — host-owned residency. The transient pool's
//! two-generation window is residency by *recency*; a lease is residency by
//! *ownership*: the value survives every walk until released. The ownership
//! law under test: **a lease is a borrow, and the `KvCacheWrite` in-place
//! move requires unique ownership** — a leased cache steps by honest copy
//! (pre-image preserved: the speculative-rollback / branch primitive), and
//! releasing the last lease restores the move. Every outcome is compared
//! bitwise against the honest-copy kernel dispatched directly.

use hologram_backend::{
    Backend, BufferRef, CpuBackend, KernelCall, KvCacheWriteCall, SplitReads, Workspace,
};
use hologram_compiler::{compile, BackendKind};
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
        .map(|i| (((i * 19 + seed * 23) % 47) as f32 - 23.0) * 0.031)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}
impl TestWorkspace {
    fn push(&mut self, data: &[u8]) -> BufferRef {
        let slot = self.slots.len() as u32;
        self.slots.push(data.to_vec());
        BufferRef {
            slot,
            offset: 0,
            length: data.len() as u64,
        }
    }
}
impl Workspace for TestWorkspace {
    fn read(&self, b: BufferRef) -> &[u8] {
        &self.slots[b.slot as usize][..]
    }
    fn write(&mut self, b: BufferRef) -> &mut [u8] {
        let len = self.slots[b.slot as usize].len();
        let _ = b.length;
        &mut self.slots[b.slot as usize][..len]
    }
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])> {
        let w = write.slot as usize;
        if reads.iter().any(|r| r.slot as usize == w) {
            return None;
        }
        let (lo, hi) = self.slots.split_at_mut(w);
        let (wbuf, hi_rest) = hi.split_first_mut()?;
        let rs = reads
            .iter()
            .map(|r| {
                let i = r.slot as usize;
                if i < w {
                    &lo[i][..]
                } else {
                    &hi_rest[i - w - 1][..]
                }
            })
            .collect();
        Some((rs, wbuf.as_mut_slice()))
    }
}

/// Graph: `KvCacheWrite(cache, new, pos) → output`.
fn write_graph(b: u64, hkv: u64, bucket: u64, d: u64) -> Graph {
    let mut g = Graph::new();
    let cache_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let new_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, 1, d));
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
    let cache = input(cache_sh, DTYPE_F32);
    let new = input(new_sh, DTYPE_F32);
    let pos = input(pos_sh, DTYPE_I32);
    let w = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([cache, new, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(w)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    g.add_output(out);
    g
}

fn direct_write(cache: &[f32], new: &[f32], pos: u32, planes: u32, bucket: u32, d: u32) -> Vec<u8> {
    let mut ws = TestWorkspace { slots: Vec::new() };
    let rc = ws.push(&to_le(cache));
    let rn = ws.push(&to_le(new));
    let rp = ws.push(&pos.to_le_bytes());
    let ro = ws.push(&vec![0u8; cache.len() * 4]);
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(
        &KernelCall::KvCacheWrite(KvCacheWriteCall {
            cache: rc,
            new: rn,
            pos: rp,
            output: ro,
            planes,
            bucket_rows: bucket,
            new_rows: 1,
            row_bytes: d * 4,
        }),
        &mut ws,
    )
    .unwrap();
    ws.slots[ro.slot as usize].clone()
}

const DIMS: (u64, u64, u64, u64) = (1, 2, 6, 4); // b, hkv, bucket, d

fn setup() -> (
    InferenceSession<CpuBackend<BufferArena>>,
    Vec<f32>, // cache values
) {
    let (b, hkv, bucket, d) = DIMS;
    let g = write_graph(b, hkv, bucket, d);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let sess = InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let cache = f32s((b * hkv * bucket * d) as usize, 1);
    (sess, cache)
}

/// **Residency by ownership.** A leased label survives walks that never
/// mention it; an unleased sibling interned at the same time ages out. After
/// release the value is still consumable once (demoted, not dropped), then
/// ages out normally.
#[test]
fn leased_label_survives_unrelated_walks_and_ages_out_after_release() {
    let (mut sess, cache) = setup();
    let (b, hkv, bucket, d) = DIMS;
    let planes = (b * hkv) as u32;

    let parked = sess.intern_input(&to_le(&cache));
    let sibling = sess.intern_input(&to_le(&f32s((b * hkv * bucket * d) as usize, 2)));
    assert!(sess.retain_label(&parked), "resident value must lease");

    // Three unrelated walks (an interleaved conversation's steps).
    let mut other = f32s((b * hkv * bucket * d) as usize, 3);
    let mut lo = sess.intern_input(&to_le(&other));
    for step in 0..3u32 {
        let new = f32s((b * hkv * d) as usize, 40 + step as usize);
        let want = direct_write(&other, &new, step, planes, bucket as u32, d as u32);
        let ln = sess.intern_input(&to_le(&new));
        let lp = sess.intern_input(&step.to_le_bytes());
        let out = sess.execute_addressed(&[lo, ln, lp]).unwrap();
        assert_eq!(sess.resolve(&out[0]).unwrap(), &want[..]);
        lo = out[0];
        other = want
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
    }

    // The lease held; recency did not.
    assert_eq!(
        sess.resolve(&parked).unwrap(),
        &to_le(&cache)[..],
        "leased label must survive unrelated walks bit-intact"
    );
    assert!(
        sess.resolve(&sibling).is_none(),
        "unleased sibling must age out of the two-generation window"
    );

    // Released ⇒ demoted: consumable in the very next walk…
    assert!(sess.release_label(&parked));
    let new = f32s((b * hkv * d) as usize, 50);
    let want = direct_write(&cache, &new, 1, planes, bucket as u32, d as u32);
    let ln = sess.intern_input(&to_le(&new));
    let lp = sess.intern_input(&1u32.to_le_bytes());
    let out = sess.execute_addressed(&[parked, ln, lp]).unwrap();
    assert_eq!(sess.resolve(&out[0]).unwrap(), &want[..]);
    // …and consumed by that walk's move, exactly as an owned value should be.
    assert!(
        sess.resolve(&parked).is_none(),
        "unleased value is owned ⇒ moved"
    );
}

/// **The ownership law.** A leased cache steps by honest copy: the kernel
/// dispatches (no elision), the pre-image survives bit-intact, and the
/// post-image equals the copy kernel. Releasing the lease restores the move.
/// This is speculative decoding's accept/reject in substrate terms.
#[test]
fn lease_forces_the_honest_copy_and_release_restores_the_move() {
    let (mut sess, cache) = setup();
    let (b, hkv, bucket, d) = DIMS;
    let planes = (b * hkv) as u32;

    let pre = sess.intern_input(&to_le(&cache));
    assert!(sess.retain_label(&pre));

    // Speculative step on the borrowed cache.
    let draft_new = f32s((b * hkv * d) as usize, 7);
    let want = direct_write(&cache, &draft_new, 4, planes, bucket as u32, d as u32);
    let ln = sess.intern_input(&to_le(&draft_new));
    let lp = sess.intern_input(&4u32.to_le_bytes());
    let out = sess.execute_addressed(&[pre, ln, lp]).unwrap();
    assert_eq!(
        sess.resolve(&out[0]).unwrap(),
        &want[..],
        "post-image != copy kernel"
    );
    assert_eq!(
        sess.last_dispatched(),
        1,
        "a borrowed cache must take the honest copy, not the move"
    );
    // The rollback point is intact.
    assert_eq!(
        sess.resolve(&pre).unwrap(),
        &to_le(&cache)[..],
        "the leased pre-image must survive the step bit-intact"
    );

    // REJECT the draft: step again from the same pre-image with other data.
    let real_new = f32s((b * hkv * d) as usize, 8);
    let want2 = direct_write(&cache, &real_new, 4, planes, bucket as u32, d as u32);
    let ln2 = sess.intern_input(&to_le(&real_new));
    // Re-interning the same position bytes: the label may live only in the
    // previous generation by now — intern promotes it (a liveness statement),
    // so the returned label is always bindable.
    let lp2 = sess.intern_input(&4u32.to_le_bytes());
    let out2 = sess.execute_addressed(&[pre, ln2, lp2]).unwrap();
    assert_eq!(
        sess.resolve(&out2[0]).unwrap(),
        &want2[..],
        "rollback step != copy kernel"
    );

    // ACCEPT path: release the lease; the next step owns the value and moves.
    assert!(sess.release_label(&pre));
    let ln3 = sess.intern_input(&to_le(&f32s((b * hkv * d) as usize, 9)));
    let lp3 = sess.intern_input(&5u32.to_le_bytes());
    let out3 = sess.execute_addressed(&[pre, ln3, lp3]).unwrap();
    assert_eq!(
        sess.last_dispatched(),
        0,
        "sole ownership must restore the move"
    );
    assert!(sess.resolve(&pre).is_none(), "the moved value is consumed");
    assert!(sess.resolve(&out3[0]).is_some());
}

/// Leases are refcounted: two holders, one release ⇒ still owned; the move
/// stays declined until the last lease drops.
#[test]
fn leases_are_refcounted() {
    let (mut sess, cache) = setup();
    let (b, hkv, bucket, d) = DIMS;
    let planes = (b * hkv) as u32;

    let l = sess.intern_input(&to_le(&cache));
    assert!(sess.retain_label(&l));
    assert!(sess.retain_label(&l));
    assert!(sess.release_label(&l));

    // One lease still held ⇒ copy, pre-image intact.
    let new = f32s((b * hkv * d) as usize, 4);
    let want = direct_write(&cache, &new, 2, planes, bucket as u32, d as u32);
    let ln = sess.intern_input(&to_le(&new));
    let lp = sess.intern_input(&2u32.to_le_bytes());
    let out = sess.execute_addressed(&[l, ln, lp]).unwrap();
    assert_eq!(sess.resolve(&out[0]).unwrap(), &want[..]);
    assert_eq!(sess.last_dispatched(), 1);
    assert!(sess.resolve(&l).is_some());

    assert!(sess.release_label(&l));
    assert!(!sess.release_label(&l), "no lease left to release");
}

/// A label that is not resident cannot be leased; leasing reports honestly.
#[test]
fn leasing_a_nonresident_label_is_refused() {
    let (mut sess, cache) = setup();
    let l = sess.intern_input(&to_le(&cache));
    // Age it out with two unrelated… simpler: a label never interned.
    let ghost = {
        let mut sess2 = {
            let (s2, _) = setup();
            s2
        };
        sess2.intern_input(&[1, 2, 3, 4])
    };
    assert!(!sess.retain_label(&ghost));
    assert!(sess.retain_label(&l));
}
