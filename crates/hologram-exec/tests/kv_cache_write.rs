//! End-to-end witnesses for the resident KV-cache write (`KvCacheWrite`) —
//! upstream gap 3: at decode, the carried K/V must never be re-hashed or
//! recopied per step. The addressed path realizes the write as a κ **move**:
//! the old cache label is retired, the buffer is mutated in place
//! (`O(new_rows)`), and the result is retained under the derived output
//! label — bound next step with no hash and no copy. Every witness here
//! compares against the honest-copy kernel dispatched directly, so the move
//! is pinned **bit-identical** to the copy; the elision itself is observed
//! through the session's dispatch counters and the retired label.

use hologram_archive::{decoder, format::SectionKind, HoloLoader};
use hologram_backend::{
    Backend, BufferRef, CpuBackend, DecodeAttentionCall, KernelCall, KvCacheWriteCall, SplitReads,
    Workspace,
};
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
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
        .map(|i| (((i * 17 + seed * 11) % 37) as f32 - 18.0) * 0.061)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Slot-indexed test workspace for the direct-kernel (honest copy) oracle leg.
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

fn input_node(g: &mut Graph, sh: hologram_graph::registry::ShapeId, dtype: u8) -> InputSource {
    let n = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(dtype),
        output_shape: sh,
    });
    g.add_input(n);
    InputSource::Node(n)
}

fn out_node(g: &mut Graph, src: InputSource, sh: hologram_graph::registry::ShapeId, dtype: u8) {
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([src]),
        output_dtype: DTypeId(dtype),
        output_shape: sh,
    });
    g.add_output(out);
}

/// Graph: `KvCacheWrite(cache, new, pos) → output`, cache `[b, hkv, bucket, d]`.
fn write_graph(b: u64, hkv: u64, bucket: u64, rows: u64, d: u64) -> Graph {
    let mut g = Graph::new();
    let cache_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let new_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, rows, d));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let cache = input_node(&mut g, cache_sh, DTYPE_F32);
    let new = input_node(&mut g, new_sh, DTYPE_F32);
    let pos = input_node(&mut g, pos_sh, DTYPE_I32);
    let w = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([cache, new, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    out_node(&mut g, InputSource::Node(w), cache_sh, DTYPE_F32);
    g
}

/// f32 oracle via the honest-copy kernel dispatched directly.
fn direct_write(
    cache: &[f32],
    new: &[f32],
    pos: u32,
    planes: u32,
    bucket: u32,
    rows: u32,
    d: u32,
) -> Vec<u8> {
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
            new_rows: rows,
            row_bytes: d * 4,
        }),
        &mut ws,
    )
    .unwrap();
    ws.slots[ro.slot as usize].clone()
}

/// The full gap-3 chain on one write: the archive carries discriminant 120
/// with the shape-derived geometry; the byte path and the addressed path both
/// equal the honest-copy kernel bit for bit; the addressed path *elides* the
/// kernel (0 dispatched) and *consumes* the input cache label (moved, no
/// longer resident, refused if re-presented under a fresh key).
#[test]
fn addressed_write_moves_the_cache_and_matches_the_copy_kernel_bitwise() {
    let (b, hkv, bucket, rows, d) = (1u64, 2u64, 8u64, 1u64, 4u64);
    let cache = f32s((b * hkv * bucket * d) as usize, 1);
    let new = f32s((b * hkv * rows * d) as usize, 2);
    let pos: u32 = 5;
    let want = direct_write(
        &cache,
        &new,
        pos,
        (b * hkv) as u32,
        bucket as u32,
        rows as u32,
        d as u32,
    );

    let g = write_graph(b, hkv, bucket, rows, d);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();

    // The archive carries the new call with shape-derived geometry.
    let plan = HoloLoader::from_bytes(&compiled.archive)
        .unwrap()
        .into_plan()
        .unwrap();
    let calls = decoder::decode_calls(plan.section(SectionKind::KernelCalls).unwrap()).unwrap();
    let wr: Vec<_> = calls
        .iter()
        .filter_map(|c| match c {
            KernelCall::KvCacheWrite(k) => Some(*k),
            _ => None,
        })
        .collect();
    assert_eq!(wr.len(), 1);
    assert_eq!(
        (
            wr[0].planes,
            wr[0].bucket_rows,
            wr[0].new_rows,
            wr[0].row_bytes
        ),
        ((b * hkv) as u32, bucket as u32, rows as u32, (d * 4) as u32)
    );

    // Byte path.
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let bufs = [to_le(&cache), to_le(&new), pos.to_le_bytes().to_vec()];
    let inputs: Vec<InputBuffer> = bufs.iter().map(|b| InputBuffer { bytes: b }).collect();
    let got = sess.execute(&inputs).unwrap();
    assert_eq!(got[0].bytes, want, "byte path != copy kernel");

    // Addressed path on a fresh session: intern once, execute by label.
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let lc = sess.intern_input(&to_le(&cache));
    let ln = sess.intern_input(&to_le(&new));
    let lp = sess.intern_input(&pos.to_le_bytes());
    let out = sess.execute_addressed(&[lc, ln, lp]).unwrap();
    assert_eq!(
        sess.resolve(&out[0]).unwrap(),
        &want[..],
        "move != copy kernel"
    );
    // The write was realized as a move: nothing dispatched, one elision.
    assert_eq!(
        sess.last_dispatched(),
        0,
        "kernel should be elided by the move"
    );
    assert_eq!(sess.last_skipped(), 1);
    // The old cache value was consumed — its label is retired, exactly the
    // "a resident value is bound by its address, never re-addressed" law.
    assert!(
        sess.resolve(&lc).is_none(),
        "moved cache label must be retired"
    );
    // Re-presenting the retired label under a fresh key (different pos) is
    // refused loud — the value is gone, not silently recomputed from
    // mutated bytes.
    let lp2 = sess.intern_input(&7u32.to_le_bytes());
    assert!(sess.execute_addressed(&[lc, ln, lp2]).is_err());

    // The decode loop shape: the step-2 cache is step 1's output label —
    // bound resident, no hash, no copy — and still equals the copy kernel.
    let new2 = f32s((b * hkv * rows * d) as usize, 9);
    let want2 = {
        let cache1: Vec<f32> = want
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        direct_write(
            &cache1,
            &new2,
            6,
            (b * hkv) as u32,
            bucket as u32,
            rows as u32,
            d as u32,
        )
    };
    let ln2 = sess.intern_input(&to_le(&new2));
    let lp6 = sess.intern_input(&6u32.to_le_bytes());
    let out2 = sess.execute_addressed(&[out[0], ln2, lp6]).unwrap();
    assert_eq!(
        sess.resolve(&out2[0]).unwrap(),
        &want2[..],
        "step 2 != copy kernel"
    );
    assert_eq!(sess.last_dispatched(), 0);
}

/// Byte-path sessions stay correct after the move: `execute` re-stores a
/// retired input label from the caller's bytes on the next walk, so the same
/// cache bytes presented twice compute (or memo-hit) the same answer.
#[test]
fn byte_path_survives_the_move_across_repeated_executes() {
    let (b, hkv, bucket, rows, d) = (1u64, 1u64, 6u64, 2u64, 4u64);
    let cache = f32s((b * hkv * bucket * d) as usize, 3);
    let new = f32s((b * hkv * rows * d) as usize, 4);
    let pos: u32 = 5; // wraps: rows land at 5 and 0
    let want = direct_write(
        &cache,
        &new,
        pos,
        (b * hkv) as u32,
        bucket as u32,
        rows as u32,
        d as u32,
    );

    let g = write_graph(b, hkv, bucket, rows, d);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let bufs = [to_le(&cache), to_le(&new), pos.to_le_bytes().to_vec()];
    let inputs: Vec<InputBuffer> = bufs.iter().map(|b| InputBuffer { bytes: b }).collect();
    for _ in 0..3 {
        let got = sess.execute(&inputs).unwrap();
        assert_eq!(got[0].bytes, want);
    }
}

/// Aliased-content hazard: two writes whose caches are *distinct input ports
/// carrying identical bytes* share one κ-label and one buffer. Neither may
/// mutate it (the other's port still reads it), so both must decline the move
/// and take the honest copy — and both outputs must still be exact.
#[test]
fn identical_cache_bytes_on_two_ports_decline_the_move_and_stay_exact() {
    let (b, hkv, bucket, rows, d) = (1u64, 1u64, 4u64, 1u64, 4u64);
    let mut g = Graph::new();
    let cache_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let new_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, rows, d));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let c1 = input_node(&mut g, cache_sh, DTYPE_F32);
    let c2 = input_node(&mut g, cache_sh, DTYPE_F32);
    let new = input_node(&mut g, new_sh, DTYPE_F32);
    let pos = input_node(&mut g, pos_sh, DTYPE_I32);
    for c in [c1, c2] {
        let w = g.add_node(Node {
            op: GraphOp::Op(OpKind::KvCacheWrite),
            inputs: SmallVec::from_iter([c, new, pos]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: cache_sh,
        });
        out_node(&mut g, InputSource::Node(w), cache_sh, DTYPE_F32);
    }
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();

    let cache = f32s((b * hkv * bucket * d) as usize, 5);
    let new_v = f32s((b * hkv * rows * d) as usize, 6);
    let posv: u32 = 2;
    let want = direct_write(
        &cache,
        &new_v,
        posv,
        (b * hkv) as u32,
        bucket as u32,
        rows as u32,
        d as u32,
    );

    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let lc = sess.intern_input(&to_le(&cache)); // one label, both ports
    let ln = sess.intern_input(&to_le(&new_v));
    let lp = sess.intern_input(&posv.to_le_bytes());
    let out = sess.execute_addressed(&[lc, lc, ln, lp]).unwrap();
    // Identical inputs ⇒ identical derived labels ⇒ the second write reuses
    // the first's result; what matters is that no write mutated the shared
    // buffer: the *input* label must still be resident and unchanged.
    assert_eq!(sess.resolve(&out[0]).unwrap(), &want[..]);
    assert_eq!(sess.resolve(&out[1]).unwrap(), &want[..]);
    assert_eq!(
        sess.resolve(&lc).unwrap(),
        &to_le(&cache)[..],
        "shared cache buffer must not be mutated by a declined move"
    );
    assert!(
        sess.last_dispatched() >= 1,
        "declined moves must dispatch the copy kernel"
    );
}

/// Ordering hazard: a compute consumer of the *written* cache forces the
/// write out of trailing placement, and a consumer of the *original* cache
/// scheduled after the write makes the move unsound. The load-time analysis
/// must mark it ineligible: `Add(cache, write(cache, …))` needs the original
/// cache bytes after the write ran. Exactness is the witness.
#[test]
fn in_graph_consumer_of_the_original_cache_declines_the_move_and_stays_exact() {
    let (b, hkv, bucket, rows, d) = (1u64, 1u64, 4u64, 1u64, 4u64);
    let mut g = Graph::new();
    let cache_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let new_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, rows, d));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let cache = input_node(&mut g, cache_sh, DTYPE_F32);
    let new = input_node(&mut g, new_sh, DTYPE_F32);
    let pos = input_node(&mut g, pos_sh, DTYPE_I32);
    let w = g.add_node(Node {
        op: GraphOp::Op(OpKind::KvCacheWrite),
        inputs: SmallVec::from_iter([cache, new, pos]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    let add = g.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([cache, InputSource::Node(w)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: cache_sh,
    });
    out_node(&mut g, InputSource::Node(add), cache_sh, DTYPE_F32);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();

    let cache_v = f32s((b * hkv * bucket * d) as usize, 7);
    let new_v = f32s((b * hkv * rows * d) as usize, 8);
    let posv: u32 = 1;
    let written = direct_write(
        &cache_v,
        &new_v,
        posv,
        (b * hkv) as u32,
        bucket as u32,
        rows as u32,
        d as u32,
    );
    let want: Vec<u8> = cache_v
        .iter()
        .zip(written.chunks_exact(4))
        .flat_map(|(a, wb)| (a + f32::from_le_bytes(wb.try_into().unwrap())).to_le_bytes())
        .collect();

    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let lc = sess.intern_input(&to_le(&cache_v));
    let ln = sess.intern_input(&to_le(&new_v));
    let lp = sess.intern_input(&posv.to_le_bytes());
    let out = sess.execute_addressed(&[lc, ln, lp]).unwrap();
    assert_eq!(sess.resolve(&out[0]).unwrap(), &want[..]);
    // Both kernels must have dispatched — the move was (correctly) refused.
    assert_eq!(sess.last_dispatched(), 2);
    assert!(sess.resolve(&lc).is_some(), "original cache must survive");
}

/// The full decode step, two tokens: `DecodeAttention` reads the resident
/// split KV while two `KvCacheWrite`s append the new K/V rows in place. The
/// addressed loop feeds each step's updated cache labels into the next —
/// no hash, no copy, writes elided — and every output equals the kernels
/// dispatched directly (honest copies) bit for bit.
#[test]
fn two_step_decode_loop_with_attention_matches_direct_kernels_bitwise() {
    let (b, h, hkv, m, bucket, new, d) = (1u64, 4u64, 2u64, 1u64, 8u64, 1u64, 8u64);
    let l = (bucket + new) as usize;

    // Graph: q, k_cache, v_cache, k_new, v_new, mask, pos →
    //   attn = DecodeAttention(q, k_cache, v_cache, k_new, v_new, mask)
    //   k'   = KvCacheWrite(k_cache, k_new, pos)
    //   v'   = KvCacheWrite(v_cache, v_new, pos)
    // outputs: [attn, k', v']
    let mut g = Graph::new();
    let q_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, h, m, d));
    let kc_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, bucket, d));
    let kn_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, new, d));
    let mask_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m, bucket + new));
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let q = input_node(&mut g, q_sh, DTYPE_F32);
    let kc = input_node(&mut g, kc_sh, DTYPE_F32);
    let vc = input_node(&mut g, kc_sh, DTYPE_F32);
    let kn = input_node(&mut g, kn_sh, DTYPE_F32);
    let vn = input_node(&mut g, kn_sh, DTYPE_F32);
    let mask = input_node(&mut g, mask_sh, DTYPE_F32);
    let pos = input_node(&mut g, pos_sh, DTYPE_I32);
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
    out_node(&mut g, InputSource::Node(attn), q_sh, DTYPE_F32);
    out_node(&mut g, InputSource::Node(wk), kc_sh, DTYPE_F32);
    out_node(&mut g, InputSource::Node(wv), kc_sh, DTYPE_F32);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();

    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    // Host-side oracle state (honest-copy kernels on raw slices).
    let mut cache_k = to_le(&f32s((b * hkv * bucket * d) as usize, 21));
    let mut cache_v = to_le(&f32s((b * hkv * bucket * d) as usize, 22));

    // Addressed-loop state.
    let mut lk = sess.intern_input(&cache_k);
    let mut lv = sess.intern_input(&cache_v);

    for step in 0..2usize {
        let realized = 3 + step as u32; // rows 0..realized of the bucket are live
        let q_v = f32s((b * h * m * d) as usize, 30 + step);
        let kn_v = f32s((b * hkv * new * d) as usize, 40 + step);
        let vn_v = f32s((b * hkv * new * d) as usize, 50 + step);
        // Mask: realized past rows + the new row visible; the rest erased.
        let mask_v: Vec<f32> = (0..m as usize * l)
            .map(|i| {
                let j = i % l;
                if j < realized as usize || j >= bucket as usize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();

        // Oracle: direct dispatch of the three kernels on the host state.
        let (attn_want, k_want, v_want) = {
            let mut ws = TestWorkspace { slots: Vec::new() };
            let rq = ws.push(&to_le(&q_v));
            let rkc = ws.push(&cache_k);
            let rvc = ws.push(&cache_v);
            let rkn = ws.push(&to_le(&kn_v));
            let rvn = ws.push(&to_le(&vn_v));
            let rm = ws.push(&to_le(&mask_v));
            let rp = ws.push(&realized.to_le_bytes());
            let ra = ws.push(&vec![0u8; (b * h * m * d) as usize * 4]);
            let rk2 = ws.push(&vec![0u8; cache_k.len()]);
            let rv2 = ws.push(&vec![0u8; cache_v.len()]);
            let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
            be.dispatch(
                &KernelCall::DecodeAttention(DecodeAttentionCall {
                    q: rq,
                    k_past: rkc,
                    v_past: rvc,
                    k_new: rkn,
                    v_new: rvn,
                    mask: rm,
                    output: ra,
                    batch: b as u32,
                    heads: h as u32,
                    kv_heads: hkv as u32,
                    q_rows: m as u32,
                    past_len: bucket as u32,
                    new_len: new as u32,
                    head_dim: d as u32,
                    scale_bits: 0,
                    dtype: DTYPE_F32,
                }),
                &mut ws,
            )
            .unwrap();
            for (rc, rn, ro) in [(rkc, rkn, rk2), (rvc, rvn, rv2)] {
                be.dispatch(
                    &KernelCall::KvCacheWrite(KvCacheWriteCall {
                        cache: rc,
                        new: rn,
                        pos: rp,
                        output: ro,
                        planes: (b * hkv) as u32,
                        bucket_rows: bucket as u32,
                        new_rows: new as u32,
                        row_bytes: (d * 4) as u32,
                    }),
                    &mut ws,
                )
                .unwrap();
            }
            (
                ws.slots[ra.slot as usize].clone(),
                ws.slots[rk2.slot as usize].clone(),
                ws.slots[rv2.slot as usize].clone(),
            )
        };

        // Addressed step: caches ride labels; only the small per-step
        // operands are interned (hashed).
        let lq = sess.intern_input(&to_le(&q_v));
        let lkn = sess.intern_input(&to_le(&kn_v));
        let lvn = sess.intern_input(&to_le(&vn_v));
        let lm = sess.intern_input(&to_le(&mask_v));
        let lp = sess.intern_input(&realized.to_le_bytes());
        let out = sess
            .execute_addressed(&[lq, lk, lv, lkn, lvn, lm, lp])
            .unwrap();
        assert_eq!(
            sess.resolve(&out[0]).unwrap(),
            &attn_want[..],
            "step {step}: attention != direct kernel"
        );
        assert_eq!(
            sess.resolve(&out[1]).unwrap(),
            &k_want[..],
            "step {step}: k cache != honest copy"
        );
        assert_eq!(
            sess.resolve(&out[2]).unwrap(),
            &v_want[..],
            "step {step}: v cache != honest copy"
        );
        // Attention dispatched; both cache writes realized as moves.
        assert_eq!(
            sess.last_dispatched(),
            1,
            "step {step}: writes must be elided"
        );
        assert_eq!(sess.last_skipped(), 2, "step {step}");
        assert!(
            sess.resolve(&lk).is_none(),
            "step {step}: k label must be moved"
        );
        assert!(
            sess.resolve(&lv).is_none(),
            "step {step}: v label must be moved"
        );

        // Advance both worlds.
        cache_k = k_want;
        cache_v = v_want;
        lk = out[1];
        lv = out[2];
    }
}

/// Confinement across the ring boundary: a long addressed decode loop whose
/// write position wraps the bucket (the `canon`/`emod` of the substrate's
/// phase space) stays bit-identical to the honest-copy kernels at every step,
/// with every write still realized as a move. The wrap is where a
/// representative leaves the naive `[0, bucket)` box and the exact remainder
/// brings it back — the step a drifting implementation would fumble.
#[test]
fn wrapping_decode_loop_matches_direct_kernels_bitwise() {
    let (b, hkv, bucket, d) = (1u64, 2u64, 4u64, 8u64);
    let planes = (b * hkv) as u32;
    let steps = 6usize; // positions 2,3,0,1,2,3 — wraps twice

    let g = write_graph(b, hkv, bucket, 1, d);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let mut cache = to_le(&f32s((b * hkv * bucket * d) as usize, 31));
    let mut label = sess.intern_input(&cache);
    for step in 0..steps {
        let pos = (2 + step as u32) % bucket as u32 + bucket as u32 * (step as u32 % 2);
        // pos deliberately exceeds the bucket on odd steps: the kernel (and
        // the move) must wrap it identically.
        let new = f32s((b * hkv * d) as usize, 60 + step);
        let want = {
            let cache_f: Vec<f32> = cache
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            direct_write(&cache_f, &new, pos, planes, bucket as u32, 1, d as u32)
        };
        let ln = sess.intern_input(&to_le(&new));
        let lp = sess.intern_input(&pos.to_le_bytes());
        let out = sess.execute_addressed(&[label, ln, lp]).unwrap();
        assert_eq!(
            sess.resolve(&out[0]).unwrap(),
            &want[..],
            "step {step} (pos {pos}): move != honest copy across the wrap"
        );
        assert_eq!(
            sess.last_dispatched(),
            0,
            "step {step}: write must stay a move"
        );
        cache = want;
        label = out[0];
    }
}

/// Ring-wrap crossing under the move: a long addressed loop whose write
/// position wraps past the bucket boundary (and past it again) stays
/// bit-identical to the honest-copy kernel at every step, with every write
/// still realized as a move. The bucket is the fundamental domain; the
/// wrapped position is `pos % bucket` — canonicalization by exact remainder,
/// and the move must respect it exactly.
#[test]
fn wrapping_decode_loop_stays_bitwise_and_moved() {
    let (b, hkv, bucket, rows, d) = (1u64, 2u64, 4u64, 1u64, 8u64);
    let planes = (b * hkv) as u32;
    let g = write_graph(b, hkv, bucket, rows, d);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let mut host: Vec<f32> = f32s((b * hkv * bucket * d) as usize, 11);
    let mut label = sess.intern_input(&to_le(&host));
    // 10 steps over a 4-row bucket: positions 3,4,5,… wrap twice.
    for step in 0..10u32 {
        let new = f32s((b * hkv * rows * d) as usize, 60 + step as usize);
        let pos = 3 + step; // raw position; the kernel wraps mod bucket
        let want = direct_write(
            &host,
            &new,
            pos,
            planes,
            bucket as u32,
            rows as u32,
            d as u32,
        );
        let ln = sess.intern_input(&to_le(&new));
        let lp = sess.intern_input(&pos.to_le_bytes());
        let out = sess.execute_addressed(&[label, ln, lp]).unwrap();
        assert_eq!(
            sess.resolve(&out[0]).unwrap(),
            &want[..],
            "step {step} (pos {pos}): move != copy kernel across the wrap"
        );
        assert_eq!(
            sess.last_dispatched(),
            0,
            "step {step}: write must stay a move"
        );
        host = want
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        label = out[0];
    }
}
