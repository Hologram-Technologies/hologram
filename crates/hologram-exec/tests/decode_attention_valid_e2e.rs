//! End-to-end witnesses for the scalar-mask decode attention (κ121): the
//! 6-input `Attention` node whose 6th operand is a `[1]` i32 lowers to the
//! new call; the mask form (rank-2 f32) still lowers to κ119; an ambiguous
//! 6th operand is refused at compile time. The headline: a full decode loop
//! whose per-token input traffic is **O(1)** — q, k_new, v_new, pos, and
//! valid_len (4 bytes) — with the caches riding κ-moves and no O(bucket)
//! mask ever built, interned, or hashed. Every step is compared bitwise
//! against the kernels dispatched directly.

use hologram_archive::{decoder, format::SectionKind, HoloLoader};
use hologram_backend::{
    Backend, BufferRef, CpuBackend, DecodeAttentionValidCall, KernelCall, KvCacheWriteCall,
    SplitReads, Workspace,
};
use hologram_compiler::{compile, BackendKind, CompileError};
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
        .map(|i| (((i * 13 + seed * 17) % 43) as f32 - 21.0) * 0.029)
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

fn out_node(g: &mut Graph, src: InputSource, sh: hologram_graph::registry::ShapeId) {
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([src]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: sh,
    });
    g.add_output(out);
}

/// The full decode-step graph on the scalar-mask form: κ121 attention plus
/// two KvCacheWrites, inputs `q, k_cache, v_cache, k_new, v_new, valid, pos`.
#[allow(clippy::type_complexity)]
fn step_graph(b: u64, h: u64, hkv: u64, bucket: u64, d: u64, sixth: (u8, u8)) -> Graph {
    // `sixth` = (rank tag: 1 ⇒ [1], 2 ⇒ [1, bucket+1]; dtype) — for the
    // discrimination tests.
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
    let sixth_sh = if sixth.0 == 1 {
        g.shape_registry_mut().intern(ShapeDescriptor::rank1(1))
    } else {
        g.shape_registry_mut()
            .intern(ShapeDescriptor::rank2(1, bucket + 1))
    };
    let pos_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
    let q = input_node(&mut g, q_sh, DTYPE_F32);
    let kc = input_node(&mut g, kc_sh, DTYPE_F32);
    let vc = input_node(&mut g, kc_sh, DTYPE_F32);
    let kn = input_node(&mut g, kn_sh, DTYPE_F32);
    let vn = input_node(&mut g, kn_sh, DTYPE_F32);
    let six = input_node(&mut g, sixth_sh, sixth.1);
    let pos = input_node(&mut g, pos_sh, DTYPE_I32);
    let attn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Attention),
        inputs: SmallVec::from_iter([q, kc, vc, kn, vn, six]),
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
    out_node(&mut g, InputSource::Node(attn), q_sh);
    out_node(&mut g, InputSource::Node(wk), kc_sh);
    out_node(&mut g, InputSource::Node(wv), kc_sh);
    g
}

/// The 6th operand selects the form: `[1]` i32 lowers to κ121, `[…]` f32
/// rank-2 lowers to κ119, and anything else is refused at compile time.
#[test]
fn sixth_operand_shape_and_dtype_select_the_form() {
    let dims = (1u64, 4u64, 2u64, 8u64, 16u64);
    let (b, h, hkv, bucket, d) = dims;

    let count = |g: Graph, want121: usize, want119: usize| {
        let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
        let plan = HoloLoader::from_bytes(&compiled.archive)
            .unwrap()
            .into_plan()
            .unwrap();
        let calls = decoder::decode_calls(plan.section(SectionKind::KernelCalls).unwrap()).unwrap();
        let n121 = calls
            .iter()
            .filter(|c| matches!(c, KernelCall::DecodeAttentionValid(_)))
            .count();
        let n119 = calls
            .iter()
            .filter(|c| matches!(c, KernelCall::DecodeAttention(_)))
            .count();
        assert_eq!((n121, n119), (want121, want119));
    };

    count(step_graph(b, h, hkv, bucket, d, (1, DTYPE_I32)), 1, 0);
    count(step_graph(b, h, hkv, bucket, d, (2, DTYPE_F32)), 0, 1);
    // Ambiguous: rank-1 f32 is neither form — refused, not guessed.
    let err = compile(
        step_graph(b, h, hkv, bucket, d, (1, DTYPE_F32)),
        BackendKind::Cpu,
        WittLevel::W32,
    )
    .err()
    .expect("rank-1 f32 sixth operand must be refused");
    assert!(matches!(err, CompileError::GraphValidation(_)));
}

/// **The O(1) decode loop.** Ten steps (crossing the ring wrap) through the
/// compiled κ121 step graph: per-token inputs are q, k_new, v_new, pos, and
/// the 4-byte valid_len — no mask exists anywhere. Attention output and both
/// moved caches are bitwise-equal to the kernels dispatched directly, every
/// step; both cache writes stay κ-moves; pool allocation is exactly steady
/// after warmup (nothing O(bucket) is being interned per token).
#[test]
fn valid_form_decode_loop_is_o1_per_token_and_bitwise() {
    let (b, h, hkv, bucket, d) = (1u64, 4u64, 2u64, 6u64, 8u64);
    let planes = (b * hkv) as u32;
    let g = step_graph(b, h, hkv, bucket, d, (1, DTYPE_I32));
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();

    let mut cache_k = to_le(&f32s((b * hkv * bucket * d) as usize, 21));
    let mut cache_v = to_le(&f32s((b * hkv * bucket * d) as usize, 22));
    let mut lk = sess.intern_input(&cache_k);
    let mut lv = sess.intern_input(&cache_v);

    let mut warm: Option<usize> = None;
    for step in 0..10u32 {
        let realized = 3 + step; // grows past the bucket: the clamp regime
        let pos = realized % bucket as u32; // ring write position
        let q_v = f32s((b * h * d) as usize, 30 + step as usize);
        let kn_v = f32s((b * hkv * d) as usize, 40 + step as usize);
        let vn_v = f32s((b * hkv * d) as usize, 50 + step as usize);

        // Oracle: the kernels dispatched directly on host state.
        let (attn_want, k_want, v_want) = {
            let mut ws = TestWorkspace { slots: Vec::new() };
            let rq = ws.push(&to_le(&q_v));
            let rkc = ws.push(&cache_k);
            let rvc = ws.push(&cache_v);
            let rkn = ws.push(&to_le(&kn_v));
            let rvn = ws.push(&to_le(&vn_v));
            let rvl = ws.push(&realized.to_le_bytes());
            let rp = ws.push(&pos.to_le_bytes());
            let ra = ws.push(&vec![0u8; (b * h * d) as usize * 4]);
            let rk2 = ws.push(&vec![0u8; cache_k.len()]);
            let rv2 = ws.push(&vec![0u8; cache_v.len()]);
            let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
            be.dispatch(
                &KernelCall::DecodeAttentionValid(DecodeAttentionValidCall {
                    q: rq,
                    k_past: rkc,
                    v_past: rvc,
                    k_new: rkn,
                    v_new: rvn,
                    valid_len: rvl,
                    output: ra,
                    batch: b as u32,
                    heads: h as u32,
                    kv_heads: hkv as u32,
                    q_rows: 1,
                    past_len: bucket as u32,
                    new_len: 1,
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
                        planes,
                        bucket_rows: bucket as u32,
                        new_rows: 1,
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

        // Addressed step: O(1) per-token interning — no mask anywhere.
        let lq = sess.intern_input(&to_le(&q_v));
        let lkn = sess.intern_input(&to_le(&kn_v));
        let lvn = sess.intern_input(&to_le(&vn_v));
        let lvl = sess.intern_input(&realized.to_le_bytes());
        let lp = sess.intern_input(&pos.to_le_bytes());
        let out = sess
            .execute_addressed(&[lq, lk, lv, lkn, lvn, lvl, lp])
            .unwrap();
        assert_eq!(
            sess.resolve(&out[0]).unwrap(),
            &attn_want[..],
            "step {step}: attention != direct kernel"
        );
        assert_eq!(
            sess.resolve(&out[1]).unwrap(),
            &k_want[..],
            "step {step}: k cache"
        );
        assert_eq!(
            sess.resolve(&out[2]).unwrap(),
            &v_want[..],
            "step {step}: v cache"
        );
        assert_eq!(
            sess.last_dispatched(),
            1,
            "step {step}: writes must stay moves"
        );

        // Confinement: nothing O(bucket) interns per token. Warm at step 4:
        // in steps 0..3 `realized` and `pos` are byte-identical and dedupe
        // to one label, so the second 4-byte size class only enters the
        // free-list cycle once they diverge at the ring wrap.
        if step == 4 {
            warm = Some(sess.pool_allocated_bytes());
        }
        if step > 4 {
            assert_eq!(
                sess.pool_allocated_bytes(),
                warm.unwrap(),
                "step {step}: allocation drifted — something O(bucket) is per-token"
            );
        }

        cache_k = k_want;
        cache_v = v_want;
        lk = out[1];
        lv = out[2];
    }
}
