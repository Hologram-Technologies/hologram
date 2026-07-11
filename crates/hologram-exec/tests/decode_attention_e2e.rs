//! End-to-end witness for the 6-input decode-attention form: a graph
//! `Attention(q, k_past, v_past, k_new, v_new, mask)` compiles, the archive
//! round-trips the new `DecodeAttention` call, the session executes it, and
//! the bytes equal the kernel invoked directly. This is the configuration
//! hologram-ai's decode rewrite lowers to — witnessed through the full
//! compile→load→execute chain, not just the kernel.

use hologram_archive::{decoder, format::SectionKind, HoloLoader};
use hologram_backend::{Backend, BufferRef, CpuBackend, KernelCall, SplitReads, Workspace};
use hologram_compiler::{compile, BackendKind, CompileError};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;

fn f32s(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.043)
        .collect()
}
fn to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_graph(
    b: u64,
    h: u64,
    hkv: u64,
    m: u64,
    past: u64,
    new: u64,
    d: u64,
    causal_attr: bool,
) -> Graph {
    let mut g = Graph::new();
    let q_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, h, m, d));
    let kp_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, past, d));
    let kn_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank4(b, hkv, new, d));
    let mask_sh = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(m, past + new));

    let mut inputs = SmallVec::new();
    for sh in [q_sh, kp_sh, kp_sh, kn_sh, kn_sh, mask_sh] {
        let n = g.add_node(Node {
            op: GraphOp::Input,
            inputs: SmallVec::new(),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: sh,
        });
        g.add_input(n);
        inputs.push(InputSource::Node(n));
    }
    let attn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Attention),
        inputs,
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: q_sh,
    });
    if causal_attr {
        g.set_attention_attrs(
            attn,
            hologram_graph::AttentionAttrs {
                causal: true,
                scale_bits: 0,
            },
        );
    }
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(attn)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: q_sh,
    });
    g.add_output(out);
    g
}

/// Slot-indexed test workspace for the direct-kernel comparison leg.
struct TestWorkspace {
    slots: Vec<Vec<u8>>,
}
impl TestWorkspace {
    fn push(&mut self, data: &[f32]) -> BufferRef {
        let slot = self.slots.len() as u32;
        self.slots.push(to_le(data));
        BufferRef {
            slot,
            offset: 0,
            length: (data.len() * 4) as u64,
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

/// The full chain: 6-input node → archive carries `DecodeAttention` (new
/// discriminant, round-tripped) → session executes → bytes equal the kernel
/// dispatched directly on the same operands.
#[test]
fn six_input_attention_compiles_executes_and_matches_the_kernel_bitwise() {
    let (b, h, hkv, m, past, new, d) = (1u64, 4u64, 2u64, 1u64, 24u64, 1u64, 16u64);
    let l = (past + new) as usize;

    let q = f32s((b * h * m * d) as usize, 1);
    let kp = f32s((b * hkv * past * d) as usize, 2);
    let vp = f32s((b * hkv * past * d) as usize, 3);
    let kn = f32s((b * hkv * new * d) as usize, 4);
    let vn = f32s((b * hkv * new * d) as usize, 5);
    // Realized-length mask: last 4 past rows unrealized.
    let mask: Vec<f32> = (0..m as usize * l)
        .map(|i| {
            let j = i % l;
            if (20..24).contains(&j) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();

    let g = build_graph(b, h, hkv, m, past, new, d, false);
    let compiled = compile(g, BackendKind::Cpu, WittLevel::W32).unwrap();

    // The archive carries the new call, decoded from its own discriminant.
    let plan = HoloLoader::from_bytes(&compiled.archive)
        .unwrap()
        .into_plan()
        .unwrap();
    let calls = decoder::decode_calls(plan.section(SectionKind::KernelCalls).unwrap()).unwrap();
    let deco: Vec<_> = calls
        .iter()
        .filter_map(|c| match c {
            KernelCall::DecodeAttention(a) => Some(*a),
            _ => None,
        })
        .collect();
    assert_eq!(deco.len(), 1, "expected exactly one DecodeAttention call");
    let call = deco[0];
    assert_eq!(call.q_rows, m as u32);
    assert_eq!(call.past_len, past as u32);
    assert_eq!(call.new_len, new as u32);
    assert_eq!(call.kv_heads, hkv as u32);

    // Execute through the session.
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let bufs = [
        to_le(&q),
        to_le(&kp),
        to_le(&vp),
        to_le(&kn),
        to_le(&vn),
        to_le(&mask),
    ];
    let inputs: Vec<InputBuffer> = bufs.iter().map(|b| InputBuffer { bytes: b }).collect();
    let got = le_to_f32(&sess.execute(&inputs).unwrap()[0].bytes);

    // Direct kernel on the same operands.
    let mut ws = TestWorkspace { slots: Vec::new() };
    let (rq, rkp, rvp, rkn, rvn, rm) = (
        ws.push(&q),
        ws.push(&kp),
        ws.push(&vp),
        ws.push(&kn),
        ws.push(&vn),
        ws.push(&mask),
    );
    let ro = ws.push(&vec![0f32; (b * h * m * d) as usize]);
    let mut direct_call = call;
    direct_call.q = rq;
    direct_call.k_past = rkp;
    direct_call.v_past = rvp;
    direct_call.k_new = rkn;
    direct_call.v_new = rvn;
    direct_call.mask = rm;
    direct_call.output = ro;
    let mut be: CpuBackend<TestWorkspace> = CpuBackend::new();
    be.dispatch(&KernelCall::DecodeAttention(direct_call), &mut ws)
        .unwrap();
    let want = le_to_f32(&ws.slots[ro.slot as usize]);

    assert_eq!(got.len(), want.len());
    for (i, (gv, wv)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "cell {i}: session {gv} vs direct kernel {wv}"
        );
    }
}

/// The mask is the single masking authority: a 6-input node that ALSO sets
/// `AttentionAttrs::causal` is refused at compile time — two conflicting
/// authorities would mean guessing which wins.
#[test]
fn six_input_attention_with_causal_attr_is_rejected() {
    let g = build_graph(1, 4, 2, 1, 8, 1, 16, true);
    let err = compile(g, BackendKind::Cpu, WittLevel::W32)
        .err()
        .expect("causal attr on the 6-input form must be refused");
    assert!(
        matches!(err, CompileError::GraphValidation(_)),
        "expected GraphValidation, got {err:?}"
    );
}
