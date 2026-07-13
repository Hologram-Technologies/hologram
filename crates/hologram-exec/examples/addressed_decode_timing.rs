//! Gap-3 measurement: the per-token cost of carrying the KV cache across the
//! byte boundary vs. keeping it resident and addressed.
//!
//! One compiled decode-step graph — `DecodeAttention(q, k, v, k_new, v_new,
//! mask)` plus two `KvCacheWrite`s — driven two ways:
//!
//! - **byte loop** (`execute`): the caller carries `past_k`/`past_v` as host
//!   bytes. Every step re-hashes both caches (BLAKE3 over O(bucket) bytes),
//!   copies them into the pool, and copies the updated caches back out.
//! - **addressed loop** (`execute_addressed`): the caches ride κ-labels; the
//!   `KvCacheWrite`s realize as in-place moves. Per step, only the small
//!   operands (q, k_new, v_new, mask, pos) are hashed; the O(bucket) KV
//!   bytes are never re-hashed and never copied.
//!
//! Both loops compute bit-identical caches (pinned by
//! `tests/kv_cache_write.rs`); this example prints what the boundary costs.
//! Native timing — the deployed wasm32 target hashes slower, so the byte
//! column is an *optimistic lower bound* on what the addressed path removes.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;
use std::time::Instant;

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

fn step_graph(b: u64, h: u64, hkv: u64, bucket: u64, d: u64) -> Graph {
    let m = 1u64;
    let new = 1u64;
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

/// The κ121 step graph: identical shape, but the 6th attention operand is
/// the `[1]` i32 `valid_len` scalar — no mask tensor exists in the graph.
fn step_graph_valid(b: u64, h: u64, hkv: u64, bucket: u64, d: u64) -> Graph {
    let m = 1u64;
    let new = 1u64;
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
    let vl_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank1(1));
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
    let vl = input(vl_sh, DTYPE_I32);
    let pos = input(pos_sh, DTYPE_I32);
    let attn = g.add_node(Node {
        op: GraphOp::Op(OpKind::Attention),
        inputs: SmallVec::from_iter([q, kc, vc, kn, vn, vl]),
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

fn main() {
    let (b, h, hkv, d) = (1u64, 12u64, 2u64, 128u64);
    println!("decode step: DecodeAttention + 2×KvCacheWrite, h={h} kv={hkv} d={d}, m=1");
    println!(
        "{:>8} {:>6} | {:>14} {:>14} {:>13} {:>13} | {:>10}",
        "bucket", "steps", "byte µs/step", "addr µs/step", "valid(3+s)", "valid(full)", "boundary"
    );
    for &bucket in &[2048u64, 8192, 32768] {
        let steps = (65536 / bucket).max(4) as usize;
        let l = (bucket + 1) as usize;
        let graph = step_graph(b, h, hkv, bucket, d);
        let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
        let kvn = (b * hkv * bucket * d) as usize;

        // Per-step operands, precomputed so the loop times only the session.
        let qs: Vec<Vec<u8>> = (0..steps)
            .map(|s| to_le(&f32s((b * h * d) as usize, s)))
            .collect();
        let kns: Vec<Vec<u8>> = (0..steps)
            .map(|s| to_le(&f32s((b * hkv * d) as usize, 100 + s)))
            .collect();
        let vns: Vec<Vec<u8>> = (0..steps)
            .map(|s| to_le(&f32s((b * hkv * d) as usize, 200 + s)))
            .collect();
        let masks: Vec<Vec<u8>> = (0..steps)
            .map(|s| {
                to_le(
                    &(0..l)
                        .map(|j| {
                            if j < 3 + s || j >= bucket as usize {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f32>>(),
                )
            })
            .collect();
        let poss: Vec<Vec<u8>> = (0..steps)
            .map(|s| ((3 + s) as u32).to_le_bytes().to_vec())
            .collect();
        let cache0_k = to_le(&f32s(kvn, 7));
        let cache0_v = to_le(&f32s(kvn, 8));

        // Byte loop: the caller carries the caches across the boundary.
        let mut sess: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
        let (mut ck, mut cv) = (cache0_k.clone(), cache0_v.clone());
        let t0 = Instant::now();
        for s in 0..steps {
            let bufs = [&qs[s][..], &ck, &cv, &kns[s], &vns[s], &masks[s], &poss[s]];
            let inputs: Vec<InputBuffer> = bufs.iter().map(|b| InputBuffer { bytes: b }).collect();
            let mut out = sess.execute(&inputs).unwrap();
            cv = out.pop().unwrap().bytes;
            ck = out.pop().unwrap().bytes;
        }
        let byte_us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;

        // Addressed loop: the caches ride labels; writes realize as moves.
        let mut sess: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
        let mut lk = sess.intern_input(&cache0_k);
        let mut lv = sess.intern_input(&cache0_v);
        let t0 = Instant::now();
        for s in 0..steps {
            let lq = sess.intern_input(&qs[s]);
            let lkn = sess.intern_input(&kns[s]);
            let lvn = sess.intern_input(&vns[s]);
            let lm = sess.intern_input(&masks[s]);
            let lp = sess.intern_input(&poss[s]);
            let out = sess
                .execute_addressed(&[lq, lk, lv, lkn, lvn, lm, lp])
                .unwrap();
            lk = out[1];
            lv = out[2];
        }
        let addr_us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;

        // κ121 loop: the mask is gone — per-token inputs are q, k_new, v_new,
        // pos, and 4 bytes of valid_len; the kernel reads only the realized
        // prefix, so attention work is O(realized), not O(bucket).
        let vgraph = step_graph_valid(b, h, hkv, bucket, d);
        let vcompiled = compile(vgraph, BackendKind::Cpu, WittLevel::W32).unwrap();
        let mut sess: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load(&vcompiled.archive, CpuBackend::new()).unwrap();
        let mut lk = sess.intern_input(&cache0_k);
        let mut lv = sess.intern_input(&cache0_v);
        let t0 = Instant::now();
        for s in 0..steps {
            let lq = sess.intern_input(&qs[s]);
            let lkn = sess.intern_input(&kns[s]);
            let lvn = sess.intern_input(&vns[s]);
            let lvl = sess.intern_input(&((3 + s) as u32).to_le_bytes());
            let lp = sess.intern_input(&poss[s]);
            let out = sess
                .execute_addressed(&[lq, lk, lv, lkn, lvn, lvl, lp])
                .unwrap();
            lk = out[1];
            lv = out[2];
        }
        let valid_us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;

        // κ121 at FULL realization (valid = bucket): the kernel now reads the
        // whole bucket — the honest compute floor at this context — while the
        // input path stays O(1). The gap between this and `addr` is the mask
        // hash + the mask-slot walk; the gap between this and `valid` above
        // is the O(realized) read law.
        let mut sess: InferenceSession<CpuBackend<BufferArena>> =
            InferenceSession::load(&vcompiled.archive, CpuBackend::new()).unwrap();
        let mut lk = sess.intern_input(&cache0_k);
        let mut lv = sess.intern_input(&cache0_v);
        let t0 = Instant::now();
        for s in 0..steps {
            let lq = sess.intern_input(&qs[s]);
            let lkn = sess.intern_input(&kns[s]);
            let lvn = sess.intern_input(&vns[s]);
            let lvl = sess.intern_input(&(bucket as u32).to_le_bytes());
            let lp = sess.intern_input(&poss[s]);
            let out = sess
                .execute_addressed(&[lq, lk, lv, lkn, lvn, lvl, lp])
                .unwrap();
            lk = out[1];
            lv = out[2];
        }
        let validfull_us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;

        println!(
            "{:>8} {:>6} | {:>14.1} {:>14.1} {:>13.1} {:>13.1} | {:>9.1}µs ({:.2}×)",
            bucket,
            steps,
            byte_us,
            addr_us,
            valid_us,
            validfull_us,
            byte_us - addr_us,
            byte_us / addr_us,
        );
    }
    println!("\n'boundary' = per-step cost the addressed path removes: 2×O(bucket)");
    println!("BLAKE3 re-hash + 2×O(bucket) copy-in + 2×O(bucket) copy-out + the");
    println!("2×O(bucket) honest-copy writes, all replaced by label binds and");
    println!("two O(1)-row in-place moves.");
}
