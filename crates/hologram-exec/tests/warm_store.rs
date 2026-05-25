//! **WS-3: persisted κ-store warms across sessions.**
//!
//! The warm-start lattice (WS-1) gives a stable κ-label per constant-only
//! cone node. A persisted store keyed by those labels lets results computed
//! in one process warm a fresh one — even from a *labels-only* archive (no
//! baked fold). This proves: (1) a session persists the cone to a
//! file-backed store; (2) a fresh session over the same labels-only archive
//! warms from it and elides the cone on the first run, output == f64 ref;
//! (3) miss-safety — an empty store warms nothing and the run recomputes
//! correctly; (4) corruption-safety — a damaged store entry is ignored
//! (recompute), never a wrong answer.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, FileWarmStore, InferenceSession, InputBuffer, MemWarmStore};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
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
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
fn ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            o[i * n + j] = acc as f32;
        }
    }
    o
}
fn ref_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a + b).collect()
}

/// cone = add(A, B) (constant-only); out = matmul(X, cone) (input-dependent).
/// Returns the labels-only compiled archive.
fn labels_only_archive(a: &[f32], b: &[f32], n: usize) -> Vec<u8> {
    let mut g = Graph::new();
    let s = g
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(n as u64, n as u64));
    let ca = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(a),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let cb = g.constants_mut().insert(ConstantEntry {
        bytes: f32_to_le(b),
        dtype: DTypeId(DTYPE_F32),
        shape: s,
    });
    let cone = g.add_node(Node {
        op: GraphOp::Op(OpKind::Add),
        inputs: SmallVec::from_iter([InputSource::Constant(ca), InputSource::Constant(cb)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let xi = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_input(xi);
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(xi), InputSource::Node(cone)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: s,
    });
    g.add_output(out);
    // Note: `compile` does NOT fold (the CLI does); this archive is
    // labels-only, so any warming must come from the store.
    compile(g, BackendKind::Cpu, WittLevel::W32)
        .unwrap()
        .archive
}

fn load(archive: &[u8]) -> InferenceSession<CpuBackend<BufferArena>> {
    InferenceSession::load(archive, CpuBackend::new()).unwrap()
}

#[test]
fn ws3_persisted_store_warms_across_sessions() {
    let n = 8usize;
    let a = fill(n * n, 0xA1);
    let b = fill(n * n, 0xB2);
    let x = fill(n * n, 0xC3);
    let archive = labels_only_archive(&a, &b, n);
    let want = ref_matmul(&x, &ref_add(&a, &b), n, n, n);
    let check = |got_bytes: &[u8]| {
        let got = le_to_f32(got_bytes);
        let scale = want.iter().fold(0f64, |mx, &w| mx.max(f64::from(w).abs())) + 1e-9;
        let err = got
            .iter()
            .zip(&want)
            .map(|(&gv, &wv)| (f64::from(gv) - f64::from(wv)).abs() / scale)
            .fold(0f64, f64::max);
        assert!(
            err <= 1e-4,
            "output diverged from reference (err {err:.3e})"
        );
    };

    // Unique temp dir for this run (stands in for a cross-process cache root).
    let dir = std::env::temp_dir().join(format!(
        "holo-ws3-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // ── Process 1: compute + persist the cone ──────────────────────────
    {
        let mut store = FileWarmStore::open(&dir).unwrap();
        let mut sess = load(&archive);
        let persisted = sess.persist_cone(&mut store).unwrap();
        assert_eq!(persisted, 1, "the constant-only cone (add) is persisted");
    }

    // ── Process 2: fresh session, warm from the store ──────────────────
    {
        let store = FileWarmStore::open(&dir).unwrap();
        let mut sess = load(&archive);
        let warmed = sess.warm_from_store(&store);
        assert_eq!(warmed, 1, "the cone is warmed from the persisted store");
        let out = sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes
            .clone();
        assert_eq!(
            sess.last_dispatched(),
            1,
            "warmed cone is elided on the first run (only the matmul dispatches)"
        );
        check(&out);
    }

    // ── Miss-safety: an empty store warms nothing; the run recomputes ──
    {
        let empty = MemWarmStore::new();
        let mut sess = load(&archive);
        assert_eq!(sess.warm_from_store(&empty), 0, "empty store warms nothing");
        let out = sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes
            .clone();
        assert_eq!(
            sess.last_dispatched(),
            2,
            "with no warm source the full graph recomputes"
        );
        check(&out);
    }

    // ── Corruption-safety: a damaged store entry is ignored (recompute) ─
    {
        // Overwrite every stored entry with garbage; `get` must reject it.
        for entry in std::fs::read_dir(&dir).unwrap() {
            std::fs::write(entry.unwrap().path(), b"corrupt").unwrap();
        }
        let store = FileWarmStore::open(&dir).unwrap();
        let mut sess = load(&archive);
        assert_eq!(
            sess.warm_from_store(&store),
            0,
            "a corrupt entry fails its integrity check and is not warmed"
        );
        let out = sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(&x),
            }])
            .unwrap()[0]
            .bytes
            .clone();
        assert_eq!(sess.last_dispatched(), 2, "corrupt store ⇒ safe recompute");
        check(&out);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
