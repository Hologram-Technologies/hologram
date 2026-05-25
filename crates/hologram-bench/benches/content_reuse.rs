//! Content-addressed reuse benchmark.
//!
//! A depth-8 chain of 128×128×128 f32 matmuls against **constant** weight
//! matrices, driven through a full `InferenceSession` and re-executed with
//! the same input. This is the shape content-addressing targets: weights
//! are leaves addressed once at load; only the single activation input is
//! hashed per execute, and that one hash is amortized over the whole
//! chain's compute. A repeated execution with identical input resolves to
//! a graph-level memo hit and returns cached outputs without re-running
//! any of the 8 matmuls.
//!
//! Run with `--baseline <name>` against a pre-content-addressing build to
//! see recompute-every-time vs memo-hit.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::{
    constant::ConstantEntry,
    node::Node,
    registry::{DTypeId, ShapeDescriptor},
    Graph, GraphOp, InputSource, OpKind,
};
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const DTYPE_F32: u8 = 8;
const DIM: u64 = 128;
const DEPTH: usize = 8;

fn build_chain_session() -> InferenceSession<CpuBackend<BufferArena>> {
    let mut graph = Graph::new();
    let shape = graph
        .shape_registry_mut()
        .intern(ShapeDescriptor::rank2(DIM, DIM));

    let x = graph.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_input(x);

    // Chain: acc = Wi · acc, with each Wi a distinct constant weight.
    let mut acc = x;
    for layer in 0..DEPTH {
        let w_bytes: Vec<u8> = (0..(DIM * DIM) as usize)
            .flat_map(|i| ((i + layer) as f32 * 0.01).to_le_bytes())
            .collect();
        let w = graph.constants_mut().insert(ConstantEntry {
            bytes: w_bytes,
            dtype: DTypeId(DTYPE_F32),
            shape,
        });
        acc = graph.add_node(Node {
            op: GraphOp::Op(OpKind::MatMul),
            inputs: SmallVec::from_iter([InputSource::Node(acc), InputSource::Constant(w)]),
            output_dtype: DTypeId(DTYPE_F32),
            output_shape: shape,
        });
    }
    let out = graph.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(acc)]),
        output_dtype: DTypeId(DTYPE_F32),
        output_shape: shape,
    });
    graph.add_output(out);

    let compiled = compile(graph, BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap()
}

/// Steady state = graph-level memo hit: identical input ⇒ cached outputs,
/// the 8-matmul chain is never re-run.
fn bench_repeated(c: &mut Criterion) {
    let mut session = build_chain_session();
    let elems = (DIM * DIM) as usize;
    let x: Vec<u8> = (0..elems).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let inputs = [InputBuffer { bytes: &x }];
    let _ = session.execute(&inputs).unwrap(); // prime the memo

    c.bench_function("content_reuse::chain8_128_repeated_memo_hit", |bencher| {
        bencher.iter(|| {
            let out = session.execute(black_box(&inputs)).unwrap();
            black_box(out);
        });
    });
}

/// Novel input every iteration ⇒ graph-memo miss ⇒ the full 8-matmul
/// chain recomputes. This is the no-reuse cost (recompute + the
/// content-addressing overhead of one input hash + per-op label folds).
fn bench_novel(c: &mut Criterion) {
    let mut session = build_chain_session();
    let elems = (DIM * DIM) as usize;
    let mut x: Vec<u8> = (0..elems).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut ctr: u32 = 0;

    c.bench_function("content_reuse::chain8_128_novel_recompute", |bencher| {
        bencher.iter(|| {
            ctr = ctr.wrapping_add(1);
            x[0..4].copy_from_slice(&ctr.to_le_bytes());
            let inputs = [InputBuffer { bytes: &x }];
            let out = session.execute(black_box(&inputs)).unwrap();
            black_box(out);
        });
    });
}

/// The fully content-addressed reuse path: intern the input once, then
/// drive `execute_addressed` on its κ-label. Steady state hashes nothing
/// and copies no tensor bytes — just a graph-memo lookup returning the
/// cached output label (TC-01 zero-cost). This is what a pipeline that
/// feeds output labels into the next step pays per repeated step.
fn bench_addressed(c: &mut Criterion) {
    let mut session = build_chain_session();
    let elems = (DIM * DIM) as usize;
    let x: Vec<u8> = (0..elems).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let lx = session.intern_input(&x);
    let _ = session.execute_addressed(&[lx]).unwrap(); // prime the memo

    c.bench_function("content_reuse::chain8_128_addressed_memo_hit", |bencher| {
        bencher.iter(|| {
            let out = session.execute_addressed(black_box(&[lx])).unwrap();
            black_box(out);
        });
    });
}

criterion_group!(benches, bench_repeated, bench_novel, bench_addressed);
criterion_main!(benches);
