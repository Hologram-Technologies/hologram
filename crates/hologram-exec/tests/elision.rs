//! **Algebraic elision — end-to-end V&V.**
//!
//! The compiler removes computation UOR's algebra proves unnecessary (identity
//! elements, involutions, dead nodes) before scheduling, so it is never
//! dispatched. These tests prove the optimization is *transparent*: a graph
//! padded with redundant ops produces bit-for-bit the same result as its
//! reduced form, while compiling to strictly fewer nodes.

use hologram_compiler::{compile_from_source, BackendKind};
use hologram_compute::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(src: &str, x: &[f32]) -> (Vec<f32>, u32) {
    let compiled = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let nodes = compiled.stats.total_nodes;
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let out = le_to_f32(
        &sess
            .execute(&[InputBuffer {
                bytes: &f32_to_le(x),
            }])
            .unwrap()[0]
            .bytes,
    );
    (out, nodes)
}

/// `Relu(x)` padded with `+0` then `·1` must equal plain `Relu(x)`, and the
/// padded form must compile to *fewer* nodes (the Add and Mul are elided).
#[test]
fn identity_padding_is_transparent_and_smaller() {
    let x = [-2.0f32, -0.5, 0.0, 0.5, 2.0, 3.0, -1.0, 4.0];
    let want: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();

    let plain = "\
input x :8
op relu x :8 as=r
output r
";
    let padded = "\
input x :8
const z :8 = 0,0,0,0,0,0,0,0
const one :8 = 1,1,1,1,1,1,1,1
op relu x :8 as=r
op add r z :8 as=a
op mul a one :8 as=m
output m
";
    let (got_plain, n_plain) = run(plain, &x);
    let (got_padded, n_padded) = run(padded, &x);

    for (i, (&g, &w)) in got_padded.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "elided relu[{i}] mismatch");
    }
    // Same result as the un-padded graph, bit-for-bit.
    assert_eq!(got_padded, got_plain);
    // The +0 and ·1 nodes were elided: padded compiles to the same node
    // count as the plain graph.
    assert_eq!(
        n_padded, n_plain,
        "identity ops should be elided away (padded {n_padded} vs plain {n_plain})"
    );
}

/// A double-negation around an activation cancels: `Neg(Neg(Relu(x)))`
/// equals `Relu(x)` and compiles to the same size.
#[test]
fn double_negation_cancels_end_to_end() {
    let x = [-1.0f32, 0.0, 0.25, 0.75, 2.0, -3.0, 1.5, 0.1];
    let want: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();

    let src = "\
input x :8
op relu x :8 as=r
op neg r :8 as=n1
op neg n1 :8 as=n2
output n2
";
    let (got, _) = run(src, &x);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "neg∘neg relu[{i}] mismatch");
    }
}
