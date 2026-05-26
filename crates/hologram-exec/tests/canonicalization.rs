//! **Algebraic κ-label canonicalization — runtime reuse V&V.**
//!
//! A commutative op's value is independent of operand order, so the executor
//! sorts operand labels to a canonical order before deriving the content
//! address. `a+b` and `b+a` then collapse to one κ-label: the second is
//! recognized as already-resident and its compute is elided — UOR's algebra
//! turned into runtime reuse, not just a compile-time graph property.

use hologram_backend::CpuBackend;
use hologram_compiler::{compile_from_source, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use prism::vocabulary::WittLevel;

fn le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn unle(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn commutative_reordering_dedups_at_runtime() {
    // c = a+b ; d = b+a (reversed operands) ; e = c+d.
    // Canonicalization makes c and d the same κ-label, so d is elided.
    let src = "\
input a :2
input b :2
op add a b :2 as=c
op add b a :2 as=d
op add c d :2 as=e
output e
";
    let compiled = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 4.0];
    let out = unle(
        &sess
            .execute(&[
                InputBuffer { bytes: &le(&a) },
                InputBuffer { bytes: &le(&b) },
            ])
            .unwrap()[0]
            .bytes,
    );
    // c = d = [4,6]; e = [8,12] — correctness preserved.
    assert_eq!(out, vec![8.0, 12.0]);
    // d = b+a was recognized as identical to c = a+b and elided (not
    // dispatched). Without canonicalization the reversed operands would hash
    // differently and d would recompute.
    assert!(
        sess.last_skipped() >= 1,
        "commutative reorder should elide a redundant op (skipped={})",
        sess.last_skipped()
    );
}

#[test]
fn noncommutative_order_is_preserved() {
    // Sub is NOT commutative: a−b and b−a must stay distinct (both compute).
    let src = "\
input a :2
input b :2
op sub a b :2 as=c
op sub b a :2 as=d
op add c d :2 as=e
output e
";
    let compiled = compile_from_source(src, WittLevel::W32, BackendKind::Cpu).unwrap();
    let mut sess: InferenceSession<CpuBackend<BufferArena>> =
        InferenceSession::load(&compiled.archive, CpuBackend::new()).unwrap();
    let a = [5.0f32, 9.0];
    let b = [1.0f32, 2.0];
    let out = unle(
        &sess
            .execute(&[
                InputBuffer { bytes: &le(&a) },
                InputBuffer { bytes: &le(&b) },
            ])
            .unwrap()[0]
            .bytes,
    );
    // c = a−b = [4,7]; d = b−a = [−4,−7]; e = c+d = [0,0].
    assert_eq!(out, vec![0.0, 0.0]);
}
