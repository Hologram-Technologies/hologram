//! PM_7 memory-affinity V&V (feature `tiered-exec`).
//!
//! The per-kernel memory tier is a pure function of the datum's quantum level
//! (Witt bit-width), recomputed at session load. These tests check the tier
//! assignment + report are coherent and that execution is byte-identical with
//! the scaffold on (it is a zero-cost observability/routing layer on CPU).
#![cfg(feature = "tiered-exec")]

use hologram_backend::CpuBackend;
use hologram_compiler::{compile, BackendKind};
use hologram_exec::coherence::TierPolicy;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::Node;
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, GraphOp, InputSource, OpKind};
use hologram_types::MemoryTier;
use prism::vocabulary::WittLevel;
use smallvec::SmallVec;

const F32: u8 = 8;

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `relu(A · B)` — A is a graph input, B a constant weight; matmul then relu.
fn build() -> Graph {
    let mut g = Graph::new();
    let a_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 3));
    let b_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank2(3, 2));
    let o_sh = g.shape_registry_mut().intern(ShapeDescriptor::rank2(2, 2));
    let a = g.add_node(Node {
        op: GraphOp::Input,
        inputs: SmallVec::new(),
        output_dtype: DTypeId(F32),
        output_shape: a_sh,
    });
    g.add_input(a);
    let w = g.constants_mut().insert(ConstantEntry {
        bytes: le_f32(&[0.5, -1.0, 2.0, 0.25, -0.5, 1.0]),
        dtype: DTypeId(F32),
        shape: b_sh,
    });
    let mm = g.add_node(Node {
        op: GraphOp::Op(OpKind::MatMul),
        inputs: SmallVec::from_iter([InputSource::Node(a), InputSource::Constant(w)]),
        output_dtype: DTypeId(F32),
        output_shape: o_sh,
    });
    let relu = g.add_node(Node {
        op: GraphOp::Op(OpKind::Relu),
        inputs: SmallVec::from_iter([InputSource::Node(mm)]),
        output_dtype: DTypeId(F32),
        output_shape: o_sh,
    });
    let out = g.add_node(Node {
        op: GraphOp::Output,
        inputs: SmallVec::from_iter([InputSource::Node(relu)]),
        output_dtype: DTypeId(F32),
        output_shape: o_sh,
    });
    g.add_output(out);
    g
}

fn session() -> InferenceSession<CpuBackend<BufferArena>> {
    let out = compile(build(), BackendKind::Cpu, WittLevel::W32).unwrap();
    InferenceSession::load(&out.archive, CpuBackend::new()).unwrap()
}

#[test]
fn tier_assignment_and_report_are_coherent() {
    let s = session();
    let tiers = s.tiers();
    assert!(!tiers.is_empty(), "every kernel gets a tier");
    let r = s.tier_report();
    // The histogram partitions the kernels exactly.
    assert_eq!(
        (r.cpu_l1_calls + r.cpu_l2_calls + r.cpu_main_calls + r.device_calls) as usize,
        tiers.len(),
        "tier histogram must sum to the kernel count"
    );
    // f32 (Witt-32) compute kernels are Q3+; small element counts stay CpuMain.
    assert!(
        tiers
            .iter()
            .all(|t| matches!(t, MemoryTier::CpuMain | MemoryTier::Device)),
        "f32 compute kernels are Q3+ tier; got {tiers:?}"
    );
}

#[test]
fn execution_is_correct_with_tiering_on() {
    let mut s = session();
    let a = le_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [2,3]
    let out = s.execute(&[InputBuffer { bytes: &a }]).unwrap();
    // A·B = [[1·0.5+2·2.0+3·(-0.5), 1·(-1.0)+2·0.25+3·1.0],
    //        [4·0.5+5·2.0+6·(-0.5), 4·(-1.0)+5·0.25+6·1.0]]
    //     = [[3.0, 2.5], [9.0, 3.25]] ; relu = same (all ≥ 0).
    assert_eq!(f32s(&out[0].bytes), vec![3.0, 2.5, 9.0, 3.25]);
}

#[test]
fn force_all_cpu_policy_keeps_every_tier_on_cpu() {
    let mut s = session();
    s.set_tier_policy(TierPolicy::ForceAllCpu);
    assert_eq!(s.tier_policy(), TierPolicy::ForceAllCpu);
    for &t in s.tiers() {
        assert!(
            TierPolicy::ForceAllCpu.apply(t).is_cpu(),
            "ForceAllCpu routes every tier to CPU"
        );
    }
}
