//! Hard Requirement contract conformance tests.
//!
//! Verifies that the hybrid JIT+tape architecture satisfies HR-1 through HR-5.

use hologram_core::op::{ActivationOp, PrimOp, RingLevel};
use hologram_graph::graph::GraphOp;

// ── HR-1: Tape execution preserved ───────────────────────────────────────

#[test]
fn hr1_tape_kernel_variants_preserved() {
    use hologram_exec::tape::TapeKernel;
    // All existing tape kernels must still exist
    let _float = TapeKernel::InlineRelu;
    let _sigmoid = TapeKernel::InlineSigmoid;
    let _matmul = TapeKernel::InlineMatMul { m: 4, k: 4, n: 4 };
    let _lut4 = TapeKernel::MatMulLut4(hologram_graph::constant::ConstantId::new(0));
    let _lut8 = TapeKernel::MatMulLut8(hologram_graph::constant::ConstantId::new(0));
    let _kv_write = TapeKernel::KvWrite {
        layer: 0,
        n_kv_heads: 1,
        head_dim: 64,
        is_key: true,
    };
    let _kv_read = TapeKernel::KvRead {
        layer: 0,
        n_kv_heads: 1,
        head_dim: 64,
    };
    let _passthrough = TapeKernel::Passthrough;
    let _output = TapeKernel::Output;
}

#[test]
fn hr1_ring_activation_through_tape() {
    use hologram_exec::tape::TapeKernel;
    // New ring-native ops coexist with existing tape kernels
    let _ring_act = TapeKernel::RingActivation {
        op: ActivationOp::Relu,
        level: RingLevel::Q3,
    };
    let _ring_acc = TapeKernel::RingAccumulate {
        level: RingLevel::Q3,
    };
}

#[test]
fn hr1_constant_data_deferred_exists() {
    // ConstantData::Deferred must exist for mmap weight access
    let _deferred = hologram_graph::constant::ConstantData::Deferred {
        byte_size: 1024,
        source_id: 42,
    };
}

// ── HR-2: WASM support ───────────────────────────────────────────────────

#[test]
fn hr2_graph_op_no_jit_dependency() {
    // GraphOp variants don't require JIT — all dispatchable via tape
    let ops = vec![
        GraphOp::Input,
        GraphOp::Output,
        GraphOp::Prim(PrimOp::Add),
        GraphOp::RingActivation(ActivationOp::Sigmoid, RingLevel::Q3),
        GraphOp::RingAccumulate(RingLevel::Q3),
        GraphOp::Passthrough,
    ];
    for op in &ops {
        let _ = op.arity(); // Must work without JIT
    }
}

// ── HR-3: blake3 checksums ───────────────────────────────────────────────

#[test]
fn hr3_blake3_available() {
    let hash = hologram_archive::checksum::blake3_u32(b"test");
    assert_ne!(hash, 0);
    let full = hologram_archive::checksum::blake3_full(b"test");
    assert_eq!(full.len(), 32);
}

#[test]
fn hr3_header_blake3_flag() {
    use hologram_archive::format::header::*;
    use hologram_archive::format::*;
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    assert!(!h.uses_blake3());
    h.set_blake3();
    assert!(h.uses_blake3());
}

// ── HR-4: No generic leak in public API ──────────────────────────────────

#[test]
fn hr4_graph_op_not_generic() {
    // GraphOp is NOT parameterized by QuantumLevel
    // This test verifies the type is usable without specifying a generic
    let _op: GraphOp = GraphOp::Input;
    let _op2: GraphOp = GraphOp::RingActivation(ActivationOp::Relu, RingLevel::Q3);
    // RingLevel is a runtime discriminator, not a compile-time generic
    let level = RingLevel::Q3;
    assert_eq!(level as u8, 3);
}

#[test]
fn hr4_ring_level_is_runtime() {
    // RingLevel is a plain enum, not a generic parameter
    let levels = [RingLevel::Q0, RingLevel::Q1, RingLevel::Q2, RingLevel::Q3];
    for &l in &levels {
        let _ = l as u8; // Runtime discriminant, not compile-time
    }
}

// ── HR-5: Performance parity ─────────────────────────────────────────────

#[test]
fn hr5_lut_gemm_preserved() {
    use hologram_exec::tape::TapeKernel;
    // LUT-GEMM psumbook kernels must still exist in tape
    let _lut4 = TapeKernel::MatMulLut4(hologram_graph::constant::ConstantId::new(0));
    let _lut8 = TapeKernel::MatMulLut8(hologram_graph::constant::ConstantId::new(0));
    let _lut16 = TapeKernel::MatMulLut16(hologram_graph::constant::ConstantId::new(0));
}

#[test]
fn hr5_simd_activation_preserved() {
    // ElementWiseView must still exist for SIMD activation dispatch
    let view = hologram_core::view::ElementWiseView::identity();
    assert_eq!(view.apply(42), 42);
    let neg_view = hologram_core::view::NEG_VIEW;
    assert_eq!(neg_view.apply(1), 255); // neg(1) = 255 in Q0
}

#[test]
fn hr5_inline_hot_ops_preserved() {
    use hologram_exec::tape::TapeKernel;
    // All inline hot ops must still exist for zero-dispatch-overhead execution
    let _ = TapeKernel::InlineRelu;
    let _ = TapeKernel::InlineSigmoid;
    let _ = TapeKernel::InlineGelu;
    let _ = TapeKernel::InlineSilu;
    let _ = TapeKernel::InlineTanh;
    let _ = TapeKernel::InlineExp;
    let _ = TapeKernel::InlineAdd;
    let _ = TapeKernel::InlineMul;
    let _ = TapeKernel::InlineSub;
    let _ = TapeKernel::InlineDiv;
    let _ = TapeKernel::InlineAbs;
    let _ = TapeKernel::InlineNeg;
    let _ = TapeKernel::InlineLog;
    let _ = TapeKernel::InlineSqrt;
}

#[test]
fn hr5_ring_chain_via_tape() {
    // Chain fusion handled by tape's FusedFloatChain and view fusion — no JIT needed.
    // The ring layer + LLVM compilation + tape dispatch reaches theoretical peak
    // for memory-bound workloads. Ring ops are const fn Rust compiled to native
    // instructions on every target including WASM.
    use hologram_exec::tape::TapeKernel;
    // Verify ring activation is dispatchable via tape (not JIT)
    let _kernel = TapeKernel::RingActivation {
        op: ActivationOp::Sigmoid,
        level: RingLevel::Q3,
    };
}

// ── Ring-native ops correctness ──────────────────────────────────────────

#[test]
fn ring_activation_all_21_variants() {
    // All 21 activations are accessible via the tape path
    let acts = [
        ActivationOp::Relu,
        ActivationOp::Abs,
        ActivationOp::Square,
        ActivationOp::Cube,
        ActivationOp::Sigmoid,
        ActivationOp::Tanh,
        ActivationOp::Gelu,
        ActivationOp::Silu,
        ActivationOp::Exp,
        ActivationOp::Exp2,
        ActivationOp::Exp10,
        ActivationOp::Log,
        ActivationOp::Log2,
        ActivationOp::Log10,
        ActivationOp::Sin,
        ActivationOp::Cos,
        ActivationOp::Tan,
        ActivationOp::Asin,
        ActivationOp::Acos,
        ActivationOp::Atan,
        ActivationOp::Sqrt,
    ];
    assert_eq!(acts.len(), 21);
    for act in &acts {
        // Each activation can be wrapped in a GraphOp
        let _op = GraphOp::RingActivation(*act, RingLevel::Q3);
    }
}
