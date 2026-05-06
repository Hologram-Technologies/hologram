//! Spec XII.3 / I-1: every op marker's emit_term restricts `Term::Application`'s
//! `operator` to the closed 10 PrimitiveOp set.
//!
//! This is statically guaranteed by the Term type signature
//! (Term::Application::operator: PrimitiveOp), so the test below is a
//! sanity check confirming op markers are reachable at compile time.

use hologram_ops::direct::*;
use hologram_ops::elementwise_unary::*;
use hologram_ops::elementwise_binary::*;
use hologram_ops::reduction::*;
use uor_foundation::PrimitiveOp;

#[test]
fn direct_ops_carry_their_primitive() {
    assert_eq!(NegOp::PRIMITIVE,  PrimitiveOp::Neg);
    assert_eq!(BnotOp::PRIMITIVE, PrimitiveOp::Bnot);
    assert_eq!(SuccOp::PRIMITIVE, PrimitiveOp::Succ);
    assert_eq!(PredOp::PRIMITIVE, PrimitiveOp::Pred);
    assert_eq!(AddOp::PRIMITIVE,  PrimitiveOp::Add);
    assert_eq!(SubOp::PRIMITIVE,  PrimitiveOp::Sub);
    assert_eq!(MulOp::PRIMITIVE,  PrimitiveOp::Mul);
    assert_eq!(XorOp::PRIMITIVE,  PrimitiveOp::Xor);
    assert_eq!(AndOp::PRIMITIVE,  PrimitiveOp::And);
    assert_eq!(OrOp::PRIMITIVE,   PrimitiveOp::Or);
}

#[test]
fn unary_iris_under_namespace() {
    assert!(ReluOp::IRI.starts_with("https://hologram.uor.foundation/op/unary/"));
    assert!(SigmoidOp::IRI.starts_with("https://hologram.uor.foundation/op/unary/"));
    assert!(ExpOp::IRI.starts_with("https://hologram.uor.foundation/op/unary/"));
}

#[test]
fn binary_iris_under_namespace() {
    assert!(DivOp::IRI.starts_with("https://hologram.uor.foundation/op/binary/"));
    assert!(MinOp::IRI.starts_with("https://hologram.uor.foundation/op/binary/"));
    assert!(EqualOp::IRI.starts_with("https://hologram.uor.foundation/op/binary/"));
}

#[test]
fn reduction_iris_under_namespace() {
    use uor_foundation::pipeline::ConstrainedTypeShape;
    use uor_foundation::HostBounds;
    use hologram_host::HologramHostBoundsCpu;
    use hologram_types::{Dim, Shape1, DTypeF32};

    type S = Shape1<Dim<128>, 1>;
    type Axes = Shape1<Dim<0>, 1>;
    type Op = ReduceSumOp<S, Axes, DTypeF32, HologramHostBoundsCpu>;
    assert!(Op::IRI.starts_with("https://hologram.uor.foundation/op/reduction/"));
    assert_eq!(Op::STEP_OP, PrimitiveOp::Add);
    let _ = HologramHostBoundsCpu::WITT_LEVEL_MAX_BITS;
    let _ = <S as ConstrainedTypeShape>::IRI;
}
