//! Spec XII.3: every op marker exposes a hologram-namespaced IRI.
//!
//! Statically enumerates every marker in the catalog and asserts the IRI
//! prefix. This anchors I-1 (no new primitives outside the closed
//! `PrimitiveOp` set — the only operator type `Term::Application::operator`
//! accepts) plus the IRI scheme from spec IV.2.

use hologram_ops::{
    direct::*, elementwise_unary::*, elementwise_binary::*,
    linalg::*, conv::*, normalization::*, reduction::*,
    layout::*, activation_reduce::*, pooling::*, structured::*,
    utility::*, backward::*,
};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use hologram_host::HologramHostBoundsCpu;
use hologram_types::{Dim, Shape1, Shape2, DTypeF32};

const PREFIX: &str = "https://hologram.uor.foundation/op/";

#[test]
fn direct_iris() {
    assert!(NegOp::IRI.starts_with(PREFIX));
    assert!(BnotOp::IRI.starts_with(PREFIX));
    assert!(SuccOp::IRI.starts_with(PREFIX));
    assert!(PredOp::IRI.starts_with(PREFIX));
    assert!(AddOp::IRI.starts_with(PREFIX));
    assert!(SubOp::IRI.starts_with(PREFIX));
    assert!(MulOp::IRI.starts_with(PREFIX));
    assert!(XorOp::IRI.starts_with(PREFIX));
    assert!(AndOp::IRI.starts_with(PREFIX));
    assert!(OrOp::IRI.starts_with(PREFIX));
}

#[test]
fn unary_iris() {
    assert!(ReluOp::IRI.starts_with(PREFIX));
    assert!(SigmoidOp::IRI.starts_with(PREFIX));
    assert!(TanhOp::IRI.starts_with(PREFIX));
    assert!(GeluOp::IRI.starts_with(PREFIX));
    assert!(SiluOp::IRI.starts_with(PREFIX));
    assert!(EluOp::IRI.starts_with(PREFIX));
    assert!(SeluOp::IRI.starts_with(PREFIX));
    assert!(ExpOp::IRI.starts_with(PREFIX));
    assert!(LogOp::IRI.starts_with(PREFIX));
    assert!(Log1pOp::IRI.starts_with(PREFIX));
    assert!(SqrtOp::IRI.starts_with(PREFIX));
    assert!(ReciprocalOp::IRI.starts_with(PREFIX));
    assert!(SinOp::IRI.starts_with(PREFIX));
    assert!(CosOp::IRI.starts_with(PREFIX));
    assert!(TanOp::IRI.starts_with(PREFIX));
    assert!(AsinOp::IRI.starts_with(PREFIX));
    assert!(AcosOp::IRI.starts_with(PREFIX));
    assert!(AtanOp::IRI.starts_with(PREFIX));
    assert!(CeilOp::IRI.starts_with(PREFIX));
    assert!(FloorOp::IRI.starts_with(PREFIX));
    assert!(RoundOp::IRI.starts_with(PREFIX));
    assert!(ErfOp::IRI.starts_with(PREFIX));
    assert!(IsNaNOp::IRI.starts_with(PREFIX));
    assert!(SignOp::IRI.starts_with(PREFIX));
    assert!(AbsOp::IRI.starts_with(PREFIX));
}

#[test]
fn binary_iris() {
    assert!(DivOp::IRI.starts_with(PREFIX));
    assert!(PowOp::IRI.starts_with(PREFIX));
    assert!(ModOp::IRI.starts_with(PREFIX));
    assert!(MinOp::IRI.starts_with(PREFIX));
    assert!(MaxOp::IRI.starts_with(PREFIX));
    assert!(EqualOp::IRI.starts_with(PREFIX));
    assert!(LessOp::IRI.starts_with(PREFIX));
    assert!(LessOrEqualOp::IRI.starts_with(PREFIX));
    assert!(GreaterOp::IRI.starts_with(PREFIX));
    assert!(GreaterOrEqualOp::IRI.starts_with(PREFIX));
}

#[test]
fn structured_iris() {
    type S = Shape2<Dim<128>, Dim<128>, 2>;
    type Op1 = MatMulOp<128, 128, 128, DTypeF32, HologramHostBoundsCpu>;
    assert!(Op1::IRI.starts_with(PREFIX));
    let _ = HologramHostBoundsCpu::WITT_LEVEL_MAX_BITS;
    let _ = <S as ConstrainedTypeShape>::IRI;

    type Op2 = GemmOp<128, 128, 128, DTypeF32, HologramHostBoundsCpu>;
    assert!(Op2::IRI.starts_with(PREFIX));

    type Cv = Conv2dOp<S, S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Cv::IRI.starts_with(PREFIX));

    type CvT = ConvTranspose2dOp<S, S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(CvT::IRI.starts_with(PREFIX));

    type LN = LayerNormOp<S, DTypeF32, HologramHostBoundsCpu>;
    assert!(LN::IRI.starts_with(PREFIX));

    type RN = RmsNormOp<S, DTypeF32, HologramHostBoundsCpu>;
    assert!(RN::IRI.starts_with(PREFIX));

    type Axes = Shape1<Dim<0>, 1>;
    type RS = ReduceSumOp<S, Axes, DTypeF32, HologramHostBoundsCpu>;
    assert!(RS::IRI.starts_with(PREFIX));

    type Sm = SoftmaxOp<S, Axes, DTypeF32, HologramHostBoundsCpu>;
    assert!(Sm::IRI.starts_with(PREFIX));

    type Mp = MaxPool2dOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Mp::IRI.starts_with(PREFIX));

    type Att = AttentionOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Att::IRI.starts_with(PREFIX));

    type Sg = FusedSwiGluOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Sg::IRI.starts_with(PREFIX));

    type Pad = PadOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Pad::IRI.starts_with(PREFIX));

    type Cl = ClipOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Cl::IRI.starts_with(PREFIX));

    type Wh = WhereOp<S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Wh::IRI.starts_with(PREFIX));
}

#[test]
fn layout_iris() {
    type S = Shape2<Dim<128>, Dim<128>, 2>;
    type Re = ReshapeOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Re::IRI.starts_with(PREFIX));

    type Tr = TransposeOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Tr::IRI.starts_with(PREFIX));

    type Co = ConcatOp<S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Co::IRI.starts_with(PREFIX));

    type Sl = SliceOp<S, S, S, DTypeF32, HologramHostBoundsCpu>;
    assert!(Sl::IRI.starts_with(PREFIX));
}

#[test]
fn backward_iris() {
    assert!(MatMulGradAOp::IRI.starts_with(PREFIX));
    assert!(MatMulGradBOp::IRI.starts_with(PREFIX));
    assert!(Conv2dGradXOp::IRI.starts_with(PREFIX));
    assert!(Conv2dGradWOp::IRI.starts_with(PREFIX));
    assert!(SoftmaxGradOp::IRI.starts_with(PREFIX));
    assert!(LogSoftmaxGradOp::IRI.starts_with(PREFIX));
    assert!(LayerNormGradOp::IRI.starts_with(PREFIX));
    assert!(RmsNormGradOp::IRI.starts_with(PREFIX));
    assert!(GroupNormGradOp::IRI.starts_with(PREFIX));
    assert!(ReduceSumGradOp::IRI.starts_with(PREFIX));
    assert!(ReduceMeanGradOp::IRI.starts_with(PREFIX));
    assert!(ReduceProdGradOp::IRI.starts_with(PREFIX));
    assert!(SubGradOp::IRI.starts_with(PREFIX));
    assert!(MulGradOp::IRI.starts_with(PREFIX));
    assert!(DivGradOp::IRI.starts_with(PREFIX));
    assert!(PowGradOp::IRI.starts_with(PREFIX));
    assert!(MinGradOp::IRI.starts_with(PREFIX));
    assert!(MaxGradOp::IRI.starts_with(PREFIX));
    assert!(ConcatGradOp::IRI.starts_with(PREFIX));
    assert!(SliceGradOp::IRI.starts_with(PREFIX));
    assert!(AvgPool2dGradOp::IRI.starts_with(PREFIX));
    assert!(GlobalAvgPoolGradOp::IRI.starts_with(PREFIX));
    assert!(PadGradOp::IRI.starts_with(PREFIX));
    assert!(AttentionGradOp::IRI.starts_with(PREFIX));
    assert!(FusedSwiGluGradOp::IRI.starts_with(PREFIX));
    assert!(UnaryGradOp::IRI.starts_with(PREFIX));
}
