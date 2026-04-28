//! The `Op` trait and the small types it depends on.
//!
//! Per-op marker structs (`Add`, `MatMul`, …) and their `Op` impls live
//! alongside their kernels in [`crate::kernels`]; this file owns only
//! the trait contract itself and its support types.

/// Broad semantic category of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Elementwise tensor transform.
    Elementwise,
    /// Linear algebra kernel such as matmul.
    LinearAlgebra,
    /// Normalisation (layer / rms / instance / group).
    Normalisation,
    /// Softmax / log-softmax style reduction.
    Reduction,
    /// Layout-only metadata transform (no value change).
    Layout,
    /// Convolution.
    Convolution,
    /// Tensor-shape rewrite (slice / concat / reshape with values).
    Shape,
    /// Pre-fused multi-op composite.
    Fused,
}

/// Planner-visible semantic metadata for an op kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpSignature {
    /// Number of consumed inputs.
    pub arity: usize,
    /// Number of produced outputs.
    pub outputs: usize,
    /// Broad semantic category.
    pub category: OpCategory,
    /// Whether the op participates in backward planning.
    pub differentiable: bool,
    /// Whether the op changes only metadata/layout, not values.
    pub layout_only: bool,
}

/// Per-op-type semantic contract.
///
/// Every canonical operation implements `Op` so all facts about it
/// (arity, name, signature, default backward rule, semantic category)
/// live with the op type. The [`crate::SemanticOp`] enum forwards to
/// these methods, keeping the closed dispatch surface that downstream
/// crates pattern-match on while the per-op declarations stay local.
///
/// See [ADR-044](../../specs/adrs/044-op-trait-canonical-semantics.md).
pub trait Op: Copy {
    /// Number of inputs this op consumes.
    fn arity(self) -> u8;

    /// Number of outputs this op produces. Defaults to one.
    #[inline]
    fn n_outputs(self) -> u8 {
        1
    }

    /// Stable machine-readable name.
    fn name(self) -> &'static str;

    /// Broad semantic category.
    fn category(self) -> OpCategory;

    /// Whether the op changes only metadata/layout, not values.
    #[inline]
    fn layout_only(self) -> bool {
        matches!(self.category(), OpCategory::Layout)
    }

    /// Whether the op participates in backward planning.
    #[inline]
    fn differentiable(self) -> bool {
        self.backward().is_some()
    }

    /// Default backward rule, if any. Non-differentiable ops return `None`.
    #[inline]
    fn backward(self) -> Option<BackwardRule> {
        None
    }

    /// Planner-visible semantic signature, derived from the other methods.
    #[inline]
    fn signature(self) -> OpSignature {
        OpSignature {
            arity: self.arity() as usize,
            outputs: self.n_outputs() as usize,
            category: self.category(),
            differentiable: self.differentiable(),
            layout_only: self.layout_only(),
        }
    }
}

/// Canonical backward rule for a differentiable op.
///
/// The transform planner consumes a `BackwardRule` and lowers it into
/// concrete executable plan nodes such as `KernelCall::MatMulGradA`.
/// Callers obtain the rule via `<Op>::backward()` (the trait method) or
/// `SemanticOp::backward()`, both of which know the originating op —
/// keeping rule and op in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackwardRule {
    /// `dA += dC`, `dB += dC`.
    AddBackward,
    /// `dA += dC`, `dB += -dC`.
    SubBackward,
    /// `dA += dC * B`, `dB += dC * A` — needs forward `A` and `B`.
    MulBackward,
    /// `dA += dC / B`, `dB += -dC * A / B²` — needs forward `A` and `B`.
    DivBackward,
    /// `dA += -dC`.
    NegBackward,
    /// `dA += dC * (A > 0)` — needs forward `A`.
    ReluBackward,
    /// `dA += dC * out * (1 - out)` — needs forward output.
    SigmoidBackward,
    /// `dA += dC * (1 - out²)` — needs forward output.
    TanhBackward,
    /// `dA += dC * out` — needs forward output.
    ExpBackward,
    /// `dA += dC / A` — needs forward `A`.
    LogBackward,
    /// `dA += dC / (2 * sqrt(A))` — uses forward output.
    SqrtBackward,
    /// `dA += dC * sign(A)` — needs forward `A`.
    AbsBackward,
    /// `dA += -dC * out²` — uses forward output.
    ReciprocalBackward,
    /// `dA += dC * (A <= B)`, `dB += dC * (A > B)` — needs forward `A` & `B`.
    MinBackward,
    /// `dA += dC * (A >= B)`, `dB += dC * (A < B)` — needs forward `A` & `B`.
    MaxBackward,
    /// Broadcast `dC` (one element per row) back across the reduced
    /// axis: `dA[r, j] += dC[r]`.
    ReduceSumBackward,
    /// Like `ReduceSumBackward` but divides by `size`:
    /// `dA[r, j] += dC[r] / size`.
    ReduceMeanBackward,
    /// `dA += dC @ Bᵀ`, `dB += Aᵀ @ dC`.
    MatMulBackward,
    /// Split `dC` along the last axis: first `size_a` cols → `dA`,
    /// next `size_b` cols → `dB`.
    ConcatBackward,
    /// Scatter `dC` into the slice region of `dA`; outside-slice
    /// positions are unaffected.
    SliceBackward,
    /// Apply the inverse permutation to `dC` to produce `dA`.
    TransposeBackward,
    /// `dA += dC * gelu'(A)` — needs forward `A`.
    GeluBackward,
    /// `dA += dC * silu'(A)` — needs forward `A`.
    SiluBackward,
    /// `dA += dC * B * A^(B-1)`, `dB += dC * out * ln(A)` — needs
    /// forward `A`, `B`, and forward output.
    PowBackward,
    /// `dA[r,j] += out[r,j] * (dC[r,j] - Σ_k dC[r,k] * out[r,k])` —
    /// needs forward output and the reduced-axis size.
    SoftmaxBackward,
    /// `dA[r,j] += dC[r,j] - exp(out[r,j]) * Σ_k dC[r,k]` — needs
    /// forward output and the reduced-axis size.
    LogSoftmaxBackward,
    /// Route `dC` to whichever row entry equals the row max — needs
    /// forward input, forward output, and the reduced-axis size.
    /// Ties: first occurrence wins.
    ReduceMaxBackward,
    /// Route `dC` to whichever row entry equals the row min — needs
    /// forward input, forward output, and the reduced-axis size.
    /// Ties: first occurrence wins.
    ReduceMinBackward,
    /// `dA[r,j] += dC[r] * Π_{k≠j} A[r,k]` — needs forward input,
    /// forward output, and the reduced-axis size. Zero-aware
    /// (reference kernel handles 0-, 1-, ≥2-zero rows separately).
    ReduceProdBackward,
    /// `dx`, `dw` for `RmsNorm`. Needs forward input, weight, and the
    /// row size + epsilon.
    RmsNormBackward,
    /// `dx`, `dw`, `db` for `LayerNorm`. Needs forward input, weight,
    /// and the row size + epsilon.
    LayerNormBackward,
    /// `dx`, `dw` for `InstanceNorm`. Same as LayerNorm minus the bias
    /// path.
    InstanceNormBackward,
    /// `d_residual`, `d_input`, `dw` for `AddRmsNorm`. The residual
    /// and input grads are identical (the forward sums them before
    /// normalising).
    AddRmsNormBackward,
    /// Broadcast `dC / (H·W)` uniformly back across spatial dims.
    GlobalAvgPoolBackward,
    /// Distribute `dC / count` into each kernel window (overlapping
    /// windows accumulate).
    AvgPool2dBackward,
    /// Route `dC` to the argmax position in each kernel window.
    MaxPool2dBackward,
    /// `dx`, `dw`, `db` for `GroupNorm`. Per-group statistics over
    /// `group_elements = input.len / num_groups` entries.
    GroupNormBackward,
    /// `d_gate`, `d_up` for `FusedSwiGlu`. `out = silu(gate) * up`, so
    /// `d_gate = dC * up * silu'(gate)` and `d_up = dC * silu(gate)`.
    FusedSwiGluBackward,
    /// `dx`, `dw`, `db` for `Conv2d`. Direct (no im2col) reference; one
    /// pass over the same `(ni, oc, oh, ow, ic, ky, kx)` index space as
    /// the forward kernel, accumulating each partial.
    Conv2dBackward,
    /// `dx`, `dw`, `db` for `ConvTranspose2d`. Mirrors the forward
    /// scatter in reverse — `dx` is a forward `Conv2d` of `dy` against
    /// the same weight; `dw` is a correlation of `x` with `dy`.
    ConvTranspose2dBackward,
    /// `dQ`, `dK`, `dV` for the canonical scaled-dot-product
    /// `Attention`. Recomputes the attention probabilities row-by-row
    /// from `Q`, `K`, `V` (no saved-mask trick — the kernel rebuilds
    /// them like the forward), then walks the standard softmax-attention
    /// backward formulas.
    AttentionBackward,
}
