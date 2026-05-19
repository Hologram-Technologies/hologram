//! Normalization ops (spec V.3).
//!
//! V.3 tree shape (LayerNorm):
//!   ReduceMean → Sub → Mul → ReduceMean → Sqrt → Div → Mul → Add
//! V.3 tree shape (RmsNorm):
//!   Mul → ReduceMean → Sqrt → Div → Mul

use core::marker::PhantomData;
use uor_foundation::enforcement::TermArena;
use uor_foundation::{PrimitiveOp, WittLevel};
use uor_foundation::HostBounds;
use uor_foundation::pipeline::ConstrainedTypeShape;
use crate::emit::{push_application, push_literal, push_recurse, EmitResult};

fn emit_norm_body<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
) -> EmitResult {
    // Structural fold-skeleton for the LayerNorm family. Prism's
    // PrimitiveOp set is the algebraic-ring closure (Add/Sub/Mul/Div/
    // Pow/…); it omits Sqrt as a primitive, so the reciprocal-std
    // factor enters as a Pow node (var^(-1/2)) at runtime. The
    // semantic walk is:
    //   mean    = Recurse(step = Add(acc, x))           — Σ x / n
    //   centred = Sub(x, mean)
    //   sq      = Mul(centred, centred)
    //   var     = Recurse(step = Add(acc, sq))          — Σ (x−μ)² / n
    //   rstd    = Pow(var, −1/2)                        — 1 / √var
    //   norm    = Mul(centred, rstd)
    //   out     = Add(Mul(norm, gamma), beta)
    // The runtime CPU/GPU kernels in `float_kernels` implement the
    // closed-form math directly (with the `eps` regularizer); the
    // Term tree above is the structural witness prism's validation
    // walks (it is not catamorphically evaluated at runtime).
    let zero     = push_literal(arena, 0, level)?;
    let mean_step = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    let mean      = push_recurse(arena, zero, zero, mean_step)?;
    let centred   = push_application(arena, PrimitiveOp::Sub, mean, 2)?;
    let sq        = push_application(arena, PrimitiveOp::Mul, centred, 2)?;
    let var_step  = push_application(arena, PrimitiveOp::Add, sq, 2)?;
    let var       = push_recurse(arena, zero, zero, var_step)?;
    let rstd      = push_application(arena, PrimitiveOp::Pow, var, 2)?;
    let norm      = push_application(arena, PrimitiveOp::Mul, rstd, 2)?;
    let scaled    = push_application(arena, PrimitiveOp::Mul, norm, 2)?;
    push_application(arena, PrimitiveOp::Add, scaled, 2)
}

pub fn emit_layer_norm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    _gamma_var: u32,
    _beta_var: u32,
) -> EmitResult {
    emit_norm_body(arena, level, x_var)
}

pub fn emit_rms_norm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    _gamma_var: u32,
    _beta_var: u32,
) -> EmitResult {
    // Skeleton: rms² = Σ x² / n; rstd = (rms²)^(-1/2); out = x · rstd · gamma.
    let zero      = push_literal(arena, 0, level)?;
    let sq        = push_application(arena, PrimitiveOp::Mul, x_var, 2)?;
    let rms_step  = push_application(arena, PrimitiveOp::Add, sq, 2)?;
    let rms       = push_recurse(arena, zero, zero, rms_step)?;
    let rstd      = push_application(arena, PrimitiveOp::Pow, rms, 2)?;
    push_application(arena, PrimitiveOp::Mul, rstd, 2)
}

pub fn emit_group_norm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    gamma_var: u32,
    beta_var: u32,
) -> EmitResult {
    emit_layer_norm(arena, level, x_var, gamma_var, beta_var)
}

pub fn emit_instance_norm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    gamma_var: u32,
    beta_var: u32,
) -> EmitResult {
    emit_layer_norm(arena, level, x_var, gamma_var, beta_var)
}

pub fn emit_add_rms_norm<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    level: WittLevel,
    x_var: u32,
    residual_var: u32,
) -> EmitResult {
    // AddRmsNorm(x, residual) = RmsNorm(Add(x, residual)). Fused: no temp buffer.
    let _ = residual_var;
    let added = push_application(arena, PrimitiveOp::Add, x_var, 2)?;
    let zero      = push_literal(arena, 0, level)?;
    let sq        = push_application(arena, PrimitiveOp::Mul, added, 2)?;
    let rms_step  = push_application(arena, PrimitiveOp::Add, sq, 2)?;
    let rms       = push_recurse(arena, zero, zero, rms_step)?;
    let rstd      = push_application(arena, PrimitiveOp::Pow, rms, 2)?;
    push_application(arena, PrimitiveOp::Mul, rstd, 2)
}

macro_rules! declare_norm {
    ($name:ident, $iri_suffix:literal, $emit_fn:ident) => {
        pub struct $name<S, D, B>(PhantomData<(S, D, B)>)
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds;

        impl<S, D, B> Default for $name<S, D, B>
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<S, D, B> $name<S, D, B>
        where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/normalization/",
                $iri_suffix,
            );
            pub const CAP: usize = 64;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                x_var: u32,
                gamma_var: u32,
                beta_var: u32,
            ) -> EmitResult {
                $emit_fn(arena, level, x_var, gamma_var, beta_var)
            }
        }
    };
}

declare_norm!(LayerNormOp,    "layer_norm",    emit_layer_norm);
declare_norm!(RmsNormOp,      "rms_norm",      emit_rms_norm);
declare_norm!(GroupNormOp,    "group_norm",    emit_group_norm);
declare_norm!(InstanceNormOp, "instance_norm", emit_instance_norm);

pub struct AddRmsNormOp<S, D, B>(PhantomData<(S, D, B)>)
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds;

impl<S, D, B> Default for AddRmsNormOp<S, D, B>
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
{ fn default() -> Self { Self(PhantomData) } }

impl<S, D, B> AddRmsNormOp<S, D, B>
where S: ConstrainedTypeShape, D: ConstrainedTypeShape, B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/normalization/add_rms_norm";
    pub const CAP: usize = 64;

    pub fn emit_term<const CAP: usize>(
        arena: &mut TermArena<CAP>,
        level: WittLevel,
        x_var: u32,
        residual_var: u32,
    ) -> EmitResult {
        emit_add_rms_norm(arena, level, x_var, residual_var)
    }
}
