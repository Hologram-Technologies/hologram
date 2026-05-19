//! Layout ops (spec V.3): no-compute address relabels.
//!
//! Per V.3, layout ops emit a single `Term::Variable` referencing a
//! remapped binding produced by the compiler's address resolver — no
//! `Application` nodes are emitted. The Validated certificate confirms
//! the bijection (the compiler exempts these from `run_completeness`'s
//! algebraic-content checks via `OpKind::is_layout_only`).

use crate::emit::{push_variable, EmitResult};
use core::marker::PhantomData;
use prism::vocabulary::WittLevel;
use uor_foundation::enforcement::TermArena;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;

/// Free emitter for any layout op. Pushes a single `Term::Variable` that
/// references the relabel binding.
pub fn emit_layout_relabel<const CAP: usize>(
    arena: &mut TermArena<CAP>,
    _level: WittLevel,
    remapped_var_index: u32,
) -> EmitResult {
    push_variable(arena, remapped_var_index)
}

macro_rules! declare_layout {
    ($name:ident, $iri_suffix:literal, [$($g:ident),*]) => {
        pub struct $name<$($g,)* D, B>(PhantomData<($($g,)* D, B)>)
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds;

        impl<$($g,)* D, B> Default for $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        { fn default() -> Self { Self(PhantomData) } }

        impl<$($g,)* D, B> $name<$($g,)* D, B>
        where $($g: ConstrainedTypeShape,)* D: ConstrainedTypeShape, B: HostBounds,
        {
            pub const IRI: &'static str = concat!(
                "https://hologram.uor.foundation/op/layout/",
                $iri_suffix,
            );
            pub const CAP: usize = 2;

            pub fn emit_term<const CAP: usize>(
                arena: &mut TermArena<CAP>,
                level: WittLevel,
                remapped_var_index: u32,
            ) -> EmitResult {
                emit_layout_relabel(arena, level, remapped_var_index)
            }
        }
    };
}

declare_layout!(ReshapeOp, "reshape", [Sin, Sout]);
declare_layout!(TransposeOp, "transpose", [S, Perm]);
declare_layout!(ConcatOp, "concat", [Axis, Inputs]);
declare_layout!(SliceOp, "slice", [Sin, Starts, Ends]);
