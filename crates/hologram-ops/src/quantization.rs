//! Quantization ops (spec X-5 / ADR-054 quantization addendum).
//!
//! `DequantizeOp<S, Qd, Td, B>` lifts a packed-integer weight buffer
//! (Qd ∈ {DTypeI8, DTypeI4}) to a dense floating-point tensor at
//! Td (typically DTypeF32 or DTypeBf16) using a per-tensor scale and
//! zero-point. The Term tree expresses the affine relation
//!
//!   y = scale · (q − zero_point)
//!
//! as a single `Application(Mul, [Sub(q, zp), scale])` chain, anchored
//! on `PrimitiveOp::Mul` and `PrimitiveOp::Sub`.
//!
//! Trillion-parameter models stage their weights as `Weight<DTypeI4, B>`
//! or `Weight<DTypeI8, B>`. `DequantizeOp` is the marker that lets the
//! compiler insert the dequant kernel before the matmul that consumes
//! the weight. The runtime `Dequantize → MatMul` fusion (`MatMulDequant`)
//! elides the intermediate dense tensor — a kernel optimization, not an
//! architectural change: the formal spec stays the affine chain.

use crate::emit::HoloArena;
use core::marker::PhantomData;
use uor_foundation::pipeline::ConstrainedTypeShape;
use uor_foundation::HostBounds;
use uor_foundation::{PrimitiveOp, WittLevel};

use crate::emit::{push_application, EmitResult};

/// Free emitter for Dequantize: `Mul(Sub(q, zp), scale)`.
pub fn emit_dequantize<const CAP: usize>(
    arena: &mut HoloArena<CAP>,
    _level: WittLevel,
    q_var: u32,
) -> EmitResult {
    // Sub anchor: q − zero_point. Mul anchor: (q − zp) · scale.
    let centered = push_application(arena, PrimitiveOp::Sub, q_var, 2)?;
    push_application(arena, PrimitiveOp::Mul, centered, 2)
}

/// Marker type. `S` is the tensor shape, `Qd` is the quantized dtype
/// (DTypeI8 / DTypeI4), `Td` is the dequantized dtype (DTypeF32 /
/// DTypeBf16), `B` is the host bounds.
pub struct DequantizeOp<S, Qd, Td, B>(PhantomData<(S, Qd, Td, B)>)
where
    S: ConstrainedTypeShape,
    Qd: ConstrainedTypeShape,
    Td: ConstrainedTypeShape,
    B: HostBounds;

impl<S, Qd, Td, B> Default for DequantizeOp<S, Qd, Td, B>
where
    S: ConstrainedTypeShape,
    Qd: ConstrainedTypeShape,
    Td: ConstrainedTypeShape,
    B: HostBounds,
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S, Qd, Td, B> DequantizeOp<S, Qd, Td, B>
where
    S: ConstrainedTypeShape,
    Qd: ConstrainedTypeShape,
    Td: ConstrainedTypeShape,
    B: HostBounds,
{
    pub const IRI: &'static str = "https://hologram.uor.foundation/op/quantization/dequantize";
    pub const CAP: usize = 8;
    pub const PRIMARY_OP: PrimitiveOp = PrimitiveOp::Mul;
    pub const ARITY: u8 = 1;

    pub fn emit_term<const CAP: usize>(
        arena: &mut HoloArena<CAP>,
        level: WittLevel,
        q_var: u32,
    ) -> EmitResult {
        emit_dequantize(arena, level, q_var)
    }
}
