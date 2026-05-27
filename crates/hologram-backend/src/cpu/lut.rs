//! LUT-accelerated low-precision activations (PM_7 Q0/Q1 tiers, uor-native).
//!
//! At a low quantum level the input domain is *finite*, so a unary activation
//! is **fully materializable as a lookup table** — the content-addressed,
//! compute-once representation of the function over that domain (the UOR
//! "materialize-and-reuse" principle, applied to a function instead of a
//! tensor). For IEEE `f16`/`bf16` (16-bit, Prism quantum level Q1) every input
//! is one of 65536 patterns, so the table is `[u16; 65536]` (128 KB, L2-
//! resident) mapping each input bit-pattern to the activation's output
//! bit-pattern.
//!
//! The table entry is `narrow(f(widen(bits)))` — **bit-identical** to the
//! compute path (`widen → f32 transcendental → narrow`), so this is a pure
//! speedup (one load + store per element instead of an `exp`/`tanh`/`erf`),
//! not an approximation. Tables are built once per (activation, dtype) and
//! cached for process lifetime; only activations actually used pay the build.

extern crate alloc;
use alloc::boxed::Box;
use std::sync::OnceLock;

use crate::cpu::dtype::{read_bf16, read_f16, write_bf16, write_f16, DTYPE_BF16, DTYPE_F16};
use crate::cpu::float_kernels::{erf_f, exp_f, gelu_f, sigmoid_f, silu_f, tanh_f};
use crate::error::BackendError;
use crate::kernel_call::lut_act;
use crate::kernel_call::UnaryCall;
use crate::workspace::Workspace;

fn act_fn(act: u8) -> fn(f32) -> f32 {
    match act {
        lut_act::SIGMOID => sigmoid_f,
        lut_act::TANH => tanh_f,
        lut_act::GELU => gelu_f,
        lut_act::SILU => silu_f,
        lut_act::EXP => exp_f,
        _ => erf_f,
    }
}

/// `true` if `dtype` is a 16-bit (Q1) float the LUT path serves.
#[inline]
pub fn is_lut_dtype(dtype: u8) -> bool {
    dtype == DTYPE_F16 || dtype == DTYPE_BF16
}

type Table = Box<[u16; 65536]>;

/// Per-(activation, dtype) cached tables. `[act][0]` = f16, `[act][1]` = bf16.
fn tables() -> &'static [[OnceLock<Table>; 2]; lut_act::COUNT] {
    static TABLES: OnceLock<[[OnceLock<Table>; 2]; lut_act::COUNT]> = OnceLock::new();
    TABLES.get_or_init(|| core::array::from_fn(|_| [OnceLock::new(), OnceLock::new()]))
}

fn build_table(act: u8, dtype: u8) -> Table {
    let f = act_fn(act);
    let mut t: Table = alloc::vec![0u16; 65536]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    let mut inb = [0u8; 2];
    let mut outb = [0u8; 2];
    for (bits, slot) in t.iter_mut().enumerate() {
        inb.copy_from_slice(&(bits as u16).to_le_bytes());
        let x = if dtype == DTYPE_F16 {
            read_f16(&inb, 0)
        } else {
            read_bf16(&inb, 0)
        };
        let y = f(x);
        if dtype == DTYPE_F16 {
            write_f16(&mut outb, 0, y);
        } else {
            write_bf16(&mut outb, 0, y);
        }
        *slot = u16::from_le_bytes(outb);
    }
    t
}

fn table(act: u8, dtype: u8) -> &'static [u16; 65536] {
    let di = if dtype == DTYPE_F16 { 0 } else { 1 };
    tables()[act as usize][di].get_or_init(|| build_table(act, dtype))
}

/// LUT-accelerated unary activation for an f16/bf16 buffer: one table lookup
/// per element (no transcendental compute). Bit-identical to `unary_float`.
pub fn unary_lut<W: Workspace>(c: &UnaryCall, ws: &mut W, act: u8) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    if n == 0 {
        return Ok(());
    }
    let dtype = c.dtype;
    debug_assert!(is_lut_dtype(dtype));
    let t = table(act, dtype);
    let (reads, out) = ws
        .split_borrow(&[c.input], c.output)
        .ok_or(BackendError::SlotOutOfRange(c.output.slot))?;
    let inp = reads[0]
        .get(..n * 2)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?;
    if out.len() < n * 2 {
        return Err(BackendError::SlotOutOfRange(c.output.slot));
    }
    for i in 0..n {
        let bits = u16::from_le_bytes([inp[2 * i], inp[2 * i + 1]]) as usize;
        out[2 * i..2 * i + 2].copy_from_slice(&t[bits].to_le_bytes());
    }
    Ok(())
}
