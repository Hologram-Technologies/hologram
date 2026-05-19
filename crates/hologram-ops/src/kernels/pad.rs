//! Canonical `Pad` op (constant / reflect / edge modes) — semantic
//! identity, executable form, and CPU reference kernel.
//!
//! Reference behaviour: 4-D NCHW input with symmetric `pad_h` /
//! `pad_w` padding around the last two axes. Three padding modes
//! (matching ONNX `Pad`):
//!
//! - **0 — constant**: pads filled with `value`.
//! - **1 — reflect**: mirrors interior values across the boundary
//!   (excluding the edge itself).
//! - **2 — edge**: replicates the boundary row/column.
//!
//! Arbitrary-axis padding is deferred — the canonical reference is
//! NCHW spatial padding only.

use crate::attrs::PadAttrs;
use crate::span::SlotSpan;
use crate::trait_def::{Op, OpCategory};

/// Pre-resolved arguments for `pad` (constant mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadCall {
    /// Input span.
    pub input: SlotSpan,
    /// Output span.
    pub output: SlotSpan,
    /// Batch size.
    pub n: u32,
    /// Channel count.
    pub c: u32,
    /// Input height.
    pub h_in: u32,
    /// Input width.
    pub w_in: u32,
    /// Vertical padding (added to top and bottom).
    pub pad_h: u32,
    /// Horizontal padding (added to left and right).
    pub pad_w: u32,
    /// Constant fill value, encoded as `f32::to_bits()`.
    pub value_bits: u32,
    /// Mode: 0 = constant, 1 = reflect, 2 = edge.
    pub mode: u8,
}

/// Marker struct for the canonical `pad` op (constant mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pad(pub PadAttrs);

impl Op for Pad {
    #[inline]
    fn arity(self) -> u8 {
        1
    }
    #[inline]
    fn name(self) -> &'static str {
        "pad"
    }
    #[inline]
    fn category(self) -> OpCategory {
        OpCategory::Shape
    }
}

/// Forward: pad a 4-D NCHW input to
/// `[N, C, h_in + 2*pad_h, w_in + 2*pad_w]` using the configured
/// mode.
pub fn pad(storage: &mut [f32], call: &PadCall) {
    let n = call.n as usize;
    let c = call.c as usize;
    let h_in = call.h_in as usize;
    let w_in = call.w_in as usize;
    let pad_h = call.pad_h as usize;
    let pad_w = call.pad_w as usize;
    let h_out = h_in + 2 * pad_h;
    let w_out = w_in + 2 * pad_w;
    let chw_in = c * h_in * w_in;
    let chw_out = c * h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            let plane_in = call.input.offset + ni * chw_in + ci * h_in * w_in;
            let plane_out = call.output.offset + ni * chw_out + ci * h_out * w_out;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let ih = src_index(oh, pad_h, h_in, call.mode);
                    let iw = src_index(ow, pad_w, w_in, call.mode);
                    let dst = plane_out + oh * w_out + ow;
                    storage[dst] = match (ih, iw) {
                        (Some(y), Some(x)) => storage[plane_in + y * w_in + x],
                        // Out-of-bounds in any dim under constant mode →
                        // use the configured fill value.
                        _ => f32::from_bits(call.value_bits),
                    };
                }
            }
        }
    }
}

/// Map a 1-D output coordinate back to the input coordinate under the
/// chosen pad mode. Returns `None` for out-of-bounds in constant mode.
#[inline]
fn src_index(out_idx: usize, pad: usize, in_size: usize, mode: u8) -> Option<usize> {
    let signed = out_idx as isize - pad as isize;
    if signed >= 0 && (signed as usize) < in_size {
        return Some(signed as usize);
    }
    match mode {
        0 => None, // constant — caller writes the fill value
        1 => Some(reflect_index(signed, in_size)),
        2 => Some(edge_index(signed, in_size)),
        _ => None,
    }
}

/// Reflect-without-repeat (matches numpy `'reflect'`): the boundary
/// element is *not* duplicated; reflecting at index `-1` lands on
/// index `1`, reflecting at `in_size` lands on `in_size - 2`.
#[inline]
fn reflect_index(signed: isize, in_size: usize) -> usize {
    if in_size == 1 {
        return 0;
    }
    let period = 2 * (in_size - 1);
    let mut p = signed.rem_euclid(period as isize) as usize;
    if p >= in_size {
        p = period - p;
    }
    p
}

/// Edge replication: clamp to `[0, in_size)`.
#[inline]
fn edge_index(signed: isize, in_size: usize) -> usize {
    if signed < 0 {
        0
    } else if (signed as usize) >= in_size {
        in_size - 1
    } else {
        signed as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(mode: u8) -> PadCall {
        PadCall {
            input: SlotSpan { offset: 0, len: 4 },
            output: SlotSpan { offset: 4, len: 16 },
            n: 1,
            c: 1,
            h_in: 2,
            w_in: 2,
            pad_h: 1,
            pad_w: 1,
            value_bits: 0,
            mode,
        }
    }

    #[test]
    fn pad_constant_writes_value_around_input() {
        let mut s = vec![0.0_f32; 4 + 16];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        pad(&mut s, &make_call(0));
        assert_eq!(
            &s[4..20],
            &[
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn pad_edge_replicates_border() {
        let mut s = vec![0.0_f32; 4 + 16];
        s[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        pad(&mut s, &make_call(2));
        // Edge mode clamps to nearest input cell.
        assert_eq!(
            &s[4..20],
            &[
                1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 4.0, 4.0, //
                3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn pad_reflect_mirrors_interior() {
        // 1×1×3×3 input padded by 1 with reflect mode. Reflect should
        // mirror without repeating the boundary. Input column [0,1,2]
        // reflected on the left at index -1 lands on index 1.
        let mut s = vec![0.0_f32; 9 + 25];
        for (i, slot) in s.iter_mut().take(9).enumerate() {
            *slot = i as f32;
        }
        let call = PadCall {
            input: SlotSpan { offset: 0, len: 9 },
            output: SlotSpan { offset: 9, len: 25 },
            n: 1,
            c: 1,
            h_in: 3,
            w_in: 3,
            pad_h: 1,
            pad_w: 1,
            value_bits: 0,
            mode: 1,
        };
        pad(&mut s, &call);
        // Centre 3×3 output rows match the input rows 0..3.
        for ih in 0..3 {
            for iw in 0..3 {
                assert_eq!(s[9 + (ih + 1) * 5 + (iw + 1)], (ih * 3 + iw) as f32);
            }
        }
        // Top border row reflects from input row 1: [4, 3, 4, 5, 4].
        assert_eq!(&s[9..14], &[4.0, 3.0, 4.0, 5.0, 4.0]);
    }
}
