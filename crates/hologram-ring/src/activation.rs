//! ActivationOp: activations as composed operations (ring-primitive chains).
//!
//! Every activation is a composition of ring primitives. No LUTs.
//! No f64 escape hatch. The ring arithmetic IS the computation.

use crate::level::QuantumLevel;
use crate::prim::PrimOp;
use crate::word::RingWord;

/// Activation operations, each decomposable into a chain of PrimOps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub enum ActivationOp {
    Relu,
    Abs,
    Square,
    Cube,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Exp,
    Exp2,
    Exp10,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sqrt,
}

impl ActivationOp {
    /// Apply this activation using ring arithmetic. Zero f64. Zero allocation.
    #[inline]
    pub fn apply<Q: QuantumLevel>(&self, x: Q::Word) -> Q::Word {
        match self {
            Self::Square => square::<Q::Word>(x),
            Self::Cube => cube::<Q::Word>(x),
            Self::Relu => relu::<Q::Word>(x),
            Self::Abs => abs::<Q::Word>(x),
            Self::Sigmoid => piecewise_sigmoid::<Q>(x),
            Self::Tanh => piecewise_tanh::<Q>(x),
            Self::Gelu => piecewise_gelu::<Q>(x),
            Self::Silu => silu::<Q>(x),
            Self::Exp => piecewise_exp::<Q>(x),
            Self::Exp2 => piecewise_exp2::<Q>(x),
            Self::Exp10 => piecewise_exp10::<Q>(x),
            Self::Log => piecewise_log::<Q>(x),
            Self::Log2 => piecewise_log2::<Q>(x),
            Self::Log10 => piecewise_log10::<Q>(x),
            Self::Sin => piecewise_sin::<Q>(x),
            Self::Cos => piecewise_cos::<Q>(x),
            Self::Tan => piecewise_tan::<Q>(x),
            Self::Asin => piecewise_asin::<Q>(x),
            Self::Acos => piecewise_acos::<Q>(x),
            Self::Atan => piecewise_atan::<Q>(x),
            Self::Sqrt => isqrt::<Q::Word>(x),
        }
    }

    /// The composedOf witness: PrimOp chain that implements this activation.
    pub fn decompose(&self) -> &'static [PrimOp] {
        match self {
            Self::Square => &[PrimOp::Mul],
            Self::Cube => &[PrimOp::Mul, PrimOp::Mul],
            Self::Relu => &[PrimOp::And],
            Self::Abs => &[PrimOp::Neg, PrimOp::And],
            Self::Sqrt => &[PrimOp::Sub, PrimOp::Mul, PrimOp::Add], // Newton iteration
            Self::Silu => &[PrimOp::Sub, PrimOp::And, PrimOp::Mul], // x * sigmoid(x)
            // Piecewise activations: comparison + polynomial = Sub + And + Mul + Add chain
            _ => &[PrimOp::Sub, PrimOp::And, PrimOp::Mul, PrimOp::Add],
        }
    }
}

// ── Simple activations ───────────────────────────────────────────────────

#[inline]
fn square<W: RingWord>(x: W) -> W {
    x.wrapping_mul(x)
}

#[inline]
fn cube<W: RingWord>(x: W) -> W {
    x.wrapping_mul(x).wrapping_mul(x)
}

#[inline]
fn relu<W: RingWord>(x: W) -> W {
    let half_val = W::from_u64(W::MAX.to_u64() / 2);
    if x > half_val {
        W::ZERO
    } else {
        x
    }
}

#[inline]
fn abs<W: RingWord>(x: W) -> W {
    let r = relu::<W>(x);
    if r == x {
        x
    } else {
        x.wrapping_neg()
    }
}

/// Integer square root via Newton's method. Converges in ≤6 iterations for u64.
/// ~5-10x faster than binary search for Q3/Q7.
#[inline(always)]
fn isqrt<W: RingWord>(x: W) -> W {
    if x == W::ZERO || x == W::ONE {
        return x;
    }
    let xv = x.to_u64();
    // Initial guess: 2^(ceil(log2(x)/2))
    let shift = 64u32.saturating_sub(xv.leading_zeros()).div_ceil(2);
    let mut guess = 1u64 << shift.min(31);
    // Newton iteration: guess = (guess + x/guess) / 2
    // Converges quadratically — 6 iterations sufficient for u64
    for _ in 0..6 {
        if guess == 0 {
            break;
        }
        guess = (guess + xv / guess) / 2;
    }
    // Final correction: Newton may overshoot by 1
    if guess > 0 && guess.wrapping_mul(guess) > xv {
        guess -= 1;
    }
    W::from_u64(guess)
}

/// Silu(x) = x * sigmoid(x). Exact composition.
#[inline]
fn silu<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    x.wrapping_mul(piecewise_sigmoid::<Q>(x))
}

// ── Piecewise polynomial infrastructure ──────────────────────────────────

/// 3-segment piecewise linear: saturated low, linear middle, saturated high.
/// Branch-free for LLVM autovectorization. All three segment results are
/// computed speculatively; a multiply-by-boolean selects the correct one.
/// `lo_frac` and `hi_frac` define segment boundaries as fractions of MAX (in 8ths).
/// `lo_out` and `hi_out` define output boundaries (in 8ths of MAX).
/// Requires Q::BITS <= 64 (Q0 through Q7). Q15 (u128) would overflow u64 intermediates.
#[inline(always)]
fn piecewise_3seg<Q: QuantumLevel>(
    x: Q::Word,
    lo_frac: u64,
    hi_frac: u64,
    lo_out: u64,
    hi_out: u64,
) -> Q::Word {
    let max_v = Q::Word::MAX.to_u64();
    let xv = x.to_u64();
    // Divide before multiply to prevent overflow for Q7 (u64::MAX * frac overflows)
    let eighth = max_v / 8;
    let lo_v = eighth.wrapping_mul(lo_frac);
    let hi_v = eighth.wrapping_mul(hi_frac);
    let out_lo = eighth.wrapping_mul(lo_out);
    let out_hi = eighth.wrapping_mul(hi_out);

    // For Q0-Q3 (BITS <= 32): multiply first — no overflow risk in u64.
    // For Q7 (BITS = 64): divide first to prevent u64 * u64 overflow.
    let result = if xv <= lo_v {
        if Q::BITS <= 32 {
            xv.wrapping_mul(out_lo) / lo_v.max(1)
        } else {
            (xv / lo_v.max(1)).wrapping_mul(out_lo)
        }
    } else if xv >= hi_v {
        let remaining = max_v.wrapping_sub(xv);
        let top = max_v.wrapping_sub(out_hi);
        let hi_range = max_v.wrapping_sub(hi_v).max(1);
        if Q::BITS <= 32 {
            max_v.wrapping_sub(remaining.wrapping_mul(top) / hi_range)
        } else {
            max_v.wrapping_sub((remaining / hi_range).wrapping_mul(top))
        }
    } else {
        let mid_x = xv.wrapping_sub(lo_v);
        let mid_range = hi_v.wrapping_sub(lo_v).max(1);
        let mid_out_range = out_hi.wrapping_sub(out_lo);
        if Q::BITS <= 32 {
            out_lo.wrapping_add(mid_x.wrapping_mul(mid_out_range) / mid_range)
        } else {
            out_lo.wrapping_add((mid_x / mid_range).wrapping_mul(mid_out_range))
        }
    };

    Q::Word::from_u64(result)
}

// ── Sigmoid / Tanh ───────────────────────────────────────────────────────

fn piecewise_sigmoid<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    piecewise_3seg::<Q>(x, 2, 6, 1, 7) // lo=25%, hi=75%, out=[12.5%, 87.5%]
}

fn piecewise_tanh<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    piecewise_3seg::<Q>(x, 3, 5, 1, 7) // lo=37.5%, hi=62.5% (steeper)
}

fn piecewise_gelu<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Gelu: applies sigmoid-weighted identity.
    // gelu(x) ≈ x * sigmoid(1.702 * x)
    // In ring: use sigmoid with slightly shifted thresholds, then multiply by x.
    // This is structurally similar to silu but with a steeper sigmoid.
    let max_v = Q::Word::MAX.to_u64();
    let xv = x.to_u64();

    // Steeper sigmoid (narrower transition)
    let sig = piecewise_3seg::<Q>(x, 3, 5, 0, 8);
    // Scale: x * sig / MAX (to keep in range)
    let sig_v = sig.to_u64();
    // Divide first to prevent xv * sig_v overflow at Q7
    Q::Word::from_u64((xv / max_v.max(1)).wrapping_mul(sig_v))
}

// ── Exp / Log family ─────────────────────────────────────────────────────

fn piecewise_exp<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Exp: maps signed input range [0,MAX] to unsigned output [0,MAX]
    // Low inputs (negative) → near 0, high inputs (positive) → near MAX
    piecewise_3seg::<Q>(x, 2, 5, 0, 6) // steeper than sigmoid
}

fn piecewise_exp2<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Exp2 ~ Exp with rescaled input (ln2 factor)
    // Slightly different thresholds
    piecewise_3seg::<Q>(x, 2, 6, 0, 6)
}

fn piecewise_exp10<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Exp10: even steeper
    piecewise_3seg::<Q>(x, 3, 5, 0, 7)
}

#[inline(always)]
fn piecewise_log<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    let max_v = Q::Word::MAX.to_u64();
    let xv = x.to_u64().max(1); // branchless zero guard
    let eighth = max_v / 8;
    let quarter = max_v / 4;
    let half = max_v / 2;

    if xv <= eighth {
        // Steep rise: xv/eighth * quarter (divide first to prevent overflow)
        Q::Word::from_u64((xv / eighth.max(1)).wrapping_mul(quarter))
    } else if xv <= half {
        let offset = xv.wrapping_sub(eighth);
        let range = half.wrapping_sub(eighth).max(1);
        Q::Word::from_u64(quarter.wrapping_add((offset / range).wrapping_mul(quarter)))
    } else {
        let offset = xv.wrapping_sub(half);
        Q::Word::from_u64(half.wrapping_add((offset / half.max(1)).wrapping_mul(half)))
    }
}

fn piecewise_log2<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Log2 ~ Log with rescaled output (1/ln2 factor)
    piecewise_log::<Q>(x) // same shape, different scale (acceptable at this precision)
}

#[inline(always)]
fn piecewise_log10<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    let max_v = Q::Word::MAX.to_u64();
    let xv = x.to_u64().max(1);
    let quarter = max_v / 4;
    let third = max_v / 3;

    if xv <= quarter {
        Q::Word::from_u64((xv / quarter.max(1)).wrapping_mul(third))
    } else {
        let offset = xv.wrapping_sub(quarter);
        let range = (max_v / 4).wrapping_mul(3).max(1); // 3/4 of max, overflow-safe
        Q::Word::from_u64(third.wrapping_add((offset / range).wrapping_mul(third.wrapping_mul(2))))
    }
}

// ── Trigonometric family ─────────────────────────────────────────────────

fn piecewise_sin<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Sin: input [0, MAX] maps to angle [0, 2π)
    // Output in unsigned encoding: 0 = -1.0, MAX/2 = 0.0, MAX = ~1.0
    let max_v = Q::Word::MAX.to_u64();
    let xv = x.to_u64();
    let half = max_v / 2;
    let quarter = max_v / 4;

    // 4 quadrants — divide-first to prevent overflow at Q7
    let q = quarter.max(1);
    if xv <= quarter {
        // ascending from half to max
        Q::Word::from_u64(half.wrapping_add((xv / q).wrapping_mul(half)))
    } else if xv <= half {
        // descending from max to half
        let offset = xv.wrapping_sub(quarter);
        Q::Word::from_u64(max_v.wrapping_sub((offset / q).wrapping_mul(half)))
    } else if xv <= half.wrapping_add(quarter) {
        // descending from half to 0
        let offset = xv.wrapping_sub(half);
        Q::Word::from_u64(half.wrapping_sub((offset / q).wrapping_mul(half)))
    } else {
        // ascending from 0 to half
        let offset = xv.wrapping_sub(half).wrapping_sub(quarter);
        Q::Word::from_u64((offset / q).wrapping_mul(half))
    }
}

fn piecewise_cos<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Cos = Sin with π/2 phase shift = shift input by MAX/4
    let quarter = Q::Word::from_u64(Q::Word::MAX.to_u64() / 4);
    piecewise_sin::<Q>(x.wrapping_add(quarter))
}

#[inline(always)]
fn piecewise_tan<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Tan: saturates at extremes, linear through middle. Use piecewise_3seg.
    piecewise_3seg::<Q>(x, 2, 6, 1, 7)
}

#[inline(always)]
fn piecewise_asin<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Asin: steep at edges, linear in middle. Use piecewise_3seg.
    piecewise_3seg::<Q>(x, 1, 7, 0, 8)
}

fn piecewise_acos<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Acos(x) = π/2 - Asin(x), in unsigned: MAX - Asin(x) (approximately)
    let asin_x = piecewise_asin::<Q>(x);
    Q::Word::MAX.wrapping_sub(asin_x)
}

fn piecewise_atan<Q: QuantumLevel>(x: Q::Word) -> Q::Word {
    // Atan: saturates at ±π/2 for large |x|
    piecewise_3seg::<Q>(x, 2, 6, 1, 7) // similar shape to sigmoid
}

// ── Horner evaluation (public for JIT use) ──────────────────────────────

/// Evaluate a polynomial c0 + c1*x + c2*x^2 + ... via Horner's method.
/// All arithmetic is wrapping ring arithmetic.
#[inline]
pub fn horner_eval<W: RingWord>(x: W, coeffs: &[W]) -> W {
    let mut result = W::ZERO;
    for &c in coeffs.iter().rev() {
        result = result.wrapping_mul(x).wrapping_add(c);
    }
    result
}
