//! Q2 (24-bit) ring — Z/2^24 Z.
//!
//! Third level of the Cayley-Dickson chain: Q0(R, dim=1) → Q1(C, dim=2) → Q2(H, dim=4).
//!
//! Memory: a full activation table at Q2 would require ~50 MB per function.
//! Arithmetic uses native u32 wrapping operations masked to 24 bits — O(1), no tables.

pub mod arith;
pub mod datum;
pub mod encoding;
pub mod ring;

pub use arith::{add_q2, bnot_q2, mul_q2, neg_q2, pow_q2, pred_q2, sub_q2, succ_q2};
pub use datum::TripleDatum;
pub use encoding::{
    AngleEncoding24, Encoding24, RawEncoding24, SignedEncoding24, UnsignedEncoding24,
};
pub use ring::{TripleInvolution, TripleRing};
