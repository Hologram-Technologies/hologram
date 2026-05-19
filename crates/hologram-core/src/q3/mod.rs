//! Q3 (32-bit) octonion ring — the first non-associative algebra in the CD chain.

pub mod arith;
pub mod datum;
pub mod ring;

pub use ring::{OctonionInvolution, OctonionRing};
