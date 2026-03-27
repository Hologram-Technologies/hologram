//! ByteRing: Z/256Z implementing uor-foundation Ring.

mod byte_ring;

pub use byte_ring::ByteInvolution;
pub use byte_ring::ByteRing;
pub use byte_ring::HoloDivisionAlgebra;
pub(crate) use byte_ring::{Q1_ALGEBRA, Q2_ALGEBRA, Q3_ALGEBRA};
