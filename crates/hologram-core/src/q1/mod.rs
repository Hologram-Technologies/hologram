//! Q1 (16-bit, Z/65536Z) quantum level support.
//!
//! Extends the Q0 (8-bit) foundation to 16-bit precision. Key differences:
//!
//! - **Arithmetic**: Direct wrapping operations (no LUT — a Q1 table would be 4 GB).
//! - **Activations**: Full 65536-entry precomputed tables (128 KB each, 2.7 MB total).
//! - **Observables**: Stratum/curvature via Q0 byte decomposition; others computed directly.
//! - **View**: `ElementWiseView16` with 128 KB heap-allocated table (vs Q0's 256-byte stack table).

pub mod activation;
pub mod arith;
pub mod datum;
pub mod encoding;
pub mod observables;
pub mod op;
pub mod ring;
#[cfg(feature = "std")]
pub mod view;

// Re-exports for convenience.
pub use arith::{add_q1, mul_q1, pow_q1, sub_q1};
pub use datum::{WordAddress, WordDatum};
pub use encoding::{
    AngleEncoding16, Encoding16, RawEncoding16, SignedEncoding16, UnsignedEncoding16,
};
pub use observables::{
    curvature_q1, domain_q1, orbit_class_q1, rank_q1, stratum_q1, torus_offset_q1, torus_page_q1,
};
pub use op::{LutOp16, Op16, PrimOp16};
pub use ring::{WordInvolution, WordRing};
#[cfg(feature = "std")]
pub use view::ElementWiseView16;
