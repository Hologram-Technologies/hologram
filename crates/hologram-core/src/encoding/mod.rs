//! Encoding: embed continuous values into discrete byte space and lift back.
//!
//! The pi-F-lambda pipeline:
//! - **pi (embed)**: continuous → byte
//! - **F**: O(1) byte→byte LUT lookup
//! - **lambda (lift)**: byte → continuous

mod angle;
mod raw;
mod signed;
mod unsigned;

pub use angle::AngleEncoding;
pub use raw::RawEncoding;
pub use signed::SignedEncoding;
pub use unsigned::UnsignedEncoding;

/// Trait for embedding continuous values into the byte ring and lifting back.
pub trait Encoding {
    /// Embed a continuous f64 value into a byte.
    fn embed(&self, value: f64) -> u8;

    /// Lift a byte back to a continuous f64 value.
    fn lift(&self, byte: u8) -> f64;

    /// Human-readable name of this encoding.
    fn name(&self) -> &'static str;
}
