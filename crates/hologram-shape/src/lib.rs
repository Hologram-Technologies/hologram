//! Runtime tensor shape tracking for hologram execution.
//!
//! Every buffer in the execution arena carries a `TensorShape` that records
//! its concrete dimensions and element dtype. Shape inference rules compute
//! output shapes from input shapes + op parameters, eliminating heuristic
//! dimension guessing from buffer byte lengths.

mod infer;
mod infer_rules;
mod registry;
mod tensor_shape;
mod validate;

pub use infer::{infer_output_shape, ShapeError};
pub use registry::ShapeRegistry;
pub use tensor_shape::TensorShape;
pub use validate::{broadcast_shapes, validate_buffer_shape};
