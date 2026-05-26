//! Hologram operations as Term-arena emitters (spec Part V).
//!
//! Each canonical op is a marker type plus a const-tagged IRI plus an
//! `emit_term` function that emits a `Term` tree into a caller-provided
//! `HoloArena<CAP>`. The Term tree is the formal specification (per
//! spec invariant I-9). Backend kernels (in `hologram-backend`) are the
//! execution form; equivalence is verified by per-op reference evaluators
//! (`reference` module).
//!
//! The closed 64-op catalog organized per spec V.3.

#![no_std]

pub mod activations;
pub mod axes;
pub mod dispatch;
pub mod emit;
pub mod grounding;
pub mod kind;
pub mod lut;
pub mod reference;

pub mod activation_reduce;
pub mod conv;
pub mod direct;
pub mod elementwise_binary;
pub mod elementwise_unary;
pub mod layout;
pub mod linalg;
pub mod normalization;
pub mod pooling;
pub mod quantization;
pub mod reduction;
pub mod structured;
pub mod utility;

pub use dispatch::emit_op_term;
pub use emit::{HoloArena, HoloTerm, HOLOGRAM_INLINE_BYTES};
pub use kind::OpKind;
pub use reference::{EvalError, ReferenceEvaluator, ScalarEvaluatorU64};
