//! Hologram operations as Term-arena emitters (spec Part V).
//!
//! Each canonical op is a marker type plus a const-tagged IRI plus an
//! `emit_term` function that emits a `Term` tree into a caller-provided
//! `TermArena<CAP>`. The Term tree is the formal specification (per
//! spec invariant I-9). Backend kernels (in `hologram-backend`) are the
//! execution form; equivalence is verified by per-op reference evaluators
//! (`reference` module).
//!
//! The closed 64-op catalog organized per spec V.3.

#![no_std]

pub mod kind;
pub mod emit;
pub mod dispatch;
pub mod reference;
pub mod lut;
pub mod activations;
pub mod grounding;
pub mod axes;

pub mod direct;
pub mod elementwise_unary;
pub mod elementwise_binary;
pub mod linalg;
pub mod conv;
pub mod normalization;
pub mod reduction;
pub mod layout;
pub mod activation_reduce;
pub mod pooling;
pub mod structured;
pub mod utility;
pub mod backward;
pub mod quantization;

pub use kind::OpKind;
pub use dispatch::emit_op_term;
pub use reference::{ReferenceEvaluator, EvalError, ScalarEvaluatorU64};
