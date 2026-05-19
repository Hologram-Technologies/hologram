//! Backend errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("buffer slot out of range: {0}")]
    SlotOutOfRange(u32),
    #[error("shape mismatch: expected {expected:?} got {actual:?}")]
    ShapeMismatch {
        expected: alloc::vec::Vec<u64>,
        actual: alloc::vec::Vec<u64>,
    },
    #[error("unsupported op: {0}")]
    UnsupportedOp(&'static str),
    #[error("backend init failed: {0}")]
    Init(&'static str),
    #[error("dispatch failed: {0}")]
    Dispatch(&'static str),
}

extern crate alloc;
