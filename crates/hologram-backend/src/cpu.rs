//! CPU backend (spec IX.2).
//!
//! `dispatch` is a closed match over `KernelCall`. Each arm calls into
//! a kernel function in `cpu_kernels::*` that implements the same
//! semantics as the corresponding op marker's Term tree (spec V.3).
//! Equivalence is verified by per-op reference-evaluator tests
//! (per spec VII.3).

use crate::backend::Backend;
use crate::error::BackendError;
use crate::kernel_call::KernelCall;
use crate::workspace::Workspace;
use core::marker::PhantomData;
use hologram_types::ActiveCpuBounds;

pub mod dtype;
mod float_kernels;

/// Doc-hidden bench/test re-entry into the decode-attention engine — the one
/// deliberate opening; the kernel module itself stays crate-internal.
#[doc(hidden)]
pub use float_kernels::decode_attention_engine_for_tests;
mod kernels;
/// LUT-accelerated low-precision activations (PM_7 Q0/Q1). Needs `OnceLock`
/// (std) for the process-lifetime table cache; under no_std the activations
/// are computed directly (a compile-time choice, not a runtime fallback).
#[cfg(feature = "std")]
pub mod lut;
#[cfg(not(feature = "std"))]
pub mod mathf;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod simd;
/// Embedder-provided wasm worker pool (shared-memory atomics builds only);
/// the plain simd128 build stays the witnessed single-threaded fallback.
#[cfg(all(
    target_arch = "wasm32",
    feature = "wasm-threads",
    target_feature = "simd128"
))]
pub mod wasm_pool;

/// CPU backend parameterized over the runtime workspace shape.
///
/// `ActiveCpuBounds` resolves at compile time per `target_arch` / `target_feature`,
/// so the inner-loop kernels select the widest available SIMD width without
/// runtime branching.
///
/// `Clone`/`Copy` are implemented manually rather than derived so that
/// they don't pick up an unwanted `W: Clone` bound — the only field is
/// `PhantomData<W>`, so the marker is always trivially copyable.
#[derive(Debug)]
pub struct CpuBackend<W: Workspace> {
    _ws: PhantomData<W>,
}

impl<W: Workspace> Copy for CpuBackend<W> {}
impl<W: Workspace> Clone for CpuBackend<W> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: Workspace> CpuBackend<W> {
    pub const fn new() -> Self {
        Self { _ws: PhantomData }
    }
}

impl<W: Workspace> Default for CpuBackend<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Workspace> Backend for CpuBackend<W> {
    type Bounds = ActiveCpuBounds;
    type WS = W;

    #[inline]
    fn dispatch(&mut self, call: &KernelCall, ws: &mut Self::WS) -> Result<(), BackendError> {
        kernels::dispatch(call, ws)
    }
}
