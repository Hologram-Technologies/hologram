//! CPU compute backend (SIMD + Accelerate BLAS).
//!
//! Delegates to the existing `float_dispatch` module which handles
//! monomorphized SIMD dispatch, Accelerate BLAS matmul, and all
//! custom op kernels.

use hologram_core::op::FloatOp;

use crate::buffer::OutputBuffer;
use crate::error::ExecResult;

use super::{ComputeBackend, KernelOutput};

/// CPU backend using monomorphized SIMD dispatch.
///
/// This is always available and serves as the fallback for GPU backends
/// that return `Skipped`.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        crate::float_dispatch::dispatch_float_into(op, inputs, None, out_buf)?;
        Ok(KernelOutput::Bytes)
    }

    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        crate::float_dispatch::matmul::dispatch_matmul_into(inputs, m, k, n, out_buf)?;
        Ok(KernelOutput::Bytes)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}
