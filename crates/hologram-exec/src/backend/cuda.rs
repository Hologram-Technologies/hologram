//! CUDA compute backend (NVIDIA GPU) — stub.
//!
//! Auto-detected when `nvcc` is on PATH or `CUDA_HOME` is set.
//! Currently returns `Ok(false)` for all ops, falling back to CPU.

use hologram_core::op::FloatOp;

use crate::error::ExecResult;

use super::ComputeBackend;

/// CUDA GPU backend (NVIDIA).
pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    fn dispatch_float(
        &self,
        _op: &FloatOp,
        _inputs: &[&[u8]],
        _out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        Ok(false)
    }

    fn dispatch_matmul(
        &self,
        _inputs: &[&[u8]],
        _m: usize,
        _k: usize,
        _n: usize,
        _out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        Ok(false)
    }

    fn name(&self) -> &'static str {
        "cuda"
    }
}
