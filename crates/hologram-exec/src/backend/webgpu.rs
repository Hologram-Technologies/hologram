//! WebGPU compute backend (browser/wgpu) — stub.
//!
//! Auto-detected for `wasm32` targets via `build.rs`.
//! Currently returns `Ok(false)` for all ops, falling back to CPU.

use hologram_core::op::FloatOp;

use crate::error::ExecResult;

use super::ComputeBackend;

/// WebGPU backend (browser via wgpu or native wgpu).
pub struct WebGpuBackend;

impl ComputeBackend for WebGpuBackend {
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
        "webgpu"
    }
}
