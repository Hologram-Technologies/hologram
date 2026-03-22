//! Metal compute backend (Apple GPU) — stub.
//!
//! Auto-detected on macOS via `build.rs` (`has_metal` cfg flag).
//! Currently returns `Ok(false)` for all ops, falling back to CPU.
//! Metal compute shader kernels will be added in a future sprint.

use hologram_core::op::FloatOp;

use crate::error::ExecResult;

use super::ComputeBackend;

/// Metal GPU backend (Apple Silicon / macOS).
pub struct MetalBackend;

impl ComputeBackend for MetalBackend {
    fn dispatch_float(
        &self,
        _op: &FloatOp,
        _inputs: &[&[u8]],
        _out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        // TODO: Metal compute shader dispatch for elementwise, matmul, softmax.
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
        // TODO: Metal GEMM via MPSMatrixMultiplication or custom compute shader.
        Ok(false)
    }

    fn name(&self) -> &'static str {
        "metal"
    }
}
