//! Multi-backend dispatch for tape execution.
//!
//! Each backend implements [`ComputeBackend`] for the ops it supports.
//! Unsupported ops return `Ok(false)`, causing fallback to the CPU backend.
//!
//! Backend availability is auto-detected at build time (`build.rs` emits
//! `has_metal`, `has_cuda`, `has_webgpu` cfg flags). Runtime selection
//! is via [`BackendSelector`].

pub mod cpu;

#[cfg(has_metal)]
pub mod metal;

#[cfg(has_cuda)]
pub mod cuda;

#[cfg(has_webgpu)]
pub mod webgpu;

use hologram_core::op::FloatOp;

use crate::error::ExecResult;

/// Compute backend for tape kernel dispatch.
///
/// Each backend implements dispatch for the op types it supports.
/// Methods return `Ok(true)` if handled, `Ok(false)` to fall back to CPU.
pub trait ComputeBackend: Send + Sync {
    /// Dispatch a float op into the output buffer.
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool>;

    /// Dispatch a matmul (M×K × K×N) into the output buffer.
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool>;

    /// Backend name for diagnostics and logging.
    fn name(&self) -> &'static str;
}

/// Runtime backend selector.
///
/// `Auto` picks the best available backend for the current build.
/// Specific variants force a particular backend (falls back to CPU
/// if the requested backend wasn't compiled in).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendSelector {
    /// Best available backend (GPU → CPU priority).
    #[default]
    Auto,
    /// Force CPU backend (SIMD + Accelerate BLAS).
    Cpu,
    /// Force Metal backend (macOS/iOS Apple GPU).
    Metal,
    /// Force CUDA backend (NVIDIA GPU).
    Cuda,
    /// Force WebGPU backend (browser/wgpu).
    WebGpu,
}

impl BackendSelector {
    /// Resolve to the best concrete backend for this build + selector.
    #[must_use]
    pub fn resolve(&self) -> Box<dyn ComputeBackend> {
        match self {
            Self::Auto => default_backend(),
            Self::Cpu => Box::new(cpu::CpuBackend),
            #[cfg(has_metal)]
            Self::Metal => Box::new(metal::MetalBackend),
            #[cfg(has_cuda)]
            Self::Cuda => Box::new(cuda::CudaBackend),
            #[cfg(has_webgpu)]
            Self::WebGpu => Box::new(webgpu::WebGpuBackend),
            // Requested backend not compiled in — fall back to CPU.
            #[allow(unreachable_patterns)]
            _ => Box::new(cpu::CpuBackend),
        }
    }
}

/// Returns the best available backend for the current build.
///
/// Priority: CUDA > Metal > WebGPU > CPU.
#[must_use]
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(has_cuda)]
    {
        return Box::new(cuda::CudaBackend);
    }

    #[cfg(has_metal)]
    {
        return Box::new(metal::MetalBackend);
    }

    #[cfg(has_webgpu)]
    {
        return Box::new(webgpu::WebGpuBackend);
    }

    #[allow(unreachable_code)]
    Box::new(cpu::CpuBackend)
}

/// List all backends available in this build.
#[must_use]
pub fn available_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    #[cfg(has_metal)]
    v.push("metal");
    #[cfg(has_cuda)]
    v.push("cuda");
    #[cfg(has_webgpu)]
    v.push("webgpu");
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_available() {
        assert!(available_backends().contains(&"cpu"));
    }

    #[test]
    fn default_backend_returns_something() {
        let b = default_backend();
        assert!(!b.name().is_empty());
    }

    #[test]
    fn auto_selector_resolves() {
        let b = BackendSelector::Auto.resolve();
        assert!(!b.name().is_empty());
    }

    #[test]
    fn cpu_selector_forces_cpu() {
        let b = BackendSelector::Cpu.resolve();
        assert_eq!(b.name(), "cpu");
    }
}
