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
            Self::Metal => {
                // Reuse the process-global cached Metal backend.
                use std::sync::{Arc, OnceLock};
                static METAL_SEL: OnceLock<Option<Arc<metal::MetalBackend>>> = OnceLock::new();
                let cached = METAL_SEL.get_or_init(|| metal::MetalBackend::new().map(Arc::new));
                match cached {
                    Some(b) => Box::new(CachedMetalBackend(Arc::clone(b))),
                    None => Box::new(cpu::CpuBackend),
                }
            }
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
/// The returned backend is cached — repeated calls return the same instance.
#[must_use]
pub fn default_backend() -> Box<dyn ComputeBackend> {
    // Use a process-global cached Metal backend to avoid re-compiling
    // shaders on every resolve() call.
    #[cfg(has_metal)]
    {
        use std::sync::{Arc, OnceLock};
        static METAL: OnceLock<Option<Arc<metal::MetalBackend>>> = OnceLock::new();
        let cached = METAL.get_or_init(|| metal::MetalBackend::new().map(Arc::new));
        if let Some(backend) = cached {
            return Box::new(CachedMetalBackend(Arc::clone(backend)));
        }
    }

    #[cfg(has_cuda)]
    {
        return Box::new(cuda::CudaBackend);
    }

    #[cfg(has_webgpu)]
    {
        return Box::new(webgpu::WebGpuBackend);
    }

    #[allow(unreachable_code)]
    Box::new(cpu::CpuBackend)
}

/// Wrapper that delegates to a shared MetalBackend via Arc.
#[cfg(has_metal)]
struct CachedMetalBackend(std::sync::Arc<metal::MetalBackend>);

#[cfg(has_metal)]
impl ComputeBackend for CachedMetalBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        self.0.dispatch_float(op, inputs, out_buf)
    }
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        self.0.dispatch_matmul(inputs, m, k, n, out_buf)
    }
    fn name(&self) -> &'static str {
        "metal"
    }
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

    #[cfg(has_metal)]
    #[test]
    fn metal_backend_available_on_macos() {
        assert!(available_backends().contains(&"metal"));
        let b = BackendSelector::Auto.resolve();
        // On macOS, Auto should pick Metal (higher priority than CPU).
        assert_eq!(b.name(), "metal");
    }

    #[cfg(has_metal)]
    #[test]
    fn metal_dispatch_matmul() {
        let b = BackendSelector::Metal.resolve();

        // 128×64 × 64×128 matmul (above Metal's 128×128 output threshold).
        let m = 128usize;
        let k = 64usize;
        let n = 128usize;

        // A = identity-like (1s on diagonal, 0s elsewhere — simplified as all 1s)
        let mut a_floats: Vec<f32> = vec![0.0; m * k];
        // Set first row to all 1.0s
        for j in 0..k {
            a_floats[j] = 1.0;
        }
        let a: Vec<u8> = bytemuck::cast_slice(&a_floats).to_vec();

        // B = all 2.0s
        let b_data: Vec<f32> = vec![2.0; k * n];
        let b_bytes: Vec<u8> = bytemuck::cast_slice(&b_data).to_vec();

        let inputs: Vec<&[u8]> = vec![&a, &b_bytes];
        let mut out_buf = Vec::new();
        let handled = b
            .dispatch_matmul(&inputs, m, k, n, &mut out_buf)
            .expect("Metal matmul dispatch failed");
        assert!(handled, "Metal should handle 128×64 × 64×128 matmul");
        assert_eq!(out_buf.len(), m * n * 4);

        let out_floats: &[f32] = bytemuck::cast_slice(&out_buf);
        // First row of C: sum(A[0,:] * B[:,j]) = sum(1.0 * 2.0, k times) = 2*k = 128.0
        for j in 0..n {
            assert!(
                (out_floats[j] - (2.0 * k as f32)).abs() < 0.1,
                "matmul C[0,{j}]: got {}, expected {}",
                out_floats[j],
                2.0 * k as f32,
            );
        }
        // Second row should be all 0s (A[1,:] = 0)
        for j in 0..n {
            assert!(
                out_floats[n + j].abs() < 0.1,
                "matmul C[1,{j}]: got {}, expected 0",
                out_floats[n + j],
            );
        }
    }

    #[cfg(has_metal)]
    #[test]
    fn metal_dispatch_relu() {
        use hologram_core::op::FloatOp;

        let b = BackendSelector::Metal.resolve();
        // Create f32 input: 1.5M floats = 6MB (above Metal threshold of 4MB)
        let input: Vec<u8> = (0..1_500_000u32)
            .flat_map(|i| {
                let v = (i as f32 - 750_000.0) * 0.001;
                v.to_le_bytes()
            })
            .collect();
        let inputs: Vec<&[u8]> = vec![&input];

        let mut out_buf = Vec::new();
        let handled = b
            .dispatch_float(&FloatOp::Relu, &inputs, &mut out_buf)
            .expect("Metal dispatch failed");
        assert!(handled, "Metal should handle Relu on 8KB buffer");
        assert_eq!(out_buf.len(), input.len());

        // Verify: negative values → 0, positive values unchanged.
        let out_floats: &[f32] = bytemuck::cast_slice(&out_buf);
        // Spot-check a few values instead of all 1.5M.
        let check_indices = [0, 1000, 750_000, 1_000_000, 1_499_999];
        for &i in &check_indices {
            let src = (i as f32 - 750_000.0) * 0.001;
            let expected = src.max(0.0);
            let got = out_floats[i];
            assert!(
                (got - expected).abs() < 1e-4,
                "Relu mismatch at {i}: got {got}, expected {expected}"
            );
        }
    }
}
