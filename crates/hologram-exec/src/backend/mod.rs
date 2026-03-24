//! Multi-backend dispatch for tape execution.
//!
//! Each backend implements [`ComputeBackend`] for the ops it supports.
//! Unsupported ops return `Ok(false)`, causing fallback to the CPU backend.
//!
//! Backend availability is auto-detected at build time (`build.rs` emits
//! `has_metal`, `has_webgpu` cfg flags). Runtime selection is via
//! [`BackendSelector`].

pub mod cpu;

#[cfg(has_metal)]
pub mod metal;

#[cfg(has_webgpu)]
pub mod webgpu;

use hologram_core::op::FloatOp;

use crate::error::ExecResult;

/// Result of a backend kernel dispatch.
///
/// Tells the tape executor HOW to store the result:
/// - `Skipped`: backend didn't handle this op → fall back to CPU
/// - `Bytes`: result written to `out_buf` (CPU path, or GPU→copy path)
/// - `MetalBuffer`: result stored in a GPU buffer → insert directly into arena
pub enum KernelOutput {
    /// Backend did not handle this op. Fall back to CPU dispatch.
    Skipped,
    /// Result written to the provided `out_buf`. Store via swap_insert.
    Bytes,
    /// Result stored in a Metal GPU buffer. Insert directly into arena (zero-copy).
    #[cfg(has_metal)]
    MetalBuffer(::metal::Buffer),
    /// Result deferred — will be available after `flush_deferred()`.
    /// Used by WebGPU batching: encode now, submit+readback at level boundary.
    #[cfg(has_webgpu)]
    WgpuDeferred,
}

impl KernelOutput {
    /// Whether the backend handled the op (not Skipped).
    #[inline]
    #[must_use]
    pub fn handled(&self) -> bool {
        !matches!(self, KernelOutput::Skipped)
    }

    /// Extract output bytes — from out_buf for Bytes, from Metal buffer for MetalBuffer.
    /// For testing: copies Metal buffer contents to Vec.
    #[cfg(test)]
    pub fn extract_bytes(self, out_buf: Vec<u8>) -> Vec<u8> {
        match self {
            KernelOutput::Skipped => Vec::new(),
            KernelOutput::Bytes => out_buf,
            #[cfg(has_metal)]
            KernelOutput::MetalBuffer(buf) => {
                let ptr = buf.contents() as *const u8;
                let len = buf.length() as usize;
                unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
            }
        }
    }
}

/// Compute backend for tape kernel dispatch.
///
/// Each backend implements dispatch for the op types it supports.
/// Returns `KernelOutput` indicating how the result should be stored.
pub trait ComputeBackend: Send + Sync {
    /// Dispatch a float op. Writes to `out_buf` or returns a Metal buffer.
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<KernelOutput>;

    /// Dispatch a matmul (M×K × K×N). Writes to `out_buf` or returns a Metal buffer.
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<KernelOutput>;

    /// Backend name for diagnostics and logging.
    fn name(&self) -> &'static str;

    /// Flush pending GPU work (commit + wait for all encoded commands).
    ///
    /// Called at level boundaries by the tape executor. After flush,
    /// all MetalBuffers returned by previous dispatch calls contain
    /// valid GPU-written data. No-op for CPU backends.
    fn flush(&self) {}

    /// Flush deferred GPU work and return readback data in dispatch order.
    ///
    /// Called at level boundaries. Returns one `Vec<u8>` per deferred dispatch
    /// in the order they were encoded. Default: calls `flush()` and returns empty.
    /// Only WebGPU overrides this (Metal uses unified memory, CPU has no deferral).
    fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
        self.flush();
        Ok(Vec::new())
    }
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
            #[cfg(has_webgpu)]
            Self::WebGpu => {
                use std::sync::{Arc, OnceLock};
                static WEBGPU_SEL: OnceLock<Option<Arc<webgpu::WebGpuBackend>>> = OnceLock::new();
                let cached = WEBGPU_SEL.get_or_init(|| webgpu::WebGpuBackend::new().map(Arc::new));
                match cached {
                    Some(b) => Box::new(CachedWebGpuBackend(Arc::clone(b))),
                    None => Box::new(cpu::CpuBackend),
                }
            }
            // Requested backend not compiled in — fall back to CPU.
            #[allow(unreachable_patterns)]
            _ => Box::new(cpu::CpuBackend),
        }
    }
}

/// Returns the best available backend for the current build.
///
/// Priority: Metal > WebGPU > CPU.
/// Metal is preferred on macOS (Apple Silicon native). WebGPU is preferred
/// over CUDA because it's cross-platform (browser + native via wgpu).
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

    #[cfg(has_webgpu)]
    {
        use std::sync::{Arc, OnceLock};
        static WEBGPU: OnceLock<Option<Arc<webgpu::WebGpuBackend>>> = OnceLock::new();
        let cached = WEBGPU.get_or_init(|| webgpu::WebGpuBackend::new().map(Arc::new));
        if let Some(backend) = cached {
            return Box::new(CachedWebGpuBackend(Arc::clone(backend)));
        }
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
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_float(op, inputs, out_buf)
    }
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_matmul(inputs, m, k, n, out_buf)
    }
    fn name(&self) -> &'static str {
        "metal"
    }
    fn flush(&self) {
        self.0.flush();
    }
}

/// Wrapper that delegates to a shared WebGpuBackend via Arc.
#[cfg(has_webgpu)]
struct CachedWebGpuBackend(std::sync::Arc<webgpu::WebGpuBackend>);

#[cfg(has_webgpu)]
impl ComputeBackend for CachedWebGpuBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_float(op, inputs, out_buf)
    }
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_matmul(inputs, m, k, n, out_buf)
    }
    fn name(&self) -> &'static str {
        "webgpu"
    }
    fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
        self.0.flush_deferred_impl()
    }
}

/// List all backends available in this build.
#[must_use]
pub fn available_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    #[cfg(has_metal)]
    v.push("metal");
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
        let result = b
            .dispatch_matmul(&inputs, m, k, n, &mut out_buf)
            .expect("Metal matmul dispatch failed");
        assert!(
            result.handled(),
            "Metal should handle 128×64 × 64×128 matmul"
        );
        b.flush(); // Commit batched GPU work before reading output.
        let output_bytes = result.extract_bytes(out_buf);
        assert_eq!(output_bytes.len(), m * n * 4);

        let out_floats: &[f32] = bytemuck::cast_slice(&output_bytes);
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
    fn metal_dispatch_softmax() {
        use hologram_core::op::FloatOp;

        let b = BackendSelector::Metal.resolve();
        // 1M floats in rows of 1024 = 6MB (above threshold)
        let row_size = 1024usize;
        let n_rows = 1024usize;
        let n_floats = row_size * n_rows;
        let input: Vec<u8> = (0..n_floats)
            .flat_map(|i| ((i % row_size) as f32 * 0.01).to_le_bytes())
            .collect();
        let inputs: Vec<&[u8]> = vec![&input];

        let mut out_buf = Vec::new();
        let result = b
            .dispatch_float(
                &FloatOp::Softmax {
                    size: row_size as u32,
                },
                &inputs,
                &mut out_buf,
            )
            .expect("Metal softmax dispatch failed");
        assert!(
            result.handled(),
            "Metal should handle softmax on large buffer"
        );
        b.flush(); // Commit batched GPU work before reading output.
        let output_bytes = result.extract_bytes(out_buf);
        assert_eq!(output_bytes.len(), input.len());

        // Each row should sum to ~1.0
        let out_floats: &[f32] = bytemuck::cast_slice(&output_bytes);
        let row_sum: f32 = out_floats[..row_size].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "Metal softmax row sum = {row_sum}, expected 1.0"
        );
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
        let result = b
            .dispatch_float(&FloatOp::Relu, &inputs, &mut out_buf)
            .expect("Metal dispatch failed");
        assert!(result.handled(), "Metal should handle Relu on 6MB buffer");
        b.flush(); // Commit batched GPU work before reading output.

        // Extract output bytes — may be in out_buf (Bytes) or Metal buffer.
        let output_bytes: Vec<u8> = match result {
            KernelOutput::Bytes => out_buf,
            #[cfg(has_metal)]
            KernelOutput::MetalBuffer(buf) => {
                let ptr = buf.contents() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, buf.length() as usize) }.to_vec()
            }
            KernelOutput::Skipped => panic!("expected handled"),
        };
        assert_eq!(output_bytes.len(), input.len());

        // Verify: negative values → 0, positive values unchanged.
        let out_floats: &[f32] = bytemuck::cast_slice(&output_bytes);
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
