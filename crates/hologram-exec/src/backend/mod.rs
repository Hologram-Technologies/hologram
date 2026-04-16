//! Multi-backend dispatch for tape execution.
//!
//! Each backend implements [`ComputeBackend`] for the ops it supports.
//! Unsupported ops return `Ok(false)`, causing fallback to the CPU backend.
//!
//! Backend availability is auto-detected at build time (`build.rs` emits
//! `has_metal`, `has_webgpu` cfg flags). Runtime selection is via
//! [`BackendSelector`].

pub mod cpu;
pub mod hardware;

#[cfg(has_metal)]
pub mod metal;

#[cfg(has_webgpu)]
pub mod webgpu;

use hologram_core::op::FloatOp;

use crate::buffer::OutputBuffer;
use crate::error::ExecResult;

/// Backend-agnostic GPU buffer handle.
///
/// Enables GPU-to-GPU op chaining without CPU readback. The executor
/// stores these alongside `OutputBuffer` slots and passes them to
/// `dispatch_*_chained` methods when the next op also runs on GPU.
pub enum GpuBuffer {
    #[cfg(has_metal)]
    Metal(::metal::Buffer),
    #[cfg(has_webgpu)]
    Wgpu(wgpu::Buffer),
}

impl GpuBuffer {
    /// Byte length of the GPU buffer.
    pub fn byte_len(&self) -> usize {
        match self {
            #[cfg(has_metal)]
            GpuBuffer::Metal(buf) => buf.length() as usize,
            #[cfg(has_webgpu)]
            GpuBuffer::Wgpu(buf) => buf.size() as usize,
        }
    }

    /// Clone the GPU buffer reference (reference-counted, zero-copy).
    /// Used for Reshape which reinterprets the same data with a new shape.
    pub fn try_clone(&self) -> Option<Self> {
        match self {
            #[cfg(has_metal)]
            GpuBuffer::Metal(buf) => Some(GpuBuffer::Metal(buf.clone())),
            #[cfg(has_webgpu)]
            GpuBuffer::Wgpu(_) => None, // WebGPU buffers can't be cheaply cloned.
        }
    }

    /// Read GPU data back to CPU. Caller must flush the backend first.
    ///
    /// For Metal (unified memory): zero-copy read from shared pointer.
    /// For WebGPU: staging buffer readback (caller must handle).
    pub fn readback_into(&self, dst: &mut [u8]) {
        match self {
            #[cfg(has_metal)]
            GpuBuffer::Metal(buf) => {
                let src = buf.contents() as *const u8;
                let len = (buf.length() as usize).min(dst.len());
                // SAFETY: Metal StorageModeShared buffers are CPU-readable
                // after the command buffer completes (flush).
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len);
                }
            }
            #[cfg(has_webgpu)]
            GpuBuffer::Wgpu(_) => {
                // WebGPU readback is async — not supported here.
                // Use flush_deferred() for WebGPU.
            }
        }
    }
}

/// A single input to a GPU kernel — either CPU bytes or a resident GPU buffer.
///
/// This enables GPU-to-GPU chaining: when a MatMul produces a `GpuBuffer`
/// and the next op is Add (which also has a GPU kernel), the Add receives
/// `GpuInput::Gpu` and passes the buffer directly to the GPU — no readback.
pub enum GpuInput<'a> {
    /// CPU byte slice (the common path, and fallback).
    Cpu(&'a [u8]),
    /// Reference to a resident GPU buffer from a prior dispatch.
    Gpu(&'a GpuBuffer),
}

impl<'a> GpuInput<'a> {
    /// Byte length of the input data.
    pub fn byte_len(&self) -> usize {
        match self {
            GpuInput::Cpu(s) => s.len(),
            GpuInput::Gpu(b) => b.byte_len(),
        }
    }
}

/// Result of a backend kernel dispatch.
///
/// Tells the tape executor HOW to store the result:
/// - `Skipped`: backend didn't handle this op → fall back to CPU
/// - `Bytes`: result written to `out_buf` (CPU path, or GPU→copy path)
/// - `GpuBuffer`: result stored in a GPU buffer for deferred readback
pub enum KernelOutput {
    /// Backend did not handle this op. Fall back to CPU dispatch.
    Skipped,
    /// Result written to the provided `out_buf`. Store via swap_insert.
    Bytes,
    /// Result stored in a backend-agnostic GPU buffer.
    /// The executor stores this for deferred readback or GPU-to-GPU chaining.
    GpuBuffer(GpuBuffer),
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

    /// Extract output bytes — from out_buf for Bytes, from GPU buffer for GpuBuffer.
    /// For testing: copies GPU buffer contents to Vec.
    #[cfg(test)]
    pub fn extract_bytes(self, out_buf: crate::buffer::OutputBuffer) -> Vec<u8> {
        match self {
            KernelOutput::Skipped => Vec::new(),
            KernelOutput::Bytes => out_buf.into_vec(),
            KernelOutput::GpuBuffer(buf) => {
                let len = buf.byte_len();
                let mut dst = vec![0u8; len];
                buf.readback_into(&mut dst);
                dst
            }
        }
    }
}

/// Shape + broadcast flags for [`ComputeBackend::dispatch_batched_matmul`].
///
/// Grouped into a builder so the trait method doesn't take 7+ positional
/// arguments — per the project rule forbidding
/// `#[allow(clippy::too_many_arguments)]`. Construct with
/// [`BatchedMatmulDims::new`] and chain [`Self::with_b_broadcast`] to enable
/// single-weight broadcast across the batch dimension.
#[derive(Debug, Clone, Copy)]
pub struct BatchedMatmulDims {
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// If true, the `b` operand is a single 2-D weight matrix broadcast across
    /// every batch slice (e.g. weight-sharing MatMul). If false, `b` has its
    /// own per-batch slice.
    pub b_broadcast: bool,
}

impl BatchedMatmulDims {
    /// New dims with `b_broadcast = false` (independent per-batch B slices).
    #[inline]
    #[must_use]
    pub fn new(batch: usize, m: usize, k: usize, n: usize) -> Self {
        Self {
            batch,
            m,
            k,
            n,
            b_broadcast: false,
        }
    }

    /// Enable broadcast of the B operand across the batch dimension.
    #[inline]
    #[must_use]
    pub fn with_b_broadcast(mut self, b_broadcast: bool) -> Self {
        self.b_broadcast = b_broadcast;
        self
    }
}

/// Parameters for Conv2d GPU dispatch.
///
/// Bundled into a struct to avoid `clippy::too_many_arguments`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dParams {
    pub ic: usize,
    pub h: usize,
    pub w: usize,
    pub oc: usize,
    pub kh: usize,
    pub kw: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dil_h: usize,
    pub dil_w: usize,
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
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput>;

    /// Dispatch a matmul (M×K × K×N). Writes to `out_buf` or returns a Metal buffer.
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput>;

    /// Dispatch a batched matmul (batch × M×K × K×N).
    /// Default: returns Skipped (falls back to per-batch CPU dispatch).
    fn dispatch_batched_matmul(
        &self,
        _inputs: &[&[u8]],
        _dims: BatchedMatmulDims,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        Ok(KernelOutput::Skipped)
    }

    /// Dispatch a float op with GPU-chained inputs.
    ///
    /// Accepts `GpuInput` (either CPU bytes or a resident GPU buffer).
    /// The default implementation reads back any GPU inputs to CPU bytes
    /// and delegates to `dispatch_float`. GPU backends override this to
    /// accept `GpuBuffer` inputs directly, avoiding CPU roundtrips.
    fn dispatch_float_chained(
        &self,
        op: &FloatOp,
        inputs: &[GpuInput<'_>],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        // Default: readback GPU inputs, then delegate to &[u8] path.
        let cpu_bufs: smallvec::SmallVec<[Vec<u8>; 4]> = inputs
            .iter()
            .map(|inp| match inp {
                GpuInput::Cpu(s) => s.to_vec(),
                GpuInput::Gpu(gb) => {
                    self.flush();
                    let mut dst = vec![0u8; gb.byte_len()];
                    gb.readback_into(&mut dst);
                    dst
                }
            })
            .collect();
        let refs: smallvec::SmallVec<[&[u8]; 4]> = cpu_bufs.iter().map(|v| v.as_slice()).collect();
        self.dispatch_float(op, &refs, out_buf)
    }

    /// Dispatch a matmul with GPU-chained inputs.
    ///
    /// Same as `dispatch_float_chained` but for matmul ops.
    fn dispatch_matmul_chained(
        &self,
        inputs: &[GpuInput<'_>],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        let cpu_bufs: smallvec::SmallVec<[Vec<u8>; 4]> = inputs
            .iter()
            .map(|inp| match inp {
                GpuInput::Cpu(s) => s.to_vec(),
                GpuInput::Gpu(gb) => {
                    self.flush();
                    let mut dst = vec![0u8; gb.byte_len()];
                    gb.readback_into(&mut dst);
                    dst
                }
            })
            .collect();
        let refs: smallvec::SmallVec<[&[u8]; 4]> = cpu_bufs.iter().map(|v| v.as_slice()).collect();
        self.dispatch_matmul(&refs, m, k, n, out_buf)
    }

    /// Dispatch a Conv2d with GPU-chained inputs.
    ///
    /// Default: returns Skipped (falls back to CPU conv dispatch).
    /// Metal backend overrides with im2col + SGEMM on GPU.
    fn dispatch_conv2d_chained(
        &self,
        _inputs: &[GpuInput<'_>],
        _params: &Conv2dParams,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        Ok(KernelOutput::Skipped)
    }

    /// Dispatch a slice (contiguous sub-range copy) on GPU.
    ///
    /// Default: returns Skipped. Metal backend uses slice_copy kernel.
    fn dispatch_slice_chained(
        &self,
        _input: &GpuInput<'_>,
        _src_offset_floats: usize,
        _count_floats: usize,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        Ok(KernelOutput::Skipped)
    }

    /// Dispatch a concat (combine two buffers) on GPU.
    ///
    /// Default: returns Skipped. Metal backend uses concat_copy kernel.
    fn dispatch_concat_chained(
        &self,
        _inputs: &[GpuInput<'_>],
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        Ok(KernelOutput::Skipped)
    }

    /// Dispatch a 4D transpose with GPU-chained inputs.
    ///
    /// Default: returns Skipped. Metal backend overrides with transpose_4d kernel.
    fn dispatch_transpose_chained(
        &self,
        _input: &GpuInput<'_>,
        _shape: [u32; 4],
        _perm: [u32; 4],
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        Ok(KernelOutput::Skipped)
    }

    /// Backend name for diagnostics and logging.
    fn name(&self) -> &'static str;

    /// Flush pending GPU work (commit + wait for all encoded commands).
    ///
    /// Called at level boundaries by the tape executor. After flush,
    /// all MetalBuffers returned by previous dispatch calls contain
    /// valid GPU-written data. No-op for CPU backends.
    fn flush(&self) {}

    /// Per-op-category minimum byte thresholds for GPU dispatch.
    ///
    /// Override in GPU backends to return hardware-detected thresholds.
    /// Default returns conservative thresholds (legacy 4MB behavior).
    fn op_thresholds(&self) -> &hardware::OpThresholds {
        &hardware::OpThresholds::DEFAULT
    }

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
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_float(op, inputs, out_buf)
    }
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_matmul(inputs, m, k, n, out_buf)
    }
    fn dispatch_batched_matmul(
        &self,
        inputs: &[&[u8]],
        dims: BatchedMatmulDims,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_batched_matmul(inputs, dims, out_buf)
    }
    fn dispatch_float_chained(
        &self,
        op: &FloatOp,
        inputs: &[GpuInput<'_>],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_float_chained(op, inputs, out_buf)
    }
    fn dispatch_matmul_chained(
        &self,
        inputs: &[GpuInput<'_>],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_matmul_chained(inputs, m, k, n, out_buf)
    }
    fn dispatch_conv2d_chained(
        &self,
        inputs: &[GpuInput<'_>],
        params: &Conv2dParams,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_conv2d_chained(inputs, params, out_buf)
    }
    fn dispatch_transpose_chained(
        &self,
        input: &GpuInput<'_>,
        shape: [u32; 4],
        perm: [u32; 4],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0
            .dispatch_transpose_chained(input, shape, perm, out_buf)
    }
    fn dispatch_slice_chained(
        &self,
        input: &GpuInput<'_>,
        src_offset_floats: usize,
        count_floats: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0
            .dispatch_slice_chained(input, src_offset_floats, count_floats, out_buf)
    }
    fn dispatch_concat_chained(
        &self,
        inputs: &[GpuInput<'_>],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_concat_chained(inputs, out_buf)
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
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_float(op, inputs, out_buf)
    }
    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_matmul(inputs, m, k, n, out_buf)
    }
    fn dispatch_batched_matmul(
        &self,
        inputs: &[&[u8]],
        dims: BatchedMatmulDims,
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<KernelOutput> {
        self.0.dispatch_batched_matmul(inputs, dims, out_buf)
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
    #[allow(unused_mut)]
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
        for v in a_floats[..k].iter_mut() {
            *v = 1.0;
        }
        let a: Vec<u8> = bytemuck::cast_slice(&a_floats).to_vec();

        // B = all 2.0s
        let b_data: Vec<f32> = vec![2.0; k * n];
        let b_bytes: Vec<u8> = bytemuck::cast_slice(&b_data).to_vec();

        let inputs: Vec<&[u8]> = vec![&a, &b_bytes];
        let mut out_buf = OutputBuffer::new();
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
        let expected = 2.0 * k as f32;
        for (j, &val) in out_floats[..n].iter().enumerate() {
            assert!(
                (val - expected).abs() < 0.1,
                "matmul C[0,{j}]: got {val}, expected {expected}",
            );
        }
        // Second row should be all 0s (A[1,:] = 0)
        for (j, &val) in out_floats[n..2 * n].iter().enumerate() {
            assert!(val.abs() < 0.1, "matmul C[1,{j}]: got {val}, expected 0",);
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

        let mut out_buf = OutputBuffer::new();
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

        let mut out_buf = OutputBuffer::new();
        let result = b
            .dispatch_float(&FloatOp::Relu, &inputs, &mut out_buf)
            .expect("Metal dispatch failed");
        assert!(result.handled(), "Metal should handle Relu on 6MB buffer");
        b.flush(); // Commit batched GPU work before reading output.

        // Extract output bytes — may be in out_buf (Bytes) or Metal buffer.
        let output_bytes: Vec<u8> = match result {
            KernelOutput::Bytes => out_buf.into_vec(),
            KernelOutput::GpuBuffer(gbuf) => {
                let len = gbuf.byte_len();
                let mut dst = vec![0u8; len];
                gbuf.readback_into(&mut dst);
                dst
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
