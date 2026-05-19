//! WebGPU compute backend (browser + wgpu).
//!
//! Implements `WebGpuMemory` and `WebGpuBackend` for device-native execution
//! in browsers (via WebGPU API) and native apps (via wgpu). Uses WGSL shader
//! source for compute kernels.
//!
//! ## Async considerations
//!
//! WebGPU is inherently async — command buffer submission and buffer readback
//! require polling. On native (wgpu), polling is synchronous. On WASM, it
//! requires integration with the browser's event loop.
//!
//! The `ComputeBackend::flush()` method is synchronous. For WASM targets,
//! a future async variant will be needed.

// WebGpuMemory and WebGpuBackend will be implemented when the `webgpu`
// feature is enabled. The WGSL shader source for compute kernels will
// mirror the MSL kernels in kernels/metal.msl.
//
// Priority: after Metal backend is production-ready.
//
// Key differences from Metal:
// - WGSL shader language instead of MSL
// - Staging buffers for upload/download (no unified memory)
// - Async command buffer submission and readback
// - Buffer mapping API for CPU access
// - Bind group layout for kernel parameters
