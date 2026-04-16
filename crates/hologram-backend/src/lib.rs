//! Device-native compute backends for hologram execution.
//!
//! This crate defines the `ComputeMemory` and `ComputeBackend` traits that
//! abstract tensor allocation and computation across devices (CPU, Metal GPU,
//! WebGPU). Every backend implements the full UOR computational model:
//!
//! - **Ring arithmetic** (Z/256Z, LUT-based transforms from uor-foundation)
//! - **Float ops** (matmul, conv2d, normalization, elementwise)
//! - **Data movement** (transpose, slice, concat, reshape)
//!
//! The core invariant: **all data lives on one device, all computation happens
//! on that device.** No CPU↔GPU transfers during execution. Weights and LUT
//! tables are uploaded to the device at initialization and stay resident.

pub mod cpu;

#[cfg(feature = "metal-backend")]
pub mod metal;

#[cfg(feature = "webgpu-backend")]
pub mod webgpu;

use hologram_core::op::FloatOp;

/// Error type for backend operations.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("device error: {0}")]
    Device(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// Manages tensor allocation on a specific device.
///
/// Every tensor buffer lives exclusively on the device managed by this
/// implementation. `upload` is called once at initialization (weights,
/// constants). `download` is called once at the end (output tensor).
/// Neither is called during the execution loop.
pub trait ComputeMemory: Send + Sync {
    /// Opaque buffer handle for this device.
    type Buffer: Send + Sync;

    /// Allocate an uninitialized buffer of `byte_len` bytes on this device.
    fn alloc(&self, byte_len: usize) -> Self::Buffer;

    /// Upload CPU bytes to a new device buffer.
    ///
    /// Called at initialization for weights and constants. The returned
    /// buffer lives on the device for the lifetime of execution.
    fn upload(&self, data: &[u8]) -> Self::Buffer;

    /// Download device buffer contents to CPU bytes.
    ///
    /// Called once at the end for output tensors. For GPU backends,
    /// the caller must flush pending work before downloading.
    fn download(&self, buf: &Self::Buffer) -> Vec<u8>;

    /// Zero-copy reshape: return a handle to the same underlying memory.
    ///
    /// The new handle shares the same device allocation — no data is
    /// copied. Shape metadata is tracked externally by the executor.
    fn alias(&self, buf: &Self::Buffer) -> Self::Buffer;

    /// Byte length of a buffer.
    fn byte_len(&self, buf: &Self::Buffer) -> usize;
}

/// Kernel parameters for dispatch.
///
/// Bundles the variable parameters that differ per-instruction (dimensions,
/// axis indices, epsilon values, etc.) without requiring per-op-type methods
/// on the trait. The backend extracts what it needs from this struct.
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Up to 8 u32 parameters (dimensions, sizes, axes, etc.).
    pub u32s: smallvec::SmallVec<[u32; 8]>,
    /// Up to 4 f32 parameters (epsilon, alpha, beta, etc.).
    pub f32s: smallvec::SmallVec<[f32; 4]>,
    /// Permutation array (for transpose).
    pub perm: [u8; 8],
    /// Number of valid perm entries.
    pub perm_len: u8,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            u32s: smallvec::SmallVec::new(),
            f32s: smallvec::SmallVec::new(),
            perm: [0, 1, 2, 3, 4, 5, 6, 7],
            perm_len: 0,
        }
    }
}

/// Dispatches tensor operations on a specific device.
///
/// Every backend implements the full UOR computational model. UOR LUT
/// tables are loaded via `load_ring_tables` at initialization and stay
/// resident on-device for all subsequent ring op dispatches.
///
/// The dispatch method handles ALL kernel types — there is no CPU fallback.
/// If a kernel is not supported, the backend returns an error (not a silent
/// skip). This ensures the caller knows immediately if coverage is incomplete.
pub trait ComputeBackend<M: ComputeMemory>: Send + Sync {
    /// Dispatch a kernel with device-native buffers.
    ///
    /// `inputs` are read-only references to device buffers.
    /// The result is written to `output` (pre-allocated by the executor).
    /// Returns the number of bytes written to `output`.
    fn dispatch(
        &self,
        op: &FloatOp,
        inputs: &[&M::Buffer],
        output: &mut M::Buffer,
        params: &KernelParams,
    ) -> Result<usize>;

    /// Dispatch a ring (byte-domain) operation using on-device LUT tables.
    fn dispatch_ring(
        &self,
        table_idx: usize,
        inputs: &[&M::Buffer],
        output: &mut M::Buffer,
    ) -> Result<usize>;

    /// Load UOR ring LUT tables onto the device.
    ///
    /// Called once at initialization. The tables stay on-device for all
    /// subsequent `dispatch_ring` calls.
    fn load_ring_tables(&mut self, tables: &[&[u8; 256]], memory: &M);

    /// Flush pending work.
    ///
    /// For GPU backends: commit the command buffer and wait for completion.
    /// For CPU: no-op.
    /// Called once at the end of execution, not per-op.
    fn flush(&self);

    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;
}
