//! Per-target dispatch (spec Part IX).
//!
//! Each backend declares a `HostBounds` from `hologram-host` and a
//! `Backend` impl whose `dispatch` consumes a `KernelCall` and writes
//! into a runtime workspace. The hot loop holds zero virtual dispatch.

pub mod kernel_call;
pub mod backend;
pub mod workspace;
pub mod error;
pub mod prism_axes;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_backend;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal_backend::MetalBackend;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;
#[cfg(feature = "wgpu")]
pub use wgpu_backend::WgpuBackend;

pub use kernel_call::*;
pub use backend::Backend;
pub use workspace::{Workspace, BufferRef};
pub use error::BackendError;

#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;

// Prism-canonical axis impls: hologram's f32 CPU kernels reachable
// through the prism-tensor `TensorAxis` / `ActivationAxis` interface
// per wiki ADR-031.
pub use prism_axes::{
    HologramF32MatmulSquare,
    HologramF32Tensor4x4Matmul, HologramF32Tensor8x8Matmul, HologramF32Tensor16x16Matmul,
    HologramF32VectorActivation,
    HologramF32VectorActivation16, HologramF32VectorActivation64, HologramF32VectorActivation256,
    HOLOGRAM_MAX_TENSOR_DIM, HOLOGRAM_MAX_ACTIVATION_LEN,
};
