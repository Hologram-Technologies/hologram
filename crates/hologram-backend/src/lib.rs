//! Per-target dispatch (spec Part IX).
//!
//! Each backend declares a `HostBounds` from `hologram-host` and a
//! `Backend` impl whose `dispatch` consumes a `KernelCall` and writes
//! into a runtime workspace. The hot loop holds zero virtual dispatch.
//!
//! `no_std` + `alloc` by default (matching prism / uor-addr) so hologram-ai
//! runs in wasm and on embedded targets; the `std` feature adds host-only
//! backends and amenities (wgpu, runtime SIMD detection, thread-local
//! scratch).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod error;
pub mod kernel_call;
pub mod layout;
pub mod prism_axes;
pub mod workspace;

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

pub use backend::Backend;
pub use error::BackendError;
pub use kernel_call::*;
pub use workspace::{BufferRef, SplitReads, Workspace};

#[cfg(feature = "cpu")]
pub use cpu::CpuBackend;

// Prism-canonical axis impls: hologram's f32 CPU kernels reachable
// through the prism-tensor `TensorAxis` / `ActivationAxis` interface
// per wiki ADR-031.
pub use prism_axes::{
    HologramF32MatmulSquare, HologramF32Tensor16x16Matmul, HologramF32Tensor4x4Matmul,
    HologramF32Tensor8x8Matmul, HologramF32VectorActivation, HologramF32VectorActivation16,
    HologramF32VectorActivation256, HologramF32VectorActivation64, HOLOGRAM_MAX_ACTIVATION_LEN,
    HOLOGRAM_MAX_TENSOR_DIM,
};
