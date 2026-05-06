//! Per-target dispatch (spec Part IX).
//!
//! Each backend declares a `HostBounds` from `hologram-host` and a
//! `Backend` impl whose `dispatch` consumes a `KernelCall` and writes
//! into a runtime workspace. The hot loop holds zero virtual dispatch.

pub mod kernel_call;
pub mod backend;
pub mod workspace;
pub mod error;

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
