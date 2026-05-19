//! Canonical-layer device backends.
//!
//! Implementations of [`hologram_transform::CanonicalBackend`] that
//! dispatch [`hologram_transform::KernelCall`]s to a device. This is
//! the Phase 3.5 surface — distinct from the legacy
//! [`crate::ComputeBackend`] / [`hologram_core::FloatOp`] path.
//!
//! Each backend is feature-gated; missing variants return
//! [`ExecError::Backend`]. The conformance harness in
//! `hologram-transform::conformance` validates each variant against
//! the reference [`hologram_transform::CpuBackend`] as it comes
//! online.
//!
//! [`ExecError::Backend`]: hologram_transform::ExecError::Backend

#[cfg(feature = "webgpu")]
pub mod wgpu;

#[cfg(feature = "webgpu")]
pub use self::wgpu::WgpuBackend;
