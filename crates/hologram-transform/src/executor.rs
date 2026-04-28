//! Plan executor.
//!
//! `Executor::run_forward` and `Executor::run_backward` walk a
//! `Box<[KernelCall]>` slice and delegate each call to a
//! [`CanonicalBackend`]. The default API uses the in-process
//! [`CpuBackend`]; pluggable variants (`run_forward_with`,
//! `run_backward_with`) accept any backend, which is how the Metal /
//! WebGPU / Atlas executors plug in (Phase 3.5).
//!
//! The executor itself never allocates, never traverses a graph, and
//! never selects algorithms — those decisions live in the planner.
//! Backends own their own device state; this module only owns the
//! per-call loop.

use crate::backend::{CanonicalBackend, CpuBackend};
use crate::buffer::BufferSet;
use crate::error::ExecError;
use crate::plan::CompiledPlan;

/// Stateless plan executor.
pub struct Executor;

impl Executor {
    /// Run the forward pass on the reference CPU backend.
    pub fn run_forward(plan: &CompiledPlan, buffers: &mut BufferSet) -> Result<(), ExecError> {
        Self::run_forward_with(plan, buffers, &mut CpuBackend::new())
    }

    /// Run the backward pass on the reference CPU backend.
    ///
    /// `plan.backward` is already in execution order (planner reversed
    /// it during lowering). The executor never re-derives the order.
    pub fn run_backward(plan: &CompiledPlan, buffers: &mut BufferSet) -> Result<(), ExecError> {
        Self::run_backward_with(plan, buffers, &mut CpuBackend::new())
    }

    /// Run the forward pass on a caller-supplied backend.
    ///
    /// The backend may batch calls internally (e.g. record a Metal /
    /// WebGPU command buffer); [`CanonicalBackend::flush`] is invoked
    /// once at the end so device-side work completes before the call
    /// returns.
    pub fn run_forward_with<B: CanonicalBackend>(
        plan: &CompiledPlan,
        buffers: &mut BufferSet,
        backend: &mut B,
    ) -> Result<(), ExecError> {
        buffers.check_fits(plan)?;
        backend.run(buffers.storage_mut(), &plan.forward)?;
        backend.flush()
    }

    /// Run the backward pass on a caller-supplied backend.
    pub fn run_backward_with<B: CanonicalBackend>(
        plan: &CompiledPlan,
        buffers: &mut BufferSet,
        backend: &mut B,
    ) -> Result<(), ExecError> {
        buffers.check_fits(plan)?;
        backend.run(buffers.storage_mut(), &plan.backward)?;
        backend.flush()
    }
}
