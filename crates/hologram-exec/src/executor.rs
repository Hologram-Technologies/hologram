//! `Executor` (spec VIII.2).

use hologram_backend::{Backend, KernelCall};
use crate::buffer::BufferArena;
use crate::error::ExecError;

/// Drives a sequence of kernel calls through a backend's dispatcher.
///
/// The schedule (parallel-execution levels) is consumed by `InferenceSession`;
/// `Executor` itself is single-pass over a flat slice of calls.
pub struct Executor;

impl Executor {
    pub fn run<B: Backend<WS = BufferArena>>(
        backend: &mut B,
        calls: &[KernelCall],
        workspace: &mut BufferArena,
    ) -> Result<(), ExecError> {
        for call in calls {
            backend.dispatch(call, workspace).map_err(|_| ExecError::Backend)?;
        }
        Ok(())
    }
}
