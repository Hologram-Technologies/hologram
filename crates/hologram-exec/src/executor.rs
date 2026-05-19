//! `Executor` (spec VIII.2).
//!
//! Walks an exec plan (per-level kernel-call indices) through a backend's
//! dispatcher. Within a level, calls have no inter-dependencies by
//! construction (the topological-level schedule guarantees disjoint
//! input/output slots), so a backend that can parallelize is free to do
//! so. The CPU executor itself walks each level sequentially because
//! `Backend::dispatch` requires unique `&mut` access to the workspace —
//! a soundly-shared `&mut BufferArena` across threads would require
//! splitting the storage into disjoint per-call mutable slices, which
//! changes the `Workspace` trait surface and is out of scope here.
//!
//! GPU backends (wgpu / Metal) have their own command-queue scheduling;
//! the runtime hands them serialized `dispatch` calls and the queue
//! parallelizes internally. The `parallel` cargo feature is therefore
//! an architectural-commitment surface (level boundaries are visible to
//! a future per-slot-disaggregated executor) rather than a code switch.

use crate::buffer::BufferArena;
use crate::error::ExecError;
use hologram_backend::{Backend, KernelCall};

pub struct Executor;

impl Executor {
    /// Sequential walk over a flat call slice. Retained for tests and
    /// consumers that have no schedule.
    pub fn run<B: Backend<WS = BufferArena>>(
        backend: &mut B,
        calls: &[KernelCall],
        workspace: &mut BufferArena,
    ) -> Result<(), ExecError> {
        for call in calls {
            backend
                .dispatch(call, workspace)
                .map_err(|_| ExecError::Backend)?;
        }
        Ok(())
    }

    /// Schedule-aware walk (spec VIII.2). Levels execute in order; the
    /// runtime preserves level boundaries so a future per-slot-split
    /// executor can dispatch a level's calls in parallel against
    /// disjoint mutable workspace fragments. The current executor walks
    /// each level sequentially — `Backend::dispatch` requires unique
    /// `&mut Workspace` access, which is incompatible with sharing a
    /// single arena reference across threads even when the underlying
    /// byte writes are disjoint.
    pub fn run_levels<B: Backend<WS = BufferArena>>(
        backend: &mut B,
        calls: &[KernelCall],
        levels: &[Vec<u32>],
        workspace: &mut BufferArena,
    ) -> Result<(), ExecError> {
        for level in levels {
            for &call_idx in level {
                let idx = call_idx as usize;
                let call = calls.get(idx).ok_or(ExecError::Backend)?;
                backend
                    .dispatch(call, workspace)
                    .map_err(|_| ExecError::Backend)?;
            }
        }
        Ok(())
    }
}
