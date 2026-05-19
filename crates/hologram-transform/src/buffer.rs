//! Buffer set: single contiguous allocation owned by the executor.
//!
//! `BufferSet` owns one `Box<[f32]>` sized by the planner. Tensors and
//! their gradients live as subslices of this single allocation. The
//! executor never allocates; kernels receive `&mut [f32]` slices.

use crate::address::TensorId;
use crate::error::ExecError;
use crate::plan::{CompiledPlan, SlotSpan};

/// Single owned workspace.
#[derive(Debug)]
pub struct BufferSet {
    storage: Box<[f32]>,
}

impl BufferSet {
    /// Allocate a buffer set sized for a plan's workspace.
    ///
    /// This *does* allocate — but it is intended to be called once, before
    /// any execution loop. The executor must never call this.
    #[must_use]
    pub fn for_plan(plan: &CompiledPlan) -> Self {
        let n = plan.workspace_elements();
        Self {
            storage: vec![0.0_f32; n].into_boxed_slice(),
        }
    }

    /// Capacity in elements.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Verify capacity matches the plan's workspace.
    #[inline]
    pub fn check_fits(&self, plan: &CompiledPlan) -> Result<(), ExecError> {
        let expected = plan.workspace_elements();
        if self.storage.len() != expected {
            return Err(ExecError::WorkspaceMismatch {
                expected,
                actual: self.storage.len(),
            });
        }
        Ok(())
    }

    /// Reset the entire buffer to zero (forward inputs + grads).
    #[inline]
    pub fn zero(&mut self) {
        self.storage.fill(0.0);
    }

    /// Write into a tensor slot, given its resolved span.
    #[inline]
    pub fn write_span(&mut self, span: SlotSpan, src: &[f32]) {
        debug_assert_eq!(span.len, src.len());
        self.storage[span.offset..span.offset + span.len].copy_from_slice(src);
    }

    /// Read a tensor slot.
    #[inline]
    #[must_use]
    pub fn read_span(&self, span: SlotSpan) -> &[f32] {
        &self.storage[span.offset..span.offset + span.len]
    }

    /// Write into a tensor slot using the plan's address table.
    #[inline]
    pub fn write_tensor(&mut self, plan: &CompiledPlan, id: TensorId, src: &[f32]) {
        self.write_span(plan.address_table.span(id), src);
    }

    /// Read a tensor slot using the plan's address table.
    #[inline]
    #[must_use]
    pub fn read_tensor<'a>(&'a self, plan: &CompiledPlan, id: TensorId) -> &'a [f32] {
        self.read_span(plan.address_table.span(id))
    }

    /// Read a tensor's gradient slot.
    #[inline]
    #[must_use]
    pub fn read_grad<'a>(&'a self, plan: &CompiledPlan, id: TensorId) -> &'a [f32] {
        self.read_span(plan.address_table.grad(id))
    }

    /// Write into a tensor's gradient slot. Used to seed `dC` before backward.
    #[inline]
    pub fn write_grad(&mut self, plan: &CompiledPlan, id: TensorId, src: &[f32]) {
        self.write_span(plan.address_table.grad(id), src);
    }

    /// Mutable raw storage — used by kernels via `Executor`.
    #[inline]
    pub(crate) fn storage_mut(&mut self) -> &mut [f32] {
        &mut self.storage
    }
}
