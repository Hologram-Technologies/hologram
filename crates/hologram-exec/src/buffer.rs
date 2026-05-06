//! Workspace buffer arena (spec VIII.3).
//!
//! Slots are pre-resolved at compile time from the graph's liveness
//! analysis. The arena performs no runtime allocation in steady state.

use hologram_backend::{Workspace, BufferRef};

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotSpan {
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Default)]
pub struct BufferArena {
    storage: Vec<u8>,
    slots: Vec<SlotSpan>,
}

impl BufferArena {
    pub fn new() -> Self { Self::default() }

    pub fn with_capacity(total_bytes: usize, slots: Vec<SlotSpan>) -> Self {
        Self {
            storage: vec![0u8; total_bytes],
            slots,
        }
    }

    pub fn slot(&self, idx: usize) -> Option<SlotSpan> {
        self.slots.get(idx).copied()
    }

    pub fn slot_count(&self) -> usize { self.slots.len() }

    pub fn capacity(&self) -> usize { self.storage.len() }

    pub fn read_slot(&self, idx: usize) -> Option<&[u8]> {
        let s = self.slots.get(idx)?;
        let start = s.offset as usize;
        let end = start + s.length as usize;
        self.storage.get(start..end)
    }

    pub fn write_slot(&mut self, idx: usize) -> Option<&mut [u8]> {
        let s = *self.slots.get(idx)?;
        let start = s.offset as usize;
        let end = start + s.length as usize;
        self.storage.get_mut(start..end)
    }
}

impl Workspace for BufferArena {
    /// Read up to `buf.length` bytes from `buf.slot`, starting at
    /// `buf.offset` *within* that slot. When `buf.length` is zero, return
    /// the slot's full contents — kernels that compute their own byte
    /// count from `element_count + dtype` can index into the returned
    /// slice without being constrained by the BufferRef's stale length.
    fn read(&self, buf: BufferRef) -> &[u8] {
        let slot = match self.slots.get(buf.slot as usize) {
            Some(s) => s,
            None => return &[],
        };
        let slot_start = slot.offset as usize;
        let slot_end = slot_start + slot.length as usize;
        if slot_end > self.storage.len() { return &[]; }
        let inner_start = slot_start + buf.offset as usize;
        let inner_end = if buf.length == 0 {
            slot_end
        } else {
            (inner_start + buf.length as usize).min(slot_end)
        };
        if inner_end > self.storage.len() || inner_start > inner_end { return &[]; }
        &self.storage[inner_start..inner_end]
    }

    fn write(&mut self, buf: BufferRef) -> &mut [u8] {
        let slot = match self.slots.get(buf.slot as usize) {
            Some(s) => *s,
            None => return &mut [],
        };
        let slot_start = slot.offset as usize;
        let slot_end = slot_start + slot.length as usize;
        if slot_end > self.storage.len() { return &mut []; }
        let inner_start = slot_start + buf.offset as usize;
        let inner_end = if buf.length == 0 {
            slot_end
        } else {
            (inner_start + buf.length as usize).min(slot_end)
        };
        if inner_end > self.storage.len() || inner_start > inner_end { return &mut []; }
        &mut self.storage[inner_start..inner_end]
    }
}

/// Caller-supplied input bytes (model input tensor body).
pub struct InputBuffer<'a> {
    pub bytes: &'a [u8],
}

/// Caller-receivable output buffer.
pub struct OutputBuffer {
    pub bytes: Vec<u8>,
}
