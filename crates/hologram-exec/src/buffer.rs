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
    fn read(&self, buf: BufferRef) -> &[u8] {
        let start = buf.offset as usize;
        let end = start + buf.length as usize;
        match (self.slots.get(buf.slot as usize), end <= self.storage.len()) {
            (Some(_), true) => &self.storage[start..end],
            _ => &[],
        }
    }

    fn write(&mut self, buf: BufferRef) -> &mut [u8] {
        let start = buf.offset as usize;
        let end = start + buf.length as usize;
        if end <= self.storage.len() {
            &mut self.storage[start..end]
        } else {
            &mut []
        }
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
