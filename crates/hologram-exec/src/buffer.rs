//! Workspace buffer arena (spec VIII.3, wiki ADR-018 zero-cost runtime
//! per TC-01).
//!
//! Slots are pre-resolved at compile time from the graph's liveness
//! analysis; the arena performs no runtime allocation in steady state.
//!
//! Performance levers driven by the prism cost-model:
//!
//! - **Aligned storage.** The backing `storage: Vec<u8>` is allocated
//!   on a 64-byte boundary via `AlignedBytes`. 64 bytes covers an
//!   x86-64 cache line, an AVX-512 ZMM register, and `bytemuck`'s
//!   alignment requirements for `&[u8] -> &[f32]` zero-copy casts.
//!   Per-slot offsets stay 4-byte-aligned by construction (sizes
//!   are multiples of `bytes_per_element` for every supported dtype
//!   except `DTYPE_I4`, whose sub-byte packing the kernels handle
//!   explicitly).
//!
//! - **Zero-copy split-borrow** (`split_borrow`). Returns disjoint
//!   `&[u8]` read slices alongside one `&mut [u8]` write slice from
//!   the same backing storage in a single API call, so kernel bodies
//!   skip the `.to_vec()` clones the previous one-borrow-at-a-time
//!   API forced. The split is `unsafe` internally but exposed safely
//!   under the invariant that the requested read slots are disjoint
//!   from the write slot — a property the schedule's per-level
//!   independence guarantees (spec VIII.2).

use core::ptr::NonNull;
use hologram_backend::{BufferRef, Workspace};

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotSpan {
    pub offset: u32,
    pub length: u32,
}

/// 64-byte-aligned byte buffer. The alignment buys:
/// - cache-line alignment on x86-64 (64-byte L1 line);
/// - AVX-512 ZMM-register alignment (512 bits = 64 bytes);
/// - safe `bytemuck::cast_slice::<u8, f32>` views without a
///   `PodCastError::TargetAlignmentGreaterAndInputNotAligned`.
#[derive(Debug)]
struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

const ARENA_ALIGN: usize = 64;

impl AlignedBytes {
    fn zeroed(len: usize) -> Self {
        // Round capacity up to the alignment so allocator behaviour is
        // predictable; minimum allocation is `ARENA_ALIGN` even for
        // zero-length arenas so the NonNull never aliases a dangling
        // address.
        let cap = len.max(ARENA_ALIGN).next_multiple_of(ARENA_ALIGN);
        // SAFETY: cap > 0 and the layout's alignment is a power of two.
        unsafe {
            let layout = core::alloc::Layout::from_size_align(cap, ARENA_ALIGN)
                .expect("layout: cap > 0 and align is a power of two");
            let raw = alloc::alloc::alloc_zeroed(layout);
            let ptr = NonNull::new(raw)
                .unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout));
            Self { ptr, len, cap }
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `cap >= len` bytes, and we expose
        // only the first `len`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for `cap >= len` bytes; `&mut self`
        // guarantees no aliasing references exist.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: ptr was produced by `alloc_zeroed` with this exact layout.
        unsafe {
            let layout = core::alloc::Layout::from_size_align(self.cap, ARENA_ALIGN)
                .expect("layout: cap > 0 and align is a power of two");
            alloc::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

impl Default for AlignedBytes {
    fn default() -> Self {
        Self::zeroed(0)
    }
}

// `AlignedBytes` owns the backing allocation exclusively; the raw ptr
// is reachable only through `&self` / `&mut self`, so the standard
// `Send + Sync` rules apply.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

extern crate alloc;

#[derive(Debug, Default)]
pub struct BufferArena {
    storage: AlignedBytes,
    slots: Vec<SlotSpan>,
}

impl BufferArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(total_bytes: usize, slots: Vec<SlotSpan>) -> Self {
        Self {
            storage: AlignedBytes::zeroed(total_bytes),
            slots,
        }
    }

    pub fn slot(&self, idx: usize) -> Option<SlotSpan> {
        self.slots.get(idx).copied()
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn capacity(&self) -> usize {
        self.storage.len
    }

    pub fn read_slot(&self, idx: usize) -> Option<&[u8]> {
        let s = self.slots.get(idx)?;
        let start = s.offset as usize;
        let end = start + s.length as usize;
        self.storage.as_slice().get(start..end)
    }

    pub fn write_slot(&mut self, idx: usize) -> Option<&mut [u8]> {
        let s = *self.slots.get(idx)?;
        let start = s.offset as usize;
        let end = start + s.length as usize;
        self.storage.as_mut_slice().get_mut(start..end)
    }

    /// Resolve a `BufferRef` to its `[start, end)` byte range within
    /// the arena's backing storage. Returns `None` on out-of-range.
    /// Used by `split_borrow` to compute the per-slot disjoint slices
    /// without intermediate allocation.
    #[inline]
    fn buf_range(&self, buf: BufferRef) -> Option<(usize, usize)> {
        let slot = self.slots.get(buf.slot as usize)?;
        let slot_start = slot.offset as usize;
        let slot_end = slot_start + slot.length as usize;
        if slot_end > self.storage.len {
            return None;
        }
        let inner_start = slot_start + buf.offset as usize;
        let inner_end = if buf.length == 0 {
            slot_end
        } else {
            (inner_start + buf.length as usize).min(slot_end)
        };
        if inner_end > self.storage.len || inner_start > inner_end {
            return None;
        }
        Some((inner_start, inner_end))
    }

    /// Zero-copy split-borrow: obtain `&[u8]` slices for each read
    /// buffer plus an `&mut [u8]` slice for the single write buffer,
    /// all backed by the same arena storage.
    ///
    /// Returns `None` if any buffer is out-of-range OR if the write
    /// range overlaps any read range. The caller is responsible for
    /// supplying disjoint buffer refs; the schedule's per-level
    /// independence (spec VIII.2) guarantees disjointness at the slot
    /// level, and within-slot writes don't overlap reads from the
    /// same slot in the executor's call sequence.
    ///
    /// Avoids the `.to_vec()` clones that the read-then-write
    /// `Workspace::read` / `Workspace::write` pair forced (the borrow
    /// checker can't see two non-overlapping ranges of a shared
    /// `Vec<u8>` as disjoint without a split).
    pub fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(Vec<&'a [u8]>, &'a mut [u8])> {
        let (w_start, w_end) = self.buf_range(write)?;
        // SAFETY: each read range is disjoint from `[w_start, w_end)` by
        // the explicit check below, and shared `&[u8]` aliasing among
        // reads is permitted. The mutable write slice has unique access
        // to its range. Lifetimes are tied to `&'a mut self`.
        let base_const = self.storage.as_slice().as_ptr();
        let base_mut = self.storage.as_mut_slice().as_mut_ptr();
        let mut read_slices: Vec<&'a [u8]> = Vec::with_capacity(reads.len());
        for r in reads {
            let (rs, re) = self.buf_range(*r)?;
            if rs < w_end && w_start < re {
                return None;
            }
            unsafe {
                read_slices.push(core::slice::from_raw_parts(base_const.add(rs), re - rs));
            }
        }
        let write_slice = unsafe {
            core::slice::from_raw_parts_mut(base_mut.add(w_start), w_end - w_start)
        };
        Some((read_slices, write_slice))
    }
}

impl Workspace for BufferArena {
    /// Read up to `buf.length` bytes from `buf.slot`, starting at
    /// `buf.offset` *within* that slot. When `buf.length` is zero, return
    /// the slot's full contents — kernels that compute their own byte
    /// count from `element_count + dtype` can index into the returned
    /// slice without being constrained by the BufferRef's stale length.
    fn read(&self, buf: BufferRef) -> &[u8] {
        match self.buf_range(buf) {
            Some((s, e)) => &self.storage.as_slice()[s..e],
            None => &[],
        }
    }

    fn write(&mut self, buf: BufferRef) -> &mut [u8] {
        match self.buf_range(buf) {
            Some((s, e)) => &mut self.storage.as_mut_slice()[s..e],
            None => &mut [],
        }
    }

    #[inline]
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(Vec<&'a [u8]>, &'a mut [u8])> {
        BufferArena::split_borrow(self, reads, write)
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
