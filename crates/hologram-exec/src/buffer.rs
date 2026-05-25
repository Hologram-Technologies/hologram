//! Workspace buffer pool (spec VIII.3, wiki ADR-018 zero-cost runtime per
//! TC-01) — the UOR content-addressed execution substrate.
//!
//! Each value lives in its **own** 64-byte-aligned buffer; a slot is a
//! *binding* to one of those buffers, not a copy of it. This is what makes
//! the runtime **zero-movement**: a value is written once by the kernel
//! that produces it and thereafter referred to by binding a slot to it —
//! reuse points at the existing buffer (no copy-back), retention keeps the
//! buffer keyed by its κ-label (no copy-out). The legacy design copied
//! tensors between a fixed byte arena and a separate content store on every
//! node; that movement is gone.
//!
//! Two buffer classes, byte-bounded so memory holds for arbitrary models
//! and run lengths (ADR-060, SC-3):
//!
//! * **pinned** — model constants/weights, deduped by content κ-label,
//!   resident for the session (the model's inherent footprint);
//! * **transient** — boundary inputs, intermediates, outputs — held in a
//!   two-generation pool: a value retained past `budget` bytes ages a whole
//!   generation out (recompute on a later miss, never a wrong answer).
//!
//! Alignment: every buffer is 64-byte aligned (x86-64 cache line, AVX-512
//! ZMM, `bytemuck::cast_slice::<u8,f32>` zero-copy).

extern crate alloc;

use alloc::vec::Vec;
use core::ptr::NonNull;

use hashbrown::HashMap;
use hologram_archive::ContentLabel;
use hologram_backend::{BufferRef, SplitReads, Workspace};

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotSpan {
    /// Byte offset (retained for API compatibility; per-slot buffers start
    /// at 0, so this is informational only).
    pub offset: u64,
    /// Byte length of the slot.
    pub length: u64,
}

/// Default transient byte budget per generation (256 MiB). Resident
/// transient bytes are bounded by `2 * budget`; pinned constants are
/// separate and model-bounded.
pub const DEFAULT_POOL_BUDGET: usize = 256 << 20;

const ARENA_ALIGN: usize = 64;

/// 64-byte-aligned owned byte buffer.
#[derive(Debug)]
struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

impl AlignedBytes {
    fn zeroed(len: usize) -> Self {
        let cap = len.max(ARENA_ALIGN).next_multiple_of(ARENA_ALIGN);
        // SAFETY: cap > 0 and ARENA_ALIGN is a power of two.
        unsafe {
            let layout = core::alloc::Layout::from_size_align(cap, ARENA_ALIGN)
                .expect("layout: cap > 0 and align is a power of two");
            let raw = alloc::alloc::alloc_zeroed(layout);
            let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout));
            Self { ptr, len, cap }
        }
    }

    /// Reuse this buffer for a value of `len` bytes, reallocating only if
    /// the existing capacity is too small. Zero-fills so a kernel that
    /// writes only a logical prefix leaves a deterministic tail.
    fn reset_to(&mut self, len: usize) {
        if len > self.cap {
            *self = Self::zeroed(len);
        } else {
            self.len = len;
            // SAFETY: ptr valid for cap >= len bytes.
            unsafe {
                core::ptr::write_bytes(self.ptr.as_ptr(), 0, len);
            }
        }
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr valid for `cap >= len` bytes; expose only `len`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr valid for `cap >= len`; `&mut self` ⇒ no aliasing.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: ptr came from `alloc_zeroed` with this exact layout.
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

// `AlignedBytes` owns its allocation exclusively; the raw ptr is reachable
// only through `&self`/`&mut self`, so the standard Send+Sync rules apply.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

/// Content-addressed buffer pool + slot binding table. Implements
/// [`Workspace`] so kernels are unchanged: a `BufferRef`'s `slot` resolves
/// through the binding table to the buffer currently bound there.
#[derive(Debug, Default)]
pub struct BufferArena {
    /// Backing storage, stable index. Freed indices are recycled.
    bufs: Vec<AlignedBytes>,
    free: Vec<usize>,
    /// `slot_buf[slot]` = index into `bufs` currently bound to `slot`
    /// (`usize::MAX` = unbound). Kernel I/O resolves through this.
    slot_buf: Vec<usize>,
    /// Declared slot lengths (informational + the fixed-arena API).
    slot_len: Vec<usize>,
    /// `slot_off[slot]` = byte offset into `bufs[slot_buf[slot]]` at which this
    /// slot's data begins. 0 for an ordinary whole-buffer binding; non-zero
    /// only for a **view** slot (a zero-movement `ProjectField`/Slice that
    /// aliases a sub-region of a parent buffer). Reset to 0 each walk.
    slot_off: Vec<usize>,
    // Content-addressed residency (label → bufs index), byte-bounded.
    pinned: HashMap<ContentLabel, usize>,
    current: HashMap<ContentLabel, usize>,
    previous: HashMap<ContentLabel, usize>,
    current_bytes: usize,
    budget: usize,
}

const UNBOUND: usize = usize::MAX;

impl BufferArena {
    pub fn new() -> Self {
        Self {
            budget: DEFAULT_POOL_BUDGET,
            ..Self::default()
        }
    }

    /// Construct a fixed arena: one buffer per slot, each bound to itself.
    /// `total_bytes` is informational (each slot owns its own buffer).
    /// This preserves the legacy fixed-slot construction used by direct
    /// `Workspace` consumers (kernels, microbenches).
    pub fn with_capacity(_total_bytes: usize, slots: Vec<SlotSpan>) -> Self {
        let n = slots.len();
        let mut bufs = Vec::with_capacity(n);
        let mut slot_buf = Vec::with_capacity(n);
        let mut slot_len = Vec::with_capacity(n);
        for (i, s) in slots.iter().enumerate() {
            bufs.push(AlignedBytes::zeroed(s.length as usize));
            slot_buf.push(i);
            slot_len.push(s.length as usize);
        }
        Self {
            bufs,
            free: Vec::new(),
            slot_buf,
            slot_off: vec![0; n],
            slot_len,
            pinned: HashMap::new(),
            current: HashMap::new(),
            previous: HashMap::new(),
            current_bytes: 0,
            budget: DEFAULT_POOL_BUDGET,
        }
    }

    pub fn slot(&self, idx: usize) -> Option<SlotSpan> {
        self.slot_len.get(idx).map(|&length| SlotSpan {
            offset: 0,
            length: length as u64,
        })
    }

    pub fn slot_count(&self) -> usize {
        self.slot_len.len()
    }

    /// Total live buffer bytes (informational).
    pub fn capacity(&self) -> usize {
        self.bufs.iter().map(|b| b.len).sum()
    }

    /// The buffer index currently bound to `slot`, if any.
    #[inline]
    fn bound(&self, slot: usize) -> Option<usize> {
        self.slot_buf.get(slot).copied().filter(|&i| i != UNBOUND)
    }

    pub fn read_slot(&self, idx: usize) -> Option<&[u8]> {
        let bi = self.bound(idx)?;
        let off = self.slot_off.get(idx).copied().unwrap_or(0);
        let buf = self.bufs[bi].as_slice();
        // A view slot exposes only its declared sub-region [off, off+len).
        let end = self
            .slot_len
            .get(idx)
            .copied()
            .filter(|&l| l > 0)
            .map_or(buf.len(), |l| (off + l).min(buf.len()));
        buf.get(off..end)
    }

    pub fn write_slot(&mut self, idx: usize) -> Option<&mut [u8]> {
        let bi = self.bound(idx)?;
        let off = self.slot_off.get(idx).copied().unwrap_or(0);
        Some(&mut self.bufs[bi].as_mut_slice()[off..])
    }

    /// Bind `slot` as a **view** onto the buffer backing `parent` at an extra
    /// `byte_offset` (composing with any offset the parent itself carries),
    /// exposing `byte_len` bytes. Zero-movement: no allocation, no copy — the
    /// UOR `ProjectField`/Slice realization. The parent buffer must outlive the
    /// view's use within the walk (it does: the producer is bound this walk).
    pub fn bind_view(&mut self, slot: usize, parent: usize, byte_offset: usize, byte_len: usize) {
        let Some(pbi) = self.bound(parent) else {
            return;
        };
        let poff = self.slot_off.get(parent).copied().unwrap_or(0);
        self.ensure_slot(slot);
        self.slot_buf[slot] = pbi;
        self.slot_off[slot] = poff + byte_offset;
        self.slot_len[slot] = byte_len;
    }

    /// Resolve a `BufferRef` to `(bufs index, start, end)` within the bound
    /// buffer, honoring any view offset on the slot. `None` if unbound/oob.
    #[inline]
    fn buf_range(&self, buf: BufferRef) -> Option<(usize, usize, usize)> {
        let slot = buf.slot as usize;
        let bi = self.bound(slot)?;
        let len = self.bufs[bi].len;
        let base = self.slot_off.get(slot).copied().unwrap_or(0);
        let start = (base + buf.offset as usize).min(len);
        let end = if buf.length == 0 {
            // Full extent of this slot's view (or whole buffer if not a view).
            self.slot_len
                .get(slot)
                .copied()
                .filter(|&l| l > 0)
                .map_or(len, |l| (base + l).min(len))
        } else {
            (start + buf.length as usize).min(len)
        };
        if start > end {
            return None;
        }
        Some((bi, start, end))
    }
}

// ─── Content-addressed pool operations (driven by the executor) ──────────

impl BufferArena {
    /// Set the transient byte budget per generation.
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget.max(1);
    }

    /// Reset the slot→buffer binding table to `n` unbound slots, keeping
    /// the resident buffer pool intact. Called at the start of a walk.
    pub fn rebind_reset(&mut self, n: usize) {
        self.slot_buf.clear();
        self.slot_buf.resize(n, UNBOUND);
        // Views are per-walk; clear all offsets so a recycled slot is a plain
        // whole-buffer binding unless `bind_view` sets it again this walk.
        self.slot_off.clear();
        self.slot_off.resize(n, 0);
        if self.slot_len.len() < n {
            self.slot_len.resize(n, 0);
        }
    }

    fn ensure_slot(&mut self, slot: usize) {
        if slot >= self.slot_buf.len() {
            self.slot_buf.resize(slot + 1, UNBOUND);
        }
        if slot >= self.slot_off.len() {
            self.slot_off.resize(slot + 1, 0);
        }
        if slot >= self.slot_len.len() {
            self.slot_len.resize(slot + 1, 0);
        }
    }

    /// Allocate (or recycle from the free list) a `len`-byte buffer and bind
    /// it to `slot`. Used both for node outputs (a reuse/memo miss) and as
    /// the backing for a freshly-interned/pinned value.
    pub fn bind_fresh(&mut self, slot: usize, len: usize) -> usize {
        let bi = match self.free.pop() {
            Some(bi) => {
                self.bufs[bi].reset_to(len);
                bi
            }
            None => {
                self.bufs.push(AlignedBytes::zeroed(len));
                self.bufs.len() - 1
            }
        };
        self.ensure_slot(slot);
        self.slot_buf[slot] = bi;
        self.slot_len[slot] = len;
        bi
    }

    /// Bind `slot` to the buffer holding `label`, if resident. Returns true
    /// on a hit (the value is now readable at `slot` with **no copy**).
    pub fn bind_resident(&mut self, slot: usize, label: &ContentLabel) -> bool {
        let bi = self
            .pinned
            .get(label)
            .or_else(|| self.current.get(label))
            .or_else(|| self.previous.get(label))
            .copied();
        match bi {
            Some(bi) => {
                self.ensure_slot(slot);
                self.slot_buf[slot] = bi;
                true
            }
            None => false,
        }
    }

    /// Pin a constant/weight by content label (resident for the session,
    /// deduped — identical weights share one buffer). No slot binding; the
    /// walk re-binds pinned constants by label each run. One-time load copy.
    pub fn pin_bytes(&mut self, label: ContentLabel, bytes: &[u8]) {
        if self.pinned.contains_key(&label) {
            return;
        }
        // Exact logical length (the allocator pads capacity to 64 for
        // alignment); `resolve` returns exactly these bytes.
        let mut buf = AlignedBytes::zeroed(bytes.len());
        buf.as_mut_slice().copy_from_slice(bytes);
        self.bufs.push(buf);
        let bi = self.bufs.len() - 1;
        self.pinned.insert(label, bi);
    }

    /// Store arbitrary bytes as a transient value addressed by `label`,
    /// without binding a slot. The byte→address boundary for inputs
    /// pre-interned ahead of `execute_addressed`.
    pub fn store_unbound(&mut self, label: ContentLabel, bytes: &[u8]) {
        if self.resident(&label) {
            return;
        }
        self.roll_if_needed(bytes.len());
        let bi = match self.free.pop() {
            Some(bi) => {
                self.bufs[bi].reset_to(bytes.len());
                bi
            }
            None => {
                self.bufs.push(AlignedBytes::zeroed(bytes.len()));
                self.bufs.len() - 1
            }
        };
        self.bufs[bi].as_mut_slice().copy_from_slice(bytes);
        self.current_bytes = self.current_bytes.saturating_add(bytes.len());
        self.current.insert(label, bi);
    }

    /// Address the value currently bound to `slot` by `label` and retain it
    /// in the transient pool — **no copy**, just records the binding's
    /// buffer under the label. Subsequent identical derivations bind to it.
    pub fn retain(&mut self, slot: usize, label: ContentLabel) {
        let bi = match self.bound(slot) {
            Some(bi) => bi,
            None => return,
        };
        if self.pinned.contains_key(&label) || self.current.contains_key(&label) {
            return;
        }
        let len = self.bufs[bi].len;
        self.roll_if_needed(len);
        self.current_bytes = self.current_bytes.saturating_add(len);
        self.current.insert(label, bi);
    }

    /// Whether a value with this label is resident (pinned or transient).
    pub fn resident(&self, label: &ContentLabel) -> bool {
        self.pinned.contains_key(label)
            || self.current.contains_key(label)
            || self.previous.contains_key(label)
    }

    /// Resolve a label to its bytes, if resident (the address→byte boundary).
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        let bi = self
            .pinned
            .get(label)
            .or_else(|| self.current.get(label))
            .or_else(|| self.previous.get(label))
            .copied()?;
        Some(self.bufs[bi].as_slice())
    }

    /// Distinct resident values across both tiers (observability).
    pub fn resident_len(&self) -> usize {
        self.pinned.len() + self.current.len() + self.previous.len()
    }

    /// Resident transient bytes (both generations, deduped) — SC-3 bound.
    pub fn transient_bytes(&self) -> usize {
        let mut seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
        let mut total = 0;
        for &bi in self.current.values().chain(self.previous.values()) {
            if seen.insert(bi) {
                total += self.bufs[bi].len;
            }
        }
        total
    }

    /// Age `current` → `previous` (recycling the dropped generation's
    /// buffers) when adding `incoming` bytes would exceed the budget, so
    /// resident transient bytes stay ≤ `2 * budget` for any run length.
    fn roll_if_needed(&mut self, incoming: usize) {
        if self.current_bytes.saturating_add(incoming) <= self.budget || self.current.is_empty() {
            return;
        }
        // Recycle the outgoing `previous` generation's buffers (deduped;
        // only those no longer referenced by a pinned/current label or a
        // live slot binding).
        let dropped = core::mem::take(&mut self.previous);
        let mut freed: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
        for (_, bi) in dropped {
            if freed.insert(bi) && !self.bufs_index_live(bi) {
                self.free.push(bi);
            }
        }
        core::mem::swap(&mut self.current, &mut self.previous);
        self.current.clear();
        self.current_bytes = 0;
    }

    /// Is buffer index `bi` referenced by a pinned/current label or a live
    /// slot binding? (Then it must not be recycled.)
    fn bufs_index_live(&self, bi: usize) -> bool {
        self.pinned.values().any(|&v| v == bi)
            || self.current.values().any(|&v| v == bi)
            || self.slot_buf.contains(&bi)
    }
}

impl Workspace for BufferArena {
    fn read(&self, buf: BufferRef) -> &[u8] {
        match self.buf_range(buf) {
            Some((bi, s, e)) => &self.bufs[bi].as_slice()[s..e],
            None => &[],
        }
    }

    fn write(&mut self, buf: BufferRef) -> &mut [u8] {
        match self.buf_range(buf) {
            Some((bi, s, e)) => &mut self.bufs[bi].as_mut_slice()[s..e],
            None => &mut [],
        }
    }

    /// Zero-copy split-borrow across the bound buffers: `&[u8]` for each
    /// read, one `&mut [u8]` for the write. Disjoint by construction —
    /// distinct values live in distinct allocations; only a read aliasing
    /// the write *buffer* with an overlapping range is rejected.
    fn split_borrow<'a>(
        &'a mut self,
        reads: &[BufferRef],
        write: BufferRef,
    ) -> Option<(SplitReads<'a>, &'a mut [u8])> {
        let (wb, ws, we) = self.buf_range(write)?;
        // Raw data pointer of the write buffer (NonNull is Copy; reading it
        // is a shared borrow that ends immediately).
        let w_ptr = self.bufs[wb].ptr.as_ptr();
        let mut read_slices: SplitReads<'a> = SplitReads::new();
        for r in reads {
            let (rb, rs, re) = self.buf_range(*r)?;
            if rb == wb && rs < we && ws < re {
                return None; // overlapping in-place read/write
            }
            let r_ptr = self.bufs[rb].ptr.as_ptr();
            // SAFETY: `rb != wb` ⇒ distinct allocations; `rb == wb` only
            // reaches here with non-overlapping ranges. Lifetimes tie to
            // `&'a mut self`, which forbids other access for `'a`.
            read_slices.push(unsafe { core::slice::from_raw_parts(r_ptr.add(rs), re - rs) });
        }
        // SAFETY: the write range is disjoint from every read range above.
        let write_slice = unsafe { core::slice::from_raw_parts_mut(w_ptr.add(ws), we - ws) };
        Some((read_slices, write_slice))
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

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_archive::address_bytes;

    /// A view slot (Slice / ProjectField) exposes a sub-region of a parent
    /// buffer with **zero movement**: no new allocation, and reads see the
    /// parent's bytes through the offset. This is the addressing-class
    /// substrate (ADR-056).
    #[test]
    fn bind_view_is_zero_movement_subregion() {
        let mut pool = BufferArena::new();
        pool.rebind_reset(2);
        pool.bind_fresh(0, 8);
        pool.write_slot(0)
            .unwrap()
            .copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let bufs_before = pool.bufs.len();

        // Slot 1 views parent slot 0 at byte offset 2, length 4 — no alloc.
        pool.bind_view(1, 0, 2, 4);
        assert_eq!(pool.bufs.len(), bufs_before, "view must not allocate");

        // Explicit-length read, full-extent read, and whole-slot read all
        // resolve to the sub-region.
        assert_eq!(
            pool.read(BufferRef {
                slot: 1,
                offset: 0,
                length: 4
            }),
            &[2, 3, 4, 5]
        );
        assert_eq!(
            pool.read(BufferRef {
                slot: 1,
                offset: 0,
                length: 0
            }),
            &[2, 3, 4, 5]
        );
        assert_eq!(pool.read_slot(1).unwrap(), &[2, 3, 4, 5]);

        // The view aliases the parent: mutating the parent shows through.
        pool.write_slot(0).unwrap()[2] = 42;
        assert_eq!(pool.read_slot(1).unwrap(), &[42, 3, 4, 5]);
    }

    /// SC-3: transient pool bytes stay bounded across an arbitrarily long
    /// run of distinct interned values (generational eviction), so memory
    /// holds regardless of run length.
    #[test]
    fn transient_bytes_are_bounded_regardless_of_run_length() {
        let budget = 4096;
        let mut pool = BufferArena::new();
        pool.set_budget(budget);
        for i in 0..100_000u32 {
            let mut p = [7u8; 256];
            p[0] = i as u8;
            p[1] = (i >> 8) as u8;
            pool.store_unbound(address_bytes(&p), &p);
        }
        assert!(
            pool.transient_bytes() <= 2 * budget + 320,
            "resident transient {} exceeded 2*budget",
            pool.transient_bytes()
        );
    }

    /// A pinned value survives arbitrary transient churn (zero movement,
    /// never evicted).
    #[test]
    fn pinned_survives_transient_churn() {
        let mut pool = BufferArena::new();
        pool.set_budget(1024);
        let w = address_bytes(b"model-weight");
        pool.pin_bytes(w, b"model-weight");
        for i in 0..100_000u32 {
            let b = i.to_le_bytes();
            pool.store_unbound(address_bytes(&b), &b);
        }
        assert_eq!(pool.resolve(&w), Some(b"model-weight".as_slice()));
    }
}
