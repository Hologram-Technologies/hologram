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
//! Two buffer classes, bounded so memory holds for arbitrary models and run
//! lengths (ADR-060, SC-3) — by the computation's structure, not a hardcoded
//! byte cap:
//!
//! * **pinned** — model constants/weights, deduped by content κ-label,
//!   resident for the session (the model's inherent footprint);
//! * **transient** — boundary inputs, intermediates, outputs — held in a
//!   two-generation pool whose generations rotate at each **walk** boundary
//!   (one `execute`): the finished walk ages to `previous` (kept so the next
//!   walk's unchanged prefix reuses it by label), the older generation is
//!   released (recompute on a later miss, never a wrong answer). Resident
//!   transient is the last two walks' working sets, which scales with the
//!   model and window — no fixed limit.
//!
//! Alignment: every buffer is 64-byte aligned (x86-64 cache line, AVX-512
//! ZMM, `bytemuck::cast_slice::<u8,f32>` zero-copy).

extern crate alloc;

use alloc::vec::Vec;
use core::ptr::NonNull;

use hashbrown::HashMap;
use hashbrown::HashSet;
use hologram_archive::{ContentLabel, WeightFingerprint, WeightProvider};
use hologram_backend::{BufferRef, SplitReads, Workspace};

/// A lazily-resident model weight: known by content label and fingerprint,
/// its body paged from the provider on first use and evictable under the
/// residency budget. `resident` is the backing buffer index while paged in,
/// `None` while evicted (re-pageable — a paged range hashes to the same κ,
/// so this never changes a result).
#[derive(Debug, Clone, Copy)]
struct LazyWeight {
    fp: WeightFingerprint,
    size: usize,
    resident: Option<usize>,
    /// Monotonic last-touch stamp for LRU eviction.
    last_touch: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotSpan {
    /// Byte offset (retained for API compatibility; per-slot buffers start
    /// at 0, so this is informational only).
    pub offset: u64,
    /// Byte length of the slot.
    pub length: u64,
}

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

    /// Reuse this buffer for a value of `len` bytes, reallocating if the
    /// existing capacity is too small **or** far too large. Zero-fills so a
    /// kernel that writes only a logical prefix leaves a deterministic tail.
    ///
    /// The free list is size-blind (LIFO), so recycling buffers across slots of
    /// differing sizes would otherwise ratchet every buffer's capacity up to the
    /// largest value it ever held and never release it — the backing arena grows
    /// unboundedly over a long autoregressive run even though the *count* of
    /// buffers is fixed. Releasing a buffer that is more than 2× oversized caps
    /// each buffer's capacity at ~2× its current use, bounding the arena to the
    /// working set (the realloc only fires on a large size mismatch).
    fn reset_to(&mut self, len: usize) {
        let want_cap = len.max(ARENA_ALIGN).next_multiple_of(ARENA_ALIGN);
        if len > self.cap || self.cap > want_cap.saturating_mul(2) {
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
    // Pinned content-addressed residency (label → bufs index): model
    // constants/weights, resident for the session, deduped by κ-label.
    pinned: HashMap<ContentLabel, usize>,
    // Two-generation content-addressed residency (label → bufs index). The
    // generation boundary is the *walk* (one `execute`), not a byte budget:
    // `current` is the in-progress walk's values, `previous` the last walk's
    // (kept so the next walk's unchanged prefix reuses them by label — the
    // content-addressed elision that replaces a KV-cache). Rotated in
    // `rebind_reset`. Resident transient is the last two walks' working sets —
    // it scales with the model and window, with no fixed cap.
    current: HashMap<ContentLabel, usize>,
    previous: HashMap<ContentLabel, usize>,
    // Lazily-resident weight tier (label → paging record): model weights the
    // session declared but did not copy resident at load. A weight's body
    // pages in from the host's `WeightProvider` on the first walk that
    // dispatches a kernel consuming it, and is evicted LRU to keep the
    // resident lazy-weight bytes within `lazy_budget`. Paged buffers survive
    // walk rotation (see `rebind_reset`), so a model whose weights fit the
    // budget pages each once and then behaves exactly like the pinned tier —
    // zero steady-state overhead — while a model that exceeds it streams its
    // weights through a bounded window instead of failing to load.
    lazy: HashMap<ContentLabel, LazyWeight>,
    /// Byte ceiling on resident lazy-weight bytes; `0` = unbounded (never
    /// evict — the fully-resident degenerate case). A structural budget the
    /// host sets from its window, not a hardcoded cap.
    lazy_budget: usize,
    /// Monotonic clock for LRU stamps.
    lazy_clock: u64,
    /// Reused scratch for `rebind_reset` so the per-walk reclaim computation
    /// allocates nothing after warmup — the `kept` reachability set (rebuilt
    /// each walk over the mostly-stable, potentially-thousands-strong pinned
    /// tier) and the reclaim list. Empty between walks.
    kept_scratch: HashSet<usize>,
    reclaim_scratch: Vec<usize>,
}

const UNBOUND: usize = usize::MAX;

impl BufferArena {
    pub fn new() -> Self {
        Self::default()
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
            lazy: HashMap::new(),
            lazy_budget: 0,
            lazy_clock: 0,
            kept_scratch: HashSet::new(),
            reclaim_scratch: Vec::new(),
        }
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
    /// Start a new walk: rotate the content-addressed generations and reset the
    /// slot→buffer binding table to `n` unbound slots.
    ///
    /// The generation boundary is the **walk** (one `execute`), not a byte
    /// budget — eviction is driven by the computation's structure, so the pool
    /// scales with the model and window with no hardcoded cap. The finished
    /// walk's values (`current`) age to `previous` (kept resident so the next
    /// walk's unchanged prefix reuses them by label — content-addressed elision,
    /// the KV-cache replacement); the older generation is released. Resident
    /// transient is therefore the last two walks' working sets.
    ///
    /// Within a walk nothing is evicted, so every value the walk produces stays
    /// available to its consumers and to output collection — correct for a graph
    /// of any size (no mid-walk drop of a still-live value).
    pub fn rebind_reset(&mut self, n: usize) {
        // Rotate: drop the older generation, age the finished walk into it.
        let dropped = core::mem::take(&mut self.previous);
        core::mem::swap(&mut self.current, &mut self.previous);
        // `current` is now the taken-empty map; `previous` is the finished walk.

        // Reclaim every buffer no longer reachable: the dropped generation's
        // buffers, plus any slot-only scratch — an un-addressable node's output
        // is bound to a slot but never retained under a label, so once we clear
        // the bindings it would leak (not in any label map, not on the free
        // list). A buffer is still needed only if pinned or in the kept
        // generation (`previous`); slot binding alone does not keep it, since we
        // clear the bindings here.
        // Paged weight buffers are kept too: a weight that fits the budget
        // pages once and stays resident across walks (steady-state pinning),
        // and one just paged for this walk's kernels must not be reclaimed.
        // Reachability set, rebuilt into reused scratch (no per-walk allocation
        // over the large, mostly-stable pinned tier; O(1) membership vs a fresh
        // BTreeSet's O(log n)).
        let kept = &mut self.kept_scratch;
        kept.clear();
        kept.extend(self.pinned.values().copied());
        kept.extend(self.previous.values().copied());
        kept.extend(self.lazy.values().filter_map(|e| e.resident));
        // Gather reclaimable buffers, then sort+dedup so the free list is pushed
        // in the same (ascending) order the previous BTreeSet produced — the
        // recycle order is preserved, only the data structure changed.
        let reclaim = &mut self.reclaim_scratch;
        reclaim.clear();
        for (_, bi) in dropped {
            if !kept.contains(&bi) {
                reclaim.push(bi);
            }
        }
        for &bi in &self.slot_buf {
            if bi != UNBOUND && !kept.contains(&bi) {
                reclaim.push(bi);
            }
        }
        reclaim.sort_unstable();
        reclaim.dedup();
        for idx in 0..self.reclaim_scratch.len() {
            let bi = self.reclaim_scratch[idx];
            if !self.free.contains(&bi) {
                self.free.push(bi);
            }
        }

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

    /// Set the resident lazy-weight byte budget (`0` = unbounded). The pool
    /// evicts LRU paged weights to keep [`Self::lazy_resident_bytes`] within
    /// it; a single weight larger than the budget still pages (a kernel
    /// cannot run a weight it cannot hold), so peak residency is
    /// `budget + largest_single_weight` in the worst case and exactly
    /// `budget` when every weight fits.
    pub fn set_lazy_budget(&mut self, bytes: usize) {
        self.lazy_budget = bytes;
    }

    /// Declare a model weight as **lazily resident**: record its label,
    /// fingerprint, and full size, but allocate nothing. Its body pages in
    /// from the provider on the first [`Self::page_and_bind`]. Idempotent;
    /// identical weights (same fingerprint ⇒ same label) collapse to one
    /// entry, so the dedup the pinned tier gives is preserved.
    pub fn pin_lazy(&mut self, label: ContentLabel, fp: WeightFingerprint, size: usize) {
        self.lazy.entry(label).or_insert(LazyWeight {
            fp,
            size,
            resident: None,
            last_touch: 0,
        });
    }

    #[inline]
    fn tick(&mut self) -> u64 {
        self.lazy_clock = self.lazy_clock.wrapping_add(1);
        self.lazy_clock
    }

    /// Resident lazy-weight bytes (deduped by construction — each label is one
    /// weight). The quantity the budget bounds; the pager's SC-3 witness.
    pub fn lazy_resident_bytes(&self) -> usize {
        self.lazy
            .values()
            .filter_map(|e| e.resident.map(|_| e.size))
            .sum()
    }

    /// Acquire a `len`-byte backing buffer (recycled from the free list or
    /// freshly allocated), returning its index. Contents are zeroed.
    fn acquire_buf(&mut self, len: usize) -> usize {
        match self.free.pop() {
            Some(bi) => {
                self.bufs[bi].reset_to(len);
                bi
            }
            None => {
                self.bufs.push(AlignedBytes::zeroed(len));
                self.bufs.len() - 1
            }
        }
    }

    /// Evict LRU paged weights until `need` more bytes fit the budget, never
    /// evicting a label in `keep` (every lazy operand of the node currently
    /// paging — a kernel reads them **simultaneously**, so none may be
    /// evicted to make room for another). No-op when unbounded or already
    /// within budget; stops when nothing else is evictable (a group whose
    /// combined footprint exceeds the budget then resides in full — a kernel
    /// cannot run operands it cannot hold).
    fn evict_lazy_protecting(&mut self, need: usize, keep: &[ContentLabel]) {
        if self.lazy_budget == 0 {
            return;
        }
        // Track the resident total incrementally (one O(L) sum up front) instead
        // of recomputing it every iteration, and pick victims from a single
        // LRU-sorted pass rather than an O(L) `min_by_key` scan per eviction —
        // the whole call is O(L log L) instead of O(L²).
        let mut resident_bytes = self.lazy_resident_bytes();
        if resident_bytes + need <= self.lazy_budget {
            return;
        }
        let mut victims: Vec<(u64, ContentLabel, usize)> = self
            .lazy
            .iter()
            .filter(|(l, e)| e.resident.is_some() && !keep.contains(l))
            .map(|(l, e)| (e.last_touch, *l, e.size))
            .collect();
        // Ascending last-touch ⇒ least-recently-used first.
        victims.sort_unstable_by_key(|&(t, _, _)| t);
        for (_, label, size) in victims {
            if resident_bytes + need <= self.lazy_budget {
                break;
            }
            if let Some(bi) = self.lazy.get_mut(&label).and_then(|e| e.resident.take()) {
                if !self.free.contains(&bi) {
                    self.free.push(bi);
                }
                resident_bytes -= size;
            }
        }
    }

    /// Page the lazy weight `label` resident if it is not (chunked whole-body
    /// fetch from `provider`), protecting `keep` from eviction. Returns the
    /// backing buffer index, or `None` if `label` is not a lazy weight or the
    /// provider cannot serve its body. Does not bind a slot.
    fn page_one<P: WeightProvider + ?Sized>(
        &mut self,
        label: &ContentLabel,
        provider: &P,
        keep: &[ContentLabel],
    ) -> Option<usize> {
        let &LazyWeight {
            fp, size, resident, ..
        } = self.lazy.get(label)?;
        if let Some(bi) = resident {
            let t = self.tick();
            if let Some(e) = self.lazy.get_mut(label) {
                e.last_touch = t;
            }
            return Some(bi);
        }
        self.evict_lazy_protecting(size, keep);
        let bi = self.acquire_buf(size);
        // Request the whole body; the provider owns its substrate's paging
        // (an in-memory store borrows zero-copy, an OPFS store assembles the
        // range however it pages internally). The `get_range` API still lets
        // a provider — or a future sub-tensor-resident kernel — address a
        // sub-range by the same κ; hologram's residency unit is the whole
        // weight, so it asks for `[0, size)` and copies once.
        let Some(body) = provider.get_range(fp, 0, size) else {
            // Paging failed — release the buffer, leave the weight
            // non-resident (fail-closed; the caller sees `None`).
            self.free.push(bi);
            return None;
        };
        if body.len() != size {
            self.free.push(bi);
            return None;
        }
        self.bufs[bi].as_mut_slice().copy_from_slice(&body);
        let t = self.tick();
        if let Some(e) = self.lazy.get_mut(label) {
            e.resident = Some(bi);
            e.last_touch = t;
        }
        Some(bi)
    }

    /// Page in and bind a node's **entire** lazy-weight operand set together,
    /// so all of them are simultaneously resident before the kernel reads
    /// them — the correct unit for a kernel that consumes more than one paged
    /// weight (e.g. a per-channel dequant matmul reading a packed weight plus
    /// large scale/zero-point vectors). Each is paged whole from `provider`
    /// (no copy on a hit, one copy on a miss), protecting the rest of the
    /// group from eviction; the resident set is held within the budget by
    /// evicting cold weights from *other* nodes. Returns `false` (fail-closed)
    /// if any operand is not a declared lazy weight or the provider cannot
    /// serve it. This is the weight-tier `resolve → miss → page → bind`.
    pub fn page_and_bind_group<P: WeightProvider + ?Sized>(
        &mut self,
        group: &[(usize, ContentLabel)],
        provider: &P,
    ) -> bool {
        // Protect every label in the group while paging any of them.
        let labels: Vec<ContentLabel> = group.iter().map(|&(_, l)| l).collect();
        for &(slot, label) in group {
            let Some(bi) = self.page_one(&label, provider, &labels) else {
                return false;
            };
            let size = self.lazy.get(&label).map_or(0, |e| e.size);
            self.ensure_slot(slot);
            self.slot_buf[slot] = bi;
            self.slot_len[slot] = size;
        }
        true
    }

    /// Convenience for a single lazy-weight operand — see
    /// [`Self::page_and_bind_group`].
    pub fn page_and_bind<P: WeightProvider + ?Sized>(
        &mut self,
        slot: usize,
        label: &ContentLabel,
        provider: &P,
    ) -> bool {
        self.page_and_bind_group(&[(slot, *label)], provider)
    }

    /// Store arbitrary bytes as a transient value addressed by `label`,
    /// without binding a slot. The byte→address boundary for inputs
    /// pre-interned ahead of `execute_addressed`.
    pub fn store_unbound(&mut self, label: ContentLabel, bytes: &[u8]) {
        if self.resident(&label) {
            return;
        }
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
        self.current.insert(label, bi);
    }

    /// Whether a value with this label is resident (pinned, transient, or a
    /// paged-in lazy weight). The lazy-tier probe is skipped entirely when no
    /// weights are paged (a non-paged session), so the fully-resident hot
    /// path pays nothing for the pager's existence.
    pub fn resident(&self, label: &ContentLabel) -> bool {
        self.pinned.contains_key(label)
            || self.current.contains_key(label)
            || self.previous.contains_key(label)
            || (!self.lazy.is_empty() && self.lazy.get(label).is_some_and(|e| e.resident.is_some()))
    }

    /// Resolve a label to its bytes, if resident (the address→byte boundary).
    /// A paged-in lazy weight resolves here too, so a weight that is also a
    /// graph output (or read back by label) works unchanged. The lazy probe
    /// is skipped when no weights are paged (zero overhead for a non-paged
    /// session).
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        let bi = self
            .pinned
            .get(label)
            .or_else(|| self.current.get(label))
            .or_else(|| self.previous.get(label))
            .or_else(|| {
                if self.lazy.is_empty() {
                    None
                } else {
                    self.lazy.get(label).and_then(|e| e.resident.as_ref())
                }
            })
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

    /// Resident pinned bytes (deduped) — the archive-embedded constants/weights
    /// pinned for the session's lifetime. Disjoint from `transient_bytes` (the
    /// pinned tier is keyed separately), so the full content-addressed footprint
    /// is `pinned_bytes() + transient_bytes()`.
    pub fn pinned_bytes(&self) -> usize {
        let mut seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
        let mut total = 0;
        for &bi in self.pinned.values() {
            if seen.insert(bi) {
                total += self.bufs[bi].len;
            }
        }
        total
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
    use alloc::borrow::Cow;
    use core::cell::Cell;
    use hologram_archive::address_bytes;

    /// In-test `WeightProvider` over an owned body table, counting bytes
    /// served and honoring a forced-fail set, so the pool's pager can be
    /// exercised in isolation (no session, no archive).
    struct MockProvider {
        bodies: HashMap<WeightFingerprint, Vec<u8>>,
        served: Cell<usize>,
        fail: HashMap<WeightFingerprint, ()>,
    }
    impl MockProvider {
        fn new() -> Self {
            Self {
                bodies: HashMap::new(),
                served: Cell::new(0),
                fail: HashMap::new(),
            }
        }
        /// Register a weight; returns its `(fingerprint, label)`.
        fn add(&mut self, body: Vec<u8>) -> (WeightFingerprint, ContentLabel) {
            let fp = WeightFingerprint::of(&body);
            let label = fp.content_label();
            self.bodies.insert(fp, body);
            (fp, label)
        }
        fn served(&self) -> usize {
            self.served.get()
        }
    }
    impl WeightProvider for MockProvider {
        fn size(&self, fp: WeightFingerprint) -> Option<usize> {
            self.bodies.get(&fp).map(Vec::len)
        }
        fn get_range(
            &self,
            fp: WeightFingerprint,
            offset: usize,
            len: usize,
        ) -> Option<Cow<'_, [u8]>> {
            if self.fail.contains_key(&fp) {
                return None;
            }
            let b = self.bodies.get(&fp)?;
            let r = b.get(offset..offset.checked_add(len)?)?;
            self.served.set(self.served.get() + r.len());
            Some(Cow::Borrowed(r))
        }
    }

    /// Whole-body read of `slot` (page_and_bind sets slot_len to the size).
    fn slot_bytes(pool: &BufferArena, slot: usize) -> Vec<u8> {
        pool.read_slot(slot).unwrap().to_vec()
    }

    #[test]
    fn lazy_pages_on_demand_and_is_readable() {
        let mut p = MockProvider::new();
        let body: Vec<u8> = (0..500u16).map(|i| i as u8).collect();
        let (fp, label) = p.add(body.clone());
        let mut pool = BufferArena::new();
        pool.rebind_reset(4);
        pool.pin_lazy(label, fp, body.len());

        // Nothing resident until paged.
        assert_eq!(pool.lazy_resident_bytes(), 0);
        assert!(!pool.resident(&label));
        assert!(pool.resolve(&label).is_none());

        assert!(pool.page_and_bind(1, &label, &p));
        assert_eq!(slot_bytes(&pool, 1), body);
        assert_eq!(pool.resolve(&label), Some(body.as_slice()));
        assert!(pool.resident(&label));
        assert_eq!(pool.lazy_resident_bytes(), body.len());
        assert_eq!(p.served(), body.len(), "paged exactly once");

        // A hit does not re-serve.
        assert!(pool.page_and_bind(2, &label, &p));
        assert_eq!(p.served(), body.len(), "hit must not re-page");
        assert_eq!(slot_bytes(&pool, 2), body);
    }

    #[test]
    fn lazy_budget_evicts_lru_and_repages_identically() {
        let mut p = MockProvider::new();
        let (fa, la) = p.add(vec![0xAA; 100]);
        let (fb, lb) = p.add(vec![0xBB; 100]);
        let (fc, lc) = p.add(vec![0xCC; 100]);
        let mut pool = BufferArena::new();
        pool.rebind_reset(8);
        pool.set_lazy_budget(250); // room for exactly two
        pool.pin_lazy(la, fa, 100);
        pool.pin_lazy(lb, fb, 100);
        pool.pin_lazy(lc, fc, 100);

        pool.page_and_bind(0, &la, &p); // resident: A
        pool.page_and_bind(1, &lb, &p); // resident: A,B
        assert_eq!(pool.lazy_resident_bytes(), 200);
        pool.page_and_bind(2, &lc, &p); // C needs room → evict LRU (A)
        assert!(
            pool.lazy_resident_bytes() <= 250,
            "budget held: {}",
            pool.lazy_resident_bytes()
        );
        assert!(!pool.resident(&la), "A (LRU) evicted");
        assert!(pool.resident(&lb) && pool.resident(&lc));
        assert_eq!(p.served(), 300, "A,B,C each paged once so far");

        // Re-paging A returns identical bytes (residency is orthogonal to
        // identity) and costs another page (A was evicted).
        assert!(pool.page_and_bind(3, &la, &p));
        assert_eq!(slot_bytes(&pool, 3), vec![0xAA; 100]);
        assert_eq!(p.served(), 400, "A re-paged");
    }

    #[test]
    fn lazy_group_keeps_all_operands_resident() {
        // Budget == exactly the group footprint: a multi-weight node must
        // hold all its operands together; a per-weight pager would evict one.
        let mut p = MockProvider::new();
        let (fa, la) = p.add(vec![1u8; 100]);
        let (fb, lb) = p.add(vec![2u8; 100]);
        let (fc, lc) = p.add(vec![3u8; 100]);
        let mut pool = BufferArena::new();
        pool.rebind_reset(8);
        // Room for the 2-weight group but not also the unrelated weight, so
        // paging the group must evict the unrelated one — never a member.
        pool.set_lazy_budget(200);
        pool.pin_lazy(la, fa, 100);
        pool.pin_lazy(lb, fb, 100);
        pool.pin_lazy(lc, fc, 100);

        pool.page_and_bind(0, &lc, &p); // resident: C (100)
        let ok = pool.page_and_bind_group(&[(1, la), (2, lb)], &p);
        assert!(ok);
        assert!(
            pool.resident(&la) && pool.resident(&lb),
            "group both resident"
        );
        assert!(!pool.resident(&lc), "unrelated LRU evicted for the group");
        assert_eq!(slot_bytes(&pool, 1), vec![1u8; 100]);
        assert_eq!(slot_bytes(&pool, 2), vec![2u8; 100]);
        assert!(pool.lazy_resident_bytes() <= 200);
    }

    #[test]
    fn lazy_group_over_budget_still_resides_in_full() {
        // A group whose combined footprint exceeds the budget must reside in
        // full — a kernel cannot run operands it cannot hold. Peak then
        // exceeds the budget by construction (the documented worst case).
        let mut p = MockProvider::new();
        let (fa, la) = p.add(vec![1u8; 100]);
        let (fb, lb) = p.add(vec![2u8; 100]);
        let mut pool = BufferArena::new();
        pool.rebind_reset(4);
        pool.set_lazy_budget(50); // below even one weight
        pool.pin_lazy(la, fa, 100);
        pool.pin_lazy(lb, fb, 100);
        assert!(pool.page_and_bind_group(&[(0, la), (1, lb)], &p));
        assert_eq!(slot_bytes(&pool, 0), vec![1u8; 100]);
        assert_eq!(slot_bytes(&pool, 1), vec![2u8; 100]);
    }

    #[test]
    fn lazy_survives_walk_rotation_without_repaging() {
        let mut p = MockProvider::new();
        let (fp, label) = p.add(vec![0x7E; 300]);
        let mut pool = BufferArena::new();
        pool.rebind_reset(4);
        pool.set_lazy_budget(0); // unbounded
        pool.pin_lazy(label, fp, 300);
        pool.page_and_bind(0, &label, &p);
        assert_eq!(p.served(), 300);

        // Rotate generations across many walks; the paged weight must persist
        // (kept from reclaim) and rebind without a re-page — steady-state
        // pinning for a fitting model.
        for _ in 0..1000 {
            pool.rebind_reset(4);
            assert!(pool.resident(&label), "paged weight survives rotation");
            assert!(pool.page_and_bind(0, &label, &p));
        }
        assert_eq!(p.served(), 300, "never re-paged across 1000 walks");
        assert_eq!(slot_bytes(&pool, 0), vec![0x7E; 300]);
    }

    #[test]
    fn lazy_page_fails_closed_on_missing_or_short_provider() {
        let mut p = MockProvider::new();
        let (fp, label) = p.add(vec![9u8; 128]);
        p.fail.insert(fp, ()); // provider will refuse this body
        let mut pool = BufferArena::new();
        pool.rebind_reset(4);
        pool.pin_lazy(label, fp, 128);
        assert!(
            !pool.page_and_bind(0, &label, &p),
            "paging failure returns false, never a wrong answer"
        );
        assert!(!pool.resident(&label));
        assert_eq!(pool.lazy_resident_bytes(), 0);

        // An unknown label is not a lazy weight → false, no panic.
        assert!(!pool.page_and_bind(0, &address_bytes(b"nope"), &p));
    }

    #[test]
    fn lazy_dedup_same_fingerprint_pages_once() {
        let mut p = MockProvider::new();
        let (fp, label) = p.add(vec![5u8; 200]);
        let mut pool = BufferArena::new();
        pool.rebind_reset(4);
        pool.set_lazy_budget(0);
        // Two declarations of the identical weight collapse to one entry.
        pool.pin_lazy(label, fp, 200);
        pool.pin_lazy(label, fp, 200);
        pool.page_and_bind(0, &label, &p);
        pool.page_and_bind(1, &label, &p);
        assert_eq!(p.served(), 200, "deduped weight paged once");
        assert_eq!(pool.lazy_resident_bytes(), 200, "counted once");
    }

    #[test]
    fn lazy_unbounded_never_evicts() {
        let mut p = MockProvider::new();
        let mut pool = BufferArena::new();
        pool.rebind_reset(64);
        pool.set_lazy_budget(0);
        let mut labels = Vec::new();
        for i in 0..32u8 {
            let (fp, l) = p.add(vec![i; 100]);
            pool.pin_lazy(l, fp, 100);
            labels.push(l);
        }
        for (slot, l) in labels.iter().enumerate() {
            pool.page_and_bind(slot, l, &p);
        }
        assert_eq!(pool.lazy_resident_bytes(), 32 * 100, "all resident");
        for l in &labels {
            assert!(pool.resident(l));
        }
    }

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

    /// SC-3: transient pool bytes stay bounded across an arbitrarily long run.
    /// Generations rotate at the walk boundary (`rebind_reset`), not on a byte
    /// budget, so resident transient is exactly the last two walks' working sets
    /// regardless of run length — bounded with no hardcoded cap.
    #[test]
    fn transient_bytes_are_bounded_regardless_of_run_length() {
        let mut pool = BufferArena::new();
        let per_walk = 16usize;
        let val = 256usize;
        for walk in 0..100_000u32 {
            pool.rebind_reset(0); // walk boundary: rotate generations
            for j in 0..per_walk {
                let mut p = [7u8; 256];
                p[0] = walk as u8;
                p[1] = (walk >> 8) as u8;
                p[2] = j as u8;
                pool.store_unbound(address_bytes(&p), &p);
            }
        }
        // Two generations (this walk + the previous), each `per_walk` distinct
        // values; nothing older survives. Independent of the 100k run length.
        assert!(
            pool.transient_bytes() <= 2 * per_walk * val + 320,
            "resident transient {} exceeded two walks",
            pool.transient_bytes()
        );
    }

    /// A pinned value survives arbitrary transient churn (zero movement,
    /// never evicted) across any number of walk rotations.
    #[test]
    fn pinned_survives_transient_churn() {
        let mut pool = BufferArena::new();
        let w = address_bytes(b"model-weight");
        pool.pin_bytes(w, b"model-weight");
        for walk in 0..100_000u32 {
            pool.rebind_reset(0);
            let b = walk.to_le_bytes();
            pool.store_unbound(address_bytes(&b), &b);
        }
        assert_eq!(pool.resolve(&w), Some(b"model-weight".as_slice()));
    }
}
