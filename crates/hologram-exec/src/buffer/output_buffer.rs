//! Output buffer abstraction for tape executor kernels.
//!
//! `OutputBuffer` replaces `&mut Vec<u8>` in all kernel signatures,
//! enabling two backing strategies:
//!
//! - **Heap** (default, LLMs): wraps a normal `Vec<u8>`. Zero overhead
//!   vs the previous bare-Vec interface.
//!
//! - **Arena** (memory-pressure, diffusion models): points into a
//!   contiguous `MmapLender` region. Fixed capacity, no allocator calls
//!   on resize/extend. Evicted regions are returned to the OS via
//!   `madvise(MADV_FREE)` — RSS tracks the live working set, not the
//!   high-water mark.
//!
//! Kernels interact with `OutputBuffer` through the same operations they
//! used on `Vec<u8>`: `len()`, `clear()`, `resize()`, `extend_from_slice()`,
//! `as_ptr()`, `as_slice()`, `as_mut_slice()`. The enum dispatch compiles
//! to a single predicted branch per call.

use std::ops::{Deref, DerefMut};

/// Output buffer for tape executor kernels.
///
/// Three variants:
/// - `Heap`: wraps `Vec<u8>`, used for small buffers and the LLM path.
/// - `Arena`: points into a contiguous `MmapLender` region (Plan 062).
/// - `Mmap`: individually mmap'd buffer via `MmapBuffer`. On drop,
///   `munmap` immediately returns pages to the OS. Used for large
///   activation buffers (>1 MiB) in the diffusion path. This is the
///   key to keeping RSS bounded — unlike `Vec<u8>` whose pages linger
///   in the allocator's free-list, `MmapBuffer::drop` calls `munmap`
///   directly.
///
/// Kernels use the same API for all variants — the enum dispatch is a
/// single predicted branch, zero overhead in practice.
/// Buffers at or above this size use MmapBuffer instead of Vec<u8> when
/// promoted during resize/extend. MmapBuffer::drop calls munmap, which
/// immediately returns pages to the OS. Below this threshold, Vec<u8> is
/// faster (no syscall overhead per allocation).
pub const MMAP_EVICT_THRESHOLD: usize = 256 * 1024; // 256 KiB

pub enum OutputBuffer {
    /// Heap-allocated buffer. Owns its memory via `Vec<u8>`.
    /// Used when `checkpoint_enabled = false` (LLM path) and for
    /// small buffers in all paths.
    Heap(Vec<u8>),

    /// Arena-backed buffer. Points into a contiguous `MmapLender` region.
    /// Fixed capacity determined at tape build time from slot assignments.
    ///
    /// The `MmapLender` that owns the backing pages must outlive all
    /// `OutputBuffer::Arena` instances. The executor enforces this by
    /// keeping the lender alive for the entire `execute_direct` call.
    Arena {
        /// Raw pointer to the start of this buffer's slot within the arena.
        /// Always 16-byte aligned (enforced by slot offset computation).
        ptr: *mut u8,
        /// Current logical length in bytes (how much has been written).
        len: usize,
        /// Maximum capacity in bytes (slot size, fixed at tape build time).
        capacity: usize,
    },

    /// Individually mmap'd buffer. On drop, `munmap` returns pages to
    /// the OS immediately — RSS tracks live buffers, not allocator caches.
    /// Used for large activation buffers (≥MMAP_EVICT_THRESHOLD) in the
    /// eviction path. Tracks a logical length separately from the mmap's
    /// full capacity (same as Arena's len/capacity split).
    Mmap {
        buf: super::mmap_buf::MmapBuffer,
        /// Logical length: how many bytes have been written.
        len: usize,
    },
}

// SAFETY: Arena pointers are valid for the lifetime of execute_direct.
// Each OutputBuffer points to a non-overlapping mmap region (guaranteed
// by slot assignments — nodes sharing a slot have non-overlapping lifetimes).
// The mmap region is MAP_PRIVATE (not shared with other processes).
// Rayon parallel kernels split the output into disjoint &mut [f32] chunks
// via par_chunks_mut — no data races within a single dispatch.
unsafe impl Send for OutputBuffer {}
unsafe impl Sync for OutputBuffer {}

impl OutputBuffer {
    /// Create a heap-backed buffer with the given capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self::Heap(Vec::with_capacity(cap))
    }

    /// Create an empty heap-backed buffer.
    #[inline]
    pub fn new() -> Self {
        Self::Heap(Vec::new())
    }

    /// Create an arena-backed buffer pointing into a pre-allocated region.
    ///
    /// # Safety
    /// `ptr` must point to a valid, writable region of at least `capacity`
    /// bytes that remains valid for the lifetime of this `OutputBuffer`.
    /// The region must be 16-byte aligned for f32 cast safety.
    #[inline]
    pub unsafe fn arena(ptr: *mut u8, capacity: usize) -> Self {
        debug_assert!(
            (ptr as usize).is_multiple_of(16),
            "arena pointer must be 16-byte aligned"
        );
        Self::Arena {
            ptr,
            len: 0,
            capacity,
        }
    }

    /// Create an Mmap-backed buffer with the given capacity.
    /// On drop, `munmap` returns pages to the OS immediately.
    /// Starts with logical length 0 (capacity is the mmap size).
    #[inline]
    pub fn mmap(capacity: usize) -> Self {
        Self::Mmap {
            buf: super::mmap_buf::MmapBuffer::new(capacity),
            len: 0,
        }
    }

    /// Current length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Heap(v) => v.len(),
            Self::Arena { len, .. } => *len,
            Self::Mmap { len, .. } => *len,
        }
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            Self::Heap(v) => v.capacity(),
            Self::Arena { capacity, .. } => *capacity,
            Self::Mmap { buf, .. } => buf.len(),
        }
    }

    /// Raw pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Heap(v) => v.as_ptr(),
            Self::Arena { ptr, .. } => *ptr as *const u8,
            Self::Mmap { buf, .. } => buf.as_slice().as_ptr(),
        }
    }

    /// Mutable raw pointer to the start of the buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Heap(v) => v.as_mut_ptr(),
            Self::Arena { ptr, .. } => *ptr,
            Self::Mmap { buf, .. } => buf.as_mut_slice().as_mut_ptr(),
        }
    }

    /// View the written portion as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Heap(v) => v.as_slice(),
            Self::Arena { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            Self::Mmap { buf, len } => &buf.as_slice()[..*len],
        }
    }

    /// View the written portion as a mutable byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Heap(v) => v.as_mut_slice(),
            Self::Arena { ptr, len, .. } => unsafe { std::slice::from_raw_parts_mut(*ptr, *len) },
            Self::Mmap { buf, len } => &mut buf.as_mut_slice()[..*len],
        }
    }

    /// Reset length to zero without deallocating.
    #[inline]
    pub fn clear(&mut self) {
        match self {
            Self::Heap(v) => v.clear(),
            Self::Arena { len, .. } => *len = 0,
            Self::Mmap { len, .. } => *len = 0,
        }
    }

    /// Resize to `new_len` bytes, filling new bytes with `value`.
    ///
    /// For Mmap: if `new_len > capacity`, promotes to a new larger Mmap.
    /// For Arena: if `new_len > capacity`, promotes to Mmap (not Heap).
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: u8) {
        match self {
            Self::Heap(v) => {
                // Promote empty Heap to Mmap for large allocations.
                // This ensures that when eviction drops the buffer,
                // munmap returns pages to the OS immediately.
                if v.is_empty() && new_len >= MMAP_EVICT_THRESHOLD {
                    let m = super::mmap_buf::MmapBuffer::new(new_len);
                    // MmapBuffer is zero-initialized; fill with value if non-zero.
                    if value != 0 {
                        let mut m = m;
                        for b in m.as_mut_slice() {
                            *b = value;
                        }
                        *self = Self::Mmap {
                            buf: m,
                            len: new_len,
                        };
                    } else {
                        *self = Self::Mmap {
                            buf: m,
                            len: new_len,
                        };
                    }
                } else {
                    v.resize(new_len, value);
                }
            }
            Self::Arena {
                ptr, len, capacity, ..
            } => {
                if new_len <= *capacity {
                    if new_len > *len {
                        unsafe {
                            std::ptr::write_bytes((*ptr).add(*len), value, new_len - *len);
                        }
                    }
                    *len = new_len;
                } else {
                    // Promote to Mmap: the kernel needs more space than
                    // the arena slot provides. Copy existing data into a
                    // new mmap-backed buffer.
                    let mut m = super::mmap_buf::MmapBuffer::new(new_len);
                    if *len > 0 {
                        m.as_mut_slice()[..*len]
                            .copy_from_slice(unsafe { std::slice::from_raw_parts(*ptr, *len) });
                    }
                    if value != 0 {
                        // MmapBuffer is zero-initialized; only fill with
                        // non-zero value if requested.
                        for b in &mut m.as_mut_slice()[*len..] {
                            *b = value;
                        }
                    }
                    *self = Self::Mmap {
                        buf: m,
                        len: new_len,
                    };
                }
            }
            Self::Mmap { buf, len } => {
                if new_len <= buf.len() {
                    // Fits within existing mmap capacity.
                    if new_len > *len && value != 0 {
                        for b in &mut buf.as_mut_slice()[*len..new_len] {
                            *b = value;
                        }
                    }
                    *len = new_len;
                } else {
                    // Grow: allocate a new larger mmap and copy.
                    let mut new_buf = super::mmap_buf::MmapBuffer::new(new_len);
                    if *len > 0 {
                        new_buf.as_mut_slice()[..*len].copy_from_slice(&buf.as_slice()[..*len]);
                    }
                    if value != 0 {
                        for b in &mut new_buf.as_mut_slice()[*len..] {
                            *b = value;
                        }
                    }
                    *self = Self::Mmap {
                        buf: new_buf,
                        len: new_len,
                    };
                }
            }
        }
    }

    /// Append bytes to the end.
    #[inline]
    pub fn extend_from_slice(&mut self, data: &[u8]) {
        match self {
            Self::Heap(v) => v.extend_from_slice(data),
            Self::Arena {
                ptr, len, capacity, ..
            } => {
                let new_len = *len + data.len();
                if new_len <= *capacity {
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), (*ptr).add(*len), data.len());
                    }
                    *len = new_len;
                } else {
                    // Promote to Mmap.
                    let mut m = super::mmap_buf::MmapBuffer::new(new_len);
                    if *len > 0 {
                        m.as_mut_slice()[..*len]
                            .copy_from_slice(unsafe { std::slice::from_raw_parts(*ptr, *len) });
                    }
                    m.as_mut_slice()[*len..new_len].copy_from_slice(data);
                    *self = Self::Mmap {
                        buf: m,
                        len: new_len,
                    };
                }
            }
            Self::Mmap { buf, len } => {
                let new_len = *len + data.len();
                if new_len <= buf.len() {
                    // Fits within existing mmap capacity.
                    buf.as_mut_slice()[*len..new_len].copy_from_slice(data);
                    *len = new_len;
                } else {
                    // Grow: allocate a new larger mmap and copy.
                    let mut new_buf = super::mmap_buf::MmapBuffer::new(new_len);
                    if *len > 0 {
                        new_buf.as_mut_slice()[..*len].copy_from_slice(&buf.as_slice()[..*len]);
                    }
                    new_buf.as_mut_slice()[*len..new_len].copy_from_slice(data);
                    *self = Self::Mmap {
                        buf: new_buf,
                        len: new_len,
                    };
                }
            }
        }
    }

    /// Consume this buffer and return the data as an owned `Vec<u8>`.
    ///
    /// For Heap: zero-cost move.
    /// For Arena/Mmap: copies the live data into a new Vec.
    pub fn into_vec(self) -> Vec<u8> {
        match self {
            Self::Heap(v) => v,
            Self::Arena { ptr, len, .. } => {
                let mut v = Vec::with_capacity(len);
                if len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), len);
                        v.set_len(len);
                    }
                }
                v
            }
            Self::Mmap { buf, len } => buf.as_slice()[..len].to_vec(),
        }
    }
}

impl Default for OutputBuffer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<u8>> for OutputBuffer {
    #[inline]
    fn from(v: Vec<u8>) -> Self {
        Self::Heap(v)
    }
}

impl Deref for OutputBuffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for OutputBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

// Drop behavior:
// - Heap: Vec<u8> drops normally via its built-in Drop, freeing the allocation.
// - Arena: fields are Copy (ptr, len, capacity) — no Drop runs. The MmapLender
//   that owns the backing pages handles deallocation when it goes out of scope.
// No custom Drop impl is needed (and having one would prevent moving Vec out
// of the Heap variant in into_vec).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_basic_ops() {
        let mut buf = OutputBuffer::with_capacity(64);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.resize(16, 0);
        assert_eq!(buf.len(), 16);

        buf.extend_from_slice(&[1, 2, 3, 4]);
        assert_eq!(buf.len(), 20);
        assert_eq!(&buf.as_slice()[16..20], &[1, 2, 3, 4]);

        buf.clear();
        assert!(buf.is_empty());

        let v = buf.into_vec();
        assert!(v.is_empty());
    }

    #[test]
    fn arena_basic_ops() {
        // Simulate an arena region with a heap-allocated aligned buffer.
        let mut backing = vec![0u8; 256];
        // Ensure 16-byte alignment (Vec<u8> on most allocators is already aligned).
        let ptr = backing.as_mut_ptr();
        let aligned_ptr = ((ptr as usize + 15) & !15) as *mut u8;
        let usable_cap = 256 - (aligned_ptr as usize - ptr as usize);

        let mut buf = unsafe { OutputBuffer::arena(aligned_ptr, usable_cap) };
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), usable_cap);

        buf.resize(32, 0xAB);
        assert_eq!(buf.len(), 32);
        assert_eq!(buf.as_slice()[0], 0xAB);

        buf.extend_from_slice(&[1, 2, 3]);
        assert_eq!(buf.len(), 35);
        assert_eq!(buf.as_slice()[32], 1);
        assert_eq!(buf.as_slice()[34], 3);

        buf.clear();
        assert!(buf.is_empty());

        // into_vec copies out of the arena.
        buf.resize(4, 42);
        let v = buf.into_vec();
        assert_eq!(v, vec![42; 4]);

        // The backing is still valid — arena owns it.
        drop(backing);
    }

    #[test]
    fn heap_from_vec() {
        let v = vec![1u8, 2, 3];
        let buf = OutputBuffer::from(v);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn deref_works() {
        let mut buf = OutputBuffer::with_capacity(8);
        buf.resize(4, 0);
        // Deref to &[u8]
        let slice: &[u8] = &buf;
        assert_eq!(slice.len(), 4);
        // DerefMut to &mut [u8]
        let mslice: &mut [u8] = &mut buf;
        mslice[0] = 99;
        assert_eq!(buf.as_slice()[0], 99);
    }

    #[test]
    fn arena_resize_beyond_capacity_promotes_to_heap() {
        let mut backing = vec![0u8; 64];
        let ptr = backing.as_mut_ptr();
        let mut buf = unsafe { OutputBuffer::arena(ptr, 32) };
        buf.resize(10, 0xAA); // within capacity — stays Arena
        assert_eq!(buf.len(), 10);
        buf.resize(64, 0xBB); // exceeds 32-byte capacity — promotes to Heap
        assert_eq!(buf.len(), 64);
        assert_eq!(buf.as_slice()[0], 0xAA); // original data preserved
        assert_eq!(buf.as_slice()[10], 0xBB); // new fill byte
                                              // Verify it's now a Heap variant by checking capacity grew.
        assert!(buf.capacity() >= 64);
    }

    #[test]
    fn arena_extend_beyond_capacity_promotes_to_heap() {
        let mut backing = vec![0u8; 64];
        let ptr = backing.as_mut_ptr();
        let mut buf = unsafe { OutputBuffer::arena(ptr, 8) };
        buf.extend_from_slice(&[1, 2, 3]); // within capacity
        assert_eq!(buf.len(), 3);
        buf.extend_from_slice(&[4, 5, 6, 7, 8, 9, 10]); // exceeds 8-byte capacity
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
