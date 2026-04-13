//! RAII buffer lending primitives.
//!
//! Provides `LentRegion` and `LentRegionMut` — non-Copy wrappers around
//! borrowed byte slices that make lending explicit in type signatures.
//! The borrow checker enforces that no `LentRegion` outlives the owning
//! buffer, eliminating the need for runtime condition variables.
//!
//! `MmapLender` wraps a contiguous mmap region and lends sub-regions
//! for Plan 061's arena-based activation buffer management.

use super::mmap_buf::MmapBuffer;

/// An immutable borrowed region of a buffer.
///
/// Non-Copy: the borrow checker tracks the lending lifetime.
/// Derefs to `&[u8]` for seamless use in existing APIs.
pub struct LentRegion<'owner> {
    data: &'owner [u8],
}

impl<'owner> LentRegion<'owner> {
    /// Create a lent region from a byte slice.
    #[inline]
    pub fn new(data: &'owner [u8]) -> Self {
        Self { data }
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the region is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl AsRef<[u8]> for LentRegion<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl std::ops::Deref for LentRegion<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.data
    }
}

/// A mutable borrowed region of a buffer.
///
/// Non-Copy: the borrow checker tracks the lending lifetime.
/// Derefs to `&mut [u8]` for seamless use in existing APIs.
pub struct LentRegionMut<'owner> {
    data: &'owner mut [u8],
}

impl<'owner> LentRegionMut<'owner> {
    /// Create a mutable lent region from a mutable byte slice.
    #[inline]
    pub fn new(data: &'owner mut [u8]) -> Self {
        Self { data }
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the region is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl AsRef<[u8]> for LentRegionMut<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl AsMut<[u8]> for LentRegionMut<'_> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

impl std::ops::Deref for LentRegionMut<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.data
    }
}

impl std::ops::DerefMut for LentRegionMut<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

/// Owns a contiguous mmap region and lends sub-regions.
///
/// Designed for Plan 061: one large mmap allocation at tape start,
/// sub-allocated per instruction output, with `madvise(MADV_FREE)`
/// on freed sub-regions.
pub struct MmapLender {
    buf: MmapBuffer,
}

impl MmapLender {
    /// Allocate a new contiguous region of `len` bytes.
    pub fn new(len: usize) -> Self {
        Self {
            buf: MmapBuffer::new(len),
        }
    }

    /// Total size of the owned region.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the region is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Lend an immutable sub-region.
    ///
    /// # Panics
    /// Panics if `offset + len > self.len()`.
    #[inline]
    pub fn lend(&self, offset: usize, len: usize) -> LentRegion<'_> {
        LentRegion::new(&self.buf.as_slice()[offset..offset + len])
    }

    /// Lend a mutable sub-region.
    ///
    /// # Panics
    /// Panics if `offset + len > self.len()`.
    #[inline]
    pub fn lend_mut(&mut self, offset: usize, len: usize) -> LentRegionMut<'_> {
        LentRegionMut::new(&mut self.buf.as_mut_slice()[offset..offset + len])
    }

    /// Advise the OS that a sub-region's pages can be reclaimed.
    ///
    /// On macOS: `MADV_FREE` (lazy — zero syscall overhead if pages
    /// aren't reclaimed). On Linux: `MADV_DONTNEED` (eager).
    /// On non-Unix: no-op.
    pub fn advise_free_region(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        #[cfg(unix)]
        {
            const PAGE_SIZE: usize = 4096;
            let base = self.buf.as_slice().as_ptr() as usize + offset;
            // Align up to page boundary.
            let aligned_start = (base + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            let end = base + len;
            if aligned_start >= end {
                return; // Region smaller than one page after alignment.
            }
            let aligned_len = end - aligned_start;
            // Round down to page boundary.
            let aligned_len = aligned_len & !(PAGE_SIZE - 1);
            if aligned_len == 0 {
                return;
            }
            unsafe {
                #[cfg(target_os = "macos")]
                libc::madvise(
                    aligned_start as *mut libc::c_void,
                    aligned_len,
                    libc::MADV_FREE,
                );
                #[cfg(not(target_os = "macos"))]
                libc::madvise(
                    aligned_start as *mut libc::c_void,
                    aligned_len,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lent_region_deref() {
        let data = vec![1u8, 2, 3, 4, 5];
        let region = LentRegion::new(&data);
        assert_eq!(region.len(), 5);
        assert_eq!(&*region, &[1, 2, 3, 4, 5]);
        assert_eq!(region.as_ref(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn lent_region_mut_deref() {
        let mut data = vec![1u8, 2, 3];
        let mut region = LentRegionMut::new(&mut data);
        region[0] = 10;
        assert_eq!(&*region, &[10, 2, 3]);
    }

    #[test]
    fn mmap_lender_basic() {
        let lender = MmapLender::new(1024);
        assert_eq!(lender.len(), 1024);
        let region = lender.lend(0, 512);
        assert_eq!(region.len(), 512);
        // mmap is zero-initialized.
        assert!(region.iter().all(|&b| b == 0));
    }

    #[test]
    fn mmap_lender_write_and_read() {
        let mut lender = MmapLender::new(256);
        {
            let mut region = lender.lend_mut(10, 4);
            region.copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        }
        let region = lender.lend(10, 4);
        assert_eq!(&*region, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn mmap_lender_non_overlapping_regions() {
        let mut lender = MmapLender::new(256);
        {
            let mut r1 = lender.lend_mut(0, 4);
            r1.copy_from_slice(&[1, 2, 3, 4]);
        }
        {
            let mut r2 = lender.lend_mut(100, 4);
            r2.copy_from_slice(&[5, 6, 7, 8]);
        }
        assert_eq!(&*lender.lend(0, 4), &[1, 2, 3, 4]);
        assert_eq!(&*lender.lend(100, 4), &[5, 6, 7, 8]);
    }

    #[test]
    fn mmap_lender_advise_free_no_crash() {
        let mut lender = MmapLender::new(8192);
        {
            let mut region = lender.lend_mut(0, 8192);
            region.fill(0xFF);
        }
        // Should not crash, even if OS doesn't honor the advice.
        lender.advise_free_region(0, 8192);
        // After advise, reading is still valid (pages may be zeros or old data).
        let _ = lender.lend(0, 8192);
    }

    #[test]
    fn mmap_lender_advise_free_small_region() {
        let lender = MmapLender::new(64);
        // Region smaller than a page — advise should be a no-op.
        lender.advise_free_region(0, 64);
    }

    #[test]
    fn mmap_lender_empty() {
        let lender = MmapLender::new(0);
        assert!(lender.is_empty());
        assert_eq!(lender.len(), 0);
    }

    #[test]
    #[should_panic]
    fn lend_out_of_bounds_panics() {
        let lender = MmapLender::new(100);
        let _ = lender.lend(50, 60); // 50 + 60 = 110 > 100
    }
}
