//! Page-aligned buffer backed by anonymous mmap.
//!
//! Unlike `Vec<u8>`, dropping an `MmapBuffer` returns pages to the OS
//! immediately via `munmap`. No allocator fragmentation — RSS tracks
//! actual usage, not allocator free-list size.
//!
//! On non-Unix platforms, falls back to `Vec<u8>` with explicit
//! capacity shrinking on drop.

/// A page-aligned byte buffer backed by anonymous mmap (Unix) or
/// a capacity-managed Vec (non-Unix).
pub struct MmapBuffer {
    #[cfg(unix)]
    ptr: *mut u8,
    #[cfg(unix)]
    len: usize,
    #[cfg(not(unix))]
    data: Vec<u8>,
}

// SAFETY: MmapBuffer owns its memory exclusively (no shared references).
unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Allocate a zero-initialized buffer of `len` bytes.
    ///
    /// On Unix, uses `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` for page-aligned
    /// allocation with OS-managed lifecycle. On other platforms, uses `Vec<u8>`.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        #[cfg(unix)]
        {
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                // Fallback to Vec if mmap fails (shouldn't happen for reasonable sizes).
                let data = vec![0u8; len];
                let leaked = data.leak();
                return Self {
                    ptr: leaked.as_mut_ptr(),
                    len,
                };
            }
            Self {
                ptr: ptr.cast::<u8>(),
                len,
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                data: vec![0u8; len],
            }
        }
    }

    /// Create an empty buffer (no allocation).
    #[inline]
    pub fn empty() -> Self {
        #[cfg(unix)]
        {
            Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            }
        }
        #[cfg(not(unix))]
        {
            Self { data: Vec::new() }
        }
    }

    /// Create from an existing `Vec<u8>`, taking ownership.
    ///
    /// Copies into a new mmap allocation and drops the Vec.
    pub fn from_vec(v: Vec<u8>) -> Self {
        if v.is_empty() {
            return Self::empty();
        }
        let mut buf = Self::new(v.len());
        buf.as_mut_slice().copy_from_slice(&v);
        // Vec drops here — `free()` returns pages to OS for large allocations.
        buf
    }

    /// View as byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.is_empty() {
            return &[];
        }
        #[cfg(unix)]
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
        #[cfg(not(unix))]
        self.data.as_slice()
    }

    /// View as mutable byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.is_empty() {
            return &mut [];
        }
        #[cfg(unix)]
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
        #[cfg(not(unix))]
        self.data.as_mut_slice()
    }

    /// Length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        #[cfg(unix)]
        {
            self.len
        }
        #[cfg(not(unix))]
        {
            self.data.len()
        }
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to `Vec<u8>` (copies data out of mmap).
    /// Used by `take()` which needs to return owned bytes to callers.
    pub fn into_vec(self) -> Vec<u8> {
        if self.is_empty() {
            return Vec::new();
        }
        self.as_slice().to_vec()
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            if self.len > 0 {
                unsafe {
                    libc::munmap(self.ptr.cast(), self.len);
                }
            }
        }
        // Non-Unix: Vec drops automatically.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer() {
        let buf = MmapBuffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.as_slice(), &[]);
    }

    #[test]
    fn allocate_and_write() {
        let mut buf = MmapBuffer::new(1024);
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.as_slice()[0], 0); // Zero-initialized.
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    fn from_vec_copies() {
        let v = vec![1u8, 2, 3, 4];
        let buf = MmapBuffer::from_vec(v);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn into_vec_copies() {
        let mut buf = MmapBuffer::new(4);
        buf.as_mut_slice().copy_from_slice(&[10, 20, 30, 40]);
        let v = buf.into_vec();
        assert_eq!(v, vec![10, 20, 30, 40]);
    }

    #[test]
    fn alignment_is_page() {
        let buf = MmapBuffer::new(4096);
        assert_eq!(buf.as_slice().as_ptr() as usize % 4096, 0);
    }

    #[test]
    fn drop_frees_memory() {
        // Just verify no crash on drop.
        let buf = MmapBuffer::new(1024 * 1024);
        assert_eq!(buf.len(), 1024 * 1024);
        drop(buf);
    }

    #[test]
    fn large_allocation() {
        let buf = MmapBuffer::new(256 * 1024 * 1024); // 256MB
        assert_eq!(buf.len(), 256 * 1024 * 1024);
        // Don't actually write all 256MB — just verify allocation works.
        drop(buf);
    }
}
