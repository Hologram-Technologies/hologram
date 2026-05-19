//! StaticBuf: fixed-size, stack-allocated byte buffer for no_alloc targets.

/// A fixed-size byte buffer backed by a `[u8; N]` array.
///
/// Provides a `push`/index interface without heap allocation, suitable for
/// `no_alloc` environments like embedded ARM and WASM without an allocator.
/// The capacity `N` is a compile-time constant.
#[derive(Clone, Copy)]
pub struct StaticBuf<const N: usize> {
    data: [u8; N],
    len: usize,
}

impl<const N: usize> StaticBuf<N> {
    /// Create an empty buffer.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: [0u8; N],
            len: 0,
        }
    }

    /// Maximum number of bytes this buffer can hold.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Number of bytes currently stored.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer contains no bytes.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is at full capacity.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Append a byte. Returns `false` if the buffer is full.
    #[inline]
    pub fn push(&mut self, byte: u8) -> bool {
        if self.len < N {
            self.data[self.len] = byte;
            self.len += 1;
            true
        } else {
            false
        }
    }

    /// Remove and return the last byte, or `None` if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<u8> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            Some(self.data[self.len])
        }
    }

    /// View the filled portion as a byte slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Mutable view of the filled portion.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.len]
    }

    /// Reset the buffer to empty (does not zero memory).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Copy bytes from a slice, up to remaining capacity.
    /// Returns the number of bytes actually written.
    #[inline]
    pub fn extend_from_slice(&mut self, src: &[u8]) -> usize {
        let available = N - self.len;
        let n = if src.len() < available {
            src.len()
        } else {
            available
        };
        self.data[self.len..self.len + n].copy_from_slice(&src[..n]);
        self.len += n;
        n
    }
}

impl<const N: usize> Default for StaticBuf<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> core::ops::Index<usize> for StaticBuf<N> {
    type Output = u8;
    #[inline]
    fn index(&self, i: usize) -> &u8 {
        &self.data[..self.len][i]
    }
}

impl<const N: usize> core::fmt::Debug for StaticBuf<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StaticBuf")
            .field("len", &self.len)
            .field("capacity", &N)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let buf = StaticBuf::<256>::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 256);
    }

    #[test]
    fn push_and_len() {
        let mut buf = StaticBuf::<4>::new();
        assert!(buf.push(10));
        assert!(buf.push(20));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn push_returns_false_when_full() {
        let mut buf = StaticBuf::<2>::new();
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(!buf.push(3));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn pop_lifo_order() {
        let mut buf = StaticBuf::<4>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn as_slice_contents() {
        let mut buf = StaticBuf::<8>::new();
        for b in 0u8..5 {
            buf.push(b);
        }
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn clear_resets_len() {
        let mut buf = StaticBuf::<8>::new();
        buf.push(42);
        buf.push(99);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn is_full() {
        let mut buf = StaticBuf::<2>::new();
        assert!(!buf.is_full());
        buf.push(1);
        buf.push(2);
        assert!(buf.is_full());
    }

    #[test]
    fn extend_from_slice_fits() {
        let mut buf = StaticBuf::<8>::new();
        let n = buf.extend_from_slice(&[10, 20, 30]);
        assert_eq!(n, 3);
        assert_eq!(buf.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn extend_from_slice_truncates() {
        let mut buf = StaticBuf::<4>::new();
        buf.push(0);
        buf.push(0);
        // 2 remaining slots, try to write 4 bytes
        let n = buf.extend_from_slice(&[1, 2, 3, 4]);
        assert_eq!(n, 2);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.as_slice(), &[0, 0, 1, 2]);
    }

    #[test]
    fn index_access() {
        let mut buf = StaticBuf::<8>::new();
        buf.push(7);
        buf.push(8);
        assert_eq!(buf[0], 7);
        assert_eq!(buf[1], 8);
    }

    #[test]
    fn default_is_empty() {
        let buf = StaticBuf::<16>::default();
        assert!(buf.is_empty());
    }

    #[test]
    fn zero_capacity_always_full() {
        let mut buf = StaticBuf::<0>::new();
        assert!(buf.is_full());
        assert!(!buf.push(1));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn extend_then_clear_then_reuse() {
        let mut buf = StaticBuf::<8>::new();
        buf.extend_from_slice(&[1, 2, 3, 4]);
        buf.clear();
        buf.extend_from_slice(&[5, 6]);
        assert_eq!(buf.as_slice(), &[5, 6]);
    }

    #[test]
    fn q0_single_byte_use_case() {
        // Canonical use case: Q0 single-byte operations, 256-byte buffer
        let mut buf = StaticBuf::<256>::new();
        for i in 0u8..=255 {
            assert!(buf.push(i));
        }
        assert!(buf.is_full());
        for (i, &b) in buf.as_slice().iter().enumerate() {
            assert_eq!(b, i as u8);
        }
    }

    #[test]
    fn as_mut_slice_write() {
        let mut buf = StaticBuf::<4>::new();
        buf.push(0);
        buf.push(0);
        buf.as_mut_slice()[0] = 99;
        assert_eq!(buf[0], 99);
    }
}
