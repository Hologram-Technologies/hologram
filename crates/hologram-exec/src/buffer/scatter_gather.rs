//! Zero-copy scatter-gather stream over discontiguous byte regions.
//!
//! Presents multiple `&[u8]` segments as a single seekable `Read + Seek`
//! stream without copying. Designed for Plan 061's mmap arena where an
//! op's logical input may span non-adjacent regions after eviction and
//! re-allocation.
//!
//! Fast path: when there is exactly one segment, `as_contiguous()` returns
//! it directly (no indirection). This covers 90%+ of calls in practice.

use std::io::{self, Read, Seek, SeekFrom};

/// A zero-copy stream over discontiguous byte segments.
///
/// Implements `Read` and `Seek` by tracking the current segment index
/// and offset within that segment. No heap allocation in the read path.
pub struct ScatterGatherStream<'a> {
    segments: &'a [&'a [u8]],
    /// Index of the current segment being read.
    seg_idx: usize,
    /// Byte offset within the current segment.
    seg_off: usize,
    /// Total logical length (cached at construction).
    total_len: u64,
}

impl<'a> ScatterGatherStream<'a> {
    /// Create a new scatter-gather stream over the given segments.
    #[inline]
    pub fn new(segments: &'a [&'a [u8]]) -> Self {
        let total_len: u64 = segments.iter().map(|s| s.len() as u64).sum();
        Self {
            segments,
            seg_idx: 0,
            seg_off: 0,
            total_len,
        }
    }

    /// Total logical length across all segments.
    #[inline]
    pub fn len(&self) -> u64 {
        self.total_len
    }

    /// Whether the stream is empty (no segments or all segments empty).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Fast path: if there is exactly one segment, return it directly.
    ///
    /// Returns `None` for zero or multiple segments. Callers should use
    /// this to avoid `Read` indirection in the common single-buffer case.
    #[inline]
    pub fn as_contiguous(&self) -> Option<&'a [u8]> {
        if self.segments.len() == 1 {
            Some(self.segments[0])
        } else {
            None
        }
    }

    /// Materialize a byte range into a new `Vec<u8>`.
    ///
    /// Use sparingly — this allocates. Intended for the rare case where
    /// a kernel needs a contiguous view across segment boundaries.
    pub fn read_range_to_vec(&self, offset: u64, len: usize) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; len];
        let mut remaining = len;
        let mut dst_off = 0;
        let mut abs_off = offset;

        for seg in self.segments {
            let seg_len = seg.len() as u64;
            if abs_off >= seg_len {
                abs_off -= seg_len;
                continue;
            }
            let start = abs_off as usize;
            let avail = seg.len() - start;
            let to_copy = remaining.min(avail);
            out[dst_off..dst_off + to_copy].copy_from_slice(&seg[start..start + to_copy]);
            dst_off += to_copy;
            remaining -= to_copy;
            if remaining == 0 {
                break;
            }
            abs_off = 0;
        }
        // Truncate if the requested range extends past the end.
        out.truncate(dst_off);
        out
    }

    /// Current absolute byte position in the logical stream.
    fn abs_position(&self) -> u64 {
        let mut pos: u64 = 0;
        for seg in &self.segments[..self.seg_idx] {
            pos += seg.len() as u64;
        }
        pos + self.seg_off as u64
    }

    /// Seek to an absolute byte position, updating seg_idx and seg_off.
    fn seek_to_abs(&mut self, target: u64) {
        let target = target.min(self.total_len);
        let mut remaining = target;
        for (i, seg) in self.segments.iter().enumerate() {
            let seg_len = seg.len() as u64;
            if remaining < seg_len {
                self.seg_idx = i;
                self.seg_off = remaining as usize;
                return;
            }
            remaining -= seg_len;
        }
        // At or past end.
        self.seg_idx = self.segments.len();
        self.seg_off = 0;
    }
}

impl Read for ScatterGatherStream<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.seg_idx >= self.segments.len() {
            return Ok(0);
        }

        let mut total_read = 0;
        while total_read < buf.len() && self.seg_idx < self.segments.len() {
            let seg = self.segments[self.seg_idx];
            let avail = seg.len() - self.seg_off;
            if avail == 0 {
                self.seg_idx += 1;
                self.seg_off = 0;
                continue;
            }
            let to_copy = (buf.len() - total_read).min(avail);
            buf[total_read..total_read + to_copy]
                .copy_from_slice(&seg[self.seg_off..self.seg_off + to_copy]);
            total_read += to_copy;
            self.seg_off += to_copy;
            if self.seg_off >= seg.len() {
                self.seg_idx += 1;
                self.seg_off = 0;
            }
        }
        Ok(total_read)
    }
}

impl Seek for ScatterGatherStream<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.total_len as i64 + offset,
            SeekFrom::Current(offset) => self.abs_position() as i64 + offset,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to negative position",
            ));
        }
        let new_pos = new_pos as u64;
        self.seek_to_abs(new_pos);
        Ok(new_pos.min(self.total_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream() {
        let segs: &[&[u8]] = &[];
        let stream = ScatterGatherStream::new(segs);
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert!(stream.as_contiguous().is_none());
    }

    #[test]
    fn single_segment_fast_path() {
        let data = &[1u8, 2, 3, 4, 5];
        let segs: &[&[u8]] = &[data];
        let stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.len(), 5);
        assert_eq!(stream.as_contiguous(), Some(data.as_slice()));
    }

    #[test]
    fn multi_segment_no_contiguous() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5];
        let segs: &[&[u8]] = &[a, b];
        let stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.len(), 5);
        assert!(stream.as_contiguous().is_none());
    }

    #[test]
    fn read_across_boundary() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5, 6];
        let segs: &[&[u8]] = &[a, b];
        let mut stream = ScatterGatherStream::new(segs);
        let mut buf = [0u8; 6];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 6);
        assert_eq!(buf, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn read_partial() {
        let a = &[10u8, 20];
        let b = &[30u8, 40, 50];
        let segs: &[&[u8]] = &[a, b];
        let mut stream = ScatterGatherStream::new(segs);
        let mut buf = [0u8; 3];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(buf, [10, 20, 30]);

        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], &[40, 50]);

        // EOF
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn seek_start() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5, 6];
        let segs: &[&[u8]] = &[a, b];
        let mut stream = ScatterGatherStream::new(segs);

        stream.seek(SeekFrom::Start(4)).unwrap();
        let mut buf = [0u8; 2];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [5, 6]);
    }

    #[test]
    fn seek_end() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5, 6];
        let segs: &[&[u8]] = &[a, b];
        let mut stream = ScatterGatherStream::new(segs);

        stream.seek(SeekFrom::End(-2)).unwrap();
        let mut buf = [0u8; 2];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [5, 6]);
    }

    #[test]
    fn seek_current() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5, 6];
        let segs: &[&[u8]] = &[a, b];
        let mut stream = ScatterGatherStream::new(segs);

        // Read 2 bytes, then seek forward 2 from current.
        let mut buf = [0u8; 2];
        stream.read_exact(&mut buf).unwrap();
        stream.seek(SeekFrom::Current(2)).unwrap();
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [5, 6]);
    }

    #[test]
    fn seek_negative_errors() {
        let a = &[1u8, 2, 3];
        let segs: &[&[u8]] = &[a];
        let mut stream = ScatterGatherStream::new(segs);
        let result = stream.seek(SeekFrom::Start(0));
        assert!(result.is_ok());
        let result = stream.seek(SeekFrom::Current(-1));
        assert!(result.is_err());
    }

    #[test]
    fn seek_past_end_clamps() {
        let a = &[1u8, 2];
        let segs: &[&[u8]] = &[a];
        let mut stream = ScatterGatherStream::new(segs);
        let pos = stream.seek(SeekFrom::Start(100)).unwrap();
        assert_eq!(pos, 2); // Clamped to total_len.
        let mut buf = [0u8; 1];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 0); // EOF.
    }

    #[test]
    fn read_range_to_vec_within_segment() {
        let a = &[1u8, 2, 3, 4, 5];
        let segs: &[&[u8]] = &[a];
        let stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.read_range_to_vec(1, 3), vec![2, 3, 4]);
    }

    #[test]
    fn read_range_to_vec_across_boundary() {
        let a = &[1u8, 2, 3];
        let b = &[4u8, 5, 6];
        let segs: &[&[u8]] = &[a, b];
        let stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.read_range_to_vec(2, 3), vec![3, 4, 5]);
    }

    #[test]
    fn read_range_to_vec_past_end_truncates() {
        let a = &[1u8, 2];
        let segs: &[&[u8]] = &[a];
        let stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.read_range_to_vec(1, 10), vec![2]);
    }

    #[test]
    fn empty_segments_skipped() {
        let a: &[u8] = &[];
        let b = &[1u8, 2];
        let c: &[u8] = &[];
        let d = &[3u8];
        let segs: &[&[u8]] = &[a, b, c, d];
        let mut stream = ScatterGatherStream::new(segs);
        assert_eq!(stream.len(), 3);
        let mut buf = [0u8; 3];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(buf, [1, 2, 3]);
    }

    #[test]
    fn many_small_segments() {
        let s1 = &[1u8];
        let s2 = &[2u8];
        let s3 = &[3u8];
        let s4 = &[4u8];
        let s5 = &[5u8];
        let segs: &[&[u8]] = &[s1, s2, s3, s4, s5];
        let mut stream = ScatterGatherStream::new(segs);
        let mut buf = [0u8; 5];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(buf, [1, 2, 3, 4, 5]);
    }
}
