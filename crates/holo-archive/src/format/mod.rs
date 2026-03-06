//! Core format types: magic bytes, page alignment, header, and serialized graph.

pub mod graph;
pub mod header;

/// Magic bytes identifying a .holo archive: ASCII "HOLO".
pub const HOLO_MAGIC: [u8; 4] = *b"HOLO";

/// Page alignment for mmap'd sections (4 KB).
pub const PAGE_SIZE: u64 = 4096;

/// Current archive format version.
pub const FORMAT_VERSION: u32 = 1;

/// Align an offset to the next page boundary.
#[inline]
#[must_use]
pub const fn align_to_page(offset: u64) -> u64 {
    (offset + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_holo() {
        assert_eq!(&HOLO_MAGIC, b"HOLO");
    }

    #[test]
    fn align_zero() {
        assert_eq!(align_to_page(0), 0);
    }

    #[test]
    fn align_at_boundary() {
        assert_eq!(align_to_page(4096), 4096);
    }

    #[test]
    fn align_off_boundary() {
        assert_eq!(align_to_page(1), 4096);
        assert_eq!(align_to_page(4095), 4096);
        assert_eq!(align_to_page(4097), 8192);
    }
}
