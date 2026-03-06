//! CRC32 checksum utilities (wraps crc32fast).

/// Compute CRC32 checksum of a byte slice.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Verify a CRC32 checksum matches an expected value.
#[inline]
pub fn verify_crc32(data: &[u8], expected: u32) -> bool {
    crc32(data) == expected
}

/// Compute CRC32 incrementally over multiple slices.
pub fn crc32_combine(slices: &[&[u8]]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    for s in slices {
        hasher.update(s);
    }
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice() {
        assert_eq!(crc32(&[]), 0);
    }

    #[test]
    fn known_value() {
        // CRC32 of "HOLO" should be deterministic
        let c = crc32(b"HOLO");
        assert_ne!(c, 0);
        assert_eq!(c, crc32(b"HOLO"));
    }

    #[test]
    fn verify_pass() {
        let data = b"test data";
        let checksum = crc32(data);
        assert!(verify_crc32(data, checksum));
    }

    #[test]
    fn verify_fail() {
        assert!(!verify_crc32(b"test data", 0x12345678));
    }

    #[test]
    fn combine_equivalence() {
        let full = b"hello world";
        let c1 = crc32(full);
        let c2 = crc32_combine(&[b"hello ", b"world"]);
        assert_eq!(c1, c2);
    }
}
