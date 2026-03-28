//! Checksum utilities: CRC32 (legacy) and blake3 (HR-3).

/// Compute CRC32 checksum of a byte slice (legacy, for v1 archives).
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Compute blake3 hash of a byte slice, truncated to 4 bytes for header compatibility.
/// Returns the first 4 bytes of the blake3 hash as a u32.
#[inline]
pub fn blake3_u32(data: &[u8]) -> u32 {
    let hash = blake3::hash(data);
    let bytes = hash.as_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Compute full 32-byte blake3 hash.
#[inline]
pub fn blake3_full(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Verify a blake3 checksum (truncated to u32) matches.
#[inline]
pub fn verify_blake3(data: &[u8], expected: u32) -> bool {
    blake3_u32(data) == expected
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
