//! Checksum utilities: BLAKE3 (primary) and truncated u32 variant for header compatibility.

/// Compute BLAKE3 checksum of a byte slice (full 32-byte hash).
#[inline]
pub fn checksum(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Compute blake3 hash of a byte slice, truncated to 4 bytes for header compatibility.
/// Returns the first 4 bytes of the blake3 hash as a u32.
#[inline]
pub fn blake3_u32(data: &[u8]) -> u32 {
    let hash = blake3::hash(data);
    let bytes = hash.as_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Verify a blake3 checksum (truncated to u32) matches.
#[inline]
pub fn verify_blake3(data: &[u8], expected: u32) -> bool {
    blake3_u32(data) == expected
}

/// Verify a BLAKE3 checksum matches an expected 32-byte value.
#[inline]
pub fn verify(data: &[u8], expected: &[u8; 32]) -> bool {
    &checksum(data) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice() {
        // BLAKE3 of empty input is a known constant (not zero).
        let c = checksum(&[]);
        assert_ne!(c, [0u8; 32]);
    }

    #[test]
    fn deterministic() {
        let c1 = checksum(b"HOLO");
        let c2 = checksum(b"HOLO");
        assert_eq!(c1, c2);
    }

    #[test]
    fn verify_pass() {
        let data = b"test data";
        let c = checksum(data);
        assert!(verify(data, &c));
    }

    #[test]
    fn verify_fail() {
        assert!(!verify(b"test data", &[0x12; 32]));
    }

    #[test]
    fn different_inputs_differ() {
        let c1 = checksum(b"hello");
        let c2 = checksum(b"world");
        assert_ne!(c1, c2);
    }
}
