//! BLAKE3 checksum utilities for archive integrity verification.

/// Compute BLAKE3 checksum of a byte slice.
#[inline]
pub fn checksum(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Verify a BLAKE3 checksum matches an expected value.
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
