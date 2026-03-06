//! Q1 const-compatible math helpers for 16-bit table generation.

/// Signed interpretation of a u16 index for Q1:
/// 0..=32767 → 0..=32767, 32768..=65535 → -32768..=-1.
pub(crate) const fn signed16(i: u32) -> i32 {
    if i < 32768 {
        i as i32
    } else {
        i as i32 - 65536
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed16_values() {
        assert_eq!(signed16(0), 0);
        assert_eq!(signed16(1), 1);
        assert_eq!(signed16(32767), 32767);
        assert_eq!(signed16(32768), -32768);
        assert_eq!(signed16(65535), -1);
    }

    #[test]
    fn signed16_matches_i16_cast() {
        for i in (0u32..=65535).step_by(256) {
            assert_eq!(signed16(i), i as u16 as i16 as i32);
        }
    }
}
