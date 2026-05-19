//! Byte-level I/O helpers for unified ring dispatch.
//!
//! These convert between byte slices and `u64` for ring arithmetic at any
//! quantum level. For widths 1, 2, 4, 8, LLVM compiles these to single
//! `movzx`/`mov` instructions (zero-extending loads / truncating stores).

/// Read up to 8 bytes from a LE byte slice into u64.
///
/// For width 1: `movzx rax, byte ptr [rsi]` (1 cycle)
/// For width 2: `movzx rax, word ptr [rsi]` (1 cycle)
/// For width 4: `mov eax, dword ptr [rsi]` (1 cycle)
/// For width 8: `mov rax, qword ptr [rsi]` (1 cycle)
#[inline(always)]
pub fn read_le_u64(bytes: &[u8], width: usize) -> u64 {
    debug_assert!(width <= 8 && width <= bytes.len());
    let mut buf = [0u8; 8];
    buf[..width].copy_from_slice(&bytes[..width]);
    u64::from_le_bytes(buf)
}

/// Write the low `width` bytes of a u64 into a LE byte slice.
#[inline(always)]
pub fn write_le_u64(bytes: &mut [u8], val: u64, width: usize) {
    debug_assert!(width <= 8 && width <= bytes.len());
    bytes[..width].copy_from_slice(&val.to_le_bytes()[..width]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_write_roundtrip_width1() {
        for v in 0..=255u8 {
            let bytes = [v];
            let val = read_le_u64(&bytes, 1);
            assert_eq!(val, v as u64);
            let mut out = [0u8];
            write_le_u64(&mut out, val, 1);
            assert_eq!(out[0], v);
        }
    }

    #[test]
    fn read_write_roundtrip_width2() {
        for v in [0u16, 1, 255, 256, 0x1234, 0xFFFF] {
            let bytes = v.to_le_bytes();
            let val = read_le_u64(&bytes, 2);
            assert_eq!(val, v as u64);
            let mut out = [0u8; 2];
            write_le_u64(&mut out, val, 2);
            assert_eq!(out, bytes);
        }
    }

    #[test]
    fn read_write_roundtrip_width4() {
        for v in [0u32, 1, 0xDEADBEEF, u32::MAX] {
            let bytes = v.to_le_bytes();
            let val = read_le_u64(&bytes, 4);
            assert_eq!(val, v as u64);
            let mut out = [0u8; 4];
            write_le_u64(&mut out, val, 4);
            assert_eq!(out, bytes);
        }
    }

    #[test]
    fn read_write_roundtrip_width8() {
        for v in [0u64, 1, 0xDEADBEEFCAFEBABE, u64::MAX] {
            let bytes = v.to_le_bytes();
            let val = read_le_u64(&bytes, 8);
            assert_eq!(val, v);
            let mut out = [0u8; 8];
            write_le_u64(&mut out, val, 8);
            assert_eq!(out, bytes);
        }
    }

    #[test]
    fn read_width3_zero_extends() {
        let bytes = [0xFF, 0xFF, 0xFF];
        let val = read_le_u64(&bytes, 3);
        assert_eq!(val, 0x00FF_FFFF); // only 3 bytes, high 5 zeroed
    }

    #[test]
    fn performance_1m_roundtrips() {
        let start = std::time::Instant::now();
        let mut buf = [0u8; 8];
        for i in 0..1_000_000u64 {
            write_le_u64(&mut buf, i, 4);
            let _ = read_le_u64(&buf, 4);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "1M byte I/O roundtrips took {}ms (target < 50ms)",
            elapsed.as_millis()
        );
    }
}
