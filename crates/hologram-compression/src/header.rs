//! Compressed block header: magic, mode, sizes.
//!
//! Layout (16 bytes):
//!   [0..4]  magic: b"HLZC"
//!   [4]     mode tag (CompressionMode)
//!   [5]     permute_id (pre-transform permutation)
//!   [6..8]  reserved
//!   [8..16] original_len as u64 little-endian

use crate::codec::CompressionMode;

/// Magic bytes identifying a hologram-compression block.
pub const MAGIC: [u8; 4] = *b"HLZC";

/// Header size in bytes.
pub const HEADER_SIZE: usize = 16;

/// Serialize a header into 16 bytes.
pub fn encode_header(
    mode: CompressionMode,
    permute_id: u8,
    original_len: u64,
) -> [u8; HEADER_SIZE] {
    let mut buf = [0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = mode as u8;
    buf[5] = permute_id;
    // buf[6..8] reserved
    buf[8..16].copy_from_slice(&original_len.to_le_bytes());
    buf
}

/// Parsed header fields.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub mode: CompressionMode,
    pub permute_id: u8,
    pub original_len: u64,
}

/// Parse a header from bytes. Returns None if magic is wrong or mode is invalid.
pub fn decode_header(buf: &[u8]) -> Option<Header> {
    if buf.len() < HEADER_SIZE {
        return None;
    }
    if buf[0..4] != MAGIC {
        return None;
    }
    let mode = CompressionMode::from_tag(buf[4])?;
    let permute_id = buf[5];
    let original_len = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);
    Some(Header {
        mode,
        permute_id,
        original_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let hdr = encode_header(CompressionMode::Stratum, 2, 1024);
        let parsed = decode_header(&hdr).unwrap();
        assert_eq!(parsed.mode, CompressionMode::Stratum);
        assert_eq!(parsed.permute_id, 2);
        assert_eq!(parsed.original_len, 1024);
    }

    #[test]
    fn large_original_len() {
        let big: u64 = 5_000_000_000; // 5 GB, exceeds u32
        let hdr = encode_header(CompressionMode::Generic, 0, big);
        let parsed = decode_header(&hdr).unwrap();
        assert_eq!(parsed.original_len, big);
    }

    #[test]
    fn bad_magic() {
        let mut hdr = encode_header(CompressionMode::Generic, 0, 100);
        hdr[0] = b'X';
        assert!(decode_header(&hdr).is_none());
    }

    #[test]
    fn bad_mode() {
        let mut hdr = encode_header(CompressionMode::Generic, 0, 100);
        hdr[4] = 99;
        assert!(decode_header(&hdr).is_none());
    }
}
