//! Helpers for content-addressable [`uor_foundation::kernel::address::Element`]
//! impls in hologram-core.
//!
//! Per ADR-052:
//! - `digest_algorithm = "blake3"` across the hologram ecosystem.
//! - `canonical_bytes` follow Amendment 43 §2: `header(k) || le_bytes(x, k+1)`,
//!   where k = ring index (0..15 for Q0..Q15) and the value is encoded in
//!   `(k + 1) * 8`-bit little-endian.
//!
//! These helpers produce the canonical bytes and the
//! `"blake3:<64 hex chars>"` digest string used across the q1/q2/q3
//! datum/address impls.
extern crate alloc;
use alloc::vec::Vec;

/// Length of the digest string: `blake3:` + 64 lowercase hex chars.
pub const DIGEST_STR_LEN: usize = 7 + 64;

/// Build the canonical-bytes vector per Amendment 43 §2.
///
/// `level_index` is the ring level k (W8 → 0, W16 → 1, W24 → 2, W32 → 3, …).
/// The value is encoded in `level_index + 1` little-endian bytes.
#[inline]
#[must_use]
pub fn canonical_bytes(level_index: u8, value: u128) -> Vec<u8> {
    let value_bytes = (level_index as usize) + 1;
    let mut out = Vec::with_capacity(1 + value_bytes);
    out.push(level_index);
    for i in 0..value_bytes {
        out.push((value >> (i * 8)) as u8);
    }
    out
}

/// Compute the `"blake3:<hex>"` digest string from canonical bytes.
#[inline]
#[must_use]
pub fn blake3_digest_str(canonical: &[u8]) -> [u8; DIGEST_STR_LEN] {
    let hash = blake3::hash(canonical);
    let mut out = [0u8; DIGEST_STR_LEN];
    out[..7].copy_from_slice(b"blake3:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, byte) in hash.as_bytes().iter().enumerate() {
        out[7 + i * 2] = HEX[(byte >> 4) as usize];
        out[7 + i * 2 + 1] = HEX[(byte & 0x0F) as usize];
    }
    out
}
