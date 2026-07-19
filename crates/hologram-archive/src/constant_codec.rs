//! Constants section codec (spec X — extension; spec X.3 weight dedup).
//!
//! Each entry is one of two forms:
//!
//!   `Inline { slot: u32, dtype: u8, body_len: u32, body_bytes... }`
//!     — bytes embedded directly in the section. Used for small literals.
//!
//!   `Reference { slot: u32, dtype: u8, fingerprint: [u8; 32] }`
//!     — slot points at a body in the `Weights` section keyed by BLAKE3
//!       fingerprint. Used for large model weights so the bytes appear
//!       in the archive exactly once (spec X.3 / X-7 trillion-param
//!       claim). The fingerprint is `WeightFingerprint::of(bytes)`.
//!
//! On decode, a `ConstantEntry` carries either the inlined `bytes` or a
//! `fingerprint` to resolve against the archive's `WeightStore`. The
//! session does a single lookup per entry at load time; the workspace
//! pre-fill copies bytes into the slot.
//!
//! Wire-level discriminant byte: `0x00` for inline, `0x01` for reference.

use crate::error::ArchiveError;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct ConstantEntry {
    pub slot: u32,
    pub dtype: u8,
    /// Inlined body. Empty if this entry is a reference (see `fingerprint`).
    pub bytes: Vec<u8>,
    /// Content fingerprint (`WeightFingerprint::of(bytes)`). `[0u8; 32]`
    /// when `bytes` is inlined; populated when this entry is a reference
    /// into the Weights section.
    pub fingerprint: [u8; 32],
    /// True when this entry's body lives in the Weights section keyed by
    /// `fingerprint`; false when bytes are embedded inline.
    pub by_reference: bool,
}

impl ConstantEntry {
    /// Construct an inlined entry.
    pub fn inline(slot: u32, dtype: u8, bytes: Vec<u8>) -> Self {
        Self {
            slot,
            dtype,
            bytes,
            fingerprint: [0u8; 32],
            by_reference: false,
        }
    }

    /// Construct a reference entry resolved from `Weights` at session load.
    pub fn reference(slot: u32, dtype: u8, fingerprint: [u8; 32]) -> Self {
        Self {
            slot,
            dtype,
            bytes: Vec::new(),
            fingerprint,
            by_reference: true,
        }
    }
}

const TAG_INLINE: u8 = 0;
const TAG_REFERENCE: u8 = 1;

pub fn encode(entries: &[ConstantEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        4 + entries
            .iter()
            .map(|e| {
                if e.by_reference {
                    1 + 4 + 1 + 32
                } else {
                    1 + 4 + 1 + 4 + e.bytes.len()
                }
            })
            .sum::<usize>(),
    );
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        if e.by_reference {
            out.push(TAG_REFERENCE);
            out.extend_from_slice(&e.slot.to_le_bytes());
            out.push(e.dtype);
            out.extend_from_slice(&e.fingerprint);
        } else {
            out.push(TAG_INLINE);
            out.extend_from_slice(&e.slot.to_le_bytes());
            out.push(e.dtype);
            out.extend_from_slice(&(e.bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&e.bytes);
        }
    }
    out
}

pub fn decode(bytes: &[u8]) -> Result<Vec<ConstantEntry>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated {
            needed: 4,
            actual: bytes.len(),
        });
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count.min(bytes.len())); // cap on untrusted count (DoS)
    let mut cur = 4usize;
    for _ in 0..count {
        if cur + 1 > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cur + 1,
                actual: bytes.len(),
            });
        }
        let tag = bytes[cur];
        cur += 1;
        if cur + 4 + 1 > bytes.len() {
            return Err(ArchiveError::Truncated {
                needed: cur + 5,
                actual: bytes.len(),
            });
        }
        let slot = u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let dtype = bytes[cur];
        cur += 1;
        match tag {
            TAG_INLINE => {
                if cur + 4 > bytes.len() {
                    return Err(ArchiveError::Truncated {
                        needed: cur + 4,
                        actual: bytes.len(),
                    });
                }
                let len = u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap()) as usize;
                cur += 4;
                if cur + len > bytes.len() {
                    return Err(ArchiveError::Truncated {
                        needed: cur + len,
                        actual: bytes.len(),
                    });
                }
                let body = bytes[cur..cur + len].to_vec();
                cur += len;
                out.push(ConstantEntry::inline(slot, dtype, body));
            }
            TAG_REFERENCE => {
                if cur + 32 > bytes.len() {
                    return Err(ArchiveError::Truncated {
                        needed: cur + 32,
                        actual: bytes.len(),
                    });
                }
                let mut fp = [0u8; 32];
                fp.copy_from_slice(&bytes[cur..cur + 32]);
                cur += 32;
                out.push(ConstantEntry::reference(slot, dtype, fp));
            }
            _ => return Err(ArchiveError::Io("unknown ConstantEntry tag")),
        }
    }
    Ok(out)
}
