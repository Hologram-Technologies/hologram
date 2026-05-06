//! Schedule section codec (spec VI.3 + VIII.2).
//!
//! Mirrors `hologram-archive::writer::encode_schedule`. Decodes the
//! `Schedule` section's payload back into a `Vec<Vec<u32>>` of node-id
//! groups indexed by parallel-execution level.

use crate::error::ArchiveError;

/// Decode a Schedule section payload into a list of levels, each carrying
/// its NodeId u32 indices.
pub fn decode(bytes: &[u8]) -> Result<Vec<Vec<u32>>, ArchiveError> {
    if bytes.len() < 4 {
        return Err(ArchiveError::Truncated { needed: 4, actual: bytes.len() });
    }
    let level_count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut levels = Vec::with_capacity(level_count);
    let mut cursor = 4usize;
    for _ in 0..level_count {
        if cursor + 4 > bytes.len() {
            return Err(ArchiveError::Truncated { needed: cursor + 4, actual: bytes.len() });
        }
        let n = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let needed = cursor + n * 4;
        if needed > bytes.len() {
            return Err(ArchiveError::Truncated { needed, actual: bytes.len() });
        }
        let mut level = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            level.push(id);
        }
        levels.push(level);
    }
    Ok(levels)
}
