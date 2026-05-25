//! BLAKE3-deduped weight store (spec X.3 + wiki ADR-031).
//!
//! The content-addressing hash routes through hologram's canonical
//! `Hasher<32>` selection — `hologram_host::HologramHasher`, which is
//! a re-export of `prism::crypto::Blake3Hasher`. No direct dependency
//! on the `blake3` crate; per ADR-031 hologram consumes its
//! content-addressing primitive from prism-crypto.

use alloc::vec::Vec;

use hashbrown::HashMap;
use hologram_host::HologramHasher;
use prism::vocabulary::Hasher;

/// 32-byte content fingerprint over a weight body. Computed via
/// prism-crypto's Blake3 `HashAxis` impl (the canonical hologram
/// hasher selection per spec III.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightFingerprint(pub [u8; 32]);

impl WeightFingerprint {
    /// Compute the fingerprint of a byte sequence via the canonical
    /// hasher (`prism::crypto::Blake3Hasher` through `Hasher<32>`).
    pub fn of(bytes: &[u8]) -> Self {
        Self(HologramHasher::initial().fold_bytes(bytes).finalize())
    }
}

#[derive(Debug, Default, Clone)]
pub struct WeightStore {
    /// Body keyed by fingerprint.
    bodies: HashMap<WeightFingerprint, Vec<u8>>,
}

impl WeightStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert weight bytes; returns the dedup key. Duplicate bodies share storage.
    pub fn insert(&mut self, bytes: Vec<u8>) -> WeightFingerprint {
        let fp = WeightFingerprint::of(&bytes);
        self.bodies.entry(fp).or_insert(bytes);
        fp
    }

    pub fn get(&self, fp: WeightFingerprint) -> Option<&[u8]> {
        self.bodies.get(&fp).map(|v| v.as_slice())
    }

    pub fn entries(&self) -> impl Iterator<Item = (&WeightFingerprint, &Vec<u8>)> {
        self.bodies.iter()
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
}
