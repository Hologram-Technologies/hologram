//! BLAKE3-deduped weight store (spec X.3 + wiki ADR-031).
//!
//! The content-addressing hash routes through hologram's canonical
//! `Hasher<32>` selection — `hologram_types::HologramHasher`, which is
//! a re-export of `prism::crypto::Blake3Hasher`. No direct dependency
//! on the `blake3` crate; per ADR-031 hologram consumes its
//! content-addressing primitive from prism-crypto.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use hashbrown::HashMap;
use hologram_types::HologramHasher;
use prism::vocabulary::Hasher;

use crate::address::{label_from_fingerprint, ContentLabel};

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

    /// The content κ-label a weight with this fingerprint addresses to —
    /// `blake3:<hex>` over the same 32-byte digest [`address_bytes`] would
    /// mint from the body. Derivable from the fingerprint **alone**, so a
    /// paged load can key residency by label without ever pulling the body.
    ///
    /// [`address_bytes`]: crate::address::address_bytes
    #[must_use]
    pub fn content_label(&self) -> ContentLabel {
        label_from_fingerprint(&self.0)
    }
}

/// Source of weight bodies for a session — the inversion of [`WeightStore`]
/// from an owned map into a provider the host implements. A paged session
/// borrows or pages ranges from here instead of copying every body resident
/// at load, so the arena becomes a **window** over the provider (the weight
/// tier's analog of the archive-tier κ-resolution that already pages).
///
/// Residency is orthogonal to identity: a range served here hashes to the
/// same κ it was addressed by, so derivation keys and kernels are unchanged.
/// A missing weight is **page-in-and-retry**, never recompute — a leaf
/// constant has no cone (contrast warm-start's recompute-on-miss).
pub trait WeightProvider {
    /// Full body length for a fingerprint, or `None` if the provider does
    /// not have it.
    fn size(&self, fp: WeightFingerprint) -> Option<usize>;

    /// Bytes `[offset, offset + len)` of the body, or `None` if absent or
    /// out of range. `Cow` so an in-memory provider borrows (zero copy) and
    /// a paging (OPFS-backed) provider can own the fetched range.
    fn get_range(&self, fp: WeightFingerprint, offset: usize, len: usize) -> Option<Cow<'_, [u8]>>;
}

/// The owned in-memory store is itself the default provider (borrowing its
/// bodies), so the fully-resident load path and the paged path share one
/// interface — a bounded host swaps in its own provider by relaxing the
/// residency budget, with no change to the executable.
impl WeightProvider for WeightStore {
    fn size(&self, fp: WeightFingerprint) -> Option<usize> {
        self.get(fp).map(<[u8]>::len)
    }

    fn get_range(&self, fp: WeightFingerprint, offset: usize, len: usize) -> Option<Cow<'_, [u8]>> {
        self.get(fp)
            .and_then(|b| b.get(offset..offset.checked_add(len)?))
            .map(Cow::Borrowed)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::address_bytes;

    #[test]
    fn fingerprint_label_equals_body_address() {
        // The label derived from a fingerprint alone must equal the label
        // `address_bytes` mints from the body — the invariant the paged load
        // relies on to key residency without materializing the weight.
        for body in [
            &b""[..],
            b"w",
            b"a-larger-weight-body-0123456789",
            &[0xABu8; 257],
        ] {
            let fp = WeightFingerprint::of(body);
            assert_eq!(
                fp.content_label(),
                address_bytes(body),
                "len {}",
                body.len()
            );
        }
    }

    #[test]
    fn weight_store_provider_serves_ranges() {
        let mut s = WeightStore::new();
        let body = (0..200u16).map(|i| i as u8).collect::<Vec<_>>();
        let fp = s.insert(body.clone());
        assert_eq!(s.size(fp), Some(200));
        assert_eq!(s.get_range(fp, 10, 5).unwrap().as_ref(), &body[10..15]);
        assert_eq!(s.get_range(fp, 0, 200).unwrap().as_ref(), &body[..]);
        assert!(s.get_range(fp, 190, 20).is_none()); // out of range
        assert!(s.size(WeightFingerprint([0xFF; 32])).is_none()); // absent
    }
}
