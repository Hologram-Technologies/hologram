//! # hologram-store-mem
//!
//! The in-memory **reference [`KappaStore`]** and the substrate's conformance fixture. A value
//! lives in one `Arc<[u8]>`; `get` returns a cheap clone, never a copy (SP zero-copy floor).
//! `put` is idempotent (spec §10.2). Eviction is by **reachability from pinned roots** computed
//! over the realization registry's `references()` inverse projection (spec §5.3 / §10.8) — the one
//! uor-native graph walk, no separate edge index.

use crate::{
    address_bytes_axis, references, Bytes, KappaLabel, KappaLabel71, KappaStore,
    RealizationRegistry, StoreError,
};
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};
use spin::Mutex;

type Key = [u8; 71];

#[derive(Default)]
struct Inner {
    /// Hologram-canonical (blake3 / sha256) — 71-byte κ-labels (the hot path).
    blobs: HashMap<Key, Bytes>,
    pinned: HashSet<Key>,
    /// Foreign-axis content (sha3-256 / keccak256 / sha512), keyed by variable-width
    /// on-the-wire κ-label bytes (architecture §3.1 G-B1).
    blobs_wide: HashMap<Vec<u8>, Bytes>,
}

/// In-memory content-addressed store. `Send + Sync` via `spin::Mutex` (no_std-uniform).
#[derive(Default)]
pub struct MemKappaStore {
    inner: Mutex<Inner>,
}

impl MemKappaStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reachability-based GC (spec §5.3 / §10.8): retain every κ reachable from a pinned root via
    /// the registry's `references()`; evict the rest from local storage. Returns the eviction
    /// count. Never evicts a reachable κ (false-eviction is the one disallowed error). Bounded by
    /// `O(reachable · refs)` — the SP "bounded walk" floor; the addressing relation is never
    /// deleted, only the local bytes (SPINE-5).
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> usize {
        let live = self.reachable(registry);
        let mut g = self.inner.lock();
        let before = g.blobs.len();
        g.blobs.retain(|k, _| live.contains(k));
        before - g.blobs.len()
    }

    /// The reachable closure from the pinned roots (the GC mark set).
    pub fn reachable(&self, registry: RealizationRegistry<'_>) -> HashSet<Key> {
        let g = self.inner.lock();
        let mut live: HashSet<Key> = HashSet::new();
        let mut frontier: Vec<Key> = g.pinned.iter().copied().collect();
        while let Some(key) = frontier.pop() {
            if !live.insert(key) {
                continue;
            }
            // A κ contributes edges only if its bytes are present locally and parse as a known
            // realization; opaque/foreign content simply has no outgoing edges (a leaf).
            if let Some(bytes) = g.blobs.get(&key) {
                if let Ok(refs) = references(bytes, registry) {
                    for r in refs {
                        frontier.push(*r.as_array());
                    }
                }
            }
        }
        live
    }
}

impl KappaStore for MemKappaStore {
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError> {
        // Hot path: 71-byte axes (blake3 / sha256). Wider axes go through `put_axis`.
        let label_bytes =
            address_bytes_axis(axis, canonical_bytes).map_err(|_| StoreError::UnknownAxis)?;
        if label_bytes.len() != 71 {
            return Err(StoreError::UnknownAxis); // wider axis: use put_axis
        }
        let arr: [u8; 71] = label_bytes
            .as_slice()
            .try_into()
            .map_err(|_| StoreError::InvalidKappa)?;
        let kappa = KappaLabel::from_bytes(&arr).map_err(|_| StoreError::InvalidKappa)?;
        let mut g = self.inner.lock();
        // Idempotent (spec §10.2): identical bytes ⇒ same κ, no duplicate write.
        if !g.blobs.contains_key(&arr) {
            g.blobs.insert(arr, Bytes::from(canonical_bytes.to_vec()));
        }
        Ok(kappa)
    }

    fn put_axis(&self, axis: &str, bytes: &[u8]) -> Result<Vec<u8>, StoreError> {
        let label = address_bytes_axis(axis, bytes).map_err(|_| StoreError::UnknownAxis)?;
        let mut g = self.inner.lock();
        if label.len() == 71 {
            let arr: [u8; 71] = label.as_slice().try_into().unwrap();
            if !g.blobs.contains_key(&arr) {
                g.blobs.insert(arr, Bytes::from(bytes.to_vec()));
            }
        } else if !g.blobs_wide.contains_key(&label) {
            g.blobs_wide
                .insert(label.clone(), Bytes::from(bytes.to_vec()));
        }
        Ok(label)
    }

    fn get_axis(&self, label_bytes: &[u8]) -> Result<Option<Bytes>, StoreError> {
        let g = self.inner.lock();
        if label_bytes.len() == 71 {
            if let Ok(arr) = <[u8; 71]>::try_from(label_bytes) {
                if let Some(b) = g.blobs.get(&arr) {
                    return Ok(Some(b.clone()));
                }
            }
        }
        Ok(g.blobs_wide.get(label_bytes).cloned())
    }

    fn contains_axis(&self, label_bytes: &[u8]) -> bool {
        let g = self.inner.lock();
        if label_bytes.len() == 71 {
            if let Ok(arr) = <[u8; 71]>::try_from(label_bytes) {
                if g.blobs.contains_key(&arr) {
                    return true;
                }
            }
        }
        g.blobs_wide.contains_key(label_bytes)
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(self.inner.lock().blobs.get(kappa.as_array()).cloned())
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        self.inner.lock().blobs.contains_key(kappa.as_array())
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        self.inner.lock().pinned.insert(*kappa.as_array());
        Ok(())
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        if self.inner.lock().pinned.remove(kappa.as_array()) {
            Ok(())
        } else {
            Err(StoreError::NotPinned)
        }
    }

    fn iterate(&self) -> Vec<KappaLabel71> {
        self.inner
            .lock()
            .blobs
            .keys()
            .filter_map(|k| KappaLabel::from_bytes(k).ok())
            .collect()
    }

    fn pinned_roots(&self) -> Vec<KappaLabel71> {
        self.inner
            .lock()
            .pinned
            .iter()
            .filter_map(|k| KappaLabel::from_bytes(k).ok())
            .collect()
    }

    fn approximate_count(&self) -> usize {
        self.inner.lock().blobs.len()
    }

    fn approximate_bytes(&self) -> u64 {
        self.inner
            .lock()
            .blobs
            .values()
            .map(|b| b.len() as u64)
            .sum()
    }
}

impl crate::GarbageCollect for MemKappaStore {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, crate::StoreError> {
        Ok(MemKappaStore::gc(self, registry))
    }
}
