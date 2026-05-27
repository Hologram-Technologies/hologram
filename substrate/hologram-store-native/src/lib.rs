//! # hologram-store-native
//!
//! The WASI/native [`KappaStore`] backend (spec §5.5), on a **redb** B-tree index. Content is
//! stored inline in the value column (the >64 KiB file-sharding split of §5.5 is a perf refinement
//! tracked separately; inline-all is correctness-equivalent). Reachability `gc` walks the
//! realization registry's `references()` exactly as the in-memory reference does, and the crate
//! passes the **same TCK** as `hologram-store-mem`.

use std::collections::HashSet;
use std::sync::Arc;

use hologram_substrate_core::{
    address_bytes, references, Bytes, KappaLabel, KappaLabel71, KappaStore, RealizationRegistry,
    StoreError,
};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};

const BLOBS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("blobs");
const PINNED: TableDefinition<&[u8], u8> = TableDefinition::new("pinned");

fn backend(_e: impl core::fmt::Debug) -> StoreError {
    StoreError::BackendFailure("redb")
}

/// redb-backed content-addressed store. `Send + Sync` (redb `Database` is).
pub struct NativeKappaStore {
    db: Database,
}

impl NativeKappaStore {
    /// Open (or create) a store at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, StoreError> {
        let db = Database::create(path).map_err(backend)?;
        Self::init(db)
    }

    /// In-memory redb (for tests / ephemeral nodes) — still the real B-tree engine.
    pub fn in_memory() -> Result<Self, StoreError> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(backend)?;
        Self::init(db)
    }

    fn init(db: Database) -> Result<Self, StoreError> {
        // Materialize both tables so empty-store reads/iters succeed.
        let tx = db.begin_write().map_err(backend)?;
        {
            tx.open_table(BLOBS).map_err(backend)?;
            tx.open_table(PINNED).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(Self { db })
    }

    /// Reachability-based GC (spec §5.3 / §10.8): retain every κ reachable from a pinned root via
    /// the registry's `references()`; evict the rest. Returns the eviction count. Never evicts a
    /// reachable κ. The addressing relation is never deleted — only the local bytes (SPINE-5).
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        let live = self.reachable(registry)?;
        let to_evict: Vec<[u8; 71]> = self
            .iterate()
            .into_iter()
            .map(|k| *k.as_array())
            .filter(|k| !live.contains(k))
            .collect();
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(BLOBS).map_err(backend)?;
            for k in &to_evict {
                t.remove(k.as_slice()).map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        Ok(to_evict.len())
    }

    /// The reachable closure from the pinned roots.
    pub fn reachable(&self, registry: RealizationRegistry<'_>) -> Result<HashSet<[u8; 71]>, StoreError> {
        let mut live: HashSet<[u8; 71]> = HashSet::new();
        let mut frontier: Vec<[u8; 71]> = self.pinned_roots().iter().map(|k| *k.as_array()).collect();
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(BLOBS).map_err(backend)?;
        while let Some(key) = frontier.pop() {
            if !live.insert(key) {
                continue;
            }
            if let Some(v) = t.get(key.as_slice()).map_err(backend)? {
                if let Ok(refs) = references(v.value(), registry) {
                    for r in refs {
                        frontier.push(*r.as_array());
                    }
                }
            }
        }
        Ok(live)
    }
}

impl KappaStore for NativeKappaStore {
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError> {
        if axis != "blake3" {
            return Err(StoreError::UnknownAxis);
        }
        let kappa = address_bytes(canonical_bytes);
        let key = *kappa.as_array();
        // Idempotent: skip the write if the κ is already present (no duplicate write, §10.2).
        if self.contains(&kappa) {
            return Ok(kappa);
        }
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(BLOBS).map_err(backend)?;
            t.insert(key.as_slice(), canonical_bytes).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(kappa)
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(BLOBS).map_err(backend)?;
        Ok(t
            .get(kappa.as_array().as_slice())
            .map_err(backend)?
            .map(|v| Arc::from(v.value())))
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        (|| -> Result<bool, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let t = tx.open_table(BLOBS).map_err(backend)?;
            Ok(t.get(kappa.as_array().as_slice()).map_err(backend)?.is_some())
        })()
        .unwrap_or(false)
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(PINNED).map_err(backend)?;
            t.insert(kappa.as_array().as_slice(), 1u8).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let existed;
        {
            let mut t = tx.open_table(PINNED).map_err(backend)?;
            existed = t.remove(kappa.as_array().as_slice()).map_err(backend)?.is_some();
        }
        tx.commit().map_err(backend)?;
        if existed {
            Ok(())
        } else {
            Err(StoreError::NotPinned)
        }
    }

    fn iterate(&self) -> Vec<KappaLabel71> {
        (|| -> Result<Vec<KappaLabel71>, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let t = tx.open_table(BLOBS).map_err(backend)?;
            let mut out = Vec::new();
            for row in t.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                if let Ok(arr) = <[u8; 71]>::try_from(k.value()) {
                    if let Ok(label) = KappaLabel::from_bytes(&arr) {
                        out.push(label);
                    }
                }
            }
            Ok(out)
        })()
        .unwrap_or_default()
    }

    fn pinned_roots(&self) -> Vec<KappaLabel71> {
        (|| -> Result<Vec<KappaLabel71>, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let t = tx.open_table(PINNED).map_err(backend)?;
            let mut out = Vec::new();
            for row in t.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                if let Ok(arr) = <[u8; 71]>::try_from(k.value()) {
                    if let Ok(label) = KappaLabel::from_bytes(&arr) {
                        out.push(label);
                    }
                }
            }
            Ok(out)
        })()
        .unwrap_or_default()
    }

    fn approximate_count(&self) -> usize {
        (|| -> Result<usize, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let t = tx.open_table(BLOBS).map_err(backend)?;
            Ok(t.len().map_err(backend)? as usize)
        })()
        .unwrap_or(0)
    }

    fn approximate_bytes(&self) -> u64 {
        (|| -> Result<u64, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let t = tx.open_table(BLOBS).map_err(backend)?;
            let mut total = 0u64;
            for row in t.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                total += v.value().len() as u64;
            }
            Ok(total)
        })()
        .unwrap_or(0)
    }
}

impl hologram_substrate_core::GarbageCollect for NativeKappaStore {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        NativeKappaStore::gc(self, registry)
    }
}
