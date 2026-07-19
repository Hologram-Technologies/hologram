//! # hologram_store::native — WASI/std redb-backed KappaStore
//!
//! The WASI/native [`KappaStore`] backend (spec §5.5), on a **redb** B-tree index.
//!
//! ## Sharding (spec §5.5)
//! Content larger than [`SHARD_THRESHOLD`] (64 KiB) is split into [`SHARD_SIZE`]-sized shards;
//! each shard is itself content-addressed and stored in the `INLINE` table, and the top-level
//! κ maps in the `SHARDED` table to a **shard manifest** (the ordered list of `(shard_κ, size)`).
//! This is the uor-native form: every fragment is a κ, identical fragments dedup across blobs,
//! and reassembly is by re-derivation. Externally `put`/`get`/`contains` are unchanged —
//! sharding is a backend refinement of the §5.5 layout.
//!
//! ## Bounded read-through cache (architecture §4 SP class)
//! A **size-bounded LRU** above redb makes `get` honor the SP zero-copy floor (consecutive gets
//! of the same κ share the same `Arc`). The cap is set per-store via [`CacheConfig`] — no
//! hardcoded ceiling. Bytes are evicted in LRU order when the total cached payload would exceed
//! `cache_max_bytes`. SPINE-6: the cap is a *resource budget*, not a structural cap on what is
//! storable; the persistent store is unbounded by this cache.
//!
//! Reachability `gc` walks the realization registry's `references()` exactly as the in-memory
//! reference does. The crate passes the **same TCK** as `hologram-store-mem`, plus G1/G2 own tests.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use hologram_space::{
    address_bytes, references, Bytes, KappaLabel, KappaLabel71, KappaStore, RealizationRegistry,
    StoreError,
};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

// ───────────────────────────── tables ─────────────────────────────

/// Inline content: keyed by `κ` (71-byte on-the-wire form). Used for content ≤ [`SHARD_THRESHOLD`]
/// AND for every shard fragment of larger content (shards are content-addressed independently).
const INLINE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("inline");

/// Shard manifest table: keyed by the top-level κ for content > [`SHARD_THRESHOLD`]; the value is
/// a packed list of `(shard_κ_71_bytes, u32 shard_size_le)`. Reassembly fetches each shard from
/// [`INLINE`] in order.
const SHARDED: TableDefinition<&[u8], &[u8]> = TableDefinition::new("sharded");

/// Pin set (reachability roots).
const PINNED: TableDefinition<&[u8], u8> = TableDefinition::new("pinned");

// ───────────────────────────── sharding policy ─────────────────────────────

/// Content larger than this triggers sharding (spec §5.5, the >64 KiB file-sharding split).
pub const SHARD_THRESHOLD: usize = 64 * 1024;

/// Shard granularity (bytes). Shards are content-addressed independently; identical shards across
/// blobs dedup automatically.
pub const SHARD_SIZE: usize = 64 * 1024;

/// Manifest entry width: 71-byte κ + 4-byte LE size.
const MANIFEST_ENTRY: usize = 71 + 4;

fn backend(_e: impl core::fmt::Debug) -> StoreError {
    StoreError::BackendFailure("redb")
}

fn encode_manifest(entries: &[([u8; 71], u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * MANIFEST_ENTRY);
    for (k, sz) in entries {
        out.extend_from_slice(k);
        out.extend_from_slice(&sz.to_le_bytes());
    }
    out
}

fn parse_manifest(bytes: &[u8]) -> Result<Vec<([u8; 71], u32)>, StoreError> {
    if !bytes.len().is_multiple_of(MANIFEST_ENTRY) {
        return Err(StoreError::BackendFailure("malformed shard manifest"));
    }
    let n = bytes.len() / MANIFEST_ENTRY;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * MANIFEST_ENTRY;
        let mut k = [0u8; 71];
        k.copy_from_slice(&bytes[off..off + 71]);
        let sz = u32::from_le_bytes([
            bytes[off + 71],
            bytes[off + 72],
            bytes[off + 73],
            bytes[off + 74],
        ]);
        out.push((k, sz));
    }
    Ok(out)
}

// ───────────────────────────── LRU cache ─────────────────────────────

/// Read-through cache configuration. `cache_max_bytes` is the soft ceiling on total **cached
/// payload bytes** (manifest reassemblies count by their reassembled size). Setting it to 0 is
/// rejected by [`NativeKappaStore::open_with_config`] — a zero-byte cache would force a redb
/// transaction on every `get` and violate the SP zero-copy floor. The architecture's "no
/// hardcoded ceiling" rule applies to the κ-graph; a memory budget for the cache is the very
/// resource budget pattern SPINE-6 allows.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Soft ceiling on total cached payload bytes.
    pub cache_max_bytes: u64,
}

impl CacheConfig {
    /// Production default: 256 MiB. Override via [`NativeKappaStore::open_with_config`] when the
    /// hosting process has a different memory budget.
    pub const DEFAULT_BYTES: u64 = 256 * 1024 * 1024;
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_max_bytes: Self::DEFAULT_BYTES,
        }
    }
}

/// Size-aware LRU over (κ → Bytes). Implemented as a HashMap of doubly-linked-list nodes.
/// Operations are O(1) amortized.
struct LruCache {
    map: HashMap<[u8; 71], Node>,
    head: Option<[u8; 71]>, // most-recently-used
    tail: Option<[u8; 71]>, // least-recently-used
    total_bytes: u64,
    max_bytes: u64,
}

struct Node {
    bytes: Bytes,
    size: u64,
    prev: Option<[u8; 71]>,
    next: Option<[u8; 71]>,
}

impl LruCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            map: HashMap::new(),
            head: None,
            tail: None,
            total_bytes: 0,
            max_bytes,
        }
    }

    fn get(&mut self, k: &[u8; 71]) -> Option<Bytes> {
        if !self.map.contains_key(k) {
            return None;
        }
        self.unlink(k);
        self.push_front(k);
        // Safety: present after unlink+push_front.
        self.map.get(k).map(|n| n.bytes.clone())
    }

    fn insert(&mut self, k: [u8; 71], bytes: Bytes) {
        if let Some(existing) = self.map.get(&k) {
            // Replace; account size delta.
            let old_size = existing.size;
            self.total_bytes = self.total_bytes.saturating_sub(old_size);
            self.unlink(&k);
            self.map.remove(&k);
        }
        let size = bytes.len() as u64;
        // A single oversize entry is still admitted (otherwise large-blob gets would always miss);
        // it will be evicted on the next insert.
        self.map.insert(
            k,
            Node {
                bytes,
                size,
                prev: None,
                next: None,
            },
        );
        self.push_front(&k);
        self.total_bytes = self.total_bytes.saturating_add(size);
        self.evict_until_within_budget();
    }

    fn remove(&mut self, k: &[u8; 71]) -> bool {
        if let Some(n) = self.map.remove(k) {
            self.total_bytes = self.total_bytes.saturating_sub(n.size);
            self.unlink_node(k, n.prev, n.next);
            true
        } else {
            false
        }
    }

    fn evict_until_within_budget(&mut self) {
        while self.total_bytes > self.max_bytes && self.map.len() > 1 {
            // Keep at least one entry to support a single oversize blob.
            let Some(tail_k) = self.tail else { break };
            self.remove(&tail_k);
        }
    }

    fn push_front(&mut self, k: &[u8; 71]) {
        let old_head = self.head;
        {
            let n = self.map.get_mut(k).expect("push_front: present");
            n.prev = None;
            n.next = old_head;
        }
        if let Some(h) = old_head {
            if let Some(hn) = self.map.get_mut(&h) {
                hn.prev = Some(*k);
            }
        }
        self.head = Some(*k);
        if self.tail.is_none() {
            self.tail = Some(*k);
        }
    }

    fn unlink(&mut self, k: &[u8; 71]) {
        let (prev, next) = match self.map.get(k) {
            Some(n) => (n.prev, n.next),
            None => return,
        };
        self.unlink_node(k, prev, next);
        if let Some(n) = self.map.get_mut(k) {
            n.prev = None;
            n.next = None;
        }
    }

    fn unlink_node(&mut self, k: &[u8; 71], prev: Option<[u8; 71]>, next: Option<[u8; 71]>) {
        if let Some(p) = prev {
            if let Some(pn) = self.map.get_mut(&p) {
                pn.next = next;
            }
        } else if self.head == Some(*k) {
            self.head = next;
        }
        if let Some(nx) = next {
            if let Some(nn) = self.map.get_mut(&nx) {
                nn.prev = prev;
            }
        } else if self.tail == Some(*k) {
            self.tail = prev;
        }
    }

    fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

// ───────────────────────────── store ─────────────────────────────

/// redb-backed content-addressed store. `Send + Sync` (redb `Database` is). A read-through
/// **bounded LRU** above redb makes `get` honor the SP zero-copy floor; sharding above
/// [`SHARD_THRESHOLD`] keeps individual redb values small (spec §5.5).
pub struct NativeKappaStore {
    db: Database,
    cache: Mutex<LruCache>,
}

impl NativeKappaStore {
    /// Open (or create) a store at `path` with the default [`CacheConfig`].
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, StoreError> {
        Self::open_with_config(path, CacheConfig::default())
    }

    /// Open (or create) a store at `path` with an explicit cache configuration.
    pub fn open_with_config(
        path: impl AsRef<std::path::Path>,
        config: CacheConfig,
    ) -> Result<Self, StoreError> {
        if config.cache_max_bytes == 0 {
            return Err(StoreError::BackendFailure(
                "cache_max_bytes must be > 0 (SP zero-copy floor requires a read-through cache)",
            ));
        }
        let db = Database::create(path).map_err(backend)?;
        Self::init(db, config)
    }

    /// In-memory redb (for tests / ephemeral nodes) with default config.
    pub fn in_memory() -> Result<Self, StoreError> {
        Self::in_memory_with_config(CacheConfig::default())
    }

    /// In-memory redb with explicit cache config.
    pub fn in_memory_with_config(config: CacheConfig) -> Result<Self, StoreError> {
        if config.cache_max_bytes == 0 {
            return Err(StoreError::BackendFailure(
                "cache_max_bytes must be > 0 (SP zero-copy floor requires a read-through cache)",
            ));
        }
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(backend)?;
        Self::init(db, config)
    }

    fn init(db: Database, config: CacheConfig) -> Result<Self, StoreError> {
        // Materialize all tables so empty-store reads/iters succeed.
        let tx = db.begin_write().map_err(backend)?;
        {
            tx.open_table(INLINE).map_err(backend)?;
            tx.open_table(SHARDED).map_err(backend)?;
            tx.open_table(PINNED).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(Self {
            db,
            cache: Mutex::new(LruCache::new(config.cache_max_bytes)),
        })
    }

    /// Bytes currently held in the LRU cache (test/diagnostic helper).
    pub fn cache_bytes(&self) -> u64 {
        self.cache.lock().unwrap().total_bytes()
    }

    /// Number of κ entries in the LRU cache (test/diagnostic helper).
    pub fn cache_entries(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Reachability-based GC (spec §5.3 / §10.8): retain every κ reachable from a pinned root via
    /// the registry's `references()`; evict the rest. For sharded κs whose top-level κ is
    /// unreachable, evict their fragments too — unless the same fragment is still referenced by
    /// another *reachable* sharded κ (content-shared shards stay alive). Returns the eviction
    /// count of **top-level** entries. The addressing relation is never deleted — only the local
    /// bytes (SPINE-5).
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        let live = self.reachable(registry)?;

        // Compute the set of fragments still alive under reachable sharded κs.
        let mut live_fragments: HashSet<[u8; 71]> = HashSet::new();
        {
            let tx = self.db.begin_read().map_err(backend)?;
            let st = tx.open_table(SHARDED).map_err(backend)?;
            for k in &live {
                if let Some(v) = st.get(k.as_slice()).map_err(backend)? {
                    for (sk, _) in parse_manifest(v.value())? {
                        live_fragments.insert(sk);
                    }
                }
            }
        }

        // Enumerate top-level κs (INLINE ∪ SHARDED) and pick the unreachable ones to evict.
        let top_level: Vec<[u8; 71]> = self.iterate().into_iter().map(|k| *k.as_array()).collect();
        let to_evict: Vec<[u8; 71]> = top_level
            .into_iter()
            .filter(|k| !live.contains(k))
            .collect();

        // For each evicted top-level, gather fragments to remove (those not held alive).
        let mut fragments_to_evict: Vec<[u8; 71]> = Vec::new();
        {
            let tx = self.db.begin_read().map_err(backend)?;
            let st = tx.open_table(SHARDED).map_err(backend)?;
            for k in &to_evict {
                if let Some(v) = st.get(k.as_slice()).map_err(backend)? {
                    for (sk, _) in parse_manifest(v.value())? {
                        if !live_fragments.contains(&sk) {
                            fragments_to_evict.push(sk);
                        }
                    }
                }
            }
        }

        // Perform the eviction in one write transaction.
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut inline = tx.open_table(INLINE).map_err(backend)?;
            let mut sharded = tx.open_table(SHARDED).map_err(backend)?;
            for k in &to_evict {
                sharded.remove(k.as_slice()).map_err(backend)?;
                inline.remove(k.as_slice()).map_err(backend)?;
            }
            for sk in &fragments_to_evict {
                inline.remove(sk.as_slice()).map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;

        // Invalidate evicted entries in the read-through cache.
        {
            let mut cache = self.cache.lock().unwrap();
            for k in &to_evict {
                cache.remove(k);
            }
            for sk in &fragments_to_evict {
                cache.remove(sk);
            }
        }
        Ok(to_evict.len())
    }

    /// The reachable closure from the pinned roots (top-level κs only). For sharded κs the
    /// reassembled bytes are inspected; for inline κs the stored bytes are.
    pub fn reachable(
        &self,
        registry: RealizationRegistry<'_>,
    ) -> Result<HashSet<[u8; 71]>, StoreError> {
        let mut live: HashSet<[u8; 71]> = HashSet::new();
        let mut frontier: Vec<[u8; 71]> =
            self.pinned_roots().iter().map(|k| *k.as_array()).collect();
        while let Some(key) = frontier.pop() {
            if !live.insert(key) {
                continue;
            }
            // Reassemble (or read inline) to extract refs.
            let label = KappaLabel::from_bytes(&key).map_err(|_| StoreError::InvalidKappa)?;
            if let Some(bytes) = self.get(&label)? {
                if let Ok(refs) = references(bytes.as_ref(), registry) {
                    for r in refs {
                        frontier.push(*r.as_array());
                    }
                }
            }
        }
        Ok(live)
    }

    // ── internal helpers ─────────────────────────────────────────────

    /// Store a single inline blob (idempotent). Used both for small top-level content AND for
    /// individual shard fragments of larger content.
    fn store_inline(&self, k: &[u8; 71], bytes: &[u8]) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(INLINE).map_err(backend)?;
            // Idempotent — skip if already present.
            if t.get(k.as_slice()).map_err(backend)?.is_none() {
                t.insert(k.as_slice(), bytes).map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn inline_contains(&self, k: &[u8; 71]) -> Result<bool, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(INLINE).map_err(backend)?;
        Ok(t.get(k.as_slice()).map_err(backend)?.is_some())
    }
}

impl KappaStore for NativeKappaStore {
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError> {
        if axis != "blake3" {
            return Err(StoreError::UnknownAxis);
        }
        let kappa = address_bytes(canonical_bytes);
        let key = *kappa.as_array();
        if self.contains(&kappa) {
            return Ok(kappa);
        }
        if canonical_bytes.len() <= SHARD_THRESHOLD {
            // Inline path: small content, single redb value.
            self.store_inline(&key, canonical_bytes)?;
        } else {
            // Sharded path (spec §5.5): split into SHARD_SIZE pieces, store each in INLINE,
            // store the manifest in SHARDED.
            let mut entries: Vec<([u8; 71], u32)> = Vec::new();
            for chunk in canonical_bytes.chunks(SHARD_SIZE) {
                let sk = address_bytes(chunk);
                let sk_arr = *sk.as_array();
                if !self.inline_contains(&sk_arr)? {
                    self.store_inline(&sk_arr, chunk)?;
                }
                entries.push((sk_arr, chunk.len() as u32));
            }
            let manifest = encode_manifest(&entries);
            let tx = self.db.begin_write().map_err(backend)?;
            {
                let mut t = tx.open_table(SHARDED).map_err(backend)?;
                t.insert(key.as_slice(), manifest.as_slice())
                    .map_err(backend)?;
            }
            tx.commit().map_err(backend)?;
        }
        Ok(kappa)
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        // 1. Cache hit (SP zero-copy floor: consecutive gets share the Arc).
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(b) = cache.get(kappa.as_array()) {
                return Ok(Some(b));
            }
        }

        // 2. Try SHARDED first (the top-level κ for a large blob) — if present, reassemble.
        let tx = self.db.begin_read().map_err(backend)?;
        if let Some(v) = tx
            .open_table(SHARDED)
            .map_err(backend)?
            .get(kappa.as_array().as_slice())
            .map_err(backend)?
        {
            let entries = parse_manifest(v.value())?;
            let total: usize = entries.iter().map(|(_, sz)| *sz as usize).sum();
            let mut buf: Vec<u8> = Vec::with_capacity(total);
            let inline = tx.open_table(INLINE).map_err(backend)?;
            for (sk, sz) in &entries {
                let shard = inline
                    .get(sk.as_slice())
                    .map_err(backend)?
                    .ok_or(StoreError::BackendFailure("missing shard"))?;
                let bytes = shard.value();
                if bytes.len() != *sz as usize {
                    return Err(StoreError::BackendFailure("shard size mismatch"));
                }
                buf.extend_from_slice(bytes);
            }
            let arc: Bytes = Arc::<[u8]>::from(buf.as_slice());
            self.cache
                .lock()
                .unwrap()
                .insert(*kappa.as_array(), arc.clone());
            return Ok(Some(arc));
        }

        // 3. Inline content (small top-level blobs or fragment-as-direct-lookup).
        let inline = tx.open_table(INLINE).map_err(backend)?;
        let bytes_opt = inline
            .get(kappa.as_array().as_slice())
            .map_err(backend)?
            .map(|v| Arc::<[u8]>::from(v.value()));
        if let Some(b) = &bytes_opt {
            self.cache
                .lock()
                .unwrap()
                .insert(*kappa.as_array(), b.clone());
        }
        Ok(bytes_opt)
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        (|| -> Result<bool, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            if tx
                .open_table(SHARDED)
                .map_err(backend)?
                .get(kappa.as_array().as_slice())
                .map_err(backend)?
                .is_some()
            {
                return Ok(true);
            }
            Ok(tx
                .open_table(INLINE)
                .map_err(backend)?
                .get(kappa.as_array().as_slice())
                .map_err(backend)?
                .is_some())
        })()
        .unwrap_or(false)
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(PINNED).map_err(backend)?;
            t.insert(kappa.as_array().as_slice(), 1u8)
                .map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let existed;
        {
            let mut t = tx.open_table(PINNED).map_err(backend)?;
            existed = t
                .remove(kappa.as_array().as_slice())
                .map_err(backend)?
                .is_some();
        }
        tx.commit().map_err(backend)?;
        if existed {
            Ok(())
        } else {
            Err(StoreError::NotPinned)
        }
    }

    fn iterate(&self) -> Vec<KappaLabel71> {
        // Top-level κs only: SHARDED keys (large) ∪ inline keys that are NOT shard fragments of
        // any sharded entry. This is what the user-facing semantics imply.
        (|| -> Result<Vec<KappaLabel71>, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let st = tx.open_table(SHARDED).map_err(backend)?;
            let it = tx.open_table(INLINE).map_err(backend)?;

            // Collect every fragment referenced by every sharded manifest.
            let mut fragments: HashSet<[u8; 71]> = HashSet::new();
            for row in st.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                for (sk, _) in parse_manifest(v.value())? {
                    fragments.insert(sk);
                }
            }

            let mut out = Vec::new();
            // Sharded top-level κs.
            for row in st.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                if let Ok(arr) = <[u8; 71]>::try_from(k.value()) {
                    if let Ok(label) = KappaLabel::from_bytes(&arr) {
                        out.push(label);
                    }
                }
            }
            // Inline top-level κs (skip fragments).
            for row in it.iter().map_err(backend)? {
                let (k, _) = row.map_err(backend)?;
                if let Ok(arr) = <[u8; 71]>::try_from(k.value()) {
                    if fragments.contains(&arr) {
                        continue;
                    }
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
        // Count of top-level κs (matches `iterate()`).
        self.iterate().len()
    }

    fn approximate_bytes(&self) -> u64 {
        (|| -> Result<u64, StoreError> {
            let tx = self.db.begin_read().map_err(backend)?;
            let inline = tx.open_table(INLINE).map_err(backend)?;
            let sharded = tx.open_table(SHARDED).map_err(backend)?;
            let mut total = 0u64;
            for row in inline.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                total += v.value().len() as u64;
            }
            for row in sharded.iter().map_err(backend)? {
                let (_, v) = row.map_err(backend)?;
                total += v.value().len() as u64;
            }
            Ok(total)
        })()
        .unwrap_or(0)
    }
}

impl hologram_space::GarbageCollect for NativeKappaStore {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        NativeKappaStore::gc(self, registry)
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn lru_evicts_in_lru_order_under_byte_budget() {
        let mut c = LruCache::new(100);
        let k = |b: u8| {
            let mut a = [0u8; 71];
            a[0] = b;
            a
        };
        let bytes = |n: usize| -> Bytes { Arc::<[u8]>::from(vec![0u8; n].as_slice()) };
        c.insert(k(1), bytes(40)); // total=40
        c.insert(k(2), bytes(40)); // total=80
        c.insert(k(3), bytes(40)); // total=120 → evicts k(1) → total=80
        assert_eq!(c.total_bytes(), 80);
        assert!(c.get(&k(1)).is_none());
        assert!(c.get(&k(2)).is_some()); // k(2) is now most-recent
        c.insert(k(4), bytes(40)); // total=120 → evicts k(3) (LRU) → total=80
        assert!(c.get(&k(3)).is_none());
        assert!(c.get(&k(2)).is_some());
        assert!(c.get(&k(4)).is_some());
    }

    #[test]
    fn lru_admits_single_oversize_entry_but_evicts_on_next_insert() {
        let mut c = LruCache::new(100);
        let k = |b: u8| {
            let mut a = [0u8; 71];
            a[0] = b;
            a
        };
        let bytes = |n: usize| -> Bytes { Arc::<[u8]>::from(vec![0u8; n].as_slice()) };
        // Single oversize entry survives (we always keep ≥1 to support a large blob).
        c.insert(k(1), bytes(500));
        assert!(c.get(&k(1)).is_some());
        // Next insert pushes us back to ≤100 by evicting the oversize entry.
        c.insert(k(2), bytes(50));
        assert!(c.get(&k(1)).is_none());
        assert!(c.get(&k(2)).is_some());
    }

    #[test]
    fn manifest_round_trip() {
        let entries = vec![([1u8; 71], 1234u32), ([2u8; 71], 5678u32)];
        let encoded = encode_manifest(&entries);
        assert_eq!(encoded.len(), 2 * MANIFEST_ENTRY);
        let decoded = parse_manifest(&encoded).unwrap();
        assert_eq!(decoded, entries);
    }

    #[test]
    fn manifest_malformed_fails_loud() {
        // Not a multiple of MANIFEST_ENTRY → backend failure (no fallback).
        assert!(parse_manifest(&[0u8; 70]).is_err());
    }
}
