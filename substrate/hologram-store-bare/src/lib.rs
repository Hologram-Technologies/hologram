#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-store-bare
//!
//! The bare-metal [`KappaStore`] over a raw [`BlockDevice`] (spec §5 + arch §11.3). No filesystem,
//! sectors are the only substrate. Layout:
//!
//! - **Dual header sectors** at LBA 0 and 1 (alternating writes): each carries `gen` + cursors;
//!   the higher-`gen` header with valid magic wins on open. Crash mid-write reverts atomically.
//! - **Index** = chained **κ-addressed leaf pages**: each leaf is a copy-on-write page with a
//!   `next_lba` and a list of `(κ_content, data_lba, data_sectors)` entries. Every page has its
//!   own κ identity (a BLAKE3 over its bytes); the header records the chain head LBA + its κ.
//! - **Pinned set** = a parallel chain of pages of κ-labels.
//! - **Data extents** = bump-allocated runs of sectors holding the `put` payloads, addressed by
//!   LBA from the leaf entries. The index never inlines content — large blobs don't bloat the
//!   index page.
//!
//! Writes are CoW: a `flush` allocates fresh LBAs for new pages, writes them, then flips the
//! inactive header. A torn write between page writes and header write leaves the previously-
//! committed header still active — the previous good state.
//!
//! `KappaStore` is sync but [`BlockDevice`] I/O is async, so device futures are driven by a minimal
//! `no_std` `block_on` (busy-poll) — immediately-ready on a RAM disk, interrupt-completing on real
//! hardware.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::future::Future;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use hashbrown::{HashMap, HashSet};
use hologram_space::BlockDevice;
use hologram_space::{
    address_bytes, references, Bytes, KappaLabel, KappaLabel71, KappaStore, RealizationRegistry,
    StoreError,
};
use spin::Mutex;

const MAGIC: &[u8; 8] = b"HGRMBARE";
/// Header format version. v2 introduced the dual-buffered headers + κ-page chain (PR #25).
/// v3 adds the persistent **free-extent list** (arch §11.3) so GC reclamation survives reboots.
/// v4 adds the **reboot-monotonic epoch** (arch §9 G-C1 → §11.3): on every successful `open`
/// the epoch is bumped, so the pair `(reboot_epoch, gen)` is a total order over all writes ever
/// made to the device — the cross-reboot ordering UorTime "since boot" cannot provide.
const VERSION: u64 = 4;
const PAGE_SECTORS: u64 = 8; // 8 × 512 B = 4 KiB pages
const NULL_LBA: u64 = u64::MAX;
type Key = [u8; 71];
type Digest = [u8; 32];

const HEADER_LBA_A: u64 = 0;
const HEADER_LBA_B: u64 = 1;
/// LBA where data + index allocation starts. We reserve room for the dual header + a small
/// pre-extent gap (so a corrupted neighbor page can't bleed into the header sectors).
const FIRST_ALLOC_LBA: u64 = 16;

// ── minimal no_std block_on (busy-poll a future to completion) ──
fn noop_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }
    RawWaker::new(
        core::ptr::null(),
        &RawWakerVTable::new(clone, no_op, no_op, no_op),
    )
}
fn block_on<F: Future>(f: F) -> F::Output {
    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut f = core::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        core::hint::spin_loop();
    }
}

#[derive(Default, Clone)]
struct Inner {
    /// κ → (data_lba, data_sectors): the resolved index from walking the leaf chain on open.
    entries: HashMap<Key, (u64, u32)>,
    /// In-memory bytes cache for `get` (built on open by reading data extents; new `put`s write
    /// through to disk + cache).
    cache: HashMap<Key, Bytes>,
    pinned: HashSet<Key>,
    /// Next free LBA for new data extents and CoW pages (bump allocator).
    alloc_cursor: u64,
    /// Active header's generation; the next flush writes generation `gen + 1` to the OTHER header.
    gen: u64,
    /// Which header slot holds the active root (false=A=LBA0, true=B=LBA1). The other is staged.
    active_slot_b: bool,
    /// Free extents the allocator can reuse before bumping the cursor (arch §11.3 "free-list").
    /// Each entry: `(lba, sectors)` for a previously-allocated extent whose κ was evicted. The
    /// allocator searches this list on `alloc` and falls back to bump when no fit exists.
    free_extents: Vec<(u64, u32)>,
    /// Reboot-monotonic epoch (G-C1): incremented on each successful `open` and persisted on
    /// every flush. Pair `(reboot_epoch, gen)` is a total order over all writes to this device,
    /// usable to order any two persisted runtime-state copies across reboots.
    reboot_epoch: u64,
}

/// A `KappaStore` persisted on a raw block device.
pub struct BareMetalKappaStore<D: BlockDevice> {
    device: D,
    inner: Mutex<Inner>,
}

fn backend(_e: impl core::fmt::Debug) -> StoreError {
    StoreError::BackendFailure("block-device")
}

impl<D: BlockDevice> BareMetalKappaStore<D> {
    /// Open a store on `device`, loading any previously-persisted image. An unformatted device
    /// loads as empty; a partially-torn header pair loads the surviving generation.
    pub fn open(device: D) -> Result<Self, StoreError> {
        let inner = Self::load(&device)?;
        Ok(Self {
            device,
            inner: Mutex::new(inner),
        })
    }

    /// Current reboot epoch (G-C1): the monotonic counter that increments on every successful
    /// `open` of a previously-formatted device. Pair `(reboot_epoch, generation)` gives a total
    /// ordering on any two persisted runtime-state copies, including across reboots.
    pub fn reboot_epoch(&self) -> u64 {
        self.inner.lock().reboot_epoch
    }

    /// Current write generation (monotonic per epoch — bumped on each `flush`).
    pub fn generation(&self) -> u64 {
        self.inner.lock().gen
    }

    fn ss(device: &D) -> usize {
        device.sector_size() as usize
    }

    fn page_bytes(device: &D) -> usize {
        Self::ss(device) * PAGE_SECTORS as usize
    }

    fn load(device: &D) -> Result<Inner, StoreError> {
        let header_a = Self::read_header(device, HEADER_LBA_A);
        let header_b = Self::read_header(device, HEADER_LBA_B);
        let chosen: Option<(bool, Header)> = match (header_a, header_b) {
            (Some(a), Some(b)) => Some(if b.gen > a.gen { (true, b) } else { (false, a) }),
            (Some(a), None) => Some((false, a)),
            (None, Some(b)) => Some((true, b)),
            (None, None) => None,
        };
        let Some((slot_b, h)) = chosen else {
            // Unformatted device: start fresh, cursor past the header reservation. Reboot epoch 1
            // is the first epoch ever (open of a fresh device == one reboot).
            return Ok(Inner {
                alloc_cursor: FIRST_ALLOC_LBA,
                reboot_epoch: 1,
                ..Default::default()
            });
        };

        let mut inner = Inner {
            entries: HashMap::new(),
            cache: HashMap::new(),
            pinned: HashSet::new(),
            alloc_cursor: h.alloc_cursor.max(FIRST_ALLOC_LBA),
            gen: h.gen,
            active_slot_b: slot_b,
            free_extents: Vec::new(),
            // G-C1: every successful open bumps the reboot epoch. The pair `(reboot_epoch, gen)`
            // is the total ordering on persisted runtime-state copies.
            reboot_epoch: h.reboot_epoch.saturating_add(1),
        };

        // Walk the leaf chain; verify each page's κ against the parent's recorded digest.
        let mut lba = h.leaf_head_lba;
        let mut expected = h.leaf_head_digest;
        while lba != NULL_LBA {
            let (page, digest) = Self::read_page(device, lba)?;
            if digest != expected {
                return Err(StoreError::BackendFailure(
                    "leaf page κ mismatch — index corrupted",
                ));
            }
            let (entries, next_lba, next_digest) = parse_leaf(&page)?;
            for (k, dlba, dsec) in entries {
                inner.entries.insert(k, (dlba, dsec));
            }
            lba = next_lba;
            expected = next_digest;
        }

        // Walk pins chain.
        let mut lba = h.pinned_head_lba;
        let mut expected = h.pinned_head_digest;
        while lba != NULL_LBA {
            let (page, digest) = Self::read_page(device, lba)?;
            if digest != expected {
                return Err(StoreError::BackendFailure(
                    "pinned page κ mismatch — index corrupted",
                ));
            }
            let (keys, next_lba, next_digest) = parse_pins(&page)?;
            for k in keys {
                inner.pinned.insert(k);
            }
            lba = next_lba;
            expected = next_digest;
        }

        // Walk the free-extent chain (arch §11.3).
        let mut lba = h.free_head_lba;
        let mut expected = h.free_head_digest;
        while lba != NULL_LBA {
            let (page, digest) = Self::read_page(device, lba)?;
            if digest != expected {
                return Err(StoreError::BackendFailure(
                    "free page κ mismatch — index corrupted",
                ));
            }
            let (entries, next_lba, next_digest) = parse_free(&page)?;
            for e in entries {
                inner.free_extents.push(e);
            }
            lba = next_lba;
            expected = next_digest;
        }

        // Re-read each data extent into the in-memory cache.
        for (k, &(dlba, dsec)) in &inner.entries {
            let ss = Self::ss(device) as u64;
            let mut buf = vec![0u8; (dsec as u64 * ss) as usize];
            block_on(device.read(dlba, dsec, &mut buf)).map_err(backend)?;
            // The extent buffer is sector-padded; the prefix that hashes to κ is the κ's content.
            // We don't know the exact length from the index entry alone (it's in sectors), so we
            // strip trailing zeros that won't round-trip. To avoid ambiguity, the extent records
            // the **exact length** as the first 4 bytes (LE).
            let payload_len = u32::from_le_bytes(buf[..4].try_into().unwrap_or([0; 4])) as usize;
            if payload_len + 4 > buf.len() {
                return Err(StoreError::BackendFailure(
                    "extent length header out of range",
                ));
            }
            let payload = &buf[4..4 + payload_len];
            // Verify the extent re-derives to κ (SPINE-4 self-check; a torn extent fails loud).
            let re = address_bytes(payload);
            if re.as_array() != k {
                return Err(StoreError::BackendFailure(
                    "data extent failed σ-axis verification",
                ));
            }
            inner.cache.insert(*k, Bytes::from(payload.to_vec()));
        }

        Ok(inner)
    }

    fn read_header(device: &D, lba: u64) -> Option<Header> {
        let ss = Self::ss(device);
        let mut buf = vec![0u8; ss];
        if block_on(device.read(lba, 1, &mut buf)).is_err() {
            return None;
        }
        Header::parse(&buf)
    }

    fn read_page(device: &D, lba: u64) -> Result<(Vec<u8>, Digest), StoreError> {
        let pb = Self::page_bytes(device);
        let mut buf = vec![0u8; pb];
        block_on(device.read(lba, PAGE_SECTORS as u32, &mut buf)).map_err(backend)?;
        let digest = blake3_digest(&buf);
        Ok((buf, digest))
    }

    /// Allocate `sectors` sectors. First reuses a freed extent (exact fit, or splits a larger one
    /// in-place); falls back to bumping `alloc_cursor`. The free-list is uor-native: every entry
    /// corresponds to an LBA whose κ has been evicted, so the bookkeeping is recoverable from the
    /// store's eviction record (no side-channel ledger).
    fn alloc(inner: &mut Inner, sectors: u32) -> u64 {
        // Best-fit: prefer exact, else smallest-larger to minimise residual fragmentation. With
        // entries assumed sorted by sectors descending isn't strictly necessary; the list is
        // typically small. We do a single pass.
        let mut best: Option<usize> = None;
        for (i, &(_, sec)) in inner.free_extents.iter().enumerate() {
            if sec < sectors {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(j) if inner.free_extents[j].1 > sec => best = Some(i),
                _ => {}
            }
            if sec == sectors {
                break;
            }
        }
        if let Some(i) = best {
            let (lba, sec) = inner.free_extents[i];
            if sec == sectors {
                inner.free_extents.swap_remove(i);
                return lba;
            }
            // Split: take the prefix, leave the suffix as a smaller free extent.
            inner.free_extents[i] = (lba + sectors as u64, sec - sectors);
            return lba;
        }
        // Fallback: bump.
        let lba = inner.alloc_cursor;
        inner.alloc_cursor += sectors as u64;
        lba
    }

    /// Write the entire in-memory state to disk: data extents (only new ones), leaf chain (always
    /// CoW-rewritten), pin chain (CoW), and finally the **inactive** header gets the new root.
    /// On torn write, the still-active header points to the prior committed state.
    fn flush(&self, inner: &mut Inner) -> Result<(), StoreError> {
        let ss = Self::ss(&self.device) as u64;
        let pb = Self::page_bytes(&self.device);

        // 1. Persist any new data extents. We rebuild `entries` from scratch each flush — any
        // blob present in `cache` gets a fresh extent allocated only if not already at an LBA.
        // Collect the writes first to avoid a simultaneous borrow of `inner.cache` (immutable) +
        // `inner.alloc_cursor` (mutable via Self::alloc).
        let mut writes: Vec<(Key, u64, u32, Vec<u8>)> = Vec::new();
        let mut new_entries: HashMap<Key, (u64, u32)> = HashMap::new();
        // Snapshot the new keys & bytes we need to extent-write, before allocating LBAs.
        let mut to_allocate: Vec<(Key, Vec<u8>)> = Vec::new();
        for (k, bytes) in &inner.cache {
            if let Some(&(lba, sec)) = inner.entries.get(k) {
                new_entries.insert(*k, (lba, sec));
                continue;
            }
            to_allocate.push((*k, bytes.as_ref().to_vec()));
        }
        for (k, bytes) in to_allocate {
            let payload_len = bytes.len() as u32;
            let total = 4 + bytes.len();
            let sectors = ((total as u64).div_ceil(ss)) as u32;
            let lba = Self::alloc(inner, sectors);
            let mut buf = vec![0u8; (sectors as u64 * ss) as usize];
            buf[..4].copy_from_slice(&payload_len.to_le_bytes());
            buf[4..4 + bytes.len()].copy_from_slice(&bytes);
            writes.push((k, lba, sectors, buf));
            new_entries.insert(k, (lba, sectors));
        }
        for (_, lba, sec, buf) in &writes {
            block_on(self.device.write(*lba, *sec, buf)).map_err(backend)?;
        }
        inner.entries = new_entries;

        // 2. CoW-write the leaf chain. Each leaf carries up to ~46 entries (4 KiB page).
        let entries_per_leaf = (pb - LEAF_HEADER_BYTES) / LEAF_ENTRY_BYTES;
        let mut sorted: Vec<(Key, u64, u32)> = inner
            .entries
            .iter()
            .map(|(k, &(l, s))| (*k, l, s))
            .collect();
        sorted.sort_by_key(|a| a.0);
        let leaf_chunks: Vec<&[(Key, u64, u32)]> = sorted.chunks(entries_per_leaf.max(1)).collect();
        let (leaf_head_lba, leaf_head_digest) = if leaf_chunks.is_empty() {
            (NULL_LBA, [0u8; 32])
        } else {
            // Build pages bottom-up so each can reference its `next_lba` + `next_digest`.
            let mut next_lba = NULL_LBA;
            let mut next_digest: Digest = [0; 32];
            let mut head_lba = 0u64;
            let mut head_digest: Digest = [0; 32];
            for chunk in leaf_chunks.iter().rev() {
                let lba = Self::alloc(inner, PAGE_SECTORS as u32);
                let page = build_leaf_page(chunk, next_lba, &next_digest, pb);
                let digest = blake3_digest(&page);
                block_on(self.device.write(lba, PAGE_SECTORS as u32, &page)).map_err(backend)?;
                next_lba = lba;
                next_digest = digest;
                head_lba = lba;
                head_digest = digest;
            }
            (head_lba, head_digest)
        };

        // 3. CoW-write the pin chain.
        let pins_per_page = (pb - PIN_HEADER_BYTES) / PIN_ENTRY_BYTES;
        let mut pin_sorted: Vec<Key> = inner.pinned.iter().copied().collect();
        pin_sorted.sort();
        let pin_chunks: Vec<&[Key]> = pin_sorted.chunks(pins_per_page.max(1)).collect();
        let (pinned_head_lba, pinned_head_digest) = if pin_chunks.is_empty() {
            (NULL_LBA, [0u8; 32])
        } else {
            let mut next_lba = NULL_LBA;
            let mut next_digest: Digest = [0; 32];
            let mut head_lba = 0u64;
            let mut head_digest: Digest = [0; 32];
            for chunk in pin_chunks.iter().rev() {
                let lba = Self::alloc(inner, PAGE_SECTORS as u32);
                let page = build_pin_page(chunk, next_lba, &next_digest, pb);
                let digest = blake3_digest(&page);
                block_on(self.device.write(lba, PAGE_SECTORS as u32, &page)).map_err(backend)?;
                next_lba = lba;
                next_digest = digest;
                head_lba = lba;
                head_digest = digest;
            }
            (head_lba, head_digest)
        };

        // 3b. CoW-write the free-extent chain (arch §11.3). Clone the entries so the borrow of
        // `inner.free_extents` ends before we call `alloc` (which mutates `inner.alloc_cursor`).
        let free_per_page = (pb - FREE_HEADER_BYTES) / FREE_ENTRY_BYTES;
        let free_snapshot: Vec<(u64, u32)> = inner.free_extents.clone();
        let free_chunks: Vec<&[(u64, u32)]> = free_snapshot.chunks(free_per_page.max(1)).collect();
        let (free_head_lba, free_head_digest) = if free_chunks.is_empty() {
            (NULL_LBA, [0u8; 32])
        } else {
            let mut next_lba = NULL_LBA;
            let mut next_digest: Digest = [0; 32];
            let mut head_lba = 0u64;
            let mut head_digest: Digest = [0; 32];
            for chunk in free_chunks.iter().rev() {
                let lba = Self::alloc(inner, PAGE_SECTORS as u32);
                let page = build_free_page(chunk, next_lba, &next_digest, pb);
                let digest = blake3_digest(&page);
                block_on(self.device.write(lba, PAGE_SECTORS as u32, &page)).map_err(backend)?;
                next_lba = lba;
                next_digest = digest;
                head_lba = lba;
                head_digest = digest;
            }
            (head_lba, head_digest)
        };

        // 4. Flush all data + page writes to ensure durability before the header swap.
        block_on(self.device.flush()).map_err(backend)?;

        // 5. Write the NEW header to the inactive slot with `gen+1`. The currently-active header
        //    is untouched until the new write completes, so a torn write reverts to it on reopen.
        let new_gen = inner.gen + 1;
        let inactive_lba = if inner.active_slot_b {
            HEADER_LBA_A
        } else {
            HEADER_LBA_B
        };
        let header = Header {
            gen: new_gen,
            alloc_cursor: inner.alloc_cursor,
            leaf_head_lba,
            leaf_head_digest,
            pinned_head_lba,
            pinned_head_digest,
            free_head_lba,
            free_head_digest,
            reboot_epoch: inner.reboot_epoch,
        };
        let mut buf = vec![0u8; Self::ss(&self.device)];
        header.write(&mut buf);
        block_on(self.device.write(inactive_lba, 1, &buf)).map_err(backend)?;
        block_on(self.device.flush()).map_err(backend)?;

        // 6. Commit: switch active slot. The old slot's older gen will be overwritten on the next
        //    flush — this is the alternating dual-buffer commit point.
        inner.gen = new_gen;
        inner.active_slot_b = !inner.active_slot_b;
        Ok(())
    }

    /// Reachability GC (spec §5.3 / §10.8) — identical semantics to the other backends; persists.
    /// Evicted κ's extents (LBA + sector count) are pushed to the free-list so future puts can
    /// reuse them (arch §11.3) — no LBA leak even on long-running stores.
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        let mut inner = self.inner.lock();
        let mut live: HashSet<Key> = HashSet::new();
        let mut frontier: Vec<Key> = inner.pinned.iter().copied().collect();
        while let Some(k) = frontier.pop() {
            if !live.insert(k) {
                continue;
            }
            if let Some(b) = inner.cache.get(&k) {
                if let Ok(refs) = references(b, registry) {
                    for r in refs {
                        frontier.push(*r.as_array());
                    }
                }
            }
        }
        let before = inner.cache.len();
        // Identify evicted κs to reclaim their extents.
        let evicted_keys: Vec<Key> = inner
            .cache
            .keys()
            .copied()
            .filter(|k| !live.contains(k))
            .collect();
        let mut reclaimed: Vec<(u64, u32)> = Vec::new();
        for k in &evicted_keys {
            if let Some((lba, sec)) = inner.entries.remove(k) {
                reclaimed.push((lba, sec));
            }
        }
        inner.cache.retain(|k, _| live.contains(k));
        // Merge into the free list.
        inner.free_extents.extend(reclaimed);
        let evicted = before - inner.cache.len();
        self.flush(&mut inner)?;
        Ok(evicted)
    }
}

impl<D: BlockDevice> KappaStore for BareMetalKappaStore<D> {
    fn put(&self, axis: &str, canonical_bytes: &[u8]) -> Result<KappaLabel71, StoreError> {
        if axis != "blake3" {
            return Err(StoreError::UnknownAxis);
        }
        let kappa = address_bytes(canonical_bytes);
        let mut inner = self.inner.lock();
        if inner.cache.contains_key(kappa.as_array()) {
            return Ok(kappa); // idempotent: no re-write
        }
        inner
            .cache
            .insert(*kappa.as_array(), Bytes::from(canonical_bytes.to_vec()));
        self.flush(&mut inner)?;
        Ok(kappa)
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(self.inner.lock().cache.get(kappa.as_array()).cloned())
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        self.inner.lock().cache.contains_key(kappa.as_array())
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let mut inner = self.inner.lock();
        inner.pinned.insert(*kappa.as_array());
        self.flush(&mut inner)
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let mut inner = self.inner.lock();
        if !inner.pinned.remove(kappa.as_array()) {
            return Err(StoreError::NotPinned);
        }
        self.flush(&mut inner)
    }

    fn iterate(&self) -> Vec<KappaLabel71> {
        self.inner
            .lock()
            .cache
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
        self.inner.lock().cache.len()
    }

    fn approximate_bytes(&self) -> u64 {
        self.inner
            .lock()
            .cache
            .values()
            .map(|b| b.len() as u64)
            .sum()
    }
}

impl<D: BlockDevice> hologram_space::GarbageCollect for BareMetalKappaStore<D> {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        BareMetalKappaStore::gc(self, registry)
    }
}

// ───────────────────────────── Page formats + headers ─────────────────────────────

struct Header {
    gen: u64,
    alloc_cursor: u64,
    leaf_head_lba: u64,
    leaf_head_digest: Digest,
    pinned_head_lba: u64,
    pinned_head_digest: Digest,
    /// Head of the **free-extent list** page chain (arch §11.3). `NULL_LBA` ⇒ no free extents.
    free_head_lba: u64,
    free_head_digest: Digest,
    /// **Reboot-monotonic epoch** (G-C1): bumped on every open; the pair `(reboot_epoch, gen)`
    /// is the total ordering on writes across reboots.
    reboot_epoch: u64,
}

impl Header {
    /// Bytes layout (≤ sector_size = 512 for the RAM device).
    fn parse(buf: &[u8]) -> Option<Self> {
        if &buf[..8] != MAGIC {
            return None;
        }
        let version = u64::from_le_bytes(buf[8..16].try_into().ok()?);
        if version != VERSION {
            return None;
        }
        let gen = u64::from_le_bytes(buf[16..24].try_into().ok()?);
        let alloc_cursor = u64::from_le_bytes(buf[24..32].try_into().ok()?);
        let leaf_head_lba = u64::from_le_bytes(buf[32..40].try_into().ok()?);
        let mut leaf_head_digest = [0u8; 32];
        leaf_head_digest.copy_from_slice(&buf[40..72]);
        let pinned_head_lba = u64::from_le_bytes(buf[72..80].try_into().ok()?);
        let mut pinned_head_digest = [0u8; 32];
        pinned_head_digest.copy_from_slice(&buf[80..112]);
        let free_head_lba = u64::from_le_bytes(buf[112..120].try_into().ok()?);
        let mut free_head_digest = [0u8; 32];
        free_head_digest.copy_from_slice(&buf[120..152]);
        let reboot_epoch = u64::from_le_bytes(buf[152..160].try_into().ok()?);
        Some(Self {
            gen,
            alloc_cursor,
            leaf_head_lba,
            leaf_head_digest,
            pinned_head_lba,
            pinned_head_digest,
            free_head_lba,
            free_head_digest,
            reboot_epoch,
        })
    }

    fn write(&self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(MAGIC);
        buf[8..16].copy_from_slice(&VERSION.to_le_bytes());
        buf[16..24].copy_from_slice(&self.gen.to_le_bytes());
        buf[24..32].copy_from_slice(&self.alloc_cursor.to_le_bytes());
        buf[32..40].copy_from_slice(&self.leaf_head_lba.to_le_bytes());
        buf[40..72].copy_from_slice(&self.leaf_head_digest);
        buf[72..80].copy_from_slice(&self.pinned_head_lba.to_le_bytes());
        buf[80..112].copy_from_slice(&self.pinned_head_digest);
        buf[112..120].copy_from_slice(&self.free_head_lba.to_le_bytes());
        buf[120..152].copy_from_slice(&self.free_head_digest);
        buf[152..160].copy_from_slice(&self.reboot_epoch.to_le_bytes());
    }
}

// Free-extent page layout: u16 num_entries | u64 next_lba | [32 B next_digest] | (lba u64, sec u32)*
const FREE_HEADER_BYTES: usize = 2 + 8 + 32;
const FREE_ENTRY_BYTES: usize = 8 + 4;

fn build_free_page(
    entries: &[(u64, u32)],
    next_lba: u64,
    next_digest: &Digest,
    page_size: usize,
) -> Vec<u8> {
    let mut page = vec![0u8; page_size];
    page[..2].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    page[2..10].copy_from_slice(&next_lba.to_le_bytes());
    page[10..42].copy_from_slice(next_digest);
    let mut off = FREE_HEADER_BYTES;
    for (lba, sec) in entries {
        page[off..off + 8].copy_from_slice(&lba.to_le_bytes());
        off += 8;
        page[off..off + 4].copy_from_slice(&sec.to_le_bytes());
        off += 4;
    }
    page
}

type FreeParse = (Vec<(u64, u32)>, u64, Digest);

fn parse_free(page: &[u8]) -> Result<FreeParse, StoreError> {
    if page.len() < FREE_HEADER_BYTES {
        return Err(StoreError::BackendFailure("free page short"));
    }
    let n = u16::from_le_bytes(page[..2].try_into().unwrap()) as usize;
    let next_lba = u64::from_le_bytes(page[2..10].try_into().unwrap());
    let mut next_digest = [0u8; 32];
    next_digest.copy_from_slice(&page[10..42]);
    let mut out = Vec::with_capacity(n);
    let mut off = FREE_HEADER_BYTES;
    for _ in 0..n {
        if off + FREE_ENTRY_BYTES > page.len() {
            return Err(StoreError::BackendFailure("free entries overflow"));
        }
        let lba = u64::from_le_bytes(page[off..off + 8].try_into().unwrap());
        off += 8;
        let sec = u32::from_le_bytes(page[off..off + 4].try_into().unwrap());
        off += 4;
        out.push((lba, sec));
    }
    Ok((out, next_lba, next_digest))
}

// Leaf layout: u16 num_entries | u64 next_lba | [32 B next_digest] | entries...
const LEAF_HEADER_BYTES: usize = 2 + 8 + 32;
const LEAF_ENTRY_BYTES: usize = 71 + 8 + 4; // κ + data_lba + sectors

fn build_leaf_page(
    entries: &[(Key, u64, u32)],
    next_lba: u64,
    next_digest: &Digest,
    page_size: usize,
) -> Vec<u8> {
    let mut page = vec![0u8; page_size];
    page[..2].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    page[2..10].copy_from_slice(&next_lba.to_le_bytes());
    page[10..42].copy_from_slice(next_digest);
    let mut off = LEAF_HEADER_BYTES;
    for (k, lba, sec) in entries {
        page[off..off + 71].copy_from_slice(k);
        off += 71;
        page[off..off + 8].copy_from_slice(&lba.to_le_bytes());
        off += 8;
        page[off..off + 4].copy_from_slice(&sec.to_le_bytes());
        off += 4;
    }
    page
}

/// A leaf-page parse: `(entries, next_lba, next_digest)`. Aliased for clippy.
type LeafParse = (Vec<(Key, u64, u32)>, u64, Digest);

fn parse_leaf(page: &[u8]) -> Result<LeafParse, StoreError> {
    if page.len() < LEAF_HEADER_BYTES {
        return Err(StoreError::BackendFailure("leaf page short"));
    }
    let n = u16::from_le_bytes(page[..2].try_into().unwrap()) as usize;
    let next_lba = u64::from_le_bytes(page[2..10].try_into().unwrap());
    let mut next_digest = [0u8; 32];
    next_digest.copy_from_slice(&page[10..42]);
    let mut out = Vec::with_capacity(n);
    let mut off = LEAF_HEADER_BYTES;
    for _ in 0..n {
        if off + LEAF_ENTRY_BYTES > page.len() {
            return Err(StoreError::BackendFailure("leaf entries overflow"));
        }
        let mut k = [0u8; 71];
        k.copy_from_slice(&page[off..off + 71]);
        off += 71;
        let lba = u64::from_le_bytes(page[off..off + 8].try_into().unwrap());
        off += 8;
        let sec = u32::from_le_bytes(page[off..off + 4].try_into().unwrap());
        off += 4;
        out.push((k, lba, sec));
    }
    Ok((out, next_lba, next_digest))
}

const PIN_HEADER_BYTES: usize = 2 + 8 + 32;
const PIN_ENTRY_BYTES: usize = 71;

fn build_pin_page(keys: &[Key], next_lba: u64, next_digest: &Digest, page_size: usize) -> Vec<u8> {
    let mut page = vec![0u8; page_size];
    page[..2].copy_from_slice(&(keys.len() as u16).to_le_bytes());
    page[2..10].copy_from_slice(&next_lba.to_le_bytes());
    page[10..42].copy_from_slice(next_digest);
    let mut off = PIN_HEADER_BYTES;
    for k in keys {
        page[off..off + 71].copy_from_slice(k);
        off += 71;
    }
    page
}

fn parse_pins(page: &[u8]) -> Result<(Vec<Key>, u64, Digest), StoreError> {
    if page.len() < PIN_HEADER_BYTES {
        return Err(StoreError::BackendFailure("pin page short"));
    }
    let n = u16::from_le_bytes(page[..2].try_into().unwrap()) as usize;
    let next_lba = u64::from_le_bytes(page[2..10].try_into().unwrap());
    let mut next_digest = [0u8; 32];
    next_digest.copy_from_slice(&page[10..42]);
    let mut out = Vec::with_capacity(n);
    let mut off = PIN_HEADER_BYTES;
    for _ in 0..n {
        if off + 71 > page.len() {
            return Err(StoreError::BackendFailure("pin entries overflow"));
        }
        let mut k = [0u8; 71];
        k.copy_from_slice(&page[off..off + 71]);
        off += 71;
        out.push(k);
    }
    Ok((out, next_lba, next_digest))
}

fn blake3_digest(bytes: &[u8]) -> Digest {
    use hologram_host::prism::vocabulary::Hasher;
    use hologram_host::HologramHasher;
    HologramHasher::initial().fold_bytes(bytes).finalize()
}

#[cfg(test)]
mod unit {
    extern crate std;
    use super::*;

    #[test]
    fn leaf_page_round_trip() {
        let k1 = address_bytes(b"a");
        let k2 = address_bytes(b"b");
        let entries = std::vec![(*k1.as_array(), 10, 1), (*k2.as_array(), 11, 2)];
        let next_digest = [0xAA; 32];
        let page = build_leaf_page(&entries, 99, &next_digest, 4096);
        let (parsed, next_lba, parsed_next_digest) = parse_leaf(&page).unwrap();
        assert_eq!(parsed, entries);
        assert_eq!(next_lba, 99);
        assert_eq!(parsed_next_digest, next_digest);
    }

    #[test]
    fn header_serialize_roundtrips() {
        let h = Header {
            gen: 7,
            alloc_cursor: 12345,
            leaf_head_lba: 16,
            leaf_head_digest: [0x55; 32],
            pinned_head_lba: 24,
            pinned_head_digest: [0xAA; 32],
            free_head_lba: 32,
            free_head_digest: [0xCC; 32],
            reboot_epoch: 42,
        };
        let mut buf = std::vec![0u8; 512];
        h.write(&mut buf);
        let back = Header::parse(&buf).unwrap();
        assert_eq!(back.gen, 7);
        assert_eq!(back.alloc_cursor, 12345);
        assert_eq!(back.leaf_head_lba, 16);
        assert_eq!(back.pinned_head_lba, 24);
        assert_eq!(back.leaf_head_digest, [0x55; 32]);
        assert_eq!(back.pinned_head_digest, [0xAA; 32]);
        assert_eq!(back.free_head_lba, 32);
        assert_eq!(back.free_head_digest, [0xCC; 32]);
        assert_eq!(back.reboot_epoch, 42);
    }

    #[test]
    fn free_page_round_trip() {
        let entries = std::vec![(100u64, 4u32), (200u64, 8u32), (300u64, 16u32)];
        let next_digest = [0xBB; 32];
        let page = build_free_page(&entries, 42, &next_digest, 4096);
        let (parsed, next_lba, parsed_next_digest) = parse_free(&page).unwrap();
        assert_eq!(parsed, entries);
        assert_eq!(next_lba, 42);
        assert_eq!(parsed_next_digest, next_digest);
    }
}
