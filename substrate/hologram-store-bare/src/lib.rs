#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-store-bare
//!
//! The bare-metal [`KappaStore`] over a raw [`BlockDevice`] (bare-metal spec §5) — no filesystem,
//! sectors are the only substrate. It formats a device (header sector + a serialized image of the
//! κ→bytes map and pinned-roots set) and persists every mutation, so the store survives a reboot
//! (reopening the same device). Passes the **shared TCK** identically to the mem/redb backends.
//!
//! `KappaStore` is sync but [`BlockDevice`] I/O is async, so device futures are driven by a minimal
//! `no_std` `block_on` (busy-poll) — immediately-ready on a RAM disk, interrupt-completing on real
//! hardware. The whole-image persistence here is the correctness-equivalent of the §5.2 B-tree +
//! extent allocator (which is the scale optimization, tracked separately — cf. redb inline-all).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::future::Future;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use hashbrown::{HashMap, HashSet};
use hologram_bare_hal::BlockDevice;
use hologram_substrate_core::{
    address_bytes, references, Bytes, KappaLabel, KappaLabel71, KappaStore, RealizationRegistry,
    StoreError,
};
use spin::Mutex;

const MAGIC: &[u8; 8] = b"HGRMBARE";
type Key = [u8; 71];

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

#[derive(Default)]
struct Inner {
    blobs: HashMap<Key, Bytes>,
    pinned: HashSet<Key>,
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
    /// Open a store on `device`, loading any previously-persisted image (empty if unformatted).
    pub fn open(device: D) -> Result<Self, StoreError> {
        let inner = Self::load(&device)?;
        Ok(Self {
            device,
            inner: Mutex::new(inner),
        })
    }

    fn ss(device: &D) -> usize {
        device.sector_size() as usize
    }

    fn load(device: &D) -> Result<Inner, StoreError> {
        let ss = Self::ss(device);
        let mut header = vec![0u8; ss];
        block_on(device.read(0, 1, &mut header)).map_err(backend)?;
        if &header[..8] != MAGIC {
            return Ok(Inner::default()); // unformatted → empty
        }
        let image_len = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        if image_len == 0 {
            return Ok(Inner::default());
        }
        let sectors = image_len.div_ceil(ss);
        let mut buf = vec![0u8; sectors * ss];
        block_on(device.read(1, sectors as u32, &mut buf)).map_err(backend)?;
        Self::deserialize(&buf[..image_len])
    }

    fn flush(&self, inner: &Inner) -> Result<(), StoreError> {
        let ss = Self::ss(&self.device);
        let image = Self::serialize(inner);
        let mut header = vec![0u8; ss];
        header[..8].copy_from_slice(MAGIC);
        header[8..16].copy_from_slice(&(image.len() as u64).to_le_bytes());
        block_on(self.device.write(0, 1, &header)).map_err(backend)?;
        if !image.is_empty() {
            let sectors = image.len().div_ceil(ss);
            let mut buf = vec![0u8; sectors * ss];
            buf[..image.len()].copy_from_slice(&image);
            block_on(self.device.write(1, sectors as u32, &buf)).map_err(backend)?;
        }
        block_on(self.device.flush()).map_err(backend)?;
        Ok(())
    }

    fn serialize(inner: &Inner) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(inner.blobs.len() as u32).to_le_bytes());
        for (k, v) in &inner.blobs {
            out.extend_from_slice(k);
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            out.extend_from_slice(v);
        }
        out.extend_from_slice(&(inner.pinned.len() as u32).to_le_bytes());
        for k in &inner.pinned {
            out.extend_from_slice(k);
        }
        out
    }

    fn deserialize(buf: &[u8]) -> Result<Inner, StoreError> {
        let mut inner = Inner::default();
        let mut cur = 0usize;
        let rd_u32 = |b: &[u8], c: &mut usize| -> Option<u32> {
            let v = b.get(*c..*c + 4)?.try_into().ok().map(u32::from_le_bytes)?;
            *c += 4;
            Some(v)
        };
        let nb = rd_u32(buf, &mut cur).ok_or(StoreError::BackendFailure("trunc"))?;
        for _ in 0..nb {
            let key: Key = buf
                .get(cur..cur + 71)
                .and_then(|s| s.try_into().ok())
                .ok_or(StoreError::BackendFailure("trunc"))?;
            cur += 71;
            let len = rd_u32(buf, &mut cur).ok_or(StoreError::BackendFailure("trunc"))? as usize;
            let bytes = buf
                .get(cur..cur + len)
                .ok_or(StoreError::BackendFailure("trunc"))?;
            cur += len;
            inner.blobs.insert(key, Bytes::from(bytes.to_vec()));
        }
        let np = rd_u32(buf, &mut cur).ok_or(StoreError::BackendFailure("trunc"))?;
        for _ in 0..np {
            let key: Key = buf
                .get(cur..cur + 71)
                .and_then(|s| s.try_into().ok())
                .ok_or(StoreError::BackendFailure("trunc"))?;
            cur += 71;
            inner.pinned.insert(key);
        }
        Ok(inner)
    }

    /// Reachability GC (spec §5.3/§10.8) — identical semantics to the other backends; persists.
    pub fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        let mut inner = self.inner.lock();
        let mut live: HashSet<Key> = HashSet::new();
        let mut frontier: Vec<Key> = inner.pinned.iter().copied().collect();
        while let Some(k) = frontier.pop() {
            if !live.insert(k) {
                continue;
            }
            if let Some(b) = inner.blobs.get(&k) {
                if let Ok(refs) = references(b, registry) {
                    for r in refs {
                        frontier.push(*r.as_array());
                    }
                }
            }
        }
        let before = inner.blobs.len();
        inner.blobs.retain(|k, _| live.contains(k));
        let evicted = before - inner.blobs.len();
        self.flush(&inner)?;
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
        if inner.blobs.contains_key(kappa.as_array()) {
            return Ok(kappa); // idempotent — no rewrite
        }
        inner
            .blobs
            .insert(*kappa.as_array(), Bytes::from(canonical_bytes.to_vec()));
        self.flush(&inner)?;
        Ok(kappa)
    }

    fn get(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(self.inner.lock().blobs.get(kappa.as_array()).cloned())
    }

    fn contains(&self, kappa: &KappaLabel71) -> bool {
        self.inner.lock().blobs.contains_key(kappa.as_array())
    }

    fn pin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let mut inner = self.inner.lock();
        inner.pinned.insert(*kappa.as_array());
        self.flush(&inner)
    }

    fn unpin(&self, kappa: &KappaLabel71) -> Result<(), StoreError> {
        let mut inner = self.inner.lock();
        if !inner.pinned.remove(kappa.as_array()) {
            return Err(StoreError::NotPinned);
        }
        self.flush(&inner)
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

impl<D: BlockDevice> hologram_substrate_core::GarbageCollect for BareMetalKappaStore<D> {
    fn gc(&self, registry: RealizationRegistry<'_>) -> Result<usize, StoreError> {
        BareMetalKappaStore::gc(self, registry)
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn image_serialize_roundtrips() {
        let mut inner = Inner::default();
        let k = address_bytes(b"x");
        inner
            .blobs
            .insert(*k.as_array(), Bytes::from(b"hello".to_vec()));
        inner.pinned.insert(*k.as_array());
        let img = BareMetalKappaStore::<hologram_bare_hal::RamBlockDevice>::serialize(&inner);
        let back =
            BareMetalKappaStore::<hologram_bare_hal::RamBlockDevice>::deserialize(&img).unwrap();
        assert_eq!(back.blobs.get(k.as_array()).unwrap().as_ref(), b"hello");
        assert!(back.pinned.contains(k.as_array()));
    }
}
