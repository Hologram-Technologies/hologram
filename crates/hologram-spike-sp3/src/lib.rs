//! **`SpikeSpace`** — a reference *external-crate* implementation of the [`hologram_space::Space`]
//! contract, using only its public API (no sealed traits, no in-tree privilege). It exists so the
//! conformance suite can witness two laws against a space defined *outside* the core crates:
//!
//! - **LAW-3 / D21** — the contract is open: the generic `hologram::Client<S: Space>` accepts this
//!   outside space with no special-casing (it compiling + running is the witness).
//! - **SP-3 / LAW-4** — composition: `Client` drives `compile → provision → run` over this space,
//!   the async network/boot seam calling straight into the synchronous compute hot path.
//!
//! Originally the P0.5 de-risk spike (D28) carried its own throwaway `Client`; that has been
//! superseded by the real `hologram::Client`, so only the reference space remains here.
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
// `async_trait` desugars to `Box`ed futures; not in the `no_std` prelude.
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use hologram_runtime::{MockEngine, Runtime};
use hologram_space::{
    Bytes, KappaLabel71, KappaSync, ManualClock, NoopSpawner, NullSurface, SeededEntropy, Space,
    SyncError,
};

/// A minimal concrete [`Space`]: a mock-engine [`Runtime`] over an in-memory store, plus a
/// sync seam that never fetches (forcing the local-store fallback). It demonstrates the
/// *composition*, not the network — a real sync seam arrives with `hologram-net`. The
/// `Space::Store` is the runtime's own store, so `store()` and `runtime()` share one content
/// store (no duplication).
pub struct SpikeSpace {
    runtime: Runtime<MockEngine, hologram_space::MemKappaStore>,
    sync: NullSync,
    entropy: SeededEntropy,
    clock: ManualClock,
    spawner: NoopSpawner,
    surface: NullSurface,
}

impl SpikeSpace {
    /// A fresh space with a mock-engine runtime over an empty in-memory store, and the
    /// deterministic reference entropy/clock (hermetic V&V).
    pub fn new() -> Self {
        Self {
            runtime: Runtime::new(MockEngine, hologram_space::MemKappaStore::new()),
            sync: NullSync,
            entropy: SeededEntropy::default(),
            clock: ManualClock::default(),
            spawner: NoopSpawner,
            surface: NullSurface,
        }
    }
}

impl Default for SpikeSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl Space for SpikeSpace {
    type Store = hologram_space::MemKappaStore;
    type Sync = NullSync;
    type Runtime = Runtime<MockEngine, hologram_space::MemKappaStore>;
    type Entropy = SeededEntropy;
    type Clock = ManualClock;
    type Spawner = NoopSpawner;
    type Surface = NullSurface;

    fn store(&self) -> &Self::Store {
        self.runtime.store()
    }
    fn sync(&self) -> &Self::Sync {
        &self.sync
    }
    fn runtime(&self) -> &Self::Runtime {
        &self.runtime
    }
    fn entropy(&self) -> &Self::Entropy {
        &self.entropy
    }
    fn clock(&self) -> &Self::Clock {
        &self.clock
    }
    fn spawner(&self) -> &Self::Spawner {
        &self.spawner
    }
    fn surface(&self) -> &Self::Surface {
        &self.surface
    }
}

/// A [`KappaSync`] that fetches nothing — the space proves composition via the local store, so
/// the async seam simply returns `Ok(None)` and the announce/discover/peering surface is inert.
pub struct NullSync;

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl KappaSync for NullSync {
    async fn fetch(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(None)
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(
        &self,
        _prefix: Option<&[u8]>,
        _limit: usize,
    ) -> alloc::vec::Vec<KappaLabel71> {
        alloc::vec::Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl KappaSync for NullSync {
    async fn fetch(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(None)
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(
        &self,
        _prefix: Option<&[u8]>,
        _limit: usize,
    ) -> alloc::vec::Vec<KappaLabel71> {
        alloc::vec::Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}
