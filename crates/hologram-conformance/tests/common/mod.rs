//! **`SpikeSpace`** — a reference implementation of the [`hologram_space::Space`] contract, used by
//! the conformance tests to witness two laws against the real `hologram::Client`:
//!
//! - **LAW-3 / D21** — the contract is open: the generic `hologram::Client<S: Space>` accepts this
//!   space with no special-casing (it compiling + running is the witness).
//! - **SP-3 / LAW-4** — composition: `Client` drives `compile → provision → run` over this space,
//!   the async network/boot seam calling straight into the synchronous compute hot path.
//!
//! Absorbed from the former standalone `hologram-spike-sp3` crate (the P0.5 spike, D28, whose
//! throwaway `Client` was long since superseded). It now lives as shared conformance **test support**
//! rather than a separate crate. The LAW-3 intent is preserved by construction: this type is built
//! from **only hologram's public API** — no sealed traits, no in-tree privilege — so an arbitrary
//! downstream implementor of the public `Space` contract is what `Client` is shown to accept.

#![allow(dead_code)] // each integration-test binary includes this module; not all use every item.

use hologram_runtime::{MockEngine, Runtime};
use hologram_space::{
    Bytes, KappaLabel71, KappaSync, ManualClock, NoopSpawner, NullSurface, SeededEntropy, Space,
    SyncError,
};

/// A minimal concrete [`Space`]: a mock-engine [`Runtime`] over an in-memory store, plus a sync seam
/// that never fetches (forcing the local-store fallback). It demonstrates *composition*, not the
/// network — a real sync seam arrives with `hologram-net`. The `Space::Store` is the runtime's own
/// store, so `store()` and `runtime()` share one content store (no duplication).
pub struct SpikeSpace {
    runtime: Runtime<MockEngine, hologram_space::MemKappaStore>,
    sync: NullSync,
    entropy: SeededEntropy,
    clock: ManualClock,
    spawner: NoopSpawner,
    surface: NullSurface,
}

impl SpikeSpace {
    /// A fresh space with a mock-engine runtime over an empty in-memory store, and the deterministic
    /// reference entropy/clock (hermetic V&V).
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

/// A [`KappaSync`] that fetches nothing — the space proves composition via the local store, so the
/// async seam simply returns `Ok(None)` and the announce/discover/peering surface is inert. The
/// conformance tests run native, so only the `Send` (non-wasm) variant is needed here.
pub struct NullSync;

#[async_trait::async_trait]
impl KappaSync for NullSync {
    async fn fetch(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(None)
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}
