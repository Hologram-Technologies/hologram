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
use hologram_space::{Bytes, KappaLabel71, Resolver, Space, StoreError};

/// A minimal concrete [`Space`]: a mock-engine [`Runtime`] over an in-memory store, plus a
/// resolver that never resolves (forcing the local-store fallback). It demonstrates the
/// *composition*, not the network — a real resolver arrives with `hologram-net`. The
/// `Space::Store` is the runtime's own store, so `store()` and `runtime()` share one content
/// store (no duplication).
pub struct SpikeSpace {
    runtime: Runtime<MockEngine, hologram_space::MemKappaStore>,
    resolver: NullResolver,
}

impl SpikeSpace {
    /// A fresh space with a mock-engine runtime over an empty in-memory store.
    pub fn new() -> Self {
        Self {
            runtime: Runtime::new(MockEngine, hologram_space::MemKappaStore::new()),
            resolver: NullResolver,
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
    type Resolver = NullResolver;
    type Runtime = Runtime<MockEngine, hologram_space::MemKappaStore>;

    fn store(&self) -> &Self::Store {
        self.runtime.store()
    }
    fn resolver(&self) -> &Self::Resolver {
        &self.resolver
    }
    fn runtime(&self) -> &Self::Runtime {
        &self.runtime
    }
}

/// A [`Resolver`] that resolves nothing — the space proves composition via the local store, so
/// the async seam simply returns `Ok(None)`.
pub struct NullResolver;

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl Resolver for NullResolver {
    async fn resolve(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(None)
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl Resolver for NullResolver {
    async fn resolve(&self, _kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError> {
        Ok(None)
    }
}
