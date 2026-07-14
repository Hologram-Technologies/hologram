//! The hologram **space contract** (`specs/refactor/02-space-contract.md`): the trait
//! surface a host implements to become a place hologram executes.
//!
//! This is the minimal **P0.5** slice of the contract — enough to prove the composition
//! bet (SP-3 / D28): a *synchronous* store plus an *async* network/boot seam under one
//! [`Space`], driving the synchronous compute hot path. It grows in P1 to the full
//! surface (engine, surface, HAL, entropy, clock, spawner — see spec 02).
//!
//! ## Async posture (LAW-4, corrected by the P0.5 spike)
//! Storage is **synchronous** (wasm-safe: the browser reaches persistent storage via a
//! synchronous OPFS handle inside a Worker). Only the network/boot seam is **async**, and
//! its `Send` bound is **maybe-Send** — `Send` on native, `?Send` on `wasm32`/bare where
//! futures are `!Send` — mirroring the substrate's `KappaSync` / `LocalKappaSync` split.
#![cfg_attr(not(feature = "std"), no_std)]

// `alloc` is used by the HAL module (and by `async_trait`'s `Box`ed futures) in both std
// and no_std builds.
extern crate alloc;
// `async_trait` desugars to `Pin<Box<dyn Future>>`; in `no_std` builds `Box` is not in
// the prelude, so bring it in from `alloc` (in `std` builds it already is).
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

pub mod hal;
pub use hal::{BlockDevice, DeviceError, NetworkInterface, NicError, RamBlockDevice};

// The portable trait surfaces + κ-addressing, absorbed from the former
// `hologram-substrate-core` crate (P1). Re-exported at the crate root so
// `hologram_space::{KappaStore, address_bytes, verify_kappa, …}` resolve directly.
mod substrate;
pub use substrate::*;

/// The async network/boot seam. `Send + Sync` on native (multi-threaded executors want
/// it); see the `wasm32` definition for the `?Send` variant.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait Resolver: Send + Sync {
    /// Resolve a κ from the network (or another peer); `Ok(None)` if unavailable.
    async fn resolve(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError>;
}

/// The async network/boot seam. On `wasm32` futures are `!Send`, so the bound is dropped
/// (`?Send`) — same trait, target-conditional bound (the maybe-Send policy, LAW-4).
#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
pub trait Resolver {
    /// Resolve a κ from the network (or another peer); `Ok(None)` if unavailable.
    async fn resolve(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, StoreError>;
}

/// A **space**: a place hologram executes. Aggregates a platform's concrete parts behind
/// associated types; everything downstream ([`crate`]'s consumers, `Client`) is generic
/// over `Space` and monomorphized per platform.
///
/// Minimal P0.5 surface: a synchronous [`KappaStore`] plus an async [`Resolver`]. P1 adds
/// the engine, surface, HAL, entropy, clock, and spawner associated types (spec 02).
pub trait Space {
    /// Local content store — **synchronous** (LAW-4).
    type Store: KappaStore;
    /// Async network/boot seam — **maybe-Send** (LAW-4).
    type Resolver: Resolver;

    /// The space's local store.
    fn store(&self) -> &Self::Store;
    /// The space's network/boot resolver.
    fn resolver(&self) -> &Self::Resolver;
}
