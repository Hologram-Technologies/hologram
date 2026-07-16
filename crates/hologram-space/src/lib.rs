//! The hologram **space contract** (`specs/refactor/02-space-contract.md`): the trait
//! surface a host implements to become a place hologram executes.
//!
//! The [`Space`] aggregate names a platform's concrete parts behind associated types: a
//! synchronous [`KappaStore`], the maybe-Send [`KappaSync`] network seam, the composed
//! [`ContainerRuntime`], and the HAL seams [`Entropy`] / [`Clock`] / [`Spawner`]. Only the
//! `Surface` (UI projection) part of spec 02 is still to be designed.
//!
//! ## Async posture (LAW-4, corrected by the P0.5 spike)
//! Storage is **synchronous** (wasm-safe: the browser reaches persistent storage via a
//! synchronous OPFS handle inside a Worker). Only the network/boot seam is **async**, and
//! its `Send` bound is **maybe-Send** — `Send` on native, `?Send` on `wasm32`/bare where
//! futures are `!Send` — via the substrate's one cfg-gated [`KappaSync`] trait.
#![cfg_attr(not(feature = "std"), no_std)]

// `alloc` is used by the HAL module (and by `async_trait`'s `Box`ed futures in the `substrate`
// submodule, which brings `Box` into its own scope) in both std and no_std builds.
extern crate alloc;

pub mod hal;
pub use hal::{
    BlockDevice, Clock, DeviceError, Entropy, ManualClock, NetworkInterface, NicError, NoopSpawner,
    RamBlockDevice, SeededEntropy, Spawner,
};

// The portable trait surfaces + κ-addressing, absorbed from the former
// `hologram-substrate-core` crate (P1). Re-exported at the crate root so
// `hologram_space::{KappaStore, address_bytes, verify_kappa, …}` resolve directly.
mod substrate;
pub use substrate::*;

// The canonical realization forms (ContainerManifest, CapabilitySet, Snapshot, …),
// absorbed from the former `hologram-realizations` crate (P1). Re-exported at the root.
mod realizations;
pub use realizations::*;

// The container-engine seam (spec §4.2/§4.4): the `ContainerEngine` Wasm-instance trait
// plus its two supporting structs, `HostContext` and `ContainerIntents`. The contract owns
// the engine seam; `hologram-runtime` orchestration and its backends reference it. Moved
// here so `Space` can gain a `type Engine: ContainerEngine` associated type (P1).
pub mod engine;
pub use engine::{ContainerEngine, ContainerIntents, HostContext};

// The reference in-memory `KappaStore` (`MemKappaStore`), re-homed here from the former
// `hologram-store-mem` crate (which move 7 had folded into the conformance TCK). It lives with
// the `KappaStore` trait it implements so runtime consumers (holospaces' κ-disk, the wasmi
// engine, the CLI) reach it without a normal dependency on the test kit; `hologram-tck`
// re-exports it for conformance authors. no_std + alloc, zero extra deps (hashbrown + spin).
mod mem;
pub use mem::MemKappaStore;

/// A **space**: a place hologram executes. Aggregates a platform's concrete parts behind
/// associated types; everything downstream ([`crate`]'s consumers, `Client`) is generic
/// over `Space` and monomorphized per platform.
///
/// Parts: a synchronous [`KappaStore`], the maybe-Send [`KappaSync`] network seam, the composed
/// [`ContainerRuntime`], and the [`Entropy`] / [`Clock`] / [`Spawner`] HAL seams (spec 02 §4).
/// The `Surface` (UI projection) part is still to be designed.
pub trait Space {
    /// Local content store — **synchronous** (LAW-4).
    type Store: KappaStore;
    /// Async network/boot seam — the substrate's maybe-Send [`KappaSync`] (LAW-4); `Send + Sync`
    /// on native, `?Send` on `wasm32`/bare (spec-02 `type Sync: KappaSync`).
    type Sync: KappaSync;
    /// The container-lifecycle runtime — `spawn`/`suspend`/`resume`/`terminate` over the
    /// space's `ContainerEngine` + `KappaStore`. `Client::open` drives a `Session` over it.
    ///
    /// Pragmatic shape (spec 02): the space exposes the **composed** runtime rather than a
    /// bare `Engine` associated type, because a `Runtime` owns its store — so an impl holds one
    /// `Runtime` and typically delegates [`store`](Space::store) to `runtime().store()`.
    type Runtime: ContainerRuntime;
    /// The platform randomness seam (spec 02 §4 HAL) — key generation and nonces draw here.
    type Entropy: Entropy;
    /// The platform monotonic-clock seam (spec 02 §4 HAL) — timeouts / fuel budgets measure here.
    type Clock: Clock;
    /// The platform background-task spawn seam (spec 02 §4 HAL) — the net pump / async work run here.
    type Spawner: Spawner;

    /// The space's local store.
    fn store(&self) -> &Self::Store;
    /// The space's network/boot sync seam (spec-02 `sync()`).
    fn sync(&self) -> &Self::Sync;
    /// The space's container runtime.
    fn runtime(&self) -> &Self::Runtime;
    /// The space's randomness source.
    fn entropy(&self) -> &Self::Entropy;
    /// The space's monotonic clock.
    fn clock(&self) -> &Self::Clock;
    /// The space's background-task spawner.
    fn spawner(&self) -> &Self::Spawner;
}
