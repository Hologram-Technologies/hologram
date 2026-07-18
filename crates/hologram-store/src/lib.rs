//! # hologram-store — the hologram `KappaStore` backends, one crate
//!
//! Consolidates the former `hologram-store-{bare,native,opfs}` sibling crates into one, feature-gated
//! per platform (each backend is disjoint in deps + target):
//!
//! - [`bare`] (feature `bare`) — `no_std` + alloc bare-metal store over a raw `BlockDevice`, no
//!   filesystem (sectors only). For bare-metal / embedded peers.
//! - [`native`] (feature `native`) — WASI/std store on a **redb** B-tree index, with sharding + a
//!   bounded read-through cache. For native hosts.
//! - `opfs` (feature `opfs`) — browser OPFS store (`wasm32` + web-sys). *(folded in by the opfs
//!   merge; see the `opfs` module when enabled.)*
//!
//! Every backend passes the shared `hologram-tck` conformance TCK identically to the in-memory
//! reference; κ is the σ-axis content address throughout (verify-by-re-derivation, SPINE-4).
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "bare")]
pub mod bare;
#[cfg(feature = "native")]
pub mod native;
