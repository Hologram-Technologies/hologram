//! # uor-distribution — the κ-Distribution protocol standard (spec 003)
//!
//! The universal, backend/transport/consensus-agnostic standard for a content-addressed `/v2/`
//! distribution registry — the UOR platform's **UOR-REGISTRY (#20)** + **UOR-RESOLUTION (#21)**
//! standards, a peer of `uor-addr` (which it builds on for identity). Any system that upholds these
//! byte contracts and operation semantics is a *conforming κ-Distribution registry*; hologram is one
//! such implementation, serving them over its `KappaStore`/`KappaSync` substrate.
//!
//! **Identity is the κ-label and nothing else (κ-only, per UOR-ADDR / SPINE-1).** Mutable tags and
//! DNS-style `org:path@version` human-names are a *resolution overlay* (UOR-RESOLUTION) that never
//! participates in addressing — as the UOR platform states of its transport realizations, "identity
//! still comes from UOR-ADDR, not the CID."
//!
//! This crate is **I/O-free**. It defines:
//! - the **canonical byte forms** ([`edge`], and — as they land — pins/witnesses/tag-snapshots) whose
//!   hashes must agree across every registry, or federation of the same content silently diverges;
//! - the **error taxonomy** ([`error`], spec §6.16);
//! - the **conformance levels** ([`level`], spec §13.1).
//!
//! Storage, transport, consensus, and hashing are the implementor's concern. (An edge's κ-label, for
//! instance, is computed by hashing [`edge::edge_canonical`] under the *source's* σ-axis — but which
//! hasher runs is the implementor's, not this crate's.)
//!
//! Incubating in-tree (plan Phase 3); extracted to its own crates.io crate at ratification (Phase 8).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod compose;
pub mod edge;
pub mod error;
pub mod level;

pub use compose::{compose_canonical, witness_blob, ComposeError, Op};
pub use error::ErrorCode;
pub use level::ConformanceLevel;
