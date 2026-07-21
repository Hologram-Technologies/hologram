//! # hologram-registry — hologram's κ-Distribution `/v2/` registry (spec 003)
//!
//! hologram's conforming implementation of the higher κ-Distribution conformance levels, served over
//! its content-addressed [`hologram_space::KappaStore`] substrate. This is the crate that consumes the
//! incubating **`uor-distribution`** standard (canonical byte forms, error taxonomy, levels): the
//! published, release-lockstep crates (`hologram-space`, `hologram-net`) cannot take a production
//! dependency on an unpublished crate, so the standard-consuming pieces live here.
//!
//! **Layering.** Levels 1–2 (blobs, tags) are served by [`hologram_net::http::kd`]; Levels 3–5
//! (edges, composition/witnesses/schemas, GC/admission/federation) are built here. All of it upholds
//! **κ-only identity** (Law 2 / SPINE-1): every stored object — edges included — is addressed by its
//! κ-label and nothing else.
//!
//! Incubating (plan Phase 5); `publish = false` and its own `[workspace]` keep it out of the release
//! and the always-green blocking jobs until ratification (Phase 8).

pub mod compose;
pub mod edge;
pub mod federation;
pub mod filter;
pub mod gc;
mod http_util;
pub mod schema;
pub mod server;
