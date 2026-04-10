//! `hologram-shapes` — conformance shape declarations, the [`PrismModule`]
//! trait, and the parameterised conformance test infrastructure.
//!
//! # Architecture role
//!
//! This crate is the **interface alphabet** layer of hologram. It declares the
//! conformance shapes that hologram's Prism modules carry, and it defines the
//! trait every Prism module implements so the compiler can dispatch against it
//! and the conformance test suite can verify it against its declared shape.
//!
//! Two shapes are declared:
//!
//! - [`F_PRISM_STRICT`](shape::F_PRISM_STRICT) — the five-primitive theoretical
//!   reference (bind, bundle, similarity, lookup, projection). Used for
//!   verification and as the directness-ratio baseline.
//! - [`F_PRISM_FUSED_COMPONENT`](shape::F_PRISM_FUSED_COMPONENT) — the
//!   transformer-component-granularity shape that hologram's first Prism
//!   module ([`hologram-fused-component`]) implements. Both bare and
//!   fused operation variants are declared as separate primitives — they are
//!   distinct irreducible operations in `T_{F_prism_fused_component}` because
//!   no Identity activation exists in the public signature to feed them into
//!   a factorisation.
//!
//! Both shapes are static `'static` data: no allocation, no runtime work to
//! produce them, no dynamic dispatch when consulting them. The conformance
//! argument is delivered through these shape signatures, *not* through enum
//! trimming on the substrate's implementation alphabet.
//!
//! # Performance
//!
//! - Shape declarations are `&'static` constants. **Perf: NEUTRAL** (no
//!   runtime allocation).
//! - The [`PrismModule`] trait is consulted at archive load time and at
//!   conformance test time, never in inner inference loops. **Perf: NEUTRAL.**
//! - The conformance test suite (state-space identity, transition fidelity,
//!   primitivity) runs only in tests, not in production builds. **Perf: NONE
//!   in release builds.**
//!
//! # Module organisation
//!
//! - [`shape`] — the two shape declarations and the `ShapeId` content-address
//! - [`prism_module`] — the [`PrismModule`] trait, [`SubstrateRequirements`],
//!   [`SubstrateClass`], [`LoadError`], [`ExecError`], [`ExecutionTrace`]
//! - [`conformance_tests`] — parameterised test families for the three
//!   carrying conditions (state-space identity, transition fidelity,
//!   primitivity)

#![no_std]
#![deny(missing_docs)]

extern crate alloc;

pub mod conformance_tests;
pub mod prism_module;
pub mod shape;

pub use prism_module::{
    ExecError, ExecutionTrace, LoadError, PrismModule, SubstrateClass, SubstrateRequirements,
};
pub use shape::{ShapeId, F_PRISM_FUSED_COMPONENT, F_PRISM_STRICT};
