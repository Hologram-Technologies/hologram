//! The **presentation / interaction seam** (spec 02 §5, D10): a κ-addressed projection of a
//! running workload's state plus an intent channel driving it (terminal I/O, file edits,
//! framebuffer regions). Generalizes holospaces' `projection.rs` (Workspace/Intent) so a
//! portable app view targets the [`Surface`] and therefore runs on every space.
//!
//! The seam is deliberately small — design systems plug in above it. It takes the running
//! workload's **κ** (not a runtime `Session`: the contract crate must not depend on the runtime,
//! RZ). **Headless** is a first-class profile: a space with no display implements [`Surface`] with
//! the null projection ([`NullSurface`]) — `project` returns the empty-projection κ, `intent`
//! refuses with [`SurfaceError::Headless`].

use alloc::string::String;
use alloc::vec::Vec;
// `async_trait` emits an unqualified `Box`; not in the `no_std` prelude (std provides it).
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

use crate::{address_bytes, KappaLabel71};

/// A canonical operator **intent** on a surface — a **closed, exhaustive** set (like the op
/// catalogue). Each variant is published as a content-addressed event on the workload's channel
/// (Laws L1/L2): identical intents address to the same κ; the intent is content, never a location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    /// A line typed into the terminal — the raw keystroke bytes (the newline is implied).
    TerminalInput(Vec<u8>),
    /// A file edit: replace the content at `path` with `content` (the editor's save).
    FileEdit {
        /// The path within the environment the operator edited.
        path: String,
        /// The new content (the editor buffer).
        content: Vec<u8>,
    },
    /// A framebuffer region update (graphical surfaces): the raw pixel bytes for a rectangle.
    FrameRegion {
        /// Left edge (pixels).
        x: u32,
        /// Top edge (pixels).
        y: u32,
        /// Region width (pixels).
        width: u32,
        /// Region height (pixels).
        height: u32,
        /// The region's raw pixel bytes (`width * height * bytes_per_pixel`).
        pixels: Vec<u8>,
    },
}

/// Why a [`Surface`] operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceError {
    /// The surface is **headless** — no display; it refuses interaction (a valid conformance
    /// profile, spec 02 §5).
    Headless,
    /// The workload κ could not be projected (absent, or not in a running phase).
    NotProjectable,
    /// A backend failure (store / render), with a static reason.
    Backend(&'static str),
}

impl core::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SurfaceError::Headless => f.write_str("surface is headless — interaction refused"),
            SurfaceError::NotProjectable => f.write_str("workload could not be projected"),
            SurfaceError::Backend(why) => write!(f, "surface backend failure: {why}"),
        }
    }
}

/// The presentation/interaction seam. **Maybe-Send** (LAW-4), the same cfg-gated posture as
/// [`KappaSync`](crate::KappaSync): `Send + Sync` on native, `?Send` on `wasm32`/bare (a browser
/// surface holds `!Send` DOM handles). `project` yields a **κ** for the projected state (render it
/// by resolving that κ); `intent` publishes a canonical operator event and returns **its κ**
/// (Law L1). Every space provides a surface; headless spaces use [`NullSurface`].
#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
pub trait Surface: Send + Sync {
    /// Project the running `workload`'s current state to a κ (the content to render).
    async fn project(&self, workload: &KappaLabel71) -> Result<KappaLabel71, SurfaceError>;
    /// Enact an operator `intent` on `workload`, returning the published event's κ.
    async fn intent(
        &self,
        workload: &KappaLabel71,
        intent: Intent,
    ) -> Result<KappaLabel71, SurfaceError>;
}

/// `?Send` `wasm32`/bare variant — see the native definition above.
#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
pub trait Surface {
    /// Project the running `workload`'s current state to a κ (the content to render).
    async fn project(&self, workload: &KappaLabel71) -> Result<KappaLabel71, SurfaceError>;
    /// Enact an operator `intent` on `workload`, returning the published event's κ.
    async fn intent(
        &self,
        workload: &KappaLabel71,
        intent: Intent,
    ) -> Result<KappaLabel71, SurfaceError>;
}

/// The **headless** reference [`Surface`]: `project` returns the canonical empty-projection κ
/// (the κ of no bytes) and `intent` refuses with [`SurfaceError::Headless`]. This is the valid
/// no-display conformance profile (esp32, a storage-sync node) — a space without a UI still
/// satisfies the contract.
pub struct NullSurface;

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl Surface for NullSurface {
    async fn project(&self, _workload: &KappaLabel71) -> Result<KappaLabel71, SurfaceError> {
        Ok(address_bytes(&[]))
    }
    async fn intent(
        &self,
        _workload: &KappaLabel71,
        _intent: Intent,
    ) -> Result<KappaLabel71, SurfaceError> {
        Err(SurfaceError::Headless)
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl Surface for NullSurface {
    async fn project(&self, _workload: &KappaLabel71) -> Result<KappaLabel71, SurfaceError> {
        Ok(address_bytes(&[]))
    }
    async fn intent(
        &self,
        _workload: &KappaLabel71,
        _intent: Intent,
    ) -> Result<KappaLabel71, SurfaceError> {
        Err(SurfaceError::Headless)
    }
}
