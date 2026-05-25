//! Warm-start fold pass (WS-2) — bake the constant-only cone's materialized
//! results into a compiled archive so the runtime cache is never cold.
//!
//! The compiler emits a *labels-only* warm-start lattice (the κ-labels of the
//! constant-only cone). This pass loads that archive, **materializes** the
//! cone through the real runtime ([`InferenceSession::materialize_cone`] —
//! the same kernels and pool as a normal execute, so the bytes are identical
//! to a cold walk), and re-emits the archive with the results spliced into
//! the `WarmStart` section. A later load pins those results under their
//! lattice labels, so the existing residency check in the node walk elides
//! the whole cone on the first run — **no walk changes, no second path**.

use alloc::vec::Vec;

use hashbrown::HashMap;
use hologram_archive::{
    format::{SectionKind, FORMAT_VERSION, MAGIC},
    warm_codec, ArchiveError, ContentLabel, HoloLoader,
};
use hologram_host::HologramHasher;
use prism::vocabulary::Hasher;

use crate::error::ExecError;
use crate::session::{InferenceSession, SessionBackend};

/// A persisted, content-addressed warm-start store (WS-3): κ-label → result
/// bytes. Lets results computed in one process warm a later one — the lattice
/// (WS-1) provides the keys, so even a labels-only archive can be made warm
/// from a store a prior run populated, extending reuse past the in-archive
/// constant cone.
///
/// **Trust model.** Like the archive itself, the store is a *trusted* cache:
/// a present entry is taken as the true result for its derivation label (the
/// key is the derivation address, not a content hash of the bytes, so it
/// cannot be self-verified). A *missing* entry is always safe — the runtime
/// recomputes it. Implementations should detect *accidental* corruption
/// (return `None` on a failed integrity check) so a damaged store degrades to
/// recompute rather than a wrong answer; [`FileWarmStore`] does.
pub trait WarmStore {
    /// Fetch the bytes stored under `label`, or `None` if absent (or failing
    /// an integrity check — the caller then recomputes).
    fn get(&self, label: &ContentLabel) -> Option<Vec<u8>>;
    /// Store `bytes` under `label` for a future process.
    fn put(&mut self, label: &ContentLabel, bytes: &[u8]);
}

/// In-memory [`WarmStore`] (no_std). Useful for embedding a warm cache in a
/// long-lived host process and as the V&V reference; persistence across OS
/// processes uses [`FileWarmStore`].
#[derive(Default)]
pub struct MemWarmStore {
    map: HashMap<Vec<u8>, Vec<u8>>,
}

impl MemWarmStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl WarmStore for MemWarmStore {
    fn get(&self, label: &ContentLabel) -> Option<Vec<u8>> {
        self.map.get(label.as_bytes()).cloned()
    }
    fn put(&mut self, label: &ContentLabel, bytes: &[u8]) {
        self.map.insert(label.as_bytes().to_vec(), bytes.to_vec());
    }
}

/// Produce a warmed copy of `archive` with the constant-only cone's results
/// materialized into its `WarmStart` section.
///
/// Returns the archive unchanged (a defensive copy) when there is no cone to
/// fold. **Idempotent**: any existing `WarmStart` section is replaced, so
/// re-folding a warmed archive reproduces it. Never produces a wrong archive
/// — the only change is adding already-correct result bytes the runtime would
/// otherwise recompute.
pub fn fold_archive<B: SessionBackend>(archive: &[u8], backend: B) -> Result<Vec<u8>, ExecError> {
    // Materialize the cone through the real runtime (same kernels, same pool).
    let mut session = InferenceSession::load(archive, backend)?;
    let entries = session.materialize_cone()?;
    if entries.is_empty() {
        return Ok(archive.to_vec());
    }
    let warm_payload = warm_codec::encode(&entries);

    // Re-emit: copy every section verbatim (dropping any prior WarmStart),
    // append the freshly-folded WarmStart, and re-serialize with a new footer.
    // Working at the section-payload level needs no per-section decoders and
    // is robust to section ordering.
    let flags = u16::from_le_bytes([archive[6], archive[7]]);
    let plan = HoloLoader::from_bytes(archive)?.into_plan()?;
    let mut sections: Vec<(SectionKind, &[u8])> = Vec::new();
    for sref in plan.sections() {
        if sref.kind == SectionKind::WarmStart {
            continue; // replaced below (idempotent re-fold)
        }
        let start = sref.offset as usize;
        let end = start + sref.length as usize;
        let body = archive
            .get(start..end)
            .ok_or(ExecError::Archive(ArchiveError::Truncated {
                needed: end,
                actual: archive.len(),
            }))?;
        sections.push((sref.kind, body));
    }
    sections.push((SectionKind::WarmStart, &warm_payload));
    Ok(serialize_archive(flags, &sections))
}

/// Serialize sections into the `.holo` wire layout (mirrors `HoloWriter`):
/// `magic ‖ version ‖ flags ‖ count ‖ table ‖ payloads ‖ footer`.
fn serialize_archive(flags: u16, sections: &[(SectionKind, &[u8])]) -> Vec<u8> {
    const HEADER: usize = 4 + 2 + 2 + 2;
    const ENTRY: usize = 1 + 7 + 8 + 8;
    let table = ENTRY * sections.len();
    let body: usize = sections.iter().map(|(_, b)| b.len()).sum();
    let mut out = Vec::with_capacity(HEADER + table + body + 32);

    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(sections.len() as u16).to_le_bytes());

    let mut offset = (HEADER + table) as u64;
    for (kind, b) in sections {
        out.push(*kind as u8);
        out.extend_from_slice(&[0u8; 7]); // pad
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(b.len() as u64).to_le_bytes());
        offset += b.len() as u64;
    }
    for (_, b) in sections {
        out.extend_from_slice(b);
    }
    let footer: [u8; 32] = HologramHasher::initial().fold_bytes(&out).finalize();
    out.extend_from_slice(&footer);
    out
}

/// Filesystem-backed [`WarmStore`] (std): one file per κ-label under a
/// directory, each `[32-byte BLAKE3 checksum ‖ result bytes]`. `get` verifies
/// the checksum and returns `None` on mismatch or any I/O error, so a damaged
/// or partially-written store degrades to recompute — never a wrong answer.
/// This is the cross-process persistence tier: process N's `put` warms
/// process N+1's [`InferenceSession::warm_from_store`].
#[cfg(feature = "std")]
pub struct FileWarmStore {
    dir: std::path::PathBuf,
}

#[cfg(feature = "std")]
impl FileWarmStore {
    /// Open (creating if needed) a store rooted at `dir`.
    pub fn open(dir: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// `<dir>/<64-hex-digest>.warm` — the label's BLAKE3 hex (the κ-label
    /// minus its `blake3:` prefix) is a filesystem-safe, collision-free name.
    fn path(&self, label: &ContentLabel) -> std::path::PathBuf {
        let s = label.as_str();
        let hex = s.strip_prefix("blake3:").unwrap_or(s);
        self.dir.join(format!("{hex}.warm"))
    }
}

#[cfg(feature = "std")]
impl WarmStore for FileWarmStore {
    fn get(&self, label: &ContentLabel) -> Option<Vec<u8>> {
        let raw = std::fs::read(self.path(label)).ok()?;
        if raw.len() < 32 {
            return None;
        }
        let (checksum, body) = raw.split_at(32);
        let actual: [u8; 32] = HologramHasher::initial().fold_bytes(body).finalize();
        if actual.as_slice() != checksum {
            return None; // accidental corruption ⇒ recompute
        }
        Some(body.to_vec())
    }

    fn put(&mut self, label: &ContentLabel, bytes: &[u8]) {
        let checksum: [u8; 32] = HologramHasher::initial().fold_bytes(bytes).finalize();
        let mut blob = Vec::with_capacity(32 + bytes.len());
        blob.extend_from_slice(&checksum);
        blob.extend_from_slice(bytes);
        let _ = std::fs::write(self.path(label), &blob);
    }
}
