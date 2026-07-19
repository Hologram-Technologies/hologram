//! `.holo` binary layout (spec X.1).

pub const MAGIC: [u8; 4] = *b"HOLO";
/// `.holo` format version (spec `refactor/03`). **v3** makes `.holo` the one
/// application container: it adds the [`SectionKind::AppManifest`] section —
/// an IRI-tagged `AppManifest` realization naming an application's ordered,
/// κ-referenced layers plus its composed children — on top of the v2
/// tensor-graph sections (kinds 0–14, unchanged). A tensor-only archive is the
/// degenerate single-layer case. **v2** enriched the `Inputs`/`Outputs` port
/// wire format with a port `name` and full `shape`, and added the open
/// [`SectionKind::Extension`] section.
///
/// Writers emit v3 only; readers accept [`MIN_READ_VERSION`]`..=FORMAT_VERSION`
/// so a v2 tensor archive stays loadable (the loader treats it as a single
/// tensor-plan layer). v1 archives (flat unnamed ports) are not loadable.
pub const FORMAT_VERSION: u16 = 3;

/// Lowest `.holo` version this build still reads (the v2 read-shim, spec 03
/// §Compatibility). Below this the archive is rejected with
/// [`crate::error::ArchiveError::UnsupportedVersion`].
pub const MIN_READ_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SectionKind {
    KernelCalls = 1,
    Schedule = 2,
    Weights = 3,
    ShapeRegistry = 4,
    DTypeRegistry = 5,
    Certificates = 6,
    Trace = 7,
    Metadata = 8,
    Inputs = 9,
    Outputs = 10,
    Constants = 11,
    /// Per-level kernel-call indices (spec VIII.2). Mirrors `Schedule`
    /// but indexes `kernel_calls[]` directly so the executor can walk
    /// levels in parallel without re-resolving NodeIds.
    ExecPlan = 12,
    /// Warm-start lattice (WS class): the κ-labels (and, at fold depth,
    /// the materialized results) of the constant-only cone — nodes whose
    /// transitive inputs are all constants. Pinned at load under their
    /// labels so the runtime cache is never cold. See `warm_codec`.
    WarmStart = 13,
    /// Open producer-defined metadata: a length-prefixed string `key` followed
    /// by arbitrary `bytes`. **Repeatable** — one section per key (tokenizer,
    /// generation config, class labels, calibration tables, provenance, …). The
    /// runtime carries extensions opaquely; a consumer fetches them by key. This
    /// is the format's escape hatch so arbitrary use-cases need not extend this
    /// closed enum.
    Extension = 14,
    /// The `.holo` v3 application manifest (spec `refactor/03`): the canonical
    /// bytes of an `AppManifest` realization (`hologram-space`), IRI-tagged and
    /// embedding every layer κ, every child `(app κ, caps κ)`, and the required
    /// CapabilitySet κ. Opaque to the archive layer — the container frames it;
    /// the app loader resolves its closure. At most one per archive. Its
    /// discriminant is appended (kinds 0–14 keep theirs, κ-stability).
    AppManifest = 15,
    /// A κ-addressed content blob embedded in a **fat** `.holo` (spec 03 §Fat and thin):
    /// `κ71 ‖ content_bytes`. **Repeatable** — one per embedded layer/closure κ. A *thin* archive
    /// omits these and resolves its manifest's closure through the store/sync; a *fat* archive
    /// carries them so it is self-contained. Fat↔thin is a packaging choice — the manifest κ (the
    /// app's identity) is unchanged either way. Opaque to the archive layer.
    ContentBlob = 16,
}

#[derive(Debug, Clone, Copy)]
pub struct SectionRef {
    pub kind: SectionKind,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone)]
pub struct HoloHeader {
    pub magic: [u8; 4],
    pub format_version: u16,
    pub flags: u16,
    pub section_count: u16,
    pub sections: alloc::vec::Vec<SectionRef>,
}
