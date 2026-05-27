//! `.holo` binary layout (spec X.1).

pub const MAGIC: [u8; 4] = *b"HOLO";
/// `.holo` format version. **v2** enriched the `Inputs`/`Outputs` port wire
/// format with a port `name` and full `shape`, and added the open
/// [`SectionKind::Extension`] section. v1 archives (flat unnamed ports, no
/// extensions) are not loadable by this build.
pub const FORMAT_VERSION: u16 = 2;

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
