//! `.holo` binary layout (spec X.1).

pub const MAGIC: [u8; 4] = *b"HOLO";
pub const FORMAT_VERSION: u16 = 1;

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
