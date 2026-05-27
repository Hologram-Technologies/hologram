//! Memory tier assignment (Prism identity PM_7: Memory Affinity).
//!
//! The memory tier of a datum is determined by its quantum level (Witt
//! bit-width). This enables the runtime to place buffers on the optimal
//! device without runtime branching — the tier is a pure function of
//! `witt_bits`, resolved at compile time.

/// Memory placement tier derived from a kernel's Witt bit-width.
///
/// Ordered from fastest-access (L1) to highest-bandwidth (Device).
/// The compiler assigns one tier per `KernelCall` based on `witt_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum MemoryTier {
    /// Q0 (≤8-bit): 256-byte LUT, L1-cache resident. CPU-native.
    CpuL1 = 0,
    /// Q1 (9–16-bit): 128 KB LUT, L2-cache resident. CPU-native.
    CpuL2 = 1,
    /// Q2 (17–24-bit): ~50 MB segmented tables. Main memory.
    CpuMain = 2,
    /// Q3+ (≥25-bit): algorithmic compute. GPU/accelerator dispatch.
    Device = 3,
}

impl MemoryTier {
    /// Assign a tier from the datum's **quantum level** (Witt bit-width) — a
    /// pure function of the quantum level (PM_7's thesis), with no element-count
    /// cutoff or other arbitrary threshold, so it scales to any workload size:
    ///
    /// - Q0 (≤8-bit): the domain has ≤256 points → fully tabled, L1-resident.
    /// - Q1 (9–16-bit): ≤65536 points → 128 KB table, L2-resident.
    /// - Q2 (17–24-bit): too large to fully table → algorithmic, main memory.
    /// - Q3+ (≥25-bit, incl. f32): algorithmic, accelerator-class (`Device`).
    ///
    /// Layout-only ops (reshape, transpose, identity) carry no compute, so they
    /// stay CPU-resident regardless of quantum level. Whether a `Device`-tier op
    /// *actually* dispatches to an accelerator vs CPU is a routing decision
    /// (`TierPolicy`), not a property of the datum — keeping this a pure,
    /// size-independent function.
    #[inline]
    pub const fn from_witt(witt_bits: u16, is_layout_only: bool) -> Self {
        if is_layout_only {
            return Self::CpuMain;
        }
        match witt_bits {
            0..=8 => Self::CpuL1,
            9..=16 => Self::CpuL2,
            17..=24 => Self::CpuMain,
            _ => Self::Device,
        }
    }

    /// Returns `true` if this tier targets CPU execution.
    #[inline]
    pub const fn is_cpu(&self) -> bool {
        (*self as u8) < (Self::Device as u8)
    }

    /// Decode from a raw `u8` discriminant (e.g. from archive bytes).
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CpuL1),
            1 => Some(Self::CpuL2),
            2 => Some(Self::CpuMain),
            3 => Some(Self::Device),
            _ => None,
        }
    }
}
