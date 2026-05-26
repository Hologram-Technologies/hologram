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
    /// Assign a tier from Witt bit-width and element count.
    ///
    /// Layout-only ops (reshape, transpose, identity) always stay on CPU
    /// regardless of bit-width. Small element counts stay on CPU because
    /// GPU launch overhead would dominate.
    #[inline]
    pub const fn from_witt(witt_bits: u16, element_count: u32, is_layout_only: bool) -> Self {
        if is_layout_only {
            return Self::CpuMain;
        }
        match witt_bits {
            0..=8 => Self::CpuL1,
            9..=16 => Self::CpuL2,
            17..=24 => Self::CpuMain,
            _ => {
                if element_count < 1024 {
                    Self::CpuMain
                } else {
                    Self::Device
                }
            }
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
