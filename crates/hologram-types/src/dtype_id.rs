//! The runtime dtype tag — one canonical spelling of "which dtype", shared by
//! the graph, the compiler, the archive wire, and the kernels.
//!
//! Before this existed the same concept was spelled three ways: a type-level
//! marker (`dtype::DType`), a `DTypeId(u8)` local to the graph registry, and a
//! loose family of `DTYPE_*: u8` constants in the backend, matched
//! independently in ~8 places. Two of those matches had `_ =>` fallbacks that
//! *silently miscomputed* on an unrecognized tag (a storage size of 1 byte, a
//! dequantized value of 0) rather than failing. Every accessor here is total:
//! an unknown tag yields `None`, so a caller must handle it or propagate an
//! error — it can never quietly produce a wrong answer.
//!
//! `#[repr(transparent)]` over `u8`: the archive wire and `op_signature` write
//! `DTypeId::raw()`, so the on-disk bytes and every content address are
//! unchanged by the introduction of this type.

/// A dtype tag. The numeric values are **wire-stable** — they appear in archive
/// bytes and in `op_signature` params, so they must never be renumbered.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DTypeId(pub u8);

impl DTypeId {
    pub const BOOL: Self = Self(0);
    pub const U8: Self = Self(1);
    pub const I8: Self = Self(2);
    pub const U64: Self = Self(3);
    pub const I32: Self = Self(4);
    pub const I64: Self = Self(5);
    pub const F16: Self = Self(6);
    pub const BF16: Self = Self(7);
    pub const F32: Self = Self(8);
    pub const F64: Self = Self(9);
    /// Packed signed 4-bit integer, two per byte (low nibble first).
    pub const I4: Self = Self(10);
    /// E8 lattice-codebook vector quantization: each **8**-element weight
    /// subvector is one `u8` codebook index. The group dimension is 8 because
    /// E8 is an 8-dimensional lattice — that is the definition, not a tuning
    /// choice. The codebook *contents* and its entry count (`1..=256`) are
    /// per-model data and travel as a constant operand.
    pub const E8CB: Self = Self(11);

    /// The E8 codebook's group dimension (the lattice's dimension).
    pub const E8CB_GROUP_DIM: u32 = 8;
    /// Largest codebook cardinality addressable by the `u8` index.
    pub const E8CB_MAX_ENTRIES: usize = 256;

    /// Every tag the engine understands, in wire order.
    pub const ALL: [Self; 12] = [
        Self::BOOL,
        Self::U8,
        Self::I8,
        Self::U64,
        Self::I32,
        Self::I64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
        Self::I4,
        Self::E8CB,
    ];

    /// The wire byte. Total by construction.
    #[must_use]
    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// `None` for a tag this build does not understand — never a guess.
    #[must_use]
    #[inline]
    pub const fn known(self) -> Option<Self> {
        if self.0 <= Self::E8CB.0 {
            Some(self)
        } else {
            None
        }
    }

    /// `true` for tags whose elements are smaller than one byte, so element
    /// count alone does not determine storage (use [`Self::storage_bytes`]).
    #[must_use]
    #[inline]
    pub const fn is_sub_byte(self) -> bool {
        matches!(self.0, x if x == Self::I4.0 || x == Self::E8CB.0)
    }

    /// Bytes per element. `None` for sub-byte tiers **and** for unknown tags —
    /// there is deliberately no `1`-byte fallback.
    #[must_use]
    pub const fn bytes_per_element(self) -> Option<usize> {
        Some(match self.0 {
            x if x == Self::BOOL.0 || x == Self::U8.0 || x == Self::I8.0 => 1,
            x if x == Self::F16.0 || x == Self::BF16.0 => 2,
            x if x == Self::I32.0 || x == Self::F32.0 => 4,
            x if x == Self::U64.0 || x == Self::I64.0 || x == Self::F64.0 => 8,
            _ => return None, // sub-byte (I4/E8CB) or unknown
        })
    }

    /// Storage bytes for an `n`-element buffer, honouring sub-byte packing
    /// (`I4` → `ceil(n/2)`, `E8CB` → `ceil(n/8)`). `None` for an unknown tag.
    #[must_use]
    pub const fn storage_bytes(self, element_count: u32) -> Option<u32> {
        if self.0 == Self::I4.0 {
            return Some(element_count.div_ceil(2));
        }
        if self.0 == Self::E8CB.0 {
            return Some(element_count.div_ceil(Self::E8CB_GROUP_DIM));
        }
        match self.bytes_per_element() {
            Some(b) => Some(element_count * b as u32),
            None => None,
        }
    }

    /// [`Self::storage_bytes`] for 64-bit element counts (graph tensors are
    /// sized in `u64`). `None` for an unrecognized tag.
    #[must_use]
    pub const fn storage_bytes_u64(self, element_count: u64) -> Option<u64> {
        if self.0 == Self::I4.0 {
            return Some(element_count.div_ceil(2));
        }
        if self.0 == Self::E8CB.0 {
            return Some(element_count.div_ceil(Self::E8CB_GROUP_DIM as u64));
        }
        match self.bytes_per_element() {
            Some(b) => Some(element_count.saturating_mul(b as u64)),
            None => None,
        }
    }

    /// IEEE-754 (or bfloat) typed — selects the native float kernel paths.
    #[must_use]
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self.0, x if x == Self::F16.0
            || x == Self::BF16.0
            || x == Self::F32.0
            || x == Self::F64.0)
    }

    /// A quantized *weight* encoding (decoded through scales/zero-points and,
    /// for `E8CB`, a codebook operand).
    #[must_use]
    #[inline]
    pub const fn is_quantized_weight(self) -> bool {
        matches!(self.0, x if x == Self::I8.0
            || x == Self::U8.0
            || x == Self::I4.0
            || x == Self::E8CB.0)
    }

    /// Stable short name, for diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.0 {
            x if x == Self::BOOL.0 => "bool",
            x if x == Self::U8.0 => "u8",
            x if x == Self::I8.0 => "i8",
            x if x == Self::U64.0 => "u64",
            x if x == Self::I32.0 => "i32",
            x if x == Self::I64.0 => "i64",
            x if x == Self::F16.0 => "f16",
            x if x == Self::BF16.0 => "bf16",
            x if x == Self::F32.0 => "f32",
            x if x == Self::F64.0 => "f64",
            x if x == Self::I4.0 => "i4",
            x if x == Self::E8CB.0 => "e8cb",
            _ => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tags are wire-stable: they are written into archive bytes and
    /// `op_signature` params. Renumbering silently re-keys every κ.
    #[test]
    fn tag_values_are_wire_stable() {
        let expect: [(DTypeId, u8); 12] = [
            (DTypeId::BOOL, 0),
            (DTypeId::U8, 1),
            (DTypeId::I8, 2),
            (DTypeId::U64, 3),
            (DTypeId::I32, 4),
            (DTypeId::I64, 5),
            (DTypeId::F16, 6),
            (DTypeId::BF16, 7),
            (DTypeId::F32, 8),
            (DTypeId::F64, 9),
            (DTypeId::I4, 10),
            (DTypeId::E8CB, 11),
        ];
        for (d, raw) in expect {
            assert_eq!(d.raw(), raw, "{} tag moved", d.name());
        }
        assert_eq!(DTypeId::ALL.len(), expect.len());
    }

    #[test]
    fn unknown_tags_are_rejected_not_guessed() {
        let bad = DTypeId(200);
        assert_eq!(bad.known(), None);
        assert_eq!(bad.bytes_per_element(), None);
        assert_eq!(bad.storage_bytes(64), None);
        assert_eq!(bad.name(), "unknown");
    }

    #[test]
    fn storage_bytes_honours_sub_byte_packing() {
        assert_eq!(DTypeId::F32.storage_bytes(10), Some(40));
        assert_eq!(DTypeId::I8.storage_bytes(10), Some(10));
        assert_eq!(DTypeId::BF16.storage_bytes(10), Some(20));
        // I4: two per byte, rounding up.
        assert_eq!(DTypeId::I4.storage_bytes(10), Some(5));
        assert_eq!(DTypeId::I4.storage_bytes(11), Some(6));
        // E8CB: one index byte per 8-element group, rounding up.
        assert_eq!(DTypeId::E8CB.storage_bytes(64), Some(8));
        assert_eq!(DTypeId::E8CB.storage_bytes(65), Some(9));
        // Sub-byte tiers have no per-element byte width.
        assert_eq!(DTypeId::I4.bytes_per_element(), None);
        assert_eq!(DTypeId::E8CB.bytes_per_element(), None);
        assert!(DTypeId::I4.is_sub_byte() && DTypeId::E8CB.is_sub_byte());
        assert!(!DTypeId::I8.is_sub_byte());
    }

    #[test]
    fn classifiers() {
        assert!(DTypeId::F32.is_float() && DTypeId::BF16.is_float());
        assert!(!DTypeId::I8.is_float() && !DTypeId::E8CB.is_float());
        assert!(DTypeId::I8.is_quantized_weight() && DTypeId::E8CB.is_quantized_weight());
        assert!(!DTypeId::F32.is_quantized_weight());
    }
}
