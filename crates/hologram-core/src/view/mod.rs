//! ElementWiseView: 256-byte lookup table for O(1) function application.
//!
//! An `ElementWiseView` captures any byte-to-byte function as a 256-entry table.
//! Composition via `.then()` produces a new table — runtime cost is always one
//! array access regardless of composition depth.

mod compose;
mod simd;

use core::fmt;

/// A 256-entry byte-to-byte lookup table for O(1) function application.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
#[repr(align(64))] // Cache-line aligned for SIMD
pub struct ElementWiseView {
    table: [u8; 256],
}

impl ElementWiseView {
    /// Create from a precomputed table.
    #[inline]
    #[must_use]
    pub const fn from_table(table: [u8; 256]) -> Self {
        Self { table }
    }

    /// Create by applying `f` to all 256 byte values.
    #[must_use]
    pub fn new<F: Fn(u8) -> u8>(f: F) -> Self {
        let mut table = [0u8; 256];
        for i in 0..=255u8 {
            table[i as usize] = f(i);
        }
        Self { table }
    }

    /// Identity view (maps every byte to itself).
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        let mut table = [0u8; 256];
        let mut i = 0u8;
        loop {
            table[i as usize] = i;
            if i == 255 {
                break;
            }
            i += 1;
        }
        Self { table }
    }

    /// Constant view (maps every byte to `value`).
    #[inline]
    #[must_use]
    pub const fn constant(value: u8) -> Self {
        Self {
            table: [value; 256],
        }
    }

    /// Apply the view to a single byte — O(1).
    #[inline(always)]
    #[must_use]
    pub const fn apply(&self, byte: u8) -> u8 {
        self.table[byte as usize]
    }

    /// Compose: `self.then(other)` → `other(self(x))` for all x.
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        compose::compose(self, other)
    }

    /// Check if this view is bijective (a permutation).
    #[must_use]
    pub fn is_bijective(&self) -> bool {
        let mut seen = [false; 256];
        for &output in &self.table {
            if seen[output as usize] {
                return false;
            }
            seen[output as usize] = true;
        }
        true
    }

    /// Compute the inverse if bijective.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        if !self.is_bijective() {
            return None;
        }
        let mut inv = [0u8; 256];
        for input in 0..=255u8 {
            inv[self.table[input as usize] as usize] = input;
        }
        Some(Self { table: inv })
    }

    /// Apply in place to a slice. Uses SIMD when available.
    pub fn apply_slice(&self, data: &mut [u8]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if data.len() >= 32 {
            return simd::apply_avx2(self, data);
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
        if data.len() >= 16 {
            return simd::apply_sse42(self, data);
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if data.len() >= 16 {
            return simd::apply_neon(self, data);
        }

        for byte in data {
            *byte = self.apply(*byte);
        }
    }

    /// Apply to `input`, writing to `output`.
    ///
    /// # Panics
    /// Panics if lengths differ.
    pub fn apply_to(&self, input: &[u8], output: &mut [u8]) {
        assert_eq!(input.len(), output.len());

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if input.len() >= 32 {
            return simd::apply_to_avx2(self, input, output);
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
        if input.len() >= 16 {
            return simd::apply_to_sse42(self, input, output);
        }

        for (i, &byte) in input.iter().enumerate() {
            output[i] = self.apply(byte);
        }
    }

    /// Reference to the underlying table.
    #[inline]
    #[must_use]
    pub const fn table(&self) -> &[u8; 256] {
        &self.table
    }

    /// Consume and return the underlying table.
    #[inline]
    #[must_use]
    pub const fn into_table(self) -> [u8; 256] {
        self.table
    }
}

impl fmt::Debug for ElementWiseView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ElementWiseView {{ bijective: {}, ", self.is_bijective())?;
        write!(f, "table: [")?;
        for (i, &byte) in self.table.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if i >= 8 {
                write!(f, "... ({} more)", 256 - 8)?;
                break;
            }
            write!(f, "{byte}")?;
        }
        write!(f, "] }}")
    }
}

impl Default for ElementWiseView {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let v = ElementWiseView::identity();
        for i in 0..=255u8 {
            assert_eq!(v.apply(i), i);
        }
    }

    #[test]
    fn constant() {
        let v = ElementWiseView::constant(42);
        for i in 0..=255u8 {
            assert_eq!(v.apply(i), 42);
        }
    }

    #[test]
    fn new_increment() {
        let v = ElementWiseView::new(|x| x.wrapping_add(1));
        assert_eq!(v.apply(0), 1);
        assert_eq!(v.apply(255), 0);
    }

    #[test]
    fn composition() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let double = inc.then(&inc);
        assert_eq!(double.apply(0), 2);
        assert_eq!(double.apply(255), 1);
    }

    #[test]
    fn composition_identity() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let id = ElementWiseView::identity();
        for i in 0..=255u8 {
            assert_eq!(inc.then(&id).apply(i), inc.apply(i));
            assert_eq!(id.then(&inc).apply(i), inc.apply(i));
        }
    }

    #[test]
    fn bijective() {
        assert!(ElementWiseView::identity().is_bijective());
        assert!(ElementWiseView::new(|x| x.wrapping_add(1)).is_bijective());
        assert!(!ElementWiseView::constant(0).is_bijective());
    }

    #[test]
    fn inverse() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let dec = inc.inverse().unwrap();
        for i in 0..=255u8 {
            assert_eq!(dec.apply(inc.apply(i)), i);
        }
        assert!(ElementWiseView::constant(0).inverse().is_none());
    }

    #[test]
    fn apply_slice() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let mut data = [0, 1, 2, 255];
        inc.apply_slice(&mut data);
        assert_eq!(data, [1, 2, 3, 0]);
    }

    #[test]
    fn apply_to_slices() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let input = [0, 1, 2, 255];
        let mut output = [0u8; 4];
        inc.apply_to(&input, &mut output);
        assert_eq!(output, [1, 2, 3, 0]);
    }

    #[test]
    fn default_is_identity() {
        let v = ElementWiseView::default();
        for i in 0..=255u8 {
            assert_eq!(v.apply(i), i);
        }
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn rkyv_round_trip() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&inc).unwrap();
        let archived =
            rkyv::access::<rkyv::Archived<ElementWiseView>, rkyv::rancor::Error>(&bytes).unwrap();
        for i in 0..=255u8 {
            assert_eq!(archived.table[i as usize], inc.apply(i));
        }
    }
}
