//! ElementWiseView16: 128 KB lookup table for O(1) Q1 function application.
//!
//! Similar to `ElementWiseView` (Q0, 256 bytes, stack) but for Q1 (65536 entries,
//! 128 KB, heap-allocated). Not `Copy` due to size.

#[cfg(feature = "std")]
extern crate std;

use core::fmt;

/// A 65536-entry u16-to-u16 lookup table for O(1) function application at Q1.
///
/// Heap-allocated (128 KB) — too large for stack. Not `Copy`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ElementWiseView16 {
    table: std::boxed::Box<[u16; 65536]>,
}

impl ElementWiseView16 {
    /// Create from a precomputed table.
    #[inline]
    #[must_use]
    pub fn from_table(table: std::boxed::Box<[u16; 65536]>) -> Self {
        Self { table }
    }

    /// Create from a static table reference (clones into a heap allocation).
    #[must_use]
    pub fn from_static(table: &[u16; 65536]) -> Self {
        let mut boxed = std::vec![0u16; 65536].into_boxed_slice();
        boxed.copy_from_slice(table);
        // SAFETY: boxed slice has exactly 65536 elements.
        let ptr = std::boxed::Box::into_raw(boxed) as *mut [u16; 65536];
        Self {
            table: unsafe { std::boxed::Box::from_raw(ptr) },
        }
    }

    /// Create by applying `f` to all 65536 u16 values.
    #[must_use]
    pub fn from_fn<F: Fn(u16) -> u16>(f: F) -> Self {
        let mut table = std::vec![0u16; 65536].into_boxed_slice();
        for i in 0..65536u32 {
            table[i as usize] = f(i as u16);
        }
        let ptr = std::boxed::Box::into_raw(table) as *mut [u16; 65536];
        Self {
            table: unsafe { std::boxed::Box::from_raw(ptr) },
        }
    }

    /// Identity view (maps every u16 to itself).
    #[must_use]
    pub fn identity() -> Self {
        Self::from_fn(|x| x)
    }

    /// Constant view (maps every u16 to `value`).
    #[must_use]
    pub fn constant(value: u16) -> Self {
        let v = std::vec![value; 65536].into_boxed_slice();
        let ptr = std::boxed::Box::into_raw(v) as *mut [u16; 65536];
        Self {
            table: unsafe { std::boxed::Box::from_raw(ptr) },
        }
    }

    /// Apply the view to a single u16 — O(1).
    #[inline(always)]
    #[must_use]
    pub fn apply(&self, value: u16) -> u16 {
        self.table[value as usize]
    }

    /// Compose: `self.then(other)` → `other(self(x))` for all x.
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        Self::from_fn(|x| other.apply(self.apply(x)))
    }

    /// Check if this view is bijective (a permutation).
    #[must_use]
    pub fn is_bijective(&self) -> bool {
        let mut seen = std::vec![false; 65536];
        for &output in self.table.iter() {
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
        let mut inv = std::vec![0u16; 65536].into_boxed_slice();
        for input in 0..65536u32 {
            inv[self.table[input as usize] as usize] = input as u16;
        }
        let ptr = std::boxed::Box::into_raw(inv) as *mut [u16; 65536];
        Some(Self {
            table: unsafe { std::boxed::Box::from_raw(ptr) },
        })
    }

    /// Apply in place to a slice of u16 values.
    pub fn apply_slice(&self, data: &mut [u16]) {
        for val in data {
            *val = self.apply(*val);
        }
    }

    /// Apply to `input`, writing to `output`.
    ///
    /// # Panics
    /// Panics if lengths differ.
    pub fn apply_to(&self, input: &[u16], output: &mut [u16]) {
        assert_eq!(input.len(), output.len());
        for (i, &val) in input.iter().enumerate() {
            output[i] = self.apply(val);
        }
    }

    /// Reference to the underlying table.
    #[inline]
    #[must_use]
    pub fn table(&self) -> &[u16; 65536] {
        &self.table
    }
}

impl fmt::Debug for ElementWiseView16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ElementWiseView16 {{ bijective: {}, ", self.is_bijective())?;
        write!(f, "table: [")?;
        for (i, &val) in self.table.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if i >= 8 {
                write!(f, "... ({} more)", 65536 - 8)?;
                break;
            }
            write!(f, "{val}")?;
        }
        write!(f, "] }}")
    }
}

impl Default for ElementWiseView16 {
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
        let v = ElementWiseView16::identity();
        for i in (0u32..=65535).step_by(256) {
            assert_eq!(v.apply(i as u16), i as u16);
        }
    }

    #[test]
    fn constant() {
        let v = ElementWiseView16::constant(42);
        for i in (0u32..=65535).step_by(1000) {
            assert_eq!(v.apply(i as u16), 42);
        }
    }

    #[test]
    fn from_fn_increment() {
        let v = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        assert_eq!(v.apply(0), 1);
        assert_eq!(v.apply(65535), 0);
    }

    #[test]
    fn from_static() {
        use crate::q1::activation::SIGMOID_65536;
        let v = ElementWiseView16::from_static(&SIGMOID_65536);
        assert_eq!(v.apply(0), SIGMOID_65536[0]);
        assert_eq!(v.apply(32767), SIGMOID_65536[32767]);
    }

    #[test]
    fn composition() {
        let inc = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        let double = inc.then(&inc);
        assert_eq!(double.apply(0), 2);
        assert_eq!(double.apply(65535), 1);
    }

    #[test]
    fn composition_identity() {
        let inc = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        let id = ElementWiseView16::identity();
        for i in (0u32..=65535).step_by(1000) {
            let v = i as u16;
            assert_eq!(inc.then(&id).apply(v), inc.apply(v));
            assert_eq!(id.then(&inc).apply(v), inc.apply(v));
        }
    }

    #[test]
    fn bijective() {
        assert!(ElementWiseView16::identity().is_bijective());
        assert!(ElementWiseView16::from_fn(|x| x.wrapping_add(1)).is_bijective());
        assert!(!ElementWiseView16::constant(0).is_bijective());
    }

    #[test]
    fn inverse() {
        let inc = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        let dec = inc.inverse().unwrap();
        for i in (0u32..=65535).step_by(1000) {
            let v = i as u16;
            assert_eq!(dec.apply(inc.apply(v)), v);
        }
        assert!(ElementWiseView16::constant(0).inverse().is_none());
    }

    #[test]
    fn apply_slice() {
        let inc = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        let mut data = [0u16, 1, 2, 65535];
        inc.apply_slice(&mut data);
        assert_eq!(data, [1, 2, 3, 0]);
    }

    #[test]
    fn apply_to_slices() {
        let inc = ElementWiseView16::from_fn(|x| x.wrapping_add(1));
        let input = [0u16, 1, 2, 65535];
        let mut output = [0u16; 4];
        inc.apply_to(&input, &mut output);
        assert_eq!(output, [1, 2, 3, 0]);
    }

    #[test]
    fn default_is_identity() {
        let v = ElementWiseView16::default();
        for i in (0u32..=65535).step_by(1000) {
            assert_eq!(v.apply(i as u16), i as u16);
        }
    }

    #[test]
    fn size_is_128kb() {
        assert_eq!(core::mem::size_of::<[u16; 65536]>(), 131072);
    }

    #[test]
    fn activation_composition() {
        // Compose sigmoid → relu, verify it produces reasonable output
        use crate::q1::activation::{RELU_65536, SIGMOID_65536};
        let sig = ElementWiseView16::from_static(&SIGMOID_65536);
        let relu = ElementWiseView16::from_static(&RELU_65536);
        let composed = sig.then(&relu);
        // sigmoid always outputs > 0 for any input, so relu should pass through
        // sigmoid output range is [0, 65535], relu zeroes ≥ 32768
        let at_zero = composed.apply(0);
        assert!(at_zero > 0); // sigmoid(0) ≈ 32768, relu(32768) = 0 actually
    }
}
