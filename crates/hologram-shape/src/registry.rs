//! Shape registry: maps buffer slot indices to their tensor shapes.

use crate::TensorShape;

/// Maps arena buffer indices to their `TensorShape`.
///
/// Pre-allocated and growable. Each slot can be set, queried, or cleared
/// independently.
pub struct ShapeRegistry {
    shapes: Vec<Option<TensorShape>>,
}

impl ShapeRegistry {
    /// Create a new registry pre-allocated for `capacity` slots.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let mut shapes = Vec::with_capacity(capacity);
        shapes.resize_with(capacity, || None);
        Self { shapes }
    }

    /// Store a shape for buffer slot `index`. Auto-grows if needed.
    pub fn set(&mut self, index: usize, shape: TensorShape) {
        if index >= self.shapes.len() {
            self.shapes.resize_with(index + 1, || None);
        }
        self.shapes[index] = Some(shape);
    }

    /// Look up the shape for buffer slot `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&TensorShape> {
        self.shapes.get(index).and_then(|s| s.as_ref())
    }

    /// Clear the shape for a buffer slot (e.g. when the buffer is evicted).
    pub fn clear(&mut self, index: usize) {
        if let Some(slot) = self.shapes.get_mut(index) {
            *slot = None;
        }
    }

    /// Reset all shapes (e.g. between inference runs).
    pub fn reset(&mut self) {
        for slot in &mut self.shapes {
            *slot = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::FloatDType;

    #[test]
    fn test_set_and_get() {
        let mut reg = ShapeRegistry::new(4);
        let shape = TensorShape::new(FloatDType::F32, &[1, 13, 2048]);
        reg.set(2, shape.clone());

        assert_eq!(reg.get(2), Some(&shape));
        assert_eq!(reg.get(0), None);
        assert_eq!(reg.get(3), None);
    }

    #[test]
    fn test_auto_grow() {
        let mut reg = ShapeRegistry::new(2);
        let shape = TensorShape::new(FloatDType::F32, &[4, 8]);
        reg.set(10, shape.clone());

        assert_eq!(reg.get(10), Some(&shape));
        assert_eq!(reg.get(0), None);
        assert_eq!(reg.get(5), None);
    }

    #[test]
    fn test_clear() {
        let mut reg = ShapeRegistry::new(4);
        let shape = TensorShape::new(FloatDType::F32, &[2, 3]);
        reg.set(1, shape);

        assert!(reg.get(1).is_some());
        reg.clear(1);
        assert_eq!(reg.get(1), None);
    }

    #[test]
    fn test_clear_out_of_range_is_noop() {
        let mut reg = ShapeRegistry::new(2);
        reg.clear(100); // should not panic
    }

    #[test]
    fn test_reset() {
        let mut reg = ShapeRegistry::new(4);
        reg.set(0, TensorShape::new(FloatDType::F32, &[1]));
        reg.set(2, TensorShape::new(FloatDType::F16, &[2, 3]));

        reg.reset();
        assert_eq!(reg.get(0), None);
        assert_eq!(reg.get(2), None);
    }

    #[test]
    fn test_overwrite() {
        let mut reg = ShapeRegistry::new(4);
        let a = TensorShape::new(FloatDType::F32, &[4]);
        let b = TensorShape::new(FloatDType::I64, &[8]);
        reg.set(0, a);
        reg.set(0, b.clone());

        assert_eq!(reg.get(0), Some(&b));
    }

    #[test]
    fn test_get_beyond_capacity() {
        let reg = ShapeRegistry::new(2);
        assert_eq!(reg.get(100), None);
    }
}
