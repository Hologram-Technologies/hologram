//! Core `TensorShape` type: concrete dimensions + element dtype.

use hologram_core::op::FloatDType;
use smallvec::SmallVec;
use std::fmt;

/// A concrete tensor shape with element data type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes (e.g. `[1, 13, 2048]`).
    pub dims: SmallVec<[usize; 4]>,
    /// Element data type.
    pub dtype: FloatDType,
}

impl TensorShape {
    /// Create a new tensor shape from a dtype and dimension slice.
    #[must_use]
    pub fn new(dtype: FloatDType, dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
            dtype,
        }
    }

    /// Scalar (0-dimensional) shape.
    #[must_use]
    pub fn scalar(dtype: FloatDType) -> Self {
        Self {
            dims: SmallVec::new(),
            dtype,
        }
    }

    /// 1-D vector shape.
    #[must_use]
    pub fn vector(dtype: FloatDType, len: usize) -> Self {
        Self {
            dims: smallvec::smallvec![len],
            dtype,
        }
    }

    /// 2-D matrix shape.
    #[must_use]
    pub fn matrix(dtype: FloatDType, rows: usize, cols: usize) -> Self {
        Self {
            dims: smallvec::smallvec![rows, cols],
            dtype,
        }
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (product of all dimensions). Returns 1 for scalars.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Total byte length: `total_elements * dtype.byte_size()`.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.total_elements() * self.dtype.byte_size()
    }

    /// Whether this is a scalar (0-dimensional).
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// The last (innermost) dimension, or `None` for scalars.
    #[must_use]
    pub fn last_dim(&self) -> Option<usize> {
        self.dims.last().copied()
    }

    /// Return a new shape with dimension `axis` replaced by `new_size`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    #[must_use]
    pub fn with_replaced_dim(&self, axis: usize, new_size: usize) -> Self {
        assert!(
            axis < self.ndim(),
            "axis {} out of range for {}-D tensor",
            axis,
            self.ndim()
        );
        let mut new_dims = self.dims.clone();
        new_dims[axis] = new_size;
        Self {
            dims: new_dims,
            dtype: self.dtype,
        }
    }
}

/// Format for display, e.g. `dtype_name`.
fn dtype_str(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F32 => "f32",
        FloatDType::F64 => "f64",
        FloatDType::I32 => "i32",
        FloatDType::I64 => "i64",
        FloatDType::F16 => "f16",
        FloatDType::BF16 => "bf16",
        FloatDType::U8 => "u8",
        FloatDType::Bool => "bool",
        FloatDType::I8 => "i8",
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]:{}", dtype_str(self.dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_accessors() {
        let s = TensorShape::new(FloatDType::F32, &[1, 13, 2048]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.total_elements(), 1 * 13 * 2048);
        assert_eq!(s.byte_len(), 1 * 13 * 2048 * 4);
        assert_eq!(s.last_dim(), Some(2048));
        assert!(!s.is_scalar());
    }

    #[test]
    fn test_scalar() {
        let s = TensorShape::scalar(FloatDType::F32);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.total_elements(), 1);
        assert_eq!(s.byte_len(), 4);
        assert!(s.is_scalar());
        assert_eq!(s.last_dim(), None);
    }

    #[test]
    fn test_vector() {
        let s = TensorShape::vector(FloatDType::F16, 256);
        assert_eq!(s.ndim(), 1);
        assert_eq!(s.total_elements(), 256);
        assert_eq!(s.byte_len(), 256 * 2);
    }

    #[test]
    fn test_matrix() {
        let s = TensorShape::matrix(FloatDType::I64, 4, 8);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.total_elements(), 32);
        assert_eq!(s.byte_len(), 32 * 8);
    }

    #[test]
    fn test_display() {
        let s = TensorShape::new(FloatDType::F32, &[1, 13, 2048]);
        assert_eq!(format!("{s}"), "[1, 13, 2048]:f32");

        let scalar = TensorShape::scalar(FloatDType::BF16);
        assert_eq!(format!("{scalar}"), "[]:bf16");
    }

    #[test]
    fn test_with_replaced_dim() {
        let s = TensorShape::new(FloatDType::F32, &[1, 13, 2048]);
        let replaced = s.with_replaced_dim(1, 77);
        assert_eq!(replaced.dims.as_slice(), &[1, 77, 2048]);
        assert_eq!(replaced.dtype, FloatDType::F32);
    }

    #[test]
    #[should_panic(expected = "axis 3 out of range")]
    fn test_with_replaced_dim_out_of_range() {
        let s = TensorShape::new(FloatDType::F32, &[1, 13, 2048]);
        let _ = s.with_replaced_dim(3, 77);
    }

    #[test]
    fn test_byte_len_f16() {
        let s = TensorShape::new(FloatDType::F16, &[2, 3]);
        assert_eq!(s.byte_len(), 6 * 2);
    }

    #[test]
    fn test_clone_eq() {
        let a = TensorShape::new(FloatDType::F32, &[4, 8]);
        let b = a.clone();
        assert_eq!(a, b);
    }
}
