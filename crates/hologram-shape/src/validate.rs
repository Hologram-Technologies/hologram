//! Validation helpers for buffer shapes.

use crate::infer::ShapeError;
use crate::TensorShape;
use smallvec::SmallVec;

/// Validate that a buffer's byte length matches the registered shape.
///
/// Returns `Ok(())` if `data.len() == shape.byte_len()`, otherwise returns
/// an `Incompatible` error with details.
pub fn validate_buffer_shape(data: &[u8], shape: &TensorShape) -> Result<(), ShapeError> {
    let expected = shape.byte_len();
    if data.len() != expected {
        Err(ShapeError::Incompatible {
            op: "validate",
            detail: format!(
                "buffer has {} bytes but shape {shape} expects {expected} bytes",
                data.len()
            ),
        })
    } else {
        Ok(())
    }
}

/// Compute numpy-style broadcast shape for two input shapes.
///
/// Aligns dimensions from the right. A dimension of 1 broadcasts to any size.
/// Returns an error if dimensions are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<SmallVec<[usize; 4]>, ShapeError> {
    let max_ndim = a.len().max(b.len());
    let mut result = SmallVec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(ShapeError::Incompatible {
                op: "broadcast",
                detail: format!("cannot broadcast dim {da} with {db} (shapes {a:?} vs {b:?})"),
            });
        }
    }

    result.reverse();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::FloatDType;

    #[test]
    fn test_validate_correct_length() {
        let shape = TensorShape::new(FloatDType::F32, &[2, 3]);
        let data = vec![0u8; 24]; // 2 * 3 * 4 bytes
        assert!(validate_buffer_shape(&data, &shape).is_ok());
    }

    #[test]
    fn test_validate_incorrect_length() {
        let shape = TensorShape::new(FloatDType::F32, &[2, 3]);
        let data = vec![0u8; 20]; // wrong
        let result = validate_buffer_shape(&data, &shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_scalar() {
        let shape = TensorShape::scalar(FloatDType::F32);
        let data = vec![0u8; 4];
        assert!(validate_buffer_shape(&data, &shape).is_ok());
    }

    #[test]
    fn test_validate_f16() {
        let shape = TensorShape::new(FloatDType::F16, &[4, 8]);
        let data = vec![0u8; 64]; // 4 * 8 * 2 bytes
        assert!(validate_buffer_shape(&data, &shape).is_ok());
    }

    // ── broadcast_shapes tests ───────────────────────────────────────

    #[test]
    fn test_broadcast_same() {
        let result = broadcast_shapes(&[4, 8], &[4, 8]).expect("same shapes should broadcast");
        assert_eq!(result.as_slice(), &[4, 8]);
    }

    #[test]
    fn test_broadcast_one_to_many() {
        let result = broadcast_shapes(&[4, 8], &[1, 8]).expect("1-to-many should broadcast");
        assert_eq!(result.as_slice(), &[4, 8]);
    }

    #[test]
    fn test_broadcast_different_ranks() {
        let result = broadcast_shapes(&[2, 3, 4], &[4]).expect("different ranks should broadcast");
        assert_eq!(result.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let result = broadcast_shapes(&[4, 8], &[]).expect("scalar should broadcast");
        assert_eq!(result.as_slice(), &[4, 8]);
    }

    #[test]
    fn test_broadcast_both_scalar() {
        let result = broadcast_shapes(&[], &[]).expect("both scalar should broadcast");
        assert!(result.is_empty());
    }

    #[test]
    fn test_broadcast_incompatible() {
        let result = broadcast_shapes(&[3, 4], &[5, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_complex() {
        let result =
            broadcast_shapes(&[1, 3, 1], &[2, 1, 4]).expect("complex broadcast should work");
        assert_eq!(result.as_slice(), &[2, 3, 4]);
    }
}
