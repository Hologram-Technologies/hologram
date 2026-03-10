//! IEEE 754 Byte-Plane Transposition.
//!
//! Observable: IEEE 754 structure — information content is highly non-uniform
//! across byte positions. The exponent bytes carry far less entropy than the
//! mantissa bytes. By transposing into separate byte planes, each plane can
//! be compressed independently with the best-suited algorithm.
//!
//! For N f32 values (4N bytes), we produce 4 planes of N bytes each:
//! - Plane 3: sign + exponent_hi (very low entropy)
//! - Plane 2: exponent_lo + mantissa_hi (medium entropy)
//! - Plane 1: mantissa_mid (moderate entropy)
//! - Plane 0: mantissa_lo (near-random)

use alloc::vec::Vec;

/// Number of byte planes for f32 data.
pub const F32_PLANES: usize = 4;

/// Number of byte planes for f64 data.
pub const F64_PLANES: usize = 8;

/// Transpose f32 data (as raw bytes, little-endian) into 4 separate byte planes.
///
/// Input: `data` with length divisible by 4 (each 4 bytes = one f32).
/// Output: 4 planes concatenated: [plane3 | plane2 | plane1 | plane0].
///
/// Returns None if input length is not divisible by 4.
pub fn transpose_f32(data: &[u8]) -> Option<Vec<u8>> {
    if !data.len().is_multiple_of(F32_PLANES) {
        return None;
    }
    let n = data.len() / F32_PLANES;
    let mut planes = alloc::vec![0u8; data.len()];

    for i in 0..n {
        let base = i * F32_PLANES;
        // Little-endian: byte 0 is LSB (mantissa_lo), byte 3 is MSB (sign+exp)
        planes[i] = data[base + 3]; // Plane 0 = MSB (sign+exp_hi)
        planes[n + i] = data[base + 2]; // Plane 1 = exp_lo+mantissa_hi
        planes[2 * n + i] = data[base + 1]; // Plane 2 = mantissa_mid
        planes[3 * n + i] = data[base]; // Plane 3 = mantissa_lo (LSB)
    }

    Some(planes)
}

/// Inverse transpose: reassemble f32 bytes from 4 byte planes.
///
/// Input: `planes` with layout [plane0 | plane1 | plane2 | plane3], length divisible by 4.
/// Output: interleaved f32 bytes (little-endian).
pub fn untranspose_f32(planes: &[u8]) -> Option<Vec<u8>> {
    if !planes.len().is_multiple_of(F32_PLANES) {
        return None;
    }
    let n = planes.len() / F32_PLANES;
    let mut data = alloc::vec![0u8; planes.len()];

    for i in 0..n {
        let base = i * F32_PLANES;
        data[base + 3] = planes[i]; // MSB
        data[base + 2] = planes[n + i];
        data[base + 1] = planes[2 * n + i];
        data[base] = planes[3 * n + i]; // LSB
    }

    Some(data)
}

/// Transpose f64 data (as raw bytes, little-endian) into 8 separate byte planes.
pub fn transpose_f64(data: &[u8]) -> Option<Vec<u8>> {
    if !data.len().is_multiple_of(F64_PLANES) {
        return None;
    }
    let n = data.len() / F64_PLANES;
    let mut planes = alloc::vec![0u8; data.len()];

    for i in 0..n {
        let base = i * F64_PLANES;
        for p in 0..F64_PLANES {
            // Plane 0 = MSB, Plane 7 = LSB
            planes[p * n + i] = data[base + (F64_PLANES - 1 - p)];
        }
    }

    Some(planes)
}

/// Inverse transpose f64: reassemble from 8 byte planes.
pub fn untranspose_f64(planes: &[u8]) -> Option<Vec<u8>> {
    if !planes.len().is_multiple_of(F64_PLANES) {
        return None;
    }
    let n = planes.len() / F64_PLANES;
    let mut data = alloc::vec![0u8; planes.len()];

    for i in 0..n {
        let base = i * F64_PLANES;
        for p in 0..F64_PLANES {
            data[base + (F64_PLANES - 1 - p)] = planes[p * n + i];
        }
    }

    Some(data)
}

/// Get a reference to a specific byte plane from transposed data.
pub fn plane_slice(transposed: &[u8], plane: usize, num_planes: usize) -> &[u8] {
    let n = transposed.len() / num_planes;
    &transposed[plane * n..(plane + 1) * n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn f32_round_trip() {
        // Some f32 values as bytes
        let values: &[f32] = &[1.0, -1.0, 0.0, 3.14, 42.0, f32::MIN, f32::MAX];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let transposed = transpose_f32(&bytes).unwrap();
        let recovered = untranspose_f32(&transposed).unwrap();
        assert_eq!(bytes, recovered);
    }

    #[test]
    fn f64_round_trip() {
        let values: &[f64] = &[1.0, -1.0, 0.0, 3.14159265, 42.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let transposed = transpose_f64(&bytes).unwrap();
        let recovered = untranspose_f64(&transposed).unwrap();
        assert_eq!(bytes, recovered);
    }

    #[test]
    fn f32_invalid_length() {
        assert!(transpose_f32(&[1, 2, 3]).is_none());
    }

    #[test]
    fn f64_invalid_length() {
        assert!(transpose_f64(&[1, 2, 3, 4, 5, 6, 7]).is_none());
    }

    #[test]
    fn f32_msb_plane_low_entropy() {
        // For small positive floats, the MSB (sign+exponent) should be very similar
        let values: Vec<f32> = (0..100).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let transposed = transpose_f32(&bytes).unwrap();

        let n = values.len();
        let msb_plane = &transposed[0..n]; // Plane 0 = MSB
                                           // Count distinct values — should be very few (same exponent range)
        let mut seen = [false; 256];
        let mut distinct = 0;
        for &b in msb_plane {
            if !seen[b as usize] {
                seen[b as usize] = true;
                distinct += 1;
            }
        }
        assert!(
            distinct <= 5,
            "MSB plane should have few distinct values, got {distinct}"
        );
    }

    #[test]
    fn plane_slice_correct() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(plane_slice(&data, 0, 4), &[1, 2]);
        assert_eq!(plane_slice(&data, 1, 4), &[3, 4]);
        assert_eq!(plane_slice(&data, 2, 4), &[5, 6]);
        assert_eq!(plane_slice(&data, 3, 4), &[7, 8]);
    }
}
