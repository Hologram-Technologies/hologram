//! Dtype tag mapping (mirrors `hologram-types::DTypeKind` numeric encoding).
//!
//! The `dtype: u8` field on every KernelCall variant carries one of these
//! constants. The CPU dispatcher uses them to select between byte-domain and
//! IEEE-754 native kernels.

pub const DTYPE_BOOL: u8 = 0;
pub const DTYPE_U8: u8 = 1;
pub const DTYPE_I8: u8 = 2;
pub const DTYPE_U64: u8 = 3;
pub const DTYPE_I32: u8 = 4;
pub const DTYPE_I64: u8 = 5;
pub const DTYPE_F16: u8 = 6;
pub const DTYPE_BF16: u8 = 7;
pub const DTYPE_F32: u8 = 8;
pub const DTYPE_F64: u8 = 9;
/// Packed signed 4-bit integer (two values per byte). Used by quantized
/// weight payloads — see spec X-5 / ADR-054 Quantization addendum.
/// `bytes_per_element` returns the integer 0; ceil-division by 2 is
/// applied at the kernel boundary. The compiler treats the storage
/// length explicitly to avoid the 0-bytes-per-element pitfall.
pub const DTYPE_I4: u8 = 10;

/// Bytes per element for a given dtype tag. Sub-byte dtypes
/// (`DTYPE_I4`) report `0`; callers compute storage size as
/// `(element_count + 1) / 2` for I4.
pub const fn bytes_per_element(dtype: u8) -> usize {
    match dtype {
        DTYPE_BOOL | DTYPE_U8 | DTYPE_I8 => 1,
        DTYPE_F16 | DTYPE_BF16 => 2,
        DTYPE_I32 | DTYPE_F32 => 4,
        DTYPE_U64 | DTYPE_I64 | DTYPE_F64 => 8,
        DTYPE_I4 => 0, // sub-byte; storage = ceil(n/2)
        _ => 1,
    }
}

/// Storage bytes for an `n`-element buffer of the given dtype, accounting
/// for sub-byte packing (I4 → ceil(n/2)).
pub const fn storage_bytes(dtype: u8, element_count: u32) -> u32 {
    match dtype {
        DTYPE_I4 => element_count.div_ceil(2),
        _ => element_count * (bytes_per_element(dtype) as u32),
    }
}

/// Whether a dtype is float-typed (selects native IEEE-754 kernel paths).
pub const fn is_float(dtype: u8) -> bool {
    matches!(dtype, DTYPE_F16 | DTYPE_BF16 | DTYPE_F32 | DTYPE_F64)
}

#[inline]
pub fn read_f32(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
}

#[inline]
pub fn write_f32(bytes: &mut [u8], i: usize, v: f32) {
    bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn read_bf16(bytes: &[u8], i: usize) -> f32 {
    let lo = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
    let bits = (lo as u32) << 16;
    f32::from_bits(bits)
}

#[inline]
pub fn write_bf16(bytes: &mut [u8], i: usize, v: f32) {
    let hi = (v.to_bits() >> 16) as u16;
    bytes[i * 2..i * 2 + 2].copy_from_slice(&hi.to_le_bytes());
}

#[inline]
pub fn read_f16(bytes: &[u8], i: usize) -> f32 {
    let bits = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
    f16_to_f32(bits)
}

#[inline]
pub fn write_f16(bytes: &mut [u8], i: usize, v: f32) {
    let bits = f32_to_f16(v);
    bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
}

/// IEEE-754 binary16 → binary32 (round-to-nearest-even).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — normalize.
            let mut m = mant;
            let mut e: i32 = 1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((127 - 15 + e) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// IEEE-754 binary32 → binary16 (round-to-nearest-even).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        let nan_bit = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | (0x1f << 10) | nan_bit;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | (0x1f << 10);
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        let m = mant | 0x800000;
        let shift = (1 - new_exp) as u32 + 13;
        let rounded = (m + (1 << (shift - 1))) >> shift;
        return (sign << 15) | (rounded as u16);
    }
    let rounded_mant = ((mant + 0x1000) >> 13) as u16;
    if rounded_mant & 0x400 != 0 {
        // Mantissa overflowed; bump exponent.
        let bumped_exp = new_exp + 1;
        if bumped_exp >= 0x1f {
            return (sign << 15) | (0x1f << 10);
        }
        return (sign << 15) | ((bumped_exp as u16) << 10);
    }
    (sign << 15) | ((new_exp as u16) << 10) | (rounded_mant & 0x3ff)
}

/// Read element `i` of a float-typed buffer as f32.
#[inline]
pub fn read_float(bytes: &[u8], i: usize, dtype: u8) -> f32 {
    match dtype {
        DTYPE_F32 => read_f32(bytes, i),
        DTYPE_BF16 => read_bf16(bytes, i),
        DTYPE_F16 => read_f16(bytes, i),
        _ => 0.0,
    }
}

/// Write element `i` of a float-typed buffer from f32.
#[inline]
pub fn write_float(bytes: &mut [u8], i: usize, v: f32, dtype: u8) {
    match dtype {
        DTYPE_F32 => write_f32(bytes, i, v),
        DTYPE_BF16 => write_bf16(bytes, i, v),
        DTYPE_F16 => write_f16(bytes, i, v),
        _ => {}
    }
}
