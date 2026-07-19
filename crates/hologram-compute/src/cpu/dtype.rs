//! Dtype tags, re-derived from the canonical [`DTypeId`] in `hologram-types`.
//!
//! The `dtype: u8` field on every KernelCall variant carries one of these
//! constants; the CPU dispatcher uses them to select between byte-domain and
//! IEEE-754 native kernels. There is exactly **one** definition of each tag
//! (`DTypeId::*`) — these aliases exist so the kernels keep matching on plain
//! `u8` (the wire representation) without a second table drifting out of sync.
//!
//! Both accessors are **total**: an unrecognized tag yields `None` rather than
//! a plausible-looking default. The previous `_ => 1` byte width and `_ => 0`
//! dequantized value turned an unknown dtype into a silently wrong answer
//! instead of an error.

pub use hologram_types::DTypeId;

pub const DTYPE_BOOL: u8 = DTypeId::BOOL.raw();
pub const DTYPE_U8: u8 = DTypeId::U8.raw();
pub const DTYPE_I8: u8 = DTypeId::I8.raw();
pub const DTYPE_U64: u8 = DTypeId::U64.raw();
pub const DTYPE_I32: u8 = DTypeId::I32.raw();
pub const DTYPE_I64: u8 = DTypeId::I64.raw();
pub const DTYPE_F16: u8 = DTypeId::F16.raw();
pub const DTYPE_BF16: u8 = DTypeId::BF16.raw();
pub const DTYPE_F32: u8 = DTypeId::F32.raw();
pub const DTYPE_F64: u8 = DTypeId::F64.raw();
/// Packed signed 4-bit integer (two values per byte, low nibble first).
/// Sub-byte: [`bytes_per_element`] is `None`; storage is `ceil(n/2)`.
pub const DTYPE_I4: u8 = DTypeId::I4.raw();
/// E8 lattice-codebook vector-quantized weight: each 8-element subvector is one
/// `u8` codebook index → 1 bit per logical weight. The group dimension is 8
/// because E8 is 8-dimensional (definitional, not a tuning knob); the codebook
/// *contents* and entry count are per-model data carried as a constant operand.
/// Sub-byte: storage is `ceil(n/8)`.
pub const DTYPE_E8CB: u8 = DTypeId::E8CB.raw();

/// Bytes per element. `None` for the sub-byte tiers (`I4`, `E8CB`) — whose
/// storage is not `n × width` — and for any unrecognized tag. Callers size
/// buffers with [`storage_bytes`].
#[must_use]
pub const fn bytes_per_element(dtype: u8) -> Option<usize> {
    DTypeId(dtype).bytes_per_element()
}

/// Storage bytes for an `n`-element buffer, honouring sub-byte packing
/// (`I4` → `ceil(n/2)`; `E8CB` → one index byte per 8-element group).
/// `None` for an unrecognized tag.
#[must_use]
pub const fn storage_bytes(dtype: u8, element_count: u32) -> Option<u32> {
    DTypeId(dtype).storage_bytes(element_count)
}

/// Whether a dtype is float-typed (selects native IEEE-754 kernel paths).
#[must_use]
pub const fn is_float(dtype: u8) -> bool {
    DTypeId(dtype).is_float()
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
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
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
        // Unreachable: dispatch rejects f64 explicitly and routes the byte ring
        // to its own element readers, so this float reader only ever sees
        // f32/f16/bf16. Fail loud rather than silently substituting 0.0 (a
        // silent-wrong surface) if that invariant is ever violated.
        other => unreachable!("read_float on non-float dtype tag {other}"),
    }
}

/// Write element `i` of a float-typed buffer from f32.
#[inline]
pub fn write_float(bytes: &mut [u8], i: usize, v: f32, dtype: u8) {
    match dtype {
        DTYPE_F32 => write_f32(bytes, i, v),
        DTYPE_BF16 => write_bf16(bytes, i, v),
        DTYPE_F16 => write_f16(bytes, i, v),
        // Unreachable for the same reason as `read_float`: fail loud rather
        // than silently dropping the write on a non-float dtype.
        other => unreachable!("write_float on non-float dtype tag {other}"),
    }
}
