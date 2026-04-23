//! Dtype conversion dispatch and half-float (f16) utilities.
//!
//! Extracted from `cpu.rs` — these are pure functions with no dependency
//! on `CpuBackend` state, used by both the cast op and Q4_0 dequantization.

use crate::BackendError;
use hologram_core::op::FloatDType;

/// Convert f16 bits to f32.
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16.
        let mut m = mantissa;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = (127 - 15 - e) as u32;
        let f32_mantissa = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa);
    }
    if exp == 0x1F {
        // Inf/NaN.
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }
    let f32_exp = (exp as i32 - 15 + 127) as u32;
    let f32_mantissa = mantissa << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
}

/// Convert f32 to f16 bits.
pub(crate) fn f32_to_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf or NaN.
        return sign | 0x7C00 | ((mantissa >> 13) as u16 & 0x3FF);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf.
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow → zero (subnormals not handled for simplicity).
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Dispatch a dtype cast operation.
pub(crate) fn dispatch_cast(
    input: &[u8],
    from: FloatDType,
    to: FloatDType,
) -> crate::Result<Vec<u8>> {
    if from == to {
        return Ok(input.to_vec());
    }

    match (from, to) {
        (FloatDType::F32, FloatDType::I64) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<i64> = in_f.iter().map(|&v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I64, FloatDType::F32) => {
            let in_i: &[i64] = bytemuck::cast_slice(input);
            let out: Vec<f32> = in_i.iter().map(|&v| v as f32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::I32) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<i32> = in_f.iter().map(|&v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::F32) => {
            let in_i: &[i32] = bytemuck::cast_slice(input);
            let out: Vec<f32> = in_i.iter().map(|&v| v as f32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::F16) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<u16> = in_f.iter().map(|&v| f32_to_half(v)).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F16, FloatDType::F32) => {
            let in_h: &[u16] = bytemuck::cast_slice(input);
            let out: Vec<f32> = in_h.iter().map(|&v| half_to_f32(v)).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::Bool) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<u8> = in_f.iter().map(|&v| u8::from(v != 0.0)).collect();
            Ok(out)
        }
        (FloatDType::Bool, FloatDType::F32) => {
            let out: Vec<f32> = input
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I64, FloatDType::I32) => {
            let in_i: &[i64] = bytemuck::cast_slice(input);
            let out: Vec<i32> = in_i.iter().map(|&v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::I64) => {
            let in_i: &[i32] = bytemuck::cast_slice(input);
            let out: Vec<i64> = in_i.iter().map(|&v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::U8) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<u8> = in_f.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            Ok(out)
        }
        (FloatDType::U8, FloatDType::F32) => {
            let out: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::F64) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<f64> = in_f.iter().map(|&v| v as f64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F64, FloatDType::F32) => {
            let in_f: &[f64] = bytemuck::cast_slice(input);
            let out: Vec<f32> = in_f.iter().map(|&v| v as f32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::I8) => {
            let in_f: &[f32] = bytemuck::cast_slice(input);
            let out: Vec<i8> = in_f.iter().map(|&v| v.clamp(-128.0, 127.0) as i8).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I8, FloatDType::F32) => {
            let in_i: &[i8] = bytemuck::cast_slice(input);
            let out: Vec<f32> = in_i.iter().map(|&v| v as f32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        _ => Err(BackendError::Unsupported(format!(
            "cast from {from:?} to {to:?} not supported"
        ))),
    }
}
