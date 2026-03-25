use super::helpers::*;
use crate::error::ExecResult;
use hologram_core::op::FloatDType;

pub(crate) fn dispatch_cast(
    inputs: &[&[u8]],
    from: FloatDType,
    to: FloatDType,
) -> ExecResult<Vec<u8>> {
    if from == to {
        return Ok(inputs[0].to_vec());
    }
    let data = inputs[0];
    let from_size = from.byte_size();

    // If the data doesn't divide evenly by the declared `from` dtype but
    // DOES divide evenly by the `to` dtype (or by 4 for f32), the upstream
    // already converted and this Cast is a no-op.  This handles chains of
    // Casts where dtype metadata wasn't fully propagated.
    if from_size > 0 && !data.len().is_multiple_of(from_size) {
        return Ok(data.to_vec());
    }

    match (from, to) {
        (FloatDType::I64, FloatDType::F32) => {
            let out: Vec<f32> = iter_i64(data).map(|v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::I32, FloatDType::F32) => {
            let out: Vec<f32> = iter_i32(data).map(|v| v as f32).collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::I64) => {
            let src = cast_f32(data)?;
            let out: Vec<i64> = src.iter().map(|&v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::F32, FloatDType::I32) => {
            let src = cast_f32(data)?;
            let out: Vec<i32> = src.iter().map(|&v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::Bool, FloatDType::F32) => {
            let out: Vec<f32> = data
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect();
            Ok(f32_vec_to_bytes(out))
        }
        (FloatDType::F32, FloatDType::Bool) => {
            let src = cast_f32(data)?;
            Ok(src
                .iter()
                .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
                .collect())
        }
        (FloatDType::I64, FloatDType::Bool) => Ok(iter_i64(data)
            .map(|v| if v != 0 { 1u8 } else { 0u8 })
            .collect()),
        (FloatDType::I64, FloatDType::I32) => {
            let out: Vec<i32> = iter_i64(data).map(|v| v as i32).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::I64) => {
            let out: Vec<i64> = iter_i32(data).map(|v| v as i64).collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::I32, FloatDType::Bool) => Ok(iter_i32(data)
            .map(|v| if v != 0 { 1u8 } else { 0u8 })
            .collect()),
        (FloatDType::Bool, FloatDType::I64) => {
            let out: Vec<i64> = data
                .iter()
                .map(|&v| if v != 0 { 1i64 } else { 0i64 })
                .collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        (FloatDType::Bool, FloatDType::I32) => {
            let out: Vec<i32> = data
                .iter()
                .map(|&v| if v != 0 { 1i32 } else { 0i32 })
                .collect();
            Ok(bytemuck::cast_slice(&out).to_vec())
        }
        // Fallback: pass-through (same bytes, different interpretation).
        _ => Ok(data.to_vec()),
    }
}

pub(crate) fn dispatch_embed(inputs: &[&[u8]], dim: usize, quant: u8) -> ExecResult<Vec<u8>> {
    // inputs[0] = token_ids (i64 or u32), inputs[1] = table (f32 or quantized) [vocab, dim]
    let raw = inputs[0];
    let table_bytes = inputs[1];
    let table = decode_weights(table_bytes, quant)?;

    let vocab = table.len() / dim;

    // Detect token ID dtype: i64 (8 bytes each) or u32 (4 bytes each).
    let token_ids: Vec<usize> = if raw.len().is_multiple_of(8) {
        iter_i64(raw).map(|v| v as usize).collect()
    } else {
        iter_i32(raw).map(|v| v as usize).collect()
    };

    let mut out = Vec::with_capacity(token_ids.len() * dim);
    for idx in &token_ids {
        if *idx >= vocab {
            return Err(crate::error::ExecError::ShapeMismatch {
                expected: format!("token id < {vocab}"),
                actual: format!("token id = {idx}"),
            });
        }
        out.extend_from_slice(&table[idx * dim..(idx + 1) * dim]);
    }
    Ok(f32_vec_to_bytes(out))
}

pub(crate) fn dispatch_dequantize(inputs: &[&[u8]]) -> ExecResult<Vec<u8>> {
    // Q4_0 dequantization: blocks of 18 bytes (2 byte scale + 16 nibbles = 32 values)
    let data = inputs[0];
    let block_size = 18;
    if !data.len().is_multiple_of(block_size) {
        // Not Q4_0 format — just pass through
        return Ok(data.to_vec());
    }
    let n_blocks = data.len() / block_size;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for block in data.chunks(block_size) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for byte_idx in 0..16 {
            let byte = block[2 + byte_idx];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    Ok(f32_vec_to_bytes(out))
}

/// Dequantize Q4_0 data: each 18-byte block produces 32 f32 values.
/// Format: 2-byte f16 scale + 16 bytes of nibble pairs (each nibble - 8).
fn dequantize_q4_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 32);
    for block in data.chunks(18) {
        if block.len() < 18 {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // ggml Q4_0 layout: low nibbles → positions 0..15, high nibbles → 16..31
        for byte_idx in 0..16 {
            let lo = (block[2 + byte_idx] & 0x0F) as i8 - 8;
            out.push(lo as f32 * scale);
        }
        for byte_idx in 0..16 {
            let hi = (block[2 + byte_idx] >> 4) as i8 - 8;
            out.push(hi as f32 * scale);
        }
    }
    out
}

/// Dequantize Q6_K data: 256 values per super-block (210 bytes each).
/// Layout: ql[128] + qh[64] + scales[16] + d(f16)[2] = 210 bytes.
/// Each value is a 6-bit signed integer (-32..31) scaled by (d * scale_i).
fn dequantize_q6_k(data: &[u8]) -> Vec<f32> {
    const QK: usize = 256;
    const BLOCK_SIZE: usize = QK / 2 + QK / 4 + QK / 16 + 2; // 128 + 64 + 16 + 2 = 210

    let n_blocks = data.len() / BLOCK_SIZE;
    let mut out = vec![0.0f32; n_blocks * QK];

    for (bi, block_data) in data.chunks(BLOCK_SIZE).enumerate() {
        if block_data.len() < BLOCK_SIZE {
            break;
        }
        let ql = &block_data[0..128];
        let qh = &block_data[128..192];
        let sc = &block_data[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block_data[208], block_data[209]]));
        let y = &mut out[bi * QK..];

        // Match ggml's dequantize_row_q6_K exactly:
        // Two passes of 128 values each, each pass processes 4 groups of 32.
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut y_off = 0usize;
        for n_pass in 0..2u8 {
            let is = (n_pass as usize) * 8; // scale index base
            for l in 0..32 {
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
                y[y_off + l] = d * sc[is] as i8 as f32 * q1 as f32;
                y[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            y_off += 128;
        }
    }
    out
}

/// Decode bytes as f32, applying dequantization if quant != 0.
/// quant: 0=f32, 1=Q4_0, 2=Q8_0, 3=Q6_K.
pub(crate) fn decode_weights(data: &[u8], quant: u8) -> ExecResult<std::borrow::Cow<'_, [f32]>> {
    match quant {
        1 => Ok(std::borrow::Cow::Owned(dequantize_q4_0(data))),
        3 => Ok(std::borrow::Cow::Owned(dequantize_q6_k(data))),
        // TODO: Q8_0 dequantization
        _ => cast_f32(data),
    }
}

#[inline]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // subnormal
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let f_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}
