//! Fused dequant-matmul paths for Q4_0, Q6_K, and Q8 quantized weights.
//!
//! Instead of dequantizing the entire K×N weight matrix to f32 (which doubles
//! memory bandwidth and allocates K×N×4 bytes), dequantize one KC×NR panel at
//! a time directly into the stack-allocated packed_b buffer.  The micro-kernel
//! is unchanged — it operates on the same packed f32 panel.

use super::matmul::{micro_kernel_packed, MatmulRemainderLayout, KC};
#[cfg(feature = "parallel")]
use super::matmul::{SendPtr, PAR_M_TILE_THRESHOLD};

// ── Fused Q4_0 dequant-matmul ─────────────────────────────────────────

/// Q4_0 block size: 18 bytes → 32 f32 values.
pub(crate) const Q4_0_BLOCK_BYTES: usize = 18;
/// Number of f32 values produced by one Q4_0 block.
pub(crate) const Q4_0_BLOCK_VALUES: usize = 32;

/// Dequantize a KC×NR panel of Q4_0 weights directly into a packed f32 buffer.
///
/// Reads Q4_0 blocks from `b_q4` (row-major K×N layout, where each row of N
/// elements is stored as N/32 blocks of 18 bytes) and writes dequantized f32s
/// into `packed` with NR stride (same layout as `pack_b_panel`).
///
/// Requires: `n` is a multiple of `Q4_0_BLOCK_VALUES` (32).
#[inline]
fn dequant_pack_q4_0_panel<const NR: usize>(
    b_q4: &[u8],
    packed: &mut [f32],
    k_start: usize,
    k_end: usize,
    j: usize,
    n: usize,
) {
    let blocks_per_row = n / Q4_0_BLOCK_VALUES;
    let block_col = j / Q4_0_BLOCK_VALUES;
    let pos_in_block = j % Q4_0_BLOCK_VALUES;

    for p_idx in 0..(k_end - k_start) {
        let p = k_start + p_idx;
        let block_offset = (p * blocks_per_row + block_col) * Q4_0_BLOCK_BYTES;
        let block = &b_q4[block_offset..block_offset + Q4_0_BLOCK_BYTES];
        let scale = super::cast::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

        let dst = &mut packed[p_idx * NR..(p_idx + 1) * NR];
        for (jj, d) in dst.iter_mut().enumerate() {
            let pos = pos_in_block + jj;
            let val = if pos < 16 {
                (block[2 + pos] & 0x0F) as i8 - 8
            } else {
                (block[2 + pos - 16] >> 4) as i8 - 8
            };
            *d = val as f32 * scale;
        }
    }
}

/// Dequantize a row segment of Q4_0 weights into a contiguous f32 buffer.
///
/// Used for remainder paths where packing is not worthwhile.  Dequantizes
/// `b_q4[row, col_start..col_end]` into `out`.
#[inline]
fn dequant_q4_0_row_segment(
    b_q4: &[u8],
    out: &mut [f32],
    row: usize,
    col_start: usize,
    n_cols: usize,
    n: usize,
) {
    let blocks_per_row = n / Q4_0_BLOCK_VALUES;
    for (jj, o) in out.iter_mut().enumerate().take(n_cols) {
        let col = col_start + jj;
        let block_col = col / Q4_0_BLOCK_VALUES;
        let pos = col % Q4_0_BLOCK_VALUES;
        let block_offset = (row * blocks_per_row + block_col) * Q4_0_BLOCK_BYTES;
        let block = &b_q4[block_offset..block_offset + Q4_0_BLOCK_BYTES];
        let scale = super::cast::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let val = if pos < 16 {
            (block[2 + pos] & 0x0F) as i8 - 8
        } else {
            (block[2 + pos - 16] >> 4) as i8 - 8
        };
        *o = val as f32 * scale;
    }
}

/// Process one MR-row strip: dequant-pack B panels and run micro-kernel.
#[inline]
fn dequant_q4_0_m_strip(a: &[f32], b_q4: &[u8], out_ptr: *mut f32, layout: MatmulRemainderLayout) {
    let MatmulRemainderLayout {
        i,
        m_rem: _,
        k,
        n,
        n_tiles,
        n_rem,
    } = layout;
    const MR: usize = 4;
    const NR: usize = 8;
    let mut packed_b = [0.0f32; KC * NR];

    // Tiled body: MR×NR output tiles with KC blocking + on-the-fly dequant.
    for jt in 0..n_tiles {
        let j = jt * NR;
        let mut acc = [[0.0f32; NR]; MR];
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            let kc_len = kc_end - kc_start;
            dequant_pack_q4_0_panel::<NR>(
                b_q4,
                &mut packed_b[..kc_len * NR],
                kc_start,
                kc_end,
                j,
                n,
            );
            micro_kernel_packed::<MR, NR>(
                a,
                &packed_b[..kc_len * NR],
                &mut acc,
                i,
                kc_start,
                kc_end,
                k,
            );
        }
        for (ii, acc_row) in acc.iter().enumerate() {
            let off = (i + ii) * n + j;
            unsafe { std::ptr::copy_nonoverlapping(acc_row.as_ptr(), out_ptr.add(off), NR) };
        }
    }

    // Remainder columns — dequant one element at a time, accumulate scalar.
    if n_rem > 0 {
        let j = n_tiles * NR;
        let mut b_val = 0.0f32;
        for jj in 0..n_rem {
            let mut acc = [0.0f32; MR];
            for kc_start in (0..k).step_by(KC) {
                let kc_end = (kc_start + KC).min(k);
                for p in kc_start..kc_end {
                    dequant_q4_0_row_segment(
                        b_q4,
                        std::slice::from_mut(&mut b_val),
                        p,
                        j + jj,
                        1,
                        n,
                    );
                    for (ii, a_acc) in acc.iter_mut().enumerate() {
                        *a_acc += a[(i + ii) * k + p] * b_val;
                    }
                }
            }
            for (ii, &a_acc) in acc.iter().enumerate() {
                unsafe { *out_ptr.add((i + ii) * n + j + jj) = a_acc };
            }
        }
    }
}

/// Process remainder rows (< MR): dequant one B row at a time, scalar accumulate.
fn dequant_q4_0_remainder_rows(
    a: &[f32],
    b_q4: &[u8],
    out: &mut [f32],
    m_start: usize,
    m_rem: usize,
    k: usize,
    n: usize,
) {
    let mut b_row = vec![0.0f32; n];
    for ii in 0..m_rem {
        let row = m_start + ii;
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            for p in kc_start..kc_end {
                let a_val = a[row * k + p];
                dequant_q4_0_row_segment(b_q4, &mut b_row, p, 0, n, n);
                let o_row = &mut out[row * n..(row + 1) * n];
                for j in 0..n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}

/// Fused Q4_0 dequantize-matmul: C[m,n] += A[m,k] × dequant(B_q4[k,n]).
///
/// Same tiling structure as `matmul_k_outer` (KC=256, MR=4, NR=8) but replaces
/// B-panel packing with on-the-fly Q4_0 dequantization.  Never materializes
/// the full K×N f32 weight matrix — only a KC×NR panel (8 KB) lives on stack.
///
/// Requires: `n` is a multiple of 32, `k * n / 32 * 18 == b_q4.len()`.
pub(crate) fn matmul_dequant_q4_0(
    a: &[f32],
    b_q4: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_tiles = m / MR;
    let n_tiles = n / NR;
    let m_rem = m % MR;
    let n_rem = n % NR;

    #[cfg(feature = "parallel")]
    if m_tiles >= PAR_M_TILE_THRESHOLD {
        use rayon::prelude::*;
        let out_ptr = SendPtr(out.as_mut_ptr());
        let n_threads = rayon::current_num_threads();
        let duty = m_tiles.div_ceil(n_threads);
        (0..m_tiles)
            .into_par_iter()
            .with_min_len(duty)
            .for_each(|it| {
                let ptr = out_ptr;
                dequant_q4_0_m_strip(
                    a,
                    b_q4,
                    ptr.0,
                    MatmulRemainderLayout::new(it * MR, k, n, n_tiles, n_rem),
                );
            });
        if m_rem > 0 {
            dequant_q4_0_remainder_rows(a, b_q4, out, m_tiles * MR, m_rem, k, n);
        }
        return;
    }

    let out_ptr = out.as_mut_ptr();
    for it in 0..m_tiles {
        dequant_q4_0_m_strip(
            a,
            b_q4,
            out_ptr,
            MatmulRemainderLayout::new(it * MR, k, n, n_tiles, n_rem),
        );
    }
    if m_rem > 0 {
        dequant_q4_0_remainder_rows(a, b_q4, out, m_tiles * MR, m_rem, k, n);
    }
}

// ── Fused Q6_K dequant-matmul ─────────────────────────────────────────
//
// Same strategy as Q4_0 above: dequantize one KC×NR panel at a time into
// the stack-allocated packed_b buffer, then run the standard micro-kernel.
// Q6_K super-blocks are 210 bytes → 256 f32 values each.

/// Q6_K super-block size in bytes.
pub(crate) const Q6_K_BLOCK_BYTES: usize = 210;
/// Number of f32 values produced by one Q6_K super-block.
pub(crate) const Q6_K_BLOCK_VALUES: usize = 256;

/// Dequantize one Q6_K value at position `pos_in_block` (0..255) from a
/// 210-byte super-block.  Returns the dequantized f32.
#[inline(always)]
fn dequant_q6_k_value(block: &[u8], pos: usize) -> f32 {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = super::cast::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    // Which pass (0 or 1) and position within pass (0..127).
    let pass = pos / 128;
    let pos_in_pass = pos % 128;
    // Which group of 32 within the pass (0..3).
    let group = pos_in_pass / 32;
    let l = pos_in_pass % 32;

    let ql_off = pass * 64;
    let qh_off = pass * 32;
    let is = pass * 8;

    let q = match group {
        0 => ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32,
        1 => ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32,
        2 => ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32,
        3 => ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32,
        _ => unreachable!(),
    };
    let scale_idx = is + group * 2;
    d * sc[scale_idx] as i8 as f32 * q as f32
}

/// Dequantize a KC×NR panel of Q6_K weights directly into a packed f32 buffer.
///
/// Reads Q6_K super-blocks from `b_q6k` (row-major K×N layout, where each row
/// of N elements is stored as N/256 super-blocks of 210 bytes) and writes
/// dequantized f32s into `packed` with NR stride.
///
/// Requires: `n` is a multiple of `Q6_K_BLOCK_VALUES` (256).
#[inline]
fn dequant_pack_q6_k_panel<const NR: usize>(
    b_q6k: &[u8],
    packed: &mut [f32],
    k_start: usize,
    k_end: usize,
    j: usize,
    n: usize,
) {
    let blocks_per_row = n / Q6_K_BLOCK_VALUES;
    let block_col = j / Q6_K_BLOCK_VALUES;
    let pos_in_block = j % Q6_K_BLOCK_VALUES;

    for p_idx in 0..(k_end - k_start) {
        let p = k_start + p_idx;
        let block_offset = (p * blocks_per_row + block_col) * Q6_K_BLOCK_BYTES;
        let block = &b_q6k[block_offset..block_offset + Q6_K_BLOCK_BYTES];

        let dst = &mut packed[p_idx * NR..(p_idx + 1) * NR];
        for (jj, d) in dst.iter_mut().enumerate() {
            *d = dequant_q6_k_value(block, pos_in_block + jj);
        }
    }
}

/// Dequantize a row segment of Q6_K weights into a contiguous f32 buffer.
#[inline]
fn dequant_q6_k_row_segment(
    b_q6k: &[u8],
    out: &mut [f32],
    row: usize,
    col_start: usize,
    n_cols: usize,
    n: usize,
) {
    let blocks_per_row = n / Q6_K_BLOCK_VALUES;
    for (jj, o) in out.iter_mut().enumerate().take(n_cols) {
        let col = col_start + jj;
        let block_col = col / Q6_K_BLOCK_VALUES;
        let pos = col % Q6_K_BLOCK_VALUES;
        let block_offset = (row * blocks_per_row + block_col) * Q6_K_BLOCK_BYTES;
        let block = &b_q6k[block_offset..block_offset + Q6_K_BLOCK_BYTES];
        *o = dequant_q6_k_value(block, pos);
    }
}

/// Process one MR-row strip: dequant-pack Q6_K B panels and run micro-kernel.
#[inline]
fn dequant_q6_k_m_strip(a: &[f32], b_q6k: &[u8], out_ptr: *mut f32, layout: MatmulRemainderLayout) {
    let MatmulRemainderLayout {
        i,
        m_rem: _,
        k,
        n,
        n_tiles,
        n_rem,
    } = layout;
    const MR: usize = 4;
    const NR: usize = 8;
    let mut packed_b = [0.0f32; KC * NR];

    for jt in 0..n_tiles {
        let j = jt * NR;
        let mut acc = [[0.0f32; NR]; MR];
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            let kc_len = kc_end - kc_start;
            dequant_pack_q6_k_panel::<NR>(
                b_q6k,
                &mut packed_b[..kc_len * NR],
                kc_start,
                kc_end,
                j,
                n,
            );
            micro_kernel_packed::<MR, NR>(
                a,
                &packed_b[..kc_len * NR],
                &mut acc,
                i,
                kc_start,
                kc_end,
                k,
            );
        }
        for (ii, acc_row) in acc.iter().enumerate() {
            let off = (i + ii) * n + j;
            unsafe { std::ptr::copy_nonoverlapping(acc_row.as_ptr(), out_ptr.add(off), NR) };
        }
    }

    // Remainder columns — dequant one element at a time, accumulate scalar.
    if n_rem > 0 {
        let j = n_tiles * NR;
        let mut b_val = 0.0f32;
        for jj in 0..n_rem {
            let mut acc = [0.0f32; MR];
            for kc_start in (0..k).step_by(KC) {
                let kc_end = (kc_start + KC).min(k);
                for p in kc_start..kc_end {
                    dequant_q6_k_row_segment(
                        b_q6k,
                        std::slice::from_mut(&mut b_val),
                        p,
                        j + jj,
                        1,
                        n,
                    );
                    for (ii, a_acc) in acc.iter_mut().enumerate() {
                        *a_acc += a[(i + ii) * k + p] * b_val;
                    }
                }
            }
            for (ii, &a_acc) in acc.iter().enumerate() {
                unsafe { *out_ptr.add((i + ii) * n + j + jj) = a_acc };
            }
        }
    }
}

/// Process remainder rows (< MR): dequant one Q6_K B row at a time, scalar accumulate.
fn dequant_q6_k_remainder_rows(
    a: &[f32],
    b_q6k: &[u8],
    out: &mut [f32],
    m_start: usize,
    m_rem: usize,
    k: usize,
    n: usize,
) {
    let mut b_row = vec![0.0f32; n];
    for ii in 0..m_rem {
        let row = m_start + ii;
        for kc_start in (0..k).step_by(KC) {
            let kc_end = (kc_start + KC).min(k);
            for p in kc_start..kc_end {
                let a_val = a[row * k + p];
                dequant_q6_k_row_segment(b_q6k, &mut b_row, p, 0, n, n);
                let o_row = &mut out[row * n..(row + 1) * n];
                for j in 0..n {
                    o_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}

/// Fused Q6_K dequantize-matmul: C[m,n] += A[m,k] × dequant(B_q6k[k,n]).
///
/// Same tiling structure as `matmul_dequant_q4_0` (KC=256, MR=4, NR=8) but
/// dequantizes Q6_K super-blocks (210 bytes → 256 values) on the fly.
/// Never materializes the full K×N f32 weight matrix.
///
/// Requires: `n` is a multiple of 256, `k * n / 256 * 210 == b_q6k.len()`.
pub(crate) fn matmul_dequant_q6_k(
    a: &[f32],
    b_q6k: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    const MR: usize = 4;
    const NR: usize = 8;

    let m_tiles = m / MR;
    let n_tiles = n / NR;
    let m_rem = m % MR;
    let n_rem = n % NR;

    #[cfg(feature = "parallel")]
    if m_tiles >= PAR_M_TILE_THRESHOLD {
        use rayon::prelude::*;
        let out_ptr = SendPtr(out.as_mut_ptr());
        let n_threads = rayon::current_num_threads();
        let duty = m_tiles.div_ceil(n_threads);
        (0..m_tiles)
            .into_par_iter()
            .with_min_len(duty)
            .for_each(|it| {
                let ptr = out_ptr;
                dequant_q6_k_m_strip(
                    a,
                    b_q6k,
                    ptr.0,
                    MatmulRemainderLayout::new(it * MR, k, n, n_tiles, n_rem),
                );
            });
        if m_rem > 0 {
            dequant_q6_k_remainder_rows(a, b_q6k, out, m_tiles * MR, m_rem, k, n);
        }
        return;
    }

    let out_ptr = out.as_mut_ptr();
    for it in 0..m_tiles {
        dequant_q6_k_m_strip(
            a,
            b_q6k,
            out_ptr,
            MatmulRemainderLayout::new(it * MR, k, n, n_tiles, n_rem),
        );
    }
    if m_rem > 0 {
        dequant_q6_k_remainder_rows(a, b_q6k, out, m_tiles * MR, m_rem, k, n);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::matmul::matmul_k_outer;
    use super::*;

    /// Minimal f32→f16 conversion for test data encoding.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3FF;
        if exp <= 0 {
            sign as u16 // flush to zero
        } else if exp >= 31 {
            (sign | 0x7C00) as u16 // infinity
        } else {
            (sign | ((exp as u32) << 10) | mant) as u16
        }
    }

    /// Encode f32 weights into Q4_0 format (18-byte blocks of 32 values each).
    fn encode_q4_0(weights: &[f32], k: usize, n: usize) -> Vec<u8> {
        assert_eq!(weights.len(), k * n);
        assert_eq!(n % Q4_0_BLOCK_VALUES, 0, "n must be a multiple of 32");
        let blocks_per_row = n / Q4_0_BLOCK_VALUES;
        let mut out = vec![0u8; k * blocks_per_row * Q4_0_BLOCK_BYTES];

        for row in 0..k {
            for bc in 0..blocks_per_row {
                let start = row * n + bc * Q4_0_BLOCK_VALUES;
                let vals = &weights[start..start + Q4_0_BLOCK_VALUES];

                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };

                let block_off = (row * blocks_per_row + bc) * Q4_0_BLOCK_BYTES;
                let scale_bits = f32_to_f16_bits(scale);
                out[block_off] = scale_bits as u8;
                out[block_off + 1] = (scale_bits >> 8) as u8;

                for i in 0..16 {
                    let lo_q = ((vals[i] / scale).round() as i8).clamp(-8, 7) + 8;
                    let hi_q = ((vals[16 + i] / scale).round() as i8).clamp(-8, 7) + 8;
                    out[block_off + 2 + i] = (lo_q as u8) | ((hi_q as u8) << 4);
                }
            }
        }
        out
    }

    /// Fused Q4_0 dequant-matmul must match dequant-then-matmul (bit-exact).
    #[test]
    fn dequant_matmul_q4_0_matches_reference() {
        let m = 5; // m_tiles=1, m_rem=1
        let k = 64;
        let n = 64;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 5) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            assert_eq!(
                r.to_bits(),
                f.to_bits(),
                "mismatch at [{idx}]: ref={r}, fused={f}"
            );
        }
    }

    /// m=1 decode path — only remainder rows, no tiled body.
    #[test]
    fn dequant_matmul_q4_0_m1_decode() {
        let m = 1;
        let k = 128;
        let n = 64;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 11 + 2) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            let diff = (r - f).abs();
            // Q4_0 dequant introduces rounding; vecmat_mul vs m_remainder_tiled
            // differ in FP accumulation order, so allow small relative error.
            let tol = r.abs().max(1e-5) * 2e-3;
            assert!(
                diff <= tol,
                "m=1 mismatch at [{idx}]: ref={r}, fused={f}, diff={diff}"
            );
        }
    }

    /// Large prefill — exercises parallel path (m_tiles >= 8).
    #[test]
    fn dequant_matmul_q4_0_large_prefill() {
        let m = 32;
        let k = 256;
        let n = 256;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 1) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_q4 = encode_q4_0(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q4_0(&b_q4);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q4_0(&a, &b_q4, &mut fused_out, m, k, n);

        let max_err = ref_out
            .iter()
            .zip(fused_out.iter())
            .map(|(&r, &f)| (r - f).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err == 0.0,
            "large prefill max absolute error: {max_err}"
        );
    }

    // ── Q6_K fused dequant-matmul tests ──────────────────────────────

    /// Encode f32 weights into Q6_K format (210-byte super-blocks of 256 values each).
    /// This is a simplified encoder for testing — it quantizes each value to 6-bit
    /// signed integers using a simple abs-max scaling per super-block.
    fn encode_q6_k(weights: &[f32], k: usize, n: usize) -> Vec<u8> {
        assert_eq!(weights.len(), k * n);
        assert_eq!(n % Q6_K_BLOCK_VALUES, 0, "n must be a multiple of 256");
        let blocks_per_row = n / Q6_K_BLOCK_VALUES;
        let mut out = vec![0u8; k * blocks_per_row * Q6_K_BLOCK_BYTES];

        for row in 0..k {
            for bc in 0..blocks_per_row {
                let start = row * n + bc * Q6_K_BLOCK_VALUES;
                let vals = &weights[start..start + Q6_K_BLOCK_VALUES];
                let block_off = (row * blocks_per_row + bc) * Q6_K_BLOCK_BYTES;

                // Find abs max for the super-block scale `d`.
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d = if max_abs == 0.0 { 1.0 } else { max_abs / 31.0 };

                // For simplicity, use a uniform per-group scale of 1 (sc[i] = 1 as i8).
                // This means each value is quantized as: round(val / d), clamped to -32..31.
                let block = &mut out[block_off..block_off + Q6_K_BLOCK_BYTES];

                // Zero the block first.
                for b in block.iter_mut() {
                    *b = 0;
                }

                // Set all group scales to 1 (as signed i8).
                for b in &mut block[192..208] {
                    *b = 1u8;
                }

                // Encode d as f16.
                let d_bits = f32_to_f16_bits(d);
                block[208] = d_bits as u8;
                block[209] = (d_bits >> 8) as u8;

                // Encode each value. Match the decoding layout exactly:
                // pass 0: vals[0..128], pass 1: vals[128..256]
                // Within each pass of 128: groups of 32 at offsets 0, 32, 64, 96.
                for pass in 0..2usize {
                    let ql_off = pass * 64;
                    let qh_off = 128 + pass * 32;
                    for group in 0..4usize {
                        for l in 0..32usize {
                            let pos = pass * 128 + group * 32 + l;
                            // Quantize: q = round(val / d) + 32, clamped to 0..63
                            let q_raw = (vals[pos] / d).round() as i32;
                            let q = q_raw.clamp(-32, 31);
                            let qu = (q + 32) as u8; // 0..63 (6-bit unsigned)

                            let lo4 = qu & 0xF;
                            let hi2 = (qu >> 4) & 0x3;

                            match group {
                                0 => {
                                    block[ql_off + l] |= lo4;
                                    block[qh_off + l] |= hi2;
                                }
                                1 => {
                                    block[ql_off + l + 32] |= lo4;
                                    block[qh_off + l] |= hi2 << 2;
                                }
                                2 => {
                                    block[ql_off + l] |= lo4 << 4;
                                    block[qh_off + l] |= hi2 << 4;
                                }
                                3 => {
                                    block[ql_off + l + 32] |= lo4 << 4;
                                    block[qh_off + l] |= hi2 << 6;
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Fused Q6_K dequant-matmul must match dequant-then-matmul (bit-exact).
    #[test]
    fn dequant_matmul_q6_k_matches_reference() {
        let m = 5; // m_tiles=1, m_rem=1
        let k = 64;
        let n = 256;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 5) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q6k = encode_q6_k(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q6_k(&b_q6k);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q6_k(&a, &b_q6k, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            assert_eq!(
                r.to_bits(),
                f.to_bits(),
                "mismatch at [{idx}]: ref={r}, fused={f}"
            );
        }
    }

    /// m=1 decode path — only remainder rows, no tiled body.
    #[test]
    fn dequant_matmul_q6_k_m1_decode() {
        let m = 1;
        let k = 128;
        let n = 256;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 11 + 2) % 100) as f32 / 100.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let b_q6k = encode_q6_k(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q6_k(&b_q6k);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q6_k(&a, &b_q6k, &mut fused_out, m, k, n);

        for (idx, (&r, &f)) in ref_out.iter().zip(fused_out.iter()).enumerate() {
            let diff = (r - f).abs();
            let tol = r.abs().max(1e-6) * 1e-4;
            assert!(
                diff <= tol,
                "m=1 mismatch at [{idx}]: ref={r}, fused={f}, diff={diff}"
            );
        }
    }

    /// Large prefill — exercises parallel path (m_tiles >= 8).
    #[test]
    fn dequant_matmul_q6_k_large_prefill() {
        let m = 32;
        let k = 256;
        let n = 256;

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 1) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let b_q6k = encode_q6_k(&b_f32, k, n);

        let b_dequant = super::super::cast::dequantize_q6_k(&b_q6k);
        let mut ref_out = vec![0.0f32; m * n];
        matmul_k_outer(&a, &b_dequant, &mut ref_out, m, k, n);

        let mut fused_out = vec![0.0f32; m * n];
        matmul_dequant_q6_k(&a, &b_q6k, &mut fused_out, m, k, n);

        let max_err = ref_out
            .iter()
            .zip(fused_out.iter())
            .map(|(&r, &f)| (r - f).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err == 0.0,
            "Q6_K large prefill max absolute error: {max_err}"
        );
    }
}
