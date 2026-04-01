use super::helpers::*;
use crate::error::ExecResult;

// ── Winograd weight transform cache ─────────────────────────────────────────
//
// 1-entry thread-local cache for the Winograd U matrix (transformed weights).
// Keyed by (weight data pointer, weight length, group, oc_per_group, ic_per_group).
// Eliminates redundant weight transforms across repeated inference calls
// (e.g., 20-50 diffusion steps with the same Conv2d weights).

struct WinogradCacheEntry {
    key: (usize, usize, usize, usize, usize), // (ptr, len, group, oc_pg, ic_pg)
    u_all: Vec<f32>,
}

std::thread_local! {
    static WINOGRAD_CACHE: std::cell::RefCell<Option<WinogradCacheEntry>> =
        const { std::cell::RefCell::new(None) };
}

// ── Depthwise conv2d fast path ───────────────────────────────────────────────

/// Fast path for depthwise convolutions (group == in_channels, 1 channel per group).
///
/// Avoids im2col entirely — direct nested loop over the kernel window for each
/// output position. Splits the spatial loop into interior (no bounds checks,
/// auto-vectorizable) and border (bounds-checked) regions for ~3-4× speedup.
#[allow(clippy::too_many_arguments)]
fn conv2d_depthwise(
    data: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
) -> Vec<f32> {
    let spatial_out = h_out * w_out;
    let mut out = vec![0.0f32; n * channels * spatial_out];

    // Compute interior region where ALL kernel elements land within input bounds.
    // For oh in [oh_safe_start, oh_safe_end): all fh in [0, kh) give valid ih.
    //   ih = oh*sh + fh*dh - ph  must be in [0, h_in)
    //   fh=0:      oh*sh >= ph           → oh >= ceil(ph / sh)
    //   fh=kh-1:   oh*sh + (kh-1)*dh < h_in + ph  → oh < (h_in + ph - (kh-1)*dh - 1) / sh + 1
    let oh_safe_start = if sh > 0 { ph.div_ceil(sh) } else { 0 };
    let oh_safe_end = if sh > 0 && h_in + ph > (kh - 1) * dh {
        (h_in + ph - (kh - 1) * dh - 1) / sh + 1
    } else {
        0
    }
    .min(h_out);
    let ow_safe_start = if sw > 0 { pw.div_ceil(sw) } else { 0 };
    let ow_safe_end = if sw > 0 && w_in + pw > (kw - 1) * dw {
        (w_in + pw - (kw - 1) * dw - 1) / sw + 1
    } else {
        0
    }
    .min(w_out);

    for batch in 0..n {
        for c in 0..channels {
            let bias_val = bias.map_or(0.0, |b| b.get(c).copied().unwrap_or(0.0));
            let w_base = c * kh * kw;
            let d_base = (batch * channels + c) * h_in * w_in;
            let o_base = (batch * channels + c) * spatial_out;

            // ── Interior region: no bounds checks needed ──────────────────
            // All kernel positions are guaranteed in-bounds. The branch-free
            // inner loop enables LLVM auto-vectorization.
            if oh_safe_start < oh_safe_end && ow_safe_start < ow_safe_end {
                for oh in oh_safe_start..oh_safe_end {
                    for ow in ow_safe_start..ow_safe_end {
                        let mut sum = bias_val;
                        for fh in 0..kh {
                            let ih_actual = oh * sh + fh * dh - ph;
                            let row_base = d_base + ih_actual * w_in;
                            let w_row = w_base + fh * kw;
                            for fw in 0..kw {
                                let iw_actual = ow * sw + fw * dw - pw;
                                sum += data[row_base + iw_actual] * weight[w_row + fw];
                            }
                        }
                        out[o_base + oh * w_out + ow] = sum;
                    }
                }
            }

            // ── Border regions: bounds-checked ────────────────────────────
            for oh in 0..h_out {
                // Skip interior rows (already processed).
                let in_h_interior = oh >= oh_safe_start && oh < oh_safe_end;

                for ow in 0..w_out {
                    // Skip fully interior pixels.
                    if in_h_interior && ow >= ow_safe_start && ow < ow_safe_end {
                        continue;
                    }
                    let mut sum = bias_val;
                    for fh in 0..kh {
                        let ih = oh * sh + fh * dh;
                        if ih < ph || ih >= h_in + ph {
                            continue;
                        }
                        let ih_actual = ih - ph;
                        for fw in 0..kw {
                            let iw = ow * sw + fw * dw;
                            if iw < pw || iw >= w_in + pw {
                                continue;
                            }
                            let iw_actual = iw - pw;
                            let d_idx = d_base + ih_actual * w_in + iw_actual;
                            let w_idx = w_base + fh * kw + fw;
                            if d_idx < data.len() && w_idx < weight.len() {
                                sum += data[d_idx] * weight[w_idx];
                            }
                        }
                    }
                    out[o_base + oh * w_out + ow] = sum;
                }
            }
        }
    }
    out
}

// ── Winograd weight transform ────────────────────────────────────────────────

/// Compute the Winograd F(2,3) weight transform: U = G × g × G^T for all groups.
/// Returns u_all[group * 16 * oc_per_group * ic_per_group].
#[allow(clippy::identity_op, clippy::erasing_op)]
fn compute_winograd_weight_transform(
    weight: &[f32],
    group: usize,
    oc_per_group: usize,
    ic_per_group: usize,
) -> Vec<f32> {
    let mut u_all = vec![0.0f32; group * 16 * oc_per_group * ic_per_group];

    for g_idx in 0..group {
        for oc_idx in 0..oc_per_group {
            for ic_idx in 0..ic_per_group {
                let abs_oc = g_idx * oc_per_group + oc_idx;
                let w_base = abs_oc * ic_per_group * 9 + ic_idx * 9;

                let mut g_k = [0.0f32; 9];
                for (i, gv) in g_k.iter_mut().enumerate() {
                    let idx = w_base + i;
                    *gv = if idx < weight.len() { weight[idx] } else { 0.0 };
                }

                // G × g (4×3 × 3×3 → 4×3).
                let mut gg = [0.0f32; 12];
                for col in 0..3 {
                    gg[0 * 3 + col] = g_k[0 * 3 + col];
                    gg[1 * 3 + col] =
                        0.5 * (g_k[0 * 3 + col] + g_k[1 * 3 + col] + g_k[2 * 3 + col]);
                    gg[2 * 3 + col] =
                        0.5 * (g_k[0 * 3 + col] - g_k[1 * 3 + col] + g_k[2 * 3 + col]);
                    gg[3 * 3 + col] = g_k[2 * 3 + col];
                }

                // (G × g) × G^T (4×3 × 3×4 → 4×4).
                let mut u = [0.0f32; 16];
                for row in 0..4 {
                    u[row * 4 + 0] = gg[row * 3 + 0];
                    u[row * 4 + 1] = 0.5 * (gg[row * 3 + 0] + gg[row * 3 + 1] + gg[row * 3 + 2]);
                    u[row * 4 + 2] = 0.5 * (gg[row * 3 + 0] - gg[row * 3 + 1] + gg[row * 3 + 2]);
                    u[row * 4 + 3] = gg[row * 3 + 2];
                }

                let u_group_base = g_idx * 16 * oc_per_group * ic_per_group;
                for e in 0..16 {
                    u_all[u_group_base
                        + e * oc_per_group * ic_per_group
                        + oc_idx * ic_per_group
                        + ic_idx] = u[e];
                }
            }
        }
    }
    u_all
}

// ── Winograd F(2,3) for 3×3 stride=1 convolutions ────────────────────────��──

/// Winograd F(2,3) convolution for 3×3 kernels with stride=1, dilation=1.
///
/// Reduces multiplications from 9 to 4 per 2×2 output tile (2.25× theoretical).
/// The algorithm transforms weights and input tiles into a domain where the
/// convolution becomes element-wise multiplication, then transforms back.
///
/// Transform matrices for F(2,3):
///   G  (weight, 4×3):  [[1,0,0],[½,½,½],[½,-½,½],[0,0,1]]
///   B^T (input, 4×4):  [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
///   A^T (output, 2×4): [[1,1,1,0],[0,1,-1,-1]]
#[allow(
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::manual_div_ceil
)]
fn conv2d_winograd_f23(
    data: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    ic: usize,
    h_in: usize,
    w_in: usize,
    oc: usize,
    h_out: usize,
    w_out: usize,
    group: usize,
) -> Vec<f32> {
    let oc_per_group = oc / group.max(1);
    let ic_per_group = ic / group.max(1);
    let spatial_out = h_out * w_out;

    // Tile dimensions: 4×4 input → 2×2 output.
    let tiles_h = (h_out + 1) / 2;
    let tiles_w = (w_out + 1) / 2;
    let n_tiles = tiles_h * tiles_w;

    let mut out = vec![0.0f32; n * oc * spatial_out];

    // ── Step 1: Transform weights (cached) ──────────────────────────────
    // U[e][oc_per_group][ic_per_group] for each of 16 Winograd elements.
    // Weight transform: U = G × g × G^T. Cached per (weight pointer, length)
    // to skip redundant transforms across repeated inference calls.
    let cache_key = (
        weight.as_ptr() as usize,
        weight.len(),
        group,
        oc_per_group,
        ic_per_group,
    );
    let u_all = WINOGRAD_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(ref entry) = *cache {
            if entry.key == cache_key {
                return entry.u_all.clone();
            }
        }
        let u = compute_winograd_weight_transform(weight, group, oc_per_group, ic_per_group);
        *cache = Some(WinogradCacheEntry {
            key: cache_key,
            u_all: u.clone(),
        });
        u
    });

    // ── Per-batch, per-group processing ────────────────────────────────
    // Allocate tile workspace once and reuse.
    let mut v_buf = vec![0.0f32; 16 * ic_per_group * n_tiles];
    let mut m_buf = vec![0.0f32; 16 * oc_per_group * n_tiles];

    for batch in 0..n {
        for g_idx in 0..group {
            // ── Step 2: Transform input tiles ──────────────────────────
            // V[e][ic_per_group][n_tiles] — transform each 4×4 input tile.
            // B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
            v_buf.fill(0.0);

            for ic_idx in 0..ic_per_group {
                let abs_ic = g_idx * ic_per_group + ic_idx;
                let d_base = (batch * ic + abs_ic) * h_in * w_in;

                for th in 0..tiles_h {
                    for tw in 0..tiles_w {
                        let tile_idx = th * tiles_w + tw;
                        let ih_start = th * 2;
                        let iw_start = tw * 2;

                        // Read 4×4 input tile with padding (pad=1 implicit).
                        let mut d = [0.0f32; 16]; // 4×4
                        for di in 0..4u32 {
                            for dj in 0..4u32 {
                                let ih = ih_start as i32 + di as i32 - 1; // pad=1
                                let iw = iw_start as i32 + dj as i32 - 1;
                                d[(di * 4 + dj) as usize] =
                                    if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                                        let idx = d_base + ih as usize * w_in + iw as usize;
                                        if idx < data.len() {
                                            data[idx]
                                        } else {
                                            0.0
                                        }
                                    } else {
                                        0.0
                                    };
                            }
                        }

                        // Compute B^T × d × B.
                        // B^T × d (4×4 × 4×4 → 4×4):
                        // Row 0: d[0,j] - d[2,j]
                        // Row 1: d[1,j] + d[2,j]
                        // Row 2: -d[1,j] + d[2,j]
                        // Row 3: d[1,j] - d[3,j]
                        let mut btd = [0.0f32; 16];
                        for j in 0..4 {
                            btd[0 * 4 + j] = d[0 * 4 + j] - d[2 * 4 + j];
                            btd[1 * 4 + j] = d[1 * 4 + j] + d[2 * 4 + j];
                            btd[2 * 4 + j] = -d[1 * 4 + j] + d[2 * 4 + j];
                            btd[3 * 4 + j] = d[1 * 4 + j] - d[3 * 4 + j];
                        }
                        // (B^T × d) × B: same column transform.
                        // Col 0: btd[i,0] - btd[i,2]
                        // Col 1: btd[i,1] + btd[i,2]
                        // Col 2: -btd[i,1] + btd[i,2]
                        // Col 3: btd[i,1] - btd[i,3]
                        let mut v = [0.0f32; 16];
                        for i in 0..4 {
                            v[i * 4 + 0] = btd[i * 4 + 0] - btd[i * 4 + 2];
                            v[i * 4 + 1] = btd[i * 4 + 1] + btd[i * 4 + 2];
                            v[i * 4 + 2] = -btd[i * 4 + 1] + btd[i * 4 + 2];
                            v[i * 4 + 3] = btd[i * 4 + 1] - btd[i * 4 + 3];
                        }

                        // Store in V[e][ic_idx][tile_idx].
                        for e in 0..16 {
                            v_buf[e * ic_per_group * n_tiles + ic_idx * n_tiles + tile_idx] = v[e];
                        }
                    }
                }
            }

            // ── Step 3: Batched GEMM in Winograd domain ────────────────
            // For each of 16 elements:
            //   M[e] = U[e] × V[e]
            //   U[e]: [oc_per_group, ic_per_group]
            //   V[e]: [ic_per_group, n_tiles]
            //   M[e]: [oc_per_group, n_tiles]
            //
            // The 16 GEMMs are independent (disjoint slices of u_all, v_buf, m_buf).
            // Parallelize with rayon when each GEMM is large enough to justify it.
            let u_group_base = g_idx * 16 * oc_per_group * ic_per_group;
            let gemm_size = oc_per_group * n_tiles;

            let do_one_gemm = |e: usize, m_slice: &mut [f32]| {
                let u_slice = &u_all[u_group_base + e * oc_per_group * ic_per_group
                    ..u_group_base + (e + 1) * oc_per_group * ic_per_group];
                let v_slice = &v_buf[e * ic_per_group * n_tiles..(e + 1) * ic_per_group * n_tiles];
                m_slice.fill(0.0);

                #[cfg(all(feature = "accelerate", target_os = "macos"))]
                {
                    super::matmul::blas::sgemm_full(
                        super::matmul::GemmParams {
                            m: oc_per_group,
                            n: n_tiles,
                            k: ic_per_group,
                            alpha: 1.0,
                            beta: 0.0,
                            trans_a: false,
                            trans_b: false,
                        },
                        u_slice,
                        v_slice,
                        m_slice,
                    );
                }
                #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                {
                    super::matmul::matmul_k_outer(
                        u_slice,
                        v_slice,
                        m_slice,
                        oc_per_group,
                        ic_per_group,
                        n_tiles,
                    );
                }
            };

            // Parallel path: 16 independent GEMMs via par_chunks_mut.
            // Gate: only parallelize when each GEMM is substantial (>= 1024 output elements).
            #[cfg(feature = "parallel")]
            if gemm_size >= 1024 {
                use rayon::prelude::*;
                m_buf
                    .par_chunks_mut(gemm_size)
                    .enumerate()
                    .for_each(|(e, m_slice)| do_one_gemm(e, m_slice));
            } else {
                for e in 0..16 {
                    do_one_gemm(e, &mut m_buf[e * gemm_size..(e + 1) * gemm_size]);
                }
            }

            #[cfg(not(feature = "parallel"))]
            for e in 0..16 {
                do_one_gemm(e, &mut m_buf[e * gemm_size..(e + 1) * gemm_size]);
            }

            // ── Step 4: Output transform + scatter ─────────────────────
            // For each output channel and tile, transform M back to spatial:
            // A^T = [[1,1,1,0],[0,1,-1,-1]]
            // Y = A^T × M_tile × A → 2×2 output
            let o_base = batch * oc * spatial_out + g_idx * oc_per_group * spatial_out;

            for oc_idx in 0..oc_per_group {
                let abs_oc = g_idx * oc_per_group + oc_idx;
                let bias_val = bias.map_or(0.0, |b| b.get(abs_oc).copied().unwrap_or(0.0));

                for th in 0..tiles_h {
                    for tw in 0..tiles_w {
                        let tile_idx = th * tiles_w + tw;

                        // Gather M[0..16] for this (oc_idx, tile_idx).
                        let mut m = [0.0f32; 16]; // 4×4
                        for e in 0..16 {
                            m[e] = m_buf[e * oc_per_group * n_tiles + oc_idx * n_tiles + tile_idx];
                        }

                        // A^T × M (2×4 × 4×4 → 2×4):
                        // Row 0: m[0,j] + m[1,j] + m[2,j]
                        // Row 1: m[1,j] - m[2,j] - m[3,j]
                        let mut atm = [0.0f32; 8]; // 2×4
                        for j in 0..4 {
                            atm[0 * 4 + j] = m[0 * 4 + j] + m[1 * 4 + j] + m[2 * 4 + j];
                            atm[1 * 4 + j] = m[1 * 4 + j] - m[2 * 4 + j] - m[3 * 4 + j];
                        }

                        // (A^T × M) × A (2×4 × 4×2 → 2×2):
                        // Col 0: atm[i,0] + atm[i,1] + atm[i,2]
                        // Col 1: atm[i,1] - atm[i,2] - atm[i,3]
                        let y00 = atm[0] + atm[1] + atm[2] + bias_val;
                        let y01 = atm[1] - atm[2] - atm[3] + bias_val;
                        let y10 = atm[4] + atm[5] + atm[6] + bias_val;
                        let y11 = atm[5] - atm[6] - atm[7] + bias_val;

                        // Scatter 2×2 tile to output.
                        let oh0 = th * 2;
                        let ow0 = tw * 2;
                        let o_ch = o_base + oc_idx * spatial_out;

                        if oh0 < h_out && ow0 < w_out {
                            out[o_ch + oh0 * w_out + ow0] = y00;
                        }
                        if oh0 < h_out && ow0 + 1 < w_out {
                            out[o_ch + oh0 * w_out + ow0 + 1] = y01;
                        }
                        if oh0 + 1 < h_out && ow0 < w_out {
                            out[o_ch + (oh0 + 1) * w_out + ow0] = y10;
                        }
                        if oh0 + 1 < h_out && ow0 + 1 < w_out {
                            out[o_ch + (oh0 + 1) * w_out + ow0 + 1] = y11;
                        }
                    }
                }
            }
        }
    }

    out
}

// ── im2col + GEMM conv2d core ────────────────────────────────────────────────

/// Core conv2d using im2col + GEMM pattern.
///
/// 1. im2col: gather input patches into a column matrix [kernel_size × spatial_out]
/// 2. GEMM: weight[oc_per_group, kernel_size] × col[kernel_size, spatial_out] → out
///
/// The GEMM phase uses BLAS sgemm when available (Accelerate on macOS),
/// falling back to the parallel tiled matmul kernel otherwise.
#[allow(clippy::too_many_arguments)]
fn conv2d_core(
    data: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    ic_per_group: usize,
    h_in: usize,
    w_in: usize,
    oc: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> Vec<f32> {
    let oc_per_group = oc / group.max(1);
    let kernel_size = ic_per_group * kh * kw;
    let spatial_out = h_out * w_out;
    let ic = ic_per_group * group;

    // Depthwise fast path: group == channels, 1 input channel per group.
    // Direct loop avoids im2col overhead for single-channel inner products.
    if ic_per_group == 1 && oc_per_group == 1 {
        return conv2d_depthwise(
            data, weight, bias, n, ic, h_in, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw,
        );
    }

    // Winograd F(2,3) fast path: 3×3 kernel, stride=1, dilation=1, sufficient channels.
    // Reduces multiplications by 2.25× — the dominant 3×3 conv case in UNet/VAE.
    if kh == 3
        && kw == 3
        && sh == 1
        && sw == 1
        && dh == 1
        && dw == 1
        && ph == 1
        && pw == 1
        && ic_per_group >= 16
    {
        return conv2d_winograd_f23(
            data, weight, bias, n, ic, h_in, w_in, oc, h_out, w_out, group,
        );
    }

    let mut out = vec![0.0f32; n * oc * spatial_out];

    // Tiled im2col: bound the col buffer to at most TILE_CAP floats.
    // Tiled im2col: bound the col buffer to at most TILE_CAP floats.
    const TILE_CAP: usize = 4 * 1024 * 1024; // 16 MB as f32
    let tile_size = if kernel_size > 0 {
        (TILE_CAP / kernel_size).max(1).min(spatial_out)
    } else {
        spatial_out
    };
    let mut col = vec![0.0f32; kernel_size * tile_size];
    // Pre-allocate tile buffers once — reused across all tiles to avoid per-tile allocation.
    let mut tile_out = vec![0.0f32; oc_per_group * tile_size];
    // For LUT-GEMM: col_t (transposed im2col) and lut_out (GEMM result).
    // These need to be separate buffers since lut_gemm writes to output while reading input.
    let mut col_t_buf = vec![0.0f32; tile_size * kernel_size];
    let mut lut_out_buf = vec![0.0f32; tile_size * oc_per_group];

    for batch in 0..n {
        for g in 0..group {
            let w_start = g * oc_per_group * kernel_size;
            let w_end = (w_start + oc_per_group * kernel_size).min(weight.len());
            let w_slice = &weight[w_start..w_end];

            // LUT-GEMM Q4 path for non-BLAS platforms (WASM, Linux without MKL).
            // On macOS with Accelerate, BLAS sgemm is faster — skip quantization.
            // Transpose W, quantize once per group, reuse across all spatial tiles.
            #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
            let qw = if group <= 1 && oc_per_group >= 64 && kernel_size >= 16 {
                let mut w_t = vec![0.0f32; oc_per_group * kernel_size];
                for oc_idx in 0..oc_per_group {
                    for k in 0..kernel_size {
                        w_t[k * oc_per_group + oc_idx] = w_slice[oc_idx * kernel_size + k];
                    }
                }
                Some(crate::lut_gemm::quantize::quantize_4bit(
                    &w_t,
                    kernel_size as u32,
                    oc_per_group as u32,
                ))
            } else {
                None
            };
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            let qw: Option<crate::lut_gemm::quantize::QuantizedWeights4> = None;

            let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;

            // Initialize output with bias.
            if let Some(b) = bias {
                for oc_idx in 0..oc_per_group {
                    let abs_oc = g * oc_per_group + oc_idx;
                    let bias_val = b.get(abs_oc).copied().unwrap_or(0.0);
                    if bias_val != 0.0 {
                        let start = o_base + oc_idx * spatial_out;
                        for v in &mut out[start..start + spatial_out] {
                            *v = bias_val;
                        }
                    }
                }
            }

            // Process spatial dimension in tiles.
            let mut tile_start = 0;
            while tile_start < spatial_out {
                let tile_end = (tile_start + tile_size).min(spatial_out);
                let tile_len = tile_end - tile_start;

                // Phase 1: im2col for this tile — col[kernel_size, tile_len].
                // Fast path for stride=1, dilation=1: consecutive output positions
                // within a row map to consecutive input positions, enabling memcpy.
                let use_fast_im2col = sh == 1 && sw == 1 && dh == 1 && dw == 1;

                for k in 0..kernel_size {
                    let ic_idx = k / (kh * kw);
                    let k_rem = k % (kh * kw);
                    let fh = k_rem / kw;
                    let fw = k_rem % kw;
                    let abs_ic = g * ic_per_group + ic_idx;
                    let col_row = &mut col[k * tile_len..(k + 1) * tile_len];

                    if use_fast_im2col {
                        let d_channel_base = (batch * ic + abs_ic) * h_in * w_in;
                        // For each output row in this tile, compute the contiguous
                        // interior range where both h and w are in-bounds, then memcpy.
                        let mut t = 0;
                        while t < tile_len {
                            let out_pos = tile_start + t;
                            let oh = out_pos / w_out;
                            let ow_start = out_pos % w_out;
                            // How many positions remain in this output row within the tile.
                            let row_remaining = (w_out - ow_start).min(tile_len - t);

                            let ih = oh + fh;
                            if ih < ph || ih >= h_in + ph {
                                // Entire row segment is padding — zero fill.
                                col_row[t..t + row_remaining].fill(0.0);
                                t += row_remaining;
                                continue;
                            }
                            let ih_actual = ih - ph;

                            // Width range in-bounds: iw_actual = ow + fw - pw must be in [0, w_in).
                            // → ow >= pw - fw  and  ow < w_in + pw - fw
                            let ow_valid_lo = pw.saturating_sub(fw);
                            let ow_valid_hi = (w_in + pw - fw).min(w_out);
                            let ow_end = ow_start + row_remaining;

                            // Leading zeros (left padding).
                            if ow_start < ow_valid_lo {
                                let zero_end = ow_valid_lo.min(ow_end);
                                let zlen = zero_end - ow_start;
                                col_row[t..t + zlen].fill(0.0);
                                t += zlen;
                                if t >= tile_len
                                    || ow_start + (t - (tile_start + out_pos - ow_start)) >= ow_end
                                {
                                    continue;
                                }
                            }

                            // Interior: contiguous copy from data.
                            let cur_ow = ow_start + (t - (tile_start + out_pos - ow_start));
                            if cur_ow < ow_valid_hi && cur_ow < ow_end {
                                let copy_end = ow_valid_hi.min(ow_end);
                                let copy_len = copy_end - cur_ow;
                                let iw_start = cur_ow + fw - pw;
                                let src_start = d_channel_base + ih_actual * w_in + iw_start;
                                let src_end = src_start + copy_len;
                                if src_end <= data.len() {
                                    col_row[t..t + copy_len]
                                        .copy_from_slice(&data[src_start..src_end]);
                                } else {
                                    // Fallback: element-wise with bounds check.
                                    for i in 0..copy_len {
                                        let idx = src_start + i;
                                        col_row[t + i] =
                                            if idx < data.len() { data[idx] } else { 0.0 };
                                    }
                                }
                                t += copy_len;
                            }

                            // Trailing zeros (right padding).
                            let final_ow = ow_start + (t - (tile_start + out_pos - ow_start));
                            if final_ow < ow_end {
                                let zlen = ow_end - final_ow;
                                col_row[t..t + zlen].fill(0.0);
                                t += zlen;
                            }
                        }
                    } else {
                        // General path: per-element with division and bounds checks.
                        for (t, col_val) in col_row.iter_mut().enumerate() {
                            let out_pos = tile_start + t;
                            let oh = out_pos / w_out;
                            let ow = out_pos % w_out;
                            let ih = oh * sh + fh * dh;
                            let iw = ow * sw + fw * dw;

                            *col_val = if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let d_idx =
                                    ((batch * ic + abs_ic) * h_in + (ih - ph)) * w_in + (iw - pw);
                                if d_idx < data.len() {
                                    data[d_idx]
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                        }
                    }
                }

                // Phase 2: GEMM — W[oc_per_group, kernel_size] × col[kernel_size, tile_len].
                if let Some(ref qw) = qw {
                    // LUT-GEMM Q4: transpose col → col_t_buf, GEMM → lut_out_buf.
                    // Both buffers are pre-allocated, zero per-tile allocation.
                    let col_t_len = tile_len * kernel_size;
                    let lut_out_len = tile_len * oc_per_group;
                    // Transpose col[K, tile_len] → col_t_buf[tile_len, K].
                    for t in 0..tile_len {
                        for k in 0..kernel_size {
                            col_t_buf[t * kernel_size + k] = col[k * tile_len + t];
                        }
                    }
                    lut_out_buf[..lut_out_len].fill(0.0);
                    #[cfg(feature = "parallel")]
                    crate::lut_gemm::lut_gemm_4bit_par(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    #[cfg(not(feature = "parallel"))]
                    crate::lut_gemm::lut_gemm_4bit(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    // Scatter from [tile_len, oc] directly into output.
                    for t in 0..tile_len {
                        for oc_idx in 0..oc_per_group {
                            let o_pos = o_base + oc_idx * spatial_out + tile_start + t;
                            out[o_pos] += lut_out_buf[t * oc_per_group + oc_idx];
                        }
                    }
                    tile_start = tile_end;
                    continue; // Skip the f32 GEMM + scatter below.
                } else {
                    let to_len = oc_per_group * tile_len;
                    tile_out[..to_len].fill(0.0);
                    #[cfg(all(feature = "accelerate", target_os = "macos"))]
                    {
                        super::matmul::blas::sgemm_full(
                            super::matmul::GemmParams {
                                m: oc_per_group,
                                n: tile_len,
                                k: kernel_size,
                                alpha: 1.0,
                                beta: 0.0,
                                trans_a: false,
                                trans_b: false,
                            },
                            w_slice,
                            &col[..kernel_size * tile_len],
                            &mut tile_out[..to_len],
                        );
                    }

                    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
                    {
                        super::matmul::matmul_k_outer(
                            w_slice,
                            &col[..kernel_size * tile_len],
                            &mut tile_out[..to_len],
                            oc_per_group,
                            kernel_size,
                            tile_len,
                        );
                    }
                    // Scatter tile results into output (add to bias if present).
                    for oc_idx in 0..oc_per_group {
                        let o_row_start = o_base + oc_idx * spatial_out + tile_start;
                        let t_row_start = oc_idx * tile_len;
                        for t in 0..tile_len {
                            out[o_row_start + t] += tile_out[t_row_start + t];
                        }
                    }
                }

                tile_start = tile_end;
            }
        }
    }

    out
}

/// Conv2d with explicit spatial dimensions from the op fields.
///
/// All dispatch paths route through this function — no shape guessing needed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv2d_direct(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    if data.is_empty() || weight.is_empty() || h_in == 0 || w_in == 0 {
        return Ok(vec![]);
    }
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let bias = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
        Some(cast_f32(bias_bytes)?)
    } else {
        None
    };

    // Trust the passed-in h_in/w_in — these come from resolve_spatial_dims()
    // which prefers runtime TensorMeta (propagated via InOutBufWithMeta) over
    // compiled values. Only fall back to heuristic derivation when both are 0.
    let (h_in, w_in) = if h_in > 0 && w_in > 0 {
        (h_in, w_in)
    } else {
        // Last resort: derive square spatial dims from total elements.
        let total = data.len();
        let side = (total as f64).sqrt() as usize;
        if side > 0 && side * side == total {
            (side, side)
        } else {
            (1, total.max(1))
        }
    };

    // Derive N, OC, IC/group from buffer lengths + known spatial dims.
    let ic = if h_in > 0 && w_in > 0 {
        data.len() / (h_in * w_in)
    } else {
        1
    };
    let n = if ic > 0 && h_in > 0 && w_in > 0 {
        data.len() / (ic * h_in * w_in)
    } else {
        1
    };
    let oc = weight.len() / (kh * kw).max(1) / (ic / group.max(1)).max(1);
    let ic_per_group = (ic / group.max(1)).max(1);

    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let out = conv2d_core(
        &data,
        &weight,
        bias.as_deref(),
        n,
        ic_per_group,
        h_in,
        w_in,
        oc,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        group,
    );
    Ok(f32_vec_to_bytes(out))
}

/// Conv2d with explicit input shapes from shape vectors (used by KvStore path).
///
/// Delegates to `dispatch_conv2d_direct` after extracting H/W from shapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv2d_with_shapes(
    inputs: &[&[u8]],
    input_shapes: &[Vec<usize>],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
) -> ExecResult<Vec<u8>> {
    let ds = input_shapes.first().cloned().unwrap_or_default();
    if ds.len() != 4 {
        return Err(crate::error::ExecError::UnsupportedOp(format!(
            "Conv2d: expected 4D input shape, got {:?}",
            ds
        )));
    }
    let h_in = ds[2];
    let w_in = ds[3];
    dispatch_conv2d_direct(inputs, kh, kw, sh, sw, ph, pw, dh, dw, group, h_in, w_in)
}

/// Conv2d with pre-quantized 4-bit LUT-GEMM weights (compile-time quantized).
///
/// The weight quantization + transpose was done at compile time. At runtime:
/// 1. im2col: gather input patches → col[kernel_size, tile_len]
/// 2. Transpose col → col_t[tile_len, kernel_size]
/// 3. LUT-GEMM: col_t × pre_quantized_weights → [tile_len, oc_per_group]
/// 4. Scatter to output
///
/// Zero quantization/transpose overhead at runtime.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv2d_lut4(
    inputs: &[&[u8]],
    cid: hologram_graph::constant::ConstantId,
    tape_ctx: &crate::tape::TapeContext<'_>,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    // On macOS with Accelerate, BLAS sgemm is faster than LUT-GEMM Q4.
    // Fall back to the f32 BLAS path — the pre-quantized weights are still in the
    // archive for non-BLAS targets (WASM, Linux without MKL).
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        let _ = (cid, tape_ctx);
        dispatch_conv2d_direct(inputs, kh, kw, sh, sw, ph, pw, dh, dw, group, h_in, w_in)
    }

    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    {
        let data = cast_f32(inputs[0])?;
        let weight = cast_f32(inputs[1])?;
        if data.is_empty() || weight.is_empty() || h_in == 0 || w_in == 0 {
            return Ok(vec![]);
        }
        let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
        let bias = if !bias_bytes.is_empty() && bias_bytes.len() >= 4 {
            Some(cast_f32(bias_bytes)?)
        } else {
            None
        };

        // Derive N, IC, OC from spatial dims and buffer lengths.
        // data is [N, IC, H, W], weight is [OC, IC/group, KH, KW].
        let h_in = h_in.max(1);
        let w_in = w_in.max(1);
        let spatial = h_in * w_in;
        let ic = if spatial > 0 { data.len() / spatial } else { 1 };
        let n = if ic > 0 && spatial > 0 {
            data.len() / (ic * spatial)
        } else {
            1
        };
        let ic_per_group = (ic / group.max(1)).max(1);
        let kernel_size = ic_per_group * kh * kw;
        let oc = if kernel_size > 0 {
            weight.len() / kernel_size
        } else {
            1
        };
        let oc_per_group = oc / group.max(1);
        let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        let spatial_out = h_out * w_out;

        // Resolve pre-quantized weights from constant store.
        let mut cache = tape_ctx.weight_cache.write();
        let qw = cache.get_q4(cid, tape_ctx.constants, tape_ctx.weights)?;

        // Validate quantized weight dimensions match runtime-derived dimensions.
        // If mismatched, fall back to the non-quantized path.
        if qw.rows as usize != kernel_size || qw.cols as usize != oc_per_group {
            tracing::warn!(
                qw_rows = qw.rows,
                qw_cols = qw.cols,
                kernel_size,
                oc_per_group,
                "Conv2dLut4 dimension mismatch — falling back to f32 path"
            );
            drop(cache);
            return dispatch_conv2d_direct(
                inputs, kh, kw, sh, sw, ph, pw, dh, dw, group, h_in, w_in,
            );
        }

        let mut out = vec![0.0f32; n * oc * spatial_out];

        // Tiled im2col (same tile sizing as conv2d_core).
        const TILE_CAP: usize = 4 * 1024 * 1024; // 16 MB as f32
        let tile_size = if kernel_size > 0 {
            (TILE_CAP / kernel_size).max(1).min(spatial_out)
        } else {
            spatial_out
        };
        let mut col = vec![0.0f32; kernel_size * tile_size];
        // Pre-allocate transpose + output buffers — reused across all tiles.
        let mut col_t_buf = vec![0.0f32; tile_size * kernel_size];
        let mut lut_out_buf = vec![0.0f32; tile_size * oc_per_group];

        for batch in 0..n {
            for g in 0..group {
                let o_base = batch * oc * spatial_out + g * oc_per_group * spatial_out;

                // Initialize output with bias.
                if let Some(ref b) = bias {
                    for oc_idx in 0..oc_per_group {
                        let abs_oc = g * oc_per_group + oc_idx;
                        let bias_val = b.get(abs_oc).copied().unwrap_or(0.0);
                        if bias_val != 0.0 {
                            let start = o_base + oc_idx * spatial_out;
                            for v in &mut out[start..start + spatial_out] {
                                *v = bias_val;
                            }
                        }
                    }
                }

                let mut tile_start = 0;
                while tile_start < spatial_out {
                    let tile_end = (tile_start + tile_size).min(spatial_out);
                    let tile_len = tile_end - tile_start;

                    // Phase 1: im2col for this tile.
                    for k in 0..kernel_size {
                        let ic_idx = k / (kh * kw);
                        let k_rem = k % (kh * kw);
                        let fh = k_rem / kw;
                        let fw = k_rem % kw;
                        let abs_ic = g * ic_per_group + ic_idx;
                        let col_row = &mut col[k * tile_len..(k + 1) * tile_len];

                        for (t, col_val) in col_row.iter_mut().enumerate() {
                            let out_pos = tile_start + t;
                            let oh = out_pos / w_out;
                            let ow = out_pos % w_out;
                            let ih = oh * sh + fh * dh;
                            let iw = ow * sw + fw * dw;

                            *col_val = if ih >= ph && ih < h_in + ph && iw >= pw && iw < w_in + pw {
                                let d_idx =
                                    ((batch * ic + abs_ic) * h_in + (ih - ph)) * w_in + (iw - pw);
                                if d_idx < data.len() {
                                    data[d_idx]
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                        }
                    }

                    // Phase 2: Transpose col → col_t_buf and LUT-GEMM → lut_out_buf.
                    let col_t_len = tile_len * kernel_size;
                    let lut_out_len = tile_len * oc_per_group;
                    for t in 0..tile_len {
                        for k in 0..kernel_size {
                            col_t_buf[t * kernel_size + k] = col[k * tile_len + t];
                        }
                    }
                    lut_out_buf[..lut_out_len].fill(0.0);
                    #[cfg(feature = "parallel")]
                    crate::lut_gemm::lut_gemm_4bit_par(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );
                    #[cfg(not(feature = "parallel"))]
                    crate::lut_gemm::lut_gemm_4bit(
                        &col_t_buf[..col_t_len],
                        qw,
                        &mut lut_out_buf[..lut_out_len],
                    );

                    // Scatter from [tile_len, oc] to output [oc, spatial_out].
                    for t in 0..tile_len {
                        for oc_idx in 0..oc_per_group {
                            let o_pos = o_base + oc_idx * spatial_out + tile_start + t;
                            out[o_pos] += lut_out_buf[t * oc_per_group + oc_idx];
                        }
                    }

                    tile_start = tile_end;
                }
            }
        }

        Ok(f32_vec_to_bytes(out))
    } // #[cfg(not(accelerate + macos))]
}

// ── ConvTranspose ────────────────────────────────────────────────────────────

/// Transposed 2-D convolution with explicit spatial dimensions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_conv_transpose(
    inputs: &[&[u8]],
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    group: usize,
    output_pad_h: usize,
    output_pad_w: usize,
    h_in: usize,
    w_in: usize,
) -> ExecResult<Vec<u8>> {
    let data = cast_f32(inputs[0])?;
    let weight = cast_f32(inputs[1])?;
    let bias_bytes = inputs.get(2).copied().unwrap_or(&[][..]);
    let has_bias = !bias_bytes.is_empty() && bias_bytes.len() >= 4;

    let ic_actual = weight.len() / (kh * kw).max(1);
    let oc_per_group = if ic_actual > 0 {
        ic_actual / group.max(1)
    } else {
        1
    };

    let h_out = (h_in.saturating_sub(1)) * sh + dh * (kh - 1) + output_pad_h + 1 - 2 * ph;
    let w_out = (w_in.saturating_sub(1)) * sw + dw * (kw - 1) + output_pad_w + 1 - 2 * pw;
    let oc = oc_per_group * group;

    let mut out = vec![0.0f32; oc * h_out * w_out];

    // Add bias — flat loop over output elements
    if has_bias {
        if let Ok(b) = cast_f32(bias_bytes) {
            let hw = h_out * w_out;
            for (flat, out_val) in out.iter_mut().enumerate() {
                let c = flat / hw;
                *out_val = if c < b.len() { b[c] } else { 0.0 };
            }
        }
    }

    // Transposed convolution: scatter input values through the kernel.
    // Flat outer loop over (group × spatial), flat inner loop over (oc_per_group × kernel).
    let hw_in = h_in * w_in;
    for flat_in in 0..(group * hw_in) {
        let g = flat_in / hw_in;
        let spatial = flat_in % hw_in;
        let ih = spatial / w_in;
        let iw = spatial % w_in;
        let abs_ic = g; // simplified: 1 input channel per group
        let d_idx = (abs_ic * h_in + ih) * w_in + iw;
        let d_val = if d_idx < data.len() {
            data[d_idx]
        } else {
            continue;
        };
        for k_flat in 0..(oc_per_group * kh * kw) {
            let oc_idx = k_flat / (kh * kw);
            let k_rem = k_flat % (kh * kw);
            let ky = k_rem / kw;
            let kx = k_rem % kw;
            let abs_oc = g * oc_per_group + oc_idx;
            let oh = ih * sh + ky * dh;
            let ow = iw * sw + kx * dw;
            if oh >= ph && ow >= pw {
                let oh = oh - ph;
                let ow = ow - pw;
                if oh < h_out && ow < w_out {
                    let w_idx = ((abs_ic * oc_per_group + oc_idx) * kh + ky) * kw + kx;
                    let o_idx = (abs_oc * h_out + oh) * w_out + ow;
                    if w_idx < weight.len() && o_idx < out.len() {
                        out[o_idx] += d_val * weight[w_idx];
                    }
                }
            }
        }
    }

    Ok(f32_vec_to_bytes(out))
}
