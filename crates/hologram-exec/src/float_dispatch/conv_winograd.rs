use super::conv::{Conv2dAttrs, Conv2dDepthwiseCall, Conv2dInputShape};

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
#[inline(always)]
pub(super) fn conv2d_depthwise(call: Conv2dDepthwiseCall<'_>) -> Vec<f32> {
    let Conv2dDepthwiseCall {
        data,
        weight,
        bias,
        input_shape:
            Conv2dInputShape {
                n,
                channels,
                h_in,
                w_in,
            },
        h_out,
        w_out,
        attrs:
            Conv2dAttrs {
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                group: _,
            },
    } = call;
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

// ── Winograd F(2,3) for 3×3 stride=1 convolutions ──────────────────────────

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
pub(super) fn conv2d_winograd_f23(
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
