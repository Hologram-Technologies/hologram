//! CPU compute backend.
//!
//! Uses SIMD (NEON on ARM, AVX2 on x86_64) and Accelerate BLAS on macOS.
//! This is the reference implementation — all other backends must produce
//! identical results for the same inputs.

use crate::cpu_cast::{dispatch_cast, half_to_f32};
use crate::{BackendError, ComputeBackend, ComputeMemory, KernelParams, Result};
use hologram_core::op::FloatOp;

// Accelerate BLAS FFI for high-performance matmul on macOS.
#[cfg(has_accelerate)]
extern "C" {
    fn cblas_sgemm(
        order: i32,   // CblasRowMajor = 101
        trans_a: i32, // CblasNoTrans = 111
        trans_b: i32, // CblasNoTrans = 111
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// CPU memory: buffers are `Vec<u8>` in main memory.
pub struct CpuMemory;

impl ComputeMemory for CpuMemory {
    type Buffer = Vec<u8>;

    fn alloc(&self, byte_len: usize) -> Vec<u8> {
        vec![0u8; byte_len]
    }

    fn upload(&self, data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    fn download(&self, buf: &Vec<u8>) -> Vec<u8> {
        buf.clone()
    }

    fn alias(&self, buf: &Vec<u8>) -> Vec<u8> {
        buf.clone()
    }

    fn byte_len(&self, buf: &Vec<u8>) -> usize {
        buf.len()
    }

    fn mmap(&self, data: &[u8]) -> Option<Vec<u8>> {
        // CPU mmap: return a copy. True mmap (backed by file pages)
        // will be added when integrating with the archive loader.
        Some(data.to_vec())
    }

    fn evict(&self, buf: &mut Vec<u8>) {
        // Drop the allocation and shrink to zero.
        *buf = Vec::new();
    }
}

/// CPU compute backend: dispatches ops using SIMD + BLAS.
pub struct CpuBackend {
    /// Ring LUT tables (loaded at initialization).
    ring_tables: Vec<[u8; 256]>,
}

impl CpuBackend {
    /// Create a new CPU backend.
    pub fn new() -> Self {
        Self {
            ring_tables: Vec::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend<CpuMemory> for CpuBackend {
    fn dispatch(
        &self,
        op: &FloatOp,
        inputs: &[&Vec<u8>],
        output: &mut Vec<u8>,
        _params: &KernelParams,
    ) -> Result<usize> {
        use hologram_core::op::OpCategory;

        match op.category() {
            OpCategory::UnaryElementwise => {
                let input = inputs
                    .first()
                    .ok_or_else(|| BackendError::Shape("unary op requires 1 input".into()))?;
                let in_floats: &[f32] = bytemuck::cast_slice(input);
                let mut out_floats = vec![0.0f32; in_floats.len()];
                for (o, &v) in out_floats.iter_mut().zip(in_floats) {
                    *o = op.apply_unary(v);
                }
                *output = bytemuck::cast_slice(&out_floats).to_vec();
                Ok(output.len())
            }
            OpCategory::BinaryElementwise => {
                if inputs.len() < 2 {
                    return Err(BackendError::Shape("binary op requires 2 inputs".into()));
                }
                let a: &[f32] = bytemuck::cast_slice(inputs[0]);
                let b: &[f32] = bytemuck::cast_slice(inputs[1]);
                let n = a.len().max(b.len());
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let va = a[i % a.len()];
                    let vb = b[i % b.len()];
                    out[i] = op.apply_binary(va, vb);
                }
                *output = bytemuck::cast_slice(&out).to_vec();
                Ok(output.len())
            }
            OpCategory::BinaryCompare => {
                if inputs.len() < 2 {
                    return Err(BackendError::Shape("compare op requires 2 inputs".into()));
                }
                let a: &[f32] = bytemuck::cast_slice(inputs[0]);
                let b: &[f32] = bytemuck::cast_slice(inputs[1]);
                let n = a.len().max(b.len());
                let mut out = vec![0u8; n];
                for i in 0..n {
                    let va = a[i % a.len()];
                    let vb = b[i % b.len()];
                    out[i] = u8::from(op.apply_compare(va, vb));
                }
                *output = out;
                Ok(output.len())
            }
            OpCategory::BinaryByteBool => {
                if inputs.len() < 2 {
                    return Err(BackendError::Shape("byte bool op requires 2 inputs".into()));
                }
                let a = inputs[0].as_slice();
                let b = inputs[1].as_slice();
                let n = a.len().max(b.len());
                let mut out = vec![0u8; n];
                for i in 0..n {
                    let va = a[i % a.len()];
                    let vb = b[i % b.len()];
                    out[i] = op.apply_byte_bool(va, vb);
                }
                *output = out;
                Ok(output.len())
            }
            OpCategory::UnaryByteBool => {
                // NOT: return 1 if input==0, else 0
                let input = inputs.first().ok_or_else(|| {
                    BackendError::Shape("unary byte bool requires 1 input".into())
                })?;
                let mut out = vec![0u8; input.len()];
                for (o, &v) in out.iter_mut().zip(input.iter()) {
                    *o = if v == 0 { 1 } else { 0 };
                }
                *output = out;
                Ok(output.len())
            }
            OpCategory::UnaryToU8 => {
                // IsNaN: interpret as f32, return 1u8 if NaN, else 0u8
                let input = inputs
                    .first()
                    .ok_or_else(|| BackendError::Shape("unary-to-u8 requires 1 input".into()))?;
                let in_f: &[f32] = bytemuck::cast_slice(input);
                let mut out = vec![0u8; in_f.len()];
                for (o, &v) in out.iter_mut().zip(in_f) {
                    *o = u8::from(v.is_nan());
                }
                *output = out;
                Ok(output.len())
            }
            OpCategory::Custom => {
                // MatMul: Accelerate BLAS on macOS, naive loop elsewhere.
                if let FloatOp::MatMul { m, k, n } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape("matmul requires 2 inputs".into()));
                    }
                    let m = *m as usize;
                    let k = *k as usize;
                    let n = *n as usize;
                    let a: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let b: &[f32] = bytemuck::cast_slice(inputs[1]);

                    // Resolve actual M from input size if baked M is 0 (variable-length).
                    let actual_m = if m > 0 {
                        m
                    } else {
                        a.len().checked_div(k).unwrap_or(0)
                    };
                    if actual_m == 0 || k == 0 || n == 0 {
                        return Err(BackendError::Shape(format!(
                            "matmul dims invalid: m={actual_m} k={k} n={n}"
                        )));
                    }

                    let mut out = vec![0.0f32; actual_m * n];
                    #[cfg(has_accelerate)]
                    {
                        // Accelerate BLAS sgemm — uses Apple's optimized AMX/NEON kernels.
                        unsafe {
                            cblas_sgemm(
                                101, // CblasRowMajor
                                111, // CblasNoTrans
                                111, // CblasNoTrans
                                actual_m as i32,
                                n as i32,
                                k as i32,
                                1.0,
                                a.as_ptr(),
                                k as i32,
                                b.as_ptr(),
                                n as i32,
                                0.0,
                                out.as_mut_ptr(),
                                n as i32,
                            );
                        }
                    }
                    #[cfg(not(has_accelerate))]
                    {
                        for i in 0..actual_m {
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for p in 0..k {
                                    sum += a[i * k + p] * b[p * n + j];
                                }
                                out[i * n + j] = sum;
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Softmax.
                if let FloatOp::Softmax { size } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("softmax requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let row_size = *size as usize;
                    let mut out = in_f.to_vec();
                    if row_size > 0 {
                        for row in out.chunks_mut(row_size) {
                            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0.0f32;
                            for v in row.iter_mut() {
                                *v = (*v - max).exp();
                                sum += *v;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for v in row.iter_mut() {
                                    *v *= inv;
                                }
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // RmsNorm.
                if let FloatOp::RmsNorm { size, epsilon } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape("rmsnorm requires 2 inputs".into()));
                    }
                    let input: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let row_size = *size as usize;
                    let eps = f32::from_bits(*epsilon);
                    let mut out = vec![0.0f32; input.len()];
                    if row_size > 0 {
                        for (chunk_idx, chunk) in input.chunks(row_size).enumerate() {
                            let ms: f32 =
                                chunk.iter().map(|v| v * v).sum::<f32>() / row_size as f32;
                            let inv_rms = 1.0 / (ms + eps).sqrt();
                            let base = chunk_idx * row_size;
                            for (i, &v) in chunk.iter().enumerate() {
                                out[base + i] = v * inv_rms * weight[i % weight.len()];
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // LayerNorm.
                if let FloatOp::LayerNorm { size, epsilon } = op {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("layernorm requires 3 inputs".into()));
                    }
                    let input: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let bias: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let row_size = *size as usize;
                    let eps = f32::from_bits(*epsilon);
                    let mut out = vec![0.0f32; input.len()];
                    if row_size > 0 {
                        for (chunk_idx, chunk) in input.chunks(row_size).enumerate() {
                            let mean: f32 = chunk.iter().sum::<f32>() / row_size as f32;
                            let var: f32 =
                                chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                                    / row_size as f32;
                            let inv_std = 1.0 / (var + eps).sqrt();
                            let base = chunk_idx * row_size;
                            for (i, &v) in chunk.iter().enumerate() {
                                out[base + i] = (v - mean) * inv_std * weight[i % weight.len()]
                                    + bias[i % bias.len()];
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Transpose.
                if let FloatOp::Transpose { perm, ndim } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("transpose requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    // Use shape from params.u32s if available.
                    let n = *ndim as usize;
                    if n >= 2 && _params.u32s.len() >= n {
                        let shape: Vec<usize> =
                            _params.u32s[..n].iter().map(|&d| d as usize).collect();
                        let total: usize = shape.iter().product();
                        if total != in_f.len() {
                            return Err(BackendError::Shape(format!(
                                "transpose shape product {total} != input len {}",
                                in_f.len()
                            )));
                        }

                        // Compute strides.
                        let mut in_strides = vec![1usize; n];
                        for i in (0..n - 1).rev() {
                            in_strides[i] = in_strides[i + 1] * shape[i + 1];
                        }

                        // Output shape = permuted input shape.
                        let out_shape: Vec<usize> =
                            (0..n).map(|i| shape[perm[i] as usize]).collect();
                        let mut out_strides = vec![1usize; n];
                        for i in (0..n - 1).rev() {
                            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                        }

                        let mut out = vec![0.0f32; total];
                        #[allow(clippy::needless_range_loop)]
                        for flat in 0..total {
                            // Decompose output flat index.
                            let mut out_coord = vec![0usize; n];
                            let mut rem = flat;
                            for i in 0..n {
                                out_coord[i] = rem / out_strides[i];
                                rem %= out_strides[i];
                            }
                            // Map to input coords via inverse perm.
                            let mut in_coord = vec![0usize; n];
                            for i in 0..n {
                                in_coord[perm[i] as usize] = out_coord[i];
                            }
                            let in_flat: usize =
                                in_coord.iter().zip(&in_strides).map(|(c, s)| c * s).sum();
                            out[flat] = in_f[in_flat];
                        }
                        *output = bytemuck::cast_slice(&out).to_vec();
                        return Ok(output.len());
                    }
                }

                // Conv2d.
                if let FloatOp::Conv2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                    group,
                    input_h,
                    input_w,
                } = op
                {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "conv2d requires at least 2 inputs".into(),
                        ));
                    }
                    let data: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let bias: Option<&[f32]> =
                        inputs.get(2).map(|b| bytemuck::cast_slice(b.as_slice()));

                    let kh = *kernel_h as usize;
                    let kw = *kernel_w as usize;
                    let sh = (*stride_h).max(1) as usize;
                    let sw = (*stride_w).max(1) as usize;
                    let ph = *pad_h as usize;
                    let pw = *pad_w as usize;
                    let dh = (*dilation_h).max(1) as usize;
                    let dw = (*dilation_w).max(1) as usize;
                    let g = (*group).max(1) as usize;
                    let h_in = *input_h as usize;
                    let w_in = *input_w as usize;

                    if h_in == 0 || w_in == 0 || kh == 0 || kw == 0 {
                        return Err(BackendError::Shape("conv2d: zero spatial dims".into()));
                    }

                    // Infer shapes.
                    let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                    let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
                    let spatial_in = h_in * w_in;
                    let spatial_out = h_out * w_out;
                    let ic = data.len().checked_div(spatial_in).unwrap_or(0);
                    let oc = (ic / g)
                        .checked_mul(kh)
                        .and_then(|v| v.checked_mul(kw))
                        .and_then(|denom| weight.len().checked_div(denom))
                        .unwrap_or(0);

                    if oc == 0 || ic == 0 {
                        return Err(BackendError::Shape("conv2d: can't infer channels".into()));
                    }

                    let ic_per_group = ic / g;
                    let oc_per_group = oc / g;
                    let n = 1; // batch=1
                    let mut out = vec![0.0f32; n * oc * spatial_out];

                    for batch in 0..n {
                        for grp in 0..g {
                            for oc_idx in 0..oc_per_group {
                                let oc_abs = grp * oc_per_group + oc_idx;
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let mut sum = 0.0f32;
                                        for ic_idx in 0..ic_per_group {
                                            let ic_abs = grp * ic_per_group + ic_idx;
                                            for ky in 0..kh {
                                                for kx in 0..kw {
                                                    let iy =
                                                        (oh * sh + ky * dh) as isize - ph as isize;
                                                    let ix =
                                                        (ow * sw + kx * dw) as isize - pw as isize;
                                                    if iy >= 0
                                                        && iy < h_in as isize
                                                        && ix >= 0
                                                        && ix < w_in as isize
                                                    {
                                                        let d_idx = batch * ic * spatial_in
                                                            + ic_abs * spatial_in
                                                            + iy as usize * w_in
                                                            + ix as usize;
                                                        let w_idx = oc_abs
                                                            * (ic_per_group * kh * kw)
                                                            + ic_idx * (kh * kw)
                                                            + ky * kw
                                                            + kx;
                                                        sum += data[d_idx] * weight[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                        let o_idx = batch * oc * spatial_out
                                            + oc_abs * spatial_out
                                            + oh * w_out
                                            + ow;
                                        out[o_idx] = sum;
                                    }
                                }
                                // Add bias.
                                if let Some(b) = bias {
                                    if oc_abs < b.len() {
                                        let base = batch * oc * spatial_out + oc_abs * spatial_out;
                                        for s in 0..spatial_out {
                                            out[base + s] += b[oc_abs];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // InstanceNorm.
                if let FloatOp::InstanceNorm { size, epsilon } = op {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("instancenorm requires 3 inputs".into()));
                    }
                    let input: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let scale: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let bias: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let spatial = *size as usize;
                    let eps = f32::from_bits(*epsilon);
                    let mut out = vec![0.0f32; input.len()];
                    if spatial > 0 {
                        for (ch_idx, chunk) in input.chunks(spatial).enumerate() {
                            let mean: f32 = chunk.iter().sum::<f32>() / spatial as f32;
                            let var: f32 =
                                chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                                    / spatial as f32;
                            let inv_std = 1.0 / (var + eps).sqrt();
                            let c = ch_idx % scale.len().max(1);
                            let base = ch_idx * spatial;
                            for (i, &v) in chunk.iter().enumerate() {
                                out[base + i] = (v - mean) * inv_std * scale[c] + bias[c];
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Slice: copy a sub-range along an axis.
                if let FloatOp::Slice { start, end, .. } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("slice requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *start as usize;
                    let e = *end as usize;
                    // For 1D slice: just copy the range.
                    if s < e && e <= in_f.len() {
                        let out: Vec<f32> = in_f[s..e].to_vec();
                        *output = bytemuck::cast_slice(&out).to_vec();
                        return Ok(output.len());
                    }
                    // Fallback: copy everything.
                    *output = input.to_vec();
                    return Ok(output.len());
                }

                // Concat: combine inputs along an axis.
                if matches!(op, FloatOp::Concat { .. }) {
                    let mut combined = Vec::new();
                    for inp in inputs {
                        combined.extend_from_slice(inp);
                    }
                    *output = combined;
                    return Ok(output.len());
                }

                // Reshape: no-op (same bytes, different shape interpretation).
                if matches!(op, FloatOp::Reshape) {
                    if let Some(inp) = inputs.first() {
                        *output = (*inp).clone();
                        return Ok(output.len());
                    }
                }

                // Gemm: general matrix multiply with alpha, beta, transpose flags.
                if let FloatOp::Gemm {
                    m,
                    k,
                    n,
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                    quant_b,
                } = op
                {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "gemm requires at least 2 inputs".into(),
                        ));
                    }
                    if *quant_b != 0 {
                        return Err(BackendError::Unsupported(
                            "gemm with quantized B not yet supported in CPU backend".into(),
                        ));
                    }
                    let a_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let b_f: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let m_val = *m as usize;
                    let k_val = *k as usize;
                    let n_val = *n as usize;
                    let alpha_f = f32::from_bits(*alpha);
                    let beta_f = f32::from_bits(*beta);

                    // Resolve actual M from input size if baked M is 0 (variable-length).
                    let actual_m = if m_val > 0 {
                        m_val
                    } else {
                        a_f.len().checked_div(k_val).unwrap_or(0)
                    };
                    if actual_m == 0 || k_val == 0 || n_val == 0 {
                        return Err(BackendError::Shape(format!(
                            "gemm dims invalid: m={actual_m} k={k_val} n={n_val}"
                        )));
                    }

                    let mut out = vec![0.0f32; actual_m * n_val];

                    // Initialize with beta * C if C input is provided.
                    if let Some(c_buf) = inputs.get(2) {
                        let c_f: &[f32] = bytemuck::cast_slice(c_buf);
                        for (i, o) in out.iter_mut().enumerate() {
                            *o = beta_f * c_f[i % c_f.len()];
                        }
                    }

                    #[cfg(has_accelerate)]
                    {
                        // Accelerate BLAS sgemm with transpose support.
                        let ta = if *trans_a { 112 } else { 111 }; // CblasTrans / CblasNoTrans
                        let tb = if *trans_b { 112 } else { 111 };
                        let lda = if *trans_a {
                            actual_m as i32
                        } else {
                            k_val as i32
                        };
                        let ldb = if *trans_b { k_val as i32 } else { n_val as i32 };
                        unsafe {
                            cblas_sgemm(
                                101, // CblasRowMajor
                                ta,
                                tb,
                                actual_m as i32,
                                n_val as i32,
                                k_val as i32,
                                alpha_f,
                                a_f.as_ptr(),
                                lda,
                                b_f.as_ptr(),
                                ldb,
                                if inputs.get(2).is_some() { 1.0 } else { 0.0 },
                                out.as_mut_ptr(),
                                n_val as i32,
                            );
                        }
                    }
                    #[cfg(not(has_accelerate))]
                    {
                        // Naive Gemm with transpose support.
                        for i in 0..actual_m {
                            for j in 0..n_val {
                                let mut sum = 0.0f32;
                                for p in 0..k_val {
                                    let a_idx = if *trans_a {
                                        p * actual_m + i
                                    } else {
                                        i * k_val + p
                                    };
                                    let b_idx = if *trans_b {
                                        j * k_val + p
                                    } else {
                                        p * n_val + j
                                    };
                                    sum += a_f[a_idx] * b_f[b_idx];
                                }
                                out[i * n_val + j] += alpha_f * sum;
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // GroupNorm: normalize over groups of channels.
                // Inputs: [x, weight, bias]. x is [N, C, spatial...].
                // We treat x as [C, spatial_per_channel] where C = weight.len().
                if let FloatOp::GroupNorm {
                    num_groups,
                    epsilon,
                } = op
                {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("group_norm requires 3 inputs".into()));
                    }
                    let input: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let bias: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let ng = *num_groups as usize;
                    let eps = f32::from_bits(*epsilon);
                    let num_channels = weight.len();

                    if ng == 0 || num_channels == 0 {
                        return Err(BackendError::Shape(
                            "group_norm: num_groups or channels is 0".into(),
                        ));
                    }
                    let channels_per_group = num_channels / ng;
                    if channels_per_group == 0 {
                        return Err(BackendError::Shape(
                            "group_norm: channels_per_group is 0".into(),
                        ));
                    }
                    let spatial = input.len() / num_channels;
                    if spatial == 0 {
                        return Err(BackendError::Shape("group_norm: spatial size is 0".into()));
                    }

                    let mut out = vec![0.0f32; input.len()];
                    let group_size = channels_per_group * spatial;

                    for g in 0..ng {
                        // Compute mean and variance over the group.
                        let mut sum = 0.0f64;
                        let mut sum_sq = 0.0f64;
                        for c_in_g in 0..channels_per_group {
                            let c = g * channels_per_group + c_in_g;
                            let base = c * spatial;
                            for s in 0..spatial {
                                let v = input[base + s] as f64;
                                sum += v;
                                sum_sq += v * v;
                            }
                        }
                        let mean = sum / group_size as f64;
                        let var = sum_sq / group_size as f64 - mean * mean;
                        let inv_std = 1.0 / (var + eps as f64).sqrt();

                        for c_in_g in 0..channels_per_group {
                            let c = g * channels_per_group + c_in_g;
                            let base = c * spatial;
                            let w = weight[c] as f64;
                            let b = bias[c] as f64;
                            for s in 0..spatial {
                                let v = input[base + s] as f64;
                                out[base + s] = ((v - mean) * inv_std * w + b) as f32;
                            }
                        }
                    }

                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // AddRmsNorm: fused Add + RmsNorm.
                // Inputs: [x, residual, weight].
                if let FloatOp::AddRmsNorm { size, epsilon } = op {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("add_rmsnorm requires 3 inputs".into()));
                    }
                    let x: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let residual: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let row_size = *size as usize;
                    let eps = f32::from_bits(*epsilon);

                    if row_size == 0 {
                        return Err(BackendError::Shape("add_rmsnorm: size is 0".into()));
                    }

                    let mut out = vec![0.0f32; x.len()];
                    for (chunk_idx, (x_chunk, r_chunk)) in x
                        .chunks(row_size)
                        .zip(residual.chunks(row_size))
                        .enumerate()
                    {
                        // Add residual first.
                        let added: Vec<f32> = x_chunk
                            .iter()
                            .zip(r_chunk.iter())
                            .map(|(&a, &b)| a + b)
                            .collect();
                        let ms: f32 = added.iter().map(|v| v * v).sum::<f32>() / row_size as f32;
                        let inv_rms = 1.0 / (ms + eps).sqrt();
                        let base = chunk_idx * row_size;
                        for (i, &v) in added.iter().enumerate() {
                            out[base + i] = v * inv_rms * weight[i % weight.len()];
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Attention: scaled dot-product attention.
                if let FloatOp::Attention {
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                    scale,
                    causal,
                    heads_first,
                    ..
                } = op
                {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape(
                            "attention requires 3 inputs (Q, K, V)".into(),
                        ));
                    }
                    let q_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let k_f: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let v_f: &[f32] = bytemuck::cast_slice(inputs[2]);

                    let hd = *head_dim as usize;
                    let nqh = *num_q_heads as usize;
                    let nkvh = *num_kv_heads as usize;
                    let sc = f32::from_bits(*scale);

                    if hd == 0 || nqh == 0 || nkvh == 0 {
                        return Err(BackendError::Shape("attention: zero dim parameter".into()));
                    }

                    // Determine seq lengths from input sizes.
                    // Q: [nqh, seq_q, hd] or [seq_q, nqh, hd]
                    let seq_q = q_f.len() / (nqh * hd);
                    let seq_k = k_f.len() / (nkvh * hd);

                    if seq_q == 0 || seq_k == 0 {
                        return Err(BackendError::Shape("attention: zero seq length".into()));
                    }

                    let gqa_ratio = nqh / nkvh;
                    let mut out = vec![0.0f32; nqh * seq_q * hd];

                    for h in 0..nqh {
                        let kv_h = h / gqa_ratio;

                        for sq in 0..seq_q {
                            // Compute attention scores for this query position.
                            let mut scores = vec![f32::NEG_INFINITY; seq_k];

                            let max_sk = if *causal { (sq + 1).min(seq_k) } else { seq_k };

                            #[allow(clippy::needless_range_loop)]
                            for sk in 0..max_sk {
                                let mut dot = 0.0f32;
                                for d in 0..hd {
                                    let q_idx = if *heads_first {
                                        h * seq_q * hd + sq * hd + d
                                    } else {
                                        sq * nqh * hd + h * hd + d
                                    };
                                    let k_idx = if *heads_first {
                                        kv_h * seq_k * hd + sk * hd + d
                                    } else {
                                        sk * nkvh * hd + kv_h * hd + d
                                    };
                                    dot += q_f[q_idx] * k_f[k_idx];
                                }
                                scores[sk] = dot * sc;
                            }

                            // Softmax over scores.
                            let max_score =
                                scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0.0f32;
                            for s in &mut scores {
                                *s = (*s - max_score).exp();
                                sum += *s;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for s in &mut scores {
                                    *s *= inv;
                                }
                            }

                            // Weighted sum over V.
                            for d in 0..hd {
                                let mut acc = 0.0f32;
                                for (sk, &score) in scores.iter().enumerate() {
                                    let v_idx = if *heads_first {
                                        kv_h * seq_k * hd + sk * hd + d
                                    } else {
                                        sk * nkvh * hd + kv_h * hd + d
                                    };
                                    acc += score * v_f[v_idx];
                                }
                                let o_idx = if *heads_first {
                                    h * seq_q * hd + sq * hd + d
                                } else {
                                    sq * nqh * hd + h * hd + d
                                };
                                out[o_idx] = acc;
                            }
                        }
                    }

                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // RotaryEmbedding: apply sin/cos rotation to pairs of features.
                if let FloatOp::RotaryEmbedding { dim, base, n_heads } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("rope requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let dim_val = *dim as usize;
                    let base_f = f32::from_bits(*base);
                    let n_h = *n_heads as usize;

                    if dim_val == 0 {
                        *output = input.to_vec();
                        return Ok(output.len());
                    }

                    let mut out = in_f.to_vec();
                    let half_dim = dim_val / 2;

                    // Each chunk of dim_val floats is one (head, position) pair.
                    // Position = chunk_index / n_heads.
                    let n_heads_eff = n_h.max(1);
                    for (chunk_idx, chunk) in out.chunks_mut(dim_val).enumerate() {
                        let pos = (chunk_idx / n_heads_eff) as f32;
                        for i in 0..half_dim.min(chunk.len() / 2) {
                            let freq = 1.0 / base_f.powf((2 * i) as f32 / dim_val as f32);
                            let angle = pos * freq;
                            let cos_a = angle.cos();
                            let sin_a = angle.sin();
                            let x0 = chunk[i];
                            let x1 = chunk[i + half_dim];
                            chunk[i] = x0 * cos_a - x1 * sin_a;
                            chunk[i + half_dim] = x0 * sin_a + x1 * cos_a;
                        }
                    }

                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Gather: row indexing along a dimension.
                if let FloatOp::Gather { dim: _, dtype } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape("gather requires 2 inputs".into()));
                    }
                    let elem_size = dtype.byte_size();
                    let data = inputs[0].as_slice();
                    let idx_bytes = inputs[1].as_slice();

                    // Indices are i64.
                    let indices: &[i64] = bytemuck::cast_slice(idx_bytes);
                    let row_size = if !indices.is_empty() && !data.is_empty() {
                        data.len() / (data.len() / elem_size).max(1) // elem_size per element
                    } else {
                        elem_size
                    };
                    // For 2D gather: data = [rows, cols], indices index rows.
                    // row_bytes = total / num_rows, but we need to know num_rows.
                    // Use params.u32s[0] as the axis-0 dim if available, otherwise infer.
                    let axis_dim = if !_params.u32s.is_empty() {
                        _params.u32s[0] as usize
                    } else {
                        data.len() / elem_size
                    };
                    let row_bytes = data.len().checked_div(axis_dim).unwrap_or(row_size);

                    let mut result = Vec::with_capacity(indices.len() * row_bytes);
                    for &idx in indices {
                        let row = if idx < 0 {
                            (axis_dim as i64 + idx) as usize
                        } else {
                            idx as usize
                        };
                        if row < axis_dim {
                            let start = row * row_bytes;
                            let end = start + row_bytes;
                            if end <= data.len() {
                                result.extend_from_slice(&data[start..end]);
                            } else {
                                result.extend(std::iter::repeat_n(0u8, row_bytes));
                            }
                        } else {
                            result.extend(std::iter::repeat_n(0u8, row_bytes));
                        }
                    }
                    *output = result;
                    return Ok(output.len());
                }

                // Embed: embedding lookup.
                // Inputs: [token_ids (u32), table (f32)].
                if let FloatOp::Embed { dim, quant } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape("embed requires 2 inputs".into()));
                    }
                    if *quant != 0 {
                        return Err(BackendError::Unsupported(
                            "embed with quantization not yet supported in CPU backend".into(),
                        ));
                    }
                    let ids: &[u32] = bytemuck::cast_slice(inputs[0]);
                    let table: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let d = *dim as usize;
                    if d == 0 {
                        return Err(BackendError::Shape("embed: dim is 0".into()));
                    }
                    let vocab = table.len() / d;
                    let mut out = Vec::with_capacity(ids.len() * d);
                    for &id in ids {
                        let row = id as usize;
                        if row < vocab {
                            out.extend_from_slice(&table[row * d..(row + 1) * d]);
                        } else {
                            out.extend(std::iter::repeat_n(0.0f32, d));
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Where: conditional select. Inputs: [cond (u8), x (f32), y (f32)].
                if matches!(op, FloatOp::Where) {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("where requires 3 inputs".into()));
                    }
                    let cond = inputs[0].as_slice();
                    let x: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let y: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let n = cond.len().max(x.len()).max(y.len());
                    let mut out = vec![0.0f32; n];
                    for i in 0..n {
                        let c = cond[i % cond.len()];
                        out[i] = if c != 0 {
                            x[i % x.len()]
                        } else {
                            y[i % y.len()]
                        };
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Range: generate [start, limit) with delta.
                if matches!(op, FloatOp::Range) {
                    if inputs.len() < 3 {
                        return Err(BackendError::Shape("range requires 3 inputs".into()));
                    }
                    let start_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let limit_f: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let delta_f: &[f32] = bytemuck::cast_slice(inputs[2]);
                    let start = *start_f
                        .first()
                        .ok_or_else(|| BackendError::Shape("range: empty start".into()))?;
                    let limit = *limit_f
                        .first()
                        .ok_or_else(|| BackendError::Shape("range: empty limit".into()))?;
                    let delta = *delta_f
                        .first()
                        .ok_or_else(|| BackendError::Shape("range: empty delta".into()))?;

                    if delta == 0.0 {
                        return Err(BackendError::Shape("range: delta is 0".into()));
                    }

                    let n = ((limit - start) / delta).ceil().max(0.0) as usize;
                    let mut out = Vec::with_capacity(n);
                    let mut v = start;
                    for _ in 0..n {
                        out.push(v);
                        v += delta;
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Cast: dtype conversion.
                if let FloatOp::Cast { from, to } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("cast requires 1 input".into()))?;
                    *output = dispatch_cast(input, *from, *to)?;
                    return Ok(output.len());
                }

                // Shape: return input shape as i64 tensor.
                if let FloatOp::Shape { start, end, .. } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("shape requires 1 input".into()))?;
                    // Use params.u32s as the actual shape if available.
                    let shape_vals: Vec<i64> = if !_params.u32s.is_empty() {
                        _params.u32s.iter().map(|&d| d as i64).collect()
                    } else {
                        // Fallback: return total element count as 1-D shape.
                        vec![input.len() as i64 / 4]
                    };
                    // Apply start/end slicing.
                    let ndim = shape_vals.len() as i64;
                    let s = if *start < 0 {
                        (ndim + *start).max(0) as usize
                    } else {
                        (*start as usize).min(shape_vals.len())
                    };
                    let e = if *end == i64::MAX {
                        shape_vals.len()
                    } else if *end < 0 {
                        (ndim + *end).max(0) as usize
                    } else {
                        (*end as usize).min(shape_vals.len())
                    };
                    let sliced = if s < e { &shape_vals[s..e] } else { &[] };
                    *output = bytemuck::cast_slice(sliced).to_vec();
                    return Ok(output.len());
                }

                // LogSoftmax: log(softmax(x)).
                if let FloatOp::LogSoftmax { size } = op {
                    let input = inputs.first().ok_or_else(|| {
                        BackendError::Shape("log_softmax requires 1 input".into())
                    })?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let row_size = *size as usize;
                    let mut out = in_f.to_vec();
                    if row_size > 0 {
                        for row in out.chunks_mut(row_size) {
                            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let log_sum_exp =
                                row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
                            for v in row.iter_mut() {
                                *v -= log_sum_exp;
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Expand: broadcast-expand data along dims.
                if let FloatOp::Expand { ndim, target_shape } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("expand requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let nd = *ndim as usize;
                    let out_shape: Vec<usize> =
                        target_shape[..nd].iter().map(|&d| d as usize).collect();
                    let total: usize = out_shape.iter().product();

                    // Input shape from params.u32s, or infer as 1-D.
                    let in_shape: Vec<usize> = if _params.u32s.len() >= nd {
                        _params.u32s[..nd].iter().map(|&d| d as usize).collect()
                    } else {
                        // Infer: rightmost dims filled from in_f.len().
                        let mut s = vec![1usize; nd];
                        s[nd - 1] = in_f.len();
                        s
                    };

                    // Compute input strides.
                    let mut in_strides = vec![1usize; nd];
                    for i in (0..nd.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
                    }
                    let mut out_strides = vec![1usize; nd];
                    for i in (0..nd.saturating_sub(1)).rev() {
                        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                    }

                    let mut out = vec![0.0f32; total];
                    for (flat, o) in out.iter_mut().enumerate() {
                        let mut rem = flat;
                        let mut in_flat = 0usize;
                        for d in 0..nd {
                            let coord = rem / out_strides[d];
                            rem %= out_strides[d];
                            // Broadcast: if input dim is 1, always index 0.
                            let in_coord = if in_shape[d] == 1 { 0 } else { coord };
                            in_flat += in_coord * in_strides[d];
                        }
                        *o = in_f[in_flat % in_f.len()];
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Reductions: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd.
                if let FloatOp::ReduceSum { size } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("reduce_sum requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    if s == 0 {
                        *output = bytemuck::cast_slice(&[in_f.iter().sum::<f32>()]).to_vec();
                    } else {
                        let out: Vec<f32> = in_f.chunks(s).map(|c| c.iter().sum::<f32>()).collect();
                        *output = bytemuck::cast_slice(&out).to_vec();
                    }
                    return Ok(output.len());
                }
                if let FloatOp::ReduceMean { size } = op {
                    let input = inputs.first().ok_or_else(|| {
                        BackendError::Shape("reduce_mean requires 1 input".into())
                    })?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    if s == 0 {
                        let mean = in_f.iter().sum::<f32>() / in_f.len().max(1) as f32;
                        *output = bytemuck::cast_slice(&[mean]).to_vec();
                    } else {
                        let out: Vec<f32> = in_f
                            .chunks(s)
                            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                            .collect();
                        *output = bytemuck::cast_slice(&out).to_vec();
                    }
                    return Ok(output.len());
                }
                if let FloatOp::ReduceMax { size } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("reduce_max requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    if s == 0 {
                        let mx = in_f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        *output = bytemuck::cast_slice(&[mx]).to_vec();
                    } else {
                        let out: Vec<f32> = in_f
                            .chunks(s)
                            .map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                            .collect();
                        *output = bytemuck::cast_slice(&out).to_vec();
                    }
                    return Ok(output.len());
                }
                if let FloatOp::ReduceMin { size } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("reduce_min requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    if s == 0 {
                        let mn = in_f.iter().copied().fold(f32::INFINITY, f32::min);
                        *output = bytemuck::cast_slice(&[mn]).to_vec();
                    } else {
                        let out: Vec<f32> = in_f
                            .chunks(s)
                            .map(|c| c.iter().copied().fold(f32::INFINITY, f32::min))
                            .collect();
                        *output = bytemuck::cast_slice(&out).to_vec();
                    }
                    return Ok(output.len());
                }
                if let FloatOp::ReduceProd { size } = op {
                    let input = inputs.first().ok_or_else(|| {
                        BackendError::Shape("reduce_prod requires 1 input".into())
                    })?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    if s == 0 {
                        let prod = in_f.iter().product::<f32>();
                        *output = bytemuck::cast_slice(&[prod]).to_vec();
                    } else {
                        let out: Vec<f32> =
                            in_f.chunks(s).map(|c| c.iter().product::<f32>()).collect();
                        *output = bytemuck::cast_slice(&out).to_vec();
                    }
                    return Ok(output.len());
                }

                // ConvTranspose: transposed convolution.
                if let FloatOp::ConvTranspose {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dilation_h,
                    dilation_w,
                    group,
                    output_pad_h,
                    output_pad_w,
                    input_h,
                    input_w,
                } = op
                {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "conv_transpose requires at least 2 inputs".into(),
                        ));
                    }
                    let data: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let weight: &[f32] = bytemuck::cast_slice(inputs[1]);
                    let bias: Option<&[f32]> =
                        inputs.get(2).map(|b| bytemuck::cast_slice(b.as_slice()));

                    let kh = *kernel_h as usize;
                    let kw = *kernel_w as usize;
                    let sh = (*stride_h).max(1) as usize;
                    let sw = (*stride_w).max(1) as usize;
                    let ph = *pad_h as usize;
                    let pw = *pad_w as usize;
                    let dh = (*dilation_h).max(1) as usize;
                    let dw = (*dilation_w).max(1) as usize;
                    let g = (*group).max(1) as usize;
                    let oph = *output_pad_h as usize;
                    let opw = *output_pad_w as usize;
                    let h_in = *input_h as usize;
                    let w_in = *input_w as usize;

                    if h_in == 0 || w_in == 0 || kh == 0 || kw == 0 {
                        return Err(BackendError::Shape(
                            "conv_transpose: zero spatial dims".into(),
                        ));
                    }

                    let h_out = (h_in - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1;
                    let w_out = (w_in - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1;
                    let spatial_in = h_in * w_in;
                    let spatial_out = h_out * w_out;

                    // weight: [ic, oc_per_group, kh, kw] for conv_transpose
                    let ic = data.len() / spatial_in;
                    let ic_per_group = ic / g;
                    let oc_per_group = ic
                        .checked_mul(kh)
                        .and_then(|v| v.checked_mul(kw))
                        .and_then(|denom| weight.len().checked_div(denom))
                        .unwrap_or(0);
                    let oc = oc_per_group * g;

                    if oc == 0 || ic == 0 {
                        return Err(BackendError::Shape(
                            "conv_transpose: can't infer channels".into(),
                        ));
                    }

                    let mut out = vec![0.0f32; oc * spatial_out];

                    for grp in 0..g {
                        for ic_idx in 0..ic_per_group {
                            let ic_abs = grp * ic_per_group + ic_idx;
                            for ih in 0..h_in {
                                for iw in 0..w_in {
                                    let x_val = data[ic_abs * spatial_in + ih * w_in + iw];
                                    for oc_idx in 0..oc_per_group {
                                        let oc_abs = grp * oc_per_group + oc_idx;
                                        for ky in 0..kh {
                                            for kx in 0..kw {
                                                let oh = ih * sh + ky * dh;
                                                let ow_val = iw * sw + kx * dw;
                                                if oh >= ph
                                                    && ow_val >= pw
                                                    && (oh - ph) < h_out
                                                    && (ow_val - pw) < w_out
                                                {
                                                    let out_h = oh - ph;
                                                    let out_w = ow_val - pw;
                                                    let w_idx = ic_abs * (oc_per_group * kh * kw)
                                                        + oc_idx * (kh * kw)
                                                        + ky * kw
                                                        + kx;
                                                    let o_idx = oc_abs * spatial_out
                                                        + out_h * w_out
                                                        + out_w;
                                                    out[o_idx] += x_val * weight[w_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Add bias.
                    if let Some(b) = bias {
                        for c in 0..oc {
                            if c < b.len() {
                                let base = c * spatial_out;
                                for s in 0..spatial_out {
                                    out[base + s] += b[c];
                                }
                            }
                        }
                    }

                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // MaxPool2d.
                if let FloatOp::MaxPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                } = op
                {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("maxpool2d requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let kh = *kernel_h as usize;
                    let kw = *kernel_w as usize;
                    let sh = (*stride_h).max(1) as usize;
                    let sw = (*stride_w).max(1) as usize;
                    let ph = *pad_h as usize;
                    let pw = *pad_w as usize;

                    // Use params for input spatial dims: [channels, h_in, w_in].
                    if _params.u32s.len() < 3 {
                        return Err(BackendError::Shape(
                            "maxpool2d: params.u32s must have [channels, h_in, w_in]".into(),
                        ));
                    }
                    let channels = _params.u32s[0] as usize;
                    let h_in = _params.u32s[1] as usize;
                    let w_in = _params.u32s[2] as usize;

                    let h_out = (h_in + 2 * ph - kh) / sh + 1;
                    let w_out = (w_in + 2 * pw - kw) / sw + 1;
                    let spatial_in = h_in * w_in;

                    let mut out = vec![f32::NEG_INFINITY; channels * h_out * w_out];
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let mut mx = f32::NEG_INFINITY;
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oh * sh + ky) as isize - ph as isize;
                                        let ix = (ow * sw + kx) as isize - pw as isize;
                                        if iy >= 0
                                            && iy < h_in as isize
                                            && ix >= 0
                                            && ix < w_in as isize
                                        {
                                            let idx =
                                                c * spatial_in + iy as usize * w_in + ix as usize;
                                            if idx < in_f.len() {
                                                mx = mx.max(in_f[idx]);
                                            }
                                        }
                                    }
                                }
                                out[c * h_out * w_out + oh * w_out + ow] = mx;
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // AvgPool2d.
                if let FloatOp::AvgPool2d {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                } = op
                {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("avgpool2d requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let kh = *kernel_h as usize;
                    let kw = *kernel_w as usize;
                    let sh = (*stride_h).max(1) as usize;
                    let sw = (*stride_w).max(1) as usize;
                    let ph = *pad_h as usize;
                    let pw = *pad_w as usize;

                    if _params.u32s.len() < 3 {
                        return Err(BackendError::Shape(
                            "avgpool2d: params.u32s must have [channels, h_in, w_in]".into(),
                        ));
                    }
                    let channels = _params.u32s[0] as usize;
                    let h_in = _params.u32s[1] as usize;
                    let w_in = _params.u32s[2] as usize;

                    let h_out = (h_in + 2 * ph - kh) / sh + 1;
                    let w_out = (w_in + 2 * pw - kw) / sw + 1;
                    let spatial_in = h_in * w_in;

                    let mut out = vec![0.0f32; channels * h_out * w_out];
                    for c in 0..channels {
                        for oh in 0..h_out {
                            for ow in 0..w_out {
                                let mut sum = 0.0f32;
                                let mut count = 0u32;
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oh * sh + ky) as isize - ph as isize;
                                        let ix = (ow * sw + kx) as isize - pw as isize;
                                        if iy >= 0
                                            && iy < h_in as isize
                                            && ix >= 0
                                            && ix < w_in as isize
                                        {
                                            let idx =
                                                c * spatial_in + iy as usize * w_in + ix as usize;
                                            if idx < in_f.len() {
                                                sum += in_f[idx];
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                                if count > 0 {
                                    out[c * h_out * w_out + oh * w_out + ow] = sum / count as f32;
                                }
                            }
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // GlobalAvgPool: spatial dims -> 1.
                if let FloatOp::GlobalAvgPool {
                    channels,
                    spatial_h,
                    spatial_w,
                } = op
                {
                    let input = inputs.first().ok_or_else(|| {
                        BackendError::Shape("global_avg_pool requires 1 input".into())
                    })?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let nc = *channels as usize;
                    let sh = *spatial_h as usize;
                    let sw = *spatial_w as usize;
                    let spatial = sh * sw;
                    if spatial == 0 || nc == 0 {
                        return Err(BackendError::Shape(
                            "global_avg_pool: zero channels or spatial".into(),
                        ));
                    }
                    let mut out = vec![0.0f32; nc];
                    for (c, o) in out.iter_mut().enumerate() {
                        let base = c * spatial;
                        let end = (base + spatial).min(in_f.len());
                        if base < in_f.len() {
                            let sum: f32 = in_f[base..end].iter().sum();
                            *o = sum / spatial as f32;
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // Resize: nearest/bilinear interpolation.
                if let FloatOp::Resize { mode } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "resize requires 2 inputs (data, scales/sizes)".into(),
                        ));
                    }
                    let in_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let scales_or_sizes: &[f32] = bytemuck::cast_slice(inputs[1]);

                    // Use params: [channels, h_in, w_in, h_out, w_out].
                    if _params.u32s.len() < 5 {
                        return Err(BackendError::Shape(
                            "resize: params.u32s must have [channels, h_in, w_in, h_out, w_out]"
                                .into(),
                        ));
                    }
                    let channels = _params.u32s[0] as usize;
                    let h_in = _params.u32s[1] as usize;
                    let w_in = _params.u32s[2] as usize;
                    let h_out = _params.u32s[3] as usize;
                    let w_out = _params.u32s[4] as usize;
                    let spatial_in = h_in * w_in;
                    let spatial_out = h_out * w_out;
                    let _ = scales_or_sizes; // sizes are in params

                    let mut out = vec![0.0f32; channels * spatial_out];

                    match mode {
                        0 => {
                            // Nearest neighbor.
                            let h_scale = if h_out > 0 {
                                h_in as f32 / h_out as f32
                            } else {
                                1.0
                            };
                            let w_scale = if w_out > 0 {
                                w_in as f32 / w_out as f32
                            } else {
                                1.0
                            };
                            for c in 0..channels {
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let ih = ((oh as f32 + 0.5) * h_scale) as usize;
                                        let iw = ((ow as f32 + 0.5) * w_scale) as usize;
                                        let ih = ih.min(h_in.saturating_sub(1));
                                        let iw = iw.min(w_in.saturating_sub(1));
                                        let idx = c * spatial_in + ih * w_in + iw;
                                        let o_idx = c * spatial_out + oh * w_out + ow;
                                        if idx < in_f.len() {
                                            out[o_idx] = in_f[idx];
                                        }
                                    }
                                }
                            }
                        }
                        1 => {
                            // Bilinear interpolation.
                            let h_scale = if h_out > 1 {
                                (h_in - 1) as f32 / (h_out - 1).max(1) as f32
                            } else {
                                0.0
                            };
                            let w_scale = if w_out > 1 {
                                (w_in - 1) as f32 / (w_out - 1).max(1) as f32
                            } else {
                                0.0
                            };
                            for c in 0..channels {
                                for oh in 0..h_out {
                                    for ow in 0..w_out {
                                        let fy = oh as f32 * h_scale;
                                        let fx = ow as f32 * w_scale;
                                        let y0 = fy as usize;
                                        let x0 = fx as usize;
                                        let y1 = (y0 + 1).min(h_in.saturating_sub(1));
                                        let x1 = (x0 + 1).min(w_in.saturating_sub(1));
                                        let dy = fy - y0 as f32;
                                        let dx = fx - x0 as f32;

                                        let base = c * spatial_in;
                                        let v00 =
                                            in_f.get(base + y0 * w_in + x0).copied().unwrap_or(0.0);
                                        let v01 =
                                            in_f.get(base + y0 * w_in + x1).copied().unwrap_or(0.0);
                                        let v10 =
                                            in_f.get(base + y1 * w_in + x0).copied().unwrap_or(0.0);
                                        let v11 =
                                            in_f.get(base + y1 * w_in + x1).copied().unwrap_or(0.0);

                                        let val = v00 * (1.0 - dy) * (1.0 - dx)
                                            + v01 * (1.0 - dy) * dx
                                            + v10 * dy * (1.0 - dx)
                                            + v11 * dy * dx;
                                        out[c * spatial_out + oh * w_out + ow] = val;
                                    }
                                }
                            }
                        }
                        _ => {
                            // Cubic or other modes — fall through to unsupported.
                            return Err(BackendError::Unsupported(format!(
                                "resize mode {mode} not supported"
                            )));
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // PadOp: N-D padding.
                if let FloatOp::PadOp { mode } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "pad requires 2 inputs (data, pads)".into(),
                        ));
                    }
                    let in_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let pads_f: &[f32] = bytemuck::cast_slice(inputs[1]);

                    // Simple 1-D constant pad as reference implementation.
                    // pads format: [begin_0, begin_1, ..., end_0, end_1, ...].
                    // For the simple case, treat as flat 1-D.
                    if pads_f.len() >= 2 {
                        let pad_begin = pads_f[0] as usize;
                        let pad_end = pads_f[pads_f.len() / 2] as usize;
                        let out_len = pad_begin + in_f.len() + pad_end;
                        let mut out = vec![0.0f32; out_len];
                        out[pad_begin..pad_begin + in_f.len()].copy_from_slice(in_f);
                        if *mode == 1 {
                            // Reflect.
                            for (i, o) in out[..pad_begin].iter_mut().enumerate() {
                                let idx = pad_begin - i;
                                *o = if idx < in_f.len() { in_f[idx] } else { 0.0 };
                            }
                            for (i, o) in out[pad_begin + in_f.len()..].iter_mut().enumerate() {
                                let idx = in_f.len().saturating_sub(2 + i);
                                *o = if idx < in_f.len() { in_f[idx] } else { 0.0 };
                            }
                        } else if *mode == 2 {
                            // Edge.
                            let first = *in_f.first().unwrap_or(&0.0);
                            let last = *in_f.last().unwrap_or(&0.0);
                            for v in out[..pad_begin].iter_mut() {
                                *v = first;
                            }
                            for v in out[pad_begin + in_f.len()..].iter_mut() {
                                *v = last;
                            }
                        }
                        *output = bytemuck::cast_slice(&out).to_vec();
                    } else {
                        // No padding needed.
                        *output = inputs[0].clone();
                    }
                    return Ok(output.len());
                }

                // Dequantize: Q4_0 -> f32.
                if matches!(op, FloatOp::Dequantize) {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("dequantize requires 1 input".into()))?;
                    // Q4_0 block: 2 bytes scale (f16) + 16 bytes nibbles = 18 bytes -> 32 f32 values.
                    let data = input.as_slice();
                    let block_size = 18;
                    let n_blocks = data.len() / block_size;
                    let mut out = Vec::with_capacity(n_blocks * 32);

                    for blk in 0..n_blocks {
                        let base = blk * block_size;
                        // Scale is stored as f16 in first 2 bytes.
                        let scale_bits = u16::from_le_bytes([data[base], data[base + 1]]);
                        let scale = half_to_f32(scale_bits);
                        for i in 0..16 {
                            let byte = data[base + 2 + i];
                            let lo = (byte & 0x0F) as i8 - 8;
                            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                            out.push(lo as f32 * scale);
                            out.push(hi as f32 * scale);
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // GatherND: stub pass-through.
                if matches!(op, FloatOp::GatherND) {
                    if let Some(inp) = inputs.first() {
                        *output = (*inp).clone();
                        return Ok(output.len());
                    }
                }

                // ScatterND: stub pass-through (data unchanged).
                if matches!(op, FloatOp::ScatterND) {
                    if let Some(inp) = inputs.first() {
                        *output = (*inp).clone();
                        return Ok(output.len());
                    }
                }

                // TopK: return top-k values and indices.
                if let FloatOp::TopK { axis: _, largest } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "topk requires 2 inputs (data, K)".into(),
                        ));
                    }
                    let in_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let k_bytes: &[i64] = bytemuck::cast_slice(inputs[1]);
                    let k = k_bytes.first().copied().unwrap_or(1) as usize;

                    // Simple 1-D top-k.
                    let mut indexed: Vec<(usize, f32)> = in_f.iter().copied().enumerate().collect();
                    if *largest {
                        indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    } else {
                        indexed.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    let k = k.min(indexed.len());
                    let values: Vec<f32> = indexed[..k].iter().map(|&(_, v)| v).collect();
                    // Output values only (indices would need a second output).
                    *output = bytemuck::cast_slice(&values).to_vec();
                    return Ok(output.len());
                }

                // CumSum: cumulative sum along an axis.
                if let FloatOp::CumSum { .. } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("cumsum requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let mut out = vec![0.0f32; in_f.len()];
                    let mut acc = 0.0f32;
                    for (o, &v) in out.iter_mut().zip(in_f) {
                        acc += v;
                        *o = acc;
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // NonZero: return indices of non-zero elements.
                if matches!(op, FloatOp::NonZero) {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("nonzero requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let indices: Vec<i64> = in_f
                        .iter()
                        .enumerate()
                        .filter(|(_, &v)| v != 0.0)
                        .map(|(i, _)| i as i64)
                        .collect();
                    *output = bytemuck::cast_slice(&indices).to_vec();
                    return Ok(output.len());
                }

                // Compress: select elements along an axis based on condition.
                if let FloatOp::Compress { .. } = op {
                    if inputs.len() < 2 {
                        return Err(BackendError::Shape(
                            "compress requires 2 inputs (data, condition)".into(),
                        ));
                    }
                    let in_f: &[f32] = bytemuck::cast_slice(inputs[0]);
                    let cond = inputs[1].as_slice();
                    let out: Vec<f32> = in_f
                        .iter()
                        .zip(cond.iter().chain(std::iter::repeat(&0u8)))
                        .filter(|(_, &c)| c != 0)
                        .map(|(&v, _)| v)
                        .collect();
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // ReverseSequence: reverse along time axis.
                if let FloatOp::ReverseSequence { .. } = op {
                    let input = inputs.first().ok_or_else(|| {
                        BackendError::Shape("reverse_sequence requires 1 input".into())
                    })?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let mut out = in_f.to_vec();
                    out.reverse();
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // ArgMax: index of maximum value along an axis.
                if let FloatOp::ArgMax { .. } = op {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("argmax requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    // Use params.u32s[0] as the axis size (last dim) if available.
                    let axis_size = if !_params.u32s.is_empty() {
                        _params.u32s[0] as usize
                    } else {
                        in_f.len()
                    };
                    if axis_size == 0 {
                        *output = Vec::new();
                        return Ok(0);
                    }
                    let mut out_indices: Vec<i64> = Vec::with_capacity(in_f.len() / axis_size);
                    for chunk in in_f.chunks(axis_size) {
                        let (max_idx, _) = chunk.iter().enumerate().fold(
                            (0, f32::NEG_INFINITY),
                            |(mi, mv), (i, &v)| {
                                if v > mv {
                                    (i, v)
                                } else {
                                    (mi, mv)
                                }
                            },
                        );
                        out_indices.push(max_idx as i64);
                    }
                    *output = bytemuck::cast_slice(&out_indices).to_vec();
                    return Ok(output.len());
                }

                // LRN: local response normalization.
                if let FloatOp::LRN {
                    size,
                    alpha,
                    beta,
                    bias,
                } = op
                {
                    let input = inputs
                        .first()
                        .ok_or_else(|| BackendError::Shape("lrn requires 1 input".into()))?;
                    let in_f: &[f32] = bytemuck::cast_slice(input);
                    let s = *size as usize;
                    let alpha_f = f32::from_bits(*alpha);
                    let beta_f = f32::from_bits(*beta);
                    let bias_f = f32::from_bits(*bias);

                    // LRN operates across channels. Use params for [channels, spatial].
                    let channels = if !_params.u32s.is_empty() {
                        _params.u32s[0] as usize
                    } else {
                        in_f.len()
                    };
                    let spatial = in_f.len().checked_div(channels).unwrap_or(1);

                    let mut out = vec![0.0f32; in_f.len()];
                    let half = s / 2;
                    for px in 0..spatial {
                        for c in 0..channels {
                            let start = c.saturating_sub(half);
                            let end = (c + half + 1).min(channels);
                            let mut sum_sq = 0.0f32;
                            for j in start..end {
                                let v = in_f[j * spatial + px];
                                sum_sq += v * v;
                            }
                            let norm = (bias_f + alpha_f / s as f32 * sum_sq).powf(beta_f);
                            out[c * spatial + px] = in_f[c * spatial + px] / norm;
                        }
                    }
                    *output = bytemuck::cast_slice(&out).to_vec();
                    return Ok(output.len());
                }

                // KvWrite / KvRead: return Unsupported (tape-level ops).
                if matches!(op, FloatOp::KvWrite { .. } | FloatOp::KvRead { .. }) {
                    return Err(BackendError::Unsupported(
                        "KV cache ops are tape-level, not dispatched to compute backend".into(),
                    ));
                }

                // Deep decode fusions: return Unsupported.
                if matches!(
                    op,
                    FloatOp::NormProjectionGemv { .. }
                        | FloatOp::AddNormProjectionGemv { .. }
                        | FloatOp::SwiGluProjectionGemv { .. }
                ) {
                    return Err(BackendError::Unsupported(format!(
                        "deep decode fusion {op:?} not yet implemented in CPU backend"
                    )));
                }

                Err(BackendError::Unsupported(format!(
                    "CPU dispatch for {op:?} not yet migrated"
                )))
            }
        }
    }

    fn dispatch_ring(
        &self,
        table_idx: usize,
        inputs: &[&Vec<u8>],
        output: &mut Vec<u8>,
    ) -> Result<usize> {
        if table_idx >= self.ring_tables.len() {
            return Err(BackendError::Unsupported(format!(
                "ring table index {table_idx} out of range (have {})",
                self.ring_tables.len()
            )));
        }
        let table = &self.ring_tables[table_idx];
        let input = inputs
            .first()
            .ok_or_else(|| BackendError::Shape("ring op requires at least one input".into()))?;

        output.resize(input.len(), 0);
        for (out, &inp) in output.iter_mut().zip(input.iter()) {
            *out = table[inp as usize];
        }
        Ok(output.len())
    }

    fn load_ring_tables(&mut self, tables: &[&[u8; 256]], _memory: &CpuMemory) {
        self.ring_tables = tables.iter().map(|t| **t).collect();
    }

    fn flush(&self) {
        // No-op for CPU.
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_core::op::FloatDType;

    #[test]
    fn cpu_memory_upload_download_roundtrip() {
        let mem = CpuMemory;
        let data = vec![1u8, 2, 3, 4];
        let buf = mem.upload(&data);
        let result = mem.download(&buf);
        assert_eq!(data, result);
    }

    #[test]
    fn cpu_memory_alloc_zeroed() {
        let mem = CpuMemory;
        let buf = mem.alloc(16);
        assert_eq!(buf.len(), 16);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_ring_dispatch() {
        let mem = CpuMemory;
        let mut backend = CpuBackend::new();

        // Identity table: table[i] = i
        let mut identity = [0u8; 256];
        for i in 0..256 {
            identity[i] = i as u8;
        }
        backend.load_ring_tables(&[&identity], &mem);

        let input = vec![0u8, 127, 255];
        let mut output = Vec::new();
        let written = backend
            .dispatch_ring(0, &[&input], &mut output)
            .expect("ring dispatch should succeed");

        assert_eq!(written, 3);
        assert_eq!(output, vec![0, 127, 255]);
    }

    #[test]
    fn cpu_ring_dispatch_transform() {
        let mem = CpuMemory;
        let mut backend = CpuBackend::new();

        // NOT table: table[i] = 255 - i
        let mut not_table = [0u8; 256];
        for i in 0..256 {
            not_table[i] = (255 - i) as u8;
        }
        backend.load_ring_tables(&[&not_table], &mem);

        let input = vec![0u8, 1, 254, 255];
        let mut output = Vec::new();
        backend
            .dispatch_ring(0, &[&input], &mut output)
            .expect("ring dispatch should succeed");

        assert_eq!(output, vec![255, 254, 1, 0]);
    }

    #[test]
    fn cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_relu_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]).to_vec();
        let mut output = Vec::new();
        let written = backend
            .dispatch(
                &FloatOp::Relu,
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("relu should succeed");
        assert_eq!(written, 20);
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn cpu_add_dispatch() {
        let backend = CpuBackend::new();
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[10.0f32, 20.0, 30.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Add,
                &[&a, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("add should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn cpu_matmul_dispatch() {
        let backend = CpuBackend::new();
        // 2x3 * 3x2 = 2x2
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::MatMul { m: 2, k: 3, n: 2 },
                &[&a, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("matmul should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        // C[0,0] = 1*1 + 2*3 + 3*5 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 64
        assert_eq!(result, &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn cpu_softmax_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Softmax { size: 3 },
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("softmax should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        // Values should be monotonically increasing.
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn cpu_conv2d_dispatch() {
        let backend = CpuBackend::new();
        // 1×1 conv: [1, 2, 2, 2] input, [3, 2, 1, 1] weight → [1, 3, 2, 2] output.
        let input: Vec<u8> =
            bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).to_vec();
        let weight: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Conv2d {
                    kernel_h: 1,
                    kernel_w: 1,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                    dilation_h: 1,
                    dilation_w: 1,
                    group: 1,
                    input_h: 2,
                    input_w: 2,
                },
                &[&input, &weight],
                &mut output,
                &KernelParams::default(),
            )
            .expect("conv2d should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result.len(), 12); // 3 output channels × 2×2 spatial
                                      // Channel 0: weight=[1,0] → copies channel 0: [1,2,3,4]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn cpu_rmsnorm_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let weight: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 1.0, 1.0, 1.0]).to_vec();
        let eps_bits = 1e-5f32.to_bits();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::RmsNorm {
                    size: 4,
                    epsilon: eps_bits,
                },
                &[&input, &weight],
                &mut output,
                &KernelParams::default(),
            )
            .expect("rmsnorm should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // Each value normalized: v / rms * weight
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0 / 7.5f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn cpu_attention_dispatch() {
        let backend = CpuBackend::new();
        // 1 head, seq=2, head_dim=2, heads_first=true
        // Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[1,2],[3,4]]
        let q: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0]).to_vec();
        let k: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0]).to_vec();
        let v: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Attention {
                    head_dim: 2,
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    scale: 1.0f32.to_bits(),
                    causal: false,
                    heads_first: true,
                    qk_norm: false,
                    rope: false,
                    rope_base: 0,
                    sparse_v: false,
                },
                &[&q, &k, &v],
                &mut output,
                &KernelParams::default(),
            )
            .expect("attention should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result.len(), 4);
        // With scale=1.0, Q[0]=[1,0] attends more to K[0]=[1,0] than K[1]=[0,1],
        // so output[0] should be closer to V[0]=[1,2].
        assert!(
            result[0] > 1.0,
            "first output should be > 1.0, got {}",
            result[0]
        );
    }

    #[test]
    fn cpu_group_norm_dispatch() {
        let backend = CpuBackend::new();
        // 2 channels, 2 spatial, 1 group (all channels in one group).
        // x=[1,2,3,4], weight=[1,1], bias=[0,0]
        let x: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let w: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 1.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[0.0f32, 0.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::GroupNorm {
                    num_groups: 1,
                    epsilon: 1e-5f32.to_bits(),
                },
                &[&x, &w, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("group_norm should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result.len(), 4);
        // Mean = 2.5, normalized values should be centered around 0.
        let sum: f32 = result.iter().sum();
        assert!(
            sum.abs() < 1e-3,
            "group_norm output should be centered, sum={sum}"
        );
    }

    #[test]
    fn cpu_gather_dispatch() {
        let backend = CpuBackend::new();
        // 3x2 table, gather rows 2, 0
        let data: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let indices: Vec<u8> = bytemuck::cast_slice(&[2i64, 0i64]).to_vec();
        let mut output = Vec::new();
        let mut params = KernelParams::default();
        params.u32s.push(3); // axis_dim = 3 rows
        backend
            .dispatch(
                &FloatOp::Gather {
                    dim: 0,
                    dtype: FloatDType::F32,
                },
                &[&data, &indices],
                &mut output,
                &params,
            )
            .expect("gather should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        // Row 2 = [5, 6], Row 0 = [1, 2]
        assert_eq!(result, &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn cpu_embed_dispatch() {
        let backend = CpuBackend::new();
        // 3 vocab, dim=2
        let ids: Vec<u8> = bytemuck::cast_slice(&[2u32, 0u32]).to_vec();
        let table: Vec<u8> =
            bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Embed { dim: 2, quant: 0 },
                &[&ids, &table],
                &mut output,
                &KernelParams::default(),
            )
            .expect("embed should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[50.0, 60.0, 10.0, 20.0]);
    }

    #[test]
    fn cpu_where_dispatch() {
        let backend = CpuBackend::new();
        let cond: Vec<u8> = vec![1, 0, 1];
        let x: Vec<u8> = bytemuck::cast_slice(&[10.0f32, 20.0, 30.0]).to_vec();
        let y: Vec<u8> = bytemuck::cast_slice(&[100.0f32, 200.0, 300.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Where,
                &[&cond, &x, &y],
                &mut output,
                &KernelParams::default(),
            )
            .expect("where should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[10.0, 200.0, 30.0]);
    }

    #[test]
    fn cpu_compare_dispatch() {
        let backend = CpuBackend::new();
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[2.0f32, 2.0, 1.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Less,
                &[&a, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("less should succeed");
        // 1<2=true, 2<2=false, 3<1=false
        assert_eq!(output, vec![1, 0, 0]);
    }

    #[test]
    fn cpu_byte_bool_dispatch() {
        let backend = CpuBackend::new();
        let a: Vec<u8> = vec![0xFF, 0x0F, 0x00];
        let b: Vec<u8> = vec![0xF0, 0xF0, 0xFF];
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::And,
                &[&a, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("and should succeed");
        assert_eq!(output, vec![0xF0, 0x00, 0x00]);
    }

    #[test]
    fn cpu_not_dispatch() {
        let backend = CpuBackend::new();
        let a: Vec<u8> = vec![0, 1, 0, 255];
        let mut output = Vec::new();
        backend
            .dispatch(&FloatOp::Not, &[&a], &mut output, &KernelParams::default())
            .expect("not should succeed");
        assert_eq!(output, vec![1, 0, 1, 0]);
    }

    #[test]
    fn cpu_isnan_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, f32::NAN, 0.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::IsNaN,
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("isnan should succeed");
        assert_eq!(output, vec![0, 1, 0]);
    }

    #[test]
    fn cpu_reduce_sum_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::ReduceSum { size: 3 },
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("reduce_sum should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[6.0, 15.0]);
    }

    #[test]
    fn cpu_gemm_dispatch() {
        let backend = CpuBackend::new();
        // 2x3 * 3x2 with alpha=1, beta=0, no transpose
        let a: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let b: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Gemm {
                    m: 2,
                    k: 3,
                    n: 2,
                    alpha: 1.0f32.to_bits(),
                    beta: 0.0f32.to_bits(),
                    trans_a: false,
                    trans_b: false,
                    quant_b: 0,
                },
                &[&a, &b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("gemm should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn cpu_cast_f32_to_i64() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.5, -3.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::Cast {
                    from: FloatDType::F32,
                    to: FloatDType::I64,
                },
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("cast should succeed");
        let result: &[i64] = bytemuck::cast_slice(&output);
        assert_eq!(result, &[1i64, 2, -3]);
    }

    #[test]
    fn cpu_log_softmax_dispatch() {
        let backend = CpuBackend::new();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec();
        let mut output = Vec::new();
        backend
            .dispatch(
                &FloatOp::LogSoftmax { size: 3 },
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("log_softmax should succeed");
        let result: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(result.len(), 3);
        // log_softmax values should be negative and sum of exp(result) should be ~1.
        let exp_sum: f32 = result.iter().map(|&v| v.exp()).sum();
        assert!(
            (exp_sum - 1.0).abs() < 1e-4,
            "exp of log_softmax should sum to 1, got {exp_sum}"
        );
    }
}
