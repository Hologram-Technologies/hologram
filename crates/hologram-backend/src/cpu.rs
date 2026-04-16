//! CPU compute backend.
//!
//! Uses SIMD (NEON on ARM, AVX2 on x86_64) and Accelerate BLAS on macOS.
//! This is the reference implementation — all other backends must produce
//! identical results for the same inputs.

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
            _ => {
                // MatMul: naive implementation (Accelerate BLAS to be wired later).
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
                    } else if k > 0 {
                        a.len() / k
                    } else {
                        0
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
                    let ic = if spatial_in > 0 {
                        data.len() / spatial_in
                    } else {
                        0
                    };
                    let oc = if ic > 0 && kh > 0 && kw > 0 {
                        weight.len() / (ic / g * kh * kw)
                    } else {
                        0
                    };

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
}
