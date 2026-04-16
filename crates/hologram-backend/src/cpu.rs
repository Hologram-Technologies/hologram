//! CPU compute backend.
//!
//! Uses SIMD (NEON on ARM, AVX2 on x86_64) and Accelerate BLAS on macOS.
//! This is the reference implementation — all other backends must produce
//! identical results for the same inputs.

use crate::{BackendError, ComputeBackend, ComputeMemory, KernelParams, Result};
use hologram_core::op::FloatOp;

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
                    for i in 0..actual_m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                sum += a[i * k + p] * b[p * n + j];
                            }
                            out[i * n + j] = sum;
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
