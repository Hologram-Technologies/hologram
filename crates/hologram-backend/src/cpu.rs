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
        _inputs: &[&Vec<u8>],
        _output: &mut Vec<u8>,
        _params: &KernelParams,
    ) -> Result<usize> {
        // TODO: wire to existing hologram-exec float_dispatch functions.
        // For now, return Unsupported — the existing tape executor handles
        // CPU dispatch. This will be filled in as we migrate ops.
        Err(BackendError::Unsupported(format!(
            "CPU dispatch for {:?} not yet migrated",
            op
        )))
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
}
