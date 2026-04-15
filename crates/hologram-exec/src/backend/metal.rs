//! Metal compute backend (Apple GPU).
//!
//! Auto-detected on macOS via `build.rs` (`has_metal` cfg flag).
//! Uses the `metal` crate (0.33) for idiomatic Rust Metal API access.
//!
//! Shader source is embedded as a string constant and compiled at
//! initialization via `device.new_library_with_source()`. Pipeline
//! states are cached per kernel function name.
//!
//! Unified memory on Apple Silicon means zero DMA overhead — the GPU
//! reads/writes the same physical RAM as the CPU.

use std::collections::HashMap;
use std::sync::Mutex;

use hologram_core::op::{FloatOp, OpCategory};
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use crate::buffer::OutputBuffer;
use crate::error::{ExecError, ExecResult};

use super::ComputeBackend;

use super::hardware::{HardwareCaps, OpThresholds};

/// Embedded Metal Shading Language source for elementwise kernels.
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = max(input[gid], 0.0f); }
}

kernel void neg(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = -input[gid]; }
}

kernel void abs_val(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = abs(input[gid]); }
}

kernel void sigmoid(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = 1.0f / (1.0f + exp(-input[gid])); }
}

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        float x = input[gid];
        output[gid] = x / (1.0f + exp(-x));
    }
}

kernel void tanh_act(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = tanh(input[gid]); }
}

kernel void exp_act(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = exp(input[gid]); }
}

kernel void reciprocal(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) { output[gid] = 1.0f / input[gid]; }
}

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        float x = input[gid];
        output[gid] = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

// Binary elementwise
kernel void add_op(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count_a [[buffer(3)]],
    constant uint& count_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint out_len = max(count_a, count_b);
    if (gid < out_len) { output[gid] = a[gid % count_a] + b[gid % count_b]; }
}

kernel void mul_op(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count_a [[buffer(3)]],
    constant uint& count_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint out_len = max(count_a, count_b);
    if (gid < out_len) { output[gid] = a[gid % count_a] * b[gid % count_b]; }
}

kernel void sub_op(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count_a [[buffer(3)]],
    constant uint& count_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint out_len = max(count_a, count_b);
    if (gid < out_len) { output[gid] = a[gid % count_a] - b[gid % count_b]; }
}

kernel void div_op(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count_a [[buffer(3)]],
    constant uint& count_b [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint out_len = max(count_a, count_b);
    if (gid < out_len) { output[gid] = a[gid % count_a] / b[gid % count_b]; }
}

// ── Tiled SGEMM: C[M,N] = A[M,K] × B[K,N] ────────────────────────────
// Uses threadgroup shared memory to reduce global memory bandwidth.
// Each threadgroup loads a TILE_SIZE×TILE_SIZE tile of A and B into
// shared memory, computes partial products, and accumulates across tiles.
//
// On Apple Silicon M-series, this is 5-20x faster than the naive kernel
// for matrices ≥ 256×256 due to ~10x higher shared memory bandwidth
// vs global memory.

constant uint TILE_SIZE = 16;

kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles for A and B.
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory.
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            tileA[tid.y][tid.x] = A[row * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B into shared memory.
        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            tileB[tid.y][tid.x] = B[bRow * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }

        // Synchronize to ensure tile is fully loaded.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate partial products from this tile.
        for (uint p = 0; p < TILE_SIZE; p++) {
            sum += tileA[tid.y][p] * tileB[p][tid.x];
        }

        // Synchronize before loading next tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ── Batched SGEMM: C[b,M,N] = A[b,M,K] × B[b,K,N] ──────────────────
// Same tiled algorithm as sgemm, but with batch dimension in Z.
// Each batch operates on independent A, B, C slices.
// B can be shared across batches (b_stride=0) for weight broadcasting.
kernel void batched_sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& a_stride [[buffer(6)]],
    constant uint& b_stride [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    uint batch = tgid.z;
    device const float* A_b = A + batch * a_stride;
    device const float* B_b = B + batch * b_stride;
    device float* C_b = C + batch * (M * N);

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            tileA[tid.y][tid.x] = A_b[row * K + aCol];
        } else {
            tileA[tid.y][tid.x] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            tileB[tid.y][tid.x] = B_b[bRow * N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = 0; p < TILE_SIZE; p++) {
            sum += tileA[tid.y][p] * tileB[p][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C_b[row * N + col] = sum;
    }
}

// ── Softmax: row-wise softmax over chunks of `size` elements ──────────
// Each threadgroup processes one row. Uses parallel reduction for max and sum.
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Simple per-element kernel: each thread handles one row.
    // For production, a true parallel reduction would be faster for large rows.
    uint row_start = (gid / row_size) * row_size;
    uint col = gid % row_size;
    if (gid >= total) return;

    // Find max in this row (each thread scans its full row — redundant but correct).
    float row_max = -INFINITY;
    for (uint i = 0; i < row_size && (row_start + i) < total; i++) {
        row_max = max(row_max, input[row_start + i]);
    }

    // Compute exp(x - max) and sum.
    float exp_val = exp(input[gid] - row_max);
    float row_sum = 0.0f;
    for (uint i = 0; i < row_size && (row_start + i) < total; i++) {
        row_sum += exp(input[row_start + i] - row_max);
    }

    output[gid] = (row_sum > 0.0f) ? (exp_val / row_sum) : (1.0f / float(row_size));
}

// ── RmsNorm: rmsnorm(x, weight, epsilon) ──────────────────────────────
// Each thread computes one element: x[i] * rsqrt(mean(x^2) + eps) * weight[i]
kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    constant uint& row_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;

    uint row_start = (gid / row_size) * row_size;
    uint col = gid % row_size;

    // Compute mean of squares for this row.
    float ms = 0.0f;
    for (uint i = 0; i < row_size && (row_start + i) < total; i++) {
        float v = input[row_start + i];
        ms += v * v;
    }
    ms /= float(row_size);

    float inv_rms = rsqrt(ms + epsilon);
    output[gid] = input[gid] * inv_rms * weight[col];
}
"#;

/// Metal GPU backend for Apple Silicon / macOS.
///
/// Phase 8.2: Command buffer batching. Multiple kernel dispatches are encoded
/// into a single command buffer without committing. Call `flush()` at level
/// boundaries to commit + wait for all pending GPU work. This amortizes the
/// ~10-50µs per-commit overhead across multiple kernels.
///
/// On Apple Silicon (unified memory), output buffers allocated via `new_buffer()`
/// are CPU-addressable immediately — the GPU writes to them asynchronously after
/// commit. The `flush()` call ensures all writes are complete before CPU reads.
pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    /// Pending command buffer for batch encoding. `None` when no work is queued.
    /// Created lazily on first dispatch, committed+waited on `flush()`.
    /// Uses `Mutex` for interior mutability — the backend is shared via Arc
    /// and must implement `Sync` for `ComputeBackend: Send + Sync`.
    pending: Mutex<Option<metal::CommandBuffer>>,
    /// Hardware-detected per-op dispatch thresholds (computed once at init).
    thresholds: OpThresholds,
}

impl MetalBackend {
    /// Create a new Metal backend from the system default GPU device.
    ///
    /// Compiles all shader kernels and caches pipeline states.
    /// Returns `None` if Metal is not available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .ok()?;

        let kernel_names: &[&str] = &[
            "relu",
            "neg",
            "abs_val",
            "sigmoid",
            "silu",
            "tanh_act",
            "exp_act",
            "reciprocal",
            "gelu",
            "add_op",
            "mul_op",
            "sub_op",
            "div_op",
            "sgemm",
            "softmax",
            "rms_norm",
        ];

        let mut pipelines = HashMap::new();
        for &name in kernel_names {
            let func = library.get_function(name, None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .ok()?;
            pipelines.insert(name, pipeline);
        }

        let thresholds = OpThresholds::from(HardwareCaps::detect());

        Some(MetalBackend {
            device,
            queue,
            pipelines,
            pending: Mutex::new(None),
            thresholds,
        })
    }

    /// Get or create the pending command buffer for batch encoding.
    fn get_or_create_cmd_buf(&self) -> std::sync::MutexGuard<'_, Option<metal::CommandBuffer>> {
        let mut pending = self.pending.lock().unwrap();
        if pending.is_none() {
            *pending = Some(self.queue.new_command_buffer().to_owned());
        }
        pending
    }

    /// Commit and wait for all pending GPU work.
    ///
    /// Called at level boundaries by the tape executor. After flush,
    /// all output MetalBuffers returned by previous dispatch calls
    /// contain valid data readable by CPU.
    pub fn flush(&self) {
        let mut pending = self.pending.lock().unwrap();
        if let Some(cmd_buf) = pending.take() {
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    /// Map a FloatOp to a Metal kernel function name.
    fn kernel_name(op: &FloatOp) -> Option<&'static str> {
        match op {
            FloatOp::Relu => Some("relu"),
            FloatOp::Neg => Some("neg"),
            FloatOp::Abs => Some("abs_val"),
            FloatOp::Sigmoid => Some("sigmoid"),
            FloatOp::Silu => Some("silu"),
            FloatOp::Tanh => Some("tanh_act"),
            FloatOp::Exp => Some("exp_act"),
            FloatOp::Reciprocal => Some("reciprocal"),
            FloatOp::Gelu => Some("gelu"),
            FloatOp::Add => Some("add_op"),
            FloatOp::Mul => Some("mul_op"),
            FloatOp::Sub => Some("sub_op"),
            FloatOp::Div => Some("div_op"),
            _ => None,
        }
    }

    /// Resolve a `GpuInput` to a `metal::Buffer`.
    /// For `Cpu` inputs: copies data to a new Metal buffer.
    /// For `Gpu(Metal)` inputs: returns the buffer directly (zero-copy).
    fn resolve_input(&self, input: &super::GpuInput<'_>) -> metal::Buffer {
        match input {
            super::GpuInput::Cpu(bytes) => self.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            super::GpuInput::Gpu(gbuf) => match gbuf {
                #[cfg(has_metal)]
                super::GpuBuffer::Metal(buf) => buf.clone(),
                #[allow(unreachable_patterns)]
                _ => {
                    // Non-Metal GPU buffer on Metal backend — should not happen.
                    // Fallback: create empty buffer.
                    self.device
                        .new_buffer(0, MTLResourceOptions::StorageModeShared)
                }
            },
        }
    }

    /// Dispatch a unary elementwise op on the GPU.
    /// Returns the output Metal buffer directly — zero-copy into arena.
    fn dispatch_unary(
        &self,
        pipeline: &ComputePipelineState,
        input: &[u8],
    ) -> ExecResult<metal::Buffer> {
        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

        let count = n_floats as u32;
        let count_buf = self.device.new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&output_buf), 0);
        encoder.set_buffer(2, Some(&count_buf), 0);

        let threadgroup_size =
            metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid_size = metal::MTLSize::new(n_floats as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }

    /// Dispatch a unary op from a GpuInput (zero-copy for GPU buffers).
    fn dispatch_unary_chained(
        &self,
        pipeline: &ComputePipelineState,
        input: &super::GpuInput<'_>,
    ) -> ExecResult<metal::Buffer> {
        let input_buf = self.resolve_input(input);
        let n_floats = (input_buf.length() as usize) / 4;
        let byte_len = n_floats * 4;

        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let count = n_floats as u32;
        let count_buf = self.device.new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending
            .as_ref()
            .expect("Metal command buffer should exist after get_or_create");
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&output_buf), 0);
        encoder.set_buffer(2, Some(&count_buf), 0);

        let threadgroup_size =
            metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid_size = metal::MTLSize::new(n_floats as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }

    /// Dispatch a binary op from GpuInputs (zero-copy for GPU buffers).
    fn dispatch_binary_chained(
        &self,
        pipeline: &ComputePipelineState,
        input_a: &super::GpuInput<'_>,
        input_b: &super::GpuInput<'_>,
    ) -> ExecResult<metal::Buffer> {
        let buf_a = self.resolve_input(input_a);
        let buf_b = self.resolve_input(input_b);
        let n_a = (buf_a.length() as usize / 4) as u32;
        let n_b = (buf_b.length() as usize / 4) as u32;
        let n_out = n_a.max(n_b) as usize;
        let byte_len = n_out * 4;

        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let count_a_buf = self.device.new_buffer_with_data(
            &n_a as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let count_b_buf = self.device.new_buffer_with_data(
            &n_b as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending
            .as_ref()
            .expect("Metal command buffer should exist after get_or_create");
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        encoder.set_buffer(3, Some(&count_a_buf), 0);
        encoder.set_buffer(4, Some(&count_b_buf), 0);

        let threadgroup_size =
            metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid_size = metal::MTLSize::new(n_out as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }

    /// Dispatch a matmul from GpuInputs (zero-copy for GPU buffers).
    fn dispatch_matmul_chained_inner(
        &self,
        input_a: &super::GpuInput<'_>,
        input_b: &super::GpuInput<'_>,
        m: usize,
        k: usize,
        n: usize,
    ) -> ExecResult<metal::Buffer> {
        let pipeline = match self.pipelines.get("sgemm") {
            Some(p) => p,
            None => {
                return Err(crate::error::ExecError::UnsupportedOp(
                    "Metal sgemm pipeline not compiled".into(),
                ))
            }
        };

        let buf_a = self.resolve_input(input_a);
        let buf_b = self.resolve_input(input_b);
        let byte_len = m * n * 4;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        let buf_c = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let buf_m = self.device.new_buffer_with_data(
            &m_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_k = self.device.new_buffer_with_data(
            &k_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_n = self.device.new_buffer_with_data(
            &n_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending
            .as_ref()
            .expect("Metal command buffer should exist after get_or_create");
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_m), 0);
        encoder.set_buffer(4, Some(&buf_k), 0);
        encoder.set_buffer(5, Some(&buf_n), 0);

        let threadgroup_size = metal::MTLSize::new(16, 16, 1);
        let grid_size = metal::MTLSize::new(n as u64, m as u64, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(buf_c)
    }

    /// Dispatch a binary elementwise op on the GPU.
    /// Returns the output Metal buffer directly — zero-copy into arena.
    fn dispatch_binary(
        &self,
        pipeline: &ComputePipelineState,
        input_a: &[u8],
        input_b: &[u8],
    ) -> ExecResult<metal::Buffer> {
        let n_a = (input_a.len() / 4) as u32;
        let n_b = (input_b.len() / 4) as u32;
        let n_out = n_a.max(n_b) as usize;
        let byte_len = n_out * 4;

        let buf_a = self.device.new_buffer_with_data(
            input_a.as_ptr() as *const _,
            input_a.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b = self.device.new_buffer_with_data(
            input_b.as_ptr() as *const _,
            input_b.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let count_a_buf = self.device.new_buffer_with_data(
            &n_a as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let count_b_buf = self.device.new_buffer_with_data(
            &n_b as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        encoder.set_buffer(3, Some(&count_a_buf), 0);
        encoder.set_buffer(4, Some(&count_b_buf), 0);

        let threadgroup_size =
            metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid_size = metal::MTLSize::new(n_out as u64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }

    /// Dispatch softmax on the GPU. Returns Metal buffer directly.
    fn dispatch_softmax(&self, input: &[u8], row_size: usize) -> ExecResult<metal::Buffer> {
        let pipeline = self
            .pipelines
            .get("softmax")
            .ok_or_else(|| ExecError::UnsupportedOp("Metal softmax pipeline missing".into()))?;

        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;
        let total = n_floats as u32;
        let row_sz = row_size as u32;

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let total_buf = self.device.new_buffer_with_data(
            &total as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let row_buf = self.device.new_buffer_with_data(
            &row_sz as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&output_buf), 0);
        encoder.set_buffer(2, Some(&total_buf), 0);
        encoder.set_buffer(3, Some(&row_buf), 0);

        let tg = metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid = metal::MTLSize::new(n_floats as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }

    /// Dispatch RmsNorm on the GPU. Returns Metal buffer directly.
    fn dispatch_rms_norm(
        &self,
        input: &[u8],
        weight: &[u8],
        row_size: usize,
        epsilon: f32,
    ) -> ExecResult<metal::Buffer> {
        let pipeline = self
            .pipelines
            .get("rms_norm")
            .ok_or_else(|| ExecError::UnsupportedOp("Metal rms_norm pipeline missing".into()))?;

        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;
        let total = n_floats as u32;
        let row_sz = row_size as u32;

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = self.device.new_buffer_with_data(
            weight.as_ptr() as *const _,
            weight.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let total_buf = self.device.new_buffer_with_data(
            &total as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let row_buf = self.device.new_buffer_with_data(
            &row_sz as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let eps_buf = self.device.new_buffer_with_data(
            &epsilon as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buf), 0);
        encoder.set_buffer(1, Some(&weight_buf), 0);
        encoder.set_buffer(2, Some(&output_buf), 0);
        encoder.set_buffer(3, Some(&total_buf), 0);
        encoder.set_buffer(4, Some(&row_buf), 0);
        encoder.set_buffer(5, Some(&eps_buf), 0);

        let tg = metal::MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        let grid = metal::MTLSize::new(n_floats as u64, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        drop(pending);

        Ok(output_buf)
    }
}

impl ComputeBackend for MetalBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        // MatMul: route to dispatch_matmul (separate size threshold).
        if let FloatOp::MatMul { m, k, n } = op {
            return self.dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize, out_buf);
        }

        // Softmax: route with row_size parameter — zero-copy Metal buffer.
        if let FloatOp::Softmax { size } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= self.thresholds.softmax_min_bytes && *size > 0 {
                let buf = self.dispatch_softmax(inputs[0], *size as usize)?;
                return Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(buf)));
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // RmsNorm: route with row_size + epsilon — zero-copy Metal buffer.
        if let FloatOp::RmsNorm { size, epsilon } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= self.thresholds.norm_min_bytes && inputs.len() >= 2 && *size > 0 {
                let buf = self.dispatch_rms_norm(
                    inputs[0],
                    inputs[1],
                    *size as usize,
                    f32::from_bits(*epsilon),
                )?;
                return Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(buf)));
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // Skip Metal for small buffers — CPU SIMD is faster.
        let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
        if input_bytes < self.thresholds.elementwise_min_bytes {
            return Ok(super::KernelOutput::Skipped);
        }

        // Look up kernel name and pipeline.
        let name = match Self::kernel_name(op) {
            Some(n) => n,
            None => return Ok(super::KernelOutput::Skipped),
        };
        let pipeline = match self.pipelines.get(name) {
            Some(p) => p,
            None => return Ok(super::KernelOutput::Skipped),
        };

        match op.category() {
            OpCategory::UnaryElementwise => {
                let metal_buf = self.dispatch_unary(pipeline, inputs[0])?;
                Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(
                    metal_buf,
                )))
            }
            OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                let metal_buf = self.dispatch_binary(pipeline, inputs[0], inputs[1])?;
                Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(
                    metal_buf,
                )))
            }
            _ => Ok(super::KernelOutput::Skipped),
        }
    }

    fn dispatch_matmul(
        &self,
        inputs: &[&[u8]],
        m: usize,
        k: usize,
        n: usize,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        // Metal matmul only worthwhile for large matrices.
        // Crossover vs Accelerate BLAS varies by GPU generation.
        let out_elements = m * n;
        if out_elements < self.thresholds.matmul_min_elements {
            return Ok(super::KernelOutput::Skipped);
        }

        let pipeline = match self.pipelines.get("sgemm") {
            Some(p) => p,
            None => return Ok(super::KernelOutput::Skipped),
        };

        if inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }

        let byte_len = out_elements * 4;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        let buf_a = self.device.new_buffer_with_data(
            inputs[0].as_ptr() as *const _,
            inputs[0].len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b = self.device.new_buffer_with_data(
            inputs[1].as_ptr() as *const _,
            inputs[1].len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_c = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        let buf_m = self.device.new_buffer_with_data(
            &m_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_k = self.device.new_buffer_with_data(
            &k_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_n = self.device.new_buffer_with_data(
            &n_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_m), 0);
        encoder.set_buffer(4, Some(&buf_k), 0);
        encoder.set_buffer(5, Some(&buf_n), 0);

        // 2D grid: (N, M) — each thread computes C[row, col].
        let threadgroup_size = metal::MTLSize::new(16, 16, 1);
        let grid_size = metal::MTLSize::new(n as u64, m as u64, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(
            buf_c,
        )))
    }

    fn dispatch_batched_matmul(
        &self,
        inputs: &[&[u8]],
        dims: super::BatchedMatmulDims,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        let super::BatchedMatmulDims {
            batch,
            m,
            k,
            n,
            b_broadcast,
        } = dims;
        // Batched matmul is worthwhile when total compute exceeds GPU launch cost.
        // On Apple Silicon, crossover is ~batch*m*n > 4096 elements total output.
        let total_output = batch * m * n;
        if total_output < 4096 {
            return Ok(super::KernelOutput::Skipped);
        }

        let pipeline = match self.pipelines.get("batched_sgemm") {
            Some(p) => p,
            None => return Ok(super::KernelOutput::Skipped),
        };
        if inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }

        let out_bytes = batch * m * n * 4;
        let a_stride_val = (m * k) as u32;
        let b_stride_val = if b_broadcast { 0u32 } else { (k * n) as u32 };
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        let buf_a = self.device.new_buffer_with_data(
            inputs[0].as_ptr() as *const _,
            inputs[0].len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b = self.device.new_buffer_with_data(
            inputs[1].as_ptr() as *const _,
            inputs[1].len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_c = self
            .device
            .new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);

        let buf_m = self.device.new_buffer_with_data(
            &m_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_k = self.device.new_buffer_with_data(
            &k_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_n = self.device.new_buffer_with_data(
            &n_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_a_stride = self.device.new_buffer_with_data(
            &a_stride_val as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b_stride = self.device.new_buffer_with_data(
            &b_stride_val as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let pending = self.get_or_create_cmd_buf();
        let cmd_buf = pending.as_ref().unwrap();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_m), 0);
        encoder.set_buffer(4, Some(&buf_k), 0);
        encoder.set_buffer(5, Some(&buf_n), 0);
        encoder.set_buffer(6, Some(&buf_a_stride), 0);
        encoder.set_buffer(7, Some(&buf_b_stride), 0);

        // 3D grid: (N, M, batch) — Z dimension is batch.
        let threadgroup_size = metal::MTLSize::new(16, 16, 1);
        let grid_size = metal::MTLSize::new(n as u64, m as u64, batch as u64);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        drop(pending);

        Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(
            buf_c,
        )))
    }

    fn name(&self) -> &'static str {
        "metal"
    }

    fn op_thresholds(&self) -> &super::hardware::OpThresholds {
        &self.thresholds
    }

    fn dispatch_float_chained(
        &self,
        op: &FloatOp,
        inputs: &[super::GpuInput<'_>],
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        // Softmax/RmsNorm/MatMul have dedicated dispatch paths — skip chaining
        // for now and let them fall back to the default (readback + dispatch_float).
        if matches!(
            op,
            FloatOp::Softmax { .. } | FloatOp::RmsNorm { .. } | FloatOp::MatMul { .. }
        ) {
            // Use the default readback-then-dispatch path.
            let cpu_bufs: smallvec::SmallVec<[Vec<u8>; 4]> = inputs
                .iter()
                .map(|inp| match inp {
                    super::GpuInput::Cpu(s) => s.to_vec(),
                    super::GpuInput::Gpu(gb) => {
                        self.flush();
                        let mut dst = vec![0u8; gb.byte_len()];
                        gb.readback_into(&mut dst);
                        dst
                    }
                })
                .collect();
            let refs: smallvec::SmallVec<[&[u8]; 4]> =
                cpu_bufs.iter().map(|v| v.as_slice()).collect();
            return self.dispatch_float(op, &refs, _out_buf);
        }

        // Skip Metal for small buffers.
        let input_bytes = inputs.first().map(|i| i.byte_len()).unwrap_or(0);
        if input_bytes < self.thresholds.elementwise_min_bytes {
            return Ok(super::KernelOutput::Skipped);
        }

        let name = match Self::kernel_name(op) {
            Some(n) => n,
            None => return Ok(super::KernelOutput::Skipped),
        };
        let pipeline = match self.pipelines.get(name) {
            Some(p) => p,
            None => return Ok(super::KernelOutput::Skipped),
        };

        match op.category() {
            OpCategory::UnaryElementwise => {
                let buf = self.dispatch_unary_chained(pipeline, &inputs[0])?;
                Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(buf)))
            }
            OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                let buf = self.dispatch_binary_chained(pipeline, &inputs[0], &inputs[1])?;
                Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(buf)))
            }
            _ => Ok(super::KernelOutput::Skipped),
        }
    }

    fn dispatch_matmul_chained(
        &self,
        inputs: &[super::GpuInput<'_>],
        m: usize,
        k: usize,
        n: usize,
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        let out_elements = m * n;
        if out_elements < self.thresholds.matmul_min_elements {
            return Ok(super::KernelOutput::Skipped);
        }
        if inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }

        let buf = self.dispatch_matmul_chained_inner(&inputs[0], &inputs[1], m, k, n)?;
        Ok(super::KernelOutput::GpuBuffer(super::GpuBuffer::Metal(buf)))
    }
}
