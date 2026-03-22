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

use hologram_core::op::{FloatOp, OpCategory};
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions};

use crate::error::ExecResult;

use super::ComputeBackend;

/// Minimum buffer size (bytes) to dispatch to Metal.
/// GPU kernel launch overhead (~10-50µs per command buffer commit+wait)
/// means Metal only wins for large buffers. On Apple Silicon M-series,
/// the crossover point for elementwise ops is ~1M floats (4MB).
/// For matmul, Metal wins much earlier (~64KB) due to higher arithmetic
/// intensity. This threshold applies to elementwise ops only.
const METAL_MIN_BYTES: usize = 4 * 1024 * 1024; // 1M floats

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
"#;

/// Metal GPU backend for Apple Silicon / macOS.
pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
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
        ];

        let mut pipelines = HashMap::new();
        for &name in kernel_names {
            let func = library.get_function(name, None).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .ok()?;
            pipelines.insert(name, pipeline);
        }

        Some(MetalBackend {
            device,
            queue,
            pipelines,
        })
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

    /// Dispatch a unary elementwise op on the GPU.
    fn dispatch_unary(
        &self,
        pipeline: &ComputePipelineState,
        input: &[u8],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;

        // Create input buffer (shared memory — no copy on Apple Silicon).
        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer.
        let output_buf = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);

        // Count buffer.
        let count = n_floats as u32;
        let count_buf = self.device.new_buffer_with_data(
            &count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Encode + dispatch.
        let cmd_buf = self.queue.new_command_buffer();
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

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Copy output to out_buf.
        let output_ptr = output_buf.contents() as *const u8;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, byte_len) };
        out_buf.extend_from_slice(output_slice);

        Ok(())
    }

    /// Dispatch a binary elementwise op on the GPU.
    fn dispatch_binary(
        &self,
        pipeline: &ComputePipelineState,
        input_a: &[u8],
        input_b: &[u8],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
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

        let cmd_buf = self.queue.new_command_buffer();
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

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let output_ptr = output_buf.contents() as *const u8;
        let output_slice = unsafe { std::slice::from_raw_parts(output_ptr, byte_len) };
        out_buf.extend_from_slice(output_slice);

        Ok(())
    }
}

impl ComputeBackend for MetalBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        // Skip Metal for small buffers — CPU SIMD is faster.
        let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
        if input_bytes < METAL_MIN_BYTES {
            return Ok(false);
        }

        // Look up kernel name and pipeline.
        let name = match Self::kernel_name(op) {
            Some(n) => n,
            None => return Ok(false), // No Metal kernel for this op.
        };
        let pipeline = match self.pipelines.get(name) {
            Some(p) => p,
            None => return Ok(false),
        };

        match op.category() {
            OpCategory::UnaryElementwise => {
                self.dispatch_unary(pipeline, inputs[0], out_buf)?;
                Ok(true)
            }
            OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                self.dispatch_binary(pipeline, inputs[0], inputs[1], out_buf)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn dispatch_matmul(
        &self,
        _inputs: &[&[u8]],
        _m: usize,
        _k: usize,
        _n: usize,
        _out_buf: &mut Vec<u8>,
    ) -> ExecResult<bool> {
        // TODO: Metal tiled SGEMM or MPS matmul.
        Ok(false)
    }

    fn name(&self) -> &'static str {
        "metal"
    }
}
