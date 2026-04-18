//! Metal compute backend (Apple GPU).
//!
//! Implements `MetalMemory` and `MetalBackend` for device-native execution
//! on Apple Silicon. All tensor data lives in Metal shared-memory buffers.
//! All computation dispatches as Metal compute shaders.
//!
//! Shader source is compiled from MSL at initialization. Pipeline states
//! are cached per kernel function name for zero-overhead dispatch.

use std::collections::HashMap;
use std::sync::Mutex;

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::{BackendError, ComputeBackend, ComputeMemory, KernelParams, Result};
use hologram_core::op::FloatOp;

/// MSL shader source — compiled at backend initialization.
const SHADER_SOURCE: &str = include_str!("kernels/metal.msl");

/// Metal device memory: tensors are `metal::Buffer` in shared GPU/CPU memory.
pub struct MetalMemory {
    device: Device,
}

impl MetalMemory {
    /// Create a new MetalMemory for the system default GPU.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        Some(Self { device })
    }
}

impl ComputeMemory for MetalMemory {
    type Buffer = metal::Buffer;

    fn alloc(&self, byte_len: usize) -> metal::Buffer {
        self.device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared)
    }

    fn upload(&self, data: &[u8]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn download(&self, buf: &metal::Buffer) -> Vec<u8> {
        let ptr = buf.contents() as *const u8;
        let len = buf.length() as usize;
        // SAFETY: StorageModeShared buffers are CPU-readable after flush.
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn alias(&self, buf: &metal::Buffer) -> metal::Buffer {
        // Metal buffers are reference-counted. clone() increments the refcount.
        buf.clone()
    }

    fn byte_len(&self, buf: &metal::Buffer) -> usize {
        buf.length() as usize
    }

    fn mmap(&self, data: &[u8]) -> Option<metal::Buffer> {
        // Metal SharedStorage buffers backed by the mmap'd region.
        // The OS shares the same physical pages between CPU and GPU.
        Some(self.upload(data))
    }

    fn evict(&self, buf: &mut metal::Buffer) {
        // Replace with a zero-length buffer to release the Metal allocation.
        *buf = self
            .device
            .new_buffer(0, MTLResourceOptions::StorageModeShared);
    }
}

/// Metal compute backend: dispatches ALL ops as Metal compute shaders.
pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    pending: Mutex<Option<metal::CommandBuffer>>,
    /// Ring LUT tables stored on device.
    ring_tables: Vec<metal::Buffer>,
}

impl MetalBackend {
    /// Create a new Metal backend, compiling all shader kernels.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = match device.new_library_with_source(SHADER_SOURCE, &options) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("[hologram-backend] Metal shader compilation failed: {e}");
                return None;
            }
        };

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
            "erf_act",
            "add_op",
            "mul_op",
            "sub_op",
            "div_op",
            "sgemm",
            "batched_sgemm",
            "im2col",
            "softmax",
            "rms_norm",
            "layer_norm",
            "instance_norm",
            "transpose_4d",
            "slice_copy",
            "concat_copy",
            "resize_nearest",
            "ring_lut",
            "ring_binary_lut",
            "sin_act",
            "cos_act",
            "log_act",
            "sqrt_act",
        ];

        let mut pipelines = HashMap::new();
        for &name in kernel_names {
            let func = match library.get_function(name, None) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[hologram-backend] Metal kernel '{name}' not found: {e}");
                    return None;
                }
            };
            let pipeline = match device.new_compute_pipeline_state_with_function(&func) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[hologram-backend] Metal pipeline for '{name}' failed: {e}");
                    return None;
                }
            };
            pipelines.insert(name, pipeline);
        }

        Some(MetalBackend {
            device,
            queue,
            pipelines,
            pending: Mutex::new(None),
            ring_tables: Vec::new(),
        })
    }

    /// Get or create the pending command buffer for batch encoding.
    fn get_or_create_cmd_buf(&self) -> std::sync::MutexGuard<'_, Option<metal::CommandBuffer>> {
        let mut pending = self
            .pending
            .lock()
            .expect("Metal pending command buffer mutex should not be poisoned");
        if pending.is_none() {
            *pending = Some(self.queue.new_command_buffer().to_owned());
        }
        pending
    }

    /// Create a small uniform buffer containing a single u32 value.
    fn u32_buf(&self, v: u32) -> metal::Buffer {
        self.device.new_buffer_with_data(
            &v as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a small uniform buffer containing a single f32 value.
    pub fn f32_buf(&self, v: f32) -> metal::Buffer {
        self.device.new_buffer_with_data(
            &v as *const f32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Dispatch a unary elementwise kernel.
    fn dispatch_unary(
        &self,
        pipeline: &ComputePipelineState,
        input: &metal::Buffer,
    ) -> metal::Buffer {
        let n_floats = input.length() as usize / 4;
        let output = self
            .device
            .new_buffer(input.length(), MTLResourceOptions::StorageModeShared);
        let count_buf = self.u32_buf(n_floats as u32);

        let pending = self.get_or_create_cmd_buf();
        let cmd = pending
            .as_ref()
            .expect("Metal command buffer for unary dispatch");
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(&output), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        let tg = MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        enc.dispatch_threads(MTLSize::new(n_floats as u64, 1, 1), tg);
        enc.end_encoding();
        drop(pending);

        output
    }

    /// Dispatch a binary elementwise kernel.
    fn dispatch_binary(
        &self,
        pipeline: &ComputePipelineState,
        a: &metal::Buffer,
        b: &metal::Buffer,
    ) -> metal::Buffer {
        let n_a = (a.length() as usize / 4) as u32;
        let n_b = (b.length() as usize / 4) as u32;
        let n_out = n_a.max(n_b) as usize;
        let output = self
            .device
            .new_buffer((n_out * 4) as u64, MTLResourceOptions::StorageModeShared);

        let pending = self.get_or_create_cmd_buf();
        let cmd = pending
            .as_ref()
            .expect("Metal command buffer for binary dispatch");
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(&output), 0);
        enc.set_buffer(3, Some(&self.u32_buf(n_a)), 0);
        enc.set_buffer(4, Some(&self.u32_buf(n_b)), 0);
        let tg = MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        enc.dispatch_threads(MTLSize::new(n_out as u64, 1, 1), tg);
        enc.end_encoding();
        drop(pending);

        output
    }

    /// Map a FloatOp to its MSL kernel function name.
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
            FloatOp::Erf => Some("erf_act"),
            FloatOp::Sin => Some("sin_act"),
            FloatOp::Cos => Some("cos_act"),
            FloatOp::Log => Some("log_act"),
            FloatOp::Sqrt => Some("sqrt_act"),
            FloatOp::Add => Some("add_op"),
            FloatOp::Mul => Some("mul_op"),
            FloatOp::Sub => Some("sub_op"),
            FloatOp::Div => Some("div_op"),
            _ => None,
        }
    }
}

impl ComputeBackend<MetalMemory> for MetalBackend {
    fn dispatch(
        &self,
        op: &FloatOp,
        inputs: &[&metal::Buffer],
        output: &mut metal::Buffer,
        _params: &KernelParams,
    ) -> Result<usize> {
        use hologram_core::op::OpCategory;

        // MatMul: extract dims from params.
        if let FloatOp::MatMul { m, k, n } = op {
            let pipeline = self
                .pipelines
                .get("sgemm")
                .ok_or_else(|| BackendError::Device("sgemm pipeline not compiled".into()))?;
            if inputs.len() < 2 {
                return Err(BackendError::Shape("matmul requires 2 inputs".into()));
            }
            let m = *m as usize;
            let k = *k as usize;
            let n = *n as usize;
            let byte_len = m * n * 4;

            let out = self
                .device
                .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
            let pending = self.get_or_create_cmd_buf();
            let cmd = pending.as_ref().expect("Metal cmd buf for matmul");
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(inputs[0]), 0);
            enc.set_buffer(1, Some(inputs[1]), 0);
            enc.set_buffer(2, Some(&out), 0);
            // Bind dimension buffers to locals so they outlive the encoder.
            let m_buf = self.u32_buf(m as u32);
            let k_buf = self.u32_buf(k as u32);
            let n_buf = self.u32_buf(n as u32);
            enc.set_buffer(3, Some(&m_buf), 0);
            enc.set_buffer(4, Some(&k_buf), 0);
            enc.set_buffer(5, Some(&n_buf), 0);
            let tg = MTLSize::new(16, 16, 1);
            // Use dispatch_threadgroups (not dispatch_threads) for correct
            // threadgroup barrier behavior with the tiled sgemm kernel.
            let grid = MTLSize::new((n as u64).div_ceil(16), (m as u64).div_ceil(16), 1);
            enc.dispatch_thread_groups(grid, tg);
            enc.end_encoding();
            drop(pending);

            *output = out;
            return Ok(byte_len);
        }

        // Elementwise ops: look up kernel by name.
        if let Some(name) = Self::kernel_name(op) {
            let pipeline = self
                .pipelines
                .get(name)
                .ok_or_else(|| BackendError::Device(format!("{name} pipeline not compiled")))?;

            match op.category() {
                OpCategory::UnaryElementwise if !inputs.is_empty() => {
                    *output = self.dispatch_unary(pipeline, inputs[0]);
                    return Ok(output.length() as usize);
                }
                OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                    *output = self.dispatch_binary(pipeline, inputs[0], inputs[1]);
                    return Ok(output.length() as usize);
                }
                _ => {}
            }
        }

        // Softmax: row-wise normalization.
        if let FloatOp::Softmax { size } = op {
            if let Some(pipeline) = self.pipelines.get("softmax") {
                if !inputs.is_empty() && *size > 0 {
                    let input_buf = inputs[0];
                    let n_floats = (input_buf.length() as usize) / 4;
                    let out = self
                        .device
                        .new_buffer(input_buf.length(), MTLResourceOptions::StorageModeShared);
                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for softmax");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(input_buf), 0);
                    enc.set_buffer(1, Some(&out), 0);
                    enc.set_buffer(2, Some(&self.u32_buf(n_floats as u32)), 0);
                    enc.set_buffer(3, Some(&self.u32_buf(*size)), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(n_floats as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // RmsNorm: input + weight → normalized output.
        if let FloatOp::RmsNorm { size, epsilon } = op {
            if let Some(pipeline) = self.pipelines.get("rms_norm") {
                if inputs.len() >= 2 && *size > 0 {
                    let total = (inputs[0].length() as usize / 4) as u32;
                    let out = self
                        .device
                        .new_buffer(inputs[0].length(), MTLResourceOptions::StorageModeShared);
                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for rmsnorm");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(inputs[0]), 0);
                    enc.set_buffer(1, Some(inputs[1]), 0);
                    enc.set_buffer(2, Some(&out), 0);
                    enc.set_buffer(3, Some(&self.u32_buf(total)), 0);
                    enc.set_buffer(4, Some(&self.u32_buf(*size)), 0);
                    enc.set_buffer(5, Some(&self.f32_buf(f32::from_bits(*epsilon))), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // LayerNorm: input + weight + bias → normalized output.
        if let FloatOp::LayerNorm { size, epsilon } = op {
            if let Some(pipeline) = self.pipelines.get("layer_norm") {
                if inputs.len() >= 3 && *size > 0 {
                    let total = (inputs[0].length() as usize / 4) as u32;
                    let out = self
                        .device
                        .new_buffer(inputs[0].length(), MTLResourceOptions::StorageModeShared);
                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for layernorm");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(inputs[0]), 0);
                    enc.set_buffer(1, Some(inputs[1]), 0);
                    enc.set_buffer(2, Some(inputs[2]), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    enc.set_buffer(4, Some(&self.u32_buf(total)), 0);
                    enc.set_buffer(5, Some(&self.u32_buf(*size)), 0);
                    enc.set_buffer(6, Some(&self.f32_buf(f32::from_bits(*epsilon))), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // InstanceNorm: input + scale + bias → normalized output.
        if let FloatOp::InstanceNorm { size, epsilon } = op {
            if let Some(pipeline) = self.pipelines.get("instance_norm") {
                if inputs.len() >= 3 && *size > 0 {
                    let total = (inputs[0].length() as usize / 4) as u32;
                    let out = self
                        .device
                        .new_buffer(inputs[0].length(), MTLResourceOptions::StorageModeShared);
                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for instancenorm");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(inputs[0]), 0);
                    enc.set_buffer(1, Some(inputs[1]), 0);
                    enc.set_buffer(2, Some(inputs[2]), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    enc.set_buffer(4, Some(&self.u32_buf(total)), 0);
                    enc.set_buffer(5, Some(&self.u32_buf(*size)), 0);
                    enc.set_buffer(6, Some(&self.f32_buf(f32::from_bits(*epsilon))), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // Transpose: 4D permutation.
        if let FloatOp::Transpose { perm, ndim } = op {
            if let Some(pipeline) = self.pipelines.get("transpose_4d") {
                if !inputs.is_empty() {
                    let n = (*ndim as usize).min(4);
                    let input_buf = inputs[0];
                    let n_floats = input_buf.length() as usize / 4;
                    let out = self
                        .device
                        .new_buffer(input_buf.length(), MTLResourceOptions::StorageModeShared);

                    // Shape from params (u32s[0..ndim]).
                    let shape = [
                        _params.u32s.first().copied().unwrap_or(1),
                        _params.u32s.get(1).copied().unwrap_or(1),
                        _params.u32s.get(2).copied().unwrap_or(1),
                        _params.u32s.get(3).copied().unwrap_or(1),
                    ];
                    let perm_u32 = [
                        perm[0] as u32,
                        if n > 1 { perm[1] as u32 } else { 1 },
                        if n > 2 { perm[2] as u32 } else { 2 },
                        if n > 3 { perm[3] as u32 } else { 3 },
                    ];

                    let shape_buf = self.device.new_buffer_with_data(
                        shape.as_ptr() as *const _,
                        16,
                        MTLResourceOptions::StorageModeShared,
                    );
                    let perm_buf = self.device.new_buffer_with_data(
                        perm_u32.as_ptr() as *const _,
                        16,
                        MTLResourceOptions::StorageModeShared,
                    );

                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for transpose");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(input_buf), 0);
                    enc.set_buffer(1, Some(&out), 0);
                    enc.set_buffer(2, Some(&self.u32_buf(n_floats as u32)), 0);
                    enc.set_buffer(3, Some(&shape_buf), 0);
                    enc.set_buffer(4, Some(&perm_buf), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(n_floats as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // Slice: contiguous sub-range copy.
        if let FloatOp::Slice {
            start,
            end,
            axis_size,
            ..
        } = op
        {
            if let Some(pipeline) = self.pipelines.get("slice_copy") {
                if !inputs.is_empty() {
                    let s = *start as usize;
                    let e = *end as usize;
                    let ax = *axis_size as usize;
                    let total_floats = inputs[0].length() as usize / 4;
                    let stride = if ax > 0 { total_floats / ax } else { 1 };
                    let src_offset = s * stride;
                    let count = (e - s) * stride;

                    let out = self
                        .device
                        .new_buffer((count * 4) as u64, MTLResourceOptions::StorageModeShared);
                    let pending = self.get_or_create_cmd_buf();
                    let cmd = pending.as_ref().expect("Metal cmd buf for slice");
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(inputs[0]), 0);
                    enc.set_buffer(1, Some(&out), 0);
                    enc.set_buffer(2, Some(&self.u32_buf(count as u32)), 0);
                    enc.set_buffer(3, Some(&self.u32_buf(src_offset as u32)), 0);
                    let tg =
                        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
                    enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), tg);
                    enc.end_encoding();
                    drop(pending);
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // Concat: combine two inputs.
        if matches!(op, FloatOp::Concat { .. }) {
            if let Some(pipeline) = self.pipelines.get("concat_copy") {
                if inputs.len() >= 2 {
                    let len_a = inputs[0].length() as usize / 4;
                    let len_b = inputs[1].length() as usize / 4;
                    let total = len_a + len_b;
                    let out = self
                        .device
                        .new_buffer((total * 4) as u64, MTLResourceOptions::StorageModeShared);
                    // Copy A at offset 0.
                    {
                        let pending = self.get_or_create_cmd_buf();
                        let cmd = pending.as_ref().expect("Metal cmd buf for concat A");
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pipeline);
                        enc.set_buffer(0, Some(inputs[0]), 0);
                        enc.set_buffer(1, Some(&out), 0);
                        enc.set_buffer(2, Some(&self.u32_buf(len_a as u32)), 0);
                        enc.set_buffer(3, Some(&self.u32_buf(0)), 0);
                        let tg = MTLSize::new(
                            pipeline.max_total_threads_per_threadgroup().min(256),
                            1,
                            1,
                        );
                        enc.dispatch_threads(MTLSize::new(len_a as u64, 1, 1), tg);
                        enc.end_encoding();
                        drop(pending);
                    }
                    // Copy B at offset len_a.
                    {
                        let pending = self.get_or_create_cmd_buf();
                        let cmd = pending.as_ref().expect("Metal cmd buf for concat B");
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pipeline);
                        enc.set_buffer(0, Some(inputs[1]), 0);
                        enc.set_buffer(1, Some(&out), 0);
                        enc.set_buffer(2, Some(&self.u32_buf(len_b as u32)), 0);
                        enc.set_buffer(3, Some(&self.u32_buf(len_a as u32)), 0);
                        let tg = MTLSize::new(
                            pipeline.max_total_threads_per_threadgroup().min(256),
                            1,
                            1,
                        );
                        enc.dispatch_threads(MTLSize::new(len_b as u64, 1, 1), tg);
                        enc.end_encoding();
                        drop(pending);
                    }
                    *output = out;
                    return Ok(output.length() as usize);
                }
            }
        }

        // Reshape: zero-copy alias.
        if matches!(op, FloatOp::Reshape) && !inputs.is_empty() {
            *output = inputs[0].clone();
            return Ok(output.length() as usize);
        }

        // Conv2d: im2col on GPU + SGEMM.
        if let FloatOp::Conv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group: _,
            input_h,
            input_w,
        } = op
        {
            let h = *input_h as usize;
            let w = *input_w as usize;
            let kh = *kernel_h as usize;
            let kw = *kernel_w as usize;
            let sh = (*stride_h).max(1) as usize;
            let sw = (*stride_w).max(1) as usize;
            let ph = *pad_h as usize;
            let pw = *pad_w as usize;
            let dh = (*dilation_h).max(1) as usize;
            let dw = (*dilation_w).max(1) as usize;

            if inputs.len() >= 2 && h > 0 && w > 0 && kh > 0 && kw > 0 {
                let w_floats = inputs[1].length() as usize / 4;
                let d_floats = inputs[0].length() as usize / 4;
                let spatial = h * w;
                let ic = if spatial > 0 && d_floats.is_multiple_of(spatial) {
                    d_floats / spatial
                } else {
                    0
                };
                let oc = if ic > 0 && kh > 0 && kw > 0 && w_floats.is_multiple_of(ic * kh * kw) {
                    w_floats / (ic * kh * kw)
                } else {
                    0
                };

                if ic > 0 && oc > 0 {
                    let out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                    let out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
                    let col_rows = ic * kh * kw;
                    let col_cols = out_h * out_w;
                    let col_elems = col_rows * col_cols;

                    // Step 1: im2col.
                    if let Some(im2col_pipe) = self.pipelines.get("im2col") {
                        let col_buf = self.device.new_buffer(
                            (col_elems * 4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        let pending = self.get_or_create_cmd_buf();
                        let cmd = pending.as_ref().expect("Metal cmd for im2col");
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(im2col_pipe);
                        enc.set_buffer(0, Some(inputs[0]), 0);
                        enc.set_buffer(1, Some(&col_buf), 0);
                        enc.set_buffer(2, Some(&self.u32_buf(ic as u32)), 0);
                        enc.set_buffer(3, Some(&self.u32_buf(h as u32)), 0);
                        enc.set_buffer(4, Some(&self.u32_buf(w as u32)), 0);
                        enc.set_buffer(5, Some(&self.u32_buf(kh as u32)), 0);
                        enc.set_buffer(6, Some(&self.u32_buf(kw as u32)), 0);
                        enc.set_buffer(7, Some(&self.u32_buf(ph as u32)), 0);
                        enc.set_buffer(8, Some(&self.u32_buf(pw as u32)), 0);
                        enc.set_buffer(9, Some(&self.u32_buf(sh as u32)), 0);
                        enc.set_buffer(10, Some(&self.u32_buf(sw as u32)), 0);
                        enc.set_buffer(11, Some(&self.u32_buf(dh as u32)), 0);
                        enc.set_buffer(12, Some(&self.u32_buf(dw as u32)), 0);
                        enc.set_buffer(13, Some(&self.u32_buf(out_h as u32)), 0);
                        enc.set_buffer(14, Some(&self.u32_buf(out_w as u32)), 0);
                        let tg = MTLSize::new(
                            im2col_pipe.max_total_threads_per_threadgroup().min(256),
                            1,
                            1,
                        );
                        enc.dispatch_threads(MTLSize::new(col_elems as u64, 1, 1), tg);
                        enc.end_encoding();
                        drop(pending);

                        // Step 2: SGEMM weight[oc, col_rows] × col[col_rows, col_cols].
                        if let Some(sgemm_pipe) = self.pipelines.get("sgemm") {
                            let out_elems = oc * col_cols;
                            let out_buf = self.device.new_buffer(
                                (out_elems * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let pending = self.get_or_create_cmd_buf();
                            let cmd = pending.as_ref().expect("Metal cmd for conv sgemm");
                            let enc = cmd.new_compute_command_encoder();
                            enc.set_compute_pipeline_state(sgemm_pipe);
                            enc.set_buffer(0, Some(inputs[1]), 0);
                            enc.set_buffer(1, Some(&col_buf), 0);
                            enc.set_buffer(2, Some(&out_buf), 0);
                            let m_buf = self.u32_buf(oc as u32);
                            let k_buf = self.u32_buf(col_rows as u32);
                            let n_buf = self.u32_buf(col_cols as u32);
                            enc.set_buffer(3, Some(&m_buf), 0);
                            enc.set_buffer(4, Some(&k_buf), 0);
                            enc.set_buffer(5, Some(&n_buf), 0);
                            let tg = MTLSize::new(16, 16, 1);
                            let grid = MTLSize::new(
                                (col_cols as u64).div_ceil(16),
                                (oc as u64).div_ceil(16),
                                1,
                            );
                            enc.dispatch_thread_groups(grid, tg);
                            enc.end_encoding();
                            drop(pending);

                            // Bias add (CPU for simplicity — bias is small).
                            if inputs.len() >= 3 && inputs[2].length() > 0 {
                                self.flush();
                                let out_ptr = out_buf.contents() as *mut f32;
                                let bias_ptr = inputs[2].contents() as *const f32;
                                let bias_len = inputs[2].length() as usize / 4;
                                for c in 0..oc.min(bias_len) {
                                    for s in 0..col_cols {
                                        unsafe {
                                            *out_ptr.add(c * col_cols + s) += *bias_ptr.add(c);
                                        }
                                    }
                                }
                            }

                            *output = out_buf;
                            return Ok(output.length() as usize);
                        }
                    }
                }
            }
        }

        // Resize: nearest neighbor upsampling (mode=0).
        if let FloatOp::Resize { mode } = op {
            if *mode == 0 && !inputs.is_empty() {
                if let Some(pipeline) = self.pipelines.get("resize_nearest") {
                    // Scale factor from params or infer from output size hint.
                    // For now, use 2× upsampling (most common in SD UNet).
                    let in_floats = inputs[0].length() as usize / 4;
                    let scale = _params.u32s.first().copied().unwrap_or(2) as usize;
                    // Infer spatial dims: assume NCHW, try common channel counts.
                    let (channels, in_h, in_w) = if _params.u32s.len() >= 4 {
                        (
                            _params.u32s[1] as usize,
                            _params.u32s[2] as usize,
                            _params.u32s[3] as usize,
                        )
                    } else {
                        // Heuristic: try sqrt for square spatial.
                        let per_channel = in_floats;
                        let s = (per_channel as f64).sqrt() as usize;
                        if s > 0 && (s * s) == per_channel {
                            (1, s, s)
                        } else {
                            (0, 0, 0)
                        }
                    };
                    if channels > 0 && in_h > 0 && in_w > 0 {
                        let out_h = in_h * scale;
                        let out_w = in_w * scale;
                        let out_total = channels * out_h * out_w;
                        let out_buf = self.device.new_buffer(
                            (out_total * 4) as u64,
                            MTLResourceOptions::StorageModeShared,
                        );
                        let pending = self.get_or_create_cmd_buf();
                        let cmd = pending.as_ref().expect("Metal cmd for resize");
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pipeline);
                        enc.set_buffer(0, Some(inputs[0]), 0);
                        enc.set_buffer(1, Some(&out_buf), 0);
                        enc.set_buffer(2, Some(&self.u32_buf(out_total as u32)), 0);
                        enc.set_buffer(3, Some(&self.u32_buf(in_h as u32)), 0);
                        enc.set_buffer(4, Some(&self.u32_buf(in_w as u32)), 0);
                        enc.set_buffer(5, Some(&self.u32_buf(out_h as u32)), 0);
                        enc.set_buffer(6, Some(&self.u32_buf(out_w as u32)), 0);
                        enc.set_buffer(7, Some(&self.u32_buf(channels as u32)), 0);
                        let tg = MTLSize::new(
                            pipeline.max_total_threads_per_threadgroup().min(256),
                            1,
                            1,
                        );
                        enc.dispatch_threads(MTLSize::new(out_total as u64, 1, 1), tg);
                        enc.end_encoding();
                        drop(pending);
                        *output = out_buf;
                        return Ok(output.length() as usize);
                    }
                }
            }
        }

        Err(BackendError::Unsupported(format!(
            "Metal dispatch for {op:?} not yet implemented"
        )))
    }

    fn dispatch_ring(
        &self,
        table_idx: usize,
        inputs: &[&metal::Buffer],
        output: &mut metal::Buffer,
    ) -> Result<usize> {
        if table_idx >= self.ring_tables.len() {
            return Err(BackendError::Unsupported(format!(
                "ring table index {table_idx} out of range (have {})",
                self.ring_tables.len()
            )));
        }
        let pipeline = self
            .pipelines
            .get("ring_lut")
            .ok_or_else(|| BackendError::Device("ring_lut pipeline not compiled".into()))?;
        let input = inputs
            .first()
            .ok_or_else(|| BackendError::Shape("ring op requires at least one input".into()))?;

        let count = input.length() as usize;
        let out = self
            .device
            .new_buffer(count as u64, MTLResourceOptions::StorageModeShared);

        let pending = self.get_or_create_cmd_buf();
        let cmd = pending.as_ref().expect("Metal cmd buf for ring_lut");
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_buffer(2, Some(&self.ring_tables[table_idx]), 0);
        enc.set_buffer(3, Some(&self.u32_buf(count as u32)), 0);
        let tg = MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
        enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), tg);
        enc.end_encoding();
        drop(pending);

        *output = out;
        Ok(count)
    }

    fn load_ring_tables(&mut self, tables: &[&[u8; 256]], memory: &MetalMemory) {
        self.ring_tables = tables
            .iter()
            .map(|table| memory.upload(&table[..]))
            .collect();
    }

    fn flush(&self) {
        let mut pending = self
            .pending
            .lock()
            .expect("Metal pending command buffer mutex should not be poisoned");
        if let Some(cmd_buf) = pending.take() {
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
    }

    fn name(&self) -> &'static str {
        "metal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_memory_roundtrip() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return, // Skip if no Metal device.
        };
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let buf = mem.upload(&data);
        assert_eq!(mem.byte_len(&buf), 8);

        // Need to create a backend just to flush (no pending work, but
        // ensures the pattern works).
        let result = mem.download(&buf);
        assert_eq!(data, result);
    }

    #[test]
    fn metal_f32_buf_creates() {
        let backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };
        let buf = backend.f32_buf(3.14);
        assert_eq!(buf.length(), 4);
    }

    #[test]
    fn metal_backend_creates() {
        let backend = MetalBackend::new();
        if let Some(b) = backend {
            assert_eq!(b.name(), "metal");
            assert!(!b.pipelines.is_empty());
        }
        // Skip if no Metal device.
    }

    #[test]
    fn metal_ring_lut_dispatch() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return,
        };
        let mut backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };

        // NOT table.
        let mut not_table = [0u8; 256];
        for i in 0..256 {
            not_table[i] = (255 - i) as u8;
        }
        backend.load_ring_tables(&[&not_table], &mem);

        let input = mem.upload(&[0u8, 1, 254, 255]);
        let mut output = mem.alloc(4);

        let written = backend
            .dispatch_ring(0, &[&input], &mut output)
            .expect("ring dispatch should succeed");
        backend.flush();

        assert_eq!(written, 4);
        let result = mem.download(&output);
        assert_eq!(result, vec![255, 254, 1, 0]);
    }

    #[test]
    fn metal_relu_dispatch() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return,
        };
        let backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };

        let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = mem.upload(bytemuck::cast_slice(&data));
        let mut output = mem.alloc(0);

        let written = backend
            .dispatch(
                &FloatOp::Relu,
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("relu dispatch should succeed");
        backend.flush();

        assert_eq!(written, 20); // 5 floats * 4 bytes
        let result_bytes = mem.download(&output);
        let result: &[f32] = bytemuck::cast_slice(&result_bytes);
        assert_eq!(result, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn metal_matmul_dispatch() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return,
        };
        let backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };
        // 32x64 × 64x32 — large enough to fill Metal threadgroups.
        let m = 32usize;
        let k = 64usize;
        let n = 32usize;
        let mut a = vec![0.0f32; m * k];
        for j in 0..k {
            a[j] = 1.0;
        } // First row = all 1s
        let b_data = vec![2.0f32; k * n]; // All 2s
        let buf_a = mem.upload(bytemuck::cast_slice(&a));
        let buf_b = mem.upload(bytemuck::cast_slice(&b_data));
        let mut output = mem.alloc(0);

        backend
            .dispatch(
                &FloatOp::MatMul {
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                },
                &[&buf_a, &buf_b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("matmul should succeed");
        backend.flush();

        let result_bytes = mem.download(&output);
        let result: &[f32] = bytemuck::cast_slice(&result_bytes);
        assert_eq!(result.len(), m * n);
        let expected = 2.0 * k as f32; // 128.0
        assert!(
            (result[0] - expected).abs() < 0.5,
            "C[0,0] should be ~{expected}, got {}",
            result[0]
        );
    }

    #[test]
    fn metal_softmax_dispatch() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return,
        };
        let backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let input = mem.upload(bytemuck::cast_slice(&data));
        let mut output = mem.alloc(0);

        backend
            .dispatch(
                &FloatOp::Softmax { size: 3 },
                &[&input],
                &mut output,
                &KernelParams::default(),
            )
            .expect("softmax should succeed");
        backend.flush();

        let result_bytes = mem.download(&output);
        let result: &[f32] = bytemuck::cast_slice(&result_bytes);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn metal_add_dispatch() {
        let mem = match MetalMemory::new() {
            Some(m) => m,
            None => return,
        };
        let backend = match MetalBackend::new() {
            Some(b) => b,
            None => return,
        };

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let buf_a = mem.upload(bytemuck::cast_slice(&a));
        let buf_b = mem.upload(bytemuck::cast_slice(&b));
        let mut output = mem.alloc(0);

        backend
            .dispatch(
                &FloatOp::Add,
                &[&buf_a, &buf_b],
                &mut output,
                &KernelParams::default(),
            )
            .expect("add dispatch should succeed");
        backend.flush();

        let result_bytes = mem.download(&output);
        let result: &[f32] = bytemuck::cast_slice(&result_bytes);
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }
}
