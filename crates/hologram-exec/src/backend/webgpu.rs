//! WebGPU compute backend via wgpu.
//!
//! Cross-platform GPU acceleration: Vulkan (Linux/Windows),
//! Metal (macOS via wgpu), DX12 (Windows), WebGPU (browser).
//! Feature-gated behind `webgpu` — not compiled by default.
//!
//! Architecture mirrors the Metal backend:
//! - Pipeline caching at init (WGSL shaders compiled once)
//! - Size thresholds (GPU only for large buffers)
//! - Synchronous dispatch with staging buffer readback
//! - `flush()` support for future command encoder batching

use std::collections::HashMap;

use hologram_core::op::{FloatOp, OpCategory};
use wgpu::util::DeviceExt;

use crate::error::{ExecError, ExecResult};

use super::ComputeBackend;

/// Minimum buffer size (bytes) to dispatch to WebGPU.
/// GPU kernel launch + staging buffer readback overhead means
/// WebGPU only wins for large buffers (~1M floats = 4MB).
const WEBGPU_MIN_BYTES: usize = 4 * 1024 * 1024;

// ── WGSL Shader Sources ─────────────────────────────────────────────────────
// Separate modules per kernel category (different bind group layouts).

/// Unary elementwise kernels (9): input, output, params { count }.
const SHADER_UNARY: &str = r#"
struct Params { count: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = max(input[gid.x], 0.0);
}

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = -input[gid.x];
}

@compute @workgroup_size(256)
fn abs_val(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = abs(input[gid.x]);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = 1.0 / (1.0 + exp(-input[gid.x]));
}

@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    let x = input[gid.x];
    output[gid.x] = x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn tanh_act(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = tanh(input[gid.x]);
}

@compute @workgroup_size(256)
fn exp_act(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = exp(input[gid.x]);
}

@compute @workgroup_size(256)
fn reciprocal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    output[gid.x] = 1.0 / input[gid.x];
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.count) { return; }
    let x = input[gid.x];
    output[gid.x] = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
"#;

/// Binary elementwise kernels (4): a, b, output, params { count_a, count_b }.
const SHADER_BINARY: &str = r#"
struct BinaryParams { count_a: u32, count_b: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryParams;

@compute @workgroup_size(256)
fn add_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_len = max(params.count_a, params.count_b);
    if (gid.x >= out_len) { return; }
    output[gid.x] = a[gid.x % params.count_a] + b[gid.x % params.count_b];
}

@compute @workgroup_size(256)
fn mul_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_len = max(params.count_a, params.count_b);
    if (gid.x >= out_len) { return; }
    output[gid.x] = a[gid.x % params.count_a] * b[gid.x % params.count_b];
}

@compute @workgroup_size(256)
fn sub_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_len = max(params.count_a, params.count_b);
    if (gid.x >= out_len) { return; }
    output[gid.x] = a[gid.x % params.count_a] - b[gid.x % params.count_b];
}

@compute @workgroup_size(256)
fn div_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_len = max(params.count_a, params.count_b);
    if (gid.x >= out_len) { return; }
    output[gid.x] = a[gid.x % params.count_a] / b[gid.x % params.count_b];
}
"#;

/// Tiled SGEMM kernel: A, B, C, params { M, K, N }.
const SHADER_SGEMM: &str = r#"
struct GemmParams { M: u32, K: u32, N: u32 }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: GemmParams;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn sgemm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
) {
    let row = tgid.y * TILE + tid.y;
    let col = tgid.x * TILE + tid.x;
    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + tid.x;
        if (row < params.M && a_col < params.K) {
            tileA[tid.y][tid.x] = A[row * params.K + a_col];
        } else {
            tileA[tid.y][tid.x] = 0.0;
        }

        let b_row = t * TILE + tid.y;
        if (b_row < params.K && col < params.N) {
            tileB[tid.y][tid.x] = B[b_row * params.N + col];
        } else {
            tileB[tid.y][tid.x] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < TILE; p = p + 1u) {
            sum = sum + tileA[tid.y][p] * tileB[p][tid.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
"#;

/// Softmax kernel: input, output, params { total, row_size }.
const SHADER_SOFTMAX: &str = r#"
struct SoftmaxParams { total: u32, row_size: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.total) { return; }
    let row_start = (gid.x / params.row_size) * params.row_size;

    var row_max: f32 = -3.402823e+38;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let idx = row_start + i;
        if (idx < params.total) {
            row_max = max(row_max, input[idx]);
        }
    }

    let exp_val = exp(input[gid.x] - row_max);
    var row_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let idx = row_start + i;
        if (idx < params.total) {
            row_sum = row_sum + exp(input[idx] - row_max);
        }
    }

    if (row_sum > 0.0) {
        output[gid.x] = exp_val / row_sum;
    } else {
        output[gid.x] = 1.0 / f32(params.row_size);
    }
}
"#;

/// RmsNorm kernel: input, weight, output, params { total, row_size, epsilon }.
const SHADER_RMS_NORM: &str = r#"
struct RmsParams { total: u32, row_size: u32, epsilon: f32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: RmsParams;

@compute @workgroup_size(256)
fn rms_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.total) { return; }
    let row_start = (gid.x / params.row_size) * params.row_size;
    let col = gid.x % params.row_size;

    var ms: f32 = 0.0;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let idx = row_start + i;
        if (idx < params.total) {
            let v = input[idx];
            ms = ms + v * v;
        }
    }
    ms = ms / f32(params.row_size);

    let inv_rms = inverseSqrt(ms + params.epsilon);
    output[gid.x] = input[gid.x] * inv_rms * weight[col];
}
"#;

// ── WebGpuBackend ────────────────────────────────────────────────────────────

/// WebGPU backend for cross-platform GPU compute via wgpu.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
}

impl WebGpuBackend {
    /// Create a new WebGPU backend. Returns `None` if no suitable GPU adapter found.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hologram-webgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        // Compile WGSL shader modules (one per kernel category).
        let modules: &[(&str, &[&str])] = &[
            (
                SHADER_UNARY,
                &[
                    "relu",
                    "neg",
                    "abs_val",
                    "sigmoid",
                    "silu",
                    "tanh_act",
                    "exp_act",
                    "reciprocal",
                    "gelu",
                ],
            ),
            (SHADER_BINARY, &["add_op", "mul_op", "sub_op", "div_op"]),
            (SHADER_SGEMM, &["sgemm"]),
            (SHADER_SOFTMAX, &["softmax"]),
            (SHADER_RMS_NORM, &["rms_norm"]),
        ];

        let mut pipelines = HashMap::new();
        for &(source, entry_points) in modules {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hologram-wgsl"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            for &name in entry_points {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None, // auto layout from WGSL bindings
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                });
                pipelines.insert(name, pipeline);
            }
        }

        Some(WebGpuBackend {
            device,
            queue,
            pipelines,
        })
    }

    /// Map a FloatOp to a WGSL kernel entry point name.
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

    /// Submit a command encoder and read back from staging buffer into out_buf.
    fn submit_and_readback(
        &self,
        encoder: wgpu::CommandEncoder,
        staging_buf: &wgpu::Buffer,
        byte_len: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| ExecError::UnsupportedOp("wgpu channel closed".into()))?
            .map_err(|e| ExecError::UnsupportedOp(format!("wgpu map failed: {e:?}")))?;

        let data = slice.get_mapped_range();
        out_buf.clear();
        out_buf.extend_from_slice(&data[..byte_len]);
        drop(data);
        staging_buf.unmap();
        Ok(())
    }

    /// Dispatch a unary elementwise op. Writes result to out_buf.
    fn dispatch_unary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input: &[u8],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;

        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: input,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let count = n_floats as u32;
        let count_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("count"),
                contents: bytemuck::bytes_of(&count),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len as u64);

        self.submit_and_readback(encoder, &staging_buf, byte_len, out_buf)
    }

    /// Dispatch a binary elementwise op. Writes result to out_buf.
    fn dispatch_binary(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input_a: &[u8],
        input_b: &[u8],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let n_a = (input_a.len() / 4) as u32;
        let n_b = (input_b.len() / 4) as u32;
        let n_out = n_a.max(n_b) as usize;
        let byte_len = n_out * 4;

        let buf_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: input_a,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: input_b,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct BinaryParams {
            count_a: u32,
            count_b: u32,
        }
        let params = BinaryParams {
            count_a: n_a,
            count_b: n_b,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_out as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len as u64);

        self.submit_and_readback(encoder, &staging_buf, byte_len, out_buf)
    }

    /// Dispatch tiled SGEMM (matmul). Writes result to out_buf.
    fn dispatch_sgemm(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        k: usize,
        n: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let byte_len = m * n * 4;

        let buf_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A"),
                contents: a,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B"),
                contents: b,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GemmParams {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32, // align to 16 bytes for uniform buffer
        }
        let params = GemmParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gemm_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self
            .pipelines
            .get("sgemm")
            .ok_or_else(|| ExecError::UnsupportedOp("wgpu sgemm pipeline missing".into()))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32 + 15) / 16;
            let wg_y = (m as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, byte_len as u64);

        self.submit_and_readback(encoder, &staging_buf, byte_len, out_buf)
    }

    /// Dispatch softmax. Writes result to out_buf.
    fn dispatch_softmax(
        &self,
        input: &[u8],
        row_size: usize,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;

        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: input,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SoftmaxParams {
            total: u32,
            row_size: u32,
        }
        let params = SoftmaxParams {
            total: n_floats as u32,
            row_size: row_size as u32,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self
            .pipelines
            .get("softmax")
            .ok_or_else(|| ExecError::UnsupportedOp("wgpu softmax pipeline missing".into()))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len as u64);

        self.submit_and_readback(encoder, &staging_buf, byte_len, out_buf)
    }

    /// Dispatch RmsNorm. Writes result to out_buf.
    fn dispatch_rms_norm(
        &self,
        input: &[u8],
        weight: &[u8],
        row_size: usize,
        epsilon: f32,
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<()> {
        let n_floats = input.len() / 4;
        let byte_len = n_floats * 4;

        let input_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: input,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let weight_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weight"),
                contents: weight,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RmsParams {
            total: u32,
            row_size: u32,
            epsilon: f32,
            _pad: u32,
        }
        let params = RmsParams {
            total: n_floats as u32,
            row_size: row_size as u32,
            epsilon,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self
            .pipelines
            .get("rms_norm")
            .ok_or_else(|| ExecError::UnsupportedOp("wgpu rms_norm pipeline missing".into()))?;
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len as u64);

        self.submit_and_readback(encoder, &staging_buf, byte_len, out_buf)
    }
}

// ── ComputeBackend impl ─────────────────────────────────────────────────────

impl ComputeBackend for WebGpuBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<super::KernelOutput> {
        // Route MatMul to dispatch_matmul.
        if let FloatOp::MatMul { m, k, n } = op {
            return self.dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize, out_buf);
        }

        // Route Softmax with threshold check.
        if let FloatOp::Softmax { size } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= WEBGPU_MIN_BYTES && *size > 0 {
                self.dispatch_softmax(inputs[0], *size as usize, out_buf)?;
                return Ok(super::KernelOutput::Bytes);
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // Route RmsNorm with threshold check.
        if let FloatOp::RmsNorm { size, epsilon } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= WEBGPU_MIN_BYTES && inputs.len() >= 2 && *size > 0 {
                self.dispatch_rms_norm(
                    inputs[0],
                    inputs[1],
                    *size as usize,
                    f32::from_bits(*epsilon),
                    out_buf,
                )?;
                return Ok(super::KernelOutput::Bytes);
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // Size threshold for elementwise ops.
        let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
        if input_bytes < WEBGPU_MIN_BYTES {
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
                self.dispatch_unary(pipeline, inputs[0], out_buf)?;
                Ok(super::KernelOutput::Bytes)
            }
            OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                self.dispatch_binary(pipeline, inputs[0], inputs[1], out_buf)?;
                Ok(super::KernelOutput::Bytes)
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
        out_buf: &mut Vec<u8>,
    ) -> ExecResult<super::KernelOutput> {
        // Same threshold as Metal: 128×128 output minimum.
        if m * n < 128 * 128 {
            return Ok(super::KernelOutput::Skipped);
        }
        if inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }
        self.dispatch_sgemm(inputs[0], inputs[1], m, k, n, out_buf)?;
        Ok(super::KernelOutput::Bytes)
    }

    fn name(&self) -> &'static str {
        "webgpu"
    }
}
