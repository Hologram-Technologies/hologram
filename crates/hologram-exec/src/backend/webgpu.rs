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
use std::sync::Mutex;

use hologram_core::op::{FloatOp, OpCategory};
use wgpu::util::DeviceExt;

use crate::buffer::OutputBuffer;
use crate::error::{ExecError, ExecResult};

use super::hardware::{HardwareCaps, OpThresholds};
use super::ComputeBackend;

// Thresholds are now per-instance via OpThresholds (see `thresholds` field).

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

/// Batched tiled SGEMM kernel: C[b,M,N] = A[b,M,K] × B[b,K,N].
/// Z workgroup dimension is batch. B can be shared (b_stride=0).
const SHADER_BATCHED_SGEMM: &str = r#"
struct BatchGemmParams { M: u32, K: u32, N: u32, a_stride: u32, b_stride: u32 }

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: BatchGemmParams;

const TILE: u32 = 16u;
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn batched_sgemm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) tgid: vec3<u32>,
) {
    let batch = tgid.z;
    let a_off = batch * params.a_stride;
    let b_off = batch * params.b_stride;
    let c_off = batch * params.M * params.N;

    let row = tgid.y * TILE + tid.y;
    let col = tgid.x * TILE + tid.x;
    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + tid.x;
        if (row < params.M && a_col < params.K) {
            tileA[tid.y][tid.x] = A[a_off + row * params.K + a_col];
        } else {
            tileA[tid.y][tid.x] = 0.0;
        }

        let b_row = t * TILE + tid.y;
        if (b_row < params.K && col < params.N) {
            tileB[tid.y][tid.x] = B[b_off + b_row * params.N + col];
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
        C[c_off + row * params.N + col] = sum;
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

/// GroupNorm kernel: input, scale, bias, output, params { total, channels, group_size, spatial, epsilon }.
const SHADER_GROUP_NORM: &str = r#"
struct GroupNormParams { total: u32, channels: u32, group_size: u32, spatial: u32, epsilon: f32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: GroupNormParams;

@compute @workgroup_size(256)
fn group_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.total) { return; }
    // Determine which group this element belongs to.
    let c = (gid.x / params.spatial) % params.channels;
    let group = c / params.group_size;
    let n = gid.x / (params.channels * params.spatial);
    let group_start = n * params.channels * params.spatial + group * params.group_size * params.spatial;
    let group_elems = params.group_size * params.spatial;

    // Compute mean and variance for this group.
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < group_elems; i = i + 1u) {
        mean = mean + input[group_start + i];
    }
    mean = mean / f32(group_elems);

    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < group_elems; i = i + 1u) {
        let diff = input[group_start + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(group_elems);

    let inv_std = inverseSqrt(variance + params.epsilon);
    output[gid.x] = (input[gid.x] - mean) * inv_std * scale[c] + bias[c];
}
"#;

// ── WebGpuBackend ────────────────────────────────────────────────────────────

/// A single deferred GPU dispatch awaiting readback.
struct DeferredEntry {
    staging_buf: wgpu::Buffer,
    byte_len: usize,
}

/// Pending GPU work: shared command encoder + deferred entries.
struct PendingWork {
    encoder: wgpu::CommandEncoder,
    entries: Vec<DeferredEntry>,
    /// Buffers referenced by the encoder that must stay alive until submit.
    kept_alive: Vec<wgpu::Buffer>,
}

/// WebGPU backend for cross-platform GPU compute via wgpu.
///
/// Phase 8.3d: Command encoder batching. Multiple dispatches encode into
/// a single `CommandEncoder`. `flush_deferred()` submits once and reads
/// back all staging buffers, returning results in dispatch order.
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    /// Pending work for batch encoding. `None` when idle.
    pending: Mutex<Option<PendingWork>>,
    /// Hardware-detected per-op dispatch thresholds.
    thresholds: OpThresholds,
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
            (SHADER_BATCHED_SGEMM, &["batched_sgemm"]),
            (SHADER_SOFTMAX, &["softmax"]),
            (SHADER_RMS_NORM, &["rms_norm"]),
            (SHADER_GROUP_NORM, &["group_norm"]),
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

        let thresholds = OpThresholds::from(HardwareCaps::detect());

        Some(WebGpuBackend {
            device,
            queue,
            pipelines,
            pending: Mutex::new(None),
            thresholds,
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

    /// Get or create the shared command encoder for batch encoding.
    fn get_or_create_encoder(&self) -> std::sync::MutexGuard<'_, Option<PendingWork>> {
        let mut pending = self.pending.lock().unwrap();
        if pending.is_none() {
            *pending = Some(PendingWork {
                encoder: self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default()),
                entries: Vec::new(),
                kept_alive: Vec::new(),
            });
        }
        pending
    }

    /// Encode a copy-to-staging command and track the deferred entry.
    /// Returns the staging buffer's entry index.
    fn enqueue_staging(
        pending: &mut PendingWork,
        output_buf: wgpu::Buffer,
        staging_buf: wgpu::Buffer,
        byte_len: usize,
    ) {
        pending
            .encoder
            .copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, byte_len as u64);
        pending.kept_alive.push(output_buf);
        pending.entries.push(DeferredEntry {
            staging_buf,
            byte_len,
        });
    }

    /// Flush all pending GPU work: submit once, poll, batch-readback all staging buffers.
    pub fn flush_deferred_impl(&self) -> ExecResult<Vec<Vec<u8>>> {
        let work = match self.pending.lock().unwrap().take() {
            Some(w) => w,
            None => return Ok(Vec::new()),
        };

        // Single submit for ALL dispatches in this level.
        self.queue.submit(std::iter::once(work.encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Issue all map_async calls at once.
        let channels: Vec<_> = work
            .entries
            .iter()
            .map(|entry| {
                let slice = entry.staging_buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                rx
            })
            .collect();

        // Single poll satisfies all pending maps.
        self.device.poll(wgpu::Maintain::Wait);

        // Read back all staging buffers in dispatch order.
        let mut results = Vec::with_capacity(work.entries.len());
        for (entry, rx) in work.entries.iter().zip(channels) {
            rx.recv()
                .map_err(|_| ExecError::UnsupportedOp("wgpu channel closed".into()))?
                .map_err(|e| ExecError::UnsupportedOp(format!("wgpu map failed: {e:?}")))?;
            let slice = entry.staging_buf.slice(..);
            let data = slice.get_mapped_range();
            let mut buf = Vec::with_capacity(entry.byte_len);
            buf.extend_from_slice(&data[..entry.byte_len]);
            results.push(buf);
            drop(data);
            entry.staging_buf.unmap();
        }

        Ok(results)
    }

    /// Encode a unary elementwise op into the shared command encoder (deferred).
    fn dispatch_unary_deferred(&self, pipeline: &wgpu::ComputePipeline, input: &[u8]) {
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        work.kept_alive.extend([input_buf, count_buf]);
        Self::enqueue_staging(work, output_buf, staging_buf, byte_len);
    }

    /// Encode a binary elementwise op into the shared command encoder (deferred).
    fn dispatch_binary_deferred(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input_a: &[u8],
        input_b: &[u8],
    ) {
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_out as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        work.kept_alive.extend([buf_a, buf_b, params_buf]);
        Self::enqueue_staging(work, output_buf, staging_buf, byte_len);
    }

    /// Encode tiled SGEMM (matmul) into the shared command encoder (deferred).
    fn dispatch_sgemm_deferred(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        k: usize,
        n: usize,
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32 + 15) / 16;
            let wg_y = (m as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        work.kept_alive.extend([buf_a, buf_b, params_buf]);
        Self::enqueue_staging(work, buf_c, staging_buf, byte_len);
        Ok(())
    }

    /// Encode batched tiled SGEMM into the shared command encoder (deferred).
    fn dispatch_batched_sgemm_deferred(
        &self,
        a: &[u8],
        b: &[u8],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        b_broadcast: bool,
    ) -> ExecResult<()> {
        let byte_len = batch * m * n * 4;

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
        struct BatchGemmParams {
            m: u32,
            k: u32,
            n: u32,
            a_stride: u32,
            b_stride: u32,
            _pad: [u32; 3], // align to 32 bytes for uniform buffer
        }
        let params = BatchGemmParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            a_stride: (m * k) as u32,
            b_stride: if b_broadcast { 0 } else { (k * n) as u32 },
            _pad: [0; 3],
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("batch_gemm_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: byte_len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = self.pipelines.get("batched_sgemm").ok_or_else(|| {
            ExecError::UnsupportedOp("wgpu batched_sgemm pipeline missing".into())
        })?;
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32 + 15) / 16;
            let wg_y = (m as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, batch as u32);
        }
        work.kept_alive.extend([buf_a, buf_b, params_buf]);
        Self::enqueue_staging(work, buf_c, staging_buf, byte_len);
        Ok(())
    }

    /// Encode softmax into the shared command encoder (deferred).
    fn dispatch_softmax_deferred(&self, input: &[u8], row_size: usize) -> ExecResult<()> {
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        work.kept_alive.extend([input_buf, params_buf]);
        Self::enqueue_staging(work, output_buf, staging_buf, byte_len);
        Ok(())
    }

    /// Encode RmsNorm into the shared command encoder (deferred).
    fn dispatch_rms_norm_deferred(
        &self,
        input: &[u8],
        weight: &[u8],
        row_size: usize,
        epsilon: f32,
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

        let mut pending = self.get_or_create_encoder();
        let work = pending.as_mut().unwrap();
        {
            let mut pass = work
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n_floats as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        work.kept_alive.extend([input_buf, weight_buf, params_buf]);
        Self::enqueue_staging(work, output_buf, staging_buf, byte_len);
        Ok(())
    }
}

// ── ComputeBackend impl ─────────────────────────────────────────────────────

impl ComputeBackend for WebGpuBackend {
    fn dispatch_float(
        &self,
        op: &FloatOp,
        inputs: &[&[u8]],
        _out_buf: &mut OutputBuffer,
    ) -> ExecResult<super::KernelOutput> {
        // Route MatMul to dispatch_matmul.
        if let FloatOp::MatMul { m, k, n } = op {
            return self.dispatch_matmul(inputs, *m as usize, *k as usize, *n as usize, _out_buf);
        }

        // Route Softmax with threshold check.
        if let FloatOp::Softmax { size } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= self.thresholds.softmax_min_bytes && *size > 0 {
                self.dispatch_softmax_deferred(inputs[0], *size as usize)?;
                return Ok(super::KernelOutput::WgpuDeferred);
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // Route RmsNorm with threshold check.
        if let FloatOp::RmsNorm { size, epsilon } = op {
            let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
            if input_bytes >= self.thresholds.norm_min_bytes && inputs.len() >= 2 && *size > 0 {
                self.dispatch_rms_norm_deferred(
                    inputs[0],
                    inputs[1],
                    *size as usize,
                    f32::from_bits(*epsilon),
                )?;
                return Ok(super::KernelOutput::WgpuDeferred);
            }
            return Ok(super::KernelOutput::Skipped);
        }

        // Size threshold for elementwise ops.
        let input_bytes = inputs.first().map(|b| b.len()).unwrap_or(0);
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
                self.dispatch_unary_deferred(pipeline, inputs[0]);
                Ok(super::KernelOutput::WgpuDeferred)
            }
            OpCategory::BinaryElementwise if inputs.len() >= 2 => {
                self.dispatch_binary_deferred(pipeline, inputs[0], inputs[1]);
                Ok(super::KernelOutput::WgpuDeferred)
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
        // Same threshold as Metal: 128×128 output minimum.
        if m * n < self.thresholds.matmul_min_elements {
            return Ok(super::KernelOutput::Skipped);
        }
        if inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }
        self.dispatch_sgemm_deferred(inputs[0], inputs[1], m, k, n)?;
        Ok(super::KernelOutput::WgpuDeferred)
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
        let total_output = batch * m * n;
        if total_output < 4096 || inputs.len() < 2 {
            return Ok(super::KernelOutput::Skipped);
        }
        self.dispatch_batched_sgemm_deferred(inputs[0], inputs[1], batch, m, k, n, b_broadcast)?;
        Ok(super::KernelOutput::WgpuDeferred)
    }

    fn name(&self) -> &'static str {
        "webgpu"
    }

    fn op_thresholds(&self) -> &super::hardware::OpThresholds {
        &self.thresholds
    }

    fn flush_deferred(&self) -> ExecResult<Vec<Vec<u8>>> {
        self.flush_deferred_impl()
    }
}
