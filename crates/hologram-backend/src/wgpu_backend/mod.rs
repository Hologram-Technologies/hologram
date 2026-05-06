//! wgpu backend (spec IX.4).
//!
//! Cross-platform GPU backend via the wgpu crate. WGSL compute kernels live
//! in `shaders.wgsl`; the backend selects a pipeline per `KernelCall`
//! variant and runs it against the workspace.
//!
//! Per ADR-051 (workspace residency), the backend keeps device-resident
//! buffers across kernel calls. Per ADR-018 / spec III.2, the binding `Bounds`
//! is `HologramHostBoundsWgpu` (WITT_LEVEL_MAX_BITS = 64).

use core::marker::PhantomData;
use std::collections::HashMap;
use std::sync::Arc;

use hologram_host::HologramHostBoundsWgpu;
use crate::backend::Backend;
use crate::kernel_call::*;
use crate::workspace::Workspace;
use crate::error::BackendError;
use crate::cpu::dtype::is_float;

const SHADERS: &str = include_str!("shaders.wgsl");

/// One-time GPU context: device + queue + compiled compute pipelines.
pub struct WgpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuContext {
    /// Construct a new wgpu context using the default adapter.
    pub fn new() -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(BackendError::Init("no compatible adapter"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hologram-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|_| BackendError::Init("device creation failed"))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hologram-wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hologram-bgl"),
            entries: &[
                bgl_entry(0, true),
                bgl_entry(1, true),
                bgl_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hologram-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mut pipelines = HashMap::new();
        for entry in [
            "add_f32", "sub_f32", "mul_f32",
            "relu_f32", "sigmoid_f32", "tanh_f32",
            "matmul_f32",
        ] {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.insert(entry, p);
        }

        Ok(Self { device, queue, pipelines, bind_group_layout: bgl })
    }
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    m: u32,
    k: u32,
    pad0: u32,
}

/// wgpu backend handle. Workspace-typed via the standard `Backend` trait.
pub struct WgpuBackend<W: Workspace> {
    ctx: WgpuContext,
    _ws: PhantomData<W>,
}

impl<W: Workspace> WgpuBackend<W> {
    pub fn new() -> Result<Self, BackendError> {
        Ok(Self { ctx: WgpuContext::new()?, _ws: PhantomData })
    }

    fn dispatch_compute(
        &self,
        entry_name: &'static str,
        a: &[u8],
        b: &[u8],
        out_len: usize,
        params: Params,
        groups: (u32, u32, u32),
    ) -> Vec<u8> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let pipeline = match self.ctx.pipelines.get(entry_name) {
            Some(p) => p,
            None => return vec![0; out_len],
        };

        // Allocate device-side buffers.
        let a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("a"),
            size: a.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b"),
            size: b.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_len.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_len.max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !a.is_empty() { queue.write_buffer(&a_buf, 0, a); }
        if !b.is_empty() { queue.write_buffer(&b_buf, 0, b); }
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hologram-bg"),
            layout: &self.ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hologram-enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hologram-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_len.max(4) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        let _ = receiver.recv();
        let bytes: Vec<u8> = slice.get_mapped_range().to_vec();
        staging.unmap();
        bytes
    }
}

impl<W: Workspace> Backend for WgpuBackend<W> {
    type Bounds = HologramHostBoundsWgpu;
    type WS = W;

    fn dispatch(&mut self, call: &KernelCall, ws: &mut Self::WS) -> Result<(), BackendError> {
        // Only float kernels run on GPU; non-f32 dtypes fall through to CPU.
        match call {
            KernelCall::Add(c) if is_float(c.dtype) => run_binary(self, ws, c, "add_f32"),
            KernelCall::Sub(c) if is_float(c.dtype) => run_binary(self, ws, c, "sub_f32"),
            KernelCall::Mul(c) if is_float(c.dtype) => run_binary(self, ws, c, "mul_f32"),
            KernelCall::Relu(c) if is_float(c.dtype) => run_unary(self, ws, c, "relu_f32"),
            KernelCall::Sigmoid(c) if is_float(c.dtype) => run_unary(self, ws, c, "sigmoid_f32"),
            KernelCall::Tanh(c) if is_float(c.dtype) => run_unary(self, ws, c, "tanh_f32"),
            KernelCall::MatMul(c) if is_float(c.dtype) => run_matmul(self, ws, c),
            _ => {
                // Fallback: route through CPU dispatch.
                crate::cpu::CpuBackend::<Self::WS>::new().dispatch(call, ws)
            }
        }
    }
}

fn run_unary<W: Workspace>(
    be: &WgpuBackend<W>, ws: &mut W, c: &UnaryCall, entry: &'static str,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = n * 4;
    let a = ws.read(c.input).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let params = Params { n: n as u32, m: 0, k: 0, pad0: 0 };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let out_bytes = be.dispatch_compute(entry, &a, &[], bytes, params, groups);
    let out = ws.write(c.output);
    let take = bytes.min(out.len()).min(out_bytes.len());
    out[..take].copy_from_slice(&out_bytes[..take]);
    Ok(())
}

fn run_binary<W: Workspace>(
    be: &WgpuBackend<W>, ws: &mut W, c: &BinaryCall, entry: &'static str,
) -> Result<(), BackendError> {
    let n = c.element_count as usize;
    let bytes = n * 4;
    let a = ws.read(c.a).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let params = Params { n: n as u32, m: 0, k: 0, pad0: 0 };
    let groups = ((n as u32).div_ceil(64), 1, 1);
    let out_bytes = be.dispatch_compute(entry, &a, &b, bytes, params, groups);
    let out = ws.write(c.output);
    let take = bytes.min(out.len()).min(out_bytes.len());
    out[..take].copy_from_slice(&out_bytes[..take]);
    Ok(())
}

fn run_matmul<W: Workspace>(
    be: &WgpuBackend<W>, ws: &mut W, c: &MatMulCall,
) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 { return Ok(()); }
    let a_bytes = m * k * 4;
    let b_bytes = k * n * 4;
    let out_bytes = m * n * 4;
    let a = ws.read(c.a).get(..a_bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws.read(c.b).get(..b_bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let params = Params { n: n as u32, m: m as u32, k: k as u32, pad0: 0 };
    let groups = ((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
    let result = be.dispatch_compute("matmul_f32", &a, &b, out_bytes, params, groups);
    let out = ws.write(c.output);
    let take = out_bytes.min(out.len()).min(result.len());
    out[..take].copy_from_slice(&result[..take]);
    Ok(())
}
