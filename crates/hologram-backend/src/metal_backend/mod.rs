//! Apple Metal backend (spec IX.3).
//!
//! macOS-only; gated on `target_os = "macos"` and the `metal` feature.
//! Implements the `Backend` trait against Metal compute shaders for the
//! float kernel set. Per spec III.2 the bound is `HologramHostBoundsMetal`
//! (WITT_LEVEL_MAX_BITS = 64).

use core::marker::PhantomData;
use std::sync::Arc;

use crate::backend::Backend;
use crate::cpu::dtype::is_float;
use crate::error::BackendError;
use crate::kernel_call::*;
use crate::workspace::Workspace;
use hologram_host::HologramHostBoundsMetal;
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

/// MSL source for the major float-typed kernels. Compiled at backend init.
const SHADER_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(device float* a [[buffer(0)]],
                    device float* b [[buffer(1)]],
                    device float* o [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    o[i] = a[i] + b[i];
}

kernel void mul_f32(device float* a [[buffer(0)]],
                    device float* b [[buffer(1)]],
                    device float* o [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    o[i] = a[i] * b[i];
}

kernel void relu_f32(device float* a [[buffer(0)]],
                     device float* o [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    o[i] = max(a[i], 0.0);
}

kernel void matmul_f32(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* o [[buffer(2)]],
                       constant uint3& mkn [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint M = mkn.x; uint K = mkn.y; uint N = mkn.z;
    if (gid.x >= M || gid.y >= N) return;
    float acc = 0.0;
    for (uint kk = 0; kk < K; kk++) {
        acc += a[gid.x * K + kk] * b[kk * N + gid.y];
    }
    o[gid.x * N + gid.y] = acc;
}
"#;

pub struct MetalContext {
    device: Arc<Device>,
    queue: Arc<CommandQueue>,
    pipelines: std::collections::HashMap<&'static str, ComputePipelineState>,
}

impl MetalContext {
    pub fn new() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or(BackendError::Init("no Metal device"))?;
        let queue = device.new_command_queue();
        let opts = metal::CompileOptions::new();
        let library: Library = device
            .new_library_with_source(SHADER_SRC, &opts)
            .map_err(|_| BackendError::Init("metal compile failed"))?;
        let mut pipelines = std::collections::HashMap::new();
        for entry in ["add_f32", "mul_f32", "relu_f32", "matmul_f32"] {
            let func = library
                .get_function(entry, None)
                .map_err(|_| BackendError::Init("metal function lookup failed"))?;
            let pipe = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|_| BackendError::Init("metal pipeline creation failed"))?;
            pipelines.insert(entry, pipe);
        }
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines,
        })
    }
}

pub struct MetalBackend<W: Workspace> {
    ctx: MetalContext,
    _ws: PhantomData<W>,
}

impl<W: Workspace> MetalBackend<W> {
    pub fn new() -> Result<Self, BackendError> {
        Ok(Self {
            ctx: MetalContext::new()?,
            _ws: PhantomData,
        })
    }

    fn dispatch_compute(
        &self,
        entry: &'static str,
        a: &[u8],
        b: Option<&[u8]>,
        out_len: usize,
        params: &[u8],
        groups: MTLSize,
        threads: MTLSize,
    ) -> Vec<u8> {
        let dev = &self.ctx.device;
        // Every caller passes a compile-time-constant entry name compiled into
        // the pipeline set at context creation; a miss is an internal invariant
        // violation — fail loud rather than silently return a zero buffer that
        // would masquerade as a valid GPU result for an unimplemented op.
        let pipe = self
            .ctx
            .pipelines
            .get(entry)
            .unwrap_or_else(|| panic!("metal pipeline '{entry}' not compiled"));
        let buf_a = dev.new_buffer_with_data(
            a.as_ptr() as *const _,
            a.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buf_b: Buffer = if let Some(bb) = b {
            dev.new_buffer_with_data(
                bb.as_ptr() as *const _,
                bb.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        } else {
            dev.new_buffer(4, MTLResourceOptions::StorageModeShared)
        };
        let buf_o = dev.new_buffer(out_len.max(4) as u64, MTLResourceOptions::StorageModeShared);
        let buf_p = dev.new_buffer_with_data(
            params.as_ptr() as *const _,
            params.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd = self.ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_o), 0);
        enc.set_buffer(3, Some(&buf_p), 0);
        enc.dispatch_thread_groups(groups, threads);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let p = buf_o.contents() as *const u8;
        unsafe { std::slice::from_raw_parts(p, out_len) }.to_vec()
    }
}

impl<W: Workspace> Backend for MetalBackend<W> {
    type Bounds = HologramHostBoundsMetal;
    type WS = W;

    fn dispatch(&mut self, call: &KernelCall, ws: &mut Self::WS) -> Result<(), BackendError> {
        match call {
            KernelCall::Add(c) if is_float(c.dtype) => run_binary(self, ws, c, "add_f32"),
            KernelCall::Mul(c) if is_float(c.dtype) => run_binary(self, ws, c, "mul_f32"),
            KernelCall::Relu(c) if is_float(c.dtype) => run_unary(self, ws, c, "relu_f32"),
            KernelCall::MatMul(c) if is_float(c.dtype) => run_matmul(self, ws, c),
            _ => crate::cpu::CpuBackend::<Self::WS>::new().dispatch(call, ws),
        }
    }
}

fn run_unary<W: Workspace>(
    be: &MetalBackend<W>,
    ws: &mut W,
    c: &UnaryCall,
    entry: &'static str,
) -> Result<(), BackendError> {
    let n = c.element_count;
    let bytes = (n as usize) * 4;
    let a = ws
        .read(c.input)
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.input.slot))?
        .to_vec();
    let n_bytes = n.to_le_bytes();
    let groups = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
    let threads = MTLSize::new(256, 1, 1);
    let result = be.dispatch_compute(entry, &a, None, bytes, &n_bytes, groups, threads);
    let out = ws.write(c.output);
    let take = bytes.min(out.len()).min(result.len());
    out[..take].copy_from_slice(&result[..take]);
    Ok(())
}

fn run_binary<W: Workspace>(
    be: &MetalBackend<W>,
    ws: &mut W,
    c: &BinaryCall,
    entry: &'static str,
) -> Result<(), BackendError> {
    let n = c.element_count;
    let bytes = (n as usize) * 4;
    let a = ws
        .read(c.a)
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws
        .read(c.b)
        .get(..bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let n_bytes = n.to_le_bytes();
    let groups = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
    let threads = MTLSize::new(256, 1, 1);
    let result = be.dispatch_compute(entry, &a, Some(&b), bytes, &n_bytes, groups, threads);
    let out = ws.write(c.output);
    let take = bytes.min(out.len()).min(result.len());
    out[..take].copy_from_slice(&result[..take]);
    Ok(())
}

fn run_matmul<W: Workspace>(
    be: &MetalBackend<W>,
    ws: &mut W,
    c: &MatMulCall,
) -> Result<(), BackendError> {
    let m = c.m as usize;
    let k = c.k as usize;
    let n = c.n as usize;
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let a_bytes = m * k * 4;
    let b_bytes = k * n * 4;
    let out_bytes = m * n * 4;
    let a = ws
        .read(c.a)
        .get(..a_bytes)
        .ok_or(BackendError::SlotOutOfRange(c.a.slot))?
        .to_vec();
    let b = ws
        .read(c.b)
        .get(..b_bytes)
        .ok_or(BackendError::SlotOutOfRange(c.b.slot))?
        .to_vec();
    let mut params = Vec::with_capacity(12);
    params.extend_from_slice(&(m as u32).to_le_bytes());
    params.extend_from_slice(&(k as u32).to_le_bytes());
    params.extend_from_slice(&(n as u32).to_le_bytes());
    let groups = MTLSize::new(((m as u64) + 7) / 8, ((n as u64) + 7) / 8, 1);
    let threads = MTLSize::new(8, 8, 1);
    let result = be.dispatch_compute(
        "matmul_f32",
        &a,
        Some(&b),
        out_bytes,
        &params,
        groups,
        threads,
    );
    let out = ws.write(c.output);
    let take = out_bytes.min(out.len()).min(result.len());
    out[..take].copy_from_slice(&result[..take]);
    Ok(())
}
