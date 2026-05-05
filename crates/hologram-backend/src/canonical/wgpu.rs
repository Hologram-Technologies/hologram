//! Canonical WebGPU backend (Phase 3.5 — partial coverage).
//!
//! `WgpuBackend` implements [`CanonicalBackend`] over a single
//! `wgpu::Device` + `wgpu::Queue` pair. Each call uploads its inputs
//! into fresh storage buffers, dispatches one compute pipeline, and
//! reads the result back into the host workspace. (A future revision
//! can keep the workspace device-resident across calls and only copy
//! at plan boundaries; today the focus is correctness, not transfer
//! cost.)
//!
//! Coverage today:
//!   - **Binary elementwise**: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`.
//!   - **Unary elementwise**: `Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`,
//!     `Sqrt`, `Abs`, `Reciprocal`, `Sin`, `Cos`, `Floor`, `Ceil`,
//!     `Round`, `Sign`, `Silu`.
//!   - **`Neg`** (via the unary path).
//!
//! All other variants return [`ExecError::Backend`] with the variant
//! name. New op coverage = one WGSL fragment + one dispatch arm +
//! conformance-test stanza. The harness pinpoints the first divergent
//! index whenever a shader doesn't match the canonical reference.
//!
//! [`CanonicalBackend`]: hologram_transform::CanonicalBackend
//! [`ExecError::Backend`]: hologram_transform::ExecError::Backend

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

use hologram_transform::{
    AddCall, AddGradCall, AddRmsNormCall, AddRmsNormGradCall, AttentionCall, BackendWorkspace,
    BinaryCall, CanonicalBackend, ConcatCall, Conv2dCall, ConvTransposeCall, ExecError,
    GlobalAvgPoolCall, GroupNormCall, InstanceNormGradCall, KernelCall, LayerNormGradCall,
    MatMulCall, MatMulGradACall, MatMulGradBCall, NegGradCall, NormFullCall, NormScaleCall,
    Pool2dCall, Pool2dKind, ReduceCall, ReduceKind, ReshapeCall, RmsNormGradCall, SliceCall,
    SlotSpan, SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind, SubGradCall, UnaryCall, UnaryKind,
};

/// Device-resident workspace for `WgpuBackend` per ADR-051.
///
/// Owns a single `wgpu::Buffer` of `capacity * 4` bytes plus a
/// reusable staging buffer for readback. `write_span` uploads via
/// `Queue::write_buffer` (no staging copy on the host side);
/// `read_span` issues a copy command into the cached staging buffer,
/// maps it, and reads the slice back.
///
/// The staging buffer is sized for the full workspace and reused
/// across every `read_span` call, so we don't pay buffer-allocation
/// cost per readback.
///
/// Today's `WgpuBackend::dispatch_resident` (added in ADR-051 step 2)
/// downloads the full workspace, runs the existing per-call
/// upload/download path, and uploads the result. That's no faster
/// than the legacy `dispatch(&mut [f32], …)` route; the per-arm
/// migration in ADR-051 step 3 binds the workspace buffer directly so
/// dispatches never round-trip through the host.
pub struct WgpuWorkspace {
    /// The single device-resident buffer. Sized at `capacity_elements
    /// * 4` bytes.
    buffer: wgpu::Buffer,
    /// Pre-allocated staging buffer for readback. Sized to match
    /// `buffer`, so any `read_span` fits without re-allocation.
    /// Reused across `read_span` calls — `unmap` after each read
    /// returns it to a writable state.
    staging: wgpu::Buffer,
    /// Element capacity, in `f32` slots.
    capacity_elements: usize,
    /// Cloned device handle. Needed by `read_span` to submit the
    /// copy/map commands without holding a borrow back to `WgpuBackend`.
    device: wgpu::Device,
    /// Cloned queue handle. Needed for `write_buffer` uploads and to
    /// submit the readback command encoder.
    queue: wgpu::Queue,
    /// Per-workspace bind-group cache. Keyed on the call's op label
    /// plus the offset/size tuple of every workspace binding plus the
    /// payload of any uniform binding (uniform buffers themselves are
    /// deduped at the backend level — same payload means same buffer
    /// — so the payload uniquely identifies the uniform participant).
    /// Inference loops repeat the same span shapes per timestep, so
    /// the cache hits on every dispatch after the first warm-up step.
    bind_group_cache: RefCell<HashMap<BindGroupKey, wgpu::BindGroup>>,
}

/// Cache key for a workspace-resident bind group.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    op: &'static str,
    /// `(offset, size)` per workspace-bound storage entry, in declared
    /// binding order. Uniform entries are not included — their value
    /// is captured in `uniform`.
    slots: Vec<(u64, u64)>,
    /// `(op, payload)` of any uniform binding, or `None` when the
    /// layout has no uniform. Uses the same key shape as
    /// [`UniformKey`] so identical params resolve to one bind group.
    uniform: Option<UniformKey>,
}

impl WgpuWorkspace {
    /// Allocate a fresh device buffer sized for the plan.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity_elements: usize) -> Self {
        let size_bytes = (capacity_elements * 4).max(4) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hologram-wgpu-workspace"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hologram-wgpu-workspace-staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            staging,
            capacity_elements,
            device: device.clone(),
            queue: queue.clone(),
            bind_group_cache: RefCell::new(HashMap::new()),
        }
    }

    /// Look up — or build and cache — a bind group sized for this
    /// workspace. The closure builds the [`wgpu::BindGroup`] on cache
    /// miss; otherwise the cached one is cloned (cheap, refcounted).
    fn cached_bind_group(
        &self,
        key: BindGroupKey,
        build: impl FnOnce() -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        if let Some(bg) = self.bind_group_cache.borrow().get(&key) {
            return bg.clone();
        }
        let bg = build();
        self.bind_group_cache.borrow_mut().insert(key, bg.clone());
        bg
    }

    /// Borrow the device buffer for bind-group construction in
    /// `dispatch_resident` arms (used after the per-arm migration).
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Issue a span readback and return a [`ReadSpanFuture`] that
    /// resolves to the contents. Unlike [`BackendWorkspace::read_span`]
    /// (which blocks via `device.poll(Wait)` until the readback
    /// completes), the async variant submits the copy + `map_async`
    /// up front and yields to the executor while polling the device
    /// non-blockingly. The caller can submit further work to the GPU
    /// (a next forward pass, another readback) before awaiting — those
    /// dispatches start running on the device in parallel with the
    /// pending readback's GPU-side copy + map.
    ///
    /// At a steady-state decode loop this hides the ~1.3 ms readback
    /// floor behind the next token's compute.
    pub fn read_span_async(&self, span: SlotSpan) -> ReadSpanFuture<'_> {
        let bytes = (span.len * 4) as u64;
        if bytes == 0 {
            return ReadSpanFuture::ready(Ok(Vec::new()));
        }
        let end = span.offset + span.len;
        if end > self.capacity_elements {
            return ReadSpanFuture::ready(Err(ExecError::WorkspaceMismatch {
                expected: end,
                actual: self.capacity_elements,
            }));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hologram-wgpu-readback-async-enc"),
            });
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            (span.offset * 4) as u64,
            &self.staging,
            0,
            bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = self.staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ReadSpanFuture {
            inner: ReadSpanFutureInner::Pending {
                device: &self.device,
                staging: &self.staging,
                bytes,
                rx,
            },
        }
    }
}

/// Future returned by [`WgpuWorkspace::read_span_async`]. Polling
/// drives the device with `wgpu::Maintain::Poll` (non-blocking) and
/// yields until the readback's `map_async` callback fires; only then
/// is the host vector materialised.
pub struct ReadSpanFuture<'a> {
    inner: ReadSpanFutureInner<'a>,
}

enum ReadSpanFutureInner<'a> {
    /// Already resolved (zero-length span or an upfront error).
    Done(Option<Result<Vec<f32>, ExecError>>),
    /// Awaiting the GPU readback callback.
    Pending {
        device: &'a wgpu::Device,
        staging: &'a wgpu::Buffer,
        bytes: u64,
        rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    },
}

impl<'a> ReadSpanFuture<'a> {
    fn ready(result: Result<Vec<f32>, ExecError>) -> Self {
        Self {
            inner: ReadSpanFutureInner::Done(Some(result)),
        }
    }
}

impl<'a> std::future::Future for ReadSpanFuture<'a> {
    type Output = Result<Vec<f32>, ExecError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        match &mut this.inner {
            ReadSpanFutureInner::Done(slot) => {
                std::task::Poll::Ready(slot.take().unwrap_or_else(|| {
                    Err(ExecError::Backend(
                        "ReadSpanFuture polled after completion".into(),
                    ))
                }))
            }
            ReadSpanFutureInner::Pending {
                device,
                staging,
                bytes,
                rx,
            } => {
                // Push the wgpu device a tick so the queued copy /
                // mapping callback can make progress without blocking
                // the executor thread.
                device.poll(wgpu::Maintain::Poll);
                match rx.try_recv() {
                    Ok(map_result) => {
                        if let Err(e) = map_result {
                            return std::task::Poll::Ready(Err(ExecError::Backend(format!(
                                "buffer map: {e:?}"
                            ))));
                        }
                        let slice = staging.slice(0..*bytes);
                        let mapped = slice.get_mapped_range();
                        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
                        drop(mapped);
                        staging.unmap();
                        std::task::Poll::Ready(Ok(out))
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        // Re-arm immediately. wgpu doesn't expose a
                        // wakeup primitive tied to the map callback,
                        // so we cooperatively re-poll on the next tick;
                        // the executor decides when to schedule us.
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => std::task::Poll::Ready(
                        Err(ExecError::Backend("readback channel disconnected".into())),
                    ),
                }
            }
        }
    }
}

impl BackendWorkspace for WgpuWorkspace {
    fn capacity(&self) -> usize {
        self.capacity_elements
    }

    fn write_span(&mut self, span: SlotSpan, data: &[f32]) -> Result<(), ExecError> {
        if data.len() != span.len {
            return Err(ExecError::Backend(format!(
                "WgpuWorkspace::write_span: data.len()={} != span.len={}",
                data.len(),
                span.len
            )));
        }
        let end = span.offset + span.len;
        if end > self.capacity_elements {
            return Err(ExecError::WorkspaceMismatch {
                expected: end,
                actual: self.capacity_elements,
            });
        }
        let byte_offset = (span.offset * 4) as u64;
        self.queue
            .write_buffer(&self.buffer, byte_offset, bytemuck::cast_slice(data));
        // No explicit submit needed: `Queue::write_buffer` is queued
        // and is flushed before the commands of the next submission.
        // Every consumer (`dispatch_resident`, `dispatch`, `read_span`)
        // performs its own submit, so the write becomes visible at
        // the right point without a per-write empty submit.
        Ok(())
    }

    fn read_span(&self, span: SlotSpan) -> Result<Vec<f32>, ExecError> {
        let end = span.offset + span.len;
        if end > self.capacity_elements {
            return Err(ExecError::WorkspaceMismatch {
                expected: end,
                actual: self.capacity_elements,
            });
        }
        let bytes = (span.len * 4) as u64;
        if bytes == 0 {
            return Ok(Vec::new());
        }
        // Reuse the workspace's pre-allocated staging buffer. The
        // staging buffer is sized for the full workspace, so any
        // span fits at offset 0.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hologram-wgpu-readback-enc"),
            });
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            (span.offset * 4) as u64,
            &self.staging,
            0,
            bytes,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| ExecError::Backend(format!("readback channel: {e}")))?
            .map_err(|e| ExecError::Backend(format!("buffer map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        Ok(out)
    }
}

// ── Helper arg structs ──────────────────────────────────────────────────
//
// These bundle the scalar parameters that would otherwise push internal
// helpers past `clippy::too_many_arguments`. The project rule forbids
// `#[allow]`-ing that lint, so each helper either takes a `&CallStruct`
// from `hologram-transform` directly or one of the small builder-style
// structs below. All the structs are private (`pub(crate)` would also
// work) — they exist purely to keep call sites readable.

/// Inputs for the shared 2-input (no-bias) norm-grad helpers
/// (`run_norm_grad2` / `run_norm_grad2_resident`). Both `RmsNormGrad`
/// and `InstanceNormGrad` reduce to the same WGSL shape; the
/// `dx_op` / `dw_op` pipeline names select which kernel runs.
#[derive(Debug, Clone, Copy)]
struct NormGrad2Args {
    input: SlotSpan,
    weight: SlotSpan,
    dy: SlotSpan,
    dx: SlotSpan,
    dw: SlotSpan,
    size: u32,
    /// `f32::to_bits()` of the stabilisation epsilon — packed straight
    /// into the WGSL uniform so the shader sees a `f32` field.
    epsilon_bits: u32,
    dx_op: &'static str,
    dw_op: &'static str,
}

impl NormGrad2Args {
    fn from_rms(call: &RmsNormGradCall) -> Self {
        Self {
            input: call.input,
            weight: call.weight,
            dy: call.dy,
            dx: call.dx,
            dw: call.dw,
            size: call.size,
            epsilon_bits: call.epsilon,
            dx_op: "rms_norm_grad_dx",
            dw_op: "rms_norm_grad_dw",
        }
    }

    fn from_instance(call: &InstanceNormGradCall) -> Self {
        Self {
            input: call.input,
            weight: call.weight,
            dy: call.dy,
            dx: call.dx,
            dw: call.dw,
            size: call.size,
            epsilon_bits: call.epsilon,
            dx_op: "instance_norm_grad_dx",
            dw_op: "instance_norm_grad_dw",
        }
    }
}

/// Inputs for the shared MatMul resident dispatch helper
/// (`run_matmul_resident_inner`). Forward + both grad arms take the
/// same shape; only the pipeline name and which slot maps to which
/// role differ.
#[derive(Debug, Clone, Copy)]
struct MatMulResidentArgs {
    /// First storage binding (a / dc / a depending on the pipeline).
    slot_a: SlotSpan,
    /// Second storage binding (b / b / dc).
    slot_b: SlotSpan,
    /// Third storage binding (c / da / db) — also the workgroup target.
    slot_c: SlotSpan,
    m: usize,
    k: usize,
    n: usize,
    pipeline_name: &'static str,
    out_total: usize,
}

impl MatMulResidentArgs {
    fn forward(call: &MatMulCall) -> Self {
        Self {
            slot_a: call.a,
            slot_b: call.b,
            slot_c: call.c,
            m: call.m,
            k: call.k,
            n: call.n,
            pipeline_name: "matmul",
            out_total: call.m * call.n,
        }
    }

    fn grad_a(call: &MatMulGradACall) -> Self {
        Self {
            slot_a: call.dc,
            slot_b: call.b,
            slot_c: call.da,
            m: call.m,
            k: call.k,
            n: call.n,
            pipeline_name: "matmul_grad_a",
            out_total: call.m * call.k,
        }
    }

    fn grad_b(call: &MatMulGradBCall) -> Self {
        Self {
            slot_a: call.a,
            slot_b: call.dc,
            slot_c: call.db,
            m: call.m,
            k: call.k,
            n: call.n,
            pipeline_name: "matmul_grad_b",
            out_total: call.k * call.n,
        }
    }
}

/// Convolution shape parameters packed into the `ConvParams` uniform.
/// Built from a `Conv2dCall` or `ConvTransposeCall` plus the resolved
/// `group` value (`call.group.max(1)` — both calls allow 0 to mean 1).
#[derive(Debug, Clone, Copy)]
struct ConvUniformParams {
    n: u32,
    c_in: u32,
    c_out: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    group: u32,
}

impl ConvUniformParams {
    fn from_conv2d(call: &Conv2dCall, group: u32) -> Self {
        Self {
            n: call.n,
            c_in: call.c_in,
            c_out: call.c_out,
            h_in: call.h_in,
            w_in: call.w_in,
            h_out: call.h_out,
            w_out: call.w_out,
            kernel_h: call.kernel_h,
            kernel_w: call.kernel_w,
            stride_h: call.stride_h,
            stride_w: call.stride_w,
            pad_h: call.pad_h,
            pad_w: call.pad_w,
            dilation_h: call.dilation_h,
            dilation_w: call.dilation_w,
            group,
        }
    }

    fn from_conv_transpose(call: &ConvTransposeCall, group: u32) -> Self {
        Self {
            n: call.n,
            c_in: call.c_in,
            c_out: call.c_out,
            h_in: call.h_in,
            w_in: call.w_in,
            h_out: call.h_out,
            w_out: call.w_out,
            kernel_h: call.kernel_h,
            kernel_w: call.kernel_w,
            stride_h: call.stride_h,
            stride_w: call.stride_w,
            pad_h: call.pad_h,
            pad_w: call.pad_w,
            dilation_h: call.dilation_h,
            dilation_w: call.dilation_w,
            group,
        }
    }

    /// Pack into the 20-u32 layout expected by the `ConvParams` WGSL
    /// uniform. The trailing four slots are padding for 16-byte
    /// alignment (`std140`-style).
    fn pack(&self) -> [u32; 20] {
        [
            self.n,
            self.c_in,
            self.c_out,
            self.h_in,
            self.w_in,
            self.h_out,
            self.w_out,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.group,
            0,
            0,
            0,
            0,
        ]
    }
}

/// Borrowed bundle for the legacy upload→dispatch→read helper. The
/// four resources travel together at every call site, so collapsing
/// them into one struct drops the helper's arity from 7 to 4.
#[derive(Copy, Clone)]
struct DispatchAndRead<'a> {
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    out_buf: &'a wgpu::Buffer,
    staging: &'a wgpu::Buffer,
}

/// WebGPU canonical backend.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    binary_bind_layout: wgpu::BindGroupLayout,
    unary_bind_layout: wgpu::BindGroupLayout,
    reduce_bind_layout: wgpu::BindGroupLayout,
    norm2_bind_layout: wgpu::BindGroupLayout,
    norm3_bind_layout: wgpu::BindGroupLayout,
    matmul_bind_layout: wgpu::BindGroupLayout,
    conv_bind_layout: wgpu::BindGroupLayout,
    norm_grad2_bind_layout: wgpu::BindGroupLayout,
    norm_grad3_bind_layout: wgpu::BindGroupLayout,
    norm_grad_addrms_bind_layout: wgpu::BindGroupLayout,
    softmax_grad_bind_layout: wgpu::BindGroupLayout,
    binary_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    unary_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    reduce_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    softmax_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    norm_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    matmul_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    pool_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    conv_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    norm_grad_pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    /// Workgroup-cooperative attention shaders for the decode-shape
    /// case (single-batch, single-head, `seq_q == 1`,
    /// `seq_kv <= ATTENTION_DECODE_MAX_SEQ_KV`). Two pipelines tuned
    /// for narrow vs wide `head_dim` — the dispatcher picks one
    /// based on `head_dim` vs [`ATTENTION_DECODE_LARGE_THRESHOLD`].
    /// Calls outside the supported envelope fall through to host
    /// fallback.
    ///
    /// The **large** pipeline declares 18 KB of workgroup memory.
    /// On Apple-Metal this declaration affects scheduling for *every*
    /// dispatch on the device, not just dispatches that use the
    /// pipeline — measured empirically: compiling the large pipeline
    /// at startup roughly doubles per-dispatch latency for unrelated
    /// matmul/softmax/etc. So we lazy-compile it on first use; if a
    /// run never hits a large-shape Attention call, the large
    /// pipeline never exists and the rest of the device runs full
    /// speed. The `RefCell` is the existing single-threaded
    /// concurrency story — `WgpuBackend` is already not `Send`.
    attention_decode_small_pipeline: wgpu::ComputePipeline,
    attention_decode_large_pipeline: RefCell<Option<wgpu::ComputePipeline>>,
    attention_decode_bind_layout: wgpu::BindGroupLayout,
    /// Active command encoder when a `run_resident` walk is in flight.
    /// Migrated `dispatch_resident` arms record into this encoder via
    /// [`Self::record_or_submit`] instead of creating + submitting their
    /// own. `None` means standalone dispatch (each call submits itself).
    pending_encoder: RefCell<Option<wgpu::CommandEncoder>>,
    /// Uniform-buffer cache keyed by op + shape so identical shapes
    /// share one device buffer across dispatches (matmul `(m, k, n)`,
    /// norm `(size, eps)`, conv full `Pool2dCall`/`ConvParams`, …).
    uniform_cache: RefCell<HashMap<UniformKey, wgpu::Buffer>>,
}

/// Cache key for shape-keyed uniform buffers. `op` distinguishes
/// shaders that pack the same `Vec<u32>` differently (`matmul.params`
/// vs `norm.params`). Stored in the `WgpuBackend` and shared across
/// all dispatches and workspaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UniformKey {
    op: &'static str,
    payload: Vec<u32>,
}

impl WgpuBackend {
    /// Synchronously initialise an adapter, device, queue, and the
    /// compiled compute pipelines. Blocks the calling thread on the
    /// underlying async wgpu APIs via `pollster`.
    pub fn new() -> Result<Self, ExecError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, ExecError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| ExecError::Backend("wgpu: no compatible adapter".into()))?;

        // Pick limits up to whatever the adapter reports — some
        // canonical kernels (norm grads) bind 5 storage buffers,
        // exceeding the `downlevel_defaults` cap of 4. The adapter's
        // actual limits are typically much higher (≥8 storage buffers
        // on Metal / Vulkan / WebGPU 1.0). Also lift the binding-size
        // cap so individual weight matrices (e.g. LLaMA-2-7B's
        // 4096×11008 = 180 MB) can be bound without splitting.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("hologram-canonical-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| ExecError::Backend(format!("wgpu: device request failed: {e}")))?;

        // ADR-051 step 3: all three binary bindings are declared as
        // read_write at the layout level so the resident path can bind
        // the same workspace buffer at three offsets within a single
        // dispatch (wgpu's usage-scope rule rejects mixing read-only
        // and read-write usages of the same buffer in one pass). The
        // shader-level `var<storage, read>` qualifier still enforces
        // immutability for inputs at the WGSL access level.
        let binary_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("binary.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                ],
            });
        // ADR-051 step 3: input + output marked read_write so the
        // resident path can bind the workspace buffer at both offsets
        // within one dispatch (wgpu usage-scope rule).
        let unary_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary.binds.layout"),
            entries: &[storage_binding(0, false), storage_binding(1, false)],
        });
        // ADR-051 step 3: input + output both read_write so the
        // resident path can bind workspace buffer windows for both.
        let reduce_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reduce.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    uniform_binding(2),
                ],
            });
        // Norm-with-scale layout: input, weight, output, params.
        // ADR-051 step 3: all storage bindings read_write so the
        // resident path can bind workspace-buffer windows.
        let norm2_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("norm2.binds.layout"),
            entries: &[
                storage_binding(0, false),
                storage_binding(1, false),
                storage_binding(2, false),
                uniform_binding(3),
            ],
        });
        // MatMul + both grad variants: two storage inputs (a, b),
        // one storage target (overwrite for forward, accumulate for
        // grad), one uniform `Params { m, k, n }`. ADR-051 step 3:
        // all storage bindings declared read_write so the resident
        // path can bind workspace windows for all three.
        let matmul_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    uniform_binding(3),
                ],
            });
        // RmsNormGrad / InstanceNormGrad: input + weight + dy + dx
        // (rw, accumulating) + dw (rw, accumulating) + uniform. Both
        // grads share the math signature (no bias).
        // ADR-051 step 3: all storage bindings read_write so the
        // resident path can bind workspace windows for input/weight/dy
        // alongside dx/dw within one dispatch.
        let norm_grad2_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("norm_grad2.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    storage_binding(3, false),
                    storage_binding(4, false),
                    uniform_binding(5),
                ],
            });
        // LayerNormGrad: same shape as norm_grad2 plus a `db`
        // accumulator at slot 5. 6 storage + uniform = 7 bindings.
        let norm_grad3_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("norm_grad3.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    storage_binding(3, false),
                    storage_binding(4, false),
                    storage_binding(5, false),
                    uniform_binding(6),
                ],
            });
        // SoftmaxGrad: forward output (read) + dC (read) + dA (rw) +
        // uniform `Params { size }`. Same shape covers both Softmax
        // and LogSoftmax variants — kind selects the WGSL pipeline.
        let softmax_grad_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("softmax_grad.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    uniform_binding(3),
                ],
            });
        // AddRmsNormGrad: residual + input + weight + dy + d_residual
        // (rw) + d_input (rw) + dw (rw) + uniform = 8 bindings.
        let norm_grad_addrms_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("norm_grad_addrms.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    storage_binding(3, false),
                    storage_binding(4, false),
                    storage_binding(5, false),
                    storage_binding(6, false),
                    uniform_binding(7),
                ],
            });
        // Conv2d forward: input + weight + bias + output + uniform
        // ConvParams. Bias is always bound (a length-`c_out` zero
        // buffer when the call carries no bias) so the bind layout
        // can stay shape-fixed.
        // ADR-051 step 3: all storage bindings read_write so the
        // resident path can bind workspace buffer windows for
        // input/weight/bias/output in one dispatch.
        let conv_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv.binds.layout"),
            entries: &[
                storage_binding(0, false),
                storage_binding(1, false),
                storage_binding(2, false),
                storage_binding(3, false),
                uniform_binding(4),
            ],
        });
        // Norm-with-3-inputs layout: covers LayerNorm (input, weight,
        // bias) and AddRmsNorm (residual, input, weight). Kernels
        // assign meanings to the bindings; the layout is shape-only.
        // ADR-051 step 3: all storage bindings read_write so the
        // resident path can bind workspace-buffer windows.
        let norm3_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("norm3.binds.layout"),
            entries: &[
                storage_binding(0, false),
                storage_binding(1, false),
                storage_binding(2, false),
                storage_binding(3, false),
                uniform_binding(4),
            ],
        });
        // Attention (decode shape) layout: q, k, v, out, params.
        let attention_decode_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("attention_decode.binds.layout"),
                entries: &[
                    storage_binding(0, false),
                    storage_binding(1, false),
                    storage_binding(2, false),
                    storage_binding(3, false),
                    uniform_binding(4),
                ],
            });

        let mut binary_pipelines = HashMap::new();
        for (name, expr) in BINARY_OPS {
            binary_pipelines.insert(
                *name,
                build_pipeline(
                    &device,
                    &binary_bind_layout,
                    name,
                    &binary_shader_source(expr),
                ),
            );
        }
        let mut unary_pipelines = HashMap::new();
        for (name, expr) in UNARY_OPS {
            unary_pipelines.insert(
                *name,
                build_pipeline(
                    &device,
                    &unary_bind_layout,
                    name,
                    &unary_shader_source(expr),
                ),
            );
        }
        let mut reduce_pipelines = HashMap::new();
        for (name, init, fold, finish) in REDUCE_OPS {
            reduce_pipelines.insert(
                *name,
                build_pipeline(
                    &device,
                    &reduce_bind_layout,
                    name,
                    &reduce_shader_source(init, fold, finish),
                ),
            );
        }
        // Softmax shares the row-bind layout (storage, storage,
        // uniform size) but the kernel makes three passes over each
        // row (max → exp+sum → finalise). Two distinct shaders cover
        // softmax / log-softmax; both reuse `reduce_bind_layout`.
        let mut softmax_pipelines = HashMap::new();
        softmax_pipelines.insert(
            "softmax",
            build_pipeline(&device, &reduce_bind_layout, "softmax", SOFTMAX_WGSL),
        );
        softmax_pipelines.insert(
            "log_softmax",
            build_pipeline(
                &device,
                &reduce_bind_layout,
                "log_softmax",
                LOG_SOFTMAX_WGSL,
            ),
        );
        let mut norm_pipelines = HashMap::new();
        norm_pipelines.insert(
            "rms_norm",
            build_pipeline(&device, &norm2_bind_layout, "rms_norm", RMS_NORM_WGSL),
        );
        norm_pipelines.insert(
            "instance_norm",
            build_pipeline(
                &device,
                &norm2_bind_layout,
                "instance_norm",
                INSTANCE_NORM_WGSL,
            ),
        );
        norm_pipelines.insert(
            "layer_norm",
            build_pipeline(&device, &norm3_bind_layout, "layer_norm", LAYER_NORM_WGSL),
        );
        norm_pipelines.insert(
            "add_rms_norm",
            build_pipeline(
                &device,
                &norm3_bind_layout,
                "add_rms_norm",
                ADD_RMS_NORM_WGSL,
            ),
        );
        let mut matmul_pipelines = HashMap::new();
        matmul_pipelines.insert(
            "matmul",
            build_pipeline(&device, &matmul_bind_layout, "matmul", MATMUL_WGSL),
        );
        matmul_pipelines.insert(
            "matmul_grad_a",
            build_pipeline(
                &device,
                &matmul_bind_layout,
                "matmul_grad_a",
                MATMUL_GRAD_A_WGSL,
            ),
        );
        matmul_pipelines.insert(
            "matmul_grad_b",
            build_pipeline(
                &device,
                &matmul_bind_layout,
                "matmul_grad_b",
                MATMUL_GRAD_B_WGSL,
            ),
        );
        // Pool family — all three forward variants reuse
        // `reduce_bind_layout` (storage in, storage out, uniform). The
        // uniform's struct fields differ per shader, but the binding
        // layout is byte-shape only.
        let mut pool_pipelines = HashMap::new();
        pool_pipelines.insert(
            "pool_max",
            build_pipeline(&device, &reduce_bind_layout, "pool_max", POOL_MAX_WGSL),
        );
        pool_pipelines.insert(
            "pool_avg",
            build_pipeline(&device, &reduce_bind_layout, "pool_avg", POOL_AVG_WGSL),
        );
        pool_pipelines.insert(
            "global_avg_pool",
            build_pipeline(
                &device,
                &reduce_bind_layout,
                "global_avg_pool",
                GLOBAL_AVG_POOL_WGSL,
            ),
        );
        let mut conv_pipelines = HashMap::new();
        conv_pipelines.insert(
            "conv2d",
            build_pipeline(&device, &conv_bind_layout, "conv2d", CONV2D_WGSL),
        );
        conv_pipelines.insert(
            "conv_transpose_2d",
            build_pipeline(
                &device,
                &conv_bind_layout,
                "conv_transpose_2d",
                CONV_TRANSPOSE_2D_WGSL,
            ),
        );
        // RmsNormGrad runs as two passes (dx + dw) on the same bind
        // layout. Each pass is its own pipeline; the dispatch helper
        // records both into a single command encoder.
        let mut norm_grad_pipelines = HashMap::new();
        for (name, source) in [
            ("rms_norm_grad_dx", RMS_NORM_GRAD_DX_WGSL),
            ("rms_norm_grad_dw", RMS_NORM_GRAD_DW_WGSL),
            ("instance_norm_grad_dx", INSTANCE_NORM_GRAD_DX_WGSL),
            ("instance_norm_grad_dw", INSTANCE_NORM_GRAD_DW_WGSL),
        ] {
            norm_grad_pipelines.insert(
                name,
                build_pipeline(&device, &norm_grad2_bind_layout, name, source),
            );
        }
        for (name, source) in [
            ("layer_norm_grad_dx", LAYER_NORM_GRAD_DX_WGSL),
            ("layer_norm_grad_dw_db", LAYER_NORM_GRAD_DW_DB_WGSL),
        ] {
            norm_grad_pipelines.insert(
                name,
                build_pipeline(&device, &norm_grad3_bind_layout, name, source),
            );
        }
        for (name, source) in [
            ("add_rms_norm_grad_dx", ADD_RMS_NORM_GRAD_DX_WGSL),
            ("add_rms_norm_grad_dw", ADD_RMS_NORM_GRAD_DW_WGSL),
        ] {
            norm_grad_pipelines.insert(
                name,
                build_pipeline(&device, &norm_grad_addrms_bind_layout, name, source),
            );
        }
        for (name, source) in [
            ("softmax_grad", SOFTMAX_GRAD_WGSL),
            ("log_softmax_grad", LOG_SOFTMAX_GRAD_WGSL),
        ] {
            norm_grad_pipelines.insert(
                name,
                build_pipeline(&device, &softmax_grad_bind_layout, name, source),
            );
        }
        let attention_decode_small_pipeline = build_pipeline(
            &device,
            &attention_decode_bind_layout,
            "attention_decode_small",
            ATTENTION_DECODE_SMALL_WGSL,
        );
        // Large pipeline is built lazily — see field doc.
        let attention_decode_large_pipeline = RefCell::new(None);

        Ok(Self {
            device,
            queue,
            binary_bind_layout,
            unary_bind_layout,
            reduce_bind_layout,
            norm2_bind_layout,
            norm3_bind_layout,
            matmul_bind_layout,
            conv_bind_layout,
            norm_grad2_bind_layout,
            norm_grad3_bind_layout,
            norm_grad_addrms_bind_layout,
            softmax_grad_bind_layout,
            binary_pipelines,
            unary_pipelines,
            reduce_pipelines,
            softmax_pipelines,
            norm_pipelines,
            matmul_pipelines,
            pool_pipelines,
            conv_pipelines,
            norm_grad_pipelines,
            attention_decode_small_pipeline,
            attention_decode_large_pipeline,
            attention_decode_bind_layout,
            pending_encoder: RefCell::new(None),
            uniform_cache: RefCell::new(HashMap::new()),
        })
    }
}

impl CanonicalBackend for WgpuBackend {
    // ADR-051 step 2: device-resident workspace landed. dispatch_resident
    // currently downloads → runs the legacy per-call dispatch on host →
    // uploads. Step 3 migrates each dispatch arm to bind the workspace
    // buffer directly so per-call transfers go away.
    type Workspace = WgpuWorkspace;

    fn alloc_workspace(&self, total_elements: usize) -> Result<Self::Workspace, ExecError> {
        Ok(WgpuWorkspace::new(
            &self.device,
            &self.queue,
            total_elements,
        ))
    }

    fn dispatch_resident(
        &mut self,
        ws: &mut Self::Workspace,
        call: &KernelCall,
    ) -> Result<(), ExecError> {
        // ADR-051 step 3: per-arm device-resident dispatch.
        //
        // Migrated arms bind the workspace buffer directly via
        // `BufferBinding`s for the call's slot spans — zero per-call
        // upload/download cost. Unmigrated arms fall through to the
        // legacy round-trip (download → host dispatch → upload).
        //
        // The binary family is migrated first since all 17 variants
        // share `run_binary_resident`. Unary, reduce, softmax, norm,
        // matmul, conv, attention, and grad arms migrate in follow-up
        // commits.
        match call {
            KernelCall::Add(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "add"),
            KernelCall::Sub(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "sub"),
            KernelCall::Mul(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "mul"),
            KernelCall::Div(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "div"),
            KernelCall::Min(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "min"),
            KernelCall::Max(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "max"),
            KernelCall::Pow(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "pow"),
            KernelCall::Mod(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "mod"),
            KernelCall::Equal(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "equal"),
            KernelCall::Less(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "less"),
            KernelCall::LessOrEqual(c) => {
                return self.run_binary_resident(ws, c.a, c.b, c.c, "less_or_equal")
            }
            KernelCall::Greater(c) => {
                return self.run_binary_resident(ws, c.a, c.b, c.c, "greater")
            }
            KernelCall::GreaterOrEqual(c) => {
                return self.run_binary_resident(ws, c.a, c.b, c.c, "greater_or_equal")
            }
            KernelCall::And(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "and"),
            KernelCall::Or(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "or"),
            KernelCall::Xor(c) => return self.run_binary_resident(ws, c.a, c.b, c.c, "xor"),
            KernelCall::FusedSwiGlu(c) => {
                return self.run_binary_resident(ws, c.a, c.b, c.c, "swiglu")
            }
            KernelCall::Unary(c, kind) => {
                if let Some(op) = unary_pipeline_key(*kind) {
                    return self.run_unary_resident(ws, c.input, c.output, op);
                }
                // Unmigrated unary kind — fall through to the slow path.
            }
            KernelCall::Reduce(c, kind) => return self.run_reduce_resident(ws, c, *kind),
            KernelCall::Softmax(c) => return self.run_softmax_resident(ws, c, "softmax"),
            KernelCall::LogSoftmax(c) => return self.run_softmax_resident(ws, c, "log_softmax"),
            KernelCall::Pool2d(c, kind) => return self.run_pool2d_resident(ws, c, *kind),
            KernelCall::GlobalAvgPool(c) => return self.run_global_avg_pool_resident(ws, c),
            KernelCall::RmsNorm(c) => return self.run_norm_scale_resident(ws, c, "rms_norm"),
            KernelCall::InstanceNorm(c) => {
                return self.run_norm_scale_resident(ws, c, "instance_norm")
            }
            KernelCall::LayerNorm(c) => return self.run_norm_full_resident(ws, c, "layer_norm"),
            KernelCall::AddRmsNorm(c) => return self.run_add_rms_norm_resident(ws, c),
            KernelCall::MatMul(c) => return self.run_matmul_resident(ws, c),
            KernelCall::MatMulGradA(c) => return self.run_matmul_grad_a_resident(ws, c),
            KernelCall::MatMulGradB(c) => return self.run_matmul_grad_b_resident(ws, c),
            KernelCall::Conv2d(c) => return self.run_conv2d_resident(ws, c),
            KernelCall::ConvTranspose2d(c) => return self.run_conv_transpose_2d_resident(ws, c),
            KernelCall::GroupNorm(c) => return self.run_group_norm_resident(ws, c),
            KernelCall::Reshape(c) => return self.run_reshape_resident(ws, c),
            KernelCall::Slice(c) => return self.run_slice_resident(ws, c),
            KernelCall::Concat(c) => return self.run_concat_resident(ws, c),
            KernelCall::SoftmaxGrad(c, kind) => {
                return self.run_softmax_grad_resident(ws, c, *kind);
            }
            KernelCall::RmsNormGrad(c) => {
                return self.run_norm_grad2_resident(ws, NormGrad2Args::from_rms(c));
            }
            KernelCall::InstanceNormGrad(c) => {
                return self.run_norm_grad2_resident(ws, NormGrad2Args::from_instance(c));
            }
            KernelCall::LayerNormGrad(c) => return self.run_layer_norm_grad_resident(ws, c),
            KernelCall::AddRmsNormGrad(c) => return self.run_add_rms_norm_grad_resident(ws, c),
            KernelCall::Attention(c) => {
                if Self::attention_decode_supports(c) {
                    return self.run_attention_decode_resident(ws, c);
                }
                // Unsupported shape — fall through to host fallback.
            }
            // Conv-grad, remaining grad arms, and a few data-movement
            // arms (Transpose, etc.) still go through the slow
            // round-trip path until they get dedicated resident helpers.
            _ => {}
        }
        // Fallback: round-trip the workspace through the host and run
        // the existing per-call dispatch path for unmigrated arms.
        // If a `run_resident` walk is in flight with batched recordings
        // queued, submit them first so the host sees the latest device
        // state before downloading.
        self.flush_pending_encoder();
        let full = SlotSpan {
            offset: 0,
            len: ws.capacity(),
        };
        let mut host = ws.read_span(full)?;
        self.dispatch(&mut host, call)?;
        ws.write_span(full, &host)?;
        Ok(())
    }

    /// Batched walk: parks a single `wgpu::CommandEncoder` in
    /// `pending_encoder` and lets each migrated `dispatch_resident`
    /// arm record into it via [`Self::record_or_submit`]. Unmigrated
    /// arms call [`Self::flush_pending_encoder`] before their host
    /// fallback, which yields the encoder, submits, and clears the
    /// slot — the next iteration installs a fresh encoder. The net
    /// effect is one device submit per *run* of consecutive migrated
    /// calls instead of one per call. For a transformer block tail
    /// (`matmul → add → rms_norm → matmul → softmax`) the submit
    /// count drops 5 → 1.
    fn run_resident(
        &mut self,
        ws: &mut Self::Workspace,
        calls: &[KernelCall],
    ) -> Result<(), ExecError> {
        if calls.is_empty() {
            return Ok(());
        }
        self.install_pending_encoder();
        for call in calls {
            self.dispatch_resident(ws, call)?;
            // Host-fallback consumed the encoder via flush_pending_encoder.
            // Reinstall a fresh one so subsequent migrated calls batch again.
            if self.pending_encoder.borrow().is_none() {
                self.install_pending_encoder();
            }
        }
        self.flush_pending_encoder();
        Ok(())
    }

    fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
        match call {
            KernelCall::Add(c) => self.dispatch_add(storage, c),
            KernelCall::Sub(c) => self.dispatch_binary(storage, c, "sub"),
            KernelCall::Mul(c) => self.dispatch_binary(storage, c, "mul"),
            KernelCall::Div(c) => self.dispatch_binary(storage, c, "div"),
            KernelCall::Min(c) => self.dispatch_binary(storage, c, "min"),
            KernelCall::Max(c) => self.dispatch_binary(storage, c, "max"),
            KernelCall::Pow(c) => self.dispatch_binary(storage, c, "pow"),
            KernelCall::Mod(c) => self.dispatch_binary(storage, c, "mod"),
            KernelCall::Equal(c) => self.dispatch_binary(storage, c, "equal"),
            KernelCall::Less(c) => self.dispatch_binary(storage, c, "less"),
            KernelCall::LessOrEqual(c) => self.dispatch_binary(storage, c, "less_or_equal"),
            KernelCall::Greater(c) => self.dispatch_binary(storage, c, "greater"),
            KernelCall::GreaterOrEqual(c) => self.dispatch_binary(storage, c, "greater_or_equal"),
            KernelCall::And(c) => self.dispatch_binary(storage, c, "and"),
            KernelCall::Or(c) => self.dispatch_binary(storage, c, "or"),
            KernelCall::Xor(c) => self.dispatch_binary(storage, c, "xor"),
            // Pure data-movement / accumulation arms — no math — handled
            // host-side. Promoting them to GPU shaders is straightforward
            // (one accumulating compute pass each) but adds two device
            // round-trips per call; deferred until a benchmark warrants.
            KernelCall::NegGrad(c) => Self::dispatch_neg_grad(storage, c),
            KernelCall::AddGrad(c) => Self::dispatch_add_grad(storage, c),
            KernelCall::SubGrad(c) => Self::dispatch_sub_grad(storage, c),
            KernelCall::Slice(c) => Self::dispatch_slice(storage, c),
            KernelCall::Concat(c) => Self::dispatch_concat(storage, c),
            KernelCall::Reshape(c) => Self::dispatch_reshape(storage, c),
            KernelCall::Unary(c, kind) => self.dispatch_unary(storage, c, *kind),
            KernelCall::Reduce(c, kind) => self.dispatch_reduce(storage, c, *kind),
            KernelCall::Softmax(c) => self.dispatch_softmax(storage, c, "softmax"),
            KernelCall::LogSoftmax(c) => self.dispatch_softmax(storage, c, "log_softmax"),
            KernelCall::RmsNorm(c) => self.dispatch_norm_scale(storage, c, "rms_norm"),
            KernelCall::InstanceNorm(c) => self.dispatch_norm_scale(storage, c, "instance_norm"),
            KernelCall::LayerNorm(c) => self.dispatch_norm_full(storage, c, "layer_norm"),
            KernelCall::AddRmsNorm(c) => self.dispatch_add_rms_norm(storage, c),
            KernelCall::GroupNorm(c) => self.dispatch_group_norm(storage, c),
            KernelCall::FusedSwiGlu(c) => self.dispatch_binary(storage, c, "swiglu"),
            KernelCall::MatMul(c) => self.dispatch_matmul(storage, c),
            KernelCall::MatMulGradA(c) => self.dispatch_matmul_grad_a(storage, c),
            KernelCall::MatMulGradB(c) => self.dispatch_matmul_grad_b(storage, c),
            KernelCall::Pool2d(c, kind) => self.dispatch_pool2d(storage, c, *kind),
            KernelCall::GlobalAvgPool(c) => self.dispatch_global_avg_pool(storage, c),
            KernelCall::Conv2d(c) => self.dispatch_conv2d(storage, c),
            KernelCall::ConvTranspose2d(c) => self.dispatch_conv_transpose_2d(storage, c),
            KernelCall::RmsNormGrad(c) => self.dispatch_rms_norm_grad(storage, c),
            KernelCall::InstanceNormGrad(c) => self.dispatch_instance_norm_grad(storage, c),
            KernelCall::LayerNormGrad(c) => self.dispatch_layer_norm_grad(storage, c),
            KernelCall::AddRmsNormGrad(c) => self.dispatch_add_rms_norm_grad(storage, c),
            KernelCall::SoftmaxGrad(c, kind) => self.dispatch_softmax_grad(storage, c, *kind),
            // ── CPU-fallback variants ──────────────────────────────────
            // Every remaining variant routes through the canonical CPU
            // reference. These are correct (the canonical CPU is the
            // semantic baseline per ADR-050) but bypass the device.
            // Promotion to a real WGSL shader is incremental: write
            // WGSL, replace the arm, validate via the conformance
            // harness. The exhaustive enumeration below means adding a
            // new `KernelCall` variant fails the build until this
            // match is updated — keeping the backend honest about
            // coverage gaps.
            KernelCall::MulGrad(_)
            | KernelCall::DivGrad(_)
            | KernelCall::PowGrad(_)
            | KernelCall::MinMaxGrad(_, _)
            | KernelCall::ReduceGrad(_, _)
            | KernelCall::ReduceArgGrad(_, _)
            | KernelCall::ReduceProdGrad(_)
            | KernelCall::UnaryGrad(_, _)
            | KernelCall::ConcatGrad(_)
            | KernelCall::SliceGrad(_)
            | KernelCall::TransposeGrad(_)
            | KernelCall::GroupNormGrad(_)
            | KernelCall::Pool2dGrad(_, _)
            | KernelCall::GlobalAvgPoolGrad(_)
            | KernelCall::FusedSwiGluGrad(_)
            | KernelCall::Conv2dGrad(_)
            | KernelCall::ConvTranspose2dGrad(_)
            | KernelCall::AttentionGrad(_)
            | KernelCall::Transpose(_)
            | KernelCall::Gemm(_)
            | KernelCall::Clip(_)
            | KernelCall::CumSum(_)
            | KernelCall::Pad(_)
            | KernelCall::Expand(_)
            | KernelCall::Where(_)
            | KernelCall::ResizeNearest(_)
            | KernelCall::ResizeLinear(_)
            | KernelCall::Lrn(_)
            | KernelCall::RotaryEmbedding(_)
            | KernelCall::Attention(_) => Self::host_cpu_fallback(storage, call),
        }
    }

    fn flush(&mut self) -> Result<(), ExecError> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }
}

impl WgpuBackend {
    fn dispatch_add(&self, storage: &mut [f32], call: &AddCall) -> Result<(), ExecError> {
        // `AddCall` and `BinaryCall` have the same shape; use the
        // shared binary path under the `add` pipeline.
        self.run_binary(storage, call.a, call.b, call.c, "add")
    }

    fn dispatch_binary(
        &self,
        storage: &mut [f32],
        call: &BinaryCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        self.run_binary(storage, call.a, call.b, call.c, op)
    }

    fn run_binary(
        &self,
        storage: &mut [f32],
        a: SlotSpan,
        b: SlotSpan,
        c: SlotSpan,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let n = a.len;
        if n == 0 {
            return Ok(());
        }
        if b.len != n || c.len != n {
            return Err(ExecError::Backend(format!(
                "WgpuBackend: {op} call has mismatched span lengths"
            )));
        }
        let pipeline = self
            .binary_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown binary op {op}")))?;

        let a_buf = self.upload(&storage[a.offset..a.offset + n], "binary.a");
        let b_buf = self.upload(&storage[b.offset..b.offset + n], "binary.b");
        let c_buf = self.alloc_storage(n, "binary.c");
        let staging = self.alloc_staging(n);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("binary.binds"),
            layout: &self.binary_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &c_buf,
                staging: &staging,
            },
            n,
            &mut storage[c.offset..c.offset + n],
        )
    }

    /// ADR-051 step 3: device-resident binary dispatch.
    ///
    /// Binds windows of the workspace buffer for the a/b/c spans
    /// directly — no per-call upload/download. Result stays on the
    /// device until the executor's [`BackendWorkspace::read_span`]
    /// pulls it back at a plan boundary.
    fn run_binary_resident(
        &self,
        ws: &WgpuWorkspace,
        a: SlotSpan,
        b: SlotSpan,
        c: SlotSpan,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let n = a.len;
        if n == 0 {
            return Ok(());
        }
        if b.len != n || c.len != n {
            return Err(ExecError::Backend(format!(
                "WgpuBackend: {op} call has mismatched span lengths"
            )));
        }
        let pipeline = self
            .binary_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown binary op {op}")))?;

        let bytes = (n * 4) as u64;
        let a_off = (a.offset * 4) as u64;
        let b_off = (b.offset * 4) as u64;
        let c_off = (c.offset * 4) as u64;
        let buffer = ws.buffer();

        let key = BindGroupKey {
            op,
            slots: vec![(a_off, bytes), (b_off, bytes), (c_off, bytes)],
            uniform: None,
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("binary.binds.resident"),
                layout: &self.binary_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: a_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: b_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: c_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "binary.resident",
            pipeline,
            &bind_group,
            n.div_ceil(64) as u32,
        );
        Ok(())
    }

    fn dispatch_reduce(
        &self,
        storage: &mut [f32],
        call: &ReduceCall,
        kind: ReduceKind,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: Reduce input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size;
        if call.output.len != rows {
            return Err(ExecError::Backend(
                "WgpuBackend: Reduce output rows mismatch".into(),
            ));
        }
        let op = reduce_pipeline_key(kind);
        let pipeline = self
            .reduce_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown reduce op {op}")))?;

        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "reduce.in",
        );
        let out_buf = self.alloc_storage(rows, "reduce.out");
        let staging = self.alloc_staging(rows);

        // Uniform buffer for the row size.
        use wgpu::util::DeviceExt;
        let params = [size as u32];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce.binds"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            rows,
            &mut storage[call.output.offset..call.output.offset + rows],
        )
    }

    /// ADR-051 step 3: device-resident Reduce dispatch.
    ///
    /// Binds workspace buffer windows for input (rows × size) and
    /// output (rows). The `size` uniform is built fresh each call —
    /// this is small (4 bytes) and the executor batches reductions at
    /// plan-build time, so the marginal cost is negligible.
    fn run_reduce_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &ReduceCall,
        kind: ReduceKind,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: Reduce input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size;
        if call.output.len != rows {
            return Err(ExecError::Backend(
                "WgpuBackend: Reduce output rows mismatch".into(),
            ));
        }
        let op = reduce_pipeline_key(kind);
        let pipeline = self
            .reduce_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown reduce op {op}")))?;

        let params_buf = self.cached_uniform("reduce.params", &[size as u32]);

        let in_bytes = (call.input.len * 4) as u64;
        let out_bytes = (rows * 4) as u64;
        let buffer = ws.buffer();
        let in_off = (call.input.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let key = BindGroupKey {
            op,
            slots: vec![(in_off, in_bytes), (out_off, out_bytes)],
            uniform: Some(UniformKey {
                op: "reduce.params",
                payload: vec![size as u32],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce.binds.resident"),
                layout: &self.reduce_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: in_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(out_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "reduce.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident Softmax / LogSoftmax dispatch.
    /// One thread per row; same uniform-size pattern as reduce.
    fn run_softmax_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &SoftmaxCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: Softmax input length not divisible by size".into(),
            ));
        }
        if call.output.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: Softmax output length must equal input length".into(),
            ));
        }
        let pipeline = self
            .softmax_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown softmax op {op}")))?;
        let rows = call.input.len / size;

        let params_buf = self.cached_uniform("softmax.params", &[size as u32]);

        let bytes = (call.input.len * 4) as u64;
        let buffer = ws.buffer();
        let in_off = (call.input.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let key = BindGroupKey {
            op,
            slots: vec![(in_off, bytes), (out_off, bytes)],
            uniform: Some(UniformKey {
                op: "softmax.params",
                payload: vec![size as u32],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax.binds.resident"),
                layout: &self.reduce_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: in_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "softmax.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// Whether the attention-decode shader can handle this call. The
    /// shader is intentionally narrow — single batch / head / decode
    /// query — so the canonical Attention op stays correct for every
    /// shape (unsupported shapes route to the host-fallback CPU
    /// reference). Promoting the general op (multi-batch, GQA,
    /// causal `seq_q > 1`) is a follow-up.
    fn attention_decode_supports(call: &AttentionCall) -> bool {
        call.batch == 1
            && call.num_q_heads == 1
            && call.num_kv_heads == 1
            && call.seq_q == 1
            && call.seq_kv > 0
            && call.seq_kv <= ATTENTION_DECODE_MAX_SEQ_KV
            && call.head_dim > 0
    }

    /// Dispatch the attention-decode shader on a resident workspace.
    /// Caller is responsible for verifying support via
    /// [`Self::attention_decode_supports`].
    fn run_attention_decode_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &AttentionCall,
    ) -> Result<(), ExecError> {
        let hd = call.head_dim as usize;
        let sk = call.seq_kv as usize;
        if call.q.len != hd || call.output.len != hd {
            return Err(ExecError::Backend(
                "WgpuBackend: Attention Q/output span lengths must equal head_dim at decode shape"
                    .into(),
            ));
        }
        if call.k.len != sk * hd || call.v.len != sk * hd {
            return Err(ExecError::Backend(
                "WgpuBackend: Attention K/V span lengths must equal seq_kv * head_dim".into(),
            ));
        }
        let params_buf = self.cached_uniform(
            "attention_decode.params",
            &[call.head_dim, call.seq_kv, call.scale_bits, 0],
        );

        let buffer = ws.buffer();
        let q_off = (call.q.offset * 4) as u64;
        let k_off = (call.k.offset * 4) as u64;
        let v_off = (call.v.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let q_bytes = (call.q.len * 4) as u64;
        let k_bytes = (call.k.len * 4) as u64;
        let v_bytes = (call.v.len * 4) as u64;
        let out_bytes = (call.output.len * 4) as u64;
        let key = BindGroupKey {
            op: "attention_decode",
            slots: vec![
                (q_off, q_bytes),
                (k_off, k_bytes),
                (v_off, v_bytes),
                (out_off, out_bytes),
            ],
            uniform: Some(UniformKey {
                op: "attention_decode.params",
                payload: vec![call.head_dim, call.seq_kv, call.scale_bits, 0],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("attention_decode.binds.resident"),
                layout: &self.attention_decode_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: q_off,
                            size: wgpu::BufferSize::new(q_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: k_off,
                            size: wgpu::BufferSize::new(k_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: v_off,
                            size: wgpu::BufferSize::new(v_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(out_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        // One workgroup handles the full call. Pick the pipeline by
        // head_dim: narrow shapes win on the 64-thread shader, wide
        // shapes win on the 256-thread shader. The large pipeline is
        // built lazily — see its field doc on the workgroup-memory
        // scheduling effect.
        let pipeline = if call.head_dim <= ATTENTION_DECODE_LARGE_THRESHOLD {
            self.attention_decode_small_pipeline.clone()
        } else {
            self.ensure_attention_decode_large_pipeline()
        };
        self.encode_and_submit("attention_decode.resident", &pipeline, &bind_group, 1);
        Ok(())
    }

    /// Build the large attention-decode pipeline on first use, then
    /// cache it. Returns a clone — `wgpu::ComputePipeline` is
    /// `Arc`-backed so the clone is a cheap refcount bump.
    fn ensure_attention_decode_large_pipeline(&self) -> wgpu::ComputePipeline {
        let mut cell = self.attention_decode_large_pipeline.borrow_mut();
        if cell.is_none() {
            *cell = Some(build_pipeline(
                &self.device,
                &self.attention_decode_bind_layout,
                "attention_decode_large",
                ATTENTION_DECODE_LARGE_WGSL,
            ));
        }
        cell.as_ref()
            .expect("large pipeline initialized above")
            .clone()
    }

    fn dispatch_softmax(
        &self,
        storage: &mut [f32],
        call: &SoftmaxCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: Softmax input length not divisible by size".into(),
            ));
        }
        if call.output.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: Softmax output length must equal input length".into(),
            ));
        }
        let pipeline = self
            .softmax_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown softmax op {op}")))?;
        let rows = call.input.len / size;

        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "softmax.in",
        );
        let out_buf = self.alloc_storage(call.input.len, "softmax.out");
        let staging = self.alloc_staging(call.input.len);

        use wgpu::util::DeviceExt;
        let params = [size as u32];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax.binds"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        // Dispatch one thread per *row* (each thread sequentially folds
        // its row); we still grid up to a multiple of 64.
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            call.input.len,
            rows.div_ceil(64) as u32,
            &mut storage[call.output.offset..call.output.offset + call.input.len],
        )
    }

    fn dispatch_norm_scale(
        &self,
        storage: &mut [f32],
        call: &NormScaleCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm output length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: norm weight length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown norm op {op}")))?;
        let rows = call.input.len / size;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "norm.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + size],
            "norm.weight",
        );
        let out_buf = self.alloc_storage(call.input.len, "norm.out");
        let staging = self.alloc_staging(call.input.len);
        let params_buf = self.upload_norm_params(call.size, call.epsilon);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm2.binds"),
            layout: &self.norm2_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            call.input.len,
            rows.div_ceil(64) as u32,
            &mut storage[call.output.offset..call.output.offset + call.input.len],
        )
    }

    fn dispatch_norm_full(
        &self,
        storage: &mut [f32],
        call: &NormFullCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm output length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size || call.bias.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: norm weight/bias length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown norm op {op}")))?;
        let rows = call.input.len / size;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "norm.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + size],
            "norm.weight",
        );
        let bias_buf = self.upload(
            &storage[call.bias.offset..call.bias.offset + size],
            "norm.bias",
        );
        let out_buf = self.alloc_storage(call.input.len, "norm.out");
        let staging = self.alloc_staging(call.input.len);
        let params_buf = self.upload_norm_params(call.size, call.epsilon);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm3.binds"),
            layout: &self.norm3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            call.input.len,
            rows.div_ceil(64) as u32,
            &mut storage[call.output.offset..call.output.offset + call.input.len],
        )
    }

    /// ADR-051 step 3: device-resident NormScale dispatch (RmsNorm /
    /// InstanceNorm). Binds workspace windows for input + weight +
    /// output via `BufferBinding` offsets — no per-call uploads.
    fn run_norm_scale_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &NormScaleCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm output length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: norm weight length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown norm op {op}")))?;
        let rows = call.input.len / size;
        let params_buf = self.upload_norm_params(call.size, call.epsilon);

        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (size * 4) as u64;
        let buffer = ws.buffer();
        let in_off = (call.input.offset * 4) as u64;
        let w_off = (call.weight.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let key = BindGroupKey {
            op,
            slots: vec![(in_off, in_bytes), (w_off, w_bytes), (out_off, in_bytes)],
            uniform: Some(UniformKey {
                op: "norm.params",
                payload: vec![call.size, call.epsilon],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("norm2.binds.resident"),
                layout: &self.norm2_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: in_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: w_off,
                            size: wgpu::BufferSize::new(w_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "norm2.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident NormFull dispatch (LayerNorm
    /// + AddRmsNorm — both use 4 storage bindings + uniform).
    fn run_norm_full_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &NormFullCall,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm output length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size || call.bias.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: norm weight/bias length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown norm op {op}")))?;
        let rows = call.input.len / size;
        let params_buf = self.upload_norm_params(call.size, call.epsilon);

        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (size * 4) as u64;
        let buffer = ws.buffer();
        let in_off = (call.input.offset * 4) as u64;
        let w_off = (call.weight.offset * 4) as u64;
        let b_off = (call.bias.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let key = BindGroupKey {
            op,
            slots: vec![
                (in_off, in_bytes),
                (w_off, w_bytes),
                (b_off, w_bytes),
                (out_off, in_bytes),
            ],
            uniform: Some(UniformKey {
                op: "norm.params",
                payload: vec![call.size, call.epsilon],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("norm3.binds.resident"),
                layout: &self.norm3_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: in_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: w_off,
                            size: wgpu::BufferSize::new(w_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: b_off,
                            size: wgpu::BufferSize::new(w_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "norm3.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident AddRmsNorm dispatch.
    /// Bindings: residual, input, weight, output, params (per
    /// `AddRmsNormCall` field order).
    fn run_add_rms_norm_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &AddRmsNormCall,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.output.len || call.input.len != call.residual.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm residual/input/output spans must match".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm weight length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get("add_rms_norm")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing add_rms_norm".into()))?;
        let rows = call.input.len / size;
        let params_buf = self.upload_norm_params(call.size, call.epsilon);

        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (size * 4) as u64;
        let buffer = ws.buffer();
        let res_off = (call.residual.offset * 4) as u64;
        let in_off = (call.input.offset * 4) as u64;
        let w_off = (call.weight.offset * 4) as u64;
        let out_off = (call.output.offset * 4) as u64;
        let key = BindGroupKey {
            op: "add_rms_norm",
            slots: vec![
                (res_off, in_bytes),
                (in_off, in_bytes),
                (w_off, w_bytes),
                (out_off, in_bytes),
            ],
            uniform: Some(UniformKey {
                op: "norm.params",
                payload: vec![call.size, call.epsilon],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("add_rms_norm.binds.resident"),
                layout: &self.norm3_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: res_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: in_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: w_off,
                            size: wgpu::BufferSize::new(w_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: out_off,
                            size: wgpu::BufferSize::new(in_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "add_rms_norm.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// Encode a single compute pass into either the active pending
    /// encoder (when a `run_resident` walk is in flight) or a freshly
    /// allocated one that gets submitted immediately. Used by all
    /// `run_*_resident` helpers — the resident path doesn't need a
    /// staging-buffer copy-back, so the encode/submit reduces to a
    /// few lines.
    fn encode_and_submit(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups_x: u32,
    ) {
        self.record_or_submit(label, |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        });
    }

    /// If a `pending_encoder` is set (we're inside a batched
    /// `run_resident` walk), record `f` into it. Otherwise allocate
    /// a fresh encoder, run `f`, and submit it. This is the single
    /// chokepoint that turns the per-call `create_encoder + submit`
    /// pattern into a one-submit batch when the executor calls
    /// `run_resident`.
    fn record_or_submit<F>(&self, label: &str, f: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut pending = self.pending_encoder.borrow_mut();
        if let Some(enc) = pending.as_mut() {
            f(enc);
        } else {
            drop(pending);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            f(&mut encoder);
            self.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Flush the in-flight pending encoder (if any) — used before host
    /// fallbacks so the workspace state on the device is current
    /// before a `read_span` mirrors it back.
    fn flush_pending_encoder(&self) {
        if let Some(enc) = self.pending_encoder.borrow_mut().take() {
            self.queue.submit(std::iter::once(enc.finish()));
        }
    }

    /// Install a fresh encoder in `pending_encoder` so the next
    /// `dispatch_resident` arms record into it.
    fn install_pending_encoder(&self) {
        let enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("run_resident.batched"),
            });
        *self.pending_encoder.borrow_mut() = Some(enc);
    }

    /// Return a cached uniform buffer for the given op + payload. The
    /// payload bytes form the cache key; identical shapes (same
    /// matmul `(m, k, n)`, same norm `(size, eps)`) share one buffer
    /// across the entire backend. Misses allocate via `create_buffer_init`.
    fn cached_uniform(&self, op: &'static str, payload: &[u32]) -> wgpu::Buffer {
        let key = UniformKey {
            op,
            payload: payload.to_vec(),
        };
        if let Some(buf) = self.uniform_cache.borrow().get(&key) {
            return buf.clone();
        }
        use wgpu::util::DeviceExt;
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(op),
                contents: bytemuck::cast_slice(payload),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        self.uniform_cache.borrow_mut().insert(key, buf.clone());
        buf
    }

    fn dispatch_group_norm(
        &self,
        storage: &mut [f32],
        call: &GroupNormCall,
    ) -> Result<(), ExecError> {
        let groups = call.num_groups;
        if groups == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(groups as usize) {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm input length not divisible by num_groups".into(),
            ));
        }
        let group_elements = call.input.len / groups as usize;
        if call.weight.len != group_elements || call.bias.len != group_elements {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm weight/bias length must equal group_elements".into(),
            ));
        }
        if call.output.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm output length must equal input length".into(),
            ));
        }
        // GroupNorm = LayerNorm with `size = group_elements` and
        // `rows = num_groups`. Same shader works as-is once we feed
        // it the per-group `size` parameter.
        let pipeline = self
            .norm_pipelines
            .get("layer_norm")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing layer_norm pipeline".into()))?;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "group_norm.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + group_elements],
            "group_norm.weight",
        );
        let bias_buf = self.upload(
            &storage[call.bias.offset..call.bias.offset + group_elements],
            "group_norm.bias",
        );
        let out_buf = self.alloc_storage(call.input.len, "group_norm.out");
        let staging = self.alloc_staging(call.input.len);
        let params_buf = self.upload_norm_params(group_elements as u32, call.epsilon);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group_norm.binds"),
            layout: &self.norm3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            call.input.len,
            (groups as usize).div_ceil(64) as u32,
            &mut storage[call.output.offset..call.output.offset + call.input.len],
        )
    }

    fn dispatch_add_rms_norm(
        &self,
        storage: &mut [f32],
        call: &AddRmsNormCall,
    ) -> Result<(), ExecError> {
        let size = call.size as usize;
        if size == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.input.len != call.residual.len || call.input.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm operand lengths must match".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm input length not divisible by size".into(),
            ));
        }
        if call.weight.len != size {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNorm weight length must equal size".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get("add_rms_norm")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing add_rms_norm".into()))?;
        let rows = call.input.len / size;
        let res_buf = self.upload(
            &storage[call.residual.offset..call.residual.offset + call.residual.len],
            "add_rms.residual",
        );
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "add_rms.input",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + size],
            "add_rms.weight",
        );
        let out_buf = self.alloc_storage(call.input.len, "add_rms.out");
        let staging = self.alloc_staging(call.input.len);
        let params_buf = self.upload_norm_params(call.size, call.epsilon);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_rms.binds"),
            layout: &self.norm3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            call.input.len,
            rows.div_ceil(64) as u32,
            &mut storage[call.output.offset..call.output.offset + call.input.len],
        )
    }

    fn dispatch_conv2d(&self, storage: &mut [f32], call: &Conv2dCall) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c_out as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d output length inconsistent with N*C_out*H_out*W_out".into(),
            ));
        }
        let group = call.group.max(1);
        if !call.c_in.is_multiple_of(group) || !call.c_out.is_multiple_of(group) {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d c_in / c_out must divide evenly by group".into(),
            ));
        }
        let pipeline = self
            .conv_pipelines
            .get("conv2d")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing conv2d pipeline".into()))?;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "conv.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + call.weight.len],
            "conv.weight",
        );
        // Bias is optional in `Conv2dCall` (zero-length span). The
        // bind layout always expects a storage buffer, so synthesise
        // a zero buffer of length `c_out` when the call has none.
        let bias_buf = if call.bias.len == call.c_out as usize {
            self.upload(
                &storage[call.bias.offset..call.bias.offset + call.bias.len],
                "conv.bias",
            )
        } else if call.bias.len == 0 {
            let zeros = vec![0.0_f32; call.c_out as usize];
            self.upload(&zeros, "conv.bias.zero")
        } else {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d bias length must be 0 or c_out".into(),
            ));
        };
        let out_buf = self.alloc_storage(total_out, "conv.out");
        let staging = self.alloc_staging(total_out);
        let params_buf = self.upload_conv_params(ConvUniformParams::from_conv2d(call, group));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv.binds"),
            layout: &self.conv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let workgroups = total_out.div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    fn dispatch_rms_norm_grad(
        &self,
        storage: &mut [f32],
        call: &RmsNormGradCall,
    ) -> Result<(), ExecError> {
        self.run_norm_grad2(storage, NormGrad2Args::from_rms(call))
    }

    fn dispatch_instance_norm_grad(
        &self,
        storage: &mut [f32],
        call: &InstanceNormGradCall,
    ) -> Result<(), ExecError> {
        self.run_norm_grad2(storage, NormGrad2Args::from_instance(call))
    }

    /// ADR-051 step 3: device-resident SoftmaxGrad / LogSoftmaxGrad.
    /// Bindings: forward output, dC, dA (rw, accumulating), params.
    fn run_softmax_grad_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &SoftmaxGradCall,
        kind: SoftmaxGradKind,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.output.len == 0 || call.da.len == 0 {
            return Ok(());
        }
        if !call.output.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: SoftmaxGrad output length not divisible by size".into(),
            ));
        }
        if call.da.len != call.output.len || call.dc.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: SoftmaxGrad span sizes inconsistent".into(),
            ));
        }
        let rows = call.output.len / size;
        let op = match kind {
            SoftmaxGradKind::Softmax => "softmax_grad",
            SoftmaxGradKind::LogSoftmax => "log_softmax_grad",
        };
        let pipeline = self
            .norm_grad_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing {op}")))?;

        let params_buf = self.cached_uniform("softmax_grad.params", &[size as u32]);

        let bytes = (call.output.len * 4) as u64;
        let buffer = ws.buffer();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_grad.binds.resident"),
            layout: &self.softmax_grad_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.dc.offset * 4) as u64,
                        size: wgpu::BufferSize::new(bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.da.offset * 4) as u64,
                        size: wgpu::BufferSize::new(bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "softmax_grad.resident",
            pipeline,
            &bind_group,
            rows.div_ceil(64) as u32,
        );
        Ok(())
    }

    fn dispatch_softmax_grad(
        &self,
        storage: &mut [f32],
        call: &SoftmaxGradCall,
        kind: SoftmaxGradKind,
    ) -> Result<(), ExecError> {
        let size = call.size;
        if size == 0 || call.output.len == 0 || call.da.len == 0 {
            return Ok(());
        }
        if !call.output.len.is_multiple_of(size) {
            return Err(ExecError::Backend(
                "WgpuBackend: SoftmaxGrad output length not divisible by size".into(),
            ));
        }
        if call.da.len != call.output.len || call.dc.len != call.output.len {
            return Err(ExecError::Backend(
                "WgpuBackend: SoftmaxGrad span sizes inconsistent".into(),
            ));
        }
        let rows = call.output.len / size;
        let op = match kind {
            SoftmaxGradKind::Softmax => "softmax_grad",
            SoftmaxGradKind::LogSoftmax => "log_softmax_grad",
        };
        let pipeline = self
            .norm_grad_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing {op}")))?;

        let out_buf = self.upload(
            &storage[call.output.offset..call.output.offset + call.output.len],
            "softmax_grad.output",
        );
        let dc_buf = self.upload(
            &storage[call.dc.offset..call.dc.offset + call.dc.len],
            "softmax_grad.dc",
        );
        let da_seed = storage[call.da.offset..call.da.offset + call.da.len].to_vec();
        let da_buf = self.upload_rw(&da_seed, "softmax_grad.da");

        use wgpu::util::DeviceExt;
        let params: [u32; 1] = [size as u32];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax_grad.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_grad.binds"),
            layout: &self.softmax_grad_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: da_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let staging = self.alloc_staging(call.output.len);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax_grad"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_grad.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &da_buf,
            0,
            &staging,
            0,
            (call.output.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        let out = read_back(&self.device, &staging, call.output.len).map_err(ExecError::Backend)?;
        storage[call.da.offset..call.da.offset + call.da.len].copy_from_slice(&out);
        Ok(())
    }

    /// ADR-051 step 3: device-resident LayerNormGrad. 6-storage layout
    /// (input, weight, dy, dx, dw, db). Two passes (dx + dw_db) sharing
    /// one bind group, single submit.
    fn run_layer_norm_grad_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &LayerNormGradCall,
    ) -> Result<(), ExecError> {
        let size = call.size;
        let size_us = size as usize;
        if size_us == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size_us;
        if call.dx.len != 0 && call.dx.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad dx length must equal input length".into(),
            ));
        }
        if (call.dw.len != 0 && call.dw.len != size_us)
            || (call.db.len != 0 && call.db.len != size_us)
        {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad dw/db length must equal size".into(),
            ));
        }
        if call.dy.len != call.input.len || call.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad span sizes inconsistent".into(),
            ));
        }
        let dx_pipe = self
            .norm_grad_pipelines
            .get("layer_norm_grad_dx")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing layer_norm_grad_dx".into()))?;
        let dw_db_pipe = self
            .norm_grad_pipelines
            .get("layer_norm_grad_dw_db")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing layer_norm_grad_dw_db".into())
            })?;

        let params_buf =
            self.cached_uniform("ln_grad.params", &[size, call.epsilon, rows as u32, 0]);

        // Synthesise zero buffers for absent dx/dw/db spans so the
        // bind layout has all slots filled.
        let zero_dx = if call.dx.len == 0 {
            Some(self.upload(&vec![0.0_f32; call.input.len], "ln_grad.dx.zero"))
        } else {
            None
        };
        let zero_dw = if call.dw.len == 0 {
            Some(self.upload(&vec![0.0_f32; size_us], "ln_grad.dw.zero"))
        } else {
            None
        };
        let zero_db = if call.db.len == 0 {
            Some(self.upload(&vec![0.0_f32; size_us], "ln_grad.db.zero"))
        } else {
            None
        };

        let buffer = ws.buffer();
        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (size_us * 4) as u64;
        let dx_resource: wgpu::BindingResource = if let Some(ref z) = zero_dx {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.dx.offset * 4) as u64,
                size: wgpu::BufferSize::new(in_bytes),
            })
        };
        let dw_resource: wgpu::BindingResource = if let Some(ref z) = zero_dw {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.dw.offset * 4) as u64,
                size: wgpu::BufferSize::new(w_bytes),
            })
        };
        let db_resource: wgpu::BindingResource = if let Some(ref z) = zero_db {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.db.offset * 4) as u64,
                size: wgpu::BufferSize::new(w_bytes),
            })
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_grad.binds.resident"),
            layout: &self.norm_grad3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.dy.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: db_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.record_or_submit("ln_grad.resident", |encoder| {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ln_grad.dx"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(dx_pipe);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln_grad.dw_db"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_db_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        });
        Ok(())
    }

    fn dispatch_layer_norm_grad(
        &self,
        storage: &mut [f32],
        call: &LayerNormGradCall,
    ) -> Result<(), ExecError> {
        let size = call.size;
        let size_us = size as usize;
        if size_us == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size_us;
        if call.dx.len != 0 && call.dx.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad dx length must equal input length".into(),
            ));
        }
        if (call.dw.len != 0 && call.dw.len != size_us)
            || (call.db.len != 0 && call.db.len != size_us)
        {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad dw/db length must equal size".into(),
            ));
        }
        if call.dy.len != call.input.len || call.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: LayerNormGrad span sizes inconsistent".into(),
            ));
        }
        let dx_pipe = self
            .norm_grad_pipelines
            .get("layer_norm_grad_dx")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing layer_norm_grad_dx".into()))?;
        let dw_db_pipe = self
            .norm_grad_pipelines
            .get("layer_norm_grad_dw_db")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing layer_norm_grad_dw_db".into())
            })?;

        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "ln_grad.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + size_us],
            "ln_grad.weight",
        );
        let dy_buf = self.upload(
            &storage[call.dy.offset..call.dy.offset + call.dy.len],
            "ln_grad.dy",
        );
        let dx_seed = if call.dx.len > 0 {
            storage[call.dx.offset..call.dx.offset + call.dx.len].to_vec()
        } else {
            vec![0.0_f32; call.input.len]
        };
        let dx_buf = self.upload_rw(&dx_seed, "ln_grad.dx");
        let dw_seed = if call.dw.len > 0 {
            storage[call.dw.offset..call.dw.offset + size_us].to_vec()
        } else {
            vec![0.0_f32; size_us]
        };
        let dw_buf = self.upload_rw(&dw_seed, "ln_grad.dw");
        let db_seed = if call.db.len > 0 {
            storage[call.db.offset..call.db.offset + size_us].to_vec()
        } else {
            vec![0.0_f32; size_us]
        };
        let db_buf = self.upload_rw(&db_seed, "ln_grad.db");

        use wgpu::util::DeviceExt;
        let params: [u32; 4] = [size, call.epsilon, rows as u32, 0];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ln_grad.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ln_grad.binds"),
            layout: &self.norm_grad3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: db_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let dx_staging = self.alloc_staging(call.input.len);
        let dw_staging = self.alloc_staging(size_us);
        let db_staging = self.alloc_staging(size_us);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ln_grad"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln_grad.dx"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dx_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ln_grad.dw_db"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_db_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &dx_buf,
            0,
            &dx_staging,
            0,
            (call.input.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_buffer(
            &dw_buf,
            0,
            &dw_staging,
            0,
            (size_us * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_buffer(
            &db_buf,
            0,
            &db_staging,
            0,
            (size_us * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        if call.dx.len > 0 {
            let out =
                read_back(&self.device, &dx_staging, call.input.len).map_err(ExecError::Backend)?;
            storage[call.dx.offset..call.dx.offset + call.dx.len].copy_from_slice(&out);
        }
        if call.dw.len > 0 {
            let out = read_back(&self.device, &dw_staging, size_us).map_err(ExecError::Backend)?;
            storage[call.dw.offset..call.dw.offset + size_us].copy_from_slice(&out);
        }
        if call.db.len > 0 {
            let out = read_back(&self.device, &db_staging, size_us).map_err(ExecError::Backend)?;
            storage[call.db.offset..call.db.offset + size_us].copy_from_slice(&out);
        }
        Ok(())
    }

    /// ADR-051 step 3: device-resident AddRmsNormGrad. 7-storage layout
    /// (residual, input, weight, dy, d_residual, d_input, dw). Two
    /// passes (dx + dw) recorded in one encoder.
    fn run_add_rms_norm_grad_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &AddRmsNormGradCall,
    ) -> Result<(), ExecError> {
        let size = call.size;
        let size_us = size as usize;
        if size_us == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.residual.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad residual length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size_us;
        let dr_len = call.d_residual.len;
        let di_len = call.d_input.len;
        if dr_len != 0 && dr_len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad d_residual length must equal input length".into(),
            ));
        }
        if di_len != 0 && di_len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad d_input length must equal input length".into(),
            ));
        }
        if call.dw.len != 0 && call.dw.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad dw length must equal size".into(),
            ));
        }
        if call.dy.len != call.input.len || call.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad span sizes inconsistent".into(),
            ));
        }
        let dx_pipe = self
            .norm_grad_pipelines
            .get("add_rms_norm_grad_dx")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing add_rms_norm_grad_dx".into())
            })?;
        let dw_pipe = self
            .norm_grad_pipelines
            .get("add_rms_norm_grad_dw")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing add_rms_norm_grad_dw".into())
            })?;

        let params_buf =
            self.cached_uniform("addrms_grad.params", &[size, call.epsilon, rows as u32, 0]);

        let zero_dr = if dr_len == 0 {
            Some(self.upload(&vec![0.0_f32; call.input.len], "addrms_grad.dr.zero"))
        } else {
            None
        };
        let zero_di = if di_len == 0 {
            Some(self.upload(&vec![0.0_f32; call.input.len], "addrms_grad.di.zero"))
        } else {
            None
        };
        let zero_dw = if call.dw.len == 0 {
            Some(self.upload(&vec![0.0_f32; size_us], "addrms_grad.dw.zero"))
        } else {
            None
        };

        let buffer = ws.buffer();
        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (size_us * 4) as u64;
        let dr_resource: wgpu::BindingResource = if let Some(ref z) = zero_dr {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.d_residual.offset * 4) as u64,
                size: wgpu::BufferSize::new(in_bytes),
            })
        };
        let di_resource: wgpu::BindingResource = if let Some(ref z) = zero_di {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.d_input.offset * 4) as u64,
                size: wgpu::BufferSize::new(in_bytes),
            })
        };
        let dw_resource: wgpu::BindingResource = if let Some(ref z) = zero_dw {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.dw.offset * 4) as u64,
                size: wgpu::BufferSize::new(w_bytes),
            })
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("addrms_grad.binds.resident"),
            layout: &self.norm_grad_addrms_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.residual.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.dy.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dr_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: di_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dw_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.record_or_submit("addrms_grad.resident", |encoder| {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("addrms_grad.dx"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(dx_pipe);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("addrms_grad.dw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        });
        Ok(())
    }

    fn dispatch_add_rms_norm_grad(
        &self,
        storage: &mut [f32],
        call: &AddRmsNormGradCall,
    ) -> Result<(), ExecError> {
        let size = call.size;
        let size_us = size as usize;
        if size_us == 0 || call.input.len == 0 {
            return Ok(());
        }
        if call.residual.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad residual length must equal input length".into(),
            ));
        }
        if !call.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad input length not divisible by size".into(),
            ));
        }
        let rows = call.input.len / size_us;
        let dr_len = call.d_residual.len;
        let di_len = call.d_input.len;
        if dr_len != 0 && dr_len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad d_residual length must equal input length".into(),
            ));
        }
        if di_len != 0 && di_len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad d_input length must equal input length".into(),
            ));
        }
        if call.dw.len != 0 && call.dw.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad dw length must equal size".into(),
            ));
        }
        if call.dy.len != call.input.len || call.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: AddRmsNormGrad span sizes inconsistent".into(),
            ));
        }
        let dx_pipe = self
            .norm_grad_pipelines
            .get("add_rms_norm_grad_dx")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing add_rms_norm_grad_dx".into())
            })?;
        let dw_pipe = self
            .norm_grad_pipelines
            .get("add_rms_norm_grad_dw")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing add_rms_norm_grad_dw".into())
            })?;

        let res_buf = self.upload(
            &storage[call.residual.offset..call.residual.offset + call.residual.len],
            "addrms_grad.residual",
        );
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "addrms_grad.input",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + size_us],
            "addrms_grad.weight",
        );
        let dy_buf = self.upload(
            &storage[call.dy.offset..call.dy.offset + call.dy.len],
            "addrms_grad.dy",
        );
        let dr_seed = if dr_len > 0 {
            storage[call.d_residual.offset..call.d_residual.offset + dr_len].to_vec()
        } else {
            vec![0.0_f32; call.input.len]
        };
        let dr_buf = self.upload_rw(&dr_seed, "addrms_grad.d_residual");
        let di_seed = if di_len > 0 {
            storage[call.d_input.offset..call.d_input.offset + di_len].to_vec()
        } else {
            vec![0.0_f32; call.input.len]
        };
        let di_buf = self.upload_rw(&di_seed, "addrms_grad.d_input");
        let dw_seed = if call.dw.len > 0 {
            storage[call.dw.offset..call.dw.offset + size_us].to_vec()
        } else {
            vec![0.0_f32; size_us]
        };
        let dw_buf = self.upload_rw(&dw_seed, "addrms_grad.dw");

        use wgpu::util::DeviceExt;
        let params: [u32; 4] = [size, call.epsilon, rows as u32, 0];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("addrms_grad.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("addrms_grad.binds"),
            layout: &self.norm_grad_addrms_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: di_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let dr_staging = self.alloc_staging(call.input.len);
        let di_staging = self.alloc_staging(call.input.len);
        let dw_staging = self.alloc_staging(size_us);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("addrms_grad"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("addrms_grad.dx"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dx_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("addrms_grad.dw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &dr_buf,
            0,
            &dr_staging,
            0,
            (call.input.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_buffer(
            &di_buf,
            0,
            &di_staging,
            0,
            (call.input.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_buffer(
            &dw_buf,
            0,
            &dw_staging,
            0,
            (size_us * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        if dr_len > 0 {
            let out =
                read_back(&self.device, &dr_staging, call.input.len).map_err(ExecError::Backend)?;
            storage[call.d_residual.offset..call.d_residual.offset + dr_len].copy_from_slice(&out);
        }
        if di_len > 0 {
            let out =
                read_back(&self.device, &di_staging, call.input.len).map_err(ExecError::Backend)?;
            storage[call.d_input.offset..call.d_input.offset + di_len].copy_from_slice(&out);
        }
        if call.dw.len > 0 {
            let out = read_back(&self.device, &dw_staging, size_us).map_err(ExecError::Backend)?;
            storage[call.dw.offset..call.dw.offset + size_us].copy_from_slice(&out);
        }
        Ok(())
    }

    /// ADR-051 step 3: device-resident `run_norm_grad2`. RmsNormGrad
    /// and InstanceNormGrad share this 5-storage layout. Two passes
    /// (`dx` + `dw`) recorded in one encoder; both pipelines share the
    /// bind group so the same workspace bindings cover both passes.
    /// `dx` and `dw` are accumulating — workspace contents are
    /// read-modify-written in place by the WGSL.
    fn run_norm_grad2_resident(
        &self,
        ws: &WgpuWorkspace,
        args: NormGrad2Args,
    ) -> Result<(), ExecError> {
        let size_us = args.size as usize;
        if size_us == 0 || args.input.len == 0 {
            return Ok(());
        }
        if !args.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad input length not divisible by size".into(),
            ));
        }
        let rows = args.input.len / size_us;
        if args.dx.len != 0 && args.dx.len != args.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dx length must equal input length".into(),
            ));
        }
        if args.dw.len != 0 && args.dw.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dw length must equal size".into(),
            ));
        }
        if args.dy.len != args.input.len || args.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad span sizes inconsistent".into(),
            ));
        }
        // Synthesise a zero-buffer for dx/dw when the call carries
        // zero-length spans (means "accumulator not used by this
        // call"). Tiny per-call alloc.
        let zero_dx = if args.dx.len == 0 {
            Some(self.upload(&vec![0.0_f32; args.input.len], "norm_grad.dx.zero"))
        } else {
            None
        };
        let zero_dw = if args.dw.len == 0 {
            Some(self.upload(&vec![0.0_f32; size_us], "norm_grad.dw.zero"))
        } else {
            None
        };
        let dx_pipeline = self.norm_grad_pipelines.get(args.dx_op).ok_or_else(|| {
            ExecError::Backend(format!("WgpuBackend: missing pipeline {}", args.dx_op))
        })?;
        let dw_pipeline = self.norm_grad_pipelines.get(args.dw_op).ok_or_else(|| {
            ExecError::Backend(format!("WgpuBackend: missing pipeline {}", args.dw_op))
        })?;

        let params_buf = self.cached_uniform(
            "norm_grad.params",
            &[args.size, args.epsilon_bits, rows as u32, 0],
        );

        let buffer = ws.buffer();
        let in_bytes = (args.input.len * 4) as u64;
        let w_bytes = (size_us * 4) as u64;
        let dx_resource: wgpu::BindingResource = if let Some(ref z) = zero_dx {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (args.dx.offset * 4) as u64,
                size: wgpu::BufferSize::new(in_bytes),
            })
        };
        let dw_resource: wgpu::BindingResource = if let Some(ref z) = zero_dw {
            z.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (args.dw.offset * 4) as u64,
                size: wgpu::BufferSize::new(w_bytes),
            })
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm_grad.binds.resident"),
            layout: &self.norm_grad2_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (args.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (args.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (args.dy.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        // Two passes (dx + dw) sharing the bind group, recorded in one
        // encoder for a single submit.
        self.record_or_submit("norm_grad.resident", |encoder| {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("norm_grad.dx"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(dx_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("norm_grad.dw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        });
        Ok(())
    }

    fn run_norm_grad2(&self, storage: &mut [f32], args: NormGrad2Args) -> Result<(), ExecError> {
        let size_us = args.size as usize;
        if size_us == 0 || args.input.len == 0 {
            return Ok(());
        }
        if !args.input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad input length not divisible by size".into(),
            ));
        }
        let rows = args.input.len / size_us;
        if args.dx.len != 0 && args.dx.len != args.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dx length must equal input length".into(),
            ));
        }
        if args.dw.len != 0 && args.dw.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dw length must equal size".into(),
            ));
        }
        if args.dy.len != args.input.len || args.weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad span sizes inconsistent".into(),
            ));
        }
        let dx_pipeline = self.norm_grad_pipelines.get(args.dx_op).ok_or_else(|| {
            ExecError::Backend(format!("WgpuBackend: missing pipeline {}", args.dx_op))
        })?;
        let dw_pipeline = self.norm_grad_pipelines.get(args.dw_op).ok_or_else(|| {
            ExecError::Backend(format!("WgpuBackend: missing pipeline {}", args.dw_op))
        })?;

        let in_buf = self.upload(
            &storage[args.input.offset..args.input.offset + args.input.len],
            "norm_grad.in",
        );
        let weight_buf = self.upload(
            &storage[args.weight.offset..args.weight.offset + size_us],
            "norm_grad.weight",
        );
        let dy_buf = self.upload(
            &storage[args.dy.offset..args.dy.offset + args.dy.len],
            "norm_grad.dy",
        );

        // Accumulating targets — seed from the host workspace.
        let dx_seed = if args.dx.len > 0 {
            storage[args.dx.offset..args.dx.offset + args.dx.len].to_vec()
        } else {
            vec![0.0_f32; args.input.len]
        };
        let dx_buf = self.upload_rw(&dx_seed, "norm_grad.dx");
        let dw_seed = if args.dw.len > 0 {
            storage[args.dw.offset..args.dw.offset + size_us].to_vec()
        } else {
            vec![0.0_f32; size_us]
        };
        let dw_buf = self.upload_rw(&dw_seed, "norm_grad.dw");

        // Uniform: [size, epsilon_bits, rows, padding] — same packing
        // as the resident path, so the cache hits when the same shape
        // runs in both forward and host-fallback modes.
        let params_buf = self.cached_uniform(
            "norm_grad.params",
            &[args.size, args.epsilon_bits, rows as u32, 0],
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm_grad.binds"),
            layout: &self.norm_grad2_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Two passes recorded in one encoder, single submission.
        let dx_staging = self.alloc_staging(args.input.len);
        let dw_staging = self.alloc_staging(size_us);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("norm_grad"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("norm_grad.dx"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dx_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(64) as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("norm_grad.dw"),
                timestamp_writes: None,
            });
            pass.set_pipeline(dw_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size_us.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &dx_buf,
            0,
            &dx_staging,
            0,
            (args.input.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        encoder.copy_buffer_to_buffer(
            &dw_buf,
            0,
            &dw_staging,
            0,
            (size_us * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read both staging buffers back.
        if args.dx.len > 0 {
            let out =
                read_back(&self.device, &dx_staging, args.input.len).map_err(ExecError::Backend)?;
            storage[args.dx.offset..args.dx.offset + args.dx.len].copy_from_slice(&out);
        }
        if args.dw.len > 0 {
            let out = read_back(&self.device, &dw_staging, size_us).map_err(ExecError::Backend)?;
            storage[args.dw.offset..args.dw.offset + size_us].copy_from_slice(&out);
        }
        Ok(())
    }

    fn dispatch_conv_transpose_2d(
        &self,
        storage: &mut [f32],
        call: &ConvTransposeCall,
    ) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c_out as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d output length inconsistent with shape".into(),
            ));
        }
        let group = call.group.max(1);
        if !call.c_in.is_multiple_of(group) || !call.c_out.is_multiple_of(group) {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d c_in / c_out must divide evenly by group".into(),
            ));
        }
        if call.stride_h == 0 || call.stride_w == 0 {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d strides must be > 0".into(),
            ));
        }
        let pipeline = self
            .conv_pipelines
            .get("conv_transpose_2d")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing conv_transpose_2d pipeline".into())
            })?;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "conv_t.in",
        );
        let weight_buf = self.upload(
            &storage[call.weight.offset..call.weight.offset + call.weight.len],
            "conv_t.weight",
        );
        let bias_buf = if call.bias.len == call.c_out as usize {
            self.upload(
                &storage[call.bias.offset..call.bias.offset + call.bias.len],
                "conv_t.bias",
            )
        } else if call.bias.len == 0 {
            let zeros = vec![0.0_f32; call.c_out as usize];
            self.upload(&zeros, "conv_t.bias.zero")
        } else {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d bias length must be 0 or c_out".into(),
            ));
        };
        let out_buf = self.alloc_storage(total_out, "conv_t.out");
        let staging = self.alloc_staging(total_out);
        let params_buf =
            self.upload_conv_params(ConvUniformParams::from_conv_transpose(call, group));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t.binds"),
            layout: &self.conv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let workgroups = total_out.div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    /// Pack the 16 dimensional fields shared by `Conv2dCall` and
    /// `ConvTransposeCall` into a uniform buffer matching the WGSL
    /// `ConvParams` struct. Padded to 20 u32s (80 bytes) for a clean
    /// 16-byte alignment.
    fn upload_conv_params(&self, params: ConvUniformParams) -> wgpu::Buffer {
        self.cached_uniform("conv.params", &params.pack())
    }

    fn dispatch_pool2d(
        &self,
        storage: &mut [f32],
        call: &Pool2dCall,
        kind: Pool2dKind,
    ) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: Pool2d output length inconsistent with N*C*H_out*W_out".into(),
            ));
        }
        let op = match kind {
            Pool2dKind::Max => "pool_max",
            Pool2dKind::Avg => "pool_avg",
        };
        let pipeline = self
            .pool_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing pool op {op}")))?;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "pool.in",
        );
        let out_buf = self.alloc_storage(total_out, "pool.out");
        let staging = self.alloc_staging(total_out);
        let params_buf = self.upload_pool_params(call);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool.binds"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let workgroups = total_out.div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    fn dispatch_global_avg_pool(
        &self,
        storage: &mut [f32],
        call: &GlobalAvgPoolCall,
    ) -> Result<(), ExecError> {
        let total_out = call.n as usize * call.c as usize;
        let plane = call.h as usize * call.w as usize;
        if total_out == 0 || plane == 0 {
            return Ok(());
        }
        if call.output.len != total_out || call.input.len != total_out * plane {
            return Err(ExecError::Backend(
                "WgpuBackend: GlobalAvgPool span lengths inconsistent with shape".into(),
            ));
        }
        let pipeline = self
            .pool_pipelines
            .get("global_avg_pool")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing global_avg_pool".into()))?;
        let in_buf = self.upload(
            &storage[call.input.offset..call.input.offset + call.input.len],
            "global_avg.in",
        );
        let out_buf = self.alloc_storage(total_out, "global_avg.out");
        let staging = self.alloc_staging(total_out);
        // Single u32 — element count of the spatial plane.
        use wgpu::util::DeviceExt;
        let params: [u32; 1] = [plane as u32];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("global_avg.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global_avg.binds"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let workgroups = total_out.div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    /// ADR-051 step 3: device-resident Pool2d dispatch.
    fn run_pool2d_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &Pool2dCall,
        kind: Pool2dKind,
    ) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: Pool2d output length inconsistent with N*C*H_out*W_out".into(),
            ));
        }
        let op = match kind {
            Pool2dKind::Max => "pool_max",
            Pool2dKind::Avg => "pool_avg",
        };
        let pipeline = self
            .pool_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing pool op {op}")))?;
        let params_buf = self.upload_pool_params(call);

        let in_bytes = (call.input.len * 4) as u64;
        let out_bytes = (total_out * 4) as u64;
        let buffer = ws.buffer();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pool.binds.resident"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(out_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "pool.resident",
            pipeline,
            &bind_group,
            total_out.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident GlobalAvgPool dispatch.
    fn run_global_avg_pool_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &GlobalAvgPoolCall,
    ) -> Result<(), ExecError> {
        let total_out = call.n as usize * call.c as usize;
        let plane = call.h as usize * call.w as usize;
        if total_out == 0 || plane == 0 {
            return Ok(());
        }
        if call.output.len != total_out || call.input.len != total_out * plane {
            return Err(ExecError::Backend(
                "WgpuBackend: GlobalAvgPool span lengths inconsistent with shape".into(),
            ));
        }
        let pipeline = self
            .pool_pipelines
            .get("global_avg_pool")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing global_avg_pool".into()))?;

        let params_buf = self.cached_uniform("global_avg.params", &[plane as u32]);

        let in_bytes = (call.input.len * 4) as u64;
        let out_bytes = (total_out * 4) as u64;
        let buffer = ws.buffer();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global_avg.binds.resident"),
            layout: &self.reduce_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(out_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "global_avg.resident",
            pipeline,
            &bind_group,
            total_out.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// Pack the 12 fields of `Pool2dCall` (excluding spans) into a
    /// uniform buffer matching the WGSL `PoolParams` struct.
    fn upload_pool_params(&self, call: &Pool2dCall) -> wgpu::Buffer {
        let params: [u32; 12] = [
            call.n,
            call.c,
            call.h_in,
            call.w_in,
            call.h_out,
            call.w_out,
            call.kernel_h,
            call.kernel_w,
            call.stride_h,
            call.stride_w,
            call.pad_h,
            call.pad_w,
        ];
        self.cached_uniform("pool.params", &params)
    }

    fn dispatch_matmul(&self, storage: &mut [f32], call: &MatMulCall) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 {
            return Ok(());
        }
        if call.a.len != m * k || call.b.len != k * n || call.c.len != m * n {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMul span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        let pipeline = self
            .matmul_pipelines
            .get("matmul")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing matmul pipeline".into()))?;
        let a_buf = self.upload(&storage[call.a.offset..call.a.offset + m * k], "matmul.a");
        let b_buf = self.upload(&storage[call.b.offset..call.b.offset + k * n], "matmul.b");
        let c_buf = self.alloc_storage(m * n, "matmul.c");
        let staging = self.alloc_staging(m * n);
        let params_buf = self.upload_matmul_params(m, k, n);
        let bind_group = self.matmul_bind_group(&a_buf, &b_buf, &c_buf, &params_buf);
        // Workgroup is 64; one flat thread per output element.
        let workgroups = (m * n).div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &c_buf,
                staging: &staging,
            },
            m * n,
            workgroups,
            &mut storage[call.c.offset..call.c.offset + m * n],
        )
    }

    fn dispatch_matmul_grad_a(
        &self,
        storage: &mut [f32],
        call: &MatMulGradACall,
    ) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 || call.da.len == 0 {
            return Ok(());
        }
        if call.dc.len != m * n || call.b.len != k * n || call.da.len != m * k {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMulGradA span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        let pipeline = self
            .matmul_pipelines
            .get("matmul_grad_a")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing matmul_grad_a".into()))?;
        let dc_buf = self.upload(
            &storage[call.dc.offset..call.dc.offset + m * n],
            "matmul.grad_a.dc",
        );
        let b_buf = self.upload(
            &storage[call.b.offset..call.b.offset + k * n],
            "matmul.grad_a.b",
        );
        // `da` accumulates — seed the device buffer with its current
        // host-side values so the kernel's `da[i] = da[i] + acc` does
        // a true read-modify-write.
        let da_buf = self.upload_rw(
            &storage[call.da.offset..call.da.offset + m * k],
            "matmul.grad_a.da",
        );
        let staging = self.alloc_staging(m * k);
        let params_buf = self.upload_matmul_params(m, k, n);
        let bind_group = self.matmul_bind_group(&dc_buf, &b_buf, &da_buf, &params_buf);
        let workgroups = (m * k).div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &da_buf,
                staging: &staging,
            },
            m * k,
            workgroups,
            &mut storage[call.da.offset..call.da.offset + m * k],
        )
    }

    fn dispatch_matmul_grad_b(
        &self,
        storage: &mut [f32],
        call: &MatMulGradBCall,
    ) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 || call.db.len == 0 {
            return Ok(());
        }
        if call.a.len != m * k || call.dc.len != m * n || call.db.len != k * n {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMulGradB span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        let pipeline = self
            .matmul_pipelines
            .get("matmul_grad_b")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing matmul_grad_b".into()))?;
        let a_buf = self.upload(
            &storage[call.a.offset..call.a.offset + m * k],
            "matmul.grad_b.a",
        );
        let dc_buf = self.upload(
            &storage[call.dc.offset..call.dc.offset + m * n],
            "matmul.grad_b.dc",
        );
        let db_buf = self.upload_rw(
            &storage[call.db.offset..call.db.offset + k * n],
            "matmul.grad_b.db",
        );
        let staging = self.alloc_staging(k * n);
        let params_buf = self.upload_matmul_params(m, k, n);
        let bind_group = self.matmul_bind_group(&a_buf, &dc_buf, &db_buf, &params_buf);
        let workgroups = (k * n).div_ceil(64) as u32;
        self.dispatch_and_read_with_workgroups(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &db_buf,
                staging: &staging,
            },
            k * n,
            workgroups,
            &mut storage[call.db.offset..call.db.offset + k * n],
        )
    }

    /// ADR-051 step 3: device-resident Conv2d dispatch.
    ///
    /// Binds workspace windows for input/weight/bias/output. Bias is
    /// optional in `Conv2dCall` (zero-length); when absent we
    /// synthesise a c_out-length zero buffer for the bind group (this
    /// is per-call, so the small allocation is fine).
    fn run_conv2d_resident(&self, ws: &WgpuWorkspace, call: &Conv2dCall) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c_out as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d output length inconsistent with N*C_out*H_out*W_out".into(),
            ));
        }
        let group = call.group.max(1);
        if !call.c_in.is_multiple_of(group) || !call.c_out.is_multiple_of(group) {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d c_in / c_out must divide evenly by group".into(),
            ));
        }
        let pipeline = self
            .conv_pipelines
            .get("conv2d")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing conv2d pipeline".into()))?;
        let params_buf = self.upload_conv_params(ConvUniformParams::from_conv2d(call, group));

        // Bias may be zero-length in the call; synthesise a c_out-zero
        // buffer when absent so the bind layout still gets a binding.
        let zero_bias_buf = if call.bias.len == 0 {
            Some(self.upload(&vec![0.0_f32; call.c_out as usize], "conv.bias.zero"))
        } else if call.bias.len == call.c_out as usize {
            None
        } else {
            return Err(ExecError::Backend(
                "WgpuBackend: Conv2d bias length must be 0 or c_out".into(),
            ));
        };

        let buffer = ws.buffer();
        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (call.weight.len * 4) as u64;
        let out_bytes = (total_out * 4) as u64;
        let bias_resource: wgpu::BindingResource = if let Some(ref zb) = zero_bias_buf {
            zb.as_entire_binding()
        } else {
            let bias_bytes = (call.c_out as usize * 4) as u64;
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.bias.offset * 4) as u64,
                size: wgpu::BufferSize::new(bias_bytes),
            })
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv.binds.resident"),
            layout: &self.conv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(out_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "conv2d.resident",
            pipeline,
            &bind_group,
            total_out.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident ConvTranspose2d dispatch.
    /// Same layout / uniform shape as Conv2d; only the pipeline
    /// changes. Bias-zero-buffer trick reused.
    fn run_conv_transpose_2d_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &ConvTransposeCall,
    ) -> Result<(), ExecError> {
        let total_out =
            call.n as usize * call.c_out as usize * call.h_out as usize * call.w_out as usize;
        if total_out == 0 {
            return Ok(());
        }
        if call.output.len != total_out {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d output length inconsistent with shape".into(),
            ));
        }
        let group = call.group.max(1);
        if !call.c_in.is_multiple_of(group) || !call.c_out.is_multiple_of(group) {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d c_in / c_out must divide evenly by group".into(),
            ));
        }
        if call.stride_h == 0 || call.stride_w == 0 {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d strides must be > 0".into(),
            ));
        }
        let pipeline = self
            .conv_pipelines
            .get("conv_transpose_2d")
            .ok_or_else(|| {
                ExecError::Backend("WgpuBackend: missing conv_transpose_2d pipeline".into())
            })?;
        let params_buf =
            self.upload_conv_params(ConvUniformParams::from_conv_transpose(call, group));
        let zero_bias_buf = if call.bias.len == 0 {
            Some(self.upload(&vec![0.0_f32; call.c_out as usize], "conv_t.bias.zero"))
        } else if call.bias.len == call.c_out as usize {
            None
        } else {
            return Err(ExecError::Backend(
                "WgpuBackend: ConvTranspose2d bias length must be 0 or c_out".into(),
            ));
        };
        let buffer = ws.buffer();
        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (call.weight.len * 4) as u64;
        let out_bytes = (total_out * 4) as u64;
        let bias_resource: wgpu::BindingResource = if let Some(ref zb) = zero_bias_buf {
            zb.as_entire_binding()
        } else {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: (call.bias.offset * 4) as u64,
                size: wgpu::BufferSize::new((call.c_out as usize * 4) as u64),
            })
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_t.binds.resident"),
            layout: &self.conv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(out_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "conv_t.resident",
            pipeline,
            &bind_group,
            total_out.div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident GroupNorm dispatch. Reuses the
    /// `layer_norm` pipeline with `size = group_elements`. Same
    /// `norm3_bind_layout` as LayerNorm; the role of binding 2 is
    /// "bias" for GroupNorm and is always present.
    fn run_group_norm_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &GroupNormCall,
    ) -> Result<(), ExecError> {
        let groups = call.num_groups;
        if groups == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(groups as usize) {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm input length not divisible by num_groups".into(),
            ));
        }
        let group_elements = call.input.len / groups as usize;
        if call.weight.len != group_elements || call.bias.len != group_elements {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm weight/bias length must equal group_elements".into(),
            ));
        }
        if call.output.len != call.input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: GroupNorm output length must equal input length".into(),
            ));
        }
        let pipeline = self
            .norm_pipelines
            .get("layer_norm")
            .ok_or_else(|| ExecError::Backend("WgpuBackend: missing layer_norm pipeline".into()))?;
        let params_buf = self.upload_norm_params(group_elements as u32, call.epsilon);

        let in_bytes = (call.input.len * 4) as u64;
        let w_bytes = (group_elements * 4) as u64;
        let buffer = ws.buffer();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("group_norm.binds.resident"),
            layout: &self.norm3_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.input.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.weight.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.bias.offset * 4) as u64,
                        size: wgpu::BufferSize::new(w_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: (call.output.offset * 4) as u64,
                        size: wgpu::BufferSize::new(in_bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.encode_and_submit(
            "group_norm.resident",
            pipeline,
            &bind_group,
            (groups as usize).div_ceil(64) as u32,
        );
        Ok(())
    }

    /// ADR-051 step 3: device-resident MatMul + MatMulGradA/B dispatch.
    /// All three share a single layout (a, b, c, params) where
    /// "a/b/c" are role-renamed for forward vs grad — see the WGSL
    /// shaders for which span maps to which role. Forward overwrites
    /// `c`; grad arms accumulate into `da` / `db`, which works
    /// naturally on the resident buffer (no separate upload_rw needed).
    fn run_matmul_resident_inner(
        &self,
        ws: &WgpuWorkspace,
        args: MatMulResidentArgs,
    ) -> Result<(), ExecError> {
        let pipeline = self
            .matmul_pipelines
            .get(args.pipeline_name)
            .ok_or_else(|| {
                ExecError::Backend(format!(
                    "WgpuBackend: missing {} pipeline",
                    args.pipeline_name
                ))
            })?;
        let params_buf = self.upload_matmul_params(args.m, args.k, args.n);

        let buffer = ws.buffer();
        let a_off = (args.slot_a.offset * 4) as u64;
        let b_off = (args.slot_b.offset * 4) as u64;
        let c_off = (args.slot_c.offset * 4) as u64;
        let a_bytes = (args.slot_a.len * 4) as u64;
        let b_bytes = (args.slot_b.len * 4) as u64;
        let c_bytes = (args.slot_c.len * 4) as u64;
        let key = BindGroupKey {
            op: args.pipeline_name,
            slots: vec![(a_off, a_bytes), (b_off, b_bytes), (c_off, c_bytes)],
            uniform: Some(UniformKey {
                op: "matmul.params",
                payload: vec![args.m as u32, args.k as u32, args.n as u32],
            }),
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul.binds.resident"),
                layout: &self.matmul_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: a_off,
                            size: wgpu::BufferSize::new(a_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: b_off,
                            size: wgpu::BufferSize::new(b_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: c_off,
                            size: wgpu::BufferSize::new(c_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "matmul.resident",
            pipeline,
            &bind_group,
            args.out_total.div_ceil(64) as u32,
        );
        Ok(())
    }

    fn run_matmul_resident(&self, ws: &WgpuWorkspace, call: &MatMulCall) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 {
            return Ok(());
        }
        if call.a.len != m * k || call.b.len != k * n || call.c.len != m * n {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMul span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        self.run_matmul_resident_inner(ws, MatMulResidentArgs::forward(call))
    }

    fn run_matmul_grad_a_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &MatMulGradACall,
    ) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 || call.da.len == 0 {
            return Ok(());
        }
        if call.dc.len != m * n || call.b.len != k * n || call.da.len != m * k {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMulGradA span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        self.run_matmul_resident_inner(ws, MatMulResidentArgs::grad_a(call))
    }

    fn run_matmul_grad_b_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &MatMulGradBCall,
    ) -> Result<(), ExecError> {
        let (m, k, n) = (call.m, call.k, call.n);
        if m == 0 || k == 0 || n == 0 || call.db.len == 0 {
            return Ok(());
        }
        if call.a.len != m * k || call.dc.len != m * n || call.db.len != k * n {
            return Err(ExecError::Backend(
                "WgpuBackend: MatMulGradB span lengths inconsistent with (m, k, n)".into(),
            ));
        }
        self.run_matmul_resident_inner(ws, MatMulResidentArgs::grad_b(call))
    }

    fn matmul_bind_group(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul.binds"),
            layout: &self.matmul_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn upload_matmul_params(&self, m: usize, k: usize, n: usize) -> wgpu::Buffer {
        self.cached_uniform("matmul.params", &[m as u32, k as u32, n as u32])
    }

    /// Build a uniform buffer holding `[size: u32, epsilon: f32-bits]`.
    /// `epsilon_bits` is the planner-side `f32::to_bits()` encoding;
    /// the WGSL shader declares the field as `f32`, so the bytes
    /// reinterpret correctly without a host-side decode.
    fn upload_norm_params(&self, size: u32, epsilon_bits: u32) -> wgpu::Buffer {
        self.cached_uniform("norm.params", &[size, epsilon_bits])
    }

    fn dispatch_unary(
        &self,
        storage: &mut [f32],
        call: &UnaryCall,
        kind: UnaryKind,
    ) -> Result<(), ExecError> {
        let op = unary_pipeline_key(kind).ok_or_else(|| {
            ExecError::Backend(format!(
                "WgpuBackend: unary kind {kind:?} not yet implemented"
            ))
        })?;
        self.run_unary(storage, call.input, call.output, op)
    }

    /// Host-side fallback that defers to the canonical CPU reference.
    /// Used for variants whose math is correct on CPU but not yet
    /// worth a dedicated shader (typically accumulating grad ops).
    fn host_cpu_fallback(storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
        let mut cpu = hologram_transform::CpuBackend::new();
        cpu.dispatch(storage, call)
    }

    fn dispatch_reshape(storage: &mut [f32], call: &ReshapeCall) -> Result<(), ExecError> {
        let n = call.input.len;
        if n == 0 {
            return Ok(());
        }
        if call.output.len != n {
            return Err(ExecError::Backend(
                "WgpuBackend: Reshape call has mismatched span lengths".into(),
            ));
        }
        copy_disjoint(storage, call.input, call.output)
    }

    fn dispatch_neg_grad(storage: &mut [f32], call: &NegGradCall) -> Result<(), ExecError> {
        if call.da.len == 0 {
            return Ok(());
        }
        if call.da.len != call.dc.len {
            return Err(ExecError::Backend(
                "WgpuBackend: NegGrad spans differ".into(),
            ));
        }
        for i in 0..call.da.len {
            storage[call.da.offset + i] -= storage[call.dc.offset + i];
        }
        Ok(())
    }

    fn dispatch_add_grad(storage: &mut [f32], call: &AddGradCall) -> Result<(), ExecError> {
        let n = call.dc.len;
        if call.da.len != 0 && call.da.len != n {
            return Err(ExecError::Backend("WgpuBackend: AddGrad da span".into()));
        }
        if call.db.len != 0 && call.db.len != n {
            return Err(ExecError::Backend("WgpuBackend: AddGrad db span".into()));
        }
        accum_each(storage, call.dc, call.da, |g| g);
        accum_each(storage, call.dc, call.db, |g| g);
        Ok(())
    }

    fn dispatch_sub_grad(storage: &mut [f32], call: &SubGradCall) -> Result<(), ExecError> {
        let n = call.dc.len;
        if call.da.len != 0 && call.da.len != n {
            return Err(ExecError::Backend("WgpuBackend: SubGrad da span".into()));
        }
        if call.db.len != 0 && call.db.len != n {
            return Err(ExecError::Backend("WgpuBackend: SubGrad db span".into()));
        }
        accum_each(storage, call.dc, call.da, |g| g);
        accum_each(storage, call.dc, call.db, |g| -g);
        Ok(())
    }

    fn dispatch_slice(storage: &mut [f32], call: &SliceCall) -> Result<(), ExecError> {
        let axis = call.axis_size as usize;
        if axis == 0 {
            return Ok(());
        }
        let start = call.start as usize;
        let end = call.end as usize;
        if end < start || end > axis {
            return Err(ExecError::Backend(
                "WgpuBackend: Slice [start, end) out of range".into(),
            ));
        }
        let out_row = end - start;
        if out_row == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(axis) {
            return Err(ExecError::Backend(
                "WgpuBackend: Slice input length not divisible by axis_size".into(),
            ));
        }
        let rows = call.input.len / axis;
        for r in 0..rows {
            let src = call.input.offset + r * axis + start;
            let dst = call.output.offset + r * out_row;
            // src and dst are disjoint subslices of `storage`; one row
            // at a time keeps the borrow narrow.
            for i in 0..out_row {
                storage[dst + i] = storage[src + i];
            }
        }
        Ok(())
    }

    /// ADR-051 step 3: device-resident Reshape — straight buffer copy.
    /// `copy_buffer_to_buffer` requires 4-byte-aligned offsets, which
    /// every `SlotSpan` already satisfies (spans are u32-element-
    /// counted). No workgroup dispatch needed.
    fn run_reshape_resident(
        &self,
        ws: &WgpuWorkspace,
        call: &ReshapeCall,
    ) -> Result<(), ExecError> {
        let n = call.input.len;
        if n == 0 {
            return Ok(());
        }
        if call.output.len != n {
            return Err(ExecError::Backend(
                "WgpuBackend: Reshape call has mismatched span lengths".into(),
            ));
        }
        if call.input.offset == call.output.offset {
            return Ok(());
        }
        let bytes = (n * 4) as u64;
        let buffer = ws.buffer();
        self.record_or_submit("reshape.resident", |encoder| {
            encoder.copy_buffer_to_buffer(
                buffer,
                (call.input.offset * 4) as u64,
                buffer,
                (call.output.offset * 4) as u64,
                bytes,
            );
        });
        Ok(())
    }

    /// ADR-051 step 3: device-resident Slice — per-row buffer copies.
    fn run_slice_resident(&self, ws: &WgpuWorkspace, call: &SliceCall) -> Result<(), ExecError> {
        let axis = call.axis_size as usize;
        if axis == 0 {
            return Ok(());
        }
        let start = call.start as usize;
        let end = call.end as usize;
        if end < start || end > axis {
            return Err(ExecError::Backend(
                "WgpuBackend: Slice [start, end) out of range".into(),
            ));
        }
        let out_row = end - start;
        if out_row == 0 || call.input.len == 0 {
            return Ok(());
        }
        if !call.input.len.is_multiple_of(axis) {
            return Err(ExecError::Backend(
                "WgpuBackend: Slice input length not divisible by axis_size".into(),
            ));
        }
        let rows = call.input.len / axis;
        let buffer = ws.buffer();
        let row_bytes = (out_row * 4) as u64;
        self.record_or_submit("slice.resident", |encoder| {
            for r in 0..rows {
                let src_off = ((call.input.offset + r * axis + start) * 4) as u64;
                let dst_off = ((call.output.offset + r * out_row) * 4) as u64;
                encoder.copy_buffer_to_buffer(buffer, src_off, buffer, dst_off, row_bytes);
            }
        });
        Ok(())
    }

    /// ADR-051 step 3: device-resident Concat — two per-row copies.
    fn run_concat_resident(&self, ws: &WgpuWorkspace, call: &ConcatCall) -> Result<(), ExecError> {
        let sa = call.size_a as usize;
        let sb = call.size_b as usize;
        if sa == 0 && sb == 0 {
            return Ok(());
        }
        if !call.a.len.is_multiple_of(sa.max(1)) || !call.b.len.is_multiple_of(sb.max(1)) {
            return Err(ExecError::Backend(
                "WgpuBackend: Concat operand length not divisible by its size".into(),
            ));
        }
        let rows_a = call.a.len.checked_div(sa).unwrap_or(0);
        let rows_b = call.b.len.checked_div(sb).unwrap_or(0);
        if rows_a != rows_b {
            return Err(ExecError::Backend(
                "WgpuBackend: Concat row counts differ".into(),
            ));
        }
        let row_out = sa + sb;
        let buffer = ws.buffer();
        self.record_or_submit("concat.resident", |encoder| {
            for r in 0..rows_a {
                let dst_row = call.output.offset + r * row_out;
                if sa > 0 {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        ((call.a.offset + r * sa) * 4) as u64,
                        buffer,
                        (dst_row * 4) as u64,
                        (sa * 4) as u64,
                    );
                }
                if sb > 0 {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        ((call.b.offset + r * sb) * 4) as u64,
                        buffer,
                        ((dst_row + sa) * 4) as u64,
                        (sb * 4) as u64,
                    );
                }
            }
        });
        Ok(())
    }

    fn dispatch_concat(storage: &mut [f32], call: &ConcatCall) -> Result<(), ExecError> {
        let sa = call.size_a as usize;
        let sb = call.size_b as usize;
        if sa == 0 && sb == 0 {
            return Ok(());
        }
        if !call.a.len.is_multiple_of(sa.max(1)) || !call.b.len.is_multiple_of(sb.max(1)) {
            return Err(ExecError::Backend(
                "WgpuBackend: Concat operand length not divisible by its size".into(),
            ));
        }
        let rows_a = call.a.len.checked_div(sa).unwrap_or(0);
        let rows_b = call.b.len.checked_div(sb).unwrap_or(0);
        if rows_a != rows_b {
            return Err(ExecError::Backend(
                "WgpuBackend: Concat row counts differ".into(),
            ));
        }
        let row_out = sa + sb;
        for r in 0..rows_a {
            let dst = call.output.offset + r * row_out;
            for i in 0..sa {
                storage[dst + i] = storage[call.a.offset + r * sa + i];
            }
            for i in 0..sb {
                storage[dst + sa + i] = storage[call.b.offset + r * sb + i];
            }
        }
        Ok(())
    }

    fn run_unary(
        &self,
        storage: &mut [f32],
        src: SlotSpan,
        dst: SlotSpan,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let n = src.len;
        if n == 0 {
            return Ok(());
        }
        if dst.len != n {
            return Err(ExecError::Backend(format!(
                "WgpuBackend: {op} call has mismatched span lengths"
            )));
        }
        let pipeline = self
            .unary_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown unary op {op}")))?;

        let in_buf = self.upload(&storage[src.offset..src.offset + n], "unary.in");
        let out_buf = self.alloc_storage(n, "unary.out");
        let staging = self.alloc_staging(n);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unary.binds"),
            layout: &self.unary_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_and_read(
            DispatchAndRead {
                pipeline,
                bind_group: &bind_group,
                out_buf: &out_buf,
                staging: &staging,
            },
            n,
            &mut storage[dst.offset..dst.offset + n],
        )
    }

    /// ADR-051 step 3: device-resident unary dispatch.
    ///
    /// Binds workspace buffer windows for src and dst spans directly —
    /// no per-call upload/download. Result stays on device until the
    /// executor's `BackendWorkspace::read_span` pulls it back at a
    /// plan boundary.
    fn run_unary_resident(
        &self,
        ws: &WgpuWorkspace,
        src: SlotSpan,
        dst: SlotSpan,
        op: &'static str,
    ) -> Result<(), ExecError> {
        let n = src.len;
        if n == 0 {
            return Ok(());
        }
        if dst.len != n {
            return Err(ExecError::Backend(format!(
                "WgpuBackend: {op} call has mismatched span lengths"
            )));
        }
        let pipeline = self
            .unary_pipelines
            .get(op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: unknown unary op {op}")))?;

        let bytes = (n * 4) as u64;
        let buffer = ws.buffer();
        let src_off = (src.offset * 4) as u64;
        let dst_off = (dst.offset * 4) as u64;

        let key = BindGroupKey {
            op,
            slots: vec![(src_off, bytes), (dst_off, bytes)],
            uniform: None,
        };
        let bind_group = ws.cached_bind_group(key, || {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("unary.binds.resident"),
                layout: &self.unary_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: src_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: dst_off,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    },
                ],
            })
        });
        self.encode_and_submit(
            "unary.resident",
            pipeline,
            &bind_group,
            n.div_ceil(64) as u32,
        );
        Ok(())
    }

    fn dispatch_and_read(
        &self,
        bufs: DispatchAndRead<'_>,
        n: usize,
        host_dst: &mut [f32],
    ) -> Result<(), ExecError> {
        // Default: one thread per element (workgroups = ceil(n / 64)).
        self.dispatch_and_read_with_workgroups(bufs, n, n.div_ceil(64) as u32, host_dst)
    }

    /// Like [`Self::dispatch_and_read`] but the caller chooses the
    /// X-dimension workgroup count. Used by reductions / softmax where
    /// a *thread* maps to a row, not an element.
    fn dispatch_and_read_with_workgroups(
        &self,
        bufs: DispatchAndRead<'_>,
        n: usize,
        workgroups_x: u32,
        host_dst: &mut [f32],
    ) -> Result<(), ExecError> {
        let DispatchAndRead {
            pipeline,
            bind_group,
            out_buf,
            staging,
        } = bufs;
        let buf_size = (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kernel"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kernel.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| ExecError::Backend(format!("wgpu: map_async channel: {e}")))?
            .map_err(|e| ExecError::Backend(format!("wgpu: map_async failed: {e}")))?;

        let view = slice.get_mapped_range();
        let out: &[f32] = bytemuck::cast_slice(&view);
        host_dst.copy_from_slice(out);
        drop(view);
        staging.unmap();
        Ok(())
    }

    fn upload(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Like `upload` but also flags the buffer as `COPY_SRC` so it can
    /// be read back after the shader mutates it. Used for accumulating
    /// targets — `dA`, `dB` in `MatMulGrad*`.
    fn upload_rw(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn alloc_storage(&self, n: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn alloc_staging(&self, n: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }
}

/// Map `UnaryKind` to the WGSL pipeline name. `None` for kinds that
/// have no shader yet (e.g. `Erf`, `Gelu`, `Not`, `IsNaN`).
fn unary_pipeline_key(kind: UnaryKind) -> Option<&'static str> {
    Some(match kind {
        UnaryKind::Neg => "neg",
        UnaryKind::Relu => "relu",
        UnaryKind::Sigmoid => "sigmoid",
        UnaryKind::Tanh => "tanh",
        UnaryKind::Exp => "exp",
        UnaryKind::Log => "log",
        UnaryKind::Sqrt => "sqrt",
        UnaryKind::Abs => "abs",
        UnaryKind::Reciprocal => "reciprocal",
        UnaryKind::Sin => "sin",
        UnaryKind::Cos => "cos",
        UnaryKind::Floor => "floor",
        UnaryKind::Ceil => "ceil",
        UnaryKind::Round => "round",
        UnaryKind::Sign => "sign",
        UnaryKind::Silu => "silu",
        UnaryKind::Gelu => "gelu",
        UnaryKind::Erf => "erf",
        UnaryKind::Not => "not",
        UnaryKind::IsNaN => "is_nan",
    })
}

// ── Conv2d forward (NCHW, direct) ─────────────────────────────────────
//
// One thread per output element. The shader decomposes the flat
// `gid.x` into (n, oc, oh, ow), walks the kernel window over
// (ic_local, ky, kx) for the matching group, and accumulates the
// convolution sum + bias. This is a correctness-first reference
// kernel; tile-based / im2col optimisations come later.
//
// Layout: input  [N, C_in,  H_in,  W_in]
//         weight [C_out, C_in/group, kH, kW]
//         bias   [C_out]   (caller passes a zero buffer if absent)
//         output [N, C_out, H_out, W_out]
//
// Group semantics: `g = oc / c_out_per_g`,
// `c_in_per_g = c_in / group`, `c_out_per_g = c_out / group`.

const CONV2D_WGSL: &str = r#"struct ConvParams {
    n: u32, c_in: u32, c_out: u32,
    h_in: u32, w_in: u32,
    h_out: u32, w_out: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
    dilation_h: u32, dilation_w: u32,
    group: u32,
};
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.c_out * params.h_out * params.w_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let ow = idx % params.w_out;
    let oh = (idx / params.w_out) % params.h_out;
    let oc = (idx / (params.w_out * params.h_out)) % params.c_out;
    let ni = idx / (params.w_out * params.h_out * params.c_out);

    let c_in_per_g = params.c_in / params.group;
    let c_out_per_g = params.c_out / params.group;
    let g = oc / c_out_per_g;
    let in_chw = params.c_in * params.h_in * params.w_in;
    let weight_per_oc = c_in_per_g * params.kernel_h * params.kernel_w;

    var acc: f32 = bias[oc];
    for (var ic_local: u32 = 0u; ic_local < c_in_per_g; ic_local = ic_local + 1u) {
        let ic = g * c_in_per_g + ic_local;
        for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
            let ih: i32 = i32(oh * params.stride_h)
                        + i32(ky * params.dilation_h)
                        - i32(params.pad_h);
            if (ih < 0 || ih >= i32(params.h_in)) { continue; }
            for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
                let iw: i32 = i32(ow * params.stride_w)
                            + i32(kx * params.dilation_w)
                            - i32(params.pad_w);
                if (iw < 0 || iw >= i32(params.w_in)) { continue; }
                let in_idx = ni * in_chw
                           + ic * params.h_in * params.w_in
                           + u32(ih) * params.w_in
                           + u32(iw);
                let w_idx = oc * weight_per_oc
                          + ic_local * params.kernel_h * params.kernel_w
                          + ky * params.kernel_w
                          + kx;
                acc = acc + input[in_idx] * weight[w_idx];
            }
        }
    }
    output[idx] = acc;
}
"#;

// ── Norm-grad shaders (RmsNorm, InstanceNorm) ─────────────────────────
//
// Both grads decompose into two passes:
//   - dx pass: one thread per row. Computes the row stats inline and
//     writes `dx[r, *]` accumulating into the seeded buffer.
//   - dw pass: one thread per column. For each column, walks every
//     row, recomputes the row stats, sums the dy * x * rstd
//     contribution, accumulates into `dw[i]`.
// The CPU reference fuses these into one row-major loop; we split
// because a single GPU thread per row would require atomic adds to
// fold the dw contributions across rows. Recomputing row stats in
// the dw pass roughly doubles the work but keeps the kernels
// atomic-free and matches downlevel limits.
//
// Bind layout (norm_grad2): input(0) weight(1) dy(2) dx(3,rw)
// dw(4,rw) params(5,uniform). Params: { size, epsilon, rows, _ }.

const RMS_NORM_GRAD_DX_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rows) { return; }
    let off = r * params.size;
    let inv_size = 1.0 / f32(params.size);

    var sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let v = input[off + i];
        sq = sq + v * v;
    }
    let rstd = 1.0 / sqrt(sq * inv_size + params.epsilon);

    var dot: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        dot = dot + dy[off + i] * weight[i] * input[off + i];
    }
    let dot_term = rstd * rstd * rstd * inv_size * dot;

    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let contrib = weight[i] * dy[off + i] * rstd - input[off + i] * dot_term;
        dx[off + i] = dx[off + i] + contrib;
    }
}
"#;

const RMS_NORM_GRAD_DW_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let inv_size = 1.0 / f32(params.size);

    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let off = r * params.size;
        var sq: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            let v = input[off + j];
            sq = sq + v * v;
        }
        let rstd = 1.0 / sqrt(sq * inv_size + params.epsilon);
        acc = acc + dy[off + i] * input[off + i] * rstd;
    }
    dw[i] = dw[i] + acc;
}
"#;

const INSTANCE_NORM_GRAD_DX_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rows) { return; }
    let off = r * params.size;
    let inv_size = 1.0 / f32(params.size);

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        sum = sum + input[off + i];
    }
    let mean = sum * inv_size;
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let d = input[off + i] - mean;
        var_sum = var_sum + d * d;
    }
    let rstd = 1.0 / sqrt(var_sum * inv_size + params.epsilon);

    var sum_dx_hat: f32 = 0.0;
    var sum_dx_hat_x_hat: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let x_hat = (input[off + i] - mean) * rstd;
        let dx_hat = dy[off + i] * weight[i];
        sum_dx_hat = sum_dx_hat + dx_hat;
        sum_dx_hat_x_hat = sum_dx_hat_x_hat + dx_hat * x_hat;
    }
    let mean_dx_hat = sum_dx_hat * inv_size;
    let mean_dx_hat_x_hat = sum_dx_hat_x_hat * inv_size;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let x_hat = (input[off + i] - mean) * rstd;
        let dx_hat = dy[off + i] * weight[i];
        let contrib = rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
        dx[off + i] = dx[off + i] + contrib;
    }
}
"#;

const INSTANCE_NORM_GRAD_DW_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let inv_size = 1.0 / f32(params.size);

    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let off = r * params.size;
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            sum = sum + input[off + j];
        }
        let mean = sum * inv_size;
        var var_sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            let d = input[off + j] - mean;
            var_sum = var_sum + d * d;
        }
        let rstd = 1.0 / sqrt(var_sum * inv_size + params.epsilon);
        let x_hat = (input[off + i] - mean) * rstd;
        acc = acc + dy[off + i] * x_hat;
    }
    dw[i] = dw[i] + acc;
}
"#;

// ── LayerNormGrad shaders (3-input layout) ────────────────────────────
//
// Same two-pass strategy as the 2-input norm grads, but the dw/db
// pass writes both `dw[i]` and `db[i]` in one column-thread sweep
// (saves one full row walk vs. running them separately).
//
// Bind layout: input(0) weight(1) dy(2) dx(3,rw) dw(4,rw) db(5,rw)
// params(6,uniform). Params: { size, epsilon, rows, _ }.

const LAYER_NORM_GRAD_DX_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rows) { return; }
    let off = r * params.size;
    let inv_size = 1.0 / f32(params.size);

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        sum = sum + input[off + i];
    }
    let mean = sum * inv_size;
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let d = input[off + i] - mean;
        var_sum = var_sum + d * d;
    }
    let rstd = 1.0 / sqrt(var_sum * inv_size + params.epsilon);

    var sum_dx_hat: f32 = 0.0;
    var sum_dx_hat_x_hat: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let x_hat = (input[off + i] - mean) * rstd;
        let dx_hat = dy[off + i] * weight[i];
        sum_dx_hat = sum_dx_hat + dx_hat;
        sum_dx_hat_x_hat = sum_dx_hat_x_hat + dx_hat * x_hat;
    }
    let mean_dx_hat = sum_dx_hat * inv_size;
    let mean_dx_hat_x_hat = sum_dx_hat_x_hat * inv_size;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let x_hat = (input[off + i] - mean) * rstd;
        let dx_hat = dy[off + i] * weight[i];
        let contrib = rstd * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
        dx[off + i] = dx[off + i] + contrib;
    }
}
"#;

const LAYER_NORM_GRAD_DW_DB_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> dw: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let inv_size = 1.0 / f32(params.size);

    var dw_acc: f32 = 0.0;
    var db_acc: f32 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let off = r * params.size;
        var sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            sum = sum + input[off + j];
        }
        let mean = sum * inv_size;
        var var_sum: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            let d = input[off + j] - mean;
            var_sum = var_sum + d * d;
        }
        let rstd = 1.0 / sqrt(var_sum * inv_size + params.epsilon);
        let x_hat = (input[off + i] - mean) * rstd;
        let dyi = dy[off + i];
        dw_acc = dw_acc + dyi * x_hat;
        db_acc = db_acc + dyi;
    }
    dw[i] = dw[i] + dw_acc;
    db[i] = db[i] + db_acc;
}
"#;

// ── AddRmsNormGrad shaders (4-input layout) ───────────────────────────
//
// Forward: y = rms_norm(s = residual + input, weight). Since y is
// symmetric in (residual, input) the partials are equal:
// d_residual = d_input = standard rms-norm-grad dx evaluated on `s`.
// dw is the standard rms-norm dw computed against `s`.
//
// Bind layout: residual(0) input(1) weight(2) dy(3) d_residual(4,rw)
// d_input(5,rw) dw(6,rw) params(7,uniform). Params: { size, epsilon,
// rows, _ }.

const ADD_RMS_NORM_GRAD_DX_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_residual: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_input: array<f32>;
@group(0) @binding(6) var<storage, read_write> dw: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rows) { return; }
    let off = r * params.size;
    let inv_size = 1.0 / f32(params.size);

    var sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let s = residual[off + i] + input[off + i];
        sq = sq + s * s;
    }
    let rstd = 1.0 / sqrt(sq * inv_size + params.epsilon);

    var dot: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let s = residual[off + i] + input[off + i];
        dot = dot + dy[off + i] * weight[i] * s;
    }
    let dot_term = rstd * rstd * rstd * inv_size * dot;

    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let s = residual[off + i] + input[off + i];
        let contrib = weight[i] * dy[off + i] * rstd - s * dot_term;
        d_residual[off + i] = d_residual[off + i] + contrib;
        d_input[off + i] = d_input[off + i] + contrib;
    }
}
"#;

const ADD_RMS_NORM_GRAD_DW_WGSL: &str = r#"struct Params { size: u32, epsilon: f32, rows: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_residual: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_input: array<f32>;
@group(0) @binding(6) var<storage, read_write> dw: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    let inv_size = 1.0 / f32(params.size);

    var acc: f32 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let off = r * params.size;
        var sq: f32 = 0.0;
        for (var j: u32 = 0u; j < params.size; j = j + 1u) {
            let s = residual[off + j] + input[off + j];
            sq = sq + s * s;
        }
        let rstd = 1.0 / sqrt(sq * inv_size + params.epsilon);
        let s_i = residual[off + i] + input[off + i];
        acc = acc + dy[off + i] * s_i * rstd;
    }
    dw[i] = dw[i] + acc;
}
"#;

// ── SoftmaxGrad shaders ───────────────────────────────────────────────
//
// Per-row backward for both Softmax and LogSoftmax variants. One
// thread per row; sequential fold over `params.size` elements.
// Bind layout: output(0) dc(1) da(2,rw) params(3,uniform).
//
// Softmax:    dA[r,j] += out[r,j] · (dC[r,j] − Σ_k(dC[r,k] · out[r,k]))
// LogSoftmax: dA[r,j] += dC[r,j] − exp(out[r,j]) · Σ_k(dC[r,k])

const SOFTMAX_GRAD_WGSL: &str = r#"struct Params { size: u32 };
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read_write> dc: array<f32>;
@group(0) @binding(2) var<storage, read_write> da: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&output)) { return; }

    var dot: f32 = 0.0;
    for (var j: u32 = 0u; j < params.size; j = j + 1u) {
        dot = dot + dc[off + j] * output[off + j];
    }
    for (var j: u32 = 0u; j < params.size; j = j + 1u) {
        let p = output[off + j];
        da[off + j] = da[off + j] + p * (dc[off + j] - dot);
    }
}
"#;

const LOG_SOFTMAX_GRAD_WGSL: &str = r#"struct Params { size: u32 };
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read_write> dc: array<f32>;
@group(0) @binding(2) var<storage, read_write> da: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&output)) { return; }

    var dc_sum: f32 = 0.0;
    for (var j: u32 = 0u; j < params.size; j = j + 1u) {
        dc_sum = dc_sum + dc[off + j];
    }
    for (var j: u32 = 0u; j < params.size; j = j + 1u) {
        let p = exp(output[off + j]);
        da[off + j] = da[off + j] + dc[off + j] - p * dc_sum;
    }
}
"#;

// ── ConvTranspose2d forward (NCHW, output-centric) ────────────────────
//
// The canonical CPU forward is a *scatter*: each input position
// contributes to many output positions. On GPU that creates write
// conflicts, so this kernel is a *gather*: one thread per output
// element, walking the same `(ic, oc_local, ky, kx)` window as the
// scatter loop and inverting the geometry to find the contributing
// input position. A position contributes iff
//
//     oh = ih·stride_h + ky·dilation_h - pad_h
//     ow = iw·stride_w + kx·dilation_w - pad_w
//
// for some integer `ih`, `iw`. Rearranging,
//
//     ih = (oh + pad_h - ky·dilation_h) / stride_h
//
// is valid only when the numerator is non-negative *and* a multiple
// of `stride_h` (else stride zeroes that position out). Same for `iw`.
// Weight layout `[C_in, C_out/group, kH, kW]` mirrors the canonical
// CPU reference.

const CONV_TRANSPOSE_2D_WGSL: &str = r#"struct ConvParams {
    n: u32, c_in: u32, c_out: u32,
    h_in: u32, w_in: u32,
    h_out: u32, w_out: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
    dilation_h: u32, dilation_w: u32,
    group: u32,
};
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.c_out * params.h_out * params.w_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let ow = idx % params.w_out;
    let oh = (idx / params.w_out) % params.h_out;
    let oc = (idx / (params.w_out * params.h_out)) % params.c_out;
    let ni = idx / (params.w_out * params.h_out * params.c_out);

    let c_in_per_g = params.c_in / params.group;
    let c_out_per_g = params.c_out / params.group;
    let g = oc / c_out_per_g;
    let oc_local = oc - g * c_out_per_g;
    let in_chw = params.c_in * params.h_in * params.w_in;
    let weight_per_ic = c_out_per_g * params.kernel_h * params.kernel_w;

    var acc: f32 = bias[oc];
    for (var ic_local: u32 = 0u; ic_local < c_in_per_g; ic_local = ic_local + 1u) {
        let ic = g * c_in_per_g + ic_local;
        for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
            let oh_pad: i32 = i32(oh) + i32(params.pad_h) - i32(ky * params.dilation_h);
            if (oh_pad < 0) { continue; }
            if (u32(oh_pad) % params.stride_h != 0u) { continue; }
            let ih: u32 = u32(oh_pad) / params.stride_h;
            if (ih >= params.h_in) { continue; }
            for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
                let ow_pad: i32 = i32(ow) + i32(params.pad_w) - i32(kx * params.dilation_w);
                if (ow_pad < 0) { continue; }
                if (u32(ow_pad) % params.stride_w != 0u) { continue; }
                let iw: u32 = u32(ow_pad) / params.stride_w;
                if (iw >= params.w_in) { continue; }
                let in_idx = ni * in_chw
                           + ic * params.h_in * params.w_in
                           + ih * params.w_in
                           + iw;
                let w_idx = ic * weight_per_ic
                          + oc_local * params.kernel_h * params.kernel_w
                          + ky * params.kernel_w
                          + kx;
                acc = acc + input[in_idx] * weight[w_idx];
            }
        }
    }
    output[idx] = acc;
}
"#;

// ── Pool family shaders (NCHW) ────────────────────────────────────────
//
// One thread per output element. The shader decomposes the flat
// `gid.x` into (n, c, oh, ow) and walks the kernel window, masking
// out-of-bounds positions. `pool_max` initialises to −∞ via bitcast;
// `pool_avg` divides by the in-bounds element count (matches the
// canonical CPU reference, which counts only in-bounds reads).
// `global_avg_pool` is a per-(n, c) mean over the H·W plane.

const POOL_MAX_WGSL: &str = r#"struct PoolParams {
    n: u32, c: u32,
    h_in: u32, w_in: u32,
    h_out: u32, w_out: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
};
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.c * params.h_out * params.w_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let ow = idx % params.w_out;
    let oh = (idx / params.w_out) % params.h_out;
    let ci = (idx / (params.w_out * params.h_out)) % params.c;
    let ni = idx / (params.w_out * params.h_out * params.c);

    let h_start: i32 = i32(oh * params.stride_h) - i32(params.pad_h);
    let w_start: i32 = i32(ow * params.stride_w) - i32(params.pad_w);
    var acc: f32 = bitcast<f32>(0xff800000u);  // −∞
    for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
        let ih: i32 = h_start + i32(ky);
        if (ih < 0 || ih >= i32(params.h_in)) { continue; }
        for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let iw: i32 = w_start + i32(kx);
            if (iw < 0 || iw >= i32(params.w_in)) { continue; }
            let plane_off = (ni * params.c + ci) * params.h_in * params.w_in;
            let v = input[plane_off + u32(ih) * params.w_in + u32(iw)];
            acc = max(acc, v);
        }
    }
    output[idx] = acc;
}
"#;

const POOL_AVG_WGSL: &str = r#"struct PoolParams {
    n: u32, c: u32,
    h_in: u32, w_in: u32,
    h_out: u32, w_out: u32,
    kernel_h: u32, kernel_w: u32,
    stride_h: u32, stride_w: u32,
    pad_h: u32, pad_w: u32,
};
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.c * params.h_out * params.w_out;
    let idx = gid.x;
    if (idx >= total) { return; }

    let ow = idx % params.w_out;
    let oh = (idx / params.w_out) % params.h_out;
    let ci = (idx / (params.w_out * params.h_out)) % params.c;
    let ni = idx / (params.w_out * params.h_out * params.c);

    let h_start: i32 = i32(oh * params.stride_h) - i32(params.pad_h);
    let w_start: i32 = i32(ow * params.stride_w) - i32(params.pad_w);
    var sum: f32 = 0.0;
    var count: u32 = 0u;
    for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
        let ih: i32 = h_start + i32(ky);
        if (ih < 0 || ih >= i32(params.h_in)) { continue; }
        for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let iw: i32 = w_start + i32(kx);
            if (iw < 0 || iw >= i32(params.w_in)) { continue; }
            let plane_off = (ni * params.c + ci) * params.h_in * params.w_in;
            sum = sum + input[plane_off + u32(ih) * params.w_in + u32(iw)];
            count = count + 1u;
        }
    }
    if (count == 0u) {
        output[idx] = 0.0;
    } else {
        output[idx] = sum / f32(count);
    }
}
"#;

const GLOBAL_AVG_POOL_WGSL: &str = r#"struct Params { plane: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = arrayLength(&output);
    let idx = gid.x;
    if (idx >= total) { return; }

    let off = idx * params.plane;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.plane; i = i + 1u) {
        sum = sum + input[off + i];
    }
    output[idx] = sum / f32(params.plane);
}
"#;

// ── MatMul + grad shaders ─────────────────────────────────────────────
//
// Naive correctness-first kernels — one thread per output element,
// sequential fold over the contracted dimension. Workgroup size 64,
// flat 1D thread-id; the shader decomposes `gid.x` into (row, col).
// Tile-based optimisation (shared memory, per-thread output blocks)
// is a future revision once a benchmark warrants it.
//
// MatMul (overwrite):    C[m,n] = Σ_k A[m,k] * B[k,n]
// MatMulGradA (accum):   dA[m,k] += Σ_n dC[m,n] * B[k,n]
// MatMulGradB (accum):   dB[k,n] += Σ_m A[m,k] * dC[m,n]

const MATMUL_WGSL: &str = r#"struct Params { m: u32, k: u32, n: u32 };
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.m * params.n;
    let idx = gid.x;
    if (idx >= total) { return; }
    let row = idx / params.n;
    let col = idx % params.n;

    var acc: f32 = 0.0;
    for (var ki: u32 = 0u; ki < params.k; ki = ki + 1u) {
        acc = acc + a[row * params.k + ki] * b[ki * params.n + col];
    }
    c[row * params.n + col] = acc;
}
"#;

const MATMUL_GRAD_A_WGSL: &str = r#"struct Params { m: u32, k: u32, n: u32 };
@group(0) @binding(0) var<storage, read_write> dc: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> da: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.m * params.k;
    let idx = gid.x;
    if (idx >= total) { return; }
    let row = idx / params.k;  // m index
    let col = idx % params.k;  // k index

    var acc: f32 = 0.0;
    for (var ni: u32 = 0u; ni < params.n; ni = ni + 1u) {
        acc = acc + dc[row * params.n + ni] * b[col * params.n + ni];
    }
    da[row * params.k + col] = da[row * params.k + col] + acc;
}
"#;

const MATMUL_GRAD_B_WGSL: &str = r#"struct Params { m: u32, k: u32, n: u32 };
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> dc: array<f32>;
@group(0) @binding(2) var<storage, read_write> db: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.k * params.n;
    let idx = gid.x;
    if (idx >= total) { return; }
    let row = idx / params.n;  // k index
    let col = idx % params.n;  // n index

    var acc: f32 = 0.0;
    for (var mi: u32 = 0u; mi < params.m; mi = mi + 1u) {
        acc = acc + a[mi * params.k + row] * dc[mi * params.n + col];
    }
    db[row * params.n + col] = db[row * params.n + col] + acc;
}
"#;

// ── Norm-family shaders ───────────────────────────────────────────────
//
// All four share `Params { size, epsilon }`. The `epsilon` field is
// declared as `f32` in WGSL but the host writes the raw `f32::to_bits`
// u32 — the bind layout is just bytes, so the GPU reinterprets
// correctly. One thread per row; sequential fold over `size` elements.

const RMS_NORM_WGSL: &str = r#"struct Params { size: u32, epsilon: f32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    var sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let v = input[off + i];
        sq = sq + v * v;
    }
    let rstd = 1.0 / sqrt(sq / f32(params.size) + params.epsilon);
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = input[off + i] * rstd * weight[i];
    }
}
"#;

const INSTANCE_NORM_WGSL: &str = r#"struct Params { size: u32, epsilon: f32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        sum = sum + input[off + i];
    }
    let mean = sum / f32(params.size);
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let d = input[off + i] - mean;
        var_sum = var_sum + d * d;
    }
    let rstd = 1.0 / sqrt(var_sum / f32(params.size) + params.epsilon);
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = (input[off + i] - mean) * rstd * weight[i];
    }
}
"#;

const LAYER_NORM_WGSL: &str = r#"struct Params { size: u32, epsilon: f32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        sum = sum + input[off + i];
    }
    let mean = sum / f32(params.size);
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let d = input[off + i] - mean;
        var_sum = var_sum + d * d;
    }
    let rstd = 1.0 / sqrt(var_sum / f32(params.size) + params.epsilon);
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = (input[off + i] - mean) * rstd * weight[i] + bias[i];
    }
}
"#;

/// Bindings: residual (0), input (1), weight (2), output (3),
/// params (4). Matches `AddRmsNormCall { residual, input, weight }`
/// field order.
const ADD_RMS_NORM_WGSL: &str = r#"struct Params { size: u32, epsilon: f32 };
@group(0) @binding(0) var<storage, read_write> residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    // Pass 1: stash s = residual + input into output, accumulate sq.
    var sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let s = residual[off + i] + input[off + i];
        output[off + i] = s;
        sq = sq + s * s;
    }
    let rstd = 1.0 / sqrt(sq / f32(params.size) + params.epsilon);
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = output[off + i] * rstd * weight[i];
    }
}
"#;

/// Softmax forward: numerically stable three-pass row-wise softmax.
/// One thread per row; sequential fold over `params.size` elements.
const SOFTMAX_WGSL: &str = r#"struct Params { size: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    // Pass 1: row max.
    var m = input[off];
    for (var i: u32 = 1u; i < params.size; i = i + 1u) {
        m = max(m, input[off + i]);
    }
    // Pass 2: write exp(x - m), accumulate sum.
    var sum = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        let e = exp(input[off + i] - m);
        output[off + i] = e;
        sum = sum + e;
    }
    // Pass 3: divide by sum.
    let inv = 1.0 / sum;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = output[off + i] * inv;
    }
}
"#;

/// Log-softmax forward: same structure as softmax but the final pass
/// writes `(x - m) - log(sum)` instead of `exp(x - m) / sum`.
const LOG_SOFTMAX_WGSL: &str = r#"struct Params { size: u32 };
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let off = r * params.size;
    if (off >= arrayLength(&input)) { return; }

    var m = input[off];
    for (var i: u32 = 1u; i < params.size; i = i + 1u) {
        m = max(m, input[off + i]);
    }
    var sum = 0.0;
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        sum = sum + exp(input[off + i] - m);
    }
    let log_z = log(sum);
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {
        output[off + i] = (input[off + i] - m) - log_z;
    }
}
"#;

/// Workgroup-memory budget for the attention-decode shader's
/// `scores` array. 4096 f32 ≈ 16 KB, well within Apple's typical
/// 32 KB per-workgroup budget. Calls with `seq_kv` greater than this
/// fall through to the host-fallback path. The bench's biggest
/// configured shape uses `seq_kv = 512`.
const ATTENTION_DECODE_MAX_SEQ_KV: u32 = 4096;

/// Workgroup-cooperative attention shader for decode shape.
///
/// Constraints (checked by [`WgpuBackend::attention_decode_supports`]):
///   * single batch (`batch == 1`)
///   * single head (`num_q_heads == num_kv_heads == 1`)
///   * decode query (`seq_q == 1`)
///   * `seq_kv <= ATTENTION_DECODE_MAX_SEQ_KV`
///
/// Two specialised pipelines pick a workgroup size by shape:
///
///   * **small** (`@workgroup_size(64)`): used when `head_dim ≤ 256`.
///     Smaller shapes don't have enough work to keep a wide
///     workgroup busy — scaling up just adds barrier overhead and
///     idle SIMD groups.
///   * **large** (`@workgroup_size(256)`): used when `head_dim > 256`.
///     With one workgroup per single-head problem, the workgroup
///     size *is* the parallelism, and the bench's medium shape
///     (`head_dim = 1024`, `seq_kv = 512`) needs more than 64
///     threads to keep the GPU busy. Apple Silicon supports up to
///     1024 threads per workgroup.
///
/// Layout: Q is `[head_dim]` contiguous; K and V are
/// `[seq_kv, head_dim]` row-major; output is `[head_dim]`. Both
/// shaders share the same algorithm:
///
///   1. Each thread fills `scores[i]` for `i` in
///      `tid, tid+WG, …, < seq_kv` — dot(Q, K[i]) * scale.
///   2. Softmax max+sum reduction across the workgroup. The small
///      shader runs the reduction sequentially on thread 0 (cheaper
///      barrier-wise at narrow workgroups); the large shader uses a
///      tree reduce.
///   3. Each thread writes `out[d_pos]` for `d_pos` in
///      `tid, tid+WG, …, < head_dim` — `Σ_k scores[k] * V[k, d_pos]`.
///
/// Causal masking is unnecessary at `seq_q == 1` (every key position
/// `< seq_kv = q_pos + (seq_kv - seq_q) + 1` so the canonical mask
/// rule `k > q + (seq_kv - seq_q)` collapses to false everywhere).
const ATTENTION_DECODE_SMALL_WGSL: &str = r#"struct Params {
    head_dim: u32,
    seq_kv: u32,
    scale_bits: u32,
    _pad: u32,
};
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scores: array<f32, 4096>;
var<workgroup> shared_inv_sum: f32;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let hd = params.head_dim;
    let sk = params.seq_kv;
    let scale = bitcast<f32>(params.scale_bits);

    for (var i: u32 = tid; i < sk; i = i + 64u) {
        var dot: f32 = 0.0;
        let k_off = i * hd;
        for (var j: u32 = 0u; j < hd; j = j + 1u) {
            dot = dot + q[j] * k[k_off + j];
        }
        scores[i] = dot * scale;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var m: f32 = scores[0];
        for (var i: u32 = 1u; i < sk; i = i + 1u) {
            m = max(m, scores[i]);
        }
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < sk; i = i + 1u) {
            let e = exp(scores[i] - m);
            scores[i] = e;
            sum = sum + e;
        }
        shared_inv_sum = 1.0 / sum;
    }
    workgroupBarrier();

    let inv = shared_inv_sum;
    for (var d_pos: u32 = tid; d_pos < hd; d_pos = d_pos + 64u) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < sk; i = i + 1u) {
            acc = acc + scores[i] * inv * v[i * hd + d_pos];
        }
        out[d_pos] = acc;
    }
}
"#;

const ATTENTION_DECODE_LARGE_WGSL: &str = r#"struct Params {
    head_dim: u32,
    seq_kv: u32,
    scale_bits: u32,
    _pad: u32,
};
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> scores: array<f32, 4096>;
var<workgroup> max_red: array<f32, 256>;
var<workgroup> sum_red: array<f32, 256>;
var<workgroup> shared_inv_sum: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let hd = params.head_dim;
    let sk = params.seq_kv;
    let scale = bitcast<f32>(params.scale_bits);

    for (var i: u32 = tid; i < sk; i = i + WG) {
        var dot: f32 = 0.0;
        let k_off = i * hd;
        for (var j: u32 = 0u; j < hd; j = j + 1u) {
            dot = dot + q[j] * k[k_off + j];
        }
        scores[i] = dot * scale;
    }
    workgroupBarrier();

    var local_max: f32 = bitcast<f32>(0xff800000u);
    for (var i: u32 = tid; i < sk; i = i + WG) {
        local_max = max(local_max, scores[i]);
    }
    max_red[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            max_red[tid] = max(max_red[tid], max_red[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let m = max_red[0];

    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < sk; i = i + WG) {
        let e = exp(scores[i] - m);
        scores[i] = e;
        local_sum = local_sum + e;
    }
    sum_red[tid] = local_sum;
    workgroupBarrier();
    stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sum_red[tid] = sum_red[tid] + sum_red[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        shared_inv_sum = 1.0 / sum_red[0];
    }
    workgroupBarrier();

    let inv = shared_inv_sum;
    for (var d_pos: u32 = tid; d_pos < hd; d_pos = d_pos + WG) {
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < sk; i = i + 1u) {
            acc = acc + scores[i] * inv * v[i * hd + d_pos];
        }
        out[d_pos] = acc;
    }
}
"#;

/// `head_dim` threshold above which the large-workgroup pipeline wins
/// over the small one. Determined empirically on the decode-step
/// bench: at `head_dim ≤ 256` the small pipeline is faster; at
/// `head_dim ≥ 512` the large pipeline pulls ahead by ~2×.
const ATTENTION_DECODE_LARGE_THRESHOLD: u32 = 256;

/// Map a `ReduceKind` to the WGSL pipeline name.
fn reduce_pipeline_key(kind: ReduceKind) -> &'static str {
    match kind {
        ReduceKind::Sum => "sum",
        ReduceKind::Mean => "mean",
        ReduceKind::Max => "max",
        ReduceKind::Min => "min",
        ReduceKind::Prod => "prod",
    }
}

/// `(pipeline_name, init, fold-expr-in-terms-of-`acc`-and-`x`,
/// finish-expr-in-terms-of-`acc`-and-`params.size`)`.
const REDUCE_OPS: &[(&str, &str, &str, &str)] = &[
    ("sum", "0.0", "acc + x", "acc"),
    ("mean", "0.0", "acc + x", "acc / f32(params.size)"),
    // ±∞ via bitcast — WGSL has no `f32::INFINITY` literal but
    // `bitcast<f32>(0xff800000u)` is the IEEE-754 bit pattern for −∞.
    ("max", "bitcast<f32>(0xff800000u)", "max(acc, x)", "acc"),
    ("min", "bitcast<f32>(0x7f800000u)", "min(acc, x)", "acc"),
    ("prod", "1.0", "acc * x", "acc"),
];

fn reduce_shader_source(init: &str, fold: &str, finish: &str) -> String {
    format!(
        r#"struct Params {{ size: u32 }};
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let r = gid.x;
    if (r >= arrayLength(&output)) {{ return; }}
    let off = r * params.size;
    var acc: f32 = {init};
    for (var i: u32 = 0u; i < params.size; i = i + 1u) {{
        let x = input[off + i];
        acc = {fold};
    }}
    output[r] = {finish};
}}
"#,
        init = init,
        fold = fold,
        finish = finish,
    )
}

/// `(pipeline_name, WGSL expression in terms of `av`/`bv`)`.
const BINARY_OPS: &[(&str, &str)] = &[
    ("add", "av + bv"),
    ("sub", "av - bv"),
    ("mul", "av * bv"),
    ("div", "av / bv"),
    ("min", "min(av, bv)"),
    ("max", "max(av, bv)"),
    ("pow", "pow(av, bv)"),
    // Explicit `fmod` formulation — WGSL `%` is spec-defined this way
    // for floats, but spelling it out keeps the contract obvious.
    ("mod", "av - bv * trunc(av / bv)"),
    // Comparisons: f32(bool) → 1.0 / 0.0.
    ("equal", "f32(av == bv)"),
    ("less", "f32(av < bv)"),
    ("less_or_equal", "f32(av <= bv)"),
    ("greater", "f32(av > bv)"),
    ("greater_or_equal", "f32(av >= bv)"),
    // Logical ops on f32-truthiness (nonzero == true).
    ("and", "f32(av != 0.0 && bv != 0.0)"),
    ("or", "f32(av != 0.0 || bv != 0.0)"),
    ("xor", "f32((av != 0.0) != (bv != 0.0))"),
    // FusedSwiGlu: out = silu(gate) * up where silu(x) = x / (1 + e^-x).
    ("swiglu", "(av / (1.0 + exp(-av))) * bv"),
];

/// `(pipeline_name, WGSL expression in terms of `x`)`.
const UNARY_OPS: &[(&str, &str)] = &[
    ("neg", "-x"),
    ("relu", "max(x, 0.0)"),
    ("sigmoid", "1.0 / (1.0 + exp(-x))"),
    ("tanh", "tanh(x)"),
    ("exp", "exp(x)"),
    ("log", "log(x)"),
    ("sqrt", "sqrt(x)"),
    ("abs", "abs(x)"),
    ("reciprocal", "1.0 / x"),
    ("sin", "sin(x)"),
    ("cos", "cos(x)"),
    ("floor", "floor(x)"),
    ("ceil", "ceil(x)"),
    // WGSL `round` is half-to-even but the canonical CPU reference
    // (`libm::roundf`) is half-away-from-zero — implement it
    // explicitly so the conformance harness sees the same value.
    ("round", "sign(x) * floor(abs(x) + 0.5)"),
    ("sign", "sign(x)"),
    ("silu", "x / (1.0 + exp(-x))"),
    // GELU (tanh approximation): matches the canonical CPU `gelu`
    // implementation in `hologram-ops` exactly.
    (
        "gelu",
        "0.5 * x * (1.0 + tanh(0.7978845 * (x + 0.044715 * x * x * x)))",
    ),
    // Erf via Abramowitz & Stegun 7.1.26 (≈1.5e-7 max abs error,
    // safely inside Tolerance::TIGHT). Sign-symmetric: erf(-x) = -erf(x).
    (
        "erf",
        "sign(x) * (1.0 - ((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs(x))) - 1.453152027) \
            * (1.0 / (1.0 + 0.3275911 * abs(x))) + 1.421413741) \
            * (1.0 / (1.0 + 0.3275911 * abs(x))) - 0.284496736) \
            * (1.0 / (1.0 + 0.3275911 * abs(x))) + 0.254829592) \
            * (1.0 / (1.0 + 0.3275911 * abs(x))) * exp(-x * x))",
    ),
    // Logical NOT on f32-truthiness: 1.0 if x == 0 else 0.0.
    ("not", "f32(x == 0.0)"),
    ("is_nan", "f32(x != x)"),
];

fn binary_shader_source(op_expr: &str) -> String {
    // ADR-051 step 3: all three bindings declared `read_write` so the
    // layout permits binding the same workspace buffer at multiple
    // offsets in one dispatch (wgpu's usage-scope rule). The shader
    // still treats `a` and `b` as input-only (no writes).
    format!(
        r#"@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&a)) {{ return; }}
    let av = a[i];
    let bv = b[i];
    c[i] = {op_expr};
}}
"#,
        op_expr = op_expr
    )
}

fn unary_shader_source(op_expr: &str) -> String {
    // ADR-051 step 3: input also declared `read_write` so the layout
    // permits the same workspace buffer at both offsets in one
    // dispatch. The shader still treats `input` as read-only at the
    // access level.
    format!(
        r#"@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&input)) {{ return; }}
    let x = input[i];
    output[i] = {op_expr};
}}
"#,
        op_expr = op_expr
    )
}

fn build_pipeline(
    device: &wgpu::Device,
    bind_layout: &wgpu::BindGroupLayout,
    name: &str,
    source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(name),
        bind_group_layouts: &[bind_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Read `dc[i]`, transform via `f`, and accumulate into `dst[i]`.
/// `dst.len == 0` is treated as "not requested" — the caller passes
/// an empty span when the gradient slot is dead.
fn accum_each<F: Fn(f32) -> f32>(storage: &mut [f32], dc: SlotSpan, dst: SlotSpan, f: F) {
    if dst.len == 0 {
        return;
    }
    for i in 0..dc.len {
        let g = storage[dc.offset + i];
        storage[dst.offset + i] += f(g);
    }
}

/// Copy `n = src.len` elements from `src` to `dst` within `storage`.
/// Caller has already validated `src.len == dst.len`. Uses
/// `split_at_mut` so non-overlapping slot spans (the planner
/// guarantee) can be moved without an intermediate `Vec`.
fn copy_disjoint(storage: &mut [f32], src: SlotSpan, dst: SlotSpan) -> Result<(), ExecError> {
    let n = src.len;
    if n == 0 {
        return Ok(());
    }
    if src.offset == dst.offset {
        return Ok(());
    }
    let (lo, hi) = if src.offset < dst.offset {
        (src.offset, dst.offset)
    } else {
        (dst.offset, src.offset)
    };
    let (left, right) = storage.split_at_mut(hi);
    if src.offset < dst.offset {
        right[..n].copy_from_slice(&left[lo..lo + n]);
    } else {
        left[lo..lo + n].copy_from_slice(&right[..n]);
    }
    Ok(())
}

/// Map a staging buffer of `n` f32s and return its contents.
/// Polls the device synchronously; safe to call from the dispatch
/// arms which are themselves already synchronous via `pollster`.
fn read_back(device: &wgpu::Device, staging: &wgpu::Buffer, n: usize) -> Result<Vec<f32>, String> {
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| format!("wgpu: map_async channel: {e}"))?
        .map_err(|e| format!("wgpu: map_async failed: {e}"))?;
    let view = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view)[..n].to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

fn storage_binding(slot: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: slot,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_binding(slot: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: slot,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
