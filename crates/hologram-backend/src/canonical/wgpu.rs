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
use std::collections::HashMap;

use hologram_transform::{
    AddCall, AddGradCall, AddRmsNormCall, AddRmsNormGradCall, BinaryCall, CanonicalBackend,
    ConcatCall, Conv2dCall, ConvTransposeCall, ExecError, GlobalAvgPoolCall, InstanceNormGradCall,
    KernelCall, LayerNormGradCall, MatMulCall, MatMulGradACall, MatMulGradBCall, NegGradCall,
    NormFullCall, NormScaleCall, Pool2dCall, Pool2dKind, ReduceCall, ReduceKind, ReshapeCall,
    RmsNormGradCall, SliceCall, SlotSpan, SoftmaxCall, SoftmaxGradCall, SoftmaxGradKind,
    SubGradCall, UnaryCall, UnaryKind,
};

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
        // on Metal / Vulkan / WebGPU 1.0).
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage,
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

        let binary_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("binary.binds.layout"),
                entries: &[
                    storage_binding(0, true),
                    storage_binding(1, true),
                    storage_binding(2, false),
                ],
            });
        let unary_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary.binds.layout"),
            entries: &[storage_binding(0, true), storage_binding(1, false)],
        });
        let reduce_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reduce.binds.layout"),
                entries: &[
                    storage_binding(0, true),
                    storage_binding(1, false),
                    uniform_binding(2),
                ],
            });
        // Norm-with-scale layout: input, weight, output, params.
        let norm2_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("norm2.binds.layout"),
            entries: &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                uniform_binding(3),
            ],
        });
        // MatMul + both grad variants: two storage-read inputs, one
        // storage-read_write target (overwrite for forward, accumulate
        // for grad), one uniform `Params { m, k, n }`.
        let matmul_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul.binds.layout"),
                entries: &[
                    storage_binding(0, true),
                    storage_binding(1, true),
                    storage_binding(2, false),
                    uniform_binding(3),
                ],
            });
        // RmsNormGrad / InstanceNormGrad: input + weight + dy + dx
        // (rw, accumulating) + dw (rw, accumulating) + uniform. Both
        // grads share the math signature (no bias).
        let norm_grad2_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("norm_grad2.binds.layout"),
                entries: &[
                    storage_binding(0, true),
                    storage_binding(1, true),
                    storage_binding(2, true),
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
                    storage_binding(0, true),
                    storage_binding(1, true),
                    storage_binding(2, true),
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
                    storage_binding(0, true),
                    storage_binding(1, true),
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
                    storage_binding(0, true),
                    storage_binding(1, true),
                    storage_binding(2, true),
                    storage_binding(3, true),
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
        let conv_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv.binds.layout"),
            entries: &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
                storage_binding(3, false),
                uniform_binding(4),
            ],
        });
        // Norm-with-3-inputs layout: covers LayerNorm (input, weight,
        // bias) and AddRmsNorm (residual, input, weight). Kernels
        // assign meanings to the bindings; the layout is shape-only.
        let norm3_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("norm3.binds.layout"),
            entries: &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, true),
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
        })
    }
}

impl CanonicalBackend for WgpuBackend {
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
            | KernelCall::GroupNorm(_)
            | KernelCall::FusedSwiGlu(_)
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
            pipeline,
            &bind_group,
            &c_buf,
            &staging,
            n,
            &mut storage[c.offset..c.offset + n],
        )
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
            rows,
            &mut storage[call.output.offset..call.output.offset + rows],
        )
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
            call.input.len,
            rows.div_ceil(64) as u32,
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
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
        let params_buf = self.upload_conv_params(
            call.n,
            call.c_in,
            call.c_out,
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
            call.dilation_h,
            call.dilation_w,
            group,
        );

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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
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
        self.run_norm_grad2(
            storage,
            call.input,
            call.weight,
            call.dy,
            call.dx,
            call.dw,
            call.size,
            call.epsilon,
            "rms_norm_grad_dx",
            "rms_norm_grad_dw",
        )
    }

    fn dispatch_instance_norm_grad(
        &self,
        storage: &mut [f32],
        call: &InstanceNormGradCall,
    ) -> Result<(), ExecError> {
        self.run_norm_grad2(
            storage,
            call.input,
            call.weight,
            call.dy,
            call.dx,
            call.dw,
            call.size,
            call.epsilon,
            "instance_norm_grad_dx",
            "instance_norm_grad_dw",
        )
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

    /// Shared dispatch path for the 2-input (no-bias) norm grads.
    /// Runs two passes in one command encoder: pass 1 computes `dx`
    /// per row, pass 2 computes `dw` per column-element.
    #[allow(clippy::too_many_arguments)]
    fn run_norm_grad2(
        &self,
        storage: &mut [f32],
        input: SlotSpan,
        weight: SlotSpan,
        dy: SlotSpan,
        dx: SlotSpan,
        dw: SlotSpan,
        size: u32,
        epsilon_bits: u32,
        dx_op: &'static str,
        dw_op: &'static str,
    ) -> Result<(), ExecError> {
        let size_us = size as usize;
        if size_us == 0 || input.len == 0 {
            return Ok(());
        }
        if !input.len.is_multiple_of(size_us) {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad input length not divisible by size".into(),
            ));
        }
        let rows = input.len / size_us;
        if dx.len != 0 && dx.len != input.len {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dx length must equal input length".into(),
            ));
        }
        if dw.len != 0 && dw.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad dw length must equal size".into(),
            ));
        }
        if dy.len != input.len || weight.len != size_us {
            return Err(ExecError::Backend(
                "WgpuBackend: norm grad span sizes inconsistent".into(),
            ));
        }
        let dx_pipeline = self
            .norm_grad_pipelines
            .get(dx_op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing pipeline {dx_op}")))?;
        let dw_pipeline = self
            .norm_grad_pipelines
            .get(dw_op)
            .ok_or_else(|| ExecError::Backend(format!("WgpuBackend: missing pipeline {dw_op}")))?;

        let in_buf = self.upload(
            &storage[input.offset..input.offset + input.len],
            "norm_grad.in",
        );
        let weight_buf = self.upload(
            &storage[weight.offset..weight.offset + size_us],
            "norm_grad.weight",
        );
        let dy_buf = self.upload(&storage[dy.offset..dy.offset + dy.len], "norm_grad.dy");

        // Accumulating targets — seed from the host workspace.
        let dx_seed = if dx.len > 0 {
            storage[dx.offset..dx.offset + dx.len].to_vec()
        } else {
            vec![0.0_f32; input.len]
        };
        let dx_buf = self.upload_rw(&dx_seed, "norm_grad.dx");
        let dw_seed = if dw.len > 0 {
            storage[dw.offset..dw.offset + size_us].to_vec()
        } else {
            vec![0.0_f32; size_us]
        };
        let dw_buf = self.upload_rw(&dw_seed, "norm_grad.dw");

        // Uniform: [size, epsilon_bits, rows, padding].
        use wgpu::util::DeviceExt;
        let params: [u32; 4] = [size, epsilon_bits, rows as u32, 0];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("norm_grad.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

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
        let dx_staging = self.alloc_staging(input.len);
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
            (input.len * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
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
        if dx.len > 0 {
            let out =
                read_back(&self.device, &dx_staging, input.len).map_err(ExecError::Backend)?;
            storage[dx.offset..dx.offset + dx.len].copy_from_slice(&out);
        }
        if dw.len > 0 {
            let out = read_back(&self.device, &dw_staging, size_us).map_err(ExecError::Backend)?;
            storage[dw.offset..dw.offset + size_us].copy_from_slice(&out);
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
        let params_buf = self.upload_conv_params(
            call.n,
            call.c_in,
            call.c_out,
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
            call.dilation_h,
            call.dilation_w,
            group,
        );
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    /// Pack the 16 dimensional fields shared by `Conv2dCall` and
    /// `ConvTransposeCall` into a uniform buffer matching the WGSL
    /// `ConvParams` struct. Padded to 20 u32s (80 bytes) for a clean
    /// 16-byte alignment.
    #[allow(clippy::too_many_arguments)]
    fn upload_conv_params(
        &self,
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
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let params: [u32; 20] = [
            n, c_in, c_out, h_in, w_in, h_out, w_out, kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, group, 0, 0, 0, 0,
        ];
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
            total_out,
            workgroups,
            &mut storage[call.output.offset..call.output.offset + total_out],
        )
    }

    /// Pack the 12 fields of `Pool2dCall` (excluding spans) into a
    /// uniform buffer matching the WGSL `PoolParams` struct.
    fn upload_pool_params(&self, call: &Pool2dCall) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
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
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pool.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
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
            pipeline,
            &bind_group,
            &c_buf,
            &staging,
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
            pipeline,
            &bind_group,
            &da_buf,
            &staging,
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
            pipeline,
            &bind_group,
            &db_buf,
            &staging,
            k * n,
            workgroups,
            &mut storage[call.db.offset..call.db.offset + k * n],
        )
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
        use wgpu::util::DeviceExt;
        let params: [u32; 3] = [m as u32, k as u32, n as u32];
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Build a uniform buffer holding `[size: u32, epsilon: f32-bits]`.
    /// `epsilon_bits` is the planner-side `f32::to_bits()` encoding;
    /// the WGSL shader declares the field as `f32`, so the bytes
    /// reinterpret correctly without a host-side decode.
    fn upload_norm_params(&self, size: u32, epsilon_bits: u32) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let params: [u32; 2] = [size, epsilon_bits];
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("norm.params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
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
            pipeline,
            &bind_group,
            &out_buf,
            &staging,
            n,
            &mut storage[dst.offset..dst.offset + n],
        )
    }

    fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        out_buf: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        n: usize,
        host_dst: &mut [f32],
    ) -> Result<(), ExecError> {
        // Default: one thread per element (workgroups = ceil(n / 64)).
        self.dispatch_and_read_with_workgroups(
            pipeline,
            bind_group,
            out_buf,
            staging,
            n,
            n.div_ceil(64) as u32,
            host_dst,
        )
    }

    /// Like [`Self::dispatch_and_read`] but the caller chooses the
    /// X-dimension workgroup count. Used by reductions / softmax where
    /// a *thread* maps to a row, not an element.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_and_read_with_workgroups(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        out_buf: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        n: usize,
        workgroups_x: u32,
        host_dst: &mut [f32],
    ) -> Result<(), ExecError> {
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
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
@group(0) @binding(0) var<storage, read> output: array<f32>;
@group(0) @binding(1) var<storage, read> dc: array<f32>;
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
@group(0) @binding(0) var<storage, read> output: array<f32>;
@group(0) @binding(1) var<storage, read> dc: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
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
@group(0) @binding(0) var<storage, read> dc: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
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
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> dc: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
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
@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
@group(0) @binding(0) var<storage, read> input: array<f32>;
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
    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
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
    format!(
        r#"@group(0) @binding(0) var<storage, read> input: array<f32>;
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
