//! `InferenceSession` (spec VIII.1).

use hologram_archive::{
    HoloLoader, format::SectionKind, decoder, decode_ports, PortDescriptor,
    constant_codec, decode_exec_plan, decode_weights, WeightFingerprint,
};
use hologram_backend::{Backend, KernelCall};
use crate::buffer::{BufferArena, InputBuffer, OutputBuffer, SlotSpan};
use crate::error::ExecError;
use crate::executor::Executor;

pub struct InferenceSession<B: SessionBackend> {
    /// Compiled kernel calls in topological order (compiler emits them
    /// per `compute_schedule` levels, flattened).
    kernel_calls: Vec<KernelCall>,
    /// Per-level kernel-call indices (spec VIII.2). Each entry holds
    /// indices into `kernel_calls`; the executor walks levels in order,
    /// parallelizing within a level when the backend permits.
    exec_plan: Vec<Vec<u32>>,
    inputs: Vec<PortDescriptor>,
    outputs: Vec<PortDescriptor>,
    workspace: BufferArena,
    backend: B,
    /// Archive's canonical 32-byte content fingerprint (spec X.1).
    /// Routed through `prism::pipeline::run` as a W256 `Term::Literal`
    /// in `execute_attested` so the `Grounded<Digest<32>>` attestation
    /// anchors to *this* session's content, not a static dummy term.
    archive_fingerprint: [u8; 32],
}

/// Backend bounds required for `InferenceSession` execute. Without the
/// `parallel` feature, plain `Backend<WS = BufferArena>` suffices. With
/// the feature on, the backend must be `Clone + Send + Sync` so that
/// per-thread copies can dispatch concurrently against disjoint slots
/// in the same schedule level.
#[cfg(not(feature = "parallel"))]
pub trait SessionBackend: Backend<WS = BufferArena> {}
#[cfg(not(feature = "parallel"))]
impl<B: Backend<WS = BufferArena>> SessionBackend for B {}

#[cfg(feature = "parallel")]
pub trait SessionBackend: Backend<WS = BufferArena> + Clone + Send + Sync {}
#[cfg(feature = "parallel")]
impl<B: Backend<WS = BufferArena> + Clone + Send + Sync> SessionBackend for B {}

impl<B: SessionBackend> InferenceSession<B> {
    /// Load and prepare an `.holo` archive for execution.
    pub fn load(bytes: &[u8], backend: B) -> Result<Self, ExecError> {
        let loader = HoloLoader::from_bytes(bytes)?;
        let archive_fingerprint = loader.fingerprint();
        let plan = loader.into_plan()?;
        let calls_section = plan.section(SectionKind::KernelCalls)?;
        let kernel_calls = decoder::decode_calls(calls_section)
            .map_err(ExecError::Archive)?;

        let inputs = plan.section(SectionKind::Inputs)
            .ok()
            .map(decode_ports)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();
        let outputs = plan.section(SectionKind::Outputs)
            .ok()
            .map(decode_ports)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();

        // Decode the per-level kernel-call schedule (spec VIII.2). If the
        // archive omits an `ExecPlan`, fall back to a single level holding
        // every call (sequential execution).
        let exec_plan: Vec<Vec<u32>> = plan
            .section(SectionKind::ExecPlan)
            .ok()
            .map(decode_exec_plan)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_else(|| vec![(0..kernel_calls.len() as u32).collect()]);

        // Constants are pre-fill payloads that the runtime writes into
        // designated workspace slots before any kernel dispatches.
        // Each entry is either inline bytes or a content-addressed
        // reference into the `Weights` section (spec X.3 + X-7).
        let constant_entries: Vec<constant_codec::ConstantEntry> = plan
            .section(SectionKind::Constants)
            .ok()
            .map(constant_codec::decode)
            .transpose()
            .map_err(ExecError::Archive)?
            .unwrap_or_default();

        // Decode the WeightStore so constant references can resolve.
        // Missing section is fine — only inline-only graphs hit that path.
        let weight_store = plan
            .section(SectionKind::Weights)
            .ok()
            .map(decode_weights)
            .transpose()
            .map_err(ExecError::Archive)?;

        // Provision workspace with **per-slot** byte sizes (spec VIII.3).
        //
        // Earlier revisions sized every slot at the maximum byte count
        // across all references. That makes total memory `slot_count *
        // max_size`, which scales catastrophically when one tensor is
        // GB-sized and the rest are KB-sized. The corrected layout
        // computes a per-slot size from the largest *referencing* call
        // (kernel BufferRef.length, port byte count, or constant body),
        // and lays slots out at cumulative offsets — total memory is
        // `Σ size_i` rather than `n · max_size_i`. This is a hard
        // requirement for trillion-parameter / UHD streaming workloads
        // (spec X-7).
        let mut slot_count: usize = 0;
        let bump = |sc: &mut usize, slot: u32| {
            let need = (slot as usize).saturating_add(1);
            if need > *sc { *sc = need; }
        };
        for b in kernel_calls.iter().flat_map(buffers) {
            if b.slot != u32::MAX { bump(&mut slot_count, b.slot); }
        }
        for p in inputs.iter().chain(outputs.iter()) {
            bump(&mut slot_count, p.slot);
        }
        for e in constant_entries.iter() {
            bump(&mut slot_count, e.slot);
        }

        let mut sizes: Vec<u32> = vec![0u32; slot_count];
        for b in kernel_calls.iter().flat_map(buffers) {
            if b.slot != u32::MAX {
                let s = &mut sizes[b.slot as usize];
                if b.length > *s { *s = b.length; }
            }
        }
        for p in inputs.iter().chain(outputs.iter()) {
            let bytes_per = port_bytes_per_element(p.dtype) as u32;
            let n = p.element_count.saturating_mul(bytes_per);
            let s = &mut sizes[p.slot as usize];
            if n > *s { *s = n; }
        }
        for e in constant_entries.iter() {
            // Inline bodies report their length directly; references
            // resolve through the WeightStore for sizing.
            let n: u32 = if e.by_reference {
                weight_store.as_ref()
                    .and_then(|s| s.get(WeightFingerprint(e.fingerprint)))
                    .map(|b| b.len() as u32)
                    .unwrap_or(0)
            } else {
                e.bytes.len() as u32
            };
            let s = &mut sizes[e.slot as usize];
            if n > *s { *s = n; }
        }
        // Floor each slot at 64 bytes so kernels that compute their own
        // strides always have headroom.
        for s in sizes.iter_mut() { if *s < 64 { *s = 64; } }
        // Round each slot to a 64-byte boundary. The arena's backing
        // storage is 64-byte aligned (see `BufferArena::AlignedBytes`),
        // and rounding individual slot lengths up to multiples of 64
        // keeps the cumulative `offset` of every slot 64-byte aligned —
        // which in turn lets `bytemuck::cast_slice::<u8, f32>` succeed
        // zero-copy on every slot. Without this, mid-arena slots can
        // sit at odd 4-byte boundaries and force the elementwise
        // fallback path. 64 bytes is the AVX-512 / cache-line width.
        for s in sizes.iter_mut() { *s = s.next_multiple_of(64); }

        let mut slots = Vec::with_capacity(slot_count);
        let mut total: usize = 0;
        for &len in &sizes {
            slots.push(SlotSpan {
                offset: total as u32,
                length: len,
            });
            total = total.saturating_add(len as usize);
        }
        let mut workspace = BufferArena::with_capacity(total, slots);

        // Pre-fill workspace slots with each constant's bytes. For
        // inline entries the bytes are local; for content-addressed
        // references the bytes come from the WeightStore (spec X.3).
        for entry in &constant_entries {
            let body: &[u8] = if entry.by_reference {
                weight_store.as_ref()
                    .and_then(|s| s.get(WeightFingerprint(entry.fingerprint)))
                    .unwrap_or(&[])
            } else {
                &entry.bytes
            };
            if let Some(dst) = workspace.write_slot(entry.slot as usize) {
                let n = body.len().min(dst.len());
                dst[..n].copy_from_slice(&body[..n]);
            }
        }

        Ok(Self { kernel_calls, exec_plan, inputs, outputs, workspace, backend, archive_fingerprint })
    }

    /// Execute one inference pass. `inputs` are written into the workspace
    /// at the slots designated by the archive's `Inputs` section. After
    /// execution, the workspace bytes at the `Outputs` slots are returned.
    pub fn execute(&mut self, inputs: &[InputBuffer]) -> Result<Vec<OutputBuffer>, ExecError> {
        if inputs.len() != self.inputs.len() {
            return Err(ExecError::InputMismatch);
        }
        // Materialize inputs into workspace slots. `port.element_count`
        // is in elements; convert to bytes via the port's dtype before
        // copying.
        for (port, buf) in self.inputs.iter().zip(inputs.iter()) {
            let bytes_per = port_bytes_per_element(port.dtype);
            let n_bytes = (port.element_count as usize).saturating_mul(bytes_per);
            let dst = self.workspace.write_slot(port.slot as usize)
                .ok_or(ExecError::InputMismatch)?;
            let take = n_bytes.min(buf.bytes.len()).min(dst.len());
            dst[..take].copy_from_slice(&buf.bytes[..take]);
            // Zero-pad any tail of the input region the caller didn't fill.
            let tail_end = n_bytes.min(dst.len());
            for byte in dst[take..tail_end].iter_mut() { *byte = 0; }
        }

        run_session_levels(
            &mut self.backend,
            &self.kernel_calls,
            &self.exec_plan,
            &mut self.workspace,
        )?;

        // Collect outputs from workspace slots; `port.element_count *
        // bytes_per_element(dtype)` is the byte count to emit.
        let mut out = Vec::with_capacity(self.outputs.len());
        for port in &self.outputs {
            let bytes_per = port_bytes_per_element(port.dtype);
            let n_bytes = (port.element_count as usize).saturating_mul(bytes_per);
            let bytes = self.workspace.read_slot(port.slot as usize)
                .ok_or(ExecError::WorkspaceExhausted)?
                .iter().take(n_bytes).copied().collect();
            out.push(OutputBuffer { bytes });
        }
        Ok(out)
    }

    pub fn kernel_count(&self) -> usize { self.kernel_calls.len() }
    pub fn input_count(&self) -> usize { self.inputs.len() }
    pub fn output_count(&self) -> usize { self.outputs.len() }
    pub fn schedule_levels(&self) -> usize { self.exec_plan.len() }

    /// Per-port descriptors (slot id, element count, dtype tag) for the
    /// archive's inputs / outputs. Callers use these to size caller-side
    /// buffers when wiring through the FFI / async bridges.
    pub fn input_ports(&self) -> &[PortDescriptor] { &self.inputs }
    pub fn output_ports(&self) -> &[PortDescriptor] { &self.outputs }

    /// Byte length of the `i`-th declared output port. Returns 0 when
    /// `i >= output_count()` so callers can pre-size buffers with a
    /// single bounded probe.
    pub fn output_byte_len(&self, i: usize) -> usize {
        self.outputs.get(i)
            .map(|p| (p.element_count as usize) * port_bytes_per_element(p.dtype))
            .unwrap_or(0)
    }

    /// The archive's canonical 32-byte content fingerprint (spec X.1).
    /// Used by `execute_attested` to anchor the prism attestation to
    /// this session's content.
    #[inline]
    pub fn archive_fingerprint(&self) -> [u8; 32] { self.archive_fingerprint }
}

/// Bridge between `InferenceSession::execute` and `Executor::run_levels`.
/// `SessionBackend` already carries the right bounds per feature flag.
fn run_session_levels<B: SessionBackend>(
    backend: &mut B,
    calls: &[KernelCall],
    exec_plan: &[Vec<u32>],
    workspace: &mut BufferArena,
) -> Result<(), ExecError> {
    Executor::run_levels(backend, calls, exec_plan, workspace)
}

impl<B: SessionBackend> InferenceSession<B> {
    pub fn workspace(&self) -> &BufferArena { &self.workspace }
    pub fn workspace_mut(&mut self) -> &mut BufferArena { &mut self.workspace }
}

/// Bytes-per-element for a port descriptor's dtype tag (mirrors
/// `hologram_backend::cpu::dtype` constants but kept local to avoid an
/// upward dependency from exec on the backend's internal module).
const fn port_bytes_per_element(dtype: u8) -> usize {
    match dtype {
        0..=2 => 1,            // BOOL, U8, I8
        6 | 7 => 2,            // F16, BF16
        4 | 8 => 4,            // I32, F32
        3 | 5 | 9 => 8,        // U64, I64, F64
        _ => 1,
    }
}

fn buffers(call: &KernelCall) -> Vec<hologram_backend::BufferRef> {
    use KernelCall as K;
    match call {
        K::Neg(c) | K::Bnot(c) | K::Succ(c) | K::Pred(c)
            | K::Relu(c) | K::Sigmoid(c) | K::Tanh(c) | K::Gelu(c) | K::Silu(c)
            | K::Elu(c) | K::Selu(c)
            | K::Exp(c) | K::Log(c) | K::Log1p(c) | K::Sqrt(c) | K::Reciprocal(c)
            | K::Sin(c) | K::Cos(c) | K::Tan(c) | K::Asin(c) | K::Acos(c) | K::Atan(c)
            | K::Ceil(c) | K::Floor(c) | K::Round(c) | K::Erf(c)
            | K::IsNaN(c) | K::Sign(c) | K::Abs(c)
            | K::RotaryEmbedding(c) | K::Clip(c) | K::Lrn(c) | K::UnaryGrad(c)
            => vec![c.input, c.output],

        K::Add(c) | K::Sub(c) | K::Mul(c) | K::Xor(c) | K::And(c) | K::Or(c)
            | K::Div(c) | K::Pow(c) | K::Mod(c) | K::Min(c) | K::Max(c)
            | K::Equal(c) | K::Less(c) | K::LessOrEqual(c) | K::Greater(c) | K::GreaterOrEqual(c)
            | K::SubGrad(c) | K::MulGrad(c) | K::DivGrad(c) | K::PowGrad(c)
            | K::MinGrad(c) | K::MaxGrad(c)
            => vec![c.a, c.b, c.output],

        K::MatMul(c) | K::FusedSwiGlu(c)
            | K::MatMulGradA(c) | K::MatMulGradB(c) | K::FusedSwiGluGrad(c)
            => vec![c.a, c.b, c.output],

        K::Gemm(c) => vec![c.a, c.b, c.c, c.output],

        K::Conv2d(c) | K::ConvTranspose2d(c)
            | K::Conv2dGradX(c) | K::Conv2dGradW(c)
            => vec![c.x, c.w, c.output],

        K::LayerNorm(c) | K::RmsNorm(c) | K::GroupNorm(c) | K::InstanceNorm(c)
            | K::AddRmsNorm(c)
            | K::LayerNormGrad(c) | K::RmsNormGrad(c) | K::GroupNormGrad(c)
            => vec![c.x, c.gamma, c.beta, c.output],

        K::ReduceSum(c) | K::ReduceMean(c) | K::ReduceProd(c)
            | K::ReduceMin(c) | K::ReduceMax(c) | K::CumSum(c)
            | K::ReduceSumGrad(c) | K::ReduceMeanGrad(c) | K::ReduceProdGrad(c)
            => vec![c.input, c.output],

        K::Reshape(c) | K::Transpose(c) | K::Concat(c) | K::Slice(c)
            | K::Pad(c) | K::Expand(c) | K::Resize(c)
            | K::ConcatGrad(c) | K::SliceGrad(c) | K::PadGrad(c)
            => vec![c.input, c.output],

        K::Softmax(c) | K::LogSoftmax(c)
            | K::SoftmaxGrad(c) | K::LogSoftmaxGrad(c)
            => vec![c.input, c.output],

        K::MaxPool2d(c) | K::AvgPool2d(c) | K::GlobalAvgPool(c)
            | K::AvgPool2dGrad(c) | K::GlobalAvgPoolGrad(c)
            => vec![c.x, c.output],

        K::Attention(c) | K::AttentionGrad(c)
            => vec![c.q, c.k, c.v, c.output],

        K::Where(c) => vec![c.cond, c.a, c.b, c.output],

        K::Dequantize(c) => vec![c.input, c.output],
    }
}
