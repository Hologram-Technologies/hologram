//! `InferenceSession` (spec VIII.1).

use hologram_archive::{HoloLoader, format::SectionKind, decoder, decode_ports, PortDescriptor, schedule_codec};
use hologram_backend::{Backend, KernelCall};
use crate::buffer::{BufferArena, InputBuffer, OutputBuffer, SlotSpan};
use crate::error::ExecError;
use crate::executor::Executor;

pub struct InferenceSession<B: Backend<WS = BufferArena>> {
    kernel_calls: Vec<KernelCall>,
    /// Topological levels (parallel-executable groups). Each entry is a
    /// list of indices into `kernel_calls`. When absent (no schedule in
    /// the archive), the executor walks `kernel_calls` flat in archive
    /// order — which is itself topological per spec VII.2.
    schedule_levels: Option<Vec<Vec<usize>>>,
    inputs: Vec<PortDescriptor>,
    outputs: Vec<PortDescriptor>,
    workspace: BufferArena,
    backend: B,
}

impl<B: Backend<WS = BufferArena>> InferenceSession<B> {
    /// Load and prepare an `.holo` archive for execution.
    pub fn load(bytes: &[u8], backend: B) -> Result<Self, ExecError> {
        let plan = HoloLoader::from_bytes(bytes)?.into_plan()?;
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

        // Decode schedule levels (parallel-execution groups), if present,
        // and project NodeId → kernel_call index.
        let schedule_levels = match plan.section(SectionKind::Schedule).ok() {
            Some(sched_bytes) => {
                let raw = schedule_codec::decode(sched_bytes)
                    .map_err(ExecError::Archive)?;
                // Each NodeId in the raw schedule needs to map to a
                // kernel-call index. Compiler emits kernel calls in
                // schedule order, so kernel_call[i] corresponds to the
                // i-th compute-only node in the topological walk. We
                // build that mapping by iterating the raw schedule's
                // node-id sequence and filtering ids that produced a kernel.
                let mut id_to_call_idx: Vec<Option<usize>> = Vec::new();
                let mut next_idx = 0usize;
                for level in &raw {
                    for &id in level {
                        let i = id as usize;
                        if id_to_call_idx.len() <= i {
                            id_to_call_idx.resize(i + 1, None);
                        }
                        // Heuristic: assume every scheduled node corresponds
                        // to a kernel call slot. Input/Output/Constant nodes
                        // are scheduled too but don't produce kernel calls;
                        // we skip ids that exceed the kernel_calls length.
                        if next_idx < kernel_calls.len() {
                            id_to_call_idx[i] = Some(next_idx);
                            next_idx += 1;
                        }
                    }
                }
                let mut levels: Vec<Vec<usize>> = Vec::with_capacity(raw.len());
                for level in &raw {
                    let mut out_level = Vec::with_capacity(level.len());
                    for &id in level {
                        if let Some(Some(idx)) = id_to_call_idx.get(id as usize) {
                            out_level.push(*idx);
                        }
                    }
                    if !out_level.is_empty() { levels.push(out_level); }
                }
                if levels.is_empty() { None } else { Some(levels) }
            }
            None => None,
        };

        // Provision workspace large enough for every distinct slot referenced
        // by any KernelCall edge or input/output port.
        let max_kernel_slot = kernel_calls.iter().flat_map(buffers).map(|b| b.slot).max().unwrap_or(0);
        let max_kernel_size: u32 = kernel_calls.iter().flat_map(buffers).map(|b| b.length).max().unwrap_or(64);
        let max_port_slot = inputs.iter().chain(outputs.iter()).map(|p| p.slot).max().unwrap_or(0);
        let max_port_size: u32 = inputs.iter().chain(outputs.iter()).map(|p| p.element_count).max().unwrap_or(0);
        let max_slot = max_kernel_slot.max(max_port_slot);
        let max_size = max_kernel_size.max(max_port_size).max(64);
        let slot_count = (max_slot + 1) as usize;
        let mut slots = Vec::with_capacity(slot_count);
        for i in 0..slot_count {
            slots.push(SlotSpan {
                offset: i as u32 * max_size,
                length: max_size,
            });
        }
        let workspace = BufferArena::with_capacity(slot_count * max_size as usize, slots);

        Ok(Self { kernel_calls, schedule_levels, inputs, outputs, workspace, backend })
    }

    /// Execute one inference pass. `inputs` are written into the workspace
    /// at the slots designated by the archive's `Inputs` section. After
    /// execution, the workspace bytes at the `Outputs` slots are returned.
    pub fn execute(&mut self, inputs: &[InputBuffer]) -> Result<Vec<OutputBuffer>, ExecError> {
        if inputs.len() != self.inputs.len() {
            return Err(ExecError::InputMismatch);
        }
        // Materialize inputs into workspace slots.
        for (port, buf) in self.inputs.iter().zip(inputs.iter()) {
            let n = port.element_count as usize;
            let dst = self.workspace.write_slot(port.slot as usize)
                .ok_or(ExecError::InputMismatch)?;
            let take = n.min(buf.bytes.len()).min(dst.len());
            dst[..take].copy_from_slice(&buf.bytes[..take]);
            for byte in dst.iter_mut().take(n).skip(take) { *byte = 0; }
        }

        // Walk the schedule level-by-level when present (spec VIII.2);
        // fall back to flat dispatch when the archive carries no schedule.
        if let Some(levels) = &self.schedule_levels {
            for level in levels {
                for &i in level {
                    if let Some(call) = self.kernel_calls.get(i) {
                        self.backend.dispatch(call, &mut self.workspace)
                            .map_err(|_| ExecError::Backend)?;
                    }
                }
            }
        } else {
            Executor::run(&mut self.backend, &self.kernel_calls, &mut self.workspace)?;
        }

        // Collect outputs from workspace slots.
        let mut out = Vec::with_capacity(self.outputs.len());
        for port in &self.outputs {
            let n = port.element_count as usize;
            let bytes = self.workspace.read_slot(port.slot as usize)
                .ok_or(ExecError::WorkspaceExhausted)?
                .iter().take(n).copied().collect();
            out.push(OutputBuffer { bytes });
        }
        Ok(out)
    }

    pub fn kernel_count(&self) -> usize { self.kernel_calls.len() }
    pub fn input_count(&self) -> usize { self.inputs.len() }
    pub fn output_count(&self) -> usize { self.outputs.len() }
    pub fn workspace(&self) -> &BufferArena { &self.workspace }
    pub fn workspace_mut(&mut self) -> &mut BufferArena { &mut self.workspace }
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
    }
}
