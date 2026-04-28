//! Canonical backend trait — the seam between [`CompiledPlan`] and a
//! device-specific dispatcher.
//!
//! [`CanonicalBackend`] is a single-method trait: given the workspace
//! storage and one `KernelCall`, execute it. The default
//! [`CpuBackend`] forwards to [`hologram_ops::dispatch`], which is the
//! reference CPU implementation co-located with the canonical op
//! definitions (per ADR-045).
//!
//! Alternative backends (Metal, WebGPU, Atlas, …) implement this same
//! trait. They typically own device buffers internally and *interpret*
//! the `[f32]` storage slice as a host-side staging buffer; the actual
//! workspace lives on the device. Backends are free to batch / fuse
//! consecutive `KernelCall`s — the trait is per-call by default but
//! [`CanonicalBackend::run`] can be overridden to walk a slice
//! directly.
//!
//! This sits in `hologram-transform` (not `hologram-ops`) because the
//! trait is fundamentally about *running a plan*, not about op
//! identity. `hologram-ops` deliberately knows nothing about plans or
//! backends.
//!
//! [`CompiledPlan`]: crate::plan::CompiledPlan
//! [`CpuBackend`]: self::CpuBackend

use crate::error::ExecError;
use crate::plan::KernelCall;

/// A device-specific dispatcher for canonical [`KernelCall`]s.
///
/// Implementors lower one `KernelCall` at a time onto their device.
/// The `storage` slice is the planner's pre-sized workspace
/// (referenced via `SlotSpan` offsets inside each call). For
/// out-of-process backends (Metal/WebGPU) the slice is the host-side
/// staging area; the backend syncs to/from device memory as needed.
pub trait CanonicalBackend {
    /// Execute one canonical kernel call.
    fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError>;

    /// Walk a slice of calls in order. The default implementation just
    /// loops over [`Self::dispatch`]; backends that benefit from
    /// command-buffer batching (Metal, WebGPU) can override this to
    /// record all calls before submitting.
    fn run(&mut self, storage: &mut [f32], calls: &[KernelCall]) -> Result<(), ExecError> {
        for call in calls {
            self.dispatch(storage, call)?;
        }
        Ok(())
    }

    /// Optional flush (commit + wait). Default is a no-op (CPU).
    /// Device backends override to drain queued work.
    fn flush(&mut self) -> Result<(), ExecError> {
        Ok(())
    }

    /// Diagnostic name (e.g. `"cpu"`, `"metal"`, `"webgpu"`).
    fn name(&self) -> &'static str;
}

/// Reference CPU backend: forwards every call to
/// [`hologram_ops::dispatch`].
///
/// `CpuBackend` is zero-sized and `Copy`. Constructing one is free,
/// so callers usually keep it as a local variable rather than passing
/// it around.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct a fresh CPU backend.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl CanonicalBackend for CpuBackend {
    #[inline]
    fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
        hologram_ops::dispatch(storage, call);
        Ok(())
    }

    #[inline]
    fn name(&self) -> &'static str {
        "cpu"
    }
}

/// Decorator that records every [`KernelCall`] dispatched, then
/// delegates to an inner backend. Useful for plan diagnostics — the
/// recorded sequence is the exact post-planner execution order.
///
/// `Inner` does the real work; `TraceBackend` only adds the audit
/// trail. Records carry the variant discriminant only (cheap +
/// `Copy`); call back through [`TraceBackend::history`] for the full
/// recorded vector.
pub struct TraceBackend<Inner> {
    inner: Inner,
    history: Vec<TraceEntry>,
}

/// One recorded dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEntry {
    /// Stable name of the dispatched call (matches the
    /// `KernelCall` variant — see [`kernel_call_name`]).
    pub name: &'static str,
}

impl<Inner: CanonicalBackend> TraceBackend<Inner> {
    /// Wrap an inner backend.
    pub const fn new(inner: Inner) -> Self {
        Self {
            inner,
            history: Vec::new(),
        }
    }

    /// Recorded calls, in dispatch order.
    #[must_use]
    pub fn history(&self) -> &[TraceEntry] {
        &self.history
    }

    /// Discard the trace and unwrap the inner backend.
    pub fn into_inner(self) -> Inner {
        self.inner
    }
}

impl<Inner: CanonicalBackend> CanonicalBackend for TraceBackend<Inner> {
    fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
        self.history.push(TraceEntry {
            name: kernel_call_name(call),
        });
        self.inner.dispatch(storage, call)
    }

    fn flush(&mut self) -> Result<(), ExecError> {
        self.inner.flush()
    }

    fn name(&self) -> &'static str {
        "trace"
    }
}

/// Stable diagnostic name for a [`KernelCall`] variant. Used by
/// [`TraceBackend`] and by callers that want a printable summary
/// without `Debug` formatting noise. The mapping mirrors the variant
/// names — adding a new `KernelCall` variant requires one new arm
/// here.
#[must_use]
pub fn kernel_call_name(call: &KernelCall) -> &'static str {
    use KernelCall::*;
    match call {
        Add(_) => "Add",
        AddGrad(_) => "AddGrad",
        Sub(_) => "Sub",
        SubGrad(_) => "SubGrad",
        Mul(_) => "Mul",
        MulGrad(_) => "MulGrad",
        DivGrad(_) => "DivGrad",
        NegGrad(_) => "NegGrad",
        UnaryGrad(_, _) => "UnaryGrad",
        MinMaxGrad(_, _) => "MinMaxGrad",
        ReduceGrad(_, _) => "ReduceGrad",
        ConcatGrad(_) => "ConcatGrad",
        SliceGrad(_) => "SliceGrad",
        TransposeGrad(_) => "TransposeGrad",
        PowGrad(_) => "PowGrad",
        SoftmaxGrad(_, _) => "SoftmaxGrad",
        ReduceArgGrad(_, _) => "ReduceArgGrad",
        ReduceProdGrad(_) => "ReduceProdGrad",
        RmsNormGrad(_) => "RmsNormGrad",
        LayerNormGrad(_) => "LayerNormGrad",
        InstanceNormGrad(_) => "InstanceNormGrad",
        AddRmsNormGrad(_) => "AddRmsNormGrad",
        Pool2dGrad(_, _) => "Pool2dGrad",
        GlobalAvgPoolGrad(_) => "GlobalAvgPoolGrad",
        GroupNormGrad(_) => "GroupNormGrad",
        FusedSwiGluGrad(_) => "FusedSwiGluGrad",
        Conv2dGrad(_) => "Conv2dGrad",
        ConvTranspose2dGrad(_) => "ConvTranspose2dGrad",
        AttentionGrad(_) => "AttentionGrad",
        Div(_) => "Div",
        Pow(_) => "Pow",
        Mod(_) => "Mod",
        Min(_) => "Min",
        Max(_) => "Max",
        Equal(_) => "Equal",
        Less(_) => "Less",
        LessOrEqual(_) => "LessOrEqual",
        Greater(_) => "Greater",
        GreaterOrEqual(_) => "GreaterOrEqual",
        And(_) => "And",
        Or(_) => "Or",
        Xor(_) => "Xor",
        Unary(_, _) => "Unary",
        Softmax(_) => "Softmax",
        LogSoftmax(_) => "LogSoftmax",
        Reshape(_) => "Reshape",
        Transpose(_) => "Transpose",
        Slice(_) => "Slice",
        Concat(_) => "Concat",
        RmsNorm(_) => "RmsNorm",
        LayerNorm(_) => "LayerNorm",
        InstanceNorm(_) => "InstanceNorm",
        GroupNorm(_) => "GroupNorm",
        AddRmsNorm(_) => "AddRmsNorm",
        FusedSwiGlu(_) => "FusedSwiGlu",
        MatMul(_) => "MatMul",
        MatMulGradA(_) => "MatMulGradA",
        MatMulGradB(_) => "MatMulGradB",
        Conv2d(_) => "Conv2d",
        ConvTranspose2d(_) => "ConvTranspose2d",
        Pool2d(_, _) => "Pool2d",
        GlobalAvgPool(_) => "GlobalAvgPool",
        Reduce(_, _) => "Reduce",
        Gemm(_) => "Gemm",
        Clip(_) => "Clip",
        CumSum(_) => "CumSum",
        Expand(_) => "Expand",
        Pad(_) => "Pad",
        ResizeNearest(_) => "ResizeNearest",
        ResizeLinear(_) => "ResizeLinear",
        RotaryEmbedding(_) => "RotaryEmbedding",
        Where(_) => "Where",
        Lrn(_) => "Lrn",
        Attention(_) => "Attention",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{AddCall, SlotSpan};

    /// Counts how many times each variant fires; a tiny sanity backend.
    struct CountingBackend {
        calls: usize,
    }

    impl CanonicalBackend for CountingBackend {
        fn dispatch(&mut self, storage: &mut [f32], call: &KernelCall) -> Result<(), ExecError> {
            self.calls += 1;
            // Still execute on CPU so downstream values are correct.
            hologram_ops::dispatch(storage, call);
            Ok(())
        }
        fn name(&self) -> &'static str {
            "counting"
        }
    }

    #[test]
    fn cpu_backend_dispatches_an_add() {
        let mut storage = [1.0_f32, 2.0, 0.0];
        let call = KernelCall::Add(AddCall {
            a: SlotSpan { offset: 0, len: 1 },
            b: SlotSpan { offset: 1, len: 1 },
            c: SlotSpan { offset: 2, len: 1 },
        });
        let mut be = CpuBackend::new();
        be.dispatch(&mut storage, &call).unwrap();
        be.flush().unwrap();
        assert_eq!(storage[2], 3.0);
        assert_eq!(be.name(), "cpu");
    }

    #[test]
    fn trace_backend_records_dispatched_call_names() {
        let mut storage = [1.0_f32, 2.0, 0.0];
        let call = KernelCall::Add(AddCall {
            a: SlotSpan { offset: 0, len: 1 },
            b: SlotSpan { offset: 1, len: 1 },
            c: SlotSpan { offset: 2, len: 1 },
        });
        let mut be = TraceBackend::new(CpuBackend::new());
        be.dispatch(&mut storage, &call).unwrap();
        be.dispatch(&mut storage, &call).unwrap();
        assert_eq!(be.history().len(), 2);
        assert_eq!(be.history()[0].name, "Add");
        assert_eq!(be.history()[1].name, "Add");
        assert_eq!(storage[2], 3.0);
    }

    #[test]
    fn run_walks_all_calls_in_order() {
        let mut storage = [1.0_f32, 2.0, 0.0, 0.0];
        let add_one = KernelCall::Add(AddCall {
            a: SlotSpan { offset: 0, len: 1 },
            b: SlotSpan { offset: 1, len: 1 },
            c: SlotSpan { offset: 2, len: 1 },
        });
        let add_two = KernelCall::Add(AddCall {
            a: SlotSpan { offset: 2, len: 1 },
            b: SlotSpan { offset: 1, len: 1 },
            c: SlotSpan { offset: 3, len: 1 },
        });
        let mut be = CountingBackend { calls: 0 };
        be.run(&mut storage, &[add_one, add_two]).unwrap();
        assert_eq!(be.calls, 2);
        assert_eq!(storage[2], 3.0);
        assert_eq!(storage[3], 5.0);
    }
}
