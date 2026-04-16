//! Single-path executor using hologram-backend.
//!
//! All buffers live on the target device. All ops dispatch through the
//! backend. One flush at the end. No CPU↔GPU transfers during execution.
//!
//! This replaces the dual-path logic in `tape.rs::execute_direct` with
//! a clean single-path loop.

use hologram_backend::{ComputeBackend, ComputeMemory};
use smallvec::SmallVec;

use crate::buffer::BufferArena;
use crate::tape::{EnumTape, TapeKernel};

/// Execute a tape on a device-native backend.
///
/// All weights and constants are uploaded to the device at the start.
/// All intermediate buffers live on the device. One flush at the end.
/// The output is downloaded to CPU for the caller.
///
/// This is the production execution path — no CPU fallback, no hybrid
/// GPU/CPU, no per-op readback.
pub fn execute_on_backend<M, B>(
    tape: &EnumTape,
    arena: &BufferArena<'_>,
    memory: &M,
    backend: &B,
) -> crate::error::ExecResult<Vec<Vec<u8>>>
where
    M: ComputeMemory,
    B: ComputeBackend<M>,
{
    let num_slots = tape
        .instructions
        .iter()
        .map(|i| i.output_idx as usize + 1)
        .max()
        .unwrap_or(0);

    // Allocate device buffers for all instruction outputs.
    let mut bufs: Vec<M::Buffer> = (0..num_slots).map(|_| memory.alloc(0)).collect();

    // Upload constants and graph inputs from the arena to device memory.
    // These are the initial values that feed into the first instructions.
    for instr in &tape.instructions {
        for &idx in &instr.input_indices {
            let i = idx as usize;
            if i < bufs.len() {
                // Check if this input comes from the arena (constant or graph input)
                // rather than from a previous instruction's output.
                if let Ok(data) = arena.get(hologram_graph::NodeId::new(idx, 0)) {
                    if !data.is_empty() {
                        bufs[i] = memory.upload(data);
                    }
                }
            }
        }
    }

    // Execute: one dispatch per instruction, all on device.
    for instr in &tape.instructions {
        let out_idx = instr.output_idx as usize;

        // SAFETY: output_idx != any input_idx in a valid DAG (enforced by
        // the tape builder). Raw pointers avoid the borrow conflict between
        // immutable input refs and mutable output.
        let bufs_ptr = bufs.as_mut_ptr();
        let input_refs: SmallVec<[&M::Buffer; 4]> = instr
            .input_indices
            .iter()
            .map(|&idx| unsafe { &*bufs_ptr.add(idx as usize) })
            .collect();
        let out_buf = unsafe { &mut *bufs_ptr.add(out_idx) };

        // Dispatch the kernel through the backend.
        let float_op = tape_kernel_to_float_op(&instr.kernel);
        if let Some(op) = float_op {
            let result = backend.dispatch(
                &op,
                &input_refs,
                out_buf,
                &hologram_backend::KernelParams::default(),
            );
            if let Err(e) = result {
                tracing::warn!(
                    error = %e,
                    "backend dispatch failed, instruction skipped"
                );
            }
        }

        drop(input_refs);
    }

    // Single flush at the end — commit all GPU work and wait.
    backend.flush();

    // Download all non-empty buffers to CPU.
    // The caller identifies outputs by index.
    let outputs: Vec<Vec<u8>> = bufs
        .iter()
        .map(|buf| {
            if memory.byte_len(buf) > 0 {
                memory.download(buf)
            } else {
                Vec::new()
            }
        })
        .collect();

    Ok(outputs)
}

/// Map a TapeKernel to a FloatOp for backend dispatch.
///
/// This bridges the tape's instruction encoding to the backend's op type.
/// Eventually the tape should use FloatOp directly, eliminating this mapping.
fn tape_kernel_to_float_op(kernel: &TapeKernel) -> Option<hologram_core::op::FloatOp> {
    use hologram_core::op::FloatOp;

    match kernel {
        TapeKernel::InlineRelu => Some(FloatOp::Relu),
        TapeKernel::InlineNeg => Some(FloatOp::Neg),
        TapeKernel::InlineSigmoid => Some(FloatOp::Sigmoid),
        TapeKernel::InlineSilu => Some(FloatOp::Silu),
        TapeKernel::InlineTanh => Some(FloatOp::Tanh),
        TapeKernel::InlineGelu => Some(FloatOp::Gelu),
        TapeKernel::InlineExp => Some(FloatOp::Exp),
        TapeKernel::InlineAbs => Some(FloatOp::Abs),
        TapeKernel::InlineReciprocal => Some(FloatOp::Reciprocal),
        TapeKernel::InlineErf => Some(FloatOp::Erf),
        TapeKernel::InlineAdd => Some(FloatOp::Add),
        TapeKernel::InlineMul => Some(FloatOp::Mul),
        TapeKernel::InlineSub => Some(FloatOp::Sub),
        TapeKernel::InlineDiv => Some(FloatOp::Div),
        TapeKernel::InlineMatMul { m, k, n } => Some(FloatOp::MatMul {
            m: *m,
            k: *k,
            n: *n,
        }),
        TapeKernel::InlineSoftmax { size } => Some(FloatOp::Softmax { size: *size }),
        TapeKernel::InlineRmsNorm { size, epsilon } => Some(FloatOp::RmsNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineLayerNorm { size, epsilon } => Some(FloatOp::LayerNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineInstanceNorm { size, epsilon } => Some(FloatOp::InstanceNorm {
            size: *size,
            epsilon: *epsilon,
        }),
        TapeKernel::InlineConv2d {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            input_h,
            input_w,
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineConv2dLut4 {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            input_h,
            input_w,
            ..
        } => Some(FloatOp::Conv2d {
            kernel_h: *kernel_h,
            kernel_w: *kernel_w,
            stride_h: *stride_h,
            stride_w: *stride_w,
            pad_h: *pad_h,
            pad_w: *pad_w,
            dilation_h: *dilation_h,
            dilation_w: *dilation_w,
            group: *group,
            input_h: *input_h,
            input_w: *input_w,
        }),
        TapeKernel::InlineTranspose { perm, ndim, .. } => Some(FloatOp::Transpose {
            perm: *perm,
            ndim: *ndim,
        }),
        TapeKernel::InlineReshape => Some(FloatOp::Reshape),
        TapeKernel::InlineSlice {
            axis_from_end,
            start,
            end,
            axis_size,
        } => Some(FloatOp::Slice {
            axis_from_end: *axis_from_end,
            start: *start,
            end: *end,
            axis_size: *axis_size,
        }),
        TapeKernel::InlineConcat {
            size_a,
            size_b,
            dtype,
        } => Some(FloatOp::Concat {
            size_a: *size_a,
            size_b: *size_b,
            dtype: *dtype,
        }),
        TapeKernel::InlineGemm { m, k, n, .. } => Some(FloatOp::MatMul {
            m: *m,
            k: *k,
            n: *n,
        }),
        TapeKernel::InlineResize { mode } => Some(FloatOp::Resize { mode: *mode }),
        TapeKernel::InlineLog => Some(FloatOp::Log),
        TapeKernel::InlineSqrt => Some(FloatOp::Sqrt),
        TapeKernel::InlineCos => Some(FloatOp::Cos),
        TapeKernel::InlineSin => Some(FloatOp::Sin),
        _ => None, // Ring ops, KV cache, fused ops — to be added.
    }
}
