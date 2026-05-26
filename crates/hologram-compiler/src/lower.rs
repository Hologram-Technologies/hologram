//! OpKind -> KernelCall lowering (spec VII.2 step 8 + IX.1).

use alloc::vec::Vec;

use crate::error::CompileError;
use hologram_backend::{
    AttentionCall, BinaryCall, BufferRef, Conv2dCall, DequantizeCall, GemmCall, KernelCall,
    ExpandCall, Im2ColCall, LayoutCall, LrnCall, MatMulCall, NormCall, PoolCall, ReduceCall,
    RoPECall, SoftmaxCall, TransposeCall, UnaryCall, WhereCall,
};
use hologram_graph::{Graph, InputSource, Node, NodeId, OpKind};

/// Op-specific shape parameters resolved from the graph.
///
/// Built by `ShapeArgs::from_graph(&graph, &node)` at compile time.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShapeArgs {
    // MatMul / Gemm
    pub m: u32,
    pub k: u32,
    pub n: u32,

    // Conv2d
    pub batch: u32,
    pub channels_in: u32,
    pub channels_out: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub h_out: u32,
    pub w_out: u32,
    pub k_h: u32,
    pub k_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,

    // Norm / softmax
    pub feature: u32,

    // Pooling (batch + channels share the conv2d fields)

    // Attention
    pub heads: u32,
    pub seq: u32,
    pub head_dim: u32,

    // Reduction
    pub axis_count: u32,
    pub keepdims: bool,

    // Gemm scalars (`Y = α·A·B + β·C`); `f32::to_bits`.
    pub alpha_bits: u32,
    pub beta_bits: u32,
}

impl ShapeArgs {
    /// Resolve op-specific shape parameters from a graph node's inputs and
    /// output shape descriptors. `node_id` is consulted for sparse-keyed
    /// per-node attributes (`Graph::conv_attrs`, etc.); pass the node's
    /// own id.
    pub fn from_graph(graph: &Graph, node_id: NodeId, node: &Node) -> Self {
        let reg = graph.shape_registry();
        let out = reg.get(node.output_shape).cloned();
        let in_shape = |idx: usize| -> Option<hologram_graph::ShapeDescriptor> {
            node.inputs.get(idx).and_then(|src| match *src {
                InputSource::Node(hologram_graph::NodeId(id)) => graph
                    .nodes()
                    .get(id as usize)
                    .and_then(|n| reg.get(n.output_shape).cloned()),
                // Constant operands (e.g. matmul weights) carry a shape too;
                // without this, weight matmuls inferred `m=k=n=0` → a no-op.
                InputSource::Constant(cid) => graph
                    .constants()
                    .get(cid)
                    .and_then(|e| reg.get(e.shape).cloned()),
                // GraphInput ports map to an input node (see compiler.rs);
                // resolve through it so a port operand isn't a silent no-op.
                InputSource::GraphInput(idx) => graph
                    .inputs()
                    .get(idx as usize)
                    .and_then(|&hologram_graph::NodeId(i)| graph.nodes().get(i as usize))
                    .and_then(|n| reg.get(n.output_shape).cloned()),
            })
        };
        let in0 = in_shape(0);
        let in1 = in_shape(1);

        let mut a = Self::default();

        // MatMul / Gemm: A is rank-2 [M, K]; B is rank-2 [K, N]; out is [M, N].
        if let (Some(a_s), Some(b_s)) = (&in0, &in1) {
            if a_s.rank >= 2 && b_s.rank >= 2 {
                a.m = a_s.dim(0).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.k = a_s.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.n = b_s.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }

        // Conv2d / Pool: input X is rank-4 [batch, ch_in, h_in, w_in];
        //                weight W is rank-4 [ch_out, ch_in, k_h, k_w];
        //                output is rank-4 [batch, ch_out, h_out, w_out].
        if let Some(x) = &in0 {
            if x.rank >= 4 {
                a.batch = x.dim(0).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.channels_in = x.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.h_in = x.dim(2).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.w_in = x.dim(3).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }
        if let Some(w) = &in1 {
            if w.rank >= 4 {
                a.channels_out = w.dim(0).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.k_h = w.dim(2).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.k_w = w.dim(3).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }
        if let Some(o) = &out {
            if o.rank >= 4 {
                a.h_out = o.dim(2).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.w_out = o.dim(3).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }

        // Pooling has no weight operand, so `k_h`/`k_w` weren't set above.
        // For non-overlapping windows the kernel equals the spatial ratio
        // `h_in / h_out` (a max/avg pool that tiles the input); derive it when
        // a weight didn't already supply one.
        if a.k_h == 0 && a.h_out > 0 {
            a.k_h = a.h_in / a.h_out;
        }
        if a.k_w == 0 && a.w_out > 0 {
            a.k_w = a.w_in / a.w_out;
        }

        // Convolution stride / padding: take per-node `ConvAttrs` if
        // attached; otherwise default to `(stride = 1, pad = 0)`.
        let conv = graph.conv_attrs(node_id).unwrap_or_default();
        a.stride_h = conv.stride_h.max(1);
        a.stride_w = conv.stride_w.max(1);
        a.pad_h = conv.pad_h;
        a.pad_w = conv.pad_w;

        // Norm / softmax: derive batch + feature from the input rank-2
        // [batch, feature] when MatMul wasn't applicable.
        if let Some(s) = &in0 {
            if s.rank == 2 {
                if a.m == 0 {
                    a.batch = s.dim(0).unwrap_or(0).min(u32::MAX as u64) as u32;
                }
                a.feature = s.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }

        // Attention: input rank-4 [batch, heads, seq, head_dim].
        if let Some(s) = &in0 {
            if s.rank == 4 && a.batch != 0 {
                a.heads = s.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.seq = s.dim(2).unwrap_or(0).min(u32::MAX as u64) as u32;
                a.head_dim = s.dim(3).unwrap_or(0).min(u32::MAX as u64) as u32;
            }
        }

        // Gemm scalars `Y = α·A·B + β·C` — from GemmAttrs (default α=β=1, the
        // plain `A·B + C`; absent attrs previously lowered to α=β=0 ⇒ zero).
        if matches!(node.op, hologram_graph::GraphOp::Op(OpKind::Gemm)) {
            let ga = graph.gemm_attrs(node_id).unwrap_or_default();
            a.alpha_bits = ga.alpha_bits;
            a.beta_bits = ga.beta_bits;
        }

        // im2col / col2im: single-instance `[Cin,Hin,Win]` image (in0 for
        // im2col, the output for col2im) plus the window in `ConvAttrs`. The
        // valid-conv output extent is derived: `Hout = (Hin − kh)/sh + 1`.
        if matches!(
            node.op,
            hologram_graph::GraphOp::Op(OpKind::Im2Col) | hologram_graph::GraphOp::Op(OpKind::Col2Im)
        ) {
            let conv = graph.conv_attrs(node_id).unwrap_or_default();
            a.k_h = conv.k_h;
            a.k_w = conv.k_w;
            a.stride_h = conv.stride_h.max(1);
            a.stride_w = conv.stride_w.max(1);
            let img = if matches!(node.op, hologram_graph::GraphOp::Op(OpKind::Im2Col)) {
                in0.clone()
            } else {
                out.clone()
            };
            if let Some(im) = img {
                if im.rank == 3 {
                    a.channels_in = im.dim(0).unwrap_or(0).min(u32::MAX as u64) as u32;
                    a.h_in = im.dim(1).unwrap_or(0).min(u32::MAX as u64) as u32;
                    a.w_in = im.dim(2).unwrap_or(0).min(u32::MAX as u64) as u32;
                    if a.k_h > 0 && a.k_w > 0 {
                        a.h_out = (a.h_in - a.k_h) / a.stride_h + 1;
                        a.w_out = (a.w_in - a.k_w) / a.stride_w + 1;
                    }
                }
            }
        }

        a
    }
}

/// Resolved per-node lowering inputs.
pub struct LoweredNode {
    pub kind: OpKind,
    pub inputs: Vec<BufferRef>,
    pub output: BufferRef,
    pub element_count: u64,
    pub witt_bits: u16,
    /// Output dtype for compute ops; for quantization ops this is the
    /// destination float dtype (the quant_dtype lives in `quant`).
    pub dtype: u8,
    pub shape: ShapeArgs,
    /// Quantization parameters (spec X-5). Default is "no quantization";
    /// the K::Dequantize arm reads these fields.
    pub quant: QuantParams,
}

/// Per-tensor quantization parameters (spec X-5). The compiler sets
/// these on the node that consumes a quantized weight (or, in fused
/// matmul-with-dequant, the matmul itself).
#[derive(Debug, Default, Clone, Copy)]
pub struct QuantParams {
    /// Source quantized dtype: `DTYPE_I8` (2) or `DTYPE_I4` (10).
    pub quant_dtype: u8,
    /// `f32::to_bits` of the per-tensor scale.
    pub scale_bits: u32,
    /// Symmetric zero-point.
    pub zero_point: i32,
}

pub fn lower(node: &LoweredNode) -> Result<KernelCall, CompileError> {
    use OpKind as K;
    let inp0 = || {
        node.inputs.first().copied().unwrap_or(BufferRef {
            slot: 0,
            offset: 0,
            length: 0,
        })
    };
    let inp1 = || {
        node.inputs.get(1).copied().unwrap_or(BufferRef {
            slot: 0,
            offset: 0,
            length: 0,
        })
    };
    let inp2 = || {
        node.inputs.get(2).copied().unwrap_or(BufferRef {
            slot: 0,
            offset: 0,
            length: 0,
        })
    };
    let s = &node.shape;
    let unary = UnaryCall {
        input: inp0(),
        output: node.output,
        element_count: node.element_count,
        witt_bits: node.witt_bits,
        dtype: node.dtype,
    };
    let binary = BinaryCall {
        a: inp0(),
        b: inp1(),
        output: node.output,
        element_count: node.element_count,
        witt_bits: node.witt_bits,
        dtype: node.dtype,
    };
    let layout = LayoutCall {
        input: inp0(),
        output: node.output,
        element_count: node.element_count,
        dtype: node.dtype,
    };
    let where_call = WhereCall {
        cond: inp0(),
        a: inp1(),
        b: inp2(),
        output: node.output,
        element_count: node.element_count,
        dtype: node.dtype,
    };
    let norm_call = NormCall {
        x: inp0(),
        gamma: inp1(),
        beta: inp2(),
        residual: NormCall::NO_RESIDUAL,
        output: node.output,
        batch: s.batch,
        feature: s.feature,
        epsilon_bits: 0,
        dtype: node.dtype,
    };
    let add_rms_norm_call = NormCall {
        x: inp0(),
        gamma: inp1(),
        beta: NormCall::NO_RESIDUAL,
        residual: inp2(),
        output: node.output,
        batch: s.batch,
        feature: s.feature,
        epsilon_bits: 0,
        dtype: node.dtype,
    };
    let reduce_call = ReduceCall {
        input: inp0(),
        output: node.output,
        element_count: node.element_count,
        axis_count: s.axis_count,
        keepdims: s.keepdims,
        dtype: node.dtype,
    };
    let softmax_call = SoftmaxCall {
        input: inp0(),
        output: node.output,
        batch: s.batch.max(1),
        feature: s.feature,
        dtype: node.dtype,
    };
    let pool_call = PoolCall {
        x: inp0(),
        output: node.output,
        batch: s.batch,
        channels: s.channels_in,
        h_in: s.h_in,
        w_in: s.w_in,
        h_out: s.h_out,
        w_out: s.w_out,
        k_h: s.k_h,
        k_w: s.k_w,
        stride_h: s.stride_h,
        stride_w: s.stride_w,
        dtype: node.dtype,
    };
    let matmul_call = MatMulCall {
        a: inp0(),
        b: inp1(),
        output: node.output,
        m: s.m,
        k: s.k,
        n: s.n,
        dtype: node.dtype,
        // Layout chosen by the post-lowering weight-packing pass; a freshly
        // lowered call is row-major until then.
        b_packed: false,
    };
    let gemm_call = GemmCall {
        a: inp0(),
        b: inp1(),
        c: inp2(),
        output: node.output,
        m: s.m,
        k: s.k,
        n: s.n,
        alpha_bits: s.alpha_bits as u64,
        beta_bits: s.beta_bits as u64,
        dtype: node.dtype,
    };
    let conv_call = Conv2dCall {
        x: inp0(),
        w: inp1(),
        output: node.output,
        batch: s.batch,
        channels_in: s.channels_in,
        channels_out: s.channels_out,
        h_in: s.h_in,
        w_in: s.w_in,
        h_out: s.h_out,
        w_out: s.w_out,
        k_h: s.k_h,
        k_w: s.k_w,
        stride_h: s.stride_h,
        stride_w: s.stride_w,
        pad_h: s.pad_h,
        pad_w: s.pad_w,
        dtype: node.dtype,
    };
    let attn_call = AttentionCall {
        q: inp0(),
        k: inp1(),
        v: inp2(),
        output: node.output,
        batch: s.batch,
        heads: s.heads,
        seq: s.seq,
        head_dim: s.head_dim,
        dtype: node.dtype,
    };
    let im2col_call = Im2ColCall {
        input: inp0(),
        output: node.output,
        channels: s.channels_in,
        h_in: s.h_in,
        w_in: s.w_in,
        h_out: s.h_out,
        w_out: s.w_out,
        k_h: s.k_h,
        k_w: s.k_w,
        stride_h: s.stride_h,
        stride_w: s.stride_w,
        dtype: node.dtype,
    };

    Ok(match node.kind {
        K::Neg => KernelCall::Neg(unary),
        K::Bnot => KernelCall::Bnot(unary),
        K::Succ => KernelCall::Succ(unary),
        K::Pred => KernelCall::Pred(unary),
        K::Add => KernelCall::Add(binary),
        K::Sub => KernelCall::Sub(binary),
        K::Mul => KernelCall::Mul(binary),
        K::Xor => KernelCall::Xor(binary),
        K::And => KernelCall::And(binary),
        K::Or => KernelCall::Or(binary),

        K::Relu => KernelCall::Relu(unary),
        K::Sigmoid => KernelCall::Sigmoid(unary),
        K::Tanh => KernelCall::Tanh(unary),
        K::Gelu => KernelCall::Gelu(unary),
        K::Silu => KernelCall::Silu(unary),
        K::Elu => KernelCall::Elu(unary),
        K::Selu => KernelCall::Selu(unary),
        K::Exp => KernelCall::Exp(unary),
        K::Log => KernelCall::Log(unary),
        K::Log1p => KernelCall::Log1p(unary),
        K::Sqrt => KernelCall::Sqrt(unary),
        K::Reciprocal => KernelCall::Reciprocal(unary),
        K::Sin => KernelCall::Sin(unary),
        K::Cos => KernelCall::Cos(unary),
        K::Tan => KernelCall::Tan(unary),
        K::Asin => KernelCall::Asin(unary),
        K::Acos => KernelCall::Acos(unary),
        K::Atan => KernelCall::Atan(unary),
        K::Ceil => KernelCall::Ceil(unary),
        K::Floor => KernelCall::Floor(unary),
        K::Round => KernelCall::Round(unary),
        K::Erf => KernelCall::Erf(unary),
        K::IsNaN => KernelCall::IsNaN(unary),
        K::Sign => KernelCall::Sign(unary),
        K::Abs => KernelCall::Abs(unary),

        K::Div => KernelCall::Div(binary),
        K::Pow => KernelCall::Pow(binary),
        K::Mod => KernelCall::Mod(binary),
        K::Min => KernelCall::Min(binary),
        K::Max => KernelCall::Max(binary),
        K::Equal => KernelCall::Equal(binary),
        K::Less => KernelCall::Less(binary),
        K::LessOrEqual => KernelCall::LessOrEqual(binary),
        K::Greater => KernelCall::Greater(binary),
        K::GreaterOrEqual => KernelCall::GreaterOrEqual(binary),

        K::MatMul => KernelCall::MatMul(matmul_call),
        K::Gemm => KernelCall::Gemm(gemm_call),
        K::Conv2d => KernelCall::Conv2d(conv_call),
        K::ConvTranspose2d => KernelCall::ConvTranspose2d(conv_call),
        K::Im2Col => KernelCall::Im2Col(im2col_call),
        K::Col2Im => KernelCall::Col2Im(im2col_call),

        K::LayerNorm => KernelCall::LayerNorm(norm_call),
        K::RmsNorm => KernelCall::RmsNorm(norm_call),
        K::GroupNorm => KernelCall::GroupNorm(norm_call),
        K::InstanceNorm => KernelCall::InstanceNorm(norm_call),
        K::AddRmsNorm => KernelCall::AddRmsNorm(add_rms_norm_call),

        K::ReduceSum => KernelCall::ReduceSum(reduce_call),
        K::ReduceMean => KernelCall::ReduceMean(reduce_call),
        K::ReduceProd => KernelCall::ReduceProd(reduce_call),
        K::ReduceMin => KernelCall::ReduceMin(reduce_call),
        K::ReduceMax => KernelCall::ReduceMax(reduce_call),

        K::Reshape => KernelCall::Reshape(layout),
        // dims/perm filled by the compiler's transpose pass (needs the perm
        // operand + input shape); a fresh call is rank-0 until then.
        K::Transpose => KernelCall::Transpose(TransposeCall {
            input: inp0(),
            output: node.output,
            rank: 0,
            dims: [0; 8],
            perm: [0; 8],
            dtype: node.dtype,
        }),
        K::Concat => KernelCall::Concat(binary),
        K::Slice => KernelCall::Slice(layout),

        K::Softmax => KernelCall::Softmax(softmax_call),
        K::LogSoftmax => KernelCall::LogSoftmax(softmax_call),

        K::MaxPool2d => KernelCall::MaxPool2d(pool_call),
        K::AvgPool2d => KernelCall::AvgPool2d(pool_call),
        K::GlobalAvgPool => KernelCall::GlobalAvgPool(pool_call),

        K::Attention => KernelCall::Attention(attn_call),
        K::FusedSwiGlu => KernelCall::FusedSwiGlu(matmul_call),

        K::Pad => KernelCall::Pad(layout),
        // in_dims/out_dims filled by the compiler's expand pass (from the
        // input + output shapes); a fresh call is rank-0 until then.
        K::Expand => KernelCall::Expand(ExpandCall {
            input: inp0(),
            output: node.output,
            rank: 0,
            in_dims: [0; 8],
            out_dims: [0; 8],
            dtype: node.dtype,
        }),
        // in_dims/out_dims filled by the compiler's resize pass.
        K::Resize => KernelCall::Resize(ExpandCall {
            input: inp0(),
            output: node.output,
            rank: 0,
            in_dims: [0; 8],
            out_dims: [0; 8],
            dtype: node.dtype,
        }),
        K::CumSum => KernelCall::CumSum(reduce_call),
        // head_dim filled by the compiler's rope pass (from the input's last
        // dim); cos/sin are operands 1 and 2.
        K::RotaryEmbedding => KernelCall::RotaryEmbedding(RoPECall {
            x: inp0(),
            cos: inp1(),
            sin: inp2(),
            output: node.output,
            head_dim: 0,
            element_count: node.element_count,
            dtype: node.dtype,
        }),
        K::Clip => KernelCall::Clip(unary),
        // size/α/β/bias + batch/channels/inner filled by the compiler's lrn
        // pass (from LrnAttrs + the input shape).
        K::Lrn => KernelCall::Lrn(LrnCall {
            input: inp0(),
            output: node.output,
            batch: 0,
            channels: 0,
            inner: 0,
            size: 1,
            alpha_bits: 0.0001f32.to_bits(),
            beta_bits: 0.75f32.to_bits(),
            bias_bits: 1.0f32.to_bits(),
            dtype: node.dtype,
        }),
        K::Where => KernelCall::Where(where_call),

        // Quantization (spec X-5).
        K::Dequantize => KernelCall::Dequantize(DequantizeCall {
            input: inp0(),
            output: node.output,
            element_count: node.element_count,
            quant_dtype: node.quant.quant_dtype,
            dtype: node.dtype,
            scale_bits: node.quant.scale_bits,
            zero_point: node.quant.zero_point,
        }),
    })
}
