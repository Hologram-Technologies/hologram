//! KernelCall serialization codec (spec X.1).
//!
//! Each KernelCall encodes as:
//!   - 2 bytes: discriminant (one per `OpKind` variant)
//!   - N bytes: payload (op-specific, see encoders below)
//!
//! The codec is total-roundtrip: `decode(encode(c)) == c`.

use alloc::vec::Vec;
use hologram_backend::{
    AttentionCall, BinaryCall, BufferRef, Conv2dCall, DequantizeCall, ExpandCall, GemmCall,
    Im2ColCall, KernelCall, LayoutCall, LrnCall, MatMulCall, NormCall, PoolCall, ReduceCall,
    RoPECall, SoftmaxCall, TransposeCall, UnaryCall, WhereCall,
};

const D_NEG: u16 = 1;
const D_BNOT: u16 = 2;
const D_SUCC: u16 = 3;
const D_PRED: u16 = 4;
const D_ADD: u16 = 5;
const D_SUB: u16 = 6;
const D_MUL: u16 = 7;
const D_XOR: u16 = 8;
const D_AND: u16 = 9;
const D_OR: u16 = 10;
const D_RELU: u16 = 11;
const D_SIGMOID: u16 = 12;
const D_TANH: u16 = 13;
const D_GELU: u16 = 14;
const D_SILU: u16 = 15;
const D_ELU: u16 = 16;
const D_SELU: u16 = 17;
const D_EXP: u16 = 18;
const D_LOG: u16 = 19;
const D_LOG1P: u16 = 20;
const D_SQRT: u16 = 21;
const D_RECIP: u16 = 22;
const D_SIN: u16 = 23;
const D_COS: u16 = 24;
const D_TAN: u16 = 25;
const D_ASIN: u16 = 26;
const D_ACOS: u16 = 27;
const D_ATAN: u16 = 28;
const D_CEIL: u16 = 29;
const D_FLOOR: u16 = 30;
const D_ROUND: u16 = 31;
const D_ERF: u16 = 32;
const D_ISNAN: u16 = 33;
const D_SIGN: u16 = 34;
const D_ABS: u16 = 35;
const D_DIV: u16 = 36;
const D_POW: u16 = 37;
const D_MOD: u16 = 38;
const D_MIN: u16 = 39;
const D_MAX: u16 = 40;
const D_EQ: u16 = 41;
const D_LT: u16 = 42;
const D_LE: u16 = 43;
const D_GT: u16 = 44;
const D_GE: u16 = 45;
const D_MATMUL: u16 = 46;
const D_GEMM: u16 = 47;
const D_CONV2D: u16 = 48;
const D_CONVT: u16 = 49;
const D_IM2COL: u16 = 109;
const D_COL2IM: u16 = 110;
const D_LN: u16 = 50;
const D_RN: u16 = 51;
const D_GN: u16 = 52;
const D_IN: u16 = 53;
const D_ARN: u16 = 54;
const D_RSUM: u16 = 55;
const D_RMEAN: u16 = 56;
const D_RPROD: u16 = 57;
const D_RMIN: u16 = 58;
const D_RMAX: u16 = 59;
const D_RESHAPE: u16 = 60;
const D_TRANSPOSE: u16 = 61;
const D_CONCAT: u16 = 62;
const D_SLICE: u16 = 63;
const D_SOFTMAX: u16 = 64;
const D_LSOFTMAX: u16 = 65;
const D_MAXPOOL: u16 = 66;
const D_AVGPOOL: u16 = 67;
const D_GAVGPOOL: u16 = 68;
const D_ATTN: u16 = 69;
const D_FSWG: u16 = 70;
const D_PAD: u16 = 71;
const D_EXPAND: u16 = 72;
const D_RESIZE: u16 = 73;
const D_CUMSUM: u16 = 74;
const D_ROTARY: u16 = 75;
const D_CLIP: u16 = 76;
const D_LRN: u16 = 77;
const D_WHERE: u16 = 78;
// 79..=104 were the backward `*Grad` op-kinds (superseded by composition
// autodiff in `hologram_graph::append_backward`); removed. Opcodes are not
// reused — the decoder rejects them.
const D_DEQ: u16 = 105;
const D_MMA: u16 = 106;
const D_MMADD: u16 = 107;
const D_MMAA: u16 = 108;

pub fn encode_calls(calls: &[KernelCall]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + calls.len() * 64);
    out.extend_from_slice(&(calls.len() as u32).to_le_bytes());
    for c in calls {
        encode_one(c, &mut out);
    }
    out
}

fn encode_one(call: &KernelCall, out: &mut Vec<u8>) {
    use KernelCall as K;
    match call {
        K::Neg(c) => {
            put_u16(out, D_NEG);
            put_unary(out, c);
        }
        K::Bnot(c) => {
            put_u16(out, D_BNOT);
            put_unary(out, c);
        }
        K::Succ(c) => {
            put_u16(out, D_SUCC);
            put_unary(out, c);
        }
        K::Pred(c) => {
            put_u16(out, D_PRED);
            put_unary(out, c);
        }
        K::Add(c) => {
            put_u16(out, D_ADD);
            put_binary(out, c);
        }
        K::Sub(c) => {
            put_u16(out, D_SUB);
            put_binary(out, c);
        }
        K::Mul(c) => {
            put_u16(out, D_MUL);
            put_binary(out, c);
        }
        K::Xor(c) => {
            put_u16(out, D_XOR);
            put_binary(out, c);
        }
        K::And(c) => {
            put_u16(out, D_AND);
            put_binary(out, c);
        }
        K::Or(c) => {
            put_u16(out, D_OR);
            put_binary(out, c);
        }

        K::Relu(c) => {
            put_u16(out, D_RELU);
            put_unary(out, c);
        }
        K::Sigmoid(c) => {
            put_u16(out, D_SIGMOID);
            put_unary(out, c);
        }
        K::Tanh(c) => {
            put_u16(out, D_TANH);
            put_unary(out, c);
        }
        K::Gelu(c) => {
            put_u16(out, D_GELU);
            put_unary(out, c);
        }
        K::Silu(c) => {
            put_u16(out, D_SILU);
            put_unary(out, c);
        }
        K::Elu(c) => {
            put_u16(out, D_ELU);
            put_unary(out, c);
        }
        K::Selu(c) => {
            put_u16(out, D_SELU);
            put_unary(out, c);
        }
        K::Exp(c) => {
            put_u16(out, D_EXP);
            put_unary(out, c);
        }
        K::Log(c) => {
            put_u16(out, D_LOG);
            put_unary(out, c);
        }
        K::Log1p(c) => {
            put_u16(out, D_LOG1P);
            put_unary(out, c);
        }
        K::Sqrt(c) => {
            put_u16(out, D_SQRT);
            put_unary(out, c);
        }
        K::Reciprocal(c) => {
            put_u16(out, D_RECIP);
            put_unary(out, c);
        }
        K::Sin(c) => {
            put_u16(out, D_SIN);
            put_unary(out, c);
        }
        K::Cos(c) => {
            put_u16(out, D_COS);
            put_unary(out, c);
        }
        K::Tan(c) => {
            put_u16(out, D_TAN);
            put_unary(out, c);
        }
        K::Asin(c) => {
            put_u16(out, D_ASIN);
            put_unary(out, c);
        }
        K::Acos(c) => {
            put_u16(out, D_ACOS);
            put_unary(out, c);
        }
        K::Atan(c) => {
            put_u16(out, D_ATAN);
            put_unary(out, c);
        }
        K::Ceil(c) => {
            put_u16(out, D_CEIL);
            put_unary(out, c);
        }
        K::Floor(c) => {
            put_u16(out, D_FLOOR);
            put_unary(out, c);
        }
        K::Round(c) => {
            put_u16(out, D_ROUND);
            put_unary(out, c);
        }
        K::Erf(c) => {
            put_u16(out, D_ERF);
            put_unary(out, c);
        }
        K::IsNaN(c) => {
            put_u16(out, D_ISNAN);
            put_unary(out, c);
        }
        K::Sign(c) => {
            put_u16(out, D_SIGN);
            put_unary(out, c);
        }
        K::Abs(c) => {
            put_u16(out, D_ABS);
            put_unary(out, c);
        }

        K::Div(c) => {
            put_u16(out, D_DIV);
            put_binary(out, c);
        }
        K::Pow(c) => {
            put_u16(out, D_POW);
            put_binary(out, c);
        }
        K::Mod(c) => {
            put_u16(out, D_MOD);
            put_binary(out, c);
        }
        K::Min(c) => {
            put_u16(out, D_MIN);
            put_binary(out, c);
        }
        K::Max(c) => {
            put_u16(out, D_MAX);
            put_binary(out, c);
        }
        K::Equal(c) => {
            put_u16(out, D_EQ);
            put_binary(out, c);
        }
        K::Less(c) => {
            put_u16(out, D_LT);
            put_binary(out, c);
        }
        K::LessOrEqual(c) => {
            put_u16(out, D_LE);
            put_binary(out, c);
        }
        K::Greater(c) => {
            put_u16(out, D_GT);
            put_binary(out, c);
        }
        K::GreaterOrEqual(c) => {
            put_u16(out, D_GE);
            put_binary(out, c);
        }

        K::MatMul(c) => {
            put_u16(out, D_MATMUL);
            put_matmul(out, c);
        }
        K::Gemm(c) => {
            put_u16(out, D_GEMM);
            put_gemm(out, c);
        }
        K::Conv2d(c) => {
            put_u16(out, D_CONV2D);
            put_conv(out, c);
        }
        K::ConvTranspose2d(c) => {
            put_u16(out, D_CONVT);
            put_conv(out, c);
        }
        K::Im2Col(c) => {
            put_u16(out, D_IM2COL);
            put_im2col(out, c);
        }
        K::Col2Im(c) => {
            put_u16(out, D_COL2IM);
            put_im2col(out, c);
        }

        K::LayerNorm(c) => {
            put_u16(out, D_LN);
            put_norm(out, c);
        }
        K::RmsNorm(c) => {
            put_u16(out, D_RN);
            put_norm(out, c);
        }
        K::GroupNorm(c) => {
            put_u16(out, D_GN);
            put_norm(out, c);
        }
        K::InstanceNorm(c) => {
            put_u16(out, D_IN);
            put_norm(out, c);
        }
        K::AddRmsNorm(c) => {
            put_u16(out, D_ARN);
            put_norm(out, c);
        }

        K::ReduceSum(c) => {
            put_u16(out, D_RSUM);
            put_reduce(out, c);
        }
        K::ReduceMean(c) => {
            put_u16(out, D_RMEAN);
            put_reduce(out, c);
        }
        K::ReduceProd(c) => {
            put_u16(out, D_RPROD);
            put_reduce(out, c);
        }
        K::ReduceMin(c) => {
            put_u16(out, D_RMIN);
            put_reduce(out, c);
        }
        K::ReduceMax(c) => {
            put_u16(out, D_RMAX);
            put_reduce(out, c);
        }

        K::Reshape(c) => {
            put_u16(out, D_RESHAPE);
            put_layout(out, c);
        }
        K::Transpose(c) => {
            put_u16(out, D_TRANSPOSE);
            put_transpose(out, c);
        }
        K::Concat(c) => {
            put_u16(out, D_CONCAT);
            put_binary(out, c);
        }
        K::Slice(c) => {
            put_u16(out, D_SLICE);
            put_layout(out, c);
        }

        K::Softmax(c) => {
            put_u16(out, D_SOFTMAX);
            put_softmax(out, c);
        }
        K::LogSoftmax(c) => {
            put_u16(out, D_LSOFTMAX);
            put_softmax(out, c);
        }

        K::MaxPool2d(c) => {
            put_u16(out, D_MAXPOOL);
            put_pool(out, c);
        }
        K::AvgPool2d(c) => {
            put_u16(out, D_AVGPOOL);
            put_pool(out, c);
        }
        K::GlobalAvgPool(c) => {
            put_u16(out, D_GAVGPOOL);
            put_pool(out, c);
        }

        K::Attention(c) => {
            put_u16(out, D_ATTN);
            put_attn(out, c);
        }
        K::FusedSwiGlu(c) => {
            put_u16(out, D_FSWG);
            put_matmul(out, c);
        }

        K::Pad(c) => {
            put_u16(out, D_PAD);
            put_layout(out, c);
        }
        K::Expand(c) => {
            put_u16(out, D_EXPAND);
            put_expand(out, c);
        }
        K::Resize(c) => {
            put_u16(out, D_RESIZE);
            put_expand(out, c);
        }
        K::CumSum(c) => {
            put_u16(out, D_CUMSUM);
            put_reduce(out, c);
        }
        K::RotaryEmbedding(c) => {
            put_u16(out, D_ROTARY);
            put_rope(out, c);
        }
        K::Clip(c) => {
            put_u16(out, D_CLIP);
            put_unary(out, c);
        }
        K::Lrn(c) => {
            put_u16(out, D_LRN);
            put_lrn(out, c);
        }
        K::Where(c) => {
            put_u16(out, D_WHERE);
            put_where(out, c);
        }

        K::Dequantize(c) => {
            put_u16(out, D_DEQ);
            put_dequantize(out, c);
        }
        K::MatMulActivation(c) => {
            put_u16(out, D_MMA);
            put_matmul(out, &c.mm);
            put_u8(out, c.act);
        }
        K::MatMulAdd(c) => {
            put_u16(out, D_MMADD);
            put_matmul(out, &c.mm);
            put_buf(out, c.residual);
        }
        K::MatMulAddActivation(c) => {
            put_u16(out, D_MMAA);
            put_matmul(out, &c.mm);
            put_buf(out, c.residual);
            put_u8(out, c.act);
        }
    }
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn put_buf(out: &mut Vec<u8>, b: BufferRef) {
    put_u32(out, b.slot);
    put_u64(out, b.offset);
    put_u64(out, b.length);
}

fn put_unary(out: &mut Vec<u8>, c: &UnaryCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u16(out, c.witt_bits);
    put_u8(out, c.dtype);
}
fn put_binary(out: &mut Vec<u8>, c: &BinaryCall) {
    put_buf(out, c.a);
    put_buf(out, c.b);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u16(out, c.witt_bits);
    put_u8(out, c.dtype);
}
fn put_matmul(out: &mut Vec<u8>, c: &MatMulCall) {
    put_buf(out, c.a);
    put_buf(out, c.b);
    put_buf(out, c.output);
    put_u32(out, c.m);
    put_u32(out, c.k);
    put_u32(out, c.n);
    put_u8(out, c.dtype);
    put_u8(out, c.b_packed as u8);
}
fn put_gemm(out: &mut Vec<u8>, c: &GemmCall) {
    put_buf(out, c.a);
    put_buf(out, c.b);
    put_buf(out, c.c);
    put_buf(out, c.output);
    put_u32(out, c.m);
    put_u32(out, c.k);
    put_u32(out, c.n);
    put_u64(out, c.alpha_bits);
    put_u64(out, c.beta_bits);
    put_u8(out, c.dtype);
}
fn put_im2col(out: &mut Vec<u8>, c: &Im2ColCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u32(out, c.channels);
    put_u32(out, c.h_in);
    put_u32(out, c.w_in);
    put_u32(out, c.h_out);
    put_u32(out, c.w_out);
    put_u32(out, c.k_h);
    put_u32(out, c.k_w);
    put_u32(out, c.stride_h);
    put_u32(out, c.stride_w);
    put_u8(out, c.dtype);
}
fn put_conv(out: &mut Vec<u8>, c: &Conv2dCall) {
    put_buf(out, c.x);
    put_buf(out, c.w);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.channels_in);
    put_u32(out, c.channels_out);
    put_u32(out, c.h_in);
    put_u32(out, c.w_in);
    put_u32(out, c.h_out);
    put_u32(out, c.w_out);
    put_u32(out, c.k_h);
    put_u32(out, c.k_w);
    put_u32(out, c.stride_h);
    put_u32(out, c.stride_w);
    put_u32(out, c.pad_h);
    put_u32(out, c.pad_w);
    put_u8(out, c.dtype);
}
fn put_norm(out: &mut Vec<u8>, c: &NormCall) {
    put_buf(out, c.x);
    put_buf(out, c.gamma);
    put_buf(out, c.beta);
    put_buf(out, c.residual);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.feature);
    put_u32(out, c.channels);
    put_u32(out, c.num_groups);
    put_u64(out, c.epsilon_bits);
    put_u8(out, c.dtype);
}
fn put_reduce(out: &mut Vec<u8>, c: &ReduceCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u8(out, c.rank);
    put_u32(out, c.axes_mask);
    put_u8(out, c.keepdims as u8);
    put_u8(out, c.dtype);
    for i in 0..c.rank as usize {
        put_u32(out, c.dims[i]);
    }
}
fn put_layout(out: &mut Vec<u8>, c: &LayoutCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u8(out, c.dtype);
}
fn put_transpose(out: &mut Vec<u8>, c: &TransposeCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u8(out, c.rank);
    for d in c.dims {
        put_u32(out, d);
    }
    for p in c.perm {
        put_u8(out, p);
    }
    put_u8(out, c.dtype);
}
fn put_lrn(out: &mut Vec<u8>, c: &LrnCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.channels);
    put_u32(out, c.inner);
    put_u32(out, c.size);
    put_u32(out, c.alpha_bits);
    put_u32(out, c.beta_bits);
    put_u32(out, c.bias_bits);
    put_u8(out, c.dtype);
}
fn put_rope(out: &mut Vec<u8>, c: &RoPECall) {
    put_buf(out, c.x);
    put_buf(out, c.cos);
    put_buf(out, c.sin);
    put_buf(out, c.output);
    put_u32(out, c.head_dim);
    put_u64(out, c.element_count);
    put_u8(out, c.dtype);
}
fn put_expand(out: &mut Vec<u8>, c: &ExpandCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u8(out, c.rank);
    for d in c.in_dims {
        put_u32(out, d);
    }
    for d in c.out_dims {
        put_u32(out, d);
    }
    put_u8(out, c.dtype);
}
fn put_softmax(out: &mut Vec<u8>, c: &SoftmaxCall) {
    put_buf(out, c.input);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.feature);
    put_u8(out, c.dtype);
}
fn put_pool(out: &mut Vec<u8>, c: &PoolCall) {
    put_buf(out, c.x);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.channels);
    put_u32(out, c.h_in);
    put_u32(out, c.w_in);
    put_u32(out, c.h_out);
    put_u32(out, c.w_out);
    put_u32(out, c.k_h);
    put_u32(out, c.k_w);
    put_u32(out, c.stride_h);
    put_u32(out, c.stride_w);
    put_u8(out, c.dtype);
}
fn put_attn(out: &mut Vec<u8>, c: &AttentionCall) {
    put_buf(out, c.q);
    put_buf(out, c.k);
    put_buf(out, c.v);
    put_buf(out, c.output);
    put_u32(out, c.batch);
    put_u32(out, c.heads);
    put_u32(out, c.seq);
    put_u32(out, c.head_dim);
    put_u8(out, c.dtype);
}
fn put_where(out: &mut Vec<u8>, c: &WhereCall) {
    put_buf(out, c.cond);
    put_buf(out, c.a);
    put_buf(out, c.b);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u8(out, c.dtype);
}
fn put_dequantize(out: &mut Vec<u8>, c: &DequantizeCall) {
    put_buf(out, c.input);
    put_buf(out, c.scales);
    put_buf(out, c.zero_points);
    put_buf(out, c.output);
    put_u64(out, c.element_count);
    put_u32(out, c.channels);
    put_u32(out, c.inner);
    put_u8(out, c.quant_dtype);
    put_u8(out, c.dtype);
    put_u32(out, c.scale_bits);
    put_u32(out, c.zero_point as u32);
}
