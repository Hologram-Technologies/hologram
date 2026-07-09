//! Decoder counterpart to `kernel_codec::encode_calls`.

use crate::error::ArchiveError;
use alloc::vec::Vec;
use hologram_backend::{
    AttentionCall, BinaryCall, BroadcastBinaryCall, BufferRef, CastCall, Conv2dCall,
    DequantActivationCall, DequantizeCall, ExpandCall, GatherCall, GemmCall, Im2ColCall,
    KernelCall, LayoutCall, LrnCall, MatMulActivationCall, MatMulAddActivationCall, MatMulAddCall,
    MatMulCall, MatMulDequantCall, NormCall, PoolCall, ReduceCall, RoPECall, SoftmaxCall,
    TransposeCall, UnaryCall, WhereCall, MAX_RANK,
};

/// Cursor over a section payload.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn need(&self, n: usize) -> Result<(), ArchiveError> {
        if self.pos + n > self.bytes.len() {
            Err(ArchiveError::Truncated {
                needed: self.pos + n,
                actual: self.bytes.len(),
            })
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, ArchiveError> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, ArchiveError> {
        self.need(2)?;
        let v = u16::from_le_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, ArchiveError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, ArchiveError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn buf(&mut self) -> Result<BufferRef, ArchiveError> {
        Ok(BufferRef {
            slot: self.u32()?,
            offset: self.u64()?,
            length: self.u64()?,
        })
    }
}

pub fn decode_calls(bytes: &[u8]) -> Result<Vec<KernelCall>, ArchiveError> {
    let mut cur = Cursor::new(bytes);
    let count = cur.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(decode_one(&mut cur)?);
    }
    Ok(out)
}

fn decode_one(cur: &mut Cursor<'_>) -> Result<KernelCall, ArchiveError> {
    let disc = cur.u16()?;
    use KernelCall as K;
    Ok(match disc {
        1 => K::Neg(read_unary(cur)?),
        2 => K::Bnot(read_unary(cur)?),
        3 => K::Succ(read_unary(cur)?),
        4 => K::Pred(read_unary(cur)?),
        5 => K::Add(read_binary(cur)?),
        6 => K::Sub(read_binary(cur)?),
        7 => K::Mul(read_binary(cur)?),
        8 => K::Xor(read_binary(cur)?),
        9 => K::And(read_binary(cur)?),
        10 => K::Or(read_binary(cur)?),
        11 => K::Relu(read_unary(cur)?),
        12 => K::Sigmoid(read_unary(cur)?),
        13 => K::Tanh(read_unary(cur)?),
        14 => K::Gelu(read_unary(cur)?),
        15 => K::Silu(read_unary(cur)?),
        16 => K::Elu(read_unary(cur)?),
        17 => K::Selu(read_unary(cur)?),
        18 => K::Exp(read_unary(cur)?),
        19 => K::Log(read_unary(cur)?),
        20 => K::Log1p(read_unary(cur)?),
        21 => K::Sqrt(read_unary(cur)?),
        22 => K::Reciprocal(read_unary(cur)?),
        23 => K::Sin(read_unary(cur)?),
        24 => K::Cos(read_unary(cur)?),
        25 => K::Tan(read_unary(cur)?),
        26 => K::Asin(read_unary(cur)?),
        27 => K::Acos(read_unary(cur)?),
        28 => K::Atan(read_unary(cur)?),
        29 => K::Ceil(read_unary(cur)?),
        30 => K::Floor(read_unary(cur)?),
        31 => K::Round(read_unary(cur)?),
        32 => K::Erf(read_unary(cur)?),
        33 => K::IsNaN(read_unary(cur)?),
        34 => K::Sign(read_unary(cur)?),
        35 => K::Abs(read_unary(cur)?),
        36 => K::Div(read_binary(cur)?),
        37 => K::Pow(read_binary(cur)?),
        38 => K::Mod(read_binary(cur)?),
        39 => K::Min(read_binary(cur)?),
        40 => K::Max(read_binary(cur)?),
        41 => K::Equal(read_binary(cur)?),
        42 => K::Less(read_binary(cur)?),
        43 => K::LessOrEqual(read_binary(cur)?),
        44 => K::Greater(read_binary(cur)?),
        45 => K::GreaterOrEqual(read_binary(cur)?),
        46 => K::MatMul(read_matmul(cur)?),
        47 => K::Gemm(read_gemm(cur)?),
        48 => K::Conv2d(read_conv(cur)?),
        49 => K::ConvTranspose2d(read_conv(cur)?),
        109 => K::Im2Col(read_im2col(cur)?),
        110 => K::Col2Im(read_im2col(cur)?),
        50 => K::LayerNorm(read_norm(cur)?),
        51 => K::RmsNorm(read_norm(cur)?),
        52 => K::GroupNorm(read_norm(cur)?),
        53 => K::InstanceNorm(read_norm(cur)?),
        54 => K::AddRmsNorm(read_norm(cur)?),
        55 => K::ReduceSum(read_reduce(cur)?),
        56 => K::ReduceMean(read_reduce(cur)?),
        57 => K::ReduceProd(read_reduce(cur)?),
        58 => K::ReduceMin(read_reduce(cur)?),
        59 => K::ReduceMax(read_reduce(cur)?),
        60 => K::Reshape(read_layout(cur)?),
        61 => K::Transpose(read_transpose(cur)?),
        62 => K::Concat(read_binary(cur)?),
        63 => K::Slice(read_layout(cur)?),
        64 => K::Softmax(read_softmax(cur)?),
        65 => K::LogSoftmax(read_softmax(cur)?),
        66 => K::MaxPool2d(read_pool(cur)?),
        67 => K::AvgPool2d(read_pool(cur)?),
        68 => K::GlobalAvgPool(read_pool(cur)?),
        69 => K::Attention(read_attn(cur)?),
        70 => K::FusedSwiGlu(read_matmul(cur)?),
        71 => K::Pad(read_layout(cur)?),
        72 => K::Expand(read_expand(cur)?),
        73 => K::Resize(read_expand(cur)?),
        74 => K::CumSum(read_reduce(cur)?),
        75 => K::RotaryEmbedding(read_rope(cur)?),
        76 => K::Clip(read_unary(cur)?),
        77 => K::Lrn(read_lrn(cur)?),
        78 => K::Where(read_where(cur)?),
        // 79..=104 were the backward `*Grad` op-kinds, removed when autodiff
        // moved to forward-op composition; those opcodes are now rejected.
        105 => K::Dequantize(read_dequantize(cur)?),
        // Dequantize + weight-slot declaration (layout / activation opt-in) and
        // an optional codebook operand. Its own discriminant, so every archive
        // that declares nothing decodes byte-identically.
        118 => {
            let mut c = read_dequantize(cur)?;
            c.weight_layout = cur.u8()?;
            c.act_quant = cur.u8()?;
            c.codebook = cur.buf()?;
            K::Dequantize(c)
        }
        106 => K::MatMulActivation(MatMulActivationCall {
            mm: read_matmul(cur)?,
            act: cur.u8()?,
        }),
        107 => K::MatMulAdd(MatMulAddCall {
            mm: read_matmul(cur)?,
            residual: cur.buf()?,
        }),
        108 => K::MatMulAddActivation(MatMulAddActivationCall {
            mm: read_matmul(cur)?,
            residual: cur.buf()?,
            act: cur.u8()?,
        }),
        111 => K::MatMulDequant(read_matmul_dequant(cur)?),
        // Extended MatMulDequant: legacy body + layout / activation-quant /
        // fused-epilogue tail.
        116 => {
            let mut c = read_matmul_dequant(cur)?;
            c.bq_omajor = cur.u8()? != 0;
            c.act_quant = cur.u8()?;
            c.act = cur.u8()?;
            c.residual = cur.buf()?;
            K::MatMulDequant(c)
        }
        // Extended MatMulDequant + a codebook operand (vector-quantized weight:
        // the stored weight is an index into the model's own codebook).
        117 => {
            let mut c = read_matmul_dequant(cur)?;
            c.bq_omajor = cur.u8()? != 0;
            c.act_quant = cur.u8()?;
            c.act = cur.u8()?;
            c.residual = cur.buf()?;
            c.codebook = cur.buf()?;
            K::MatMulDequant(c)
        }
        112 => K::BroadcastBinary(read_broadcast_binary(cur)?),
        113 => K::DequantActivation(read_dequant_activation(cur)?),
        114 => K::Gather(read_gather(cur)?),
        115 => K::Cast(CastCall {
            input: cur.buf()?,
            output: cur.buf()?,
            element_count: cur.u64()?,
            src_dtype: cur.u8()?,
            dst_dtype: cur.u8()?,
        }),
        _ => return Err(ArchiveError::Io("unknown KernelCall discriminant")),
    })
}

fn read_broadcast_binary(c: &mut Cursor<'_>) -> Result<BroadcastBinaryCall, ArchiveError> {
    let small = c.buf()?;
    let other = c.buf()?;
    let output = c.buf()?;
    let rank = c.u8()?;
    let op = c.u8()?;
    let small_is_lhs = c.u8()? != 0;
    let dtype = c.u8()?;
    let mut in_dims = [0u32; MAX_RANK];
    let mut out_dims = [0u32; MAX_RANK];
    for i in 0..rank as usize {
        in_dims[i] = c.u32()?;
        out_dims[i] = c.u32()?;
    }
    Ok(BroadcastBinaryCall {
        small,
        other,
        output,
        rank,
        in_dims,
        out_dims,
        op,
        small_is_lhs,
        dtype,
    })
}

fn read_matmul_dequant(c: &mut Cursor<'_>) -> Result<MatMulDequantCall, ArchiveError> {
    Ok(MatMulDequantCall {
        a: c.buf()?,
        bq: c.buf()?,
        scales: c.buf()?,
        zero_points: c.buf()?,
        output: c.buf()?,
        m: c.u32()?,
        k: c.u32()?,
        n: c.u32()?,
        channels: c.u32()?,
        inner: c.u32()?,
        quant_dtype: c.u8()?,
        dtype: c.u8()?,
        scale_bits: c.u32()?,
        zero_point: c.u32()? as i32,
        bq_omajor: false,
        act_quant: 0,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
        codebook: MatMulDequantCall::NO_CODEBOOK,
    })
}

fn read_dequant_activation(c: &mut Cursor<'_>) -> Result<DequantActivationCall, ArchiveError> {
    Ok(DequantActivationCall {
        input: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        quant_dtype: c.u8()?,
        act: c.u8()?,
        dtype: c.u8()?,
        scale_bits: c.u32()?,
        zero_point: c.u32()? as i32,
    })
}

fn read_unary(c: &mut Cursor<'_>) -> Result<UnaryCall, ArchiveError> {
    Ok(UnaryCall {
        input: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        witt_bits: c.u16()?,
        dtype: c.u8()?,
    })
}
fn read_binary(c: &mut Cursor<'_>) -> Result<BinaryCall, ArchiveError> {
    Ok(BinaryCall {
        a: c.buf()?,
        b: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        witt_bits: c.u16()?,
        dtype: c.u8()?,
    })
}
fn read_matmul(c: &mut Cursor<'_>) -> Result<MatMulCall, ArchiveError> {
    Ok(MatMulCall {
        a: c.buf()?,
        b: c.buf()?,
        output: c.buf()?,
        m: c.u32()?,
        k: c.u32()?,
        n: c.u32()?,
        dtype: c.u8()?,
        b_packed: c.u8()? != 0,
    })
}
fn read_gemm(c: &mut Cursor<'_>) -> Result<GemmCall, ArchiveError> {
    Ok(GemmCall {
        a: c.buf()?,
        b: c.buf()?,
        c: c.buf()?,
        output: c.buf()?,
        m: c.u32()?,
        k: c.u32()?,
        n: c.u32()?,
        alpha_bits: c.u64()?,
        beta_bits: c.u64()?,
        dtype: c.u8()?,
    })
}
fn read_im2col(c: &mut Cursor<'_>) -> Result<Im2ColCall, ArchiveError> {
    Ok(Im2ColCall {
        input: c.buf()?,
        output: c.buf()?,
        channels: c.u32()?,
        h_in: c.u32()?,
        w_in: c.u32()?,
        h_out: c.u32()?,
        w_out: c.u32()?,
        k_h: c.u32()?,
        k_w: c.u32()?,
        stride_h: c.u32()?,
        stride_w: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_gather(c: &mut Cursor<'_>) -> Result<GatherCall, ArchiveError> {
    Ok(GatherCall {
        data: c.buf()?,
        indices: c.buf()?,
        output: c.buf()?,
        outer: c.u64()?,
        axis_dim: c.u64()?,
        inner: c.u64()?,
        num_indices: c.u64()?,
        idx_dtype: c.u8()?,
        dtype: c.u8()?,
    })
}
fn read_conv(c: &mut Cursor<'_>) -> Result<Conv2dCall, ArchiveError> {
    Ok(Conv2dCall {
        x: c.buf()?,
        w: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        channels_in: c.u32()?,
        channels_out: c.u32()?,
        h_in: c.u32()?,
        w_in: c.u32()?,
        h_out: c.u32()?,
        w_out: c.u32()?,
        k_h: c.u32()?,
        k_w: c.u32()?,
        stride_h: c.u32()?,
        stride_w: c.u32()?,
        pad_h: c.u32()?,
        pad_w: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_norm(c: &mut Cursor<'_>) -> Result<NormCall, ArchiveError> {
    Ok(NormCall {
        x: c.buf()?,
        gamma: c.buf()?,
        beta: c.buf()?,
        residual: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        feature: c.u32()?,
        channels: c.u32()?,
        num_groups: c.u32()?,
        epsilon_bits: c.u64()?,
        dtype: c.u8()?,
    })
}
fn read_reduce(c: &mut Cursor<'_>) -> Result<ReduceCall, ArchiveError> {
    let input = c.buf()?;
    let output = c.buf()?;
    let element_count = c.u64()?;
    let rank = c.u8()?;
    let axes_mask = c.u32()?;
    let keepdims = c.u8()? != 0;
    let dtype = c.u8()?;
    let mut dims = [0u32; MAX_RANK];
    for d in dims.iter_mut().take(rank as usize) {
        *d = c.u32()?;
    }
    Ok(ReduceCall {
        input,
        output,
        element_count,
        rank,
        dims,
        axes_mask,
        keepdims,
        dtype,
    })
}
fn read_layout(c: &mut Cursor<'_>) -> Result<LayoutCall, ArchiveError> {
    Ok(LayoutCall {
        input: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        dtype: c.u8()?,
    })
}
fn read_transpose(c: &mut Cursor<'_>) -> Result<TransposeCall, ArchiveError> {
    let input = c.buf()?;
    let output = c.buf()?;
    let rank = c.u8()?;
    let mut dims = [0u32; MAX_RANK];
    for d in &mut dims {
        *d = c.u32()?;
    }
    let mut perm = [0u8; MAX_RANK];
    for p in &mut perm {
        *p = c.u8()?;
    }
    let dtype = c.u8()?;
    Ok(TransposeCall {
        input,
        output,
        rank,
        dims,
        perm,
        dtype,
    })
}
fn read_expand(c: &mut Cursor<'_>) -> Result<ExpandCall, ArchiveError> {
    let input = c.buf()?;
    let output = c.buf()?;
    let rank = c.u8()?;
    let mut in_dims = [0u32; MAX_RANK];
    for d in &mut in_dims {
        *d = c.u32()?;
    }
    let mut out_dims = [0u32; MAX_RANK];
    for d in &mut out_dims {
        *d = c.u32()?;
    }
    let dtype = c.u8()?;
    Ok(ExpandCall {
        input,
        output,
        rank,
        in_dims,
        out_dims,
        dtype,
    })
}
fn read_lrn(c: &mut Cursor<'_>) -> Result<LrnCall, ArchiveError> {
    Ok(LrnCall {
        input: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        channels: c.u32()?,
        inner: c.u32()?,
        size: c.u32()?,
        alpha_bits: c.u32()?,
        beta_bits: c.u32()?,
        bias_bits: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_rope(c: &mut Cursor<'_>) -> Result<RoPECall, ArchiveError> {
    Ok(RoPECall {
        x: c.buf()?,
        cos: c.buf()?,
        sin: c.buf()?,
        output: c.buf()?,
        head_dim: c.u32()?,
        element_count: c.u64()?,
        dtype: c.u8()?,
    })
}
fn read_softmax(c: &mut Cursor<'_>) -> Result<SoftmaxCall, ArchiveError> {
    Ok(SoftmaxCall {
        input: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        feature: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_pool(c: &mut Cursor<'_>) -> Result<PoolCall, ArchiveError> {
    Ok(PoolCall {
        x: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        channels: c.u32()?,
        h_in: c.u32()?,
        w_in: c.u32()?,
        h_out: c.u32()?,
        w_out: c.u32()?,
        k_h: c.u32()?,
        k_w: c.u32()?,
        stride_h: c.u32()?,
        stride_w: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_attn(c: &mut Cursor<'_>) -> Result<AttentionCall, ArchiveError> {
    Ok(AttentionCall {
        q: c.buf()?,
        k: c.buf()?,
        v: c.buf()?,
        output: c.buf()?,
        batch: c.u32()?,
        heads: c.u32()?,
        seq: c.u32()?,
        head_dim: c.u32()?,
        kv_heads: c.u32()?,
        causal: c.u8()? != 0,
        scale_bits: c.u32()?,
        dtype: c.u8()?,
    })
}
fn read_where(c: &mut Cursor<'_>) -> Result<WhereCall, ArchiveError> {
    Ok(WhereCall {
        cond: c.buf()?,
        a: c.buf()?,
        b: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        dtype: c.u8()?,
    })
}
fn read_dequantize(c: &mut Cursor<'_>) -> Result<DequantizeCall, ArchiveError> {
    Ok(DequantizeCall {
        input: c.buf()?,
        scales: c.buf()?,
        zero_points: c.buf()?,
        output: c.buf()?,
        element_count: c.u64()?,
        channels: c.u32()?,
        inner: c.u32()?,
        quant_dtype: c.u8()?,
        dtype: c.u8()?,
        scale_bits: c.u32()?,
        zero_point: c.u32()? as i32,
        // Declaration-free legacy form; the `D_DEQ2` arm overwrites these.
        codebook: DequantizeCall::NO_CODEBOOK,
        weight_layout: 0,
        act_quant: 0,
    })
}
