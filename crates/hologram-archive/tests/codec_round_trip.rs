//! KernelCall codec round-trip tests.

use hologram_archive::{decoder, kernel_codec};
use hologram_compute::{
    BinaryCall, BufferRef, Conv2dCall, GemmCall, KernelCall, LayoutCall, MatMulCall, UnaryCall,
};

fn ref_buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 64,
    }
}

#[test]
fn unary_round_trip() {
    let calls = vec![
        KernelCall::Neg(UnaryCall {
            input: ref_buf(0),
            output: ref_buf(1),
            element_count: 16,
            witt_bits: 8,
            dtype: 1,
        }),
        KernelCall::Relu(UnaryCall {
            input: ref_buf(2),
            output: ref_buf(3),
            element_count: 32,
            witt_bits: 16,
            dtype: 1,
        }),
        KernelCall::Sin(UnaryCall {
            input: ref_buf(4),
            output: ref_buf(5),
            element_count: 64,
            witt_bits: 32,
            dtype: 8,
        }),
    ];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    assert_eq!(decoded.len(), calls.len());
    for (a, b) in calls.iter().zip(decoded.iter()) {
        match (a, b) {
            (KernelCall::Neg(x), KernelCall::Neg(y))
            | (KernelCall::Relu(x), KernelCall::Relu(y))
            | (KernelCall::Sin(x), KernelCall::Sin(y)) => {
                assert_eq!(x.element_count, y.element_count);
                assert_eq!(x.witt_bits, y.witt_bits);
                assert_eq!(x.input.slot, y.input.slot);
                assert_eq!(x.output.slot, y.output.slot);
            }
            _ => panic!("variant mismatch"),
        }
    }
}

#[test]
fn binary_round_trip() {
    let c = BinaryCall {
        a: ref_buf(0),
        b: ref_buf(1),
        output: ref_buf(2),
        element_count: 16,
        witt_bits: 8,
        dtype: 1,
    };
    let calls = vec![KernelCall::Add(c), KernelCall::Mul(c), KernelCall::Xor(c)];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    assert_eq!(decoded.len(), 3);
    assert!(matches!(decoded[0], KernelCall::Add(_)));
    assert!(matches!(decoded[1], KernelCall::Mul(_)));
    assert!(matches!(decoded[2], KernelCall::Xor(_)));
}

#[test]
fn matmul_round_trip() {
    let calls = vec![
        KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 128,
            k: 256,
            n: 128,
            dtype: 0,
            b_packed: false,
        }),
        // The weight-layout monomorphism flag must survive the round-trip.
        KernelCall::MatMul(MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 64,
            k: 64,
            n: 64,
            dtype: 8,
            b_packed: true,
        }),
    ];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::MatMul(d) = &decoded[0] {
        assert_eq!(d.m, 128);
        assert_eq!(d.k, 256);
        assert_eq!(d.n, 128);
        assert!(!d.b_packed);
    } else {
        panic!("not matmul");
    }
    match &decoded[1] {
        KernelCall::MatMul(d) => assert!(d.b_packed, "b_packed must round-trip"),
        _ => panic!("not matmul"),
    }
}

#[test]
fn matmul_activation_round_trip() {
    use hologram_compute::{fused_activation, MatMulActivationCall};
    let calls = vec![KernelCall::MatMulActivation(MatMulActivationCall {
        mm: MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(2),
            m: 64,
            k: 128,
            n: 256,
            dtype: 8,
            b_packed: false,
        },
        act: fused_activation::GELU,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::MatMulActivation(d) = &decoded[0] {
        assert_eq!(d.mm.m, 64);
        assert_eq!(d.mm.k, 128);
        assert_eq!(d.mm.n, 256);
        assert_eq!(d.mm.output.slot, 2);
        assert_eq!(d.act, fused_activation::GELU);
    } else {
        panic!("not matmul-activation");
    }
}

#[test]
fn broadcast_binary_round_trip() {
    use hologram_compute::{broadcast_op, BroadcastBinaryCall};
    let calls = vec![KernelCall::BroadcastBinary(BroadcastBinaryCall {
        small: ref_buf(0),
        other: ref_buf(1),
        output: ref_buf(2),
        rank: 3,
        in_dims: [1, 3, 1, 0, 0, 0, 0, 0],
        out_dims: [2, 3, 4, 0, 0, 0, 0, 0],
        op: broadcast_op::MUL,
        small_is_lhs: false,
        dtype: 8,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::BroadcastBinary(d) = &decoded[0] {
        assert_eq!(d.rank, 3);
        assert_eq!(&d.in_dims[..3], &[1, 3, 1]);
        assert_eq!(&d.out_dims[..3], &[2, 3, 4]);
        assert_eq!(d.op, broadcast_op::MUL);
        assert!(!d.small_is_lhs);
        assert_eq!(d.other.slot, 1);
    } else {
        panic!("not broadcast-binary");
    }
}

#[test]
fn matmul_dequant_round_trip() {
    use hologram_compute::MatMulDequantCall;
    let calls = vec![KernelCall::MatMulDequant(MatMulDequantCall {
        a: ref_buf(0),
        bq: ref_buf(1),
        scales: ref_buf(2),
        zero_points: ref_buf(3),
        output: ref_buf(4),
        m: 32,
        k: 64,
        n: 48,
        channels: 48,
        inner: 1,
        quant_dtype: 2,
        dtype: 8,
        scale_bits: 0.0125f32.to_bits(),
        zero_point: -3,
        bq_omajor: false,
        act_quant: 0,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
        codebook: MatMulDequantCall::NO_CODEBOOK,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::MatMulDequant(d) = &decoded[0] {
        assert_eq!((d.m, d.k, d.n), (32, 64, 48));
        assert_eq!((d.channels, d.inner), (48, 1));
        assert_eq!(d.bq.slot, 1);
        assert_eq!(d.scales.slot, 2);
        assert_eq!(d.zero_points.slot, 3);
        assert_eq!(d.output.slot, 4);
        assert_eq!(d.quant_dtype, 2);
        assert_eq!(d.zero_point, -3);
        assert_eq!(d.scale_bits, 0.0125f32.to_bits());
    } else {
        panic!("not matmul-dequant");
    }
    // Default-form calls must stay on the legacy discriminant so archives
    // that don't use the extension remain byte-identical (κ-stable).
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 111);
}

#[test]
fn matmul_dequant_omajor_w8a8_round_trip() {
    use hologram_compute::{mm_act_quant, MatMulDequantCall};
    let calls = vec![KernelCall::MatMulDequant(MatMulDequantCall {
        a: ref_buf(0),
        bq: ref_buf(1),
        scales: ref_buf(2),
        zero_points: ref_buf(3),
        output: ref_buf(4),
        m: 1,
        k: 896,
        n: 4864,
        channels: 4864,
        inner: 1,
        quant_dtype: 2,
        dtype: 8,
        scale_bits: 0,
        zero_point: 0,
        bq_omajor: true,
        act_quant: mm_act_quant::W8A8_TOKEN_SYM,
        act: 5, // fused_activation::TANH
        residual: ref_buf(7),
        codebook: MatMulDequantCall::NO_CODEBOOK,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    // Extended fields force the v2 discriminant. A codebook-free call must never
    // take the v3 tag — that would re-key every existing W8A8 decode address.
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 116);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::MatMulDequant(d) = &decoded[0] {
        assert_eq!((d.m, d.k, d.n), (1, 896, 4864));
        assert!(d.bq_omajor);
        assert_eq!(d.act_quant, mm_act_quant::W8A8_TOKEN_SYM);
        assert_eq!(d.act, 5, "fused epilogue act must round-trip");
        assert_eq!(d.residual.slot, 7, "epilogue residual must round-trip");
    } else {
        panic!("not matmul-dequant");
    }
}

#[test]
fn dequant_activation_round_trip() {
    use hologram_compute::{lut_act, DequantActivationCall};
    let calls = vec![KernelCall::DequantActivation(DequantActivationCall {
        input: ref_buf(0),
        output: ref_buf(1),
        element_count: 4096,
        quant_dtype: 2,
        act: lut_act::GELU,
        dtype: 8,
        scale_bits: 0.05f32.to_bits(),
        zero_point: -7,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::DequantActivation(d) = &decoded[0] {
        assert_eq!(d.input.slot, 0);
        assert_eq!(d.output.slot, 1);
        assert_eq!(d.element_count, 4096);
        assert_eq!(d.quant_dtype, 2);
        assert_eq!(d.act, lut_act::GELU);
        assert_eq!(d.dtype, 8);
        assert_eq!(d.scale_bits, 0.05f32.to_bits());
        assert_eq!(d.zero_point, -7);
    } else {
        panic!("not dequant-activation");
    }
}

#[test]
fn matmul_add_activation_round_trip() {
    use hologram_compute::{fused_activation, MatMulAddActivationCall};
    let calls = vec![KernelCall::MatMulAddActivation(MatMulAddActivationCall {
        mm: MatMulCall {
            a: ref_buf(0),
            b: ref_buf(1),
            output: ref_buf(3),
            m: 32,
            k: 64,
            n: 48,
            dtype: 8,
            b_packed: false,
        },
        residual: ref_buf(2),
        act: fused_activation::SILU,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::MatMulAddActivation(d) => {
            assert_eq!(d.mm.m, 32);
            assert_eq!(d.mm.n, 48);
            assert_eq!(d.residual.slot, 2);
            assert_eq!(d.act, fused_activation::SILU);
        }
        _ => panic!("not matmul-add-activation"),
    }
}

#[test]
fn im2col_col2im_round_trip() {
    use hologram_compute::Im2ColCall;
    let geom = Im2ColCall {
        input: ref_buf(0),
        output: ref_buf(1),
        channels: 3,
        h_in: 8,
        w_in: 8,
        h_out: 6,
        w_out: 6,
        k_h: 3,
        k_w: 3,
        stride_h: 1,
        stride_w: 1,
        dtype: 8,
    };
    let calls = vec![KernelCall::Im2Col(geom), KernelCall::Col2Im(geom)];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match (&decoded[0], &decoded[1]) {
        (KernelCall::Im2Col(a), KernelCall::Col2Im(b)) => {
            assert_eq!(a.channels, 3);
            assert_eq!(a.k_h, 3);
            assert_eq!(a.h_out, 6);
            assert_eq!(b.w_in, 8);
            assert_eq!(b.stride_h, 1);
        }
        _ => panic!("im2col/col2im did not round-trip"),
    }
}

#[test]
fn gemm_round_trip() {
    let calls = vec![KernelCall::Gemm(GemmCall {
        a: ref_buf(0),
        b: ref_buf(1),
        c: ref_buf(2),
        output: ref_buf(3),
        m: 32,
        k: 64,
        n: 32,
        alpha_bits: 0x3F800000,
        beta_bits: 0,
        dtype: 1,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::Gemm(d) = &decoded[0] {
        assert_eq!(d.alpha_bits, 0x3F800000);
        assert_eq!(d.dtype, 1);
    } else {
        panic!("not gemm");
    }
}

#[test]
fn conv_round_trip() {
    let calls = vec![KernelCall::Conv2d(Conv2dCall {
        x: ref_buf(0),
        w: ref_buf(1),
        output: ref_buf(2),
        batch: 1,
        channels_in: 3,
        channels_out: 16,
        h_in: 224,
        w_in: 224,
        h_out: 222,
        w_out: 222,
        k_h: 3,
        k_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        dtype: 0,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::Conv2d(d) = &decoded[0] {
        assert_eq!(d.h_in, 224);
        assert_eq!(d.k_h, 3);
    } else {
        panic!("not conv");
    }
}

#[test]
fn empty_round_trip() {
    let calls: Vec<KernelCall> = vec![];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn layout_round_trip() {
    let calls = vec![
        KernelCall::Reshape(LayoutCall {
            input: ref_buf(0),
            output: ref_buf(1),
            element_count: 100,
            dtype: 0,
        }),
        KernelCall::Transpose(hologram_compute::TransposeCall {
            input: ref_buf(2),
            output: ref_buf(3),
            rank: 2,
            dims: [3, 4, 0, 0, 0, 0, 0, 0],
            perm: [1, 0, 0, 0, 0, 0, 0, 0],
            dtype: 1,
        }),
    ];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    assert!(matches!(decoded[0], KernelCall::Reshape(_)));
    assert!(matches!(decoded[1], KernelCall::Transpose(_)));
}

#[test]
fn warm_start_round_trip() {
    use hologram_archive::{address_bytes, warm_codec, WarmEntry};
    let entries = vec![
        // Labels-only entry (Layer 1).
        WarmEntry {
            slot: 3,
            label: address_bytes(b"cone-node-a"),
            result: Vec::new(),
        },
        // Materialized-result entry (Layer 2 shape).
        WarmEntry {
            slot: 7,
            label: address_bytes(b"cone-node-b"),
            result: vec![1u8, 2, 3, 4, 5],
        },
    ];
    let bytes = warm_codec::encode(&entries);
    let decoded = warm_codec::decode(&bytes).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].slot, 3);
    assert_eq!(decoded[0].label, address_bytes(b"cone-node-a"));
    assert!(decoded[0].result.is_empty());
    assert_eq!(decoded[1].slot, 7);
    assert_eq!(decoded[1].label, address_bytes(b"cone-node-b"));
    assert_eq!(decoded[1].result, vec![1u8, 2, 3, 4, 5]);
}

#[test]
fn gather_round_trip() {
    use hologram_compute::GatherCall;
    let calls = vec![KernelCall::Gather(GatherCall {
        data: ref_buf(0),
        indices: ref_buf(1),
        output: ref_buf(2),
        outer: 1,
        axis_dim: 32000,
        inner: 768,
        num_indices: 17,
        idx_dtype: 5, // i64
        dtype: 8,     // f32
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::Gather(d) => {
            assert_eq!(d.data.slot, 0);
            assert_eq!(d.indices.slot, 1);
            assert_eq!(d.output.slot, 2);
            assert_eq!(d.outer, 1);
            assert_eq!(d.axis_dim, 32000);
            assert_eq!(d.inner, 768);
            assert_eq!(d.num_indices, 17);
            assert_eq!(d.idx_dtype, 5);
            assert_eq!(d.dtype, 8);
        }
        _ => panic!("not gather"),
    }
}

#[test]
fn kv_cache_write_round_trip() {
    use hologram_compute::KvCacheWriteCall;
    let calls = vec![KernelCall::KvCacheWrite(KvCacheWriteCall {
        cache: ref_buf(0),
        new: ref_buf(1),
        pos: ref_buf(2),
        output: ref_buf(3),
        planes: 2,
        bucket_rows: 32768,
        new_rows: 1,
        row_bytes: 512,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::KvCacheWrite(d) => {
            assert_eq!(d.cache.slot, 0);
            assert_eq!(d.new.slot, 1);
            assert_eq!(d.pos.slot, 2);
            assert_eq!(d.output.slot, 3);
            assert_eq!(d.planes, 2);
            assert_eq!(d.bucket_rows, 32768);
            assert_eq!(d.new_rows, 1);
            assert_eq!(d.row_bytes, 512);
        }
        _ => panic!("not kv_cache_write"),
    }
}

#[test]
fn decode_attention_valid_round_trip() {
    use hologram_compute::DecodeAttentionValidCall;
    let calls = vec![KernelCall::DecodeAttentionValid(DecodeAttentionValidCall {
        q: ref_buf(0),
        k_past: ref_buf(1),
        v_past: ref_buf(2),
        k_new: ref_buf(3),
        v_new: ref_buf(4),
        valid_len: ref_buf(5),
        output: ref_buf(6),
        batch: 1,
        heads: 12,
        kv_heads: 2,
        q_rows: 1,
        past_len: 32768,
        new_len: 1,
        head_dim: 128,
        scale_bits: 0,
        dtype: 8,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::DecodeAttentionValid(d) => {
            assert_eq!(d.q.slot, 0);
            assert_eq!(d.valid_len.slot, 5);
            assert_eq!(d.output.slot, 6);
            assert_eq!(d.heads, 12);
            assert_eq!(d.kv_heads, 2);
            assert_eq!(d.past_len, 32768);
            assert_eq!(d.q_rows, 1);
            assert_eq!(d.new_len, 1);
            assert_eq!(d.head_dim, 128);
            assert_eq!(d.dtype, 8);
        }
        _ => panic!("not decode_attention_valid"),
    }
}

#[test]
fn cast_round_trip() {
    use hologram_compute::CastCall;
    let calls = vec![KernelCall::Cast(CastCall {
        input: ref_buf(0),
        output: ref_buf(1),
        element_count: 256,
        src_dtype: 5, // i64
        dst_dtype: 8, // f32
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::Cast(d) => {
            assert_eq!(d.input.slot, 0);
            assert_eq!(d.output.slot, 1);
            assert_eq!(d.element_count, 256);
            assert_eq!(d.src_dtype, 5);
            assert_eq!(d.dst_dtype, 8);
        }
        _ => panic!("not cast"),
    }
}

/// A vector-quantized `MatMulDequant` carries a **codebook operand**. It takes
/// its own discriminant (117) so no codebook-free archive re-keys, and the
/// operand must survive the round-trip.
#[test]
fn matmul_dequant_with_codebook_round_trips_on_its_own_discriminant() {
    use hologram_compute::{mm_act_quant, MatMulDequantCall};
    let calls = vec![KernelCall::MatMulDequant(MatMulDequantCall {
        a: ref_buf(0),
        bq: ref_buf(1),
        scales: ref_buf(2),
        zero_points: ref_buf(3),
        output: ref_buf(4),
        m: 1,
        k: 16,
        n: 4,
        channels: 4,
        inner: 1,
        quant_dtype: 11, // e8cb
        dtype: 8,
        scale_bits: 0,
        zero_point: 0,
        bq_omajor: true,
        act_quant: mm_act_quant::W8A8_TOKEN_SYM,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
        codebook: ref_buf(9),
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        117,
        "a codebook-carrying call must take the v3 discriminant"
    );
    let decoded = decoder::decode_calls(&bytes).unwrap();
    match &decoded[0] {
        KernelCall::MatMulDequant(c) => {
            assert!(c.has_codebook());
            assert_eq!(c.codebook.slot, ref_buf(9).slot);
            assert_eq!(c.codebook.length, ref_buf(9).length);
            assert_eq!(c.quant_dtype, 11);
        }
        other => panic!("wrong variant: {other:?}"),
    }
    // Re-encoding is byte-stable.
    assert_eq!(kernel_codec::encode_calls(&decoded), bytes);
}

/// The codebook is a **read operand**: it must appear in `buffers()` so it folds
/// into the κ-label. Two models with different codebooks must address
/// differently, and a codebook-free call's operand order must be unchanged.
#[test]
fn codebook_participates_in_the_operand_set() {
    use hologram_compute::MatMulDequantCall;
    let mut c = match &decode_calls_of_vq()[0] {
        KernelCall::MatMulDequant(c) => *c,
        _ => unreachable!(),
    };
    let with = hologram_compute::buffers(&KernelCall::MatMulDequant(c));
    c.codebook = MatMulDequantCall::NO_CODEBOOK;
    let without = hologram_compute::buffers(&KernelCall::MatMulDequant(c));
    assert_eq!(
        with.len(),
        without.len() + 1,
        "the codebook must be an operand"
    );
    // Output stays last; the codebook slots in before it.
    assert_eq!(with.last().unwrap().slot, without.last().unwrap().slot);
    assert!(with.iter().any(|b| b.slot == 9));
    assert!(!without.iter().any(|b| b.slot == 9));
}

fn decode_calls_of_vq() -> Vec<KernelCall> {
    use hologram_compute::{mm_act_quant, MatMulDequantCall};
    vec![KernelCall::MatMulDequant(MatMulDequantCall {
        a: ref_buf(0),
        bq: ref_buf(1),
        scales: ref_buf(2),
        zero_points: ref_buf(3),
        output: ref_buf(4),
        m: 1,
        k: 16,
        n: 4,
        channels: 4,
        inner: 1,
        quant_dtype: 11,
        dtype: 8,
        scale_bits: 0,
        zero_point: 0,
        bq_omajor: true,
        act_quant: mm_act_quant::W8A8_TOKEN_SYM,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
        codebook: ref_buf(9),
    })]
}
