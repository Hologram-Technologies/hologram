//! KernelCall codec round-trip tests.

use hologram_archive::{decoder, kernel_codec};
use hologram_backend::{
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
    use hologram_backend::{fused_activation, MatMulActivationCall};
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
fn matmul_dequant_round_trip() {
    use hologram_backend::MatMulDequantCall;
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
}

#[test]
fn matmul_add_activation_round_trip() {
    use hologram_backend::{fused_activation, MatMulAddActivationCall};
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
    use hologram_backend::Im2ColCall;
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
        KernelCall::Transpose(hologram_backend::TransposeCall {
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
