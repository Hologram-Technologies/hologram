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
    let calls = vec![KernelCall::MatMul(MatMulCall {
        a: ref_buf(0),
        b: ref_buf(1),
        output: ref_buf(2),
        m: 128,
        k: 256,
        n: 128,
        dtype: 0,
    })];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    if let KernelCall::MatMul(d) = &decoded[0] {
        assert_eq!(d.m, 128);
        assert_eq!(d.k, 256);
        assert_eq!(d.n, 128);
    } else {
        panic!("not matmul");
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
        KernelCall::Transpose(LayoutCall {
            input: ref_buf(2),
            output: ref_buf(3),
            element_count: 200,
            dtype: 1,
        }),
    ];
    let bytes = kernel_codec::encode_calls(&calls);
    let decoded = decoder::decode_calls(&bytes).unwrap();
    assert!(matches!(decoded[0], KernelCall::Reshape(_)));
    assert!(matches!(decoded[1], KernelCall::Transpose(_)));
}
