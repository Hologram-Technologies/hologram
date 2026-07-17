//! Golden pin on the two byte-strings that feed **content addressing**:
//! `KernelCall::op_signature()` (opcode + params) and `kernel_codec::encode_calls()`
//! (the archive wire form).
//!
//! A κ-label is `derive_label(opcode, params, operand-labels)`, and the archive
//! replays from the wire bytes — so if either changes for an *existing* call
//! shape, every previously-minted address for that op silently re-keys and all
//! previously deduped/cached content is invalidated. New capabilities must
//! therefore take **new discriminants / new signature tags**, never widen an
//! existing encoding in place.
//!
//! These assertions freeze the legacy shapes (`MatMul`, plain `MatMulDequant`,
//! and the output-major W8A8 decode GEMV for the `i8`/`i4` tiers). Adding a new
//! variant must leave every byte here untouched; a failure here means a
//! previously-addressable computation just changed identity.

use hologram_archive::kernel_codec::encode_calls;
use hologram_compute::{BufferRef, KernelCall, MatMulCall, MatMulDequantCall};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn buf(slot: u32) -> BufferRef {
    BufferRef {
        slot,
        offset: 0,
        length: 64,
    }
}

/// A plain (legacy) dequant matmul — wire `D_MMDQ` (111), signature tag 108.
fn mmdq_plain() -> KernelCall {
    KernelCall::MatMulDequant(MatMulDequantCall {
        a: buf(0),
        bq: buf(1),
        scales: buf(2),
        zero_points: buf(3),
        output: buf(4),
        m: 2,
        k: 8,
        n: 4,
        channels: 4,
        inner: 1,
        quant_dtype: 2, // i8
        dtype: 8,       // f32
        scale_bits: 0,
        zero_point: 0,
        bq_omajor: false,
        act_quant: 0,
        act: 0,
        residual: MatMulDequantCall::NO_RESIDUAL,
        codebook: MatMulDequantCall::NO_CODEBOOK,
    })
}

/// Output-major W8A8 decode GEMV — wire `D_MMDQ2` (116), signature tag 116.
fn mmdq_omajor(quant_dtype: u8) -> KernelCall {
    let mut c = match mmdq_plain() {
        KernelCall::MatMulDequant(c) => c,
        _ => unreachable!(),
    };
    c.m = 1;
    c.quant_dtype = quant_dtype;
    c.bq_omajor = true;
    c.act_quant = 1; // W8A8_TOKEN_SYM
    KernelCall::MatMulDequant(c)
}

fn plain_matmul() -> KernelCall {
    KernelCall::MatMul(MatMulCall {
        a: buf(0),
        b: buf(1),
        output: buf(2),
        m: 2,
        k: 8,
        n: 4,
        dtype: 8,
        b_packed: false,
    })
}

/// `(name, call, expected opcode, expected signature params, expected wire)`.
/// Frozen 2026-07 (v0.7.2). Do not "update" these to make a change pass — a
/// diff here means the content address of an existing op moved.
#[allow(clippy::type_complexity)]
fn frozen() -> Vec<(&'static str, KernelCall, u16, &'static str, &'static str)> {
    vec![
        (
            "mmdq_plain",
            mmdq_plain(),
            108,
            "020000000800000004000000040000000100000002080000000000000000",
            "010000006f0000000000000000000000000040000000000000000100000000000000000000004000000000000000020000000000000000000000400000000000000003000000000000000000000040000000000000000400000000000000000000004000000000000000020000000800000004000000040000000100000002080000000000000000",
        ),
        (
            "mmdq_omajor_i8",
            mmdq_omajor(2),
            116,
            "010000000800000004000000040000000100000002080000000000000000010000",
            "01000000740000000000000000000000000040000000000000000100000000000000000000004000000000000000020000000000000000000000400000000000000003000000000000000000000040000000000000000400000000000000000000004000000000000000010000000800000004000000040000000100000002080000000000000000010100ffffffff00000000000000000000000000000000",
        ),
        (
            "mmdq_omajor_i4",
            mmdq_omajor(10),
            116,
            "01000000080000000400000004000000010000000a080000000000000000010000",
            "0100000074000000000000000000000000004000000000000000010000000000000000000000400000000000000002000000000000000000000040000000000000000300000000000000000000004000000000000000040000000000000000000000400000000000000001000000080000000400000004000000010000000a080000000000000000010100ffffffff00000000000000000000000000000000",
        ),
        (
            "plain_matmul",
            plain_matmul(),
            45,
            "02000000080000000400000008",
            "010000002e000000000000000000000000004000000000000000010000000000000000000000400000000000000002000000000000000000000040000000000000000200000008000000040000000800",
        ),
    ]
}

#[test]
fn op_signature_bytes_are_frozen_for_legacy_calls() {
    for (name, call, opcode, params, _) in frozen() {
        let sig = call.op_signature();
        assert_eq!(sig.opcode, opcode, "{name}: opcode moved — κ re-key");
        assert_eq!(
            hex(sig.params()),
            params,
            "{name}: op_signature params moved — every κ for this op re-keys"
        );
    }
}

#[test]
fn archive_wire_bytes_are_frozen_for_legacy_calls() {
    for (name, call, _, _, wire) in frozen() {
        let got = encode_calls(std::slice::from_ref(&call));
        assert_eq!(
            hex(&got),
            wire,
            "{name}: archive wire encoding moved — existing archives no longer replay"
        );
    }
}

/// Every legacy call must survive `encode → decode → encode` unchanged.
#[test]
fn legacy_calls_round_trip_through_the_wire() {
    for (name, call, _, _, _) in frozen() {
        let bytes = encode_calls(std::slice::from_ref(&call));
        let decoded = hologram_archive::decoder::decode_calls(&bytes)
            .unwrap_or_else(|e| panic!("{name}: decode failed: {e:?}"));
        assert_eq!(decoded.len(), 1, "{name}: call count");
        let re = encode_calls(&decoded);
        assert_eq!(hex(&re), hex(&bytes), "{name}: re-encode differs");
    }
}
