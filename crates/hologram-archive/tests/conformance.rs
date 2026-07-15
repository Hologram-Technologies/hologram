//! **Addressing conformance — external V&V (class AS).**
//!
//! These tests validate hologram's content-addressing against *external
//! authorities*, never against hologram itself (the discipline uor-addr
//! and prism-btc establish: SHA vs FIPS-180-4, merkle vs rust-bitcoin):
//!
//! * **AS-1 σ-axis correctness** — hologram's σ-axis (`HologramHasher` =
//!   `prism::crypto::Blake3Hasher`) must equal the upstream BLAKE3
//!   reference implementation (`blake3` crate, by the algorithm's authors)
//!   byte-for-byte, over inputs spanning chunk/subtree boundaries.
//! * **AS-2 κ-label realization** — `address_bytes` must emit exactly
//!   `blake3:<lowercase-hex of the reference digest>` (71 bytes), the
//!   canonical κ-label wire form.
//! * **AS-3 determinism** — equal content ⇒ equal κ-label.
//! * **AS-4 collision-sensitivity** — a single-bit input change changes
//!   the κ-label.
//! * **AS-5 streaming equivalence** — incremental `fold_bytes` equals a
//!   one-shot hash (ADR-060 streaming must not change the address).

use hologram_archive::{address_bytes, derive_label_witnessed};
use hologram_types::HologramHasher;
use prism::vocabulary::Hasher;

/// Inputs chosen to cross BLAKE3's 1024-byte chunk and subtree-merge
/// boundaries, where a non-conformant implementation would diverge.
fn corpus() -> Vec<Vec<u8>> {
    let lens = [
        0usize, 1, 2, 63, 64, 65, 127, 1023, 1024, 1025, 2048, 4096, 100_000,
    ];
    lens.iter()
        .map(|&n| (0..n).map(|i| (i % 251) as u8).collect::<Vec<u8>>())
        .collect()
}

fn holo_digest(bytes: &[u8]) -> [u8; 32] {
    HologramHasher::initial().fold_bytes(bytes).finalize()
}

#[test]
fn as1_sigma_axis_equals_blake3_reference() {
    for input in corpus() {
        let expected = *blake3::hash(&input).as_bytes();
        let got = holo_digest(&input);
        assert_eq!(
            got,
            expected,
            "σ-axis diverged from BLAKE3 reference at len {}",
            input.len()
        );
    }
}

#[test]
fn as2_address_bytes_is_canonical_kappa_label() {
    for input in corpus() {
        let reference = blake3::hash(&input);
        let expected = format!("blake3:{}", reference.to_hex());
        let label = address_bytes(&input);
        assert_eq!(
            label.as_str(),
            expected,
            "κ-label wire form at len {}",
            input.len()
        );
        assert_eq!(label.as_str().len(), 71);
    }
}

#[test]
fn as3_addressing_is_deterministic() {
    let a = address_bytes(b"hologram content addressing");
    let b = address_bytes(b"hologram content addressing");
    assert_eq!(a, b);
}

#[test]
fn as4_single_bit_change_changes_label() {
    let base = [0u8; 64];
    let mut flipped = base;
    flipped[40] ^= 0x01;
    assert_ne!(address_bytes(&base), address_bytes(&flipped));
}

// ─── CA-3: witnessed derivation addresses (composition, à la prism-btc) ──

#[test]
fn ca3_derived_address_carries_replayable_witness() {
    let a = address_bytes(b"operand-A");
    let b = address_bytes(b"operand-B");
    let outcome = derive_label_witnessed(45 /*matmul*/, &[1, 2, 3], &[a, b]).expect("compose");
    // TC-05: the witness re-certifies the derivation to the same κ-label
    // without re-running the σ-axis (QS-05 fingerprint equivalence).
    assert_eq!(outcome.witness.verify().expect("replay"), outcome.address);
    // It is a well-formed blake3 κ-label.
    assert!(outcome.address.as_str().starts_with("blake3:"));
    assert_eq!(outcome.address.as_str().len(), 71);
}

#[test]
fn ca3_derivation_is_deterministic_and_order_sensitive() {
    let a = address_bytes(b"operand-A");
    let b = address_bytes(b"operand-B");
    let ab1 = derive_label_witnessed(45, &[7], &[a, b]).unwrap().address;
    let ab2 = derive_label_witnessed(45, &[7], &[a, b]).unwrap().address;
    assert_eq!(ab1, ab2, "derivation must be deterministic");
    // Order-sensitivity is *proven*, not assumed: matmul(A,B) ≠ matmul(B,A).
    let ba = derive_label_witnessed(45, &[7], &[b, a]).unwrap().address;
    assert_ne!(
        ab1, ba,
        "ordered composition must distinguish operand order"
    );
    // Opcode and params also separate the derivation.
    assert_ne!(
        ab1,
        derive_label_witnessed(46, &[7], &[a, b]).unwrap().address
    );
    assert_ne!(
        ab1,
        derive_label_witnessed(45, &[8], &[a, b]).unwrap().address
    );
}

#[test]
fn as5_streaming_equals_one_shot() {
    // ADR-060 streaming: folding in chunks must equal a single fold.
    let data: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
    let one_shot = HologramHasher::initial().fold_bytes(&data).finalize();
    let mut h = HologramHasher::initial();
    for chunk in data.chunks(257) {
        h = h.fold_bytes(chunk);
    }
    let streamed = h.finalize();
    assert_eq!(one_shot, streamed);
    // and both equal the reference
    assert_eq!(one_shot, *blake3::hash(&data).as_bytes());
}

// ─── OV: no fixed byte ceiling — operations beyond u32 don't overflow ──
//
// ADR-060 removed fixed byte ceilings. hologram previously stored byte
// offsets/lengths and element counts as u32 (a silent 4 GiB / 4.29 B-element
// cap, with saturating/truncating arithmetic). They are now u64; this proves
// values beyond u32::MAX survive the archive codec round-trip intact — i.e.
// large-tensor operations don't overflow their size representation.
#[test]
fn ov_codec_roundtrips_beyond_u32_without_truncation() {
    use hologram_archive::decoder::decode_calls;
    use hologram_archive::kernel_codec::encode_calls;
    use hologram_backend::{BufferRef, KernelCall, MatMulCall, UnaryCall};

    let big_len: u64 = 5_000_000_000; // > 4 GiB byte length/offset
    let big_count: u64 = u32::MAX as u64 + 1_000; // > 4.29 B elements

    let calls = vec![
        KernelCall::Relu(UnaryCall {
            input: BufferRef {
                slot: 0,
                offset: big_len,
                length: big_len,
            },
            output: BufferRef {
                slot: 1,
                offset: 0,
                length: big_len,
            },
            element_count: big_count,
            witt_bits: 32,
            dtype: 8,
        }),
        KernelCall::MatMul(MatMulCall {
            a: BufferRef {
                slot: 0,
                offset: 0,
                length: big_len,
            },
            b: BufferRef {
                slot: 1,
                offset: big_len,
                length: big_len,
            },
            output: BufferRef {
                slot: 2,
                offset: 2 * big_len,
                length: big_len,
            },
            m: 70_000,
            k: 70_000,
            n: 70_000,
            dtype: 8,
            b_packed: false,
        }),
    ];
    let decoded = decode_calls(&encode_calls(&calls)).expect("decode");
    assert_eq!(decoded.len(), 2);

    let KernelCall::Relu(r) = &decoded[0] else {
        panic!("variant")
    };
    assert_eq!(
        r.element_count, big_count,
        "element_count truncated past u32"
    );
    assert_eq!(r.input.offset, big_len, "offset truncated past u32");
    assert_eq!(r.input.length, big_len, "length truncated past u32");

    let KernelCall::MatMul(m) = &decoded[1] else {
        panic!("variant")
    };
    assert_eq!(
        m.output.offset,
        2 * big_len,
        "matmul output offset truncated"
    );
    assert_eq!(m.output.length, big_len);
}
