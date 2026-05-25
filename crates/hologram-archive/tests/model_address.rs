//! UOR-streaming model addressing for hologram-ai (gated on `model-formats`).
//!
//! Exercises the gguf realization re-exported through `hologram-archive`:
//! a GGUF model is content-addressed to a κ-label whose tensor-data leaves
//! are streamed through the σ-axis with bounded resident memory (ADR-060),
//! so model size never bounds the addressing memory. The resulting κ-label
//! composes with hologram's other part labels under the E₈ operations.
#![cfg(feature = "model-formats")]

use hologram_archive::address::{compose_model, gguf};

// GGUF v3 wire constants.
const GGUF_MAGIC: u32 = 0x4655_4747;
const T_UINT32: u32 = 4;
const GGML_F32: u32 = 0;
const ALIGN: u64 = 32;

fn align_up(o: usize, a: usize) -> usize {
    o.div_ceil(a) * a
}

/// Serialize a minimal but well-formed GGUF v3 file with a `general.alignment`
/// KV and one f32 tensor (whose data bytes are bound into the κ-label).
fn build_gguf(tensor_name: &str, tensor: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&1u64.to_le_bytes()); // kv_count

    // KV: general.alignment = 32 (UINT32).
    let key = b"general.alignment";
    out.extend_from_slice(&(key.len() as u64).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&T_UINT32.to_le_bytes());
    out.extend_from_slice(&(ALIGN as u32).to_le_bytes());

    // Tensor info: name, n_dims, dims, ggml_type, offset.
    let name = tensor_name.as_bytes();
    out.extend_from_slice(&(name.len() as u64).to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    out.extend_from_slice(&(tensor.len() as u64).to_le_bytes());
    out.extend_from_slice(&GGML_F32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // data offset

    // Aligned tensor data section.
    let data_start = align_up(out.len(), ALIGN as usize);
    out.resize(data_start, 0);
    for v in tensor {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn is_kappa_71(s: &str) -> bool {
    s.len() == 71
        && s.starts_with("blake3:")
        && s[7..]
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[test]
fn gguf_model_addresses_to_verifiable_kappa_label() {
    let bytes = build_gguf("token_embd.weight", &[1.0, 2.0, 3.0, 4.0]);
    // BLAKE3 σ-axis to match hologram's canonical axis (and `compose_model`).
    let outcome = gguf::address_blake3(&bytes).expect("well-formed gguf");
    assert!(is_kappa_71(outcome.address.as_str()));
    // The witness re-certifies the streamed derivation without re-hashing.
    assert_eq!(outcome.witness.verify().expect("replay"), outcome.address);
}

#[test]
fn distinct_weights_yield_distinct_labels() {
    let a = gguf::address_blake3(&build_gguf("w", &[1.0, 2.0]))
        .unwrap()
        .address;
    let b = gguf::address_blake3(&build_gguf("w", &[1.0, 3.0]))
        .unwrap()
        .address;
    assert_ne!(a, b, "different tensor bytes ⇒ different κ-label");
}

#[test]
fn gguf_label_composes_with_e8_operations() {
    // A composite identity over two model shards (decomposition → composition).
    let shard0 = gguf::address_blake3(&build_gguf("shard0", &[1.0, 2.0]))
        .unwrap()
        .address;
    let shard1 = gguf::address_blake3(&build_gguf("shard1", &[3.0, 4.0]))
        .unwrap()
        .address;
    let model = compose_model(&[shard0, shard1]).expect("composes");
    let model_rev = compose_model(&[shard1, shard0]).expect("composes");
    assert_eq!(model, model_rev, "CS-G2 composition is order-independent");
}
