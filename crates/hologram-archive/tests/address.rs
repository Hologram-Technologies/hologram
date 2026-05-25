//! UOR-ADDR content-addressing + E₈ composition integration (ADR-061).

use std::collections::HashMap;

use hologram_archive::address::{address_ring, compose_model, KappaLabel};

/// Amendment-43 canonical bytes for a ring element: `[witt_level] ||
/// le_coefficient[witt_level + 1]`.
fn ring_bytes(witt_level: u8, coefficient: u64) -> Vec<u8> {
    let coeff_bytes = (witt_level + 1) as usize;
    let mut out = Vec::with_capacity(1 + coeff_bytes);
    out.push(witt_level);
    out.extend_from_slice(&coefficient.to_le_bytes()[..coeff_bytes]);
    out
}

#[test]
fn ring_element_addresses_to_blake3_kappa_label() {
    let a = address_ring(&ring_bytes(1, 0x0102)).expect("well-formed ring element");
    // hologram's canonical σ-axis is blake3.
    assert_eq!(a.address.sigma_axis(), Some("blake3"));
    // The witness re-certifies through prism replay without re-hashing.
    assert_eq!(a.witness.verify().expect("replay certifies"), a.address);
}

#[test]
fn equivalent_content_collapses_to_one_label() {
    // Same canonical bytes minted twice → identical κ-label → one cache bucket.
    let mut cache: HashMap<KappaLabel<71>, u32> = HashMap::new();
    for _ in 0..3 {
        let label = address_ring(&ring_bytes(2, 0x0010_2030)).unwrap().address;
        *cache.entry(label).or_default() += 1;
    }
    assert_eq!(cache.len(), 1, "structural equivalence ⇒ one κ-label");
}

#[test]
fn compose_model_is_order_independent() {
    let a = address_ring(&ring_bytes(0, 0x07)).unwrap().address;
    let b = address_ring(&ring_bytes(1, 0x1234)).unwrap().address;

    let ab = compose_model(&[a, b]).expect("compose a∘b");
    let ba = compose_model(&[b, a]).expect("compose b∘a");
    assert_eq!(ab, ba, "CS-G2 commutativity is structural");
    // The composite identity is distinct from either part.
    assert_ne!(ab, a);
    assert_ne!(ab, b);
}

#[test]
fn single_part_model_is_the_part() {
    let a = address_ring(&ring_bytes(3, 0xDEAD_BEEF)).unwrap().address;
    assert_eq!(compose_model(&[a]).unwrap(), a);
}

#[test]
fn empty_model_is_rejected() {
    assert!(compose_model(&[]).is_err());
}
