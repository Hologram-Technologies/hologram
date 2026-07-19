//! Smoke-test the prism Blake3 hasher as hologram's canonical
//! `Hasher<32>` selection (spec III.3, wiki ADR-031). The full
//! standard-vector conformance suite lives in `prism-crypto`'s
//! `tests/conformance.rs`; this file checks only the integration
//! surface — hologram-host re-exports `prism::crypto::Blake3Hasher`
//! as `HologramHasher` and the `Hasher<32>` trait surface resolves.

use hologram_types::HologramHasher;
use prism::vocabulary::Hasher;

#[test]
fn fold_byte_associative_with_fold_bytes() {
    let by_byte = HologramHasher::initial()
        .fold_byte(b'a')
        .fold_byte(b'b')
        .fold_byte(b'c')
        .finalize();
    let by_slice = HologramHasher::initial().fold_bytes(b"abc").finalize();
    assert_eq!(by_byte, by_slice);
}

#[test]
fn output_bytes_is_32() {
    assert_eq!(<HologramHasher as Hasher<32>>::OUTPUT_BYTES, 32);
}

#[test]
fn empty_input_is_deterministic() {
    let a = HologramHasher::initial().finalize();
    let b = HologramHasher::initial().finalize();
    assert_eq!(a, b);
}
