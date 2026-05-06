//! Spec XII.3: HologramHasher produces BLAKE3-equivalent output.

use uor_foundation::enforcement::Hasher;
use hologram_host::HologramHasher;

#[test]
fn blake3_known_vector_empty() {
    let h = HologramHasher::initial();
    let out = h.finalize();
    let expected: [u8; 32] = blake3::hash(&[]).into();
    assert_eq!(out, expected);
}

#[test]
fn blake3_known_vector_abc() {
    let h = HologramHasher::initial().fold_bytes(b"abc");
    let out = h.finalize();
    let expected: [u8; 32] = blake3::hash(b"abc").into();
    assert_eq!(out, expected);
}

#[test]
fn fold_byte_is_associative_with_fold_bytes() {
    let by_byte = HologramHasher::initial()
        .fold_byte(b'a').fold_byte(b'b').fold_byte(b'c').finalize();
    let by_slice = HologramHasher::initial().fold_bytes(b"abc").finalize();
    assert_eq!(by_byte, by_slice);
}

#[test]
fn output_bytes_is_32() {
    assert_eq!(<HologramHasher as Hasher<32>>::OUTPUT_BYTES, 32);
}
