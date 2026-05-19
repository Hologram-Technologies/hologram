//! Spec X.1 footer-verification tests.

use hologram_archive::{ArchiveError, HoloLoader, HoloWriter};

fn build_minimal_archive() -> Vec<u8> {
    HoloWriter::new().finish().unwrap()
}

#[test]
fn intact_archive_loads() {
    let bytes = build_minimal_archive();
    let _ = HoloLoader::from_bytes(&bytes).expect("intact archive must verify");
}

#[test]
fn tampered_body_fails_checksum() {
    let mut bytes = build_minimal_archive();
    // Flip a byte in the section table (after the header, before the footer).
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    match HoloLoader::from_bytes(&bytes) {
        Err(ArchiveError::ChecksumMismatch) => {}
        other => panic!("expected ChecksumMismatch, got {:?}", other.err()),
    }
}

#[test]
fn tampered_footer_fails() {
    let mut bytes = build_minimal_archive();
    let len = bytes.len();
    bytes[len - 1] ^= 0x01;
    match HoloLoader::from_bytes(&bytes) {
        Err(ArchiveError::ChecksumMismatch) => {}
        other => panic!("expected ChecksumMismatch, got {:?}", other.err()),
    }
}

#[test]
fn unchecked_loader_skips_footer() {
    let mut bytes = build_minimal_archive();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    // Should still parse despite tampering when verification is skipped.
    let _ = HoloLoader::from_bytes_unchecked(&bytes).unwrap();
}
