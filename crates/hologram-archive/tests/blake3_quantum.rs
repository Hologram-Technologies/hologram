//! BLAKE3 checksum + header flag conformance tests.
//!
//! Verifies BLAKE3 checksum support and the header flag bits. The Witt
//! level is carried by the
//! [`ConformanceShapeSection`](hologram_archive::section::conformance_shape::ConformanceShapeSection)
//! rather than the header itself.

use hologram_archive::checksum;
use hologram_archive::format::header::*;
use hologram_archive::format::*;

// ── BLAKE3 checksums ─────────────────────────────────────────────────────

#[test]
fn blake3_deterministic() {
    let data = b"test data for blake3";
    let h1 = checksum::blake3_u32(data);
    let h2 = checksum::blake3_u32(data);
    assert_eq!(h1, h2, "blake3 must be deterministic");
    assert_ne!(h1, 0, "blake3 should not be 0 for non-empty data");
}

#[test]
fn blake3_u32_differs_from_full() {
    let data = b"test data";
    let truncated = checksum::blake3_u32(data);
    let full = checksum::checksum(data);
    // The u32 should be the first 4 bytes of the full hash
    let expected = u32::from_le_bytes([full[0], full[1], full[2], full[3]]);
    assert_eq!(truncated, expected);
}

#[test]
fn blake3_full_32_bytes() {
    let data = b"full hash test";
    let hash = checksum::checksum(data);
    assert_eq!(hash.len(), 32);
    // Non-zero
    assert!(hash.iter().any(|&b| b != 0));
}

#[test]
fn blake3_verify() {
    let data = b"verify me";
    let expected = checksum::blake3_u32(data);
    assert!(checksum::verify_blake3(data, expected));
    assert!(!checksum::verify_blake3(data, expected ^ 1)); // corrupt
}

// ── BLAKE3 flag in header ────────────────────────────────────────────────

fn empty_header() -> HoloHeader {
    HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: [0u8; 32],
        weights_checksum: [0u8; 32],
        unit_address: [0u8; 32],
        section_count: 0,
        flags: 0,
    }
}

#[test]
fn blake3_flag_default_off() {
    let h = empty_header();
    assert!(!h.uses_blake3());
}

#[test]
fn blake3_flag_set() {
    let mut h = empty_header();
    h.set_blake3();
    assert!(h.uses_blake3());
}

#[test]
fn blake3_flag_survives_bytemuck_roundtrip() {
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 4096,
        graph_size: 512,
        weights_offset: 8192,
        weights_size: 1024,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 12288,
        graph_checksum: [0u8; 32],
        weights_checksum: [0u8; 32],
        unit_address: [0u8; 32],
        section_count: 0,
        flags: 0,
    };
    h.set_blake3();

    let bytes = h.as_bytes();
    let h2 = HoloHeader::from_bytes(bytes).unwrap();
    assert!(h2.uses_blake3());
}
