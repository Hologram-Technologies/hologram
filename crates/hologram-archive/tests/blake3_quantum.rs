//! blake3 + quantum_index conformance tests.
//!
//! Verifies blake3 checksum support and quantum_index in header flags.

use hologram_archive::checksum;
use hologram_archive::format::header::*;
use hologram_archive::format::*;

// ── blake3 checksums ─────────────────────────────────────────────────────

#[test]
fn blake3_deterministic() {
    let data = b"test data for blake3";
    let h1 = checksum::blake3_u32(data);
    let h2 = checksum::blake3_u32(data);
    assert_eq!(h1, h2, "blake3 must be deterministic");
    assert_ne!(h1, 0, "blake3 should not be 0 for non-empty data");
}

#[test]
fn blake3_differs_from_crc32() {
    let data = b"test data";
    let blake = checksum::blake3_u32(data);
    let crc = checksum::crc32(data);
    // While collisions are theoretically possible, for this specific input they differ
    assert_ne!(
        blake, crc,
        "blake3 and crc32 should produce different values"
    );
}

#[test]
fn blake3_full_32_bytes() {
    let data = b"full hash test";
    let hash = checksum::blake3_full(data);
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

// ── quantum_index in header ──────────────────────────────────────────────

#[test]
fn quantum_index_default_zero() {
    let h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    assert_eq!(h.quantum_index(), 0);
}

#[test]
fn quantum_index_set_q3() {
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    h.set_quantum_index(3);
    assert_eq!(h.quantum_index(), 3);
    // Other flags should not be affected
    assert!(!h.is_graph_compressed());
    assert!(!h.uses_blake3());
}

#[test]
fn quantum_index_set_q7() {
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    h.set_quantum_index(7);
    assert_eq!(h.quantum_index(), 7);
}

#[test]
fn quantum_index_coexists_with_blake3_flag() {
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    h.set_quantum_index(3);
    h.set_blake3();
    assert_eq!(h.quantum_index(), 3);
    assert!(h.uses_blake3());
}

#[test]
fn quantum_index_survives_bytemuck_roundtrip() {
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
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    h.set_quantum_index(7);
    h.set_blake3();

    let bytes = h.as_bytes();
    let h2 = HoloHeader::from_bytes(bytes).unwrap();
    assert_eq!(h2.quantum_index(), 7);
    assert!(h2.uses_blake3());
}

// ── blake3 flag ──────────────────────────────────────────────────────────

#[test]
fn blake3_flag_default_off() {
    let h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    assert!(!h.uses_blake3());
}

#[test]
fn blake3_flag_set() {
    let mut h = HoloHeader {
        magic: HOLO_MAGIC,
        version: FORMAT_VERSION,
        graph_offset: 0,
        graph_size: 0,
        weights_offset: 0,
        weights_size: 0,
        section_table_offset: 0,
        section_table_size: 0,
        total_size: 0,
        graph_checksum: 0,
        weights_checksum: 0,
        section_count: 0,
        flags: 0,
    };
    h.set_blake3();
    assert!(h.uses_blake3());
}
