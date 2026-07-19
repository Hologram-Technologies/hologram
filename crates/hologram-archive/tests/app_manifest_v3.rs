//! `.holo` v3: the `AppManifest` section + the v2 read-shim (spec `refactor/03`).
//!
//! v3 makes `.holo` the one application container. The archive layer carries the
//! manifest opaquely (its bytes are an `AppManifest` realization decoded by the
//! app-load layer); a bare tensor archive omits it. Writers emit v3; readers
//! accept `MIN_READ_VERSION..=FORMAT_VERSION` so a v2 tensor archive still loads.

use hologram_archive::{ArchiveError, HoloLoader, HoloWriter, SectionKind, FORMAT_VERSION};

#[test]
fn writer_stamps_version_3() {
    let bytes = HoloWriter::new().finish().unwrap();
    assert_eq!(FORMAT_VERSION, 3);
    // Header: magic[4] || version[2 LE] || …
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 3);
}

#[test]
fn app_manifest_section_round_trips_opaquely() {
    // The archive treats the manifest as opaque bytes — here a stand-in for an
    // AppManifest realization's canonical form.
    let manifest = b"IRI:app-manifest\x00...operand-embedding bytes...".to_vec();
    let mut w = HoloWriter::new();
    w.set_app_manifest(manifest.clone());
    let bytes = w.finish().unwrap();

    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    assert_eq!(plan.app_manifest(), Some(manifest.as_slice()));
    assert_eq!(
        plan.section(SectionKind::AppManifest).unwrap(),
        &manifest[..]
    );
}

#[test]
fn bare_tensor_archive_has_no_manifest() {
    // A writer that never sets a manifest emits no AppManifest section (the
    // degenerate tensor archive; the compiler will later default to a
    // single-tensor-plan manifest).
    let bytes = HoloWriter::new().finish().unwrap();
    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    assert_eq!(plan.app_manifest(), None);
    assert!(plan.section(SectionKind::AppManifest).is_err());
}

#[test]
fn read_shim_accepts_v2_and_v3_but_rejects_others() {
    // The version gate runs BEFORE footer verification, so mutating the version
    // byte (without re-signing) distinguishes acceptance from rejection by the
    // error kind: a rejected version fails at the gate (UnsupportedVersion); an
    // accepted version gets past it and fails only at the (now-broken) footer
    // (ChecksumMismatch).
    let mut bytes = HoloWriter::new().finish().unwrap();

    // v2 accepted (read-shim): reaches the footer check.
    bytes[4] = 2;
    bytes[5] = 0;
    assert!(matches!(
        HoloLoader::from_bytes(&bytes),
        Err(ArchiveError::ChecksumMismatch)
    ));

    // v1 rejected at the gate (below MIN_READ_VERSION).
    bytes[4] = 1;
    assert!(matches!(
        HoloLoader::from_bytes(&bytes),
        Err(ArchiveError::UnsupportedVersion(1))
    ));

    // v4 rejected at the gate (above the current version).
    bytes[4] = 4;
    assert!(matches!(
        HoloLoader::from_bytes(&bytes),
        Err(ArchiveError::UnsupportedVersion(4))
    ));
}
