//! `.holo` v3/v4: the `AppManifest` section + the v2 read-shim (spec `refactor/03`).
//!
//! v3 makes `.holo` the one application container; v4 appends the `inference-model`
//! layer kind to the manifest's closed kind set (the section set is unchanged). The
//! archive layer carries the manifest opaquely (its bytes are an `AppManifest`
//! realization decoded by the app-load layer); a bare tensor archive omits it.
//! Writers emit v4; readers accept `MIN_READ_VERSION..=FORMAT_VERSION` so v2/v3
//! archives still load.

use hologram_archive::{ArchiveError, HoloLoader, HoloWriter, SectionKind, FORMAT_VERSION};

#[test]
fn writer_stamps_version_4() {
    let bytes = HoloWriter::new().finish().unwrap();
    assert_eq!(FORMAT_VERSION, 4);
    // Header: magic[4] || version[2 LE] || …
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 4);
}

#[test]
fn v4_archive_round_trips_an_inference_model_layer() {
    use hologram_space::{address_bytes, AppManifest, Layer, LayerKind, Realization};

    // A v4 model-only app: no primary (non-executable / library artifact), one
    // inference-model layer whose aux tag is the engine identifier and whose
    // entry names the callable service.
    let manifest = AppManifest {
        primary: None,
        requires: address_bytes(b"caps"),
        layers: vec![Layer::inference_model(
            address_bytes(b"r4g1-model"),
            "ai.default",
            "uor-r4",
        )],
        children: vec![],
    };
    manifest.validate().unwrap();

    let mut w = HoloWriter::new();
    w.set_app_manifest(manifest.canonicalize());
    let bytes = w.finish().unwrap();
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 4);

    let plan = HoloLoader::from_bytes(&bytes).unwrap().into_plan().unwrap();
    let decoded = AppManifest::decode(plan.app_manifest().unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.primary, None);
    assert_eq!(decoded.layers.len(), 1);
    assert_eq!(decoded.layers[0].kind, LayerKind::InferenceModel);
    assert_eq!(decoded.layers[0].entry, "ai.default");
    assert_eq!(decoded.layers[0].aux, "uor-r4");
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
fn read_shim_accepts_v2_through_v4_but_rejects_others() {
    // The version gate runs BEFORE footer verification, so mutating the version
    // byte (without re-signing) distinguishes acceptance from rejection by the
    // error kind: a rejected version fails at the gate (UnsupportedVersion); an
    // accepted version gets past it and fails only at the (now-broken) footer
    // (ChecksumMismatch).
    let mut bytes = HoloWriter::new().finish().unwrap();

    // v2 and v3 accepted (read-shim): reach the footer check.
    for v in [2u8, 3] {
        bytes[4] = v;
        bytes[5] = 0;
        assert!(matches!(
            HoloLoader::from_bytes(&bytes),
            Err(ArchiveError::ChecksumMismatch)
        ));
    }

    // v1 rejected at the gate (below MIN_READ_VERSION).
    bytes[4] = 1;
    assert!(matches!(
        HoloLoader::from_bytes(&bytes),
        Err(ArchiveError::UnsupportedVersion(1))
    ));

    // v5 rejected at the gate (above the current version) — a v3 reader rejects
    // a v4 archive the same way, one version lower.
    bytes[4] = 5;
    assert!(matches!(
        HoloLoader::from_bytes(&bytes),
        Err(ArchiveError::UnsupportedVersion(5))
    ));
}
