//! Fat/thin `.holo` conversion over `resolve_closure` (spec `refactor/03` §Fat and thin).
//!
//! `Client::fat` embeds every store-resolvable layer/closure κ as a content blob (self-contained);
//! `Client::thin` drops the blobs (manifest + certificates only), resolving through the store at
//! load. The invariant under test: **the manifest κ — the app's identity — is unchanged either way**
//! (fat↔thin is packaging, never identity), and `is_fat` reflects whether the archive is
//! self-contained.

use hologram::{Client, Holo};
use hologram_archive::HoloWriter;
use hologram_space::{address_bytes, AppManifest, KappaStore, Layer, Realization};
use hologram_spike_sp3::SpikeSpace;

/// A thin `.holo` (manifest only) whose layers' content has been provisioned into the client store.
fn provisioned_app(client: &Client<SpikeSpace>) -> Holo {
    let store = client.store();
    let wasm = store.put("blake3", b"fat-thin-wasm-layer").unwrap();
    let plan = store.put("blake3", b"fat-thin-tensor-plan").unwrap();
    let caps = store.put("blake3", b"fat-thin-caps").unwrap();
    let manifest = AppManifest {
        primary: Some(0),
        requires: caps,
        layers: vec![Layer::wasm(wasm, "_start"), Layer::tensor(plan, "sess")],
        children: vec![],
    };
    let mut w = HoloWriter::new();
    w.set_app_manifest(manifest.canonicalize());
    Holo::from_bytes(w.finish().unwrap())
}

#[test]
fn fat_thin_round_trip_preserves_the_manifest_kappa() {
    let client = Client::new(SpikeSpace::new());
    let thin = provisioned_app(&client);

    // A manifest-only archive is thin — its layer content is not embedded.
    assert!(!client.is_fat(&thin), "a manifest-only archive is thin");

    // Fat embeds the store-resolvable closure — the archive becomes self-contained.
    let fat = client.fat(&thin).unwrap();
    assert!(
        client.is_fat(&fat),
        "fat embeds the closure; it resolves with no store"
    );

    // Thinning the fat archive drops the blobs again.
    let rethin = client.thin(&fat).unwrap();
    assert!(
        !client.is_fat(&rethin),
        "thinning drops the embedded content"
    );

    // Identity is invariant across fat↔thin — the app κ never changes (packaging, not identity).
    let id = |h: &Holo| client.inspect(h).unwrap().app;
    assert_eq!(id(&thin), id(&fat), "fat must not change the app κ");
    assert_eq!(id(&thin), id(&rethin), "thin must not change the app κ");
}

#[test]
fn is_fat_requires_every_layer_embedded() {
    // A fat archive whose store is missing a layer is not fully self-contained. Build the manifest
    // against a κ whose content is never provisioned, then fat: the missing layer cannot be embedded.
    let client = Client::new(SpikeSpace::new());
    let store = client.store();
    let present = store.put("blake3", b"present-layer").unwrap();
    let absent = address_bytes(b"never-provisioned-layer");
    let caps = store.put("blake3", b"caps2").unwrap();
    let manifest = AppManifest {
        primary: Some(0),
        requires: caps,
        layers: vec![
            Layer::wasm(present, "_start"),
            Layer::tensor(absent, "sess"),
        ],
        children: vec![],
    };
    let mut w = HoloWriter::new();
    w.set_app_manifest(manifest.canonicalize());
    let holo = Holo::from_bytes(w.finish().unwrap());

    let fat = client.fat(&holo).unwrap();
    // The absent layer's bytes were never in the store, so the fat archive cannot be self-contained.
    assert!(
        !client.is_fat(&fat),
        "a fat archive missing a layer is not complete"
    );
}
