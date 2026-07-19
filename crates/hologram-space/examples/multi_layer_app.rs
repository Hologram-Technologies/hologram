//! **Exit-criteria demo — P4** (spec 06 §P4): a multi-layer `.holo` v3 application.
//!
//! A wasm code-module layer + a tensor-plan layer + a rootfs-image layer + a view layer, assembled
//! into ONE `AppManifest`, validated, and round-tripped. There is one format: the manifest embeds
//! every layer κ as an operand, so `references()` recovers the whole application's reachability
//! closure — migrating the app between peers is exactly `resolve_closure(app κ)`.
//!
//! Run: `cargo run -p hologram-space --example multi_layer_app`

use hologram_space::{address_bytes, AppManifest, Layer, LayerKind, Realization};

fn main() {
    // Four layers, each addressed by the κ of its content. Dedup spans layers (Law L3): a model
    // shared by two layers is stored once.
    let layers = vec![
        Layer::wasm(address_bytes(b"wasm-codemodule-bytes"), "_start"),
        Layer::tensor(address_bytes(b"tensor-plan-bytes"), "decode"),
        Layer::rootfs(address_bytes(b"rootfs-image-bytes"), "boot", "riscv64"),
        Layer::view(address_bytes(b"view-bundle-bytes"), "webgpu"),
    ];
    let manifest = AppManifest {
        primary: Some(0), // the wasm layer's exit code IS the application's exit code
        requires: address_bytes(b"required-capability-set"),
        layers,
        children: Vec::new(),
    };

    // Validated at load, before any layer boots (the execution invariants of the format).
    manifest
        .validate()
        .expect("a well-formed multi-layer app validates");
    println!(
        "built a {}-layer .holo v3 app — primary = layer {:?}",
        manifest.layers.len(),
        manifest.primary
    );
    for (i, layer) in manifest.layers.iter().enumerate() {
        println!(
            "  layer {i}: {:?}  entry={:?}  aux={:?}",
            layer.kind, layer.entry, layer.aux
        );
    }

    // One format: canonicalize → decode recovers every layer.
    let bytes = manifest.canonicalize();
    let decoded = AppManifest::decode(&bytes).expect("the canonical form round-trips");
    assert_eq!(decoded.layers.len(), 4);
    assert_eq!(decoded.layers[0].kind, LayerKind::WasmCodemodule);
    assert_eq!(
        decoded.layers[2].aux, "riscv64",
        "rootfs layer keeps its arch tag"
    );

    // `references()` recovers the app's whole content graph — the migration closure.
    let refs = <AppManifest as Realization>::references(&bytes).expect("reachability closure");
    assert!(refs.contains(&address_bytes(b"tensor-plan-bytes")));
    assert!(refs.contains(&address_bytes(b"required-capability-set")));
    println!(
        "references() recovered {} operand κs — the migration closure",
        refs.len()
    );

    println!("\nP4 multi-layer .holo demo: OK");
}
