//! Drivers (and container bodies, and the engine itself) are **codemodule κ-labels** — addressed
//! via uor-addr's CCMAS realization, not hand-authored by the substrate. A driver is a
//! `CodeModuleValue` AST → a blake3 κ-label → referenced by a manifest and loaded through the same
//! store/fetch/verify path as any container code.

use hologram_realizations::ContainerManifest;
use hologram_space::Realization;
use uor_addr::codemodule::{address_blake3, CodeModuleValue};

/// Build a tiny "block-device driver" code module AST and address it (CCMAS → blake3 κ).
fn driver_kappa(name: &str) -> hologram_space::KappaLabel71 {
    // fn read(...) and fn write(...) — the BlockDevice interface, expressed as code.
    let body = CodeModuleValue::atom("0");
    let ret = CodeModuleValue::atom("DeviceError");
    let read = CodeModuleValue::function("read", &[], &ret, &body);
    let write = CodeModuleValue::function("write", &[], &ret, &body);
    let module = CodeModuleValue::module(name, &[read, write]);
    address_blake3(module.tagged_bytes())
        .expect("κ-label")
        .address
}

#[test]
fn a_driver_is_a_codemodule_kappa_addressed_not_authored() {
    let nvme = driver_kappa("nvme-driver");
    // Deterministic: the same driver AST always content-addresses to the same κ.
    assert_eq!(nvme, driver_kappa("nvme-driver"));
    // Distinct code ⇒ distinct κ (a different driver is a different artifact).
    assert_ne!(nvme, driver_kappa("e1000-driver"));
    // It's a blake3 κ-label — hologram's native width — usable anywhere a code-κ is.
    assert_eq!(nvme.sigma_axis(), Some("blake3"));
}

#[test]
fn driver_loads_through_the_same_manifest_machinery_as_a_container() {
    // A driver "container": its code operand is the codemodule κ; state/params are leaves.
    let code = driver_kappa("virtio-blk-driver");
    let state = hologram_space::address_bytes(b"driver-config");
    let params = hologram_space::address_bytes(b"{}");
    let manifest = ContainerManifest {
        code,
        initial_state: state,
        parameters: params,
    };

    // The driver's code-κ is recovered by the manifest's references() inverse projection (SPINE-3) —
    // the engine fetches + verifies + loads it exactly like container Wasm. No bespoke driver code.
    let refs = ContainerManifest::references(&manifest.canonicalize()).unwrap();
    assert_eq!(
        refs[0], code,
        "the manifest references the driver codemodule κ"
    );
}
