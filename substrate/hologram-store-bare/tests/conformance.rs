//! The bare-metal block-device store runs the **same TCK** as mem/redb (integration), plus
//! format/reachability/**reboot persistence** end-to-end over a RAM block device.

use hologram_bare_hal::RamBlockDevice;
use hologram_realizations::{ContainerManifest, REGISTRY};
use hologram_store_bare::BareMetalKappaStore;
use hologram_substrate_core::{KappaStore, Realization};
use hologram_substrate_tck::store_battery;

fn dev() -> RamBlockDevice {
    RamBlockDevice::new(512, 8192) // 4 MiB RAM disk
}

#[test]
fn bare_passes_the_kappastore_tck() {
    let store = BareMetalKappaStore::open(dev()).unwrap();
    store_battery(&store);
}

#[test]
fn bare_reachability_gc() {
    let store = BareMetalKappaStore::open(dev()).unwrap();
    let code = store.put("blake3", b"wasm").unwrap();
    let st = store.put("blake3", b"state").unwrap();
    let pa = store.put("blake3", b"params").unwrap();
    let mk = store.put("blake3", &ContainerManifest { code, initial_state: st, parameters: pa }.canonicalize()).unwrap();
    let orphan = store.put("blake3", b"orphan").unwrap();
    store.pin(&mk).unwrap();
    assert_eq!(store.gc(REGISTRY).unwrap(), 1);
    assert!(store.contains(&mk) && store.contains(&code));
    assert!(!store.contains(&orphan));
}

#[test]
fn bare_persists_across_reboot() {
    // Format + write on one store; "reboot" = open a fresh store on the SAME device → data survives.
    let device = dev();
    let (k, pinned) = {
        let store = BareMetalKappaStore::open(device.clone()).unwrap();
        let k = store.put("blake3", b"durable-on-disk").unwrap();
        store.pin(&k).unwrap();
        (k, k)
    };
    // Fresh engine instance over the same block device — reads the persisted on-disk image.
    let rebooted = BareMetalKappaStore::open(device).unwrap();
    assert_eq!(rebooted.get(&k).unwrap().unwrap().as_ref(), b"durable-on-disk");
    assert!(rebooted.pinned_roots().contains(&pinned), "pinned roots survive reboot");
}
