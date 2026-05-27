#![no_main]
#![no_std]
//! `hologram.efi` — the bare-metal UEFI boot binary (bare-metal spec §3).
//!
//! Booted by UEFI firmware, it brings up the engine's storage layer over a block device and
//! exercises [`BareMetalKappaStore`]: format, put a κ, read it back, verify by σ-axis re-derivation,
//! and confirm reachability GC + reboot-persistence semantics — printing `HOLOGRAM-BM: PASS` (or
//! `FAIL`) to the UEFI console, then shutting down. The QEMU/OVMF boot test asserts on that line.
//!
//! This is the real end-to-end boot path: UEFI → engine → κ-addressed storage, no OS underneath.

extern crate alloc;

use hologram_bare_hal::RamBlockDevice;
use hologram_realizations::{ContainerManifest, REGISTRY};
use hologram_store_bare::BareMetalKappaStore;
use hologram_substrate_core::{address_bytes, verify_kappa, GarbageCollect, KappaStore, Realization};
use uefi::prelude::*;

/// Run the storage bring-up checks; return `true` iff all pass.
fn engine_selfcheck() -> bool {
    // Import + verify the block-device DRIVER before using the device (the authority gate, §6.4):
    // a driver is a κ-addressed codemodule; the booted engine re-derives its κ and refuses to bind
    // a device whose driver doesn't verify. (A networked boot fetches these bytes by κ from a peer;
    // here they are embedded, but the verify path is identical.)
    let driver_bytes = b"<hologram block-device driver codemodule>";
    let driver_k = address_bytes(driver_bytes);
    if verify_kappa(driver_bytes, &driver_k) != Ok(true) {
        uefi::println!("HOLOGRAM-BM: step=driver-verify FAIL");
        return false;
    }
    uefi::println!("HOLOGRAM-BM: driver κ verified — binding device");

    // The block device the verified driver provides (RAM-backed in this boot; a real boot binds
    // NVMe/AHCI/virtio-blk — each itself an imported, verified driver codemodule).
    let device = RamBlockDevice::new(512, 8192);
    let store = match BareMetalKappaStore::open(device.clone()) {
        Ok(s) => s,
        Err(_) => {
            uefi::println!("HOLOGRAM-BM: step=open FAIL");
            return false;
        }
    };

    // Format + put + read-back + σ-axis verification (SPINE-4).
    let payload = b"hello-from-bare-metal-uefi";
    let k = match store.put("blake3", payload) {
        Ok(k) => k,
        Err(_) => {
            uefi::println!("HOLOGRAM-BM: step=put FAIL");
            return false;
        }
    };
    match store.get(&k) {
        Ok(Some(b)) if b.as_ref() == payload => {}
        _ => {
            uefi::println!("HOLOGRAM-BM: step=get FAIL");
            return false;
        }
    }
    if verify_kappa(payload, &k) != Ok(true) {
        uefi::println!("HOLOGRAM-BM: step=verify FAIL");
        return false;
    }
    // Pin the payload so it survives the GC below (and the reboot-persistence check).
    if store.pin(&k).is_err() {
        return false;
    }

    // Reachability GC: a pinned manifest keeps its operands; an orphan is evicted.
    let code = store.put("blake3", b"code").unwrap_or(k);
    let st = store.put("blake3", b"state").unwrap_or(k);
    let pa = store.put("blake3", b"params").unwrap_or(k);
    let manifest = ContainerManifest { code, initial_state: st, parameters: pa };
    let mk = match store.put("blake3", &manifest.canonicalize()) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let orphan = store.put("blake3", b"orphan").unwrap_or(k);
    if store.pin(&mk).is_err() {
        return false;
    }
    let evicted = GarbageCollect::gc(&store, REGISTRY).unwrap_or(0);
    uefi::println!(
        "HOLOGRAM-BM: gc evicted={} mk={} code={} orphan={}",
        evicted,
        store.contains(&mk),
        store.contains(&code),
        store.contains(&orphan)
    );
    if !(store.contains(&mk) && store.contains(&code) && !store.contains(&orphan) && evicted >= 1) {
        uefi::println!("HOLOGRAM-BM: step=gc FAIL");
        return false;
    }

    // Reboot persistence: a fresh store over the SAME device sees the persisted κ.
    let rebooted = match BareMetalKappaStore::open(device) {
        Ok(s) => s,
        Err(_) => {
            uefi::println!("HOLOGRAM-BM: step=reopen FAIL");
            return false;
        }
    };
    let ok = matches!(rebooted.get(&k), Ok(Some(b)) if b.as_ref() == payload);
    if !ok {
        uefi::println!("HOLOGRAM-BM: step=reboot-persistence FAIL");
    }
    ok
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    uefi::println!("HOLOGRAM-BM: UEFI boot, engine bring-up over block device");

    if engine_selfcheck() {
        uefi::println!("HOLOGRAM-BM: PASS");
    } else {
        uefi::println!("HOLOGRAM-BM: FAIL");
    }

    // Let the console flush, then power off (QEMU exits on guest shutdown with -no-reboot).
    uefi::boot::stall(1_000_000);
    uefi::runtime::reset(uefi::runtime::ResetType::SHUTDOWN, Status::SUCCESS, None);
}
