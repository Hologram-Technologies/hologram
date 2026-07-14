//! B1 / G-C1 witness — **reboot-monotonic ordering across reboots**.
//!
//! UorTime is "since boot" and resets across reboots; bare-metal §4.5 / §5.5 needs a way to order
//! runtime-state copies written on *different boots*. The store's header now persists a
//! `reboot_epoch` bumped on every successful `open`; the `RuntimeStateRegion` realization carries
//! the pair `(reboot_epoch, generation)`, which is a total order on persisted copies.

use hologram_realizations::RuntimeStateRegion;
use hologram_space::RamBlockDevice;
use hologram_space::{address_bytes, KappaStore, Realization};
use hologram_store_bare::BareMetalKappaStore;

/// On a brand-new device the first `open` is reboot epoch 1.
#[test]
fn b1_fresh_device_starts_at_epoch_one() {
    let dev = RamBlockDevice::new(512, 4096);
    let store = BareMetalKappaStore::open(dev).unwrap();
    assert_eq!(store.reboot_epoch(), 1);
}

/// Re-opening a previously-flushed device bumps the epoch. The pair `(reboot_epoch, gen)` gives
/// a total order over all writes ever made (across reboots), which is what G-C1 requires.
#[test]
fn b1_reboot_epoch_bumps_across_open() {
    // `RamBlockDevice::clone` shares backing bytes — the canonical bare-metal reboot fixture.
    let dev = RamBlockDevice::new(512, 4096);

    let store1 = BareMetalKappaStore::open(dev.clone()).unwrap();
    let _ = store1.put("blake3", b"first-boot-payload").unwrap();
    let epoch_b1 = store1.reboot_epoch();
    let gen_b1 = store1.generation();
    drop(store1);

    let store2 = BareMetalKappaStore::open(dev.clone()).unwrap();
    let epoch_b2 = store2.reboot_epoch();
    let gen_b2 = store2.generation();
    // Persist the new epoch.
    let _ = store2.put("blake3", b"second-boot-payload").unwrap();
    drop(store2);

    let store3 = BareMetalKappaStore::open(dev).unwrap();
    let epoch_b3 = store3.reboot_epoch();

    assert_eq!(epoch_b1, 1);
    assert_eq!(epoch_b2, 2);
    assert_eq!(epoch_b3, 3);
    assert!(
        (epoch_b2, gen_b2) > (epoch_b1, gen_b1),
        "(epoch, gen) total order across reboots: ({epoch_b1},{gen_b1}) < ({epoch_b2},{gen_b2})"
    );
}

/// `RuntimeStateRegion` realizations recorded across reboots compare correctly via `(epoch, gen)`.
#[test]
fn b1_runtime_state_region_pairs_compare_correctly_across_reboots() {
    let mk = |epoch: u64, gen_: u64| -> RuntimeStateRegion {
        let state = address_bytes(b"runtime-state-fixture");
        let mut payload = Vec::new();
        payload.extend_from_slice(&64u64.to_le_bytes()); // region_lba
        payload.extend_from_slice(&8u32.to_le_bytes()); // sectors
        payload.extend_from_slice(&gen_.to_le_bytes());
        payload.extend_from_slice(&epoch.to_le_bytes());
        RuntimeStateRegion {
            state,
            region_payload: payload,
        }
    };

    let earlier = mk(1, 5);
    let later_in_same_epoch = mk(1, 6);
    let later_in_new_epoch = mk(2, 1); // newer epoch even though gen reset to 1

    let decode = |r: &RuntimeStateRegion| RuntimeStateRegion::decode(&r.canonicalize()).unwrap();
    let (_, _, g_e, e_e) = decode(&earlier);
    let (_, _, g_s, e_s) = decode(&later_in_same_epoch);
    let (_, _, g_n, e_n) = decode(&later_in_new_epoch);

    assert!((e_e, g_e) < (e_s, g_s), "later-in-epoch dominates");
    assert!(
        (e_s, g_s) < (e_n, g_n),
        "newer epoch dominates regardless of gen"
    );
    assert!(g_n < g_s, "newer-epoch gen happens to be smaller");
    assert!(e_n > e_s, "but newer epoch dominates");
}
