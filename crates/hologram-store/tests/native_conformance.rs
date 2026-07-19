#![cfg(feature = "native")]
//! The native redb backend runs the *same* TCK as the in-memory reference (TR: one conformance
//! definition, validated identically per substrate) plus the reachability-GC witness (ST/§10.8).

use hologram_space::{ContainerManifest, REGISTRY};
use hologram_space::{KappaStore, Realization};
use hologram_store::native::NativeKappaStore;
use hologram_tck::store_battery;

#[test]
fn native_passes_the_kappastore_tck() {
    let store = NativeKappaStore::in_memory().unwrap();
    store_battery(&store);
}

#[test]
fn native_reachability_gc_matches_reference_semantics() {
    let store = NativeKappaStore::in_memory().unwrap();
    let code = store.put("blake3", b"wasm").unwrap();
    let state = store.put("blake3", b"state").unwrap();
    let params = store.put("blake3", b"params").unwrap();
    let manifest = ContainerManifest {
        code,
        initial_state: state,
        parameters: params,
    };
    let mk = store.put("blake3", &manifest.canonicalize()).unwrap();
    let orphan = store.put("blake3", b"orphan").unwrap();

    store.pin(&mk).unwrap();
    let evicted = store.gc(REGISTRY).unwrap();

    assert_eq!(evicted, 1);
    assert!(
        store.contains(&mk)
            && store.contains(&code)
            && store.contains(&state)
            && store.contains(&params)
    );
    assert!(!store.contains(&orphan));
}

#[test]
fn native_persists_across_reopen() {
    // Durability: a κ put into a file-backed store is present after reopen (the addressing
    // relation survives a restart — SPINE-5 / crash-safety direction).
    let dir = std::env::temp_dir().join(format!("hologram-tck-{}", std::process::id()));
    let path = dir.with_extension("redb");
    let _ = std::fs::remove_file(&path);
    let k = {
        let store = NativeKappaStore::open(&path).unwrap();
        store.put("blake3", b"durable").unwrap()
    };
    {
        let store = NativeKappaStore::open(&path).unwrap();
        assert_eq!(store.get(&k).unwrap().unwrap().as_ref(), b"durable");
    }
    let _ = std::fs::remove_file(&path);
}
