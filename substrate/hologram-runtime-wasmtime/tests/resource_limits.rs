//! Resource budget enforcement (spec §7.6 / §10 resource errors): the runtime bounds a container's
//! **CPU per event** (Wasmtime fuel), **linear memory** (StoreLimits), and **storage quota** (a
//! per-container byte ledger on `storage_put`) — declared in its Capability Set. `0 = unbounded`.

use hologram_realizations::{CapabilitySet, ContainerManifest};
use hologram_runtime::Runtime;
use hologram_runtime_wasmtime::WasmtimeEngine;
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{address_bytes, Capabilities, ContainerRuntime, KappaStore, Realization};

fn caps(mem_max: u64, cpu_ms: u64, quota: u64) -> Capabilities {
    Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: quota,
        network_fetch: false,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: mem_max,
        cpu_time_per_event_ms: cpu_ms,
    }
}

fn provision(store: &MemKappaStore, tag: &[u8], wasm: &[u8], c: Capabilities) -> (hologram_substrate_core::KappaLabel71, hologram_substrate_core::KappaLabel71) {
    let code = store.put("blake3", wasm).unwrap();
    let st = store.put("blake3", tag).unwrap();
    let cid = store.put("blake3", &ContainerManifest { code, initial_state: st, parameters: st }.canonicalize()).unwrap();
    let ck = store.put("blake3", &CapabilitySet::new(c).canonicalize()).unwrap();
    (cid, ck)
}

#[test]
fn cpu_budget_bounds_a_runaway_event() {
    // hg_event loops forever; a 1 ms (= bounded fuel) budget traps it instead of hanging (§7.5).
    const LOOP: &str = r#"
    (module (memory (export "memory") 1)
      (func (export "hg_init") (param i32 i32) (result i32) (i32.const 0))
      (func (export "hg_suspend") (result i32) (i32.const 0))
      (func (export "hg_resume")  (result i32) (i32.const 0))
      (func (export "hg_event") (param i32 i32) (result i32) (loop (br 0)) (i32.const 0)))"#;
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let (cid, ck) = provision(&store, b"loop", &wat::parse_str(LOOP).unwrap(), caps(0, 1, 0));
        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let h = rt.spawn(&cid, &ck).await.unwrap();
        // Returns a nonzero status (fuel-exhausted PipelineFailure) — crucially, it *returns* (no hang).
        assert_ne!(rt.deliver_event(h, b"").unwrap(), 0, "runaway event is fuel-bounded");
        // The container survives for subsequent events (the engine refuels per call).
        assert_ne!(rt.deliver_event(h, b"").unwrap(), 0);
    });
}

#[test]
fn memory_budget_caps_linear_memory_growth() {
    // hg_event grows memory by 10 pages and returns the result (-1 if capped).
    const GROW: &str = r#"
    (module (memory (export "memory") 1)
      (func (export "hg_init") (param i32 i32) (result i32) (i32.const 0))
      (func (export "hg_suspend") (result i32) (i32.const 0))
      (func (export "hg_resume")  (result i32) (i32.const 0))
      (func (export "hg_event") (param i32 i32) (result i32) (memory.grow (i32.const 10))))"#;
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let wasm = wat::parse_str(GROW).unwrap();
        // Capped at 1 page (64 KiB): a 10-page grow is refused → -1 (0xFFFFFFFF).
        let (cid, ck) = provision(&store, b"grow-capped", &wasm, caps(64 * 1024, 100, 0));
        // Unbounded: the same grow succeeds → old size (1 page).
        let (cid2, ck2) = provision(&store, b"grow-free", &wasm, caps(0, 100, 0));
        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let capped = rt.spawn(&cid, &ck).await.unwrap();
        let free = rt.spawn(&cid2, &ck2).await.unwrap();
        assert_eq!(rt.deliver_event(capped, b"").unwrap(), u32::MAX, "grow past memory cap refused");
        assert_eq!(rt.deliver_event(free, b"").unwrap(), 1, "unbounded memory grows (old size = 1 page)");
    });
}

#[test]
fn storage_quota_bounds_what_a_container_persists() {
    // hg_event storage_puts its event bytes; a small quota refuses an over-budget put (§7.6).
    const PUT: &str = r#"
    (module
      (import "hologram" "storage_put" (func $put (param i32 i32 i32) (result i32)))
      (memory (export "memory") 2)
      (func (export "hg_init") (param i32 i32) (result i32) (i32.const 0))
      (func (export "hg_suspend") (result i32) (i32.const 0))
      (func (export "hg_resume")  (result i32) (i32.const 0))
      (func (export "hg_event") (param i32 i32) (result i32)
        (call $put (i32.const 0) (local.get 1) (i32.const 600))))"#;
    pollster::block_on(async {
        let store = MemKappaStore::new();
        // Quota of 8 bytes.
        let (cid, ck) = provision(&store, b"quota", &wat::parse_str(PUT).unwrap(), caps(0, 100, 8));
        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let h = rt.spawn(&cid, &ck).await.unwrap();

        // A within-quota put (5 bytes) succeeds.
        assert_eq!(rt.deliver_event(h, b"12345").unwrap(), 0);
        assert!(rt.store().contains(&address_bytes(b"12345")));
        // An over-quota put (the remaining budget is 3 bytes) is refused → -1, and not stored.
        let big = [0u8; 50];
        assert_eq!(rt.deliver_event(h, &big).unwrap(), u32::MAX, "over-quota put refused");
        assert!(!rt.store().contains(&address_bytes(&big)));
    });
}
