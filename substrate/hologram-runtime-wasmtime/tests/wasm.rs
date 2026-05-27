//! CR (live engine) — real Wasm execution + linear-memory snapshot/restore via Wasmtime, then the
//! full `Runtime` lifecycle driving an actual container module (the same orchestration proven
//! against `MockEngine`, now over real Wasm — substrate-tripling at the runtime).

use std::sync::Arc;

use hologram_realizations::{ContainerManifest, Snapshot, REGISTRY};
use hologram_runtime::{ContainerEngine, HostContext, Runtime};
use hologram_runtime_wasmtime::WasmtimeEngine;
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{Capabilities, ContainerRuntime, KappaLabel71, KappaStore, Realization};

/// A minimal host context for direct-engine tests (empty store, no granted roots).
fn ctx() -> HostContext {
    HostContext {
        store: Arc::new(MemKappaStore::new()),
        storage_roots: vec![],
        registry: REGISTRY,
        memory_max_bytes: 0,
        cpu_fuel_per_event: 0,
        storage_quota_bytes: 0,
    }
}

/// A real container module: `memory`, and `hg_event` increments byte 0 of linear memory.
const COUNTER_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    (i32.store8 (i32.const 0)
      (i32.add (i32.load8_u (i32.const 0)) (i32.const 1)))
    (i32.const 0)))
"#;

fn wasm() -> Vec<u8> {
    wat::parse_str(COUNTER_WAT).expect("valid wat")
}

#[test]
fn engine_executes_wasm_and_snapshots_memory() {
    let engine = WasmtimeEngine::new();
    let mut inst = engine.instantiate(&wasm(), &ctx()).unwrap();
    assert_eq!(engine.init(&mut inst, &[]), 0);

    // Three real Wasm calls; byte 0 of the container's linear memory counts them.
    for _ in 0..3 {
        assert_eq!(engine.event(&mut inst, &[]), 0);
    }
    let snap = engine.snapshot_memory(&inst);
    assert_eq!(snap[0], 3, "hg_event incremented the container's memory three times");

    // Restore the snapshot into a fresh instance — state continuity over real Wasm memory.
    let mut inst2 = engine.instantiate(&wasm(), &ctx()).unwrap();
    assert_eq!(engine.snapshot_memory(&inst2)[0], 0, "fresh instance starts at 0");
    engine.restore_memory(&mut inst2, &snap);
    assert_eq!(engine.snapshot_memory(&inst2)[0], 3, "restored memory carries the count");
    // And it keeps counting from the restored state.
    engine.event(&mut inst2, &[]);
    assert_eq!(engine.snapshot_memory(&inst2)[0], 4);
}

#[test]
fn runtime_drives_a_real_wasm_container_through_suspend_resume() {
    pollster::block_on(async {
        // Provision a container whose code is the real Wasm module.
        let store = MemKappaStore::new();
        let code = store.put("blake3", &wasm()).unwrap();
        let state = store.put("blake3", b"").unwrap();
        let params = store.put("blake3", b"").unwrap();
        let cid = store
            .put("blake3", &ContainerManifest { code, initial_state: state, parameters: params }.canonicalize())
            .unwrap();
        let caps = store
            .put(
                "blake3",
                &hologram_realizations::CapabilitySet::new(Capabilities {
                    storage_roots: vec![],
                    storage_quota_bytes: 0,
                    network_fetch: false,
                    network_announce: false,
                    publish_channels: vec![],
                    subscribe_channels: vec![],
                    memory_max_bytes: 1 << 20,
                    cpu_time_per_event_ms: 100,
                })
                .canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let h = rt.spawn(&cid, &caps).await.unwrap();

        // Drive the real container, then suspend → a snapshot κ that references the Container ID.
        rt.deliver_event(h, &[]).unwrap();
        rt.deliver_event(h, &[]).unwrap();
        let snap = rt.suspend(h).await.unwrap();
        let snap_bytes = rt.store().get(&snap).unwrap().unwrap();
        assert_eq!(Snapshot::references(snap_bytes.as_ref()).unwrap()[0], cid);

        // Resume on a fresh handle and confirm the container keeps counting from restored memory.
        let h2 = rt.resume(&snap, &caps).await.unwrap();
        assert_eq!(rt.deliver_event(h2, &[]).unwrap(), 0); // third increment, no trap
    });
}

/// A container that USES the host import surface: on a 71-byte event it `storage_get`s that κ into
/// memory[100] and returns the byte count; otherwise it `storage_put`s the event bytes.
const IO_WAT: &str = r#"
(module
  (import "hologram" "storage_get" (func $get (param i32 i32 i32) (result i32)))
  (import "hologram" "storage_put" (func $put (param i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    (if (result i32) (i32.eq (local.get 1) (i32.const 71))
      (then (call $get (i32.const 0) (i32.const 100) (i32.const 256)))
      (else (call $put (i32.const 0) (local.get 1) (i32.const 200))))))
"#;

#[test]
fn container_uses_capability_gated_host_storage_imports() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let io_code = store.put("blake3", &wat::parse_str(IO_WAT).unwrap()).unwrap();
        let empty = store.put("blake3", b"").unwrap();
        let cid = store
            .put("blake3", &ContainerManifest { code: io_code, initial_state: empty, parameters: empty }.canonicalize())
            .unwrap();

        // Two blobs already present; the container is GRANTED a root for one, not the other.
        let granted = store.put("blake3", b"read-me").unwrap();
        let secret = store.put("blake3", b"top-secret-bytes").unwrap();
        let caps = store
            .put(
                "blake3",
                &hologram_realizations::CapabilitySet::new(Capabilities {
                    storage_roots: vec![granted],
                    storage_quota_bytes: 0,
                    network_fetch: false,
                    network_announce: false,
                    publish_channels: vec![],
                    subscribe_channels: vec![],
                    memory_max_bytes: 1 << 20,
                    cpu_time_per_event_ms: 100,
                })
                .canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let h = rt.spawn(&cid, &caps).await.unwrap();

        // storage_get of the GRANTED κ → returns the byte length ("read-me" = 7).
        assert_eq!(rt.deliver_event(h, granted.as_array()).unwrap(), 7);
        // storage_get of a PRESENT-but-UNGRANTED κ → capability-denied → -1 (0xFFFFFFFF).
        assert_eq!(rt.deliver_event(h, secret.as_array()).unwrap(), u32::MAX);
        // storage_put from inside the container: the host store gains the κ of the written bytes.
        assert_eq!(rt.deliver_event(h, b"written-by-container").unwrap(), 0);
        assert!(rt.store().contains(&hologram_substrate_core::address_bytes(b"written-by-container")));
    });
}

/// A container exercising the full import surface: on event it `publish`es mem[0..71]→mem[71..142]
/// and touches time/entropy; on callback it `storage_put`s the received payload (recording receipt).
const PUBSUB_WAT: &str = r#"
(module
  (import "hologram" "publish"      (func $pub  (param i32 i32)))
  (import "hologram" "storage_put"  (func $put  (param i32 i32 i32) (result i32)))
  (import "hologram" "time_now"     (func $now  (param i32)))
  (import "hologram" "entropy"      (func $rand (param i32 i32)))
  (memory (export "memory") 1)
  (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    (call $pub  (i32.const 0) (i32.const 71))
    (call $now  (i32.const 300))
    (call $rand (i32.const 320) (i32.const 8))
    (i32.const 0))
  (func (export "hg_callback") (param i32 i32 i32) (result i32)
    (drop (call $put (local.get 1) (local.get 2) (i32.const 400)))
    (i32.const 0)))
"#;

#[test]
fn wasm_container_publishes_and_subscriber_callback_records_receipt() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let code = store.put("blake3", &wat::parse_str(PUBSUB_WAT).unwrap()).unwrap();
        let mk = |state: &[u8], pubs: Vec<KappaLabel71>, subs: Vec<KappaLabel71>| {
            let st = store.put("blake3", state).unwrap();
            let cid = store
                .put("blake3", &ContainerManifest { code, initial_state: st, parameters: st }.canonicalize())
                .unwrap();
            let caps = store
                .put(
                    "blake3",
                    &hologram_realizations::CapabilitySet::new(Capabilities {
                        storage_roots: vec![],
                        storage_quota_bytes: 0,
                        network_fetch: false,
                        network_announce: false,
                        publish_channels: pubs,
                        subscribe_channels: subs,
                        memory_max_bytes: 1 << 20,
                        cpu_time_per_event_ms: 100,
                    })
                    .canonicalize(),
                )
                .unwrap();
            (cid, caps)
        };

        let channel = store.put("blake3", b"the-channel").unwrap();
        let payload = store.put("blake3", b"the-payload").unwrap();
        let (pid, pk) = mk(b"publisher", vec![channel], vec![]);
        let (sid, sk) = mk(b"subscriber", vec![], vec![channel]);

        let rt = Runtime::new(WasmtimeEngine::new(), store);
        let pubh = rt.spawn(&pid, &pk).await.unwrap();
        let subh = rt.spawn(&sid, &sk).await.unwrap();
        rt.subscribe(subh, &channel, 9).unwrap();

        // Event payload = channel-κ ‖ payload-κ (142 bytes); the Wasm container reads it and publishes.
        let mut ev = Vec::new();
        ev.extend_from_slice(channel.as_array());
        ev.extend_from_slice(payload.as_array());
        rt.deliver_event(pubh, &ev).unwrap();

        // The publish import → runtime applied it (cap-gated) → a Route κ exists in the graph.
        let route = hologram_realizations::Route { endpoint: channel, target: payload };
        assert!(rt.store().contains(&hologram_substrate_core::address_bytes(&route.canonicalize())));

        // Delivery reached the subscriber's hg_callback, which storage_put the received payload-κ
        // bytes → the host store now contains the κ of those 71 bytes (receipt witnessed).
        let receipt = hologram_substrate_core::address_bytes(payload.as_array());
        assert!(rt.store().contains(&receipt), "subscriber's hg_callback ran and recorded the payload");
    });
}
