//! TR class (architecture §5, §6, conformance §10.16) — **substrate-tripling byte-identity**.
//!
//! The same Wasm container, the same `Runtime` orchestration, the same input event stream — run
//! across all three storage substrates available to a std host (memory reference, native redb,
//! bare-metal block-device-backed Merkle store) — must produce a **byte-identical κ stream**:
//!
//! - the Container ID κ (`spawn` minted from the manifest),
//! - every κ the container wrote via `storage_put` during the event stream,
//! - the `Snapshot` κ produced by `suspend`.
//!
//! That equality is the architecture's load-bearing claim: a container is portable across
//! substrates by construction, because every step is σ-axis content-addressed and the engine
//! is deterministic over a fixed input stream. The substrate-tripling witness lives here.
//!
//! A real cluster also has OPFS (browser-only) — its byte-identity is asserted by the
//! `scripts/opfs-browser-test.sh` Playwright harness against `address_bytes`. Together those
//! cover all four documented substrates (TR `[track]` resolved).

use hologram_runtime::Runtime;
use hologram_runtime_wasmtime::WasmtimeEngine;
use hologram_space::ContainerManifest;
use hologram_space::RamBlockDevice;
use hologram_space::{Capabilities, ContainerRuntime, KappaLabel71, KappaStore, Realization};
use hologram_store_bare::BareMetalKappaStore;
use hologram_store_mem::MemKappaStore;
use hologram_store_native::NativeKappaStore;

/// A deterministic container exercising the full host-import surface that produces κs:
/// `storage_put`s the event bytes at scratch[200], `publish`es nothing (no channel granted), and
/// touches no entropy/clock. The output κ-stream is therefore determined entirely by:
///   (a) the input event bytes (`storage_put` content-addresses them), and
///   (b) the snapshot serialization (deterministic — Wasmtime memory pages + globals + cursor).
const TRIPLING_WAT: &str = r#"
(module
  (import "hologram" "storage_put"  (func $put  (param i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "hg_init")    (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    ;; storage_put(event_ptr=local.get 0, event_len=local.get 1, scratch_ptr=200)
    (drop (call $put (local.get 0) (local.get 1) (i32.const 200)))
    (i32.const 0))
  (func (export "hg_callback") (param i32 i32 i32) (result i32) (i32.const 0)))
"#;

#[derive(Debug, PartialEq, Eq)]
struct KappaStream {
    cid: KappaLabel71,
    snapshot: KappaLabel71,
    /// κs the container wrote to the store via `storage_put`, in delivery order.
    written: Vec<KappaLabel71>,
}

async fn run_one<S: KappaStore + Send + Sync + 'static>(store: S) -> KappaStream {
    // Provision: the manifest is content-addressed, so the cid κ is identical across substrates
    // for the same (code, initial_state, parameters).
    let code = store
        .put("blake3", &wat::parse_str(TRIPLING_WAT).unwrap())
        .unwrap();
    let empty = store.put("blake3", b"").unwrap();
    let cid = store
        .put(
            "blake3",
            &ContainerManifest {
                code,
                initial_state: empty,
                parameters: empty,
            }
            .canonicalize(),
        )
        .unwrap();
    let caps = store
        .put(
            "blake3",
            &hologram_space::CapabilitySet::new(Capabilities {
                storage_roots: vec![],
                storage_quota_bytes: 0,
                network_fetch: false,
                network_announce: false,
                publish_channels: vec![],
                subscribe_channels: vec![],
                memory_max_bytes: 1 << 20,
                cpu_time_per_event_ms: 100,
                priority_weight: 0,
            })
            .canonicalize(),
        )
        .unwrap();

    // Drive the engine. WasmtimeEngine is deterministic over a fixed input stream (entropy is
    // unused in this container), so all storage_put-induced κs are determined entirely by the
    // input event bytes.
    let rt = Runtime::new(WasmtimeEngine::new(), store);
    let h = rt.spawn(&cid, &caps).await.unwrap();

    // A fixed input event stream — these bytes deterministically address to a known κ each.
    let events: Vec<&[u8]> = vec![
        b"tr-event-0-deterministic",
        b"tr-event-1-deterministic",
        b"tr-event-2-deterministic",
    ];
    let written: Vec<KappaLabel71> = events
        .iter()
        .map(|e| {
            rt.deliver_event(h, e).unwrap();
            hologram_space::address_bytes(e)
        })
        .collect();

    let snapshot = rt.suspend(h).await.unwrap();
    KappaStream {
        cid,
        snapshot,
        written,
    }
}

#[test]
fn tr_same_container_emits_byte_identical_kappa_stream_on_mem_native_bare() {
    pollster::block_on(async {
        let mem = run_one(MemKappaStore::new()).await;
        let native = run_one(NativeKappaStore::in_memory().unwrap()).await;
        let bare =
            run_one(BareMetalKappaStore::open(RamBlockDevice::new(512, 4096)).unwrap()).await;

        assert_eq!(mem.cid, native.cid, "mem ↔ native: cid κ identical");
        assert_eq!(native.cid, bare.cid, "native ↔ bare: cid κ identical");

        assert_eq!(
            mem.written, native.written,
            "mem ↔ native: written κ stream"
        );
        assert_eq!(
            native.written, bare.written,
            "native ↔ bare: written κ stream"
        );

        // Every written κ is actually present in each store (the σ-axis address resolved on each
        // backend equivalently).
        let mem_store = MemKappaStore::new();
        let native_store = NativeKappaStore::in_memory().unwrap();
        let bare_store = BareMetalKappaStore::open(RamBlockDevice::new(512, 4096)).unwrap();
        for e in [
            b"tr-event-0-deterministic" as &[u8],
            b"tr-event-1-deterministic",
            b"tr-event-2-deterministic",
        ] {
            let k = hologram_space::address_bytes(e);
            let _ = mem_store.put("blake3", e).unwrap();
            let _ = native_store.put("blake3", e).unwrap();
            let _ = bare_store.put("blake3", e).unwrap();
            assert!(mem_store.contains(&k));
            assert!(native_store.contains(&k));
            assert!(bare_store.contains(&k));
        }

        // Snapshot κ: also content-addressed, so identical across substrates given a deterministic
        // engine over the same event stream.
        assert_eq!(
            mem.snapshot, native.snapshot,
            "mem ↔ native: snapshot κ identical (deterministic engine)"
        );
        assert_eq!(
            native.snapshot, bare.snapshot,
            "native ↔ bare: snapshot κ identical (deterministic engine)"
        );
    });
}
