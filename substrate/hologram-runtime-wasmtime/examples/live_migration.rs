//! Real-world use-case: **checkpoint / restore + live migration** (serverless cold-start-from-
//! snapshot, VM/container migration).
//!
//! A stateful session container runs on node A, is suspended to a snapshot κ, the reachable bytes
//! are shipped to node B, and the session resumes on B with its linear memory intact — addressed
//! entirely by κ. Run: `cargo run -p hologram-runtime-wasmtime --example live_migration`.

use hologram_realizations::{CapabilitySet, ContainerManifest, Snapshot};
use hologram_runtime::Runtime;
use hologram_runtime_wasmtime::WasmtimeEngine;
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{Capabilities, ContainerRuntime, KappaStore, Realization};

/// A session container: hg_event increments a counter at memory[0] (its session state).
const SESSION_WAT: &str = r#"
(module (memory (export "memory") 1)
  (func (export "hg_init") (param i32 i32) (result i32) (i32.const 0))
  (func (export "hg_suspend") (result i32) (i32.const 0))
  (func (export "hg_resume")  (result i32) (i32.const 0))
  (func (export "hg_event") (param i32 i32) (result i32)
    (i32.store8 (i32.const 0) (i32.add (i32.load8_u (i32.const 0)) (i32.const 1)))
    (i32.load8_u (i32.const 0))))
"#;

fn provision(
    store: &MemKappaStore,
) -> (
    hologram_substrate_core::KappaLabel71,
    hologram_substrate_core::KappaLabel71,
) {
    let code = store
        .put("blake3", &wat::parse_str(SESSION_WAT).unwrap())
        .unwrap();
    let cid = store
        .put(
            "blake3",
            &ContainerManifest {
                code,
                initial_state: code,
                parameters: code,
            }
            .canonicalize(),
        )
        .unwrap();
    let caps = store
        .put(
            "blake3",
            &CapabilitySet::new(Capabilities {
                storage_roots: vec![],
                storage_quota_bytes: 0,
                network_fetch: false,
                network_announce: false,
                publish_channels: vec![],
                subscribe_channels: vec![],
                memory_max_bytes: 0,
                cpu_time_per_event_ms: 100,
                priority_weight: 0,
            })
            .canonicalize(),
        )
        .unwrap();
    (cid, caps)
}

fn main() {
    pollster::block_on(async {
        // Node A: run the session through three requests, then checkpoint.
        let node_a = MemKappaStore::new();
        let (cid, caps) = provision(&node_a);
        let rt_a = Runtime::new(WasmtimeEngine::new(), node_a);
        let h = rt_a.spawn(&cid, &caps).await.unwrap();
        // Empty events: the counter lives at memory[0]; an event payload would be written there too.
        for _ in 0..3 {
            rt_a.deliver_event(h, b"").unwrap();
        }
        let snap = rt_a.suspend(h).await.unwrap();
        println!(
            "node A    : 3 requests served; checkpointed → snapshot κ {}",
            snap.as_str()
        );

        // Ship the reachable bytes (snapshot + manifest + code) to node B.
        let node_b = MemKappaStore::new();
        for k in rt_a.store().iterate() {
            node_b
                .put("blake3", rt_a.store().get(&k).unwrap().unwrap().as_ref())
                .unwrap();
        }
        let snap_refs = Snapshot::references(node_b.get(&snap).unwrap().unwrap().as_ref()).unwrap();
        println!(
            "transfer  : snapshot references Container ID {} (graph continuity)",
            snap_refs[0].as_str()
        );

        // Node B: resume from the snapshot κ — session state (counter == 3) is intact.
        let rt_b = Runtime::new(WasmtimeEngine::new(), node_b);
        let h2 = rt_b.resume(&snap, &caps).await.unwrap();
        let next = rt_b.deliver_event(h2, b"").unwrap();
        println!(
            "node B    : resumed; next request returns {} (state continued from 3 → 4)",
            next
        );
        assert_eq!(next, 4, "session counter survived migration");

        println!(
            "OK — live migration: checkpoint on A → resume on B from the snapshot κ, state intact"
        );
    });
}
