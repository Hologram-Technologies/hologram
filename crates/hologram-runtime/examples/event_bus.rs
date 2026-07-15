//! Real-world use-case: an **inter-container event bus** (IoT telemetry / message queue).
//!
//! A "sensor" container publishes readings to a channel; a "logger" subscribes and receives each via
//! `hg_callback` — including readings published while the logger was **suspended** (durable
//! subscriptions, §10.11). Run: `cargo run -p hologram-runtime-wasmtime --example event_bus`.

use hologram_runtime::{MockEngine, Runtime};
use hologram_space::{Capabilities, ContainerRuntime, KappaLabel71, KappaStore, Realization};
use hologram_space::{CapabilitySet, ContainerManifest};
use hologram_tck::MemKappaStore;

fn caps(publish: Vec<KappaLabel71>, subscribe: Vec<KappaLabel71>) -> Capabilities {
    Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: 0,
        network_fetch: false,
        network_announce: false,
        publish_channels: publish,
        subscribe_channels: subscribe,
        memory_max_bytes: 0,
        cpu_time_per_event_ms: 0,
        priority_weight: 0,
    }
}

fn provision(store: &MemKappaStore, tag: &[u8], c: Capabilities) -> (KappaLabel71, KappaLabel71) {
    let code = store.put("blake3", tag).unwrap();
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
    (
        cid,
        store
            .put("blake3", &CapabilitySet::new(c).canonicalize())
            .unwrap(),
    )
}

fn main() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let topic = store.put("blake3", b"channel:telemetry").unwrap();
        let (sensor, sk) = provision(&store, b"sensor", caps(vec![topic], vec![]));
        let (logger, lk) = provision(&store, b"logger", caps(vec![], vec![topic]));

        let rt = Runtime::new(MockEngine, store);
        let s = rt.spawn(&sensor, &sk).await.unwrap();
        let l = rt.spawn(&logger, &lk).await.unwrap();
        rt.subscribe(l, &topic, 1).unwrap();

        // The logger goes offline (suspended); the sensor publishes two readings while it's down.
        let snap = rt.suspend(l).await.unwrap();
        let r1 = rt.store().put("blake3", b"temp=21.5C").unwrap();
        let r2 = rt.store().put("blake3", b"temp=22.1C").unwrap();
        rt.publish(s, &topic, &r1).unwrap();
        rt.publish(s, &topic, &r2).unwrap();
        println!("publish   : sensor → telemetry: temp=21.5C, temp=22.1C (logger offline)");

        // Logger resumes and catches up on everything it missed (durable subscription).
        let l2 = rt.resume(&snap, &lk).await.unwrap();
        let received = rt.delivered_callbacks(l2);
        println!(
            "delivered : logger caught up on {} reading(s) published while offline",
            received.len()
        );
        assert_eq!(received.len(), 2);

        // Capability gate: a container without publish rights is refused.
        let intruder = rt.spawn(&logger, &lk).await.unwrap();
        assert!(rt.publish(intruder, &topic, &r1).is_err());
        println!("OK — durable pub/sub event bus with capability-gated publish + offline catch-up");
    });
}
