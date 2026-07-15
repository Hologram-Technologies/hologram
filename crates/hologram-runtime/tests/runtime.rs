//! CR — Container Runtime conformance (spec §4 / §10). Lifecycle, snapshot-as-κ continuity,
//! capability **enforcement** at delegation (the `admits` containment wired at `spawn_child`),
//! revocation, and cross-runtime migration — all hermetic against the mock engine.

use hologram_runtime::{MockEngine, Runtime};
use hologram_space::{
    Capabilities, ContainerHandle, ContainerRuntime, ContainerState, KappaLabel71, KappaStore,
    Realization,
};
use hologram_space::{CapabilitySet, ContainerManifest};
use hologram_tck::MemKappaStore;

fn caps(roots: &[KappaLabel71], quota: u64, fetch: bool) -> Capabilities {
    Capabilities {
        storage_roots: roots.to_vec(),
        storage_quota_bytes: quota,
        network_fetch: fetch,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: 0,
    }
}

/// Provision a container (manifest + code + caps) into a store; return (container_id, caps_kappa).
fn provision(
    store: &MemKappaStore,
    body: &[u8],
    state: &[u8],
    c: Capabilities,
) -> (KappaLabel71, KappaLabel71) {
    let code = store.put("blake3", body).unwrap();
    let st = store.put("blake3", state).unwrap();
    let params = store.put("blake3", b"params").unwrap();
    let manifest = ContainerManifest {
        code,
        initial_state: st,
        parameters: params,
    };
    let cid = store.put("blake3", &manifest.canonicalize()).unwrap();
    let caps_k = store
        .put("blake3", &CapabilitySet::new(c).canonicalize())
        .unwrap();
    (cid, caps_k)
}

#[test]
fn cr_lifecycle_spawn_suspend_resume_preserves_state() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let (cid, caps_k) = provision(&store, b"<wasm>", b"INIT", caps(&[], 1000, false));
        let rt = Runtime::new(MockEngine, store);

        let h = rt.spawn(&cid, &caps_k).await.unwrap();
        assert_eq!(rt.info(h).unwrap().state, ContainerState::Running);

        // Mutate state through an event, then suspend → snapshot κ.
        rt.deliver_event(h, b"-EVENT").unwrap();
        let snap = rt.suspend(h).await.unwrap();
        assert_eq!(rt.info(h).unwrap().state, ContainerState::Suspended);

        // The snapshot's references resolve to the same Container ID (continuity).
        let snap_bytes = rt.store().get(&snap).unwrap().unwrap();
        assert_eq!(
            hologram_space::Snapshot::references(snap_bytes.as_ref()).unwrap()[0],
            cid
        );

        // Resume into a fresh handle; the restored linear memory is exactly the pre-suspend state.
        let h2 = rt.resume(&snap, &caps_k).await.unwrap();
        assert_eq!(rt.info(h2).unwrap().state, ContainerState::Running);
        assert_eq!(
            rt.info(h2).unwrap().memory_bytes,
            ("INIT-EVENT".len()) as u64
        );
    });
}

#[test]
fn cr_terminate_removes_the_instance() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let (cid, caps_k) = provision(&store, b"<wasm>", b"", caps(&[], 0, false));
        let rt = Runtime::new(MockEngine, store);
        let h = rt.spawn(&cid, &caps_k).await.unwrap();
        assert_eq!(rt.list().len(), 1);
        rt.terminate(h).await.unwrap();
        assert!(rt.list().is_empty());
        assert!(rt.info(h).is_none());
    });
}

#[test]
fn cr_spawn_child_enforces_capability_containment() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let r1 = store.put("blake3", b"root-1").unwrap();
        let r2 = store.put("blake3", b"root-2").unwrap();

        // Parent holds {r1} with quota 1000, no fetch.
        let (cid, parent_caps) = provision(&store, b"<wasm>", b"", caps(&[r1], 1000, false));
        // A properly-narrowed child caps {r1}, quota 500.
        let narrow = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(&[r1], 500, false)).canonicalize(),
            )
            .unwrap();
        // An over-broad child caps {r1, r2} (extra root) — must be refused.
        let broad = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(&[r1, r2], 500, false)).canonicalize(),
            )
            .unwrap();
        // An over-broad child requesting a flag the parent lacks — must be refused.
        let flagged = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(&[r1], 500, true)).canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(MockEngine, store);
        let parent = rt.spawn(&cid, &parent_caps).await.unwrap();

        assert!(
            rt.spawn_child(parent, &cid, &narrow).is_ok(),
            "narrowed delegation admitted"
        );
        assert!(
            rt.spawn_child(parent, &cid, &broad).is_err(),
            "extra storage root refused"
        );
        assert!(
            rt.spawn_child(parent, &cid, &flagged).is_err(),
            "un-held network flag refused"
        );
    });
}

#[test]
fn cr_revocation_refuses_subsequent_operations() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let (cid, caps_k) = provision(&store, b"<wasm>", b"", caps(&[], 0, false));
        let rt = Runtime::new(MockEngine, store);
        let h = rt.spawn(&cid, &caps_k).await.unwrap();

        rt.deliver_event(h, b"ok").unwrap(); // works before revocation
        rt.revoke(&caps_k);
        assert!(
            rt.deliver_event(h, b"nope").is_err(),
            "events refused after revocation"
        );
        assert!(
            rt.spawn(&cid, &caps_k).await.is_err(),
            "spawn refused with a revoked cap set"
        );
    });
}

#[test]
fn rv_transitive_revoke_cascades_through_delegation() {
    // arch §11.8: revoking the grandparent's caps cascades through the κ-graph Delegation cone —
    // A → B → C, revoke(A) refuses C's future operations because Delegation κ-edges connect them.
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let r = store.put("blake3", b"shared-root").unwrap();
        let (cid, caps_a) = provision(&store, b"<wasm>", b"", caps(&[r], 1000, false));
        let caps_b = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(&[r], 500, false)).canonicalize(),
            )
            .unwrap();
        let caps_c = store
            .put(
                "blake3",
                &CapabilitySet::new(caps(&[r], 100, false)).canonicalize(),
            )
            .unwrap();

        let rt = Runtime::new(MockEngine, store);
        let a = rt.spawn(&cid, &caps_a).await.unwrap();
        let b = rt.spawn_child(a, &cid, &caps_b).unwrap();
        // C is spawned under B (NOT under A) — the cascade must traverse the κ-graph two hops.
        let _c = rt.spawn_child(b, &cid, &caps_c).unwrap();

        // Before revoke: all three operate normally.
        assert!(rt.deliver_event(a, b"ok").is_ok());
        // Revoking A cascades through the Delegation κ-graph: B AND C become unable to operate.
        rt.revoke(&caps_a);
        assert!(rt.deliver_event(a, b"nope").is_err(), "A directly revoked");
        assert!(
            rt.spawn(&cid, &caps_b).await.is_err(),
            "B refused — descendant of A in the delegation cone"
        );
        assert!(
            rt.spawn(&cid, &caps_c).await.is_err(),
            "C refused — two-hop descendant via the κ-graph"
        );
    });
}

#[test]
fn cr_cross_runtime_migration_from_snapshot() {
    pollster::block_on(async {
        // Peer A: spawn, run, suspend → snapshot κ.
        let store_a = MemKappaStore::new();
        let (cid, caps_k) = provision(&store_a, b"<wasm>", b"STATE", caps(&[], 0, false));
        let rt_a = Runtime::new(MockEngine, store_a);
        let h = rt_a.spawn(&cid, &caps_k).await.unwrap();
        rt_a.deliver_event(h, b"+A").unwrap();
        let snap = rt_a.suspend(h).await.unwrap();

        // Transfer the reachable bytes (snapshot + manifest + code) to peer B's store.
        let store_b = MemKappaStore::new();
        for k in rt_a.store().iterate() {
            store_b
                .put("blake3", rt_a.store().get(&k).unwrap().unwrap().as_ref())
                .unwrap();
        }
        // Peer B resumes from the same snapshot κ — same Container ID, restored memory.
        let rt_b = Runtime::new(MockEngine, store_b);
        let h2 = rt_b.resume(&snap, &caps_k).await.unwrap();
        assert_eq!(rt_b.info(h2).unwrap().container_id, cid);
        assert_eq!(
            rt_b.info(h2).unwrap().memory_bytes,
            ("STATE+A".len()) as u64
        );
    });
}

// ───────────────────────────── channels / pub-sub (spec §4.4, §10.4, §10.11) ─────────────────────────────

fn caps_chan(publish: &[KappaLabel71], subscribe: &[KappaLabel71]) -> Capabilities {
    Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: 0,
        network_fetch: false,
        network_announce: false,
        publish_channels: publish.to_vec(),
        subscribe_channels: subscribe.to_vec(),
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: 0,
    }
}

fn provision_caps(
    store: &MemKappaStore,
    tag: &[u8],
    c: Capabilities,
) -> (KappaLabel71, KappaLabel71) {
    // `tag` distinguishes containers — identical manifests content-address to the SAME Container ID,
    // and subscriptions are keyed by Container ID (§10.11), so publisher and subscriber must differ.
    let code = store.put("blake3", tag).unwrap();
    let e = store.put("blake3", b"").unwrap();
    let cid = store
        .put(
            "blake3",
            &ContainerManifest {
                code,
                initial_state: e,
                parameters: e,
            }
            .canonicalize(),
        )
        .unwrap();
    let ck = store
        .put("blake3", &CapabilitySet::new(c).canonicalize())
        .unwrap();
    (cid, ck)
}

#[test]
fn ch_publish_delivers_to_subscriber_via_callback() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let chan = store.put("blake3", b"channel:topic").unwrap();
        let payload = store.put("blake3", b"a-message").unwrap();
        let (pid, pk) = provision_caps(&store, b"publisher", caps_chan(&[chan], &[]));
        let (sid, sk) = provision_caps(&store, b"subscriber", caps_chan(&[], &[chan]));
        let rt = Runtime::new(MockEngine, store);

        let pubh = rt.spawn(&pid, &pk).await.unwrap();
        let subh = rt.spawn(&sid, &sk).await.unwrap();
        rt.subscribe(subh, &chan, 42).unwrap();
        rt.publish(pubh, &chan, &payload).unwrap();

        // The subscriber's hg_callback received (callback_id=42, payload-κ bytes).
        let info = rt.info(subh).unwrap();
        assert_eq!(info.container_id, sid);
        // MockEngine recorded the callback on the instance; expose via a fresh subscribe-then-check.
        assert!(
            rt.delivered_callbacks(subh)
                .contains(&(42, payload.as_array().to_vec())),
            "subscriber received the published κ via hg_callback"
        );
    });
}

#[test]
fn ch_publish_and_subscribe_are_capability_gated() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let chan = store.put("blake3", b"chan").unwrap();
        let other = store.put("blake3", b"other").unwrap();
        let payload = store.put("blake3", b"m").unwrap();
        // Container may publish to `chan` only; subscribe to `chan` only.
        let (cid, ck) = provision_caps(&store, b"both", caps_chan(&[chan], &[chan]));
        let rt = Runtime::new(MockEngine, store);
        let h = rt.spawn(&cid, &ck).await.unwrap();

        assert!(
            rt.publish(h, &other, &payload).is_err(),
            "publish to ungranted channel refused"
        );
        assert!(
            rt.subscribe(h, &other, 1).is_err(),
            "subscribe to ungranted channel refused"
        );
        assert!(rt.publish(h, &chan, &payload).is_ok());
        assert!(rt.subscribe(h, &chan, 1).is_ok());
    });
}

#[test]
fn ch_subscription_persists_across_suspend_resume() {
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let chan = store.put("blake3", b"durable-channel").unwrap();
        let payload = store.put("blake3", b"published-during-suspension").unwrap();
        let (pid, pk) = provision_caps(&store, b"publisher", caps_chan(&[chan], &[]));
        let (sid, sk) = provision_caps(&store, b"subscriber", caps_chan(&[], &[chan]));
        let rt = Runtime::new(MockEngine, store);

        let pubh = rt.spawn(&pid, &pk).await.unwrap();
        let subh = rt.spawn(&sid, &sk).await.unwrap();
        rt.subscribe(subh, &chan, 7).unwrap();

        // Suspend the subscriber, publish while it's down, then resume → delivery replays (§10.11).
        let snap = rt.suspend(subh).await.unwrap();
        rt.publish(pubh, &chan, &payload).unwrap(); // subscriber suspended → not delivered yet
        let subh2 = rt.resume(&snap, &sk).await.unwrap();
        assert!(
            rt.delivered_callbacks(subh2)
                .contains(&(7, payload.as_array().to_vec())),
            "message published during suspension replayed on resume"
        );
    });
}

fn caps_weighted(weight: u32) -> Capabilities {
    Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: 0,
        network_fetch: false,
        network_announce: false,
        publish_channels: vec![],
        subscribe_channels: vec![],
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: weight,
    }
}

#[test]
fn sc_drr_fairness_over_uortime() {
    // arch §11.7: a deficit-round-robin scheduler keyed by `(handle, UorTime)` — NOT wall-clock —
    // delivers events at a ratio matching `priority_weight`. With weights {1, 1, 4} and quantum=1
    // we expect roughly {1, 1, 4} events per round. No backoff timing involved: the order is
    // deterministic over uor-native quantities.
    pollster::block_on(async {
        let store = MemKappaStore::new();
        let (cid_lo1, ck_lo1) = provision(&store, b"<wasm>", b"", caps_weighted(1));
        let (cid_lo2, ck_lo2) = provision(&store, b"<wasm>", b"", caps_weighted(1));
        let (cid_hi, ck_hi) = provision(&store, b"<wasm>", b"", caps_weighted(4));
        let rt = Runtime::new(MockEngine, store);
        let lo1 = rt.spawn(&cid_lo1, &ck_lo1).await.unwrap();
        let lo2 = rt.spawn(&cid_lo2, &ck_lo2).await.unwrap();
        let hi = rt.spawn(&cid_hi, &ck_hi).await.unwrap();

        // Flood every container with ample work (50 events each).
        for _ in 0..50 {
            rt.enqueue_event(lo1, b"e".to_vec()).unwrap();
            rt.enqueue_event(lo2, b"e".to_vec()).unwrap();
            rt.enqueue_event(hi, b"e".to_vec()).unwrap();
        }

        // One round at quantum=1 should serve {1, 1, 4} (deficit covers exactly the weight).
        let r1 = rt.pump_round(1);
        let count = |h: ContainerHandle| r1.iter().filter(|(x, _)| *x == h).count();
        assert_eq!(count(lo1), 1, "weight 1 → 1 event/round");
        assert_eq!(count(lo2), 1, "weight 1 → 1 event/round");
        assert_eq!(
            count(hi),
            4,
            "weight 4 → 4 events/round (no starvation either way)"
        );
    });
}
