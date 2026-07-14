//! Worked example: a **hologram-ai LLM-inference container**, end-to-end over the substrate —
//! and a trustless migration where the snapshot is fetched from an untrusted peer.
//!
//! This is a real-world scenario exercising the whole Phase-0 reference: realizations
//! (container-manifest, capability-set, snapshot), the κ-store (put/get/pin/gc), and the
//! eviction-tolerant async read path `get_with_fetch` with **verify-on-receipt** (SPINE-4) —
//! including rejection of a malicious peer (the property that makes the network trustless).

use std::collections::HashMap;

use async_trait::async_trait;
use hologram_realizations::{CapabilitySet, ContainerManifest, Snapshot, REGISTRY};
use hologram_space::{
    address_bytes, get_with_fetch, Bytes, Capabilities, KappaLabel71, KappaStore, KappaSync,
    Realization, SyncError,
};
use hologram_store_mem::MemKappaStore;

fn caps_with(
    storage_roots: Vec<KappaLabel71>,
    publish_channels: Vec<KappaLabel71>,
) -> Capabilities {
    Capabilities {
        storage_roots,
        publish_channels,
        subscribe_channels: vec![],
        storage_quota_bytes: 0,
        memory_max_bytes: 0,
        cpu_time_per_event_ms: 0,
        priority_weight: 0,
        network_fetch: false,
        network_announce: false,
    }
}

/// A peer/gateway that serves κ-addressed bytes (spec §6). `honest=false` simulates a malicious
/// peer that returns forged bytes for every request — the receiver must reject it (SPINE-4).
struct MockPeer {
    blobs: HashMap<[u8; 71], Vec<u8>>,
    honest: bool,
}

#[async_trait]
impl KappaSync for MockPeer {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        if !self.honest {
            return Ok(Some(Bytes::from(b"forged-content".to_vec())));
        }
        Ok(self
            .blobs
            .get(kappa.as_array())
            .map(|v| Bytes::from(v.clone())))
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

/// Build and store a hologram-ai inference container's manifest + capability set on peer A.
fn provision(store: &MemKappaStore) -> (KappaLabel71, KappaLabel71) {
    // The container body: a Prism inference model compiled to Wasm (κ of its bytes), plus the
    // model weights as initial state, plus instantiation params. These are leaves.
    let code = store.put("blake3", b"<llm-runtime.wasm>").unwrap();
    let weights = store.put("blake3", b"<gguf-weights>").unwrap();
    let params = store
        .put("blake3", br#"{"ctx":4096,"axis":"blake3"}"#)
        .unwrap();

    // The Container ID *is* the manifest (spec §4.1): identity binds code+state+params.
    let manifest = ContainerManifest {
        code,
        initial_state: weights,
        parameters: params,
    };
    let container_id = store.put("blake3", &manifest.canonicalize()).unwrap();

    // A capability set: may read the model-data root, publish on the "completions" channel,
    // bounded budgets. The authority is itself a κ-label (SPINE-1).
    let model_root = store.put("blake3", b"<model-data-root>").unwrap();
    let completions = store.put("blake3", b"<channel:completions>").unwrap();
    let caps = CapabilitySet::new(caps_with(vec![model_root], vec![completions]));
    let caps_k = store.put("blake3", &caps.canonicalize()).unwrap();

    (container_id, caps_k)
}

#[test]
fn llm_container_lifecycle_and_trustless_migration() {
    pollster::block_on(async {
        // ── Peer A: provision the container and suspend it to a snapshot ──
        let peer_a = MemKappaStore::new();
        let (container_id, caps_k) = provision(&peer_a);

        // Suspend → snapshot (operands: the container id; payload: opaque linear-memory digest).
        let snapshot = Snapshot {
            container_id,
            previous: None,
            storage_used: 0,
            state_payload: b"<linear-memory+globals+cursor>".to_vec(),
        };
        let snapshot_k = peer_a.put("blake3", &snapshot.canonicalize()).unwrap();

        // Pin the runtime root set and GC — everything reachable from the pinned snapshot/manifest
        // survives; nothing is lost.
        peer_a.pin(&snapshot_k).unwrap();
        peer_a.pin(&container_id).unwrap();
        peer_a.pin(&caps_k).unwrap();
        let before = peer_a.approximate_count();
        peer_a.gc(REGISTRY);
        assert_eq!(
            peer_a.approximate_count(),
            before,
            "all pinned-reachable κ retained"
        );

        // ── Migration to peer B via an HONEST peer: fetch the snapshot, verify on receipt ──
        let mut wire = HashMap::new();
        for k in peer_a.iterate() {
            wire.insert(*k.as_array(), peer_a.get(&k).unwrap().unwrap().to_vec());
        }
        let honest = MockPeer {
            blobs: wire,
            honest: true,
        };

        let peer_b = MemKappaStore::new();
        // The snapshot isn't local on B → get_with_fetch pulls it from the peer, verifies the
        // κ by re-derivation (SPINE-4), and caches it. Resume would proceed from these bytes.
        let got = get_with_fetch(&peer_b, &honest, &snapshot_k).await.unwrap();
        assert!(got.is_some(), "honest peer serves the snapshot");
        assert!(
            peer_b.contains(&snapshot_k),
            "verified bytes are cached locally"
        );

        // The recovered snapshot's references resolve to the same Container ID (graph continuity).
        let refs =
            Snapshot::references(peer_b.get(&snapshot_k).unwrap().unwrap().as_ref()).unwrap();
        assert_eq!(refs, vec![container_id]);

        // ── Migration via a MALICIOUS peer: forged bytes are rejected (trustless) ──
        let evil = MockPeer {
            blobs: HashMap::new(),
            honest: false,
        };
        let peer_c = MemKappaStore::new();
        let result = get_with_fetch(&peer_c, &evil, &snapshot_k).await;
        assert!(
            result.is_err(),
            "forged content fails σ-axis re-derivation (SPINE-4)"
        );
        assert!(!peer_c.contains(&snapshot_k), "nothing forged is cached");
    });
}

#[test]
fn capability_set_grants_are_reachability_roots() {
    // A capability set's granted κ-labels (storage roots, channels) are its references — so
    // pinning the capability set keeps the authority's targets reachable (spec §5.3).
    let root = address_bytes(b"data-root");
    let chan = address_bytes(b"chan");
    let caps = CapabilitySet::new(caps_with(vec![root], vec![chan]));
    let refs = CapabilitySet::references(&caps.canonicalize()).unwrap();
    assert_eq!(refs, vec![root, chan]);
    // And the canonical form decodes back to the same Capabilities (round-trip for enforcement).
    assert_eq!(
        CapabilitySet::to_capabilities(&caps.canonicalize()).unwrap(),
        caps.caps
    );
}
