//! F1 — **cross-peer channel fanout** (architecture §4.4, V&V row CH).
//!
//! Two separate `Runtime`s ("peer A" and "peer B") each own a local store; both share a
//! `KappaSync` that announces κs cross-peer and answers fetches. The publisher on peer A calls
//! `publish` (which queues an `announce(route_κ)` intent); after `process_pending_network`,
//! peer B sees the announcement, `poll_channel_fanout` discovers it, fetches the Route κ, and
//! delivers the payload to its local subscriber. The Wasm container's `hg_callback` fires —
//! cross-peer end-to-end via only the existing `KappaSync` surface.

use std::sync::Arc;

use async_trait::async_trait;
use hologram_realizations::{CapabilitySet, ContainerManifest};
use hologram_runtime::{MockEngine, Runtime};
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{
    Bytes, Capabilities, ContainerRuntime, KappaLabel71, KappaStore, KappaSync, Realization,
    SyncError,
};

fn caps(pubs: &[KappaLabel71], subs: &[KappaLabel71]) -> Capabilities {
    Capabilities {
        storage_roots: vec![],
        storage_quota_bytes: 0,
        network_fetch: true,
        network_announce: true,
        publish_channels: pubs.to_vec(),
        subscribe_channels: subs.to_vec(),
        memory_max_bytes: 1 << 20,
        cpu_time_per_event_ms: 100,
        priority_weight: 0,
    }
}

fn provision(store: &MemKappaStore, tag: &[u8], c: Capabilities) -> (KappaLabel71, KappaLabel71) {
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

/// A two-peer `KappaSync`: announcements propagate to the *other* peer's store; fetches look in
/// the other peer's store. The "network" is just two `Arc<MemKappaStore>` references.
struct TwoPeerSync {
    /// The peer this sync writes to on announce / reads from on fetch (i.e. the *other* peer).
    other: Arc<MemKappaStore>,
    /// Our local store — used to read announced bytes back for cross-peer delivery.
    local: Arc<MemKappaStore>,
}

#[async_trait]
impl KappaSync for TwoPeerSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(self.other.get(kappa).unwrap_or(None))
    }
    async fn announce(&self, kappa: &KappaLabel71) {
        // Mirror the announced bytes from our local store into the other peer's store. This is
        // what the uor-native `hologram-net-tcp` `Provide` + `FetchReq` flow achieves on the wire.
        if let Ok(Some(b)) = self.local.get(kappa) {
            let _ = self.other.put("blake3", b.as_ref());
        }
    }
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        // Surface every κ the other peer has (the "DHT crawl" — bounded here by the test set).
        self.other.iterate()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

#[test]
fn f1_cross_peer_channel_publish_fans_out_to_remote_subscriber() {
    pollster::block_on(async {
        // Shared channel κ + payload κ — content-addressed, identical on every peer (TR class).
        let bootstrap = MemKappaStore::new();
        let chan = bootstrap.put("blake3", b"cross-peer-channel").unwrap();
        let payload = bootstrap
            .put("blake3", b"the-message-published-by-A")
            .unwrap();

        // Peer A — publisher.
        let store_a_seed = MemKappaStore::new();
        let (pid, pk) = provision(&store_a_seed, b"publisher", caps(&[chan], &[]));
        let _ = store_a_seed
            .put("blake3", b"the-message-published-by-A")
            .unwrap();
        // Peer B — subscriber.
        let store_b_seed = MemKappaStore::new();
        let (sid, sk) = provision(&store_b_seed, b"subscriber", caps(&[], &[chan]));

        // Build runtimes first; then obtain Arc handles to their stores for the sync wiring.
        let rt_a = Runtime::new(MockEngine, store_a_seed);
        let rt_b = Runtime::new(MockEngine, store_b_seed);
        let store_a: Arc<MemKappaStore> = rt_a.store_arc();
        let store_b: Arc<MemKappaStore> = rt_b.store_arc();

        let sync_a: Arc<dyn KappaSync> = Arc::new(TwoPeerSync {
            other: store_b.clone(),
            local: store_a.clone(),
        });
        let sync_b: Arc<dyn KappaSync> = Arc::new(TwoPeerSync {
            other: store_a.clone(),
            local: store_b.clone(),
        });

        let rt_a = rt_a.with_sync(sync_a);
        let rt_b = rt_b.with_sync(sync_b);

        let pub_h = rt_a.spawn(&pid, &pk).await.unwrap();
        let sub_h = rt_b.spawn(&sid, &sk).await.unwrap();
        rt_b.subscribe(sub_h, &chan, 7).unwrap();

        // 1. Publish on A — locally mints a Route κ; queues an Announce intent.
        let route_k = rt_a.publish(pub_h, &chan, &payload).unwrap();

        // 2. Drive A's network tick — announces the Route κ to peer B (writes it into B's store).
        let announced = rt_a.process_pending_network().await;
        assert!(
            announced >= 1,
            "publish should have queued at least one announce intent"
        );
        assert!(
            rt_b.store().contains(&route_k),
            "peer B's store now holds the announced Route κ"
        );

        // 3. Drive B's subscriber-side poll — discovers the Route, classifies it under `chan`,
        //    and `pump()` delivers to the local subscriber's hg_callback.
        let delivered = rt_b.poll_channel_fanout(&chan).await.unwrap();
        assert!(
            delivered >= 1,
            "subscriber peer must have observed the Route fan-out"
        );

        // The cross-peer subscriber's callback fired with the publisher's payload κ.
        assert!(
            rt_b.delivered_callbacks(sub_h)
                .contains(&(7, payload.as_array().to_vec())),
            "remote subscriber received the κ via hg_callback (cross-peer fanout)"
        );
    });
}
