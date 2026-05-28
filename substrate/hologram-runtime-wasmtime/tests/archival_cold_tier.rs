//! AR class (arch §11.4): the **archival cold tier** is a hologram peer backed by the bare-metal
//! substrate (`BareMetalKappaStore` over a `BlockDevice`). It participates in the federated read
//! path via the same `KappaSync` surface every other peer uses; ordering the federation chain
//! hot → cold produces cold-tier latency without external hosting. Test: a κ that lives **only**
//! on the bare-metal cold peer is reached by a federated fetch that walked the chain past two
//! warmer peers that didn't have it.

use async_trait::async_trait;
use hologram_bare_hal::RamBlockDevice;
use hologram_store_bare::BareMetalKappaStore;
use hologram_store_mem::MemKappaStore;
use hologram_substrate_core::{
    Bytes, FederatedKappaSync, KappaLabel71, KappaStore, KappaSync, SyncError,
};
use std::sync::Arc;

/// Trivial adapter: present any local [`KappaStore`] as a [`KappaSync`] so the federation chain
/// can include in-process peers. A real cluster would wrap each store behind `serve(...)` (HTTP)
/// or a `TcpKappaSync` (the uor-native `hologram-net-tcp` transport) — both are KappaSyncs.
struct StoreAsSync<S: KappaStore + Send + Sync>(Arc<S>);

#[async_trait]
impl<S: KappaStore + Send + Sync + 'static> KappaSync for StoreAsSync<S> {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        Ok(self.0.get(kappa).unwrap_or(None))
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new()
    }
    async fn add_peer(&self, _peer_addr: &str) -> Result<(), SyncError> {
        Err(SyncError::BackendFailure("local"))
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Err(SyncError::BackendFailure("local"))
    }
}

#[test]
fn archival_cold_tier_via_bare_metal_peer_in_federation() {
    pollster::block_on(async {
        // 1. Hot peer: RAM-backed MemKappaStore. Empty.
        let hot = Arc::new(MemKappaStore::new());

        // 2. Warm peer: a second RAM peer (acting as a redb-style native cache; same trait surface).
        //    Empty too.
        let warm = Arc::new(MemKappaStore::new());

        // 3. Cold peer: **bare-metal** store over a RAM block device. This is the substrate-tripling
        //    cold tier (§11.4) — same wire as the others, durable across reboots via §11.3.
        let device = RamBlockDevice::new(512, 4096);
        let cold_store = Arc::new(BareMetalKappaStore::open(device).unwrap());
        let payload = b"archived-on-bare-metal-cold-tier";
        let archived_kappa = cold_store.put("blake3", payload).unwrap();

        // 4. Build the federation chain hot → warm → cold, all over the same KappaSync surface.
        let fed = FederatedKappaSync::new(vec![
            Arc::new(StoreAsSync(hot.clone())) as Arc<dyn KappaSync>,
            Arc::new(StoreAsSync(warm.clone())) as Arc<dyn KappaSync>,
            Arc::new(StoreAsSync(cold_store.clone())) as Arc<dyn KappaSync>,
        ]);

        // 5. The archived κ lives ONLY on the bare-metal cold peer. A federated fetch resolves it
        //    by falling through hot (Ok(None)), warm (Ok(None)), and finally cold (Ok(Some(_))).
        let got = fed.fetch(&archived_kappa).await.unwrap();
        assert_eq!(got.unwrap().as_ref(), payload);

        // 6. A κ the network doesn't have falls through all three peers → Ok(None).
        let absent = hologram_substrate_core::address_bytes(b"present-nowhere");
        assert_eq!(fed.fetch(&absent).await.unwrap(), None);

        // 7. Sanity: neither hot nor warm picked up the archived bytes (no write-through here);
        //    the federation chain alone exposed cold without writing into the warmer tiers.
        assert!(!hot.contains(&archived_kappa));
        assert!(!warm.contains(&archived_kappa));
    });
}
