//! # hologram-net-libp2p
//!
//! The peer-to-peer [`KappaSync`] transport (spec §6.2): κ-label fetch over a libp2p
//! **request-response** protocol on **TCP + Noise + Yamux**, with **verify-on-receipt** (SPINE-4).
//! A served node answers κ requests from its local store; a client dials a peer and fetches, then
//! re-derives the κ before accepting the bytes (a peer cannot serve forged content).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use hologram_substrate_core::{
    verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaStore, KappaSync, SyncError,
};
use libp2p_core::transport::Transport;
use libp2p_core::upgrade::Version;
use libp2p_core::Multiaddr;
use libp2p_request_response::{self as request_response, Message, ProtocolSupport};
use libp2p_swarm::{Config as SwarmConfig, StreamProtocol};
use libp2p_swarm::{Swarm, SwarmEvent};
use tokio::sync::oneshot;

const PROTOCOL: &str = "/hologram/cas/1";

type Behaviour = request_response::cbor::Behaviour<Vec<u8>, Option<Vec<u8>>>;

// Built from libp2p sub-crates rather than the `libp2p` umbrella's `SwarmBuilder`
// (which would lock the DNS/mDNS resolver into the dependency graph). The transport
// is exactly TCP → Noise authentication → Yamux multiplexing, the same stack the
// umbrella's `.with_tcp(...)` produces.
fn build_swarm() -> Swarm<Behaviour> {
    let id_keys = libp2p_identity::Keypair::generate_ed25519();
    let peer_id = id_keys.public().to_peer_id();
    let transport = libp2p_tcp::tokio::Transport::new(libp2p_tcp::Config::default())
        .upgrade(Version::V1Lazy)
        .authenticate(libp2p_noise::Config::new(&id_keys).expect("noise"))
        .multiplex(libp2p_yamux::Config::default())
        .boxed();
    let behaviour = Behaviour::new(
        [(StreamProtocol::new(PROTOCOL), ProtocolSupport::Full)],
        request_response::Config::default(),
    );
    Swarm::new(
        transport,
        behaviour,
        peer_id,
        SwarmConfig::with_tokio_executor(),
    )
}

/// A running κ-serving node; holds its listen address. Dropping it stops the node.
pub struct ServedNode {
    addr: Multiaddr,
    task: tokio::task::JoinHandle<()>,
}

impl ServedNode {
    pub fn addr(&self) -> &Multiaddr {
        &self.addr
    }
}

impl Drop for ServedNode {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Serve κ requests from `store` over libp2p; returns once the node is listening.
pub async fn serve(store: Arc<dyn KappaStore>) -> ServedNode {
    let mut swarm = build_swarm();
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .expect("listen");
    let (addr_tx, addr_rx) = oneshot::channel::<Multiaddr>();
    let mut addr_tx = Some(addr_tx);

    let task = tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::NewListenAddr { address, .. } => {
                    if let Some(tx) = addr_tx.take() {
                        let _ = tx.send(address);
                    }
                }
                SwarmEvent::Behaviour(request_response::Event::Message {
                    message:
                        Message::Request {
                            request, channel, ..
                        },
                    ..
                }) => {
                    let resp: Option<Vec<u8>> = <[u8; 71]>::try_from(request.as_slice())
                        .ok()
                        .and_then(|a| KappaLabel::from_bytes(&a).ok())
                        .and_then(|k| store.get(&k).ok().flatten())
                        .map(|b| b.as_ref().to_vec());
                    let _ = swarm.behaviour_mut().send_response(channel, resp);
                }
                _ => {}
            }
        }
    });

    let addr = addr_rx.await.expect("listen addr");
    ServedNode { addr, task }
}

/// Dial `peer_addr`, request `kappa`, and **verify on receipt**. `Ok(None)` ⇒ the peer doesn't have
/// it; `Err(VerificationFailed)` ⇒ the peer served bytes that don't re-derive to `kappa`.
pub async fn fetch_once(
    peer_addr: &Multiaddr,
    kappa: &KappaLabel71,
) -> Result<Option<Bytes>, SyncError> {
    let mut swarm = build_swarm();
    swarm
        .dial(peer_addr.clone())
        .map_err(|_| SyncError::BackendFailure("dial"))?;
    let kappa = *kappa;

    let fut = async {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    swarm
                        .behaviour_mut()
                        .send_request(&peer_id, kappa.as_array().to_vec());
                }
                SwarmEvent::Behaviour(request_response::Event::Message {
                    message: Message::Response { response, .. },
                    ..
                }) => return Ok(response),
                SwarmEvent::OutgoingConnectionError { .. } => {
                    return Err(SyncError::BackendFailure("connect"));
                }
                SwarmEvent::Behaviour(request_response::Event::OutboundFailure { .. }) => {
                    return Err(SyncError::AllSourcesFailed);
                }
                _ => {}
            }
        }
    };

    let response: Option<Vec<u8>> = match tokio::time::timeout(Duration::from_secs(10), fut).await {
        Ok(r) => r?,
        Err(_) => return Err(SyncError::AllSourcesFailed),
    };

    match response {
        None => Ok(None),
        Some(bytes) => match verify_kappa(&bytes, &kappa) {
            Ok(true) => Ok(Some(Bytes::from(bytes))),
            _ => Err(SyncError::VerificationFailed), // forged peer rejected (§6.4)
        },
    }
}

/// A [`KappaSync`] over a set of libp2p peer multiaddrs.
#[derive(Default)]
pub struct Libp2pKappaSync {
    peers: std::sync::Mutex<Vec<Multiaddr>>,
}

impl Libp2pKappaSync {
    pub fn new(peers: Vec<Multiaddr>) -> Self {
        Self {
            peers: std::sync::Mutex::new(peers),
        }
    }
}

#[async_trait]
impl KappaSync for Libp2pKappaSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        let peers = self.peers.lock().unwrap().clone();
        if peers.is_empty() {
            return Err(SyncError::NotEnabled);
        }
        let mut last = SyncError::AllSourcesFailed;
        for p in peers {
            match fetch_once(&p, kappa).await {
                Ok(Some(b)) => return Ok(Some(b)),
                Ok(None) => last = SyncError::AllSourcesFailed,
                Err(e) => last = e, // skip a failing/forging peer, try the next
            }
        }
        if matches!(last, SyncError::AllSourcesFailed) {
            Ok(None)
        } else {
            Err(last)
        }
    }
    async fn announce(&self, _kappa: &KappaLabel71) {}
    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        Vec::new() // Kademlia provider records are the discovery layer (follow-on); fetch is here.
    }
    async fn add_peer(&self, multiaddr: &str) -> Result<(), SyncError> {
        let a: Multiaddr = multiaddr
            .parse()
            .map_err(|_| SyncError::BackendFailure("multiaddr"))?;
        self.peers.lock().unwrap().push(a);
        Ok(())
    }
    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_store_mem::MemKappaStore;
    use hologram_substrate_core::address_bytes;

    #[tokio::test]
    async fn two_node_libp2p_fetch_verifies_and_404s() {
        // Node B serves a store holding κ over libp2p (TCP+Noise+Yamux).
        let store = Arc::new(MemKappaStore::new());
        let k = store.put("blake3", b"fetched-over-libp2p").unwrap();
        let node_b = serve(store).await;

        // Node A fetches κ from B's multiaddr → verified bytes.
        let sync = Libp2pKappaSync::new(vec![node_b.addr().clone()]);
        let got = sync.fetch(&k).await.unwrap();
        assert_eq!(got.unwrap().as_ref(), b"fetched-over-libp2p");

        // A κ the peer doesn't have → Ok(None).
        let absent = address_bytes(b"absent-from-peer");
        assert_eq!(sync.fetch(&absent).await.unwrap(), None);
    }

    #[tokio::test]
    async fn libp2p_fetch_with_no_peers_is_not_enabled() {
        let sync = Libp2pKappaSync::new(vec![]);
        assert_eq!(
            sync.fetch(&address_bytes(b"x")).await,
            Err(SyncError::NotEnabled)
        );
    }
}
