//! # hologram-net-libp2p
//!
//! The peer-to-peer [`KappaSync`] transport (spec §6.2 + arch §11.1): κ-label fetch over libp2p
//! **request-response** on **TCP + Noise + Yamux**, with **verify-on-receipt** (SPINE-4), AND
//! **content discovery via Kademlia DHT** — `announce(κ)` calls `start_providing(κ)`, `fetch(κ)`
//! falls through to `get_providers(κ)` when the κ isn't local + isn't on a known peer, and
//! `discover(prefix, limit)` does `get_closest_peers(prefix)` + a `List` RR request to each closest
//! peer. The DHT key IS the κ-label (`κ.as_bytes()`) — the κ is its own routing key, no parallel
//! naming scheme. Every byte received still re-derives through the σ-axis before being accepted.
//!
//! Built from libp2p sub-crates rather than the umbrella (the umbrella unconditionally locks the
//! DNS/mDNS resolver path → `hickory-proto`, which `cargo audit` flags on a whole-lockfile scan).
//! The transport is the same TCP→Noise→Yamux stack `SwarmBuilder::with_tcp(...)` would produce.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use hologram_substrate_core::{
    verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaStore, KappaSync, SyncError,
};
use libp2p_core::transport::Transport;
use libp2p_core::upgrade::Version;
use libp2p_core::{multiaddr::Protocol, Multiaddr, PeerId};
use libp2p_kad::store::MemoryStore;
use libp2p_kad::{self as kad};
use libp2p_request_response::{
    self as request_response, Message, OutboundRequestId, ProtocolSupport, ResponseChannel,
};
use libp2p_swarm::{Config as SwarmConfig, NetworkBehaviour, StreamProtocol, Swarm, SwarmEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

const PROTOCOL: &str = "/hologram/cas/1";

/// Request envelope — `Get(κ)` (fetch-by-κ, existing) or `List{prefix, limit}` (κ-prefix discovery).
/// κ-labels are carried as `Vec<u8>` on the wire (serde's stock array impls cap at N=32, and our
/// labels are 71 bytes); convert at the boundary.
#[derive(Debug, Serialize, Deserialize)]
enum Req {
    Get(Vec<u8>),
    List { prefix: Vec<u8>, limit: u32 },
}

/// Response envelope — paired with the request variant.
#[derive(Debug, Serialize, Deserialize)]
enum Resp {
    Get(Option<Vec<u8>>),
    /// κ-labels are transmitted as 71-byte vectors.
    List(Vec<Vec<u8>>),
}

type Cas = request_response::cbor::Behaviour<Req, Resp>;
type Kad = kad::Behaviour<MemoryStore>;

// `prelude` redirects the derive macro's code generation to libp2p-swarm's re-export module —
// without it the macro emits paths like `libp2p::swarm::...` which only resolve under the umbrella
// crate (which we deliberately do not depend on — see Cargo.toml comment).
#[derive(NetworkBehaviour)]
#[behaviour(prelude = "libp2p_swarm::derive_prelude")]
struct PeerBehaviour {
    cas: Cas,
    kad: Kad,
}

/// Commands sent from the API to the persistent swarm task.
enum Cmd {
    AddPeer(Multiaddr, oneshot::Sender<Result<(), SyncError>>),
    Announce(KappaLabel71, oneshot::Sender<()>),
    Fetch(
        KappaLabel71,
        oneshot::Sender<Result<Option<Bytes>, SyncError>>,
    ),
    Discover(Option<Vec<u8>>, usize, oneshot::Sender<Vec<KappaLabel71>>),
}

/// In-flight fetch state — a κ-label whose providers are being walked via the DHT.
struct PendingFetch {
    reply: oneshot::Sender<Result<Option<Bytes>, SyncError>>,
    /// Providers found by `get_providers`, still to try (in order). Each provider is queried with
    /// `Req::Get(κ)`; on `Resp::Get(None)` or RR failure, the next provider is tried. When the
    /// list is exhausted (and `get_providers` has finished), the reply is `Ok(None)`.
    providers: Vec<PeerId>,
    /// `Some(_)` while a `Req::Get` is in flight to one of the providers above.
    in_flight: Option<OutboundRequestId>,
    /// `true` until `GetProviders` returns its last step, then we know `providers` won't grow.
    providers_more: bool,
}

/// In-flight discover state — a `List` query fanned out to closest peers.
struct PendingDiscover {
    prefix: Vec<u8>,
    limit: usize,
    reply: oneshot::Sender<Vec<KappaLabel71>>,
    /// Outstanding `Req::List` requests waiting on responses. When zero, the reply is sent.
    outstanding: usize,
    results: Vec<KappaLabel71>,
    /// `true` until `GetClosestPeers` returns its last step.
    peers_more: bool,
}

/// A running libp2p peer: serves κ requests from `store`, participates in the Kademlia DHT, and
/// answers fetch/announce/discover from the [`KappaSync`] surface. Drop to stop.
pub struct LibPeer {
    listen_addr: Multiaddr,
    cmd_tx: mpsc::UnboundedSender<Cmd>,
    task: tokio::task::JoinHandle<()>,
}

impl LibPeer {
    /// Spawn a new peer; returns once it's listening and its routable multiaddr is known. The
    /// multiaddr includes the `/p2p/<peer-id>` suffix so other peers can dial + populate their
    /// Kademlia routing table.
    pub async fn new(store: Arc<dyn KappaStore>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<Cmd>();
        let (addr_tx, addr_rx) = oneshot::channel::<Multiaddr>();
        let task = tokio::spawn(swarm_task(store, cmd_rx, addr_tx));
        let listen_addr = addr_rx.await.expect("listen addr");
        Self {
            listen_addr,
            cmd_tx,
            task,
        }
    }

    pub fn addr(&self) -> &Multiaddr {
        &self.listen_addr
    }
}

impl Drop for LibPeer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[async_trait]
impl KappaSync for LibPeer {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Cmd::Fetch(*kappa, tx))
            .map_err(|_| SyncError::BackendFailure("peer task gone"))?;
        rx.await
            .map_err(|_| SyncError::BackendFailure("fetch dropped"))?
    }

    async fn announce(&self, kappa: &KappaLabel71) {
        let (tx, rx) = oneshot::channel();
        if self.cmd_tx.send(Cmd::Announce(*kappa, tx)).is_ok() {
            let _ = rx.await;
        }
    }

    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71> {
        let (tx, rx) = oneshot::channel();
        if self
            .cmd_tx
            .send(Cmd::Discover(prefix.map(|p| p.to_vec()), limit, tx))
            .is_ok()
        {
            rx.await.unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    async fn add_peer(&self, multiaddr: &str) -> Result<(), SyncError> {
        let a: Multiaddr = multiaddr
            .parse()
            .map_err(|_| SyncError::BackendFailure("multiaddr"))?;
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Cmd::AddPeer(a, tx))
            .map_err(|_| SyncError::BackendFailure("peer task gone"))?;
        rx.await
            .map_err(|_| SyncError::BackendFailure("add_peer dropped"))?
    }

    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        // libp2p peers carry multiaddrs, not URLs. HTTP/IPFS gateways are wired by
        // FederatedKappaSync in hologram-substrate-core (architecture §11.2), which forwards
        // gateway URLs to HttpKappaSync / IpfsKappaSync. Surfacing "wrong transport" here makes
        // the caller pick the right backend.
        Err(SyncError::BackendFailure(
            "libp2p does not consume HTTP gateway URLs — use FederatedKappaSync",
        ))
    }
}

// ============================================================================
// Internals — the persistent swarm task
// ============================================================================

fn build_swarm(local_keys: libp2p_identity::Keypair) -> Swarm<PeerBehaviour> {
    let peer_id = local_keys.public().to_peer_id();
    let transport = libp2p_tcp::tokio::Transport::new(libp2p_tcp::Config::default())
        .upgrade(Version::V1Lazy)
        .authenticate(libp2p_noise::Config::new(&local_keys).expect("noise"))
        .multiplex(libp2p_yamux::Config::default())
        .boxed();

    let cas = Cas::new(
        [(StreamProtocol::new(PROTOCOL), ProtocolSupport::Full)],
        request_response::Config::default(),
    );
    let mut kad_cfg = kad::Config::new(StreamProtocol::new("/hologram/kad/1"));
    kad_cfg.set_query_timeout(Duration::from_secs(20));
    let mut kad = Kad::with_config(peer_id, MemoryStore::new(peer_id), kad_cfg);
    // Always-Server: don't wait for AutoNAT-style reachability confirmation before serving queries.
    // Peer multiaddrs are explicit in the substrate's deployment models — no NAT, no probing.
    kad.set_mode(Some(kad::Mode::Server));

    Swarm::new(
        transport,
        PeerBehaviour { cas, kad },
        peer_id,
        SwarmConfig::with_tokio_executor(),
    )
}

async fn swarm_task(
    store: Arc<dyn KappaStore>,
    mut cmd_rx: mpsc::UnboundedReceiver<Cmd>,
    addr_tx: oneshot::Sender<Multiaddr>,
) {
    let id_keys = libp2p_identity::Keypair::generate_ed25519();
    let local_peer_id = id_keys.public().to_peer_id();
    let mut swarm = build_swarm(id_keys);
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .expect("listen");

    let mut addr_tx = Some(addr_tx);

    // Pending fetches keyed by κ.
    let mut pending_fetch_by_kappa: HashMap<KappaLabel71, PendingFetch> = HashMap::new();
    // RR Get OutboundRequestId → the κ that's waiting on it.
    let mut fetch_inflight_to_kappa: HashMap<OutboundRequestId, KappaLabel71> = HashMap::new();
    // GetProviders QueryId → κ.
    let mut providers_query_to_kappa: HashMap<kad::QueryId, KappaLabel71> = HashMap::new();

    // Pending discover queries.
    let mut pending_discover_by_id: HashMap<kad::QueryId, PendingDiscover> = HashMap::new();
    // RR List OutboundRequestId → discover query id.
    let mut discover_inflight_to_query: HashMap<OutboundRequestId, kad::QueryId> = HashMap::new();

    loop {
        tokio::select! {
            ev = swarm.select_next_some() => match ev {
                SwarmEvent::NewListenAddr { address, .. } => {
                    if let Some(tx) = addr_tx.take() {
                        let mut full = address;
                        full.push(Protocol::P2p(local_peer_id));
                        let _ = tx.send(full);
                    }
                }

                SwarmEvent::Behaviour(PeerBehaviourEvent::Cas(request_response::Event::Message {
                    message: Message::Request { request, channel, .. },
                    ..
                })) => {
                    handle_inbound_request(&store, &mut swarm, request, channel);
                }

                SwarmEvent::Behaviour(PeerBehaviourEvent::Cas(request_response::Event::Message {
                    message: Message::Response { request_id, response },
                    ..
                })) => {
                    handle_inbound_response(
                        &mut swarm, request_id, response,
                        &mut pending_fetch_by_kappa, &mut fetch_inflight_to_kappa,
                        &mut pending_discover_by_id, &mut discover_inflight_to_query,
                    );
                }

                SwarmEvent::Behaviour(PeerBehaviourEvent::Cas(
                    request_response::Event::OutboundFailure { request_id, .. },
                )) => {
                    if let Some(kappa) = fetch_inflight_to_kappa.remove(&request_id) {
                        try_next_provider(&mut swarm, kappa, &mut pending_fetch_by_kappa, &mut fetch_inflight_to_kappa);
                    } else if let Some(qid) = discover_inflight_to_query.remove(&request_id) {
                        if let Some(pd) = pending_discover_by_id.get_mut(&qid) {
                            pd.outstanding = pd.outstanding.saturating_sub(1);
                        }
                        finalize_discover_if_idle(qid, &mut pending_discover_by_id);
                    }
                }

                SwarmEvent::Behaviour(PeerBehaviourEvent::Kad(kad::Event::OutboundQueryProgressed {
                    id, result, step, ..
                })) => {
                    handle_kad_progress(
                        &mut swarm, id, result, step.last,
                        &mut providers_query_to_kappa,
                        &mut pending_fetch_by_kappa, &mut fetch_inflight_to_kappa,
                        &mut pending_discover_by_id, &mut discover_inflight_to_query,
                    );
                }

                _ => {}
            },

            cmd = cmd_rx.recv() => match cmd {
                Some(Cmd::AddPeer(addr, reply)) => {
                    // The multiaddr must end in /p2p/<peer-id> so we know the PeerId for the routing
                    // table. Without it, libp2p-kad has no way to use the address.
                    if let Some(pid) = extract_peer_id(&addr) {
                        swarm.behaviour_mut().kad.add_address(&pid, addr.clone());
                        let _ = swarm.dial(addr);
                        let _ = reply.send(Ok(()));
                    } else {
                        let _ = reply.send(Err(SyncError::BackendFailure("multiaddr missing /p2p/<peer-id>")));
                    }
                }

                Some(Cmd::Announce(kappa, reply)) => {
                    let key = kad::RecordKey::new(&kappa.as_array());
                    let _ = swarm.behaviour_mut().kad.start_providing(key);
                    let _ = reply.send(());
                }

                Some(Cmd::Fetch(kappa, reply)) => {
                    // Fast path: serve from the local store.
                    if let Ok(Some(b)) = store.get(&kappa) {
                        let _ = reply.send(Ok(Some(b)));
                        continue;
                    }
                    // Otherwise, find providers via the DHT and dial them.
                    let key = kad::RecordKey::new(&kappa.as_array());
                    let qid = swarm.behaviour_mut().kad.get_providers(key);
                    providers_query_to_kappa.insert(qid, kappa);
                    pending_fetch_by_kappa.insert(kappa, PendingFetch {
                        reply, providers: Vec::new(), in_flight: None, providers_more: true,
                    });
                }

                Some(Cmd::Discover(prefix, limit, reply)) => {
                    let target_bytes = prefix.clone().unwrap_or_default();
                    let qid = swarm.behaviour_mut().kad.get_closest_peers(target_bytes);
                    pending_discover_by_id.insert(qid, PendingDiscover {
                        prefix: prefix.unwrap_or_default(),
                        limit, reply, outstanding: 0, results: Vec::new(), peers_more: true,
                    });
                }

                None => return,
            },
        }
    }
}

fn extract_peer_id(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| match p {
        Protocol::P2p(pid) => Some(pid),
        _ => None,
    })
}

fn handle_inbound_request(
    store: &Arc<dyn KappaStore>,
    swarm: &mut Swarm<PeerBehaviour>,
    request: Req,
    channel: ResponseChannel<Resp>,
) {
    let resp = match request {
        Req::Get(arr_vec) => {
            let bytes = <[u8; 71]>::try_from(arr_vec.as_slice())
                .ok()
                .and_then(|a| KappaLabel::<71>::from_bytes(&a).ok())
                .and_then(|k| store.get(&k).ok().flatten())
                .map(|b| b.as_ref().to_vec());
            Resp::Get(bytes)
        }
        Req::List { prefix, limit } => {
            let mut hits: Vec<Vec<u8>> = Vec::new();
            for k in store.iterate() {
                if hits.len() >= limit as usize {
                    break;
                }
                if k.as_array().starts_with(&prefix) {
                    hits.push(k.as_array().to_vec());
                }
            }
            Resp::List(hits)
        }
    };
    let _ = swarm.behaviour_mut().cas.send_response(channel, resp);
}

#[allow(clippy::too_many_arguments)]
fn handle_inbound_response(
    swarm: &mut Swarm<PeerBehaviour>,
    request_id: OutboundRequestId,
    response: Resp,
    pending_fetch_by_kappa: &mut HashMap<KappaLabel71, PendingFetch>,
    fetch_inflight_to_kappa: &mut HashMap<OutboundRequestId, KappaLabel71>,
    pending_discover_by_id: &mut HashMap<kad::QueryId, PendingDiscover>,
    discover_inflight_to_query: &mut HashMap<OutboundRequestId, kad::QueryId>,
) {
    if let Some(kappa) = fetch_inflight_to_kappa.remove(&request_id) {
        match response {
            Resp::Get(Some(bytes)) => match verify_kappa(&bytes, &kappa) {
                Ok(true) => {
                    if let Some(p) = pending_fetch_by_kappa.remove(&kappa) {
                        let _ = p.reply.send(Ok(Some(Bytes::from(bytes))));
                    }
                }
                _ => {
                    if let Some(p) = pending_fetch_by_kappa.remove(&kappa) {
                        let _ = p.reply.send(Err(SyncError::VerificationFailed));
                    }
                }
            },
            Resp::Get(None) | Resp::List(_) => {
                try_next_provider(
                    swarm,
                    kappa,
                    pending_fetch_by_kappa,
                    fetch_inflight_to_kappa,
                );
            }
        }
        return;
    }

    if let Some(qid) = discover_inflight_to_query.remove(&request_id) {
        if let Some(pd) = pending_discover_by_id.get_mut(&qid) {
            if let Resp::List(items) = response {
                for v in items {
                    if let Ok(arr) = <[u8; 71]>::try_from(v.as_slice()) {
                        if let Ok(k) = KappaLabel::<71>::from_bytes(&arr) {
                            if !pd.results.contains(&k) {
                                pd.results.push(k);
                            }
                        }
                    }
                }
            }
            pd.outstanding = pd.outstanding.saturating_sub(1);
        }
        finalize_discover_if_idle(qid, pending_discover_by_id);
    }
}

fn try_next_provider(
    swarm: &mut Swarm<PeerBehaviour>,
    kappa: KappaLabel71,
    pending_fetch_by_kappa: &mut HashMap<KappaLabel71, PendingFetch>,
    fetch_inflight_to_kappa: &mut HashMap<OutboundRequestId, KappaLabel71>,
) {
    let Some(pending) = pending_fetch_by_kappa.get_mut(&kappa) else {
        return;
    };
    pending.in_flight = None;
    if let Some(peer) = pending.providers.pop() {
        let req_id = swarm
            .behaviour_mut()
            .cas
            .send_request(&peer, Req::Get(kappa.as_array().to_vec()));
        pending.in_flight = Some(req_id);
        fetch_inflight_to_kappa.insert(req_id, kappa);
        return;
    }
    // No more providers right now; if GetProviders has also finished, we're done.
    if !pending.providers_more {
        if let Some(p) = pending_fetch_by_kappa.remove(&kappa) {
            let _ = p.reply.send(Ok(None));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_kad_progress(
    swarm: &mut Swarm<PeerBehaviour>,
    id: kad::QueryId,
    result: kad::QueryResult,
    is_last: bool,
    providers_query_to_kappa: &mut HashMap<kad::QueryId, KappaLabel71>,
    pending_fetch_by_kappa: &mut HashMap<KappaLabel71, PendingFetch>,
    fetch_inflight_to_kappa: &mut HashMap<OutboundRequestId, KappaLabel71>,
    pending_discover_by_id: &mut HashMap<kad::QueryId, PendingDiscover>,
    discover_inflight_to_query: &mut HashMap<OutboundRequestId, kad::QueryId>,
) {
    match result {
        kad::QueryResult::GetProviders(Ok(kad::GetProvidersOk::FoundProviders {
            providers,
            ..
        })) => {
            let Some(&kappa) = providers_query_to_kappa.get(&id) else {
                return;
            };
            if let Some(p) = pending_fetch_by_kappa.get_mut(&kappa) {
                for pid in providers {
                    if !p.providers.contains(&pid) {
                        p.providers.push(pid);
                    }
                }
                if p.in_flight.is_none() {
                    try_next_provider(
                        swarm,
                        kappa,
                        pending_fetch_by_kappa,
                        fetch_inflight_to_kappa,
                    );
                }
            }
            if is_last {
                providers_query_to_kappa.remove(&id);
                if let Some(p) = pending_fetch_by_kappa.get_mut(&kappa) {
                    p.providers_more = false;
                    if p.in_flight.is_none() && p.providers.is_empty() {
                        if let Some(p) = pending_fetch_by_kappa.remove(&kappa) {
                            let _ = p.reply.send(Ok(None));
                        }
                    }
                }
            }
        }
        kad::QueryResult::GetProviders(Ok(
            kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
        ))
        | kad::QueryResult::GetProviders(Err(_)) => {
            if let Some(kappa) = providers_query_to_kappa.remove(&id) {
                if let Some(p) = pending_fetch_by_kappa.get_mut(&kappa) {
                    p.providers_more = false;
                    if p.in_flight.is_none() && p.providers.is_empty() {
                        if let Some(p) = pending_fetch_by_kappa.remove(&kappa) {
                            let _ = p.reply.send(Ok(None));
                        }
                    }
                }
            }
        }
        kad::QueryResult::GetClosestPeers(res) => {
            let peers = match res {
                Ok(ok) => ok.peers,
                Err(kad::GetClosestPeersError::Timeout { peers, .. }) => peers,
            };
            let Some(pd) = pending_discover_by_id.get_mut(&id) else {
                return;
            };
            for peer in peers {
                let req_id = swarm.behaviour_mut().cas.send_request(
                    &peer.peer_id,
                    Req::List {
                        prefix: pd.prefix.clone(),
                        limit: pd.limit as u32,
                    },
                );
                pd.outstanding += 1;
                discover_inflight_to_query.insert(req_id, id);
            }
            if is_last {
                pd.peers_more = false;
                finalize_discover_if_idle(id, pending_discover_by_id);
            }
        }
        _ => {}
    }
}

fn finalize_discover_if_idle(
    qid: kad::QueryId,
    pending_discover_by_id: &mut HashMap<kad::QueryId, PendingDiscover>,
) {
    let Some(pd) = pending_discover_by_id.get(&qid) else {
        return;
    };
    if pd.outstanding == 0 && !pd.peers_more {
        if let Some(pd) = pending_discover_by_id.remove(&qid) {
            let mut out = pd.results;
            out.truncate(pd.limit);
            let _ = pd.reply.send(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_store_mem::MemKappaStore;
    use hologram_substrate_core::address_bytes;

    #[tokio::test]
    async fn kad_announce_then_fetch_via_provider_discovery() {
        // B has the data and announces κ to the DHT; A bootstraps off B and fetches κ
        // **without** ever calling add_peer for κ's holder by-κ — the DHT provider lookup resolves
        // it. Coordinator-free content discovery (NW class extended).
        let store_b = Arc::new(MemKappaStore::new());
        let kappa = store_b.put("blake3", b"resolved-via-kademlia").unwrap();
        let peer_b = LibPeer::new(store_b).await;

        let store_a = Arc::new(MemKappaStore::new());
        let peer_a = LibPeer::new(store_a).await;

        // A bootstraps off B (one peer in A's routing table). A does NOT call add_peer per κ.
        peer_a.add_peer(&peer_b.addr().to_string()).await.unwrap();
        // B announces κ as a provider on the DHT.
        peer_b.announce(&kappa).await;
        // Give the DHT a moment to settle the providing record + connectivity.
        tokio::time::sleep(Duration::from_millis(300)).await;

        let got = peer_a.fetch(&kappa).await.unwrap();
        assert_eq!(got.unwrap().as_ref(), b"resolved-via-kademlia");

        // κ the network doesn't have → Ok(None).
        let absent = address_bytes(b"absent-from-network");
        assert_eq!(peer_a.fetch(&absent).await.unwrap(), None);
    }

    #[tokio::test]
    async fn kad_discover_returns_keys_from_closest_peers() {
        // B has three κ-labels; A bootstraps off B and discovers them via get_closest_peers + List
        // RR — no prior knowledge of B's contents.
        let store_b = Arc::new(MemKappaStore::new());
        let k1 = store_b.put("blake3", b"item-1").unwrap();
        let _k2 = store_b.put("blake3", b"item-2").unwrap();
        let _k3 = store_b.put("blake3", b"item-3").unwrap();
        let peer_b = LibPeer::new(store_b).await;

        let store_a = Arc::new(MemKappaStore::new());
        let peer_a = LibPeer::new(store_a).await;
        peer_a.add_peer(&peer_b.addr().to_string()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;

        // Empty-prefix discover with high limit returns the union of every peer's iterate().
        let found = peer_a.discover(None, 32).await;
        assert!(
            found.iter().any(|k| k == &k1),
            "discover must surface κ that B holds"
        );
    }
}
