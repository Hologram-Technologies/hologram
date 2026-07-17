//! # hologram-net-tcp
//!
//! **uor-native TCP `KappaSync`** for std hosts. Content-addressed networking with **no PeerIds,
//! no Multiaddrs, no second naming surface** (SPINE-1) — every peer is identified by the
//! κ-label of its [`PeerEndpoint`] realization, every routed key is a κ-label, every fetched
//! byte is σ-axis re-derived on receipt (SPINE-4). Replaces `hologram-net-libp2p`, whose
//! PeerId + Multiaddr layer was the structural reason that crate was *not* uor-native.
//!
//! ## What this crate is
//!
//! - **Transport**: raw TCP. No Noise handshake (content integrity is provided by σ-axis
//!   verify-on-receipt at the application layer; transport encryption is a separate, future
//!   concern that can be added by wrapping the `TcpStream` without changing the protocol).
//! - **Framing**: length-prefixed `u32 LE len | u8 kind | payload` (same shape as
//!   `hologram-net-bare`'s wire format).
//! - **Peer identity**: κ of a [`PeerEndpoint`] realization. No public keys, no PeerIds.
//! - **DHT**: a κ-XOR Kademlia (256 k-buckets, K=20 peers per bucket) over **κ keys** —
//!   `announce(κ)` issues a `Provide` to the K peers nearest κ; `discover` does the standard
//!   Kademlia `find_node` + `get_providers` walk.
//! - **`KappaSync` trait**: `fetch` / `announce` / `discover` / `add_peer` / `add_gateway`.
//!   `add_peer` accepts `host:port` syntax (resolved into a `PeerEndpoint` κ and bootstrapped
//!   into the routing table); `add_gateway` is a no-op success only for symmetry with the
//!   trait — TCP-CAS knows no URL surface.
//!
//! ## What this crate is NOT
//!
//! - **Not** a libp2p compatible peer. There is no Multiaddr, no Noise XX handshake, no
//!   PeerId on the wire — by design.
//! - **Not** a substitute for confidentiality where required. SPINE-4 gives integrity; if a
//!   deployment needs confidentiality, wrap the transport in TLS — the wire format is layered
//!   so this is additive.

use std::collections::HashMap;
use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;
use hologram_space::PeerEndpoint;
use hologram_space::{
    address_bytes, verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaStore, KappaSync,
    Realization, SyncError,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex as AsyncMutex;
use tokio::time::{timeout, Duration};

mod dht;
mod wire;

pub use dht::RoutingTable;
pub use wire::{decode_frame, encode_frame, Kind};

// ── Configuration ───────────────────────────────────────────────────────────

/// Kademlia bucket capacity. K=20 is the canonical Kademlia choice from the original paper;
/// this is the **protocol parameter**, not an arbitrary cap — the routing table's design
/// degrades gracefully if a bucket fills.
pub const K: usize = 20;

/// Kademlia concurrency factor. α=3 — protocol parameter (Kademlia paper).
pub const ALPHA: usize = 3;

/// Per-instance configuration — every value an operator might reasonably tune. Defaults are
/// labeled at use sites; none are hardcoded into the protocol itself.
#[derive(Clone, Debug)]
pub struct TcpConfig {
    /// Per-RPC timeout for outbound DHT queries. `None` ⇒ no timeout (bounded only by the
    /// network / the caller's own future-timeout). Defaults to `Some(Duration::from_secs(5))`,
    /// but the operator should set whatever fits their deployment — there is **no policy
    /// constant** baked into the wire (SPINE-6).
    pub rpc_timeout: Option<Duration>,
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            rpc_timeout: Some(Duration::from_secs(5)),
        }
    }
}

// ── TcpKappaSync ────────────────────────────────────────────────────────────

/// A peer entry in the routing table — its identity κ + its TCP endpoint (so we can dial it).
#[derive(Clone, Debug)]
pub struct Peer {
    pub id: KappaLabel71,
    pub addr: SocketAddr,
}

impl Peer {
    /// Construct a peer from a socket address; identity κ = `address_bytes(PeerEndpoint(addr))`.
    /// Both IPv4 and IPv6 are first-class — the proto byte in the `PeerEndpoint` payload
    /// distinguishes them, and the κ is derived from the proto-tagged bytes (so the same
    /// host:port at v4 and v6 yields distinct identity κs, by design).
    pub fn from_addr(addr: SocketAddr) -> Self {
        let endpoint = match addr.ip() {
            IpAddr::V4(v4) => PeerEndpoint::tcp4(v4.octets(), addr.port()),
            IpAddr::V6(v6) => PeerEndpoint::tcp6(v6.octets(), addr.port()),
        };
        let id = address_bytes(&endpoint.canonicalize());
        Self { id, addr }
    }
}

/// Serialize a peer's transport endpoint for wire framing (Provide / FindNodeRes / GetProvidersRes).
fn endpoint_payload(addr: SocketAddr) -> Vec<u8> {
    match addr.ip() {
        IpAddr::V4(v4) => PeerEndpoint::tcp4(v4.octets(), addr.port())
            .transport_payload
            .clone(),
        IpAddr::V6(v6) => PeerEndpoint::tcp6(v6.octets(), addr.port())
            .transport_payload
            .clone(),
    }
}

/// Parse the variable-length endpoint payload starting at `bytes`. Returns
/// `(SocketAddr, bytes_consumed)`. Looks at `bytes[0]` to size the payload — supports both
/// IPv4 (7 bytes) and IPv6 (19 bytes). Returns `None` if the proto is unknown or the slice
/// is too short.
fn parse_endpoint_payload(bytes: &[u8]) -> Option<(SocketAddr, usize)> {
    let proto = *bytes.first()?;
    let size = PeerEndpoint::payload_size_for_proto(proto)?;
    if bytes.len() < size {
        return None;
    }
    let addr = match proto {
        PeerEndpoint::PROTO_TCP4 => {
            let (host, port) = PeerEndpoint::parse_tcp4(&bytes[..size])?;
            SocketAddr::new(IpAddr::V4(Ipv4Addr::from(host)), port)
        }
        PeerEndpoint::PROTO_TCP6 => {
            let (host, port) = PeerEndpoint::parse_tcp6(&bytes[..size])?;
            SocketAddr::new(IpAddr::V6(std::net::Ipv6Addr::from(host)), port)
        }
        _ => return None,
    };
    Some((addr, size))
}

/// uor-native TCP `KappaSync` implementation. Owns the routing table, a local store handle for
/// answering inbound fetches, and the listener task.
pub struct TcpKappaSync {
    local_id: KappaLabel71,
    listen_addr: SocketAddr,
    store: Arc<dyn KappaStore>,
    config: TcpConfig,
    routing: Arc<AsyncMutex<RoutingTable>>,
    /// Provider records held *for* other peers — `content_κ → set of (provider id, addr)`.
    /// Standard Kademlia: peers nearest a κ in XOR space store its provider records.
    providers: Arc<AsyncMutex<HashMap<[u8; 71], Vec<Peer>>>>,
    /// Outbound dialer cache — open `TcpStream`s per peer addr, so repeated fetches reuse a
    /// connection. Eviction is by Drop when the entry is removed.
    dialer: Arc<AsyncMutex<HashMap<SocketAddr, Arc<AsyncMutex<TcpStream>>>>>,
}

impl TcpKappaSync {
    /// Bind a TCP listener with the default [`TcpConfig`]. See [`bind_with_config`] for
    /// operator-tuned configuration.
    ///
    /// [`bind_with_config`]: TcpKappaSync::bind_with_config
    pub async fn bind(
        addr: SocketAddr,
        store: Arc<dyn KappaStore>,
    ) -> Result<Arc<Self>, SyncError> {
        Self::bind_with_config(addr, store, TcpConfig::default()).await
    }

    /// Bind with an explicit [`TcpConfig`] — the operator controls every tunable; no policy
    /// constants are baked into the wire protocol (SPINE-6).
    pub async fn bind_with_config(
        addr: SocketAddr,
        store: Arc<dyn KappaStore>,
        config: TcpConfig,
    ) -> Result<Arc<Self>, SyncError> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|_| SyncError::BackendFailure("bind"))?;
        let listen_addr = listener
            .local_addr()
            .map_err(|_| SyncError::BackendFailure("local_addr"))?;
        let local = Peer::from_addr(listen_addr);
        let routing = Arc::new(AsyncMutex::new(RoutingTable::new(local.id)));
        let providers = Arc::new(AsyncMutex::new(HashMap::new()));
        let dialer = Arc::new(AsyncMutex::new(HashMap::new()));
        let this = Arc::new(Self {
            local_id: local.id,
            listen_addr,
            store,
            config,
            routing,
            providers,
            dialer,
        });
        // Spawn the accept loop.
        let inbound = this.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let inbound = inbound.clone();
                tokio::spawn(async move {
                    let _ = inbound.handle_connection(stream).await;
                });
            }
        });
        Ok(this)
    }

    /// Our local peer identity κ.
    pub fn local_id(&self) -> KappaLabel71 {
        self.local_id
    }

    /// The TCP address we're listening on.
    pub fn local_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// Bootstrap by adding a known peer. The peer's identity κ is derived from its socket
    /// address (the κ of its `PeerEndpoint` realization); we record it in the routing table
    /// and immediately ping its bucket via a `find_node` for our own id (the standard
    /// Kademlia bootstrap step).
    pub async fn add_peer_addr(&self, addr: SocketAddr) -> Result<KappaLabel71, SyncError> {
        let peer = Peer::from_addr(addr);
        self.routing.lock().await.insert(peer.clone());
        // Kademlia bootstrap: query the bootstrap peer for nodes close to us.
        let _ = self.do_find_node(&peer, self.local_id).await;
        Ok(peer.id)
    }

    // ── inbound frame handling ──────────────────────────────────────────────

    async fn handle_connection(&self, mut stream: TcpStream) -> Result<(), SyncError> {
        // One request per connection (simpler than multiplexing; the dialer cache reuses the
        // open socket for back-to-back queries when needed).
        let mut buf = Vec::with_capacity(64);
        let mut tmp = [0u8; 4096];
        loop {
            let n = match stream.read(&mut tmp).await {
                Ok(0) => return Ok(()),
                Ok(n) => n,
                Err(_) => return Err(SyncError::BackendFailure("inbound-read")),
            };
            buf.extend_from_slice(&tmp[..n]);
            while let Some((kind, payload, consumed)) = decode_frame(&buf) {
                self.handle_frame(&mut stream, kind, payload.to_vec())
                    .await?;
                buf.drain(..consumed);
            }
        }
    }

    async fn handle_frame(
        &self,
        stream: &mut TcpStream,
        kind: u8,
        payload: Vec<u8>,
    ) -> Result<(), SyncError> {
        // Wire-version handshake (spec 04 §Protocol hardening). Additive + backward-compatible: a
        // peer that opens with a HELLO gets our HELLO back and a negotiated version; a peer that
        // starts straight into a DHT/fetch frame (no HELLO) is unaffected. An incompatible or
        // malformed HELLO closes the connection — refuse, never a silent downgrade.
        if kind == crate::bare::KIND_HELLO {
            use crate::protocol::WireVersionRange;
            match WireVersionRange::decode(&payload) {
                Some(peer) if WireVersionRange::CURRENT.negotiate(peer).is_some() => {
                    stream
                        .write_all(&crate::bare::hello_frame(WireVersionRange::CURRENT))
                        .await
                        .ok();
                    return Ok(());
                }
                _ => return Err(SyncError::BackendFailure("incompatible wire version")),
            }
        }
        match Kind::from_u8(kind) {
            Some(Kind::FetchReq) => {
                if payload.len() != 71 {
                    return Ok(());
                }
                let arr: [u8; 71] = payload[..71].try_into().unwrap();
                let label =
                    KappaLabel::from_bytes(&arr).map_err(|_| SyncError::VerificationFailed)?;
                let resp = match self.store.get(&label) {
                    Ok(Some(bytes)) => {
                        let mut p = Vec::with_capacity(71 + bytes.len());
                        p.extend_from_slice(&arr);
                        p.extend_from_slice(bytes.as_ref());
                        encode_frame(Kind::FetchResOk as u8, &p)
                    }
                    _ => encode_frame(Kind::FetchRes404 as u8, &arr),
                };
                stream.write_all(&resp).await.ok();
            }
            Some(Kind::Announce) => {
                if payload.len() != 71 {
                    return Ok(());
                }
                // An announce from a peer means "I have this κ" — record a provider entry.
                // The peer's identity is derivable from the connection's remote addr.
                if let Ok(peer_addr) = stream.peer_addr() {
                    let provider = Peer::from_addr(peer_addr);
                    let mut p = self.providers.lock().await;
                    let arr: [u8; 71] = payload[..71].try_into().unwrap();
                    let entry = p.entry(arr).or_default();
                    if !entry.iter().any(|x| x.id == provider.id) {
                        entry.push(provider);
                    }
                }
            }
            Some(Kind::FindNodeReq) => {
                if payload.len() != 71 {
                    return Ok(());
                }
                let arr: [u8; 71] = payload[..71].try_into().unwrap();
                let closest = self.routing.lock().await.k_closest(&arr, K);
                let resp = encode_peer_list(Kind::FindNodeRes, &closest);
                stream.write_all(&resp).await.ok();
            }
            Some(Kind::GetProvidersReq) => {
                if payload.len() != 71 {
                    return Ok(());
                }
                let arr: [u8; 71] = payload[..71].try_into().unwrap();
                let providers = self
                    .providers
                    .lock()
                    .await
                    .get(&arr)
                    .cloned()
                    .unwrap_or_default();
                let resp = encode_peer_list(Kind::GetProvidersRes, &providers);
                stream.write_all(&resp).await.ok();
            }
            Some(Kind::Provide) => {
                // payload: content_κ (71 bytes) | endpoint (variable: tcp4=7, tcp6=19)
                if payload.len() < 71 + 1 {
                    return Ok(());
                }
                let mut arr = [0u8; 71];
                arr.copy_from_slice(&payload[..71]);
                let Some((addr, _)) = parse_endpoint_payload(&payload[71..]) else {
                    return Ok(()); // unknown proto / malformed — drop silently (forward-compat)
                };
                let provider = Peer::from_addr(addr);
                let mut p = self.providers.lock().await;
                let entry = p.entry(arr).or_default();
                if !entry.iter().any(|x| x.id == provider.id) {
                    entry.push(provider);
                }
            }
            // Inbound responses arrive on the *outbound* socket (the caller's), not on the
            // listener path — so seeing a *_RES on this side means the peer is misbehaving.
            // Drop silently (forward-compat: future kinds may appear here).
            _ => {}
        }
        // Record the peer in the routing table for any inbound contact.
        if let Ok(peer_addr) = stream.peer_addr() {
            let peer = Peer::from_addr(peer_addr);
            self.routing.lock().await.insert(peer);
        }
        Ok(())
    }

    // ── outbound RPCs ───────────────────────────────────────────────────────

    /// The outbound half of the connect handshake (spec 04 §Protocol hardening): send our HELLO,
    /// read the peer's, negotiate. Runs once per new connection, before any request frame; an
    /// incompatible peer returns an error so the dial is aborted (never a silent downgrade). A peer
    /// on the shared `bare`/`tcp` protocol answers a HELLO with its own HELLO (see
    /// `handle_frame`).
    async fn dialer_handshake(&self, stream: &mut TcpStream) -> Result<(), SyncError> {
        use crate::protocol::WireVersionRange;
        stream
            .write_all(&crate::bare::hello_frame(WireVersionRange::CURRENT))
            .await
            .map_err(|_| SyncError::BackendFailure("handshake-write"))?;
        let mut buf = Vec::with_capacity(16);
        let mut tmp = [0u8; 256];
        loop {
            let n = match self.config.rpc_timeout {
                Some(d) => timeout(d, stream.read(&mut tmp))
                    .await
                    .map_err(|_| SyncError::BackendFailure("handshake-timeout"))?
                    .map_err(|_| SyncError::BackendFailure("handshake-read"))?,
                None => stream
                    .read(&mut tmp)
                    .await
                    .map_err(|_| SyncError::BackendFailure("handshake-read"))?,
            };
            if n == 0 {
                return Err(SyncError::BackendFailure("handshake-eof"));
            }
            buf.extend_from_slice(&tmp[..n]);
            if decode_frame(&buf).is_some() {
                break;
            }
        }
        crate::bare::negotiate_from_hello(WireVersionRange::CURRENT, &buf)
            .map(|_| ())
            .map_err(|_| SyncError::BackendFailure("incompatible wire version"))
    }

    /// Send a single frame to `peer`, await one response frame.
    async fn rpc(&self, peer: &Peer, frame: Vec<u8>) -> Result<(u8, Vec<u8>), SyncError> {
        let stream_arc = {
            let mut d = self.dialer.lock().await;
            match d.get(&peer.addr).cloned() {
                Some(s) => s,
                None => {
                    let mut s = TcpStream::connect(peer.addr)
                        .await
                        .map_err(|_| SyncError::BackendFailure("dial"))?;
                    // Dialer-side wire-version handshake (spec 04): negotiate once, before any
                    // request; an incompatible peer aborts the dial (refuse, no silent downgrade).
                    self.dialer_handshake(&mut s).await?;
                    let s = Arc::new(AsyncMutex::new(s));
                    d.insert(peer.addr, s.clone());
                    s
                }
            }
        };
        let mut stream = stream_arc.lock().await;
        if stream.write_all(&frame).await.is_err() {
            // Stale connection — drop the cache entry and let the caller retry by re-dialing.
            self.dialer.lock().await.remove(&peer.addr);
            return Err(SyncError::BackendFailure("rpc-write"));
        }
        let mut buf = Vec::with_capacity(128);
        let mut tmp = [0u8; 4096];
        loop {
            let read_result = match self.config.rpc_timeout {
                Some(d) => match timeout(d, stream.read(&mut tmp)).await {
                    Ok(r) => r,
                    Err(_) => {
                        self.dialer.lock().await.remove(&peer.addr);
                        return Err(SyncError::BackendFailure("rpc-timeout"));
                    }
                },
                // No operator-set timeout — bounded only by the network / future-drop.
                None => stream.read(&mut tmp).await,
            };
            let n = match read_result {
                Ok(0) => return Err(SyncError::BackendFailure("rpc-eof")),
                Ok(n) => n,
                Err(_) => {
                    self.dialer.lock().await.remove(&peer.addr);
                    return Err(SyncError::BackendFailure("rpc-read"));
                }
            };
            buf.extend_from_slice(&tmp[..n]);
            if let Some((kind, payload, _)) = decode_frame(&buf) {
                return Ok((kind, payload.to_vec()));
            }
        }
    }

    async fn do_find_node(
        &self,
        peer: &Peer,
        target: KappaLabel71,
    ) -> Result<Vec<Peer>, SyncError> {
        let frame = encode_frame(Kind::FindNodeReq as u8, target.as_array());
        let (kind, payload) = self.rpc(peer, frame).await?;
        if kind != Kind::FindNodeRes as u8 {
            return Err(SyncError::BackendFailure("find-node-kind"));
        }
        let peers = decode_peer_list(&payload);
        for p in &peers {
            self.routing.lock().await.insert(p.clone());
        }
        Ok(peers)
    }

    async fn do_get_providers(
        &self,
        peer: &Peer,
        key: KappaLabel71,
    ) -> Result<Vec<Peer>, SyncError> {
        let frame = encode_frame(Kind::GetProvidersReq as u8, key.as_array());
        let (kind, payload) = self.rpc(peer, frame).await?;
        if kind != Kind::GetProvidersRes as u8 {
            return Err(SyncError::BackendFailure("get-providers-kind"));
        }
        Ok(decode_peer_list(&payload))
    }

    /// Iterative Kademlia lookup for `target`. Returns the K closest peers we discovered.
    async fn kademlia_find(&self, target: KappaLabel71) -> Vec<Peer> {
        let mut closest: Vec<Peer> = self.routing.lock().await.k_closest(target.as_array(), K);
        let mut queried: HashSet<[u8; 71]> = HashSet::new();
        loop {
            // Pick α closest unqueried peers.
            let mut to_query: Vec<Peer> = Vec::new();
            for p in &closest {
                if to_query.len() == ALPHA {
                    break;
                }
                if !queried.contains(p.id.as_array()) {
                    to_query.push(p.clone());
                }
            }
            if to_query.is_empty() {
                break;
            }
            for p in &to_query {
                queried.insert(*p.id.as_array());
            }
            // Query them sequentially (the routing-table mutex makes parallel queries trickier
            // and α=3 keeps the round-count small). Merge new peers into `closest`.
            let mut new_seen = false;
            for peer in to_query {
                if let Ok(peers) = self.do_find_node(&peer, target).await {
                    for p in peers {
                        if !closest.iter().any(|x| x.id == p.id) {
                            closest.push(p);
                            new_seen = true;
                        }
                    }
                }
            }
            // Re-rank by distance and trim to K.
            closest.sort_by_key(|p| dht::xor_distance(p.id.as_array(), target.as_array()));
            closest.truncate(K);
            if !new_seen {
                break;
            }
        }
        closest
    }
}

#[async_trait]
impl KappaSync for TcpKappaSync {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        // 1. Local hit short-circuits.
        if let Ok(Some(b)) = self.store.get(kappa) {
            return Ok(Some(b));
        }
        // 2. Find K closest peers to κ; ask each for providers.
        let closest = self.kademlia_find(*kappa).await;
        let mut providers: Vec<Peer> = Vec::new();
        for p in &closest {
            if let Ok(prov) = self.do_get_providers(p, *kappa).await {
                for q in prov {
                    if !providers.iter().any(|x| x.id == q.id) {
                        providers.push(q);
                    }
                }
            }
        }
        // 3. Also try the closest peers themselves — they may serve the κ directly even if no
        //    provider record exists for it.
        let try_order: Vec<Peer> = providers.into_iter().chain(closest).collect();
        for p in try_order {
            let frame = encode_frame(Kind::FetchReq as u8, kappa.as_array());
            let Ok((kind, payload)) = self.rpc(&p, frame).await else {
                continue;
            };
            if kind == Kind::FetchResOk as u8 && payload.len() >= 71 {
                let bytes = &payload[71..];
                if verify_kappa(bytes, kappa).unwrap_or(false) {
                    let arc: Bytes = Arc::<[u8]>::from(bytes);
                    // Cache locally (verify-on-receipt complete).
                    let _ = self.store.put("blake3", bytes);
                    return Ok(Some(arc));
                }
                // Forged response — skip this peer, try the next.
            }
            // 404 or wrong kind — try next.
        }
        Ok(None)
    }

    async fn announce(&self, kappa: &KappaLabel71) {
        // Provide-to-K: send our endpoint as the provider record to the K closest peers to κ.
        // The endpoint payload is variable-length (tcp4=7 / tcp6=19), proto byte distinguishes.
        let closest = self.kademlia_find(*kappa).await;
        let local_addr_bytes = endpoint_payload(self.listen_addr);
        let mut payload = Vec::with_capacity(71 + local_addr_bytes.len());
        payload.extend_from_slice(kappa.as_array());
        payload.extend_from_slice(&local_addr_bytes);
        let frame = encode_frame(Kind::Provide as u8, &payload);
        for peer in closest {
            let Ok(mut s) = TcpStream::connect(peer.addr).await else {
                continue;
            };
            let _ = s.write_all(&frame).await;
            // No response expected for Provide.
        }
    }

    async fn discover(&self, prefix: Option<&[u8]>, limit: usize) -> Vec<KappaLabel71> {
        // The Kademlia view of "discover near a prefix": treat the prefix as the high bytes of
        // a virtual target κ, find_node toward it, return the closest peers' provider records.
        // If the caller passed garbage that doesn't parse as a κ-label, return empty — no
        // silent fallback to our own id (SPINE-6).
        let target = match prefix {
            Some(p) if !p.is_empty() => {
                let mut arr = [0u8; 71];
                let n = p.len().min(71);
                arr[..n].copy_from_slice(&p[..n]);
                match KappaLabel::from_bytes(&arr) {
                    Ok(k) => k,
                    Err(_) => return Vec::new(),
                }
            }
            // No prefix ⇒ explore the routing table around our own id (standard Kademlia idiom).
            _ => self.local_id,
        };
        let closest = self.kademlia_find(target).await;
        closest.into_iter().take(limit).map(|p| p.id).collect()
    }

    async fn add_peer(&self, host_port: &str) -> Result<(), SyncError> {
        // Parse `host:port` — the uor-native peer-bootstrap form. No Multiaddr, no `/p2p/<PeerId>`
        // suffix. The peer's identity κ is derived from its endpoint (SPINE-1).
        let addr: SocketAddr = host_port
            .parse()
            .map_err(|_| SyncError::BackendFailure("addr-parse"))?;
        self.add_peer_addr(addr).await.map(|_| ())
    }

    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        // TCP-CAS has no URL surface. HTTP gateways live behind `hologram-net-http`; this
        // crate's `KappaSync` rejects URL bootstrap to keep the failure mode explicit.
        Err(SyncError::NotEnabled)
    }
}

// ── peer-list codec ─────────────────────────────────────────────────────────

/// Encode a list of peers as `u32 LE count | (κ_71 + endpoint_payload)*` framed by `kind`.
/// Endpoint payloads are variable length (tcp4 = 7 bytes, tcp6 = 19 bytes); the proto byte at
/// the start of each endpoint distinguishes.
fn encode_peer_list(kind: Kind, peers: &[Peer]) -> Vec<u8> {
    // Rough capacity estimate (tcp4-sized); the codec auto-grows for tcp6.
    let mut payload = Vec::with_capacity(4 + peers.len() * (71 + 7));
    payload.extend_from_slice(&(peers.len() as u32).to_le_bytes());
    for p in peers {
        payload.extend_from_slice(p.id.as_array());
        payload.extend_from_slice(&endpoint_payload(p.addr));
    }
    encode_frame(kind as u8, &payload)
}

fn decode_peer_list(payload: &[u8]) -> Vec<Peer> {
    if payload.len() < 4 {
        return Vec::new();
    }
    let n = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut off = 4;
    for _ in 0..n {
        if off + 71 + 1 > payload.len() {
            break;
        }
        let mut id = [0u8; 71];
        id.copy_from_slice(&payload[off..off + 71]);
        off += 71;
        let Some((addr, consumed)) = parse_endpoint_payload(&payload[off..]) else {
            // Unknown proto: we can't know how many bytes to skip; drop the rest as
            // forward-compat (SPINE-5).
            break;
        };
        off += consumed;
        if let Ok(label) = KappaLabel::from_bytes(&id) {
            out.push(Peer { id: label, addr });
        }
    }
    out
}
