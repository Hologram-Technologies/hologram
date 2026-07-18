//! # QUIC transport (spec 04) — encrypted P2P over QUIC.
//!
//! The modern, dependency-light P2P substrate: [quinn](https://docs.rs/quinn) (QUIC over TLS 1.3)
//! carrying the *same* `u32 LE len | u8 kind | payload` frame protocol and wire-version handshake as
//! [`crate::bare`] / `crate::tcp`. A [`QuicPeer`] both serves inbound fetches and dials peers from
//! one endpoint; every fetched byte is re-derived through the σ-axis at the receiver (SPINE-4).
//!
//! This is the transport primitive **iroh** layers relay + NAT-traversal onto. Full iroh is
//! currently unreachable in this workspace: modern iroh requires `blake3 1.8`, but the κ core pins
//! `blake3 1.5` (via `uor-prism-crypto`) — an irreconcilable conflict, and the only iroh version
//! that resolves (0.28) pulls ~291 packages with an outdated API. quinn carries no `blake3`, so it
//! drops in cleanly (14 packages) and gives the encrypted-P2P substrate directly.
//!
//! ## Why a skip-verify TLS client is correct here
//! QUIC's TLS gives an encrypted, integrity-protected **channel**; it does *not* need PKI because
//! **content is content-addressed**. Every response is re-derived (`verify_kappa`) at the receiver:
//! a forging responder is rejected no matter which certificate terminated the tunnel. So the peer
//! uses a self-signed transport cert plus a skip-verify client verifier — confidentiality from TLS,
//! integrity from κ. A space that also wants peer *authentication* binds it at the κ layer (the
//! `AttestationKey` / signed handshake), never via TLS PKI (SPINE-1: κ is the only identity).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use hologram_space::{verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaSync, SyncError};
use quinn::{ClientConfig, Connection, Endpoint, ServerConfig};

use crate::bare::{
    decode_frame, encode_frame, hello_frame, negotiate_from_hello, KIND_FETCH_REQ,
    KIND_FETCH_RES_404, KIND_FETCH_RES_OK,
};
use crate::protocol::WireVersionRange;

/// The resolver hook: given a κ, produce the canonical bytes if locally available (the substrate
/// wires this to its `KappaStore::get`). Same shape as [`crate::bare::LocalResolver`].
pub type LocalResolver = crate::bare::LocalResolver;

/// Upper bound on a single request/response body read from a QUIC stream. One fetch carries a HELLO
/// frame plus one content frame; 64 MiB caps a hostile peer's memory pressure without truncating any
/// realistic realization (a codemodule/tensor layer is chunked into per-κ blobs upstream).
const MAX_BODY: usize = 64 * 1024 * 1024;

/// A QUIC peer: one bound endpoint that both **serves** inbound fetches (from a resolver) and
/// **dials** other peers. Implements [`KappaSync`]: a `fetch(κ)` short-circuits on a local hit, then
/// routes to explicitly-joined peers in order. Cheap to clone (the handles are `Arc`-backed).
#[derive(Clone)]
pub struct QuicPeer {
    endpoint: Endpoint,
    range: WireVersionRange,
    /// Local content resolver — a `KappaSync::fetch` checks this before any network round-trip.
    resolver: LocalResolver,
    /// Explicitly-joined peers the `KappaSync::fetch` impl routes to, in join order. QUIC is a
    /// direct-dial transport (κ is the only identity — no DHT here): peers are joined via
    /// [`QuicPeer::join`] / `add_peer`, never gossiped or discovered.
    peers: Arc<Mutex<Vec<SocketAddr>>>,
}

impl QuicPeer {
    /// Bind a QUIC endpoint on `addr` (use `0` for the port to get an ephemeral one) that answers
    /// fetches from `resolver`, and spawn its accept loop. The endpoint can also dial peers via
    /// [`QuicPeer::fetch_from`].
    pub fn bind(addr: SocketAddr, resolver: LocalResolver) -> Result<Self, SyncError> {
        ensure_crypto_provider();
        let server_config =
            server_config().map_err(|_| SyncError::BackendFailure("quic-tls-server"))?;
        let mut endpoint =
            Endpoint::server(server_config, addr).map_err(|_| SyncError::BackendFailure("quic-bind"))?;
        endpoint.set_default_client_config(
            client_config().map_err(|_| SyncError::BackendFailure("quic-tls-client"))?,
        );
        let range = WireVersionRange::CURRENT;
        // Serve inbound connections in the background — one task per accepted connection, one task
        // per bi-stream. Fetches are stateless per stream, so this scales without shared locks.
        tokio::spawn(serve_endpoint(endpoint.clone(), resolver.clone(), range));
        Ok(Self {
            endpoint,
            range,
            resolver,
            peers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// The bound local address (resolves the ephemeral port when bound to `:0`).
    pub fn local_addr(&self) -> Result<SocketAddr, SyncError> {
        self.endpoint
            .local_addr()
            .map_err(|_| SyncError::BackendFailure("quic-addr"))
    }

    /// Join `peer` to this node's routing set — a subsequent `KappaSync::fetch` tries it (in join
    /// order) after a local miss. Idempotent: a peer already present is not added twice.
    pub fn join(&self, peer: SocketAddr) {
        let mut peers = self.peers.lock().expect("peer table poisoned");
        if !peers.contains(&peer) {
            peers.push(peer);
        }
    }

    /// A snapshot of the currently-joined peers, in routing order.
    pub fn peers(&self) -> Vec<SocketAddr> {
        self.peers.lock().expect("peer table poisoned").clone()
    }

    /// Fetch `kappa` from a specific peer over QUIC, verifying on receipt (SPINE-4). Returns
    /// `Ok(None)` for a resolved-absent (404) response and `Err(VerificationFailed)` if the peer
    /// returns content whose κ does not match — a forging responder is never trusted.
    pub async fn fetch_from(
        &self,
        peer: SocketAddr,
        kappa: &KappaLabel71,
    ) -> Result<Option<Bytes>, SyncError> {
        let conn = self
            .endpoint
            .connect(peer, "localhost")
            .map_err(|_| SyncError::BackendFailure("quic-connect"))?
            .await
            .map_err(|_| SyncError::BackendFailure("quic-handshake"))?;
        let res = self.fetch_over(&conn, kappa).await;
        conn.close(0u32.into(), b"done");
        res
    }

    async fn fetch_over(
        &self,
        conn: &Connection,
        kappa: &KappaLabel71,
    ) -> Result<Option<Bytes>, SyncError> {
        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|_| SyncError::BackendFailure("quic-open-bi"))?;
        // Request = wire-version HELLO ++ FETCH_REQ(κ). The handshake rides the same stream so a
        // stateless responder negotiates and serves in one round trip.
        let mut req = hello_frame(self.range);
        req.extend_from_slice(&encode_frame(KIND_FETCH_REQ, kappa.as_array()));
        send.write_all(&req)
            .await
            .map_err(|_| SyncError::BackendFailure("quic-write"))?;
        send.finish()
            .map_err(|_| SyncError::BackendFailure("quic-finish"))?;
        let buf = recv
            .read_to_end(MAX_BODY)
            .await
            .map_err(|_| SyncError::BackendFailure("quic-read"))?;
        parse_fetch_response(self.range, &buf, kappa)
    }
}

impl core::fmt::Debug for QuicPeer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuicPeer")
            .field("local_addr", &self.endpoint.local_addr().ok())
            .field("peers", &self.peers())
            .finish()
    }
}

// A `KappaSync` over explicitly-joined QUIC peers. `quic` is a std/native-only feature, so the
// trait's native `Send` futures apply — plain `#[async_trait]` (never the `?Send` wasm variant).
#[async_trait]
impl KappaSync for QuicPeer {
    async fn fetch(&self, kappa: &KappaLabel71) -> Result<Option<Bytes>, SyncError> {
        // A local hit resolves without any network round-trip.
        if let Some(bytes) = (self.resolver)(kappa) {
            return Ok(Some(bytes));
        }
        // Snapshot the routing set and drop the lock *before* awaiting (never hold a std mutex
        // across `.await`). Try peers in join order; the first honest hit wins. A dead or forging
        // peer is skipped — one bad peer must not blind a fetch that another peer can answer.
        for peer in self.peers() {
            match self.fetch_from(peer, kappa).await {
                Ok(Some(bytes)) => return Ok(Some(bytes)),
                Ok(None) => continue,
                Err(_) => continue,
            }
        }
        Ok(None)
    }

    async fn announce(&self, _kappa: &KappaLabel71) {
        // Best-effort no-op: the direct-dial QUIC transport has no announce channel (peers are
        // joined explicitly, not gossiped). A DHT-backed transport is where announce belongs.
    }

    async fn discover(&self, _prefix: Option<&[u8]>, _limit: usize) -> Vec<KappaLabel71> {
        // No discovery in the direct-dial model — peers are joined, content is fetched by κ.
        Vec::new()
    }

    async fn add_peer(&self, peer_addr: &str) -> Result<(), SyncError> {
        let addr: SocketAddr = peer_addr
            .parse()
            .map_err(|_| SyncError::BackendFailure("quic-bad-addr"))?;
        self.join(addr);
        Ok(())
    }

    async fn add_gateway(&self, _url: &str) -> Result<(), SyncError> {
        // No HTTP gateway surface on the QUIC transport — fail-loud (SPINE-6).
        Err(SyncError::NotEnabled)
    }
}

// ── server side ──────────────────────────────────────────────────────────────

/// Accept connections forever; spawn a per-connection task that serves each bi-stream.
async fn serve_endpoint(endpoint: Endpoint, resolver: LocalResolver, range: WireVersionRange) {
    while let Some(incoming) = endpoint.accept().await {
        let resolver = resolver.clone();
        tokio::spawn(async move {
            let Ok(conn) = incoming.await else { return };
            // Each fetch opens a fresh bi-stream; loop until the peer closes the connection.
            while let Ok((send, recv)) = conn.accept_bi().await {
                let resolver = resolver.clone();
                tokio::spawn(async move {
                    let _ = serve_stream(send, recv, resolver, range).await;
                });
            }
        });
    }
}

/// Serve one fetch stream: read HELLO ++ FETCH_REQ, negotiate (refuse if incompatible), answer
/// HELLO ++ FETCH_RES. A refused handshake drops the stream without a response (never a silent
/// downgrade — the dialer's read then errors).
async fn serve_stream(
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    resolver: LocalResolver,
    range: WireVersionRange,
) -> Result<(), SyncError> {
    let buf = recv
        .read_to_end(MAX_BODY)
        .await
        .map_err(|_| SyncError::BackendFailure("quic-read"))?;
    // Frame 1 — the peer's HELLO. `negotiate_from_hello` refuses hostile/incompatible bytes.
    let (_k, _p, n1) = decode_frame(&buf).ok_or(SyncError::BackendFailure("quic-badhello"))?;
    negotiate_from_hello(range, &buf[..n1]).map_err(|_| SyncError::BackendFailure("quic-refused"))?;
    // Frame 2 — the FETCH_REQ. Build the response: our HELLO ++ FETCH_RES.
    let (kind, payload, _n2) =
        decode_frame(&buf[n1..]).ok_or(SyncError::BackendFailure("quic-badreq"))?;
    let mut out = hello_frame(range);
    if kind == KIND_FETCH_REQ && payload.len() == 71 {
        let mut k = [0u8; 71];
        k.copy_from_slice(payload);
        if let Ok(label) = KappaLabel::from_bytes(&k) {
            match resolver(&label) {
                Some(bytes) => {
                    let mut body = Vec::with_capacity(71 + bytes.len());
                    body.extend_from_slice(&k);
                    body.extend_from_slice(bytes.as_ref());
                    out.extend_from_slice(&encode_frame(KIND_FETCH_RES_OK, &body));
                }
                None => out.extend_from_slice(&encode_frame(KIND_FETCH_RES_404, &k)),
            }
        }
    }
    send.write_all(&out)
        .await
        .map_err(|_| SyncError::BackendFailure("quic-write"))?;
    send.finish()
        .map_err(|_| SyncError::BackendFailure("quic-finish"))?;
    Ok(())
}

// ── client-side response parsing ───────────────────────────────────────────────

/// Parse a `HELLO ++ FETCH_RES` response: negotiate the peer's HELLO, then verify-on-receipt any
/// returned content. The returned κ must equal the requested κ *and* re-derive to it — a responder
/// cannot substitute other content or forge a hash (SPINE-4).
fn parse_fetch_response(
    local: WireVersionRange,
    buf: &[u8],
    requested: &KappaLabel71,
) -> Result<Option<Bytes>, SyncError> {
    let (_k, _p, n1) = decode_frame(buf).ok_or(SyncError::BackendFailure("quic-badhello"))?;
    negotiate_from_hello(local, &buf[..n1])
        .map_err(|_| SyncError::BackendFailure("quic-incompatible"))?;
    let (kind, payload, _n2) =
        decode_frame(&buf[n1..]).ok_or(SyncError::BackendFailure("quic-badres"))?;
    match kind {
        KIND_FETCH_RES_OK => {
            if payload.len() < 71 {
                return Err(SyncError::VerificationFailed);
            }
            let mut k = [0u8; 71];
            k.copy_from_slice(&payload[..71]);
            if &k != requested.as_array() {
                return Err(SyncError::VerificationFailed); // wrong κ — not what we asked for
            }
            let label = KappaLabel::from_bytes(&k).map_err(|_| SyncError::VerificationFailed)?;
            let bytes = &payload[71..];
            if verify_kappa(bytes, &label) == Ok(true) {
                Ok(Some(Arc::<[u8]>::from(bytes)))
            } else {
                Err(SyncError::VerificationFailed) // forging responder
            }
        }
        KIND_FETCH_RES_404 => Ok(None),
        _ => Err(SyncError::BackendFailure("quic-badkind")),
    }
}

// ── TLS plumbing (self-signed transport cert + skip-verify client) ─────────────

/// Install the ring `CryptoProvider` as the process default once, so rustls' builder-based configs
/// (used by quinn's `with_single_cert` / `ClientConfig::builder`) have a provider without aws-lc-rs.
fn ensure_crypto_provider() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// A fresh self-signed server config. The cert authenticates nothing hologram trusts (integrity is
/// κ, not PKI) — it only lets QUIC/TLS 1.3 establish the encrypted channel.
fn server_config() -> Result<ServerConfig, Box<dyn std::error::Error + Send + Sync>> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])?;
    let cert_der = cert.cert.der().clone();
    let key_der =
        rustls::pki_types::PrivateKeyDer::try_from(cert.key_pair.serialize_der()).map_err(|e| {
            Box::<dyn std::error::Error + Send + Sync>::from(e.to_string())
        })?;
    Ok(ServerConfig::with_single_cert(vec![cert_der], key_der)?)
}

/// A client config that skips certificate verification: content is κ-verified downstream, so the
/// transport cert need not chain to any authority (see the module docs).
fn client_config() -> Result<ClientConfig, Box<dyn std::error::Error + Send + Sync>> {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification::new()))
        .with_no_client_auth();
    Ok(ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?,
    )))
}

/// A rustls verifier that accepts any server certificate. Sound *for this transport* because κ
/// re-derivation is the real integrity check; TLS here is confidentiality only.
#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Self {
        Self(Arc::new(rustls::crypto::ring::default_provider()))
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
