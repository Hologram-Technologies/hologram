#![allow(mixed_script_confusables)]
//! NW-tcp V&V — the uor-native replacement for the prior libp2p-backed `KappaSync`.
//!
//! Every assertion here matches a property the architecture demands of the network layer:
//!   - **Identity is κ.** Peers are addressed by the κ of their `PeerEndpoint` realization;
//!     there are no PeerIds, no Multiaddrs, no second naming surface (SPINE-1).
//!   - **Content addressing is κ.** Fetch / announce / discover all carry κ-labels over a
//!     wire that is bounded by the σ-axis on every response (verify-on-receipt — SPINE-4).
//!   - **DHT is κ-XOR.** k-buckets index peers by XOR distance over the **decoded blake3
//!     digest portion** of their identity κ — exactly the Kademlia metric, but over κ-space.
//!   - **Forgery is refused.** A peer that returns bytes that don't re-derive to the
//!     requested κ has its response dropped; the caller falls through to the next peer.

use std::net::SocketAddr;
use std::sync::Arc;

use hologram_net::tcp::TcpKappaSync;
use hologram_space::PeerEndpoint;
use hologram_space::{address_bytes, KappaStore, KappaSync, Realization};
use hologram_tck::MemKappaStore;

#[tokio::test(flavor = "current_thread")]
async fn nw_tcp_peer_identity_is_kappa_of_peer_endpoint() {
    // The peer's identity κ is the content-address of its `PeerEndpoint` — no PeerIds.
    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store)
        .await
        .unwrap();
    let addr = sync.local_addr();
    let expected = address_bytes(
        &PeerEndpoint::tcp4(
            match addr.ip() {
                std::net::IpAddr::V4(v4) => v4.octets(),
                _ => panic!("v4 only in this test"),
            },
            addr.port(),
        )
        .canonicalize(),
    );
    assert_eq!(
        sync.local_id(),
        expected,
        "local peer identity κ must equal address_bytes(PeerEndpoint(local_addr))"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nw_tcp_two_node_fetch_verifies_on_receipt() {
    // A serves κ; B bootstraps off A; B.fetch resolves κ over the wire and verifies.
    let store_a = Arc::new(MemKappaStore::new());
    let payload = b"content-served-by-A";
    let k = store_a.put("blake3", payload).unwrap();

    let sync_a = TcpKappaSync::bind(
        "127.0.0.1:0".parse().unwrap(),
        store_a.clone() as Arc<dyn KappaStore>,
    )
    .await
    .unwrap();

    let store_b = Arc::new(MemKappaStore::new());
    let sync_b = TcpKappaSync::bind(
        "127.0.0.1:0".parse().unwrap(),
        store_b.clone() as Arc<dyn KappaStore>,
    )
    .await
    .unwrap();

    // Bootstrap: B knows A by host:port (NOT by Multiaddr / PeerId).
    let a_addr: SocketAddr = sync_a.local_addr();
    sync_b.add_peer(&a_addr.to_string()).await.unwrap();

    // Fetch κ over the network.
    let got = sync_b.fetch(&k).await.unwrap().expect("κ resolved");
    assert_eq!(got.as_ref(), payload);

    // SPINE-4: bytes verified on the B side and now cached locally.
    assert!(store_b.contains(&k));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nw_tcp_announce_then_get_providers_resolves_without_prior_knowledge() {
    // Three nodes: A, B, C. A announces κ. B bootstraps off A, then off C. C fetches κ — it
    // should resolve via the DHT (get_providers → fetch from A) without C having any prior
    // direct knowledge of A.
    let store_a = Arc::new(MemKappaStore::new());
    let payload = b"DHT-resolved-payload";
    let k = store_a.put("blake3", payload).unwrap();
    let sync_a = TcpKappaSync::bind(
        "127.0.0.1:0".parse().unwrap(),
        store_a.clone() as Arc<dyn KappaStore>,
    )
    .await
    .unwrap();

    let store_b = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync_b = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store_b)
        .await
        .unwrap();

    let store_c = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync_c = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store_c)
        .await
        .unwrap();

    // A announces κ (provider records propagate to B via the kademlia_find walk).
    // First A and B need to know each other for A's announce to reach B.
    sync_b
        .add_peer(&sync_a.local_addr().to_string())
        .await
        .unwrap();
    sync_a.announce(&k).await;

    // C bootstraps off B only — has no direct knowledge of A.
    sync_c
        .add_peer(&sync_b.local_addr().to_string())
        .await
        .unwrap();
    // The Kademlia find_node walk from C reaches A through B; then get_providers returns A;
    // then C fetches κ from A and verifies.
    let got = sync_c.fetch(&k).await.unwrap();
    assert!(
        got.is_some(),
        "DHT walk should resolve κ via providers without C knowing A directly"
    );
    assert_eq!(got.unwrap().as_ref(), payload);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nw_tcp_forged_response_is_refused_silent_to_caller() {
    // A "forging" peer that serves arbitrary bytes for any FetchReq — the receiver must reject
    // them on σ-axis re-derivation (SPINE-4) and fall through to None.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let forger = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let forger_addr = forger.local_addr().unwrap();
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = forger.accept().await {
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            // Parse the FetchReq κ from the frame: skip 4-byte len + 1-byte kind.
            if buf.len() < 5 + 71 {
                return;
            }
            let κ = &buf[5..5 + 71];
            // Build a forged FetchResOk: κ + bogus bytes.
            let mut payload = Vec::with_capacity(71 + 5);
            payload.extend_from_slice(κ);
            payload.extend_from_slice(b"FORGE");
            let len = (1 + payload.len()) as u32;
            let mut frame = Vec::new();
            frame.extend_from_slice(&len.to_le_bytes());
            frame.push(0x02); // Kind::FetchResOk
            frame.extend_from_slice(&payload);
            let _ = stream.write_all(&frame).await;
        }
    });

    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store)
        .await
        .unwrap();
    sync.add_peer(&forger_addr.to_string()).await.unwrap();

    // The κ we ask for resolves to "FORGE" on the forger's wire, but `address_bytes(b"FORGE")`
    // is NOT what we requested — verify_kappa fails, the response is dropped, no peers remain,
    // we get None.
    let k = address_bytes(b"authentic-content-not-forge");
    let got = sync.fetch(&k).await.unwrap();
    assert!(
        got.is_none(),
        "forged response must be rejected at the σ-axis verification (SPINE-4)"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn nw_tcp_add_gateway_is_not_enabled() {
    // TCP-CAS has no URL surface — `add_gateway` must fail-loud.
    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store)
        .await
        .unwrap();
    let err = sync.add_gateway("http://example.invalid").await;
    assert!(matches!(err, Err(hologram_space::SyncError::NotEnabled)));
}

#[tokio::test(flavor = "current_thread")]
async fn nw_tcp_ipv4_and_ipv6_endpoints_have_distinct_identity_kappa() {
    // IPv4 and IPv6 endpoints at the same host:port pattern produce **distinct** identity κs
    // because the `PeerEndpoint` realization tags its payload with a proto byte (0 vs 1) — the
    // κ-graph treats them as different peers, by design (SPINE-1: identity = content address).
    let v4 = PeerEndpoint::tcp4([127, 0, 0, 1], 4001);
    let v6 = PeerEndpoint::tcp6(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // ::1
        4001,
    );
    assert_ne!(
        hologram_space::address_bytes(&v4.canonicalize()),
        hologram_space::address_bytes(&v6.canonicalize()),
        "v4 and v6 endpoints must have distinct identity κs (proto byte is part of the address)"
    );
    // Both round-trip through the realization payload codec.
    let (h4, p4) = PeerEndpoint::parse_tcp4(&v4.transport_payload).unwrap();
    assert_eq!((h4, p4), ([127, 0, 0, 1], 4001));
    let (h6, p6) = PeerEndpoint::parse_tcp6(&v6.transport_payload).unwrap();
    assert_eq!(p6, 4001);
    assert_eq!(h6[15], 1);
    // Cross-decode refuses: a v4 payload doesn't parse as v6, and vice versa (proto byte).
    assert!(PeerEndpoint::parse_tcp6(&v4.transport_payload).is_none());
    assert!(PeerEndpoint::parse_tcp4(&v6.transport_payload).is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn nw_tcp_rpc_timeout_is_operator_configurable_not_a_baked_constant() {
    use hologram_net::tcp::TcpConfig;

    // Default config has a finite timeout (operator-visible default; not an arbitrary baked cap).
    let default = TcpConfig::default();
    assert!(
        default.rpc_timeout.is_some(),
        "the default tunes a sensible timeout — but the operator can set None to disable it"
    );

    // The operator can construct an unbounded-wait config (None ⇒ no policy timeout — the wait
    // is bounded only by the network or the caller's own future-timeout).
    let unbounded = TcpConfig { rpc_timeout: None };
    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let _sync = hologram_net::tcp::TcpKappaSync::bind_with_config(
        "127.0.0.1:0".parse().unwrap(),
        store,
        unbounded,
    )
    .await
    .unwrap();
    // The mere fact that we can construct + bind a sync with `rpc_timeout = None` is the
    // structural witness: there is no baked-in policy cap (SPINE-6).
}

#[tokio::test(flavor = "current_thread")]
async fn nw_tcp_unparseable_addr_in_add_peer_is_loud() {
    // Caller passed a non-host:port string — fail-loud (SPINE-6), not silent success.
    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let sync = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store)
        .await
        .unwrap();
    let err = sync.add_peer("/dns/example/tcp/4001").await;
    assert!(
        matches!(
            err,
            Err(hologram_space::SyncError::BackendFailure("addr-parse"))
        ),
        "Multiaddr-style strings are NOT accepted (SPINE-1: no second naming surface)"
    );
}
