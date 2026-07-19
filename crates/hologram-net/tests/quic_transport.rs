//! Localhost integration for the QUIC transport (`quic`) — two real quinn endpoints on ephemeral
//! ports, one dialing the other over QUIC/TLS 1.3. Exercises the exact path a native peer rides:
//! wire-version handshake → FETCH_REQ → FETCH_RES → verify-on-receipt (SPINE-4). Gated behind the
//! `quic` feature so the default (no_std-capable) build is untouched.
#![cfg(feature = "quic")]

use std::sync::Arc;

use hologram_net::quic::QuicPeer;
use hologram_space::{KappaLabel71, KappaStore, KappaSync};
use hologram_tck::MemKappaStore;

/// A resolver backed by a `MemKappaStore` — the substrate's `KappaStore::get`, as a fetch hook.
fn resolver(store: Arc<MemKappaStore>) -> hologram_net::quic::LocalResolver {
    Arc::new(move |k: &KappaLabel71| store.get(k).ok().flatten())
}

fn loopback() -> std::net::SocketAddr {
    ([127, 0, 0, 1], 0).into()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_quic_peers_fetch_over_localhost() {
    // Peer A holds content; peer B (empty) dials A over QUIC and fetches it — the full request/
    // response path with the handshake and κ re-derivation, no local shortcut.
    let a_store = Arc::new(MemKappaStore::new());
    let payload = b"content-that-only-peer-A-holds-over-quic";
    let k = a_store.put("blake3", payload).unwrap();

    let peer_a = QuicPeer::bind(loopback(), resolver(a_store)).unwrap();
    let a_addr = peer_a.local_addr().unwrap();
    let peer_b = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();

    let got = peer_b
        .fetch_from(a_addr, &k)
        .await
        .expect("fetch succeeds")
        .expect("A holds the content");
    assert_eq!(got.as_ref(), payload, "B re-derived A's content over QUIC");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quic_fetch_miss_yields_404_not_a_hang() {
    // A κ neither peer holds → A answers a resolved-absent 404 → B records the miss as `None`.
    let peer_a = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();
    let a_addr = peer_a.local_addr().unwrap();
    let peer_b = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();

    let absent = hologram_space::address_bytes(b"nobody-has-this-over-quic");
    let got = peer_b
        .fetch_from(a_addr, &absent)
        .await
        .expect("fetch completes");
    assert!(
        got.is_none(),
        "an absent κ resolves to None, not an error or a hang"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quic_rejects_a_forging_responder() {
    // A responder that returns bytes not matching the requested κ must be rejected: verify-on-receipt
    // re-derives the σ-axis at B, so a forged FETCH_RES_OK never passes (SPINE-4).
    let liar: hologram_net::quic::LocalResolver = Arc::new(|_k| {
        Some(Arc::<[u8]>::from(
            &b"these-bytes-do-not-hash-to-the-requested-kappa"[..],
        ))
    });
    let peer_a = QuicPeer::bind(loopback(), liar).unwrap();
    let a_addr = peer_a.local_addr().unwrap();
    let peer_b = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();

    // Ask for the κ of *honest* content; A lies and ships different bytes under that κ.
    let honest_k = hologram_space::address_bytes(b"the-real-content");
    let err = peer_b
        .fetch_from(a_addr, &honest_k)
        .await
        .expect_err("a forging responder must be rejected, not trusted");
    assert!(
        matches!(err, hologram_space::SyncError::VerificationFailed),
        "forged content fails κ re-derivation → VerificationFailed, got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quic_fetch_short_circuits_on_local_content() {
    // With no peers joined, a `KappaSync::fetch` for locally-held content resolves from the resolver
    // — no network round-trip, no peer required.
    let store = Arc::new(MemKappaStore::new());
    let k = store.put("blake3", b"held-locally").unwrap();
    let peer = QuicPeer::bind(loopback(), resolver(store)).unwrap();
    assert!(peer.peers().is_empty());

    let got = peer.fetch(&k).await.unwrap().expect("local hit");
    assert_eq!(got.as_ref(), b"held-locally");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quic_fetch_routes_across_joined_peers() {
    // A holds the content; C is empty. B joins C (via the `add_peer` string API) *then* A, and does a
    // `KappaSync::fetch` by κ: routing tries C (a miss) then A (the holder), verify-on-receipt at the
    // hop. Join order is honored and the first hit wins.
    let a_store = Arc::new(MemKappaStore::new());
    let payload = b"content-reached-by-routing";
    let k = a_store.put("blake3", payload).unwrap();

    let peer_a = QuicPeer::bind(loopback(), resolver(a_store)).unwrap();
    let peer_c = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();
    let peer_b = QuicPeer::bind(loopback(), resolver(Arc::new(MemKappaStore::new()))).unwrap();

    peer_b
        .add_peer(&peer_c.local_addr().unwrap().to_string())
        .await
        .unwrap(); // string-addr join (the KappaSync surface)
    peer_b.join(peer_a.local_addr().unwrap()); // typed join
    assert_eq!(peer_b.peers().len(), 2, "both peers joined, in order");

    let got = peer_b
        .fetch(&k)
        .await
        .unwrap()
        .expect("routed to the holder A");
    assert_eq!(got.as_ref(), payload);

    // A κ no joined peer holds → routed miss → None (every peer tried, none had it; not an error).
    let absent = hologram_space::address_bytes(b"held-by-nobody");
    assert!(peer_b.fetch(&absent).await.unwrap().is_none());

    // Idempotent join: re-joining an existing peer does not duplicate it.
    peer_b.join(peer_a.local_addr().unwrap());
    assert_eq!(
        peer_b.peers().len(),
        2,
        "re-joining a known peer is a no-op"
    );
}
