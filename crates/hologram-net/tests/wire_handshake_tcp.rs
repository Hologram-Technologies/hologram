#![cfg(feature = "tcp")]
//! Wire-version handshake over a **real localhost TCP connection** (spec 04 §Protocol hardening).
//!
//! The in-process loopback test proves the handshake over a paired NIC; this proves the same
//! `bare` connect handshake negotiates over actual sockets — the integration a channel cannot cover.
//! `127.0.0.1:0` (OS-assigned port) so runs never collide, and `current_thread` tokio for
//! determinism. The `bare`/`tcp` transports share the `u32 LE len | u8 kind | payload` frame format,
//! so `hello_frame` (KIND_HELLO) is wire-compatible with both.

use hologram_net::bare::{hello_frame, negotiate_from_hello, HandshakeError};
use hologram_net::protocol::{WireVersionRange, WIRE_VERSION};
use hologram_net::tcp::TcpKappaSync;
use hologram_space::KappaStore;
use hologram_tck::MemKappaStore;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Read exactly one `len | kind | payload` frame off the stream (blocking on the socket).
async fn read_frame(stream: &mut TcpStream) -> Vec<u8> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.unwrap();
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut rest = vec![0u8; len];
    stream.read_exact(&mut rest).await.unwrap();
    let mut frame = len_buf.to_vec();
    frame.extend_from_slice(&rest);
    frame
}

/// Run a HELLO handshake between a listening server (advertising `server_range`) and a connecting
/// client (advertising `client_range`) over real localhost TCP; return `(client_result,
/// server_result)`. Each side sends its HELLO first, then reads the peer's and negotiates — no
/// ordering deadlock.
async fn handshake(
    server_range: WireVersionRange,
    client_range: WireVersionRange,
) -> (Result<u16, HandshakeError>, Result<u16, HandshakeError>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.unwrap();
        s.write_all(&hello_frame(server_range)).await.unwrap();
        let peer_hello = read_frame(&mut s).await;
        negotiate_from_hello(server_range, &peer_hello)
    });

    let mut client = TcpStream::connect(addr).await.unwrap();
    client.write_all(&hello_frame(client_range)).await.unwrap();
    let peer_hello = read_frame(&mut client).await;
    let client_result = negotiate_from_hello(client_range, &peer_hello);

    (client_result, server.await.unwrap())
}

#[tokio::test(flavor = "current_thread")]
async fn wire_version_handshake_negotiates_over_tcp() {
    // Overlapping ranges [1,3] and [2,5] → both sides independently negotiate the highest common
    // version (3) over the real socket.
    let (client, server) = handshake(
        WireVersionRange { min: 1, max: 3 },
        WireVersionRange { min: 2, max: 5 },
    )
    .await;
    assert_eq!(client, Ok(3));
    assert_eq!(server, Ok(3));
}

#[tokio::test(flavor = "current_thread")]
async fn tcp_kappa_sync_answers_the_handshake() {
    // The real transport, not a raw socket: a client that opens a `TcpKappaSync` connection with a
    // HELLO must get the server's HELLO back and negotiate its current wire version — the handshake
    // wired into `handle_connection` (additive; a peer that skips HELLO is unaffected — see the DHT
    // suite).
    let store = Arc::new(MemKappaStore::new()) as Arc<dyn KappaStore>;
    let server = TcpKappaSync::bind("127.0.0.1:0".parse().unwrap(), store)
        .await
        .unwrap();
    let addr = server.local_addr();

    let mut client = TcpStream::connect(addr).await.unwrap();
    let client_range = WireVersionRange { min: 1, max: 5 };
    client.write_all(&hello_frame(client_range)).await.unwrap();
    let server_hello = read_frame(&mut client).await;
    assert_eq!(
        negotiate_from_hello(client_range, &server_hello),
        Ok(WIRE_VERSION),
        "the transport answers the handshake at its current wire version"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn incompatible_peer_is_refused_over_tcp() {
    // Disjoint ranges [1,2] and [9,9] → both sides refuse; the connection is dropped, never a silent
    // downgrade to a version one side cannot parse.
    let (client, server) = handshake(
        WireVersionRange { min: 1, max: 2 },
        WireVersionRange { min: 9, max: 9 },
    )
    .await;
    assert_eq!(client, Err(HandshakeError::Incompatible));
    assert_eq!(server, Err(HandshakeError::Incompatible));
}
