//! Level 5 — federation fetch with verify-on-receipt (spec §8.3, §11.6).
//!
//! When a registry lacks a κ it MAY fetch it from a peer registry's `/v2/` blob endpoint. The
//! transport is **untrusted**: every received byte sequence is re-hashed against the requested κ
//! before it is accepted or cached (§8.1). A compromised or buggy peer that returns wrong content is
//! detected here and rejected — the same verify-on-receipt invariant the whole substrate upholds.
//! (Multi-hop relay chains this: each hop re-verifies, so end-to-end integrity holds across untrusted
//! relays without pre-established trust.)

use std::io::{Read, Write};
use std::net::TcpStream;

use hologram_space::{verify_kappa_axis, KappaLabel71, KappaStore};

/// Fetch `kappa` from peer registry `peer` (`host:port`) under `path`, verifying on receipt and
/// caching locally on success. Returns `true` iff a byte-verified copy was obtained and stored.
/// A non-200 response, a transport error, or a hash mismatch (forgery/corruption) returns `false`
/// and stores nothing.
pub fn federate_fetch(
    peer: &str,
    path: &str,
    kappa: &KappaLabel71,
    store: &dyn KappaStore,
) -> bool {
    let Ok(mut stream) = TcpStream::connect(peer) else {
        return false;
    };
    let req = format!(
        "GET /v2/{path}/blobs/{} HTTP/1.1\r\nHost: {peer}\r\nConnection: close\r\n\r\n",
        kappa.as_str()
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut resp = Vec::new();
    if stream.read_to_end(&mut resp).is_err() {
        return false;
    }
    let Some(split) = resp.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    // Only a 200 carries content.
    let status_ok = core::str::from_utf8(&resp[..split])
        .ok()
        .and_then(|h| h.lines().next())
        .and_then(|line| line.split_whitespace().nth(1))
        .map(|code| code == "200")
        .unwrap_or(false);
    if !status_ok {
        return false;
    }
    let body = &resp[split + 4..];
    // Verify-on-receipt: re-hash the body against the requested κ (§8.1). A mismatch is rejected.
    match verify_kappa_axis(body, kappa.as_bytes()) {
        Ok(true) => {
            let axis = kappa.sigma_axis().unwrap_or("blake3");
            store.put_axis(axis, body).is_ok()
        }
        _ => false, // forged / corrupt / axis error → reject, cache nothing
    }
}
