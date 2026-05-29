#![cfg_attr(not(feature = "std"), no_std)]
//! # hologram-net-http — HTTP-CAS gateway protocol (spec §6.3)
//!
//! The **protocol layer** of the Network Layer: map `/cas/{kappa}` requests to/from a
//! [`KappaStore`], format responses, and **verify received bytes by σ-axis re-derivation**
//! (SPINE-4 / §10.3). This is pure and hermetically testable (per the V&V hermetic-first policy);
//! the live transport (a thin `std::net::TcpListener` server + client in [`live`]) is layered
//! on top — it does not change these bytes (§10.6 wire-format byte-identity).

extern crate alloc;

use alloc::format;
use alloc::string::String;
use hologram_substrate_core::{verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaStore};

/// The canonical request path for a κ-label (spec §6.3 `GET /cas/{kappa}`).
pub fn cas_path(kappa: &KappaLabel71) -> String {
    format!("/cas/{}", kappa.as_str())
}

/// Parse a `/cas/{kappa}` path back to a κ-label. `None` if malformed (the gateway returns 400).
pub fn parse_cas_path(path: &str) -> Option<KappaLabel71> {
    let rest = path.strip_prefix("/cas/")?;
    let bytes: [u8; 71] = rest.as_bytes().try_into().ok()?;
    KappaLabel::from_bytes(&bytes).ok()
}

/// HTTP-CAS response (status + optional body), per spec §6.3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CasResponse {
    /// `200 OK` — `Content-Type: application/octet-stream`, `Cache-Control: immutable`.
    Ok(Bytes),
    /// `404 Not Found` — not present locally (the gateway does NOT proactively fetch).
    NotFound,
    /// `400 Bad Request` — the path's κ-label is malformed.
    BadRequest,
}

impl CasResponse {
    pub fn status(&self) -> u16 {
        match self {
            CasResponse::Ok(_) => 200,
            CasResponse::NotFound => 404,
            CasResponse::BadRequest => 400,
        }
    }
}

/// **Server** side: answer a `GET /cas/{kappa}` from a local [`KappaStore`] (spec §6.5). Serves the
/// stored canonical bytes verbatim (§10.6 wire-format byte-identity) — never proactively fetches.
pub fn serve_get(store: &dyn KappaStore, path: &str) -> CasResponse {
    let Some(kappa) = parse_cas_path(path) else {
        return CasResponse::BadRequest;
    };
    match store.get(&kappa) {
        Ok(Some(bytes)) => CasResponse::Ok(bytes),
        _ => CasResponse::NotFound,
    }
}

/// Outcome of a client receiving a gateway response for `requested`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiveError {
    /// Re-derivation through the σ-axis did not match the requested κ — drop + mark untrusted (§6.4).
    VerificationFailed,
    /// Malformed κ in the received label.
    AxisError,
}

/// **Client** side: accept gateway bytes for `requested` **only after re-deriving the κ-label and
/// matching** (SPINE-4 / §6.4 / §10.3). This is what makes the network trustless — a gateway
/// cannot serve forged content.
pub fn accept_received(requested: &KappaLabel71, body: &[u8]) -> Result<Bytes, ReceiveError> {
    match verify_kappa(body, requested) {
        Ok(true) => Ok(Bytes::from(body.to_vec())),
        Ok(false) => Err(ReceiveError::VerificationFailed),
        Err(_) => Err(ReceiveError::AxisError),
    }
}

#[cfg(feature = "live")]
pub mod live;

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_store_mem::MemKappaStore;
    use hologram_substrate_core::address_bytes;

    #[test]
    fn path_roundtrips() {
        let k = address_bytes(b"x");
        assert_eq!(parse_cas_path(&cas_path(&k)), Some(k));
        assert_eq!(parse_cas_path("/cas/not-a-kappa"), None);
        assert_eq!(parse_cas_path("/wrong/path"), None);
    }

    #[test]
    fn server_serves_present_404s_absent_400s_malformed() {
        let store = MemKappaStore::new();
        let k = store.put("blake3", b"served-bytes").unwrap();
        // §10.6: the body is byte-identical to the stored canonical bytes.
        assert_eq!(
            serve_get(&store, &cas_path(&k)),
            CasResponse::Ok(store.get(&k).unwrap().unwrap())
        );
        assert_eq!(
            serve_get(&store, &cas_path(&address_bytes(b"absent"))),
            CasResponse::NotFound
        );
        assert_eq!(serve_get(&store, "/cas/garbage"), CasResponse::BadRequest);
    }

    #[test]
    fn client_verifies_on_receipt_and_rejects_forgery() {
        let store = MemKappaStore::new();
        let k = store.put("blake3", b"authentic-payload").unwrap();
        let CasResponse::Ok(body) = serve_get(&store, &cas_path(&k)) else {
            panic!("expected 200");
        };
        // Honest body verifies.
        assert_eq!(
            accept_received(&k, body.as_ref()).unwrap().as_ref(),
            b"authentic-payload"
        );
        // Forged body for the same κ is rejected (§6.4 trustless).
        assert_eq!(
            accept_received(&k, b"forged"),
            Err(ReceiveError::VerificationFailed)
        );
    }
}
