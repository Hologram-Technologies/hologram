//! # HTTP-CAS gateway protocol (spec §6.3) — the `http` module of `hologram-net`
//!
//! The **protocol layer** of the Network Layer: map `/cas/{kappa}` requests to/from a
//! [`KappaStore`], format responses, and **verify received bytes by σ-axis re-derivation**
//! (SPINE-4 / §10.3). This is pure and hermetically testable (per the V&V hermetic-first policy);
//! the live transport (a thin `std::net::TcpListener` server + client in [`live`]) is layered
//! on top — it does not change these bytes (§10.6 wire-format byte-identity).

extern crate alloc;

use alloc::format;
use alloc::string::String;
use hologram_space::{verify_kappa, Bytes, KappaLabel, KappaLabel71, KappaStore};

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

/// The piece-table resource for a κ: `/cas/{kappa}/pieces` (see [`hologram_space::PieceTree`]).
pub fn pieces_path(kappa: &KappaLabel71) -> String {
    format!("/cas/{}/pieces", kappa.as_str())
}

/// What a request path names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasResource {
    /// `/cas/{kappa}`: the content.
    Blob(KappaLabel71),
    /// `/cas/{kappa}/pieces`: the content's 1 MiB piece table, which hashes to the κ.
    Pieces(KappaLabel71),
    /// `/cas/` or `/cas/?prefix=..&limit=..`: discovery.
    Discover,
}

/// Route a request path. `None` if it names nothing (the gateway returns 400).
pub fn parse_cas_resource(path: &str) -> Option<CasResource> {
    if path == "/cas/" || path.starts_with("/cas/?") {
        return Some(CasResource::Discover);
    }
    if let Some(k) = path.strip_suffix("/pieces").and_then(parse_cas_path) {
        return Some(CasResource::Pieces(k));
    }
    parse_cas_path(path).map(CasResource::Blob)
}

/// A request's `Range` header resolved against a content length (RFC 9110 §14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeSpec {
    /// No usable `Range`: serve the whole content with 200. Absent, malformed, a unit other than
    /// `bytes`, and multiple ranges all land here (a server may ignore `Range`, §14.2).
    Full,
    /// Serve bytes `[start, end)` with 206.
    Partial { start: u64, end: u64 },
    /// The range starts at or past the end: 416 with `Content-Range: bytes */len`.
    Unsatisfiable,
}

/// Resolve a `Range` header value against content of `len` bytes. Supports one range of the forms
/// `bytes=a-b`, `bytes=a-` and `bytes=-n`.
pub fn parse_range(header: Option<&str>, len: u64) -> RangeSpec {
    let Some(h) = header else {
        return RangeSpec::Full;
    };
    let h = h.trim();
    let Some(spec) = h
        .get(..6)
        .filter(|u| u.eq_ignore_ascii_case("bytes="))
        .map(|_| h[6..].trim())
    else {
        return RangeSpec::Full;
    };
    if spec.contains(',') {
        return RangeSpec::Full;
    }
    let Some((a, b)) = spec.split_once('-') else {
        return RangeSpec::Full;
    };
    let (a, b) = (a.trim(), b.trim());
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if a.is_empty() {
        // Suffix range: the last n bytes.
        let Some(n) = digits(b).then(|| b.parse::<u64>().ok()).flatten() else {
            return RangeSpec::Full;
        };
        if n == 0 || len == 0 {
            return RangeSpec::Unsatisfiable;
        }
        return RangeSpec::Partial {
            start: len - n.min(len),
            end: len,
        };
    }
    let Some(start) = digits(a).then(|| a.parse::<u64>().ok()).flatten() else {
        return RangeSpec::Full;
    };
    let last = if b.is_empty() {
        u64::MAX
    } else {
        match digits(b).then(|| b.parse::<u64>().ok()).flatten() {
            Some(v) if v >= start => v,
            _ => return RangeSpec::Full,
        }
    };
    if start >= len {
        return RangeSpec::Unsatisfiable;
    }
    RangeSpec::Partial {
        start,
        end: last.min(len - 1) + 1,
    }
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
pub mod client;
#[cfg(feature = "live")]
pub mod fetch;
#[cfg(feature = "live")]
pub mod live;

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_space::address_bytes;
    use hologram_tck::MemKappaStore;

    #[test]
    fn path_roundtrips() {
        let k = address_bytes(b"x");
        assert_eq!(parse_cas_path(&cas_path(&k)), Some(k));
        assert_eq!(parse_cas_path("/cas/not-a-kappa"), None);
        assert_eq!(parse_cas_path("/wrong/path"), None);
    }

    #[test]
    fn resources_route_by_path() {
        let k = address_bytes(b"x");
        assert_eq!(
            parse_cas_resource(&cas_path(&k)),
            Some(CasResource::Blob(k))
        );
        assert_eq!(
            parse_cas_resource(&pieces_path(&k)),
            Some(CasResource::Pieces(k))
        );
        assert_eq!(parse_cas_resource("/cas/"), Some(CasResource::Discover));
        assert_eq!(
            parse_cas_resource("/cas/?prefix=blake3:&limit=3"),
            Some(CasResource::Discover)
        );
        assert_eq!(parse_cas_resource("/cas/nope/pieces"), None);
        assert_eq!(parse_cas_resource("/other"), None);
    }

    #[test]
    fn range_header_follows_rfc_9110() {
        use RangeSpec::*;
        let p = |s, e| Partial { start: s, end: e };
        assert_eq!(parse_range(None, 100), Full);
        assert_eq!(parse_range(Some("bytes=0-9"), 100), p(0, 10));
        assert_eq!(parse_range(Some("bytes=90-"), 100), p(90, 100));
        assert_eq!(parse_range(Some("bytes=90-1000"), 100), p(90, 100));
        assert_eq!(parse_range(Some("bytes=-10"), 100), p(90, 100));
        assert_eq!(parse_range(Some("bytes=-1000"), 100), p(0, 100));
        assert_eq!(parse_range(Some(" Bytes = 5-5 "), 100), Full); // space inside the unit
        assert_eq!(parse_range(Some("BYTES=5-5"), 100), p(5, 6));
        assert_eq!(parse_range(Some("bytes=100-"), 100), Unsatisfiable);
        assert_eq!(parse_range(Some("bytes=200-300"), 100), Unsatisfiable);
        assert_eq!(parse_range(Some("bytes=-0"), 100), Unsatisfiable);
        assert_eq!(parse_range(Some("bytes=0-"), 0), Unsatisfiable);
        // Ignored (served whole): malformed, reversed, other units, multiple ranges.
        assert_eq!(parse_range(Some("bytes=9-0"), 100), Full);
        assert_eq!(parse_range(Some("bytes=a-b"), 100), Full);
        assert_eq!(parse_range(Some("bytes=-"), 100), Full);
        assert_eq!(parse_range(Some("items=0-1"), 100), Full);
        assert_eq!(parse_range(Some("bytes=0-1,5-6"), 100), Full);
        assert_eq!(parse_range(Some("bytes=+1-2"), 100), Full);
        assert_eq!(
            parse_range(Some("bytes=0-99999999999999999999999"), 100),
            Full
        );
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
