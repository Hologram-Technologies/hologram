//! Wire codec for the uor-native TCP transport. Frames are `u32 LE len | u8 kind | payload`.
//! Frame kinds are **append-only** (SPINE-5) — older nodes silently ignore unknown kinds for
//! forward-compatibility. No checksum on the frame itself: the κ-verified content payload
//! provides integrity at the layer that matters (SPINE-4).

/// Frame kinds. Append-only.
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Kind {
    /// `payload = κ_71` — please return your stored bytes for this κ.
    FetchReq = 0x01,
    /// `payload = κ_71 | bytes` — here are bytes addressed by κ. Receiver MUST verify
    /// `verify_kappa(bytes, κ)` before accepting (SPINE-4).
    FetchResOk = 0x02,
    /// `payload = κ_71` — I do not have this κ.
    FetchRes404 = 0x03,
    /// `payload = κ_71` — I am announcing I now hold this κ.
    Announce = 0x10,
    /// `payload = target_κ_71` — DHT find_node toward target.
    FindNodeReq = 0x30,
    /// `payload = u32 count | (κ_71 + endpoint_7)*`.
    FindNodeRes = 0x31,
    /// `payload = κ_71` — DHT get_providers for this content κ.
    GetProvidersReq = 0x50,
    /// `payload = u32 count | (κ_71 + endpoint_7)*`.
    GetProvidersRes = 0x51,
    /// `payload = κ_71 | endpoint_7` — record this peer as a provider for this κ.
    Provide = 0x40,
}

impl Kind {
    /// Parse a wire byte into a Kind, returning `None` for unknown kinds (which receivers
    /// should silently ignore for forward-compatibility — SPINE-5).
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::FetchReq),
            0x02 => Some(Self::FetchResOk),
            0x03 => Some(Self::FetchRes404),
            0x10 => Some(Self::Announce),
            0x30 => Some(Self::FindNodeReq),
            0x31 => Some(Self::FindNodeRes),
            0x40 => Some(Self::Provide),
            0x50 => Some(Self::GetProvidersReq),
            0x51 => Some(Self::GetProvidersRes),
            _ => None,
        }
    }
}

/// Build an outbound frame: `u32 LE len | u8 kind | payload`.
pub fn encode_frame(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + payload.len());
    let len = (1 + payload.len()) as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

/// Parse one frame from the front of `buf`. Returns `(kind, payload, total_bytes_consumed)`,
/// or `None` if the frame is incomplete.
pub fn decode_frame(buf: &[u8]) -> Option<(u8, &[u8], usize)> {
    if buf.len() < 5 {
        return None;
    }
    let len = u32::from_le_bytes(buf[..4].try_into().ok()?) as usize;
    if buf.len() < 4 + len || len < 1 {
        return None;
    }
    let kind = buf[4];
    let payload = &buf[5..4 + len];
    Some((kind, payload, 4 + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let f = encode_frame(Kind::Announce as u8, b"hello");
        let (k, p, n) = decode_frame(&f).unwrap();
        assert_eq!(k, Kind::Announce as u8);
        assert_eq!(p, b"hello");
        assert_eq!(n, f.len());
    }

    #[test]
    fn partial_frame_returns_none() {
        let f = encode_frame(Kind::FetchReq as u8, &[0u8; 71]);
        assert!(decode_frame(&f[..3]).is_none());
    }

    #[test]
    fn unknown_kind_parses_but_kind_enum_says_none() {
        let f = encode_frame(0x7F, b"x");
        let (k, _, _) = decode_frame(&f).unwrap();
        assert_eq!(k, 0x7F);
        assert!(Kind::from_u8(k).is_none());
    }
}
