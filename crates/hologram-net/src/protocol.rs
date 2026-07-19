//! Wire-protocol version negotiation (spec 04 §Protocol hardening).
//!
//! Peers exchange their supported wire-version range on connect and negotiate the **highest common**
//! version — an incompatible peer is refused at the protocol boundary, never silently downgraded
//! (SPINE-6 fail-loud). κ identity and the frame codec are version-stable; the version gates protocol
//! *semantics* (new frame kinds / fields) so old and new peers interoperate predictably.

/// The wire-protocol version this build speaks (the current `bare`/frame protocol). Bump on a
/// backward-incompatible change and widen [`WireVersionRange::CURRENT`] rather than renumbering
/// existing frame kinds (those are append-only, SPINE-5).
pub const WIRE_VERSION: u16 = 1;

/// A peer's supported wire-version range `[min, max]` (inclusive), advertised in the connect
/// handshake as `min:u16 LE ‖ max:u16 LE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WireVersionRange {
    /// Oldest protocol version this peer still understands.
    pub min: u16,
    /// Newest protocol version this peer speaks.
    pub max: u16,
}

impl WireVersionRange {
    /// This build's advertised range — currently exactly [`WIRE_VERSION`].
    pub const CURRENT: WireVersionRange = WireVersionRange {
        min: WIRE_VERSION,
        max: WIRE_VERSION,
    };

    /// Negotiate the version to speak with a peer: the **highest** version both support. `None` when
    /// the ranges are disjoint — the peers are incompatible, so the connection is refused (fail-loud,
    /// never a silent downgrade to something one side cannot parse).
    #[must_use]
    pub fn negotiate(self, peer: WireVersionRange) -> Option<u16> {
        let lo = self.min.max(peer.min);
        let hi = self.max.min(peer.max);
        (lo <= hi).then_some(hi)
    }

    /// Encode as the 4-byte handshake payload `min:u16 LE ‖ max:u16 LE`.
    #[must_use]
    pub fn encode(self) -> [u8; 4] {
        let mut out = [0u8; 4];
        out[..2].copy_from_slice(&self.min.to_le_bytes());
        out[2..].copy_from_slice(&self.max.to_le_bytes());
        out
    }

    /// Decode a handshake payload. `None` if it is not exactly 4 bytes, or if `min > max` (a
    /// malformed range) — a hostile peer's garbage handshake is rejected, never coerced.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<WireVersionRange> {
        let arr: [u8; 4] = bytes.try_into().ok()?;
        let min = u16::from_le_bytes([arr[0], arr[1]]);
        let max = u16::from_le_bytes([arr[2], arr[3]]);
        (min <= max).then_some(WireVersionRange { min, max })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(min: u16, max: u16) -> WireVersionRange {
        WireVersionRange { min, max }
    }

    #[test]
    fn negotiate_picks_the_highest_common_version() {
        assert_eq!(range(1, 3).negotiate(range(2, 5)), Some(3));
        assert_eq!(range(1, 1).negotiate(range(1, 1)), Some(1));
        assert_eq!(range(2, 4).negotiate(range(4, 9)), Some(4)); // touch at a single version
                                                                 // This build negotiates with itself to its own version.
        assert_eq!(
            WireVersionRange::CURRENT.negotiate(WireVersionRange::CURRENT),
            Some(WIRE_VERSION)
        );
    }

    #[test]
    fn disjoint_ranges_are_incompatible_no_silent_downgrade() {
        assert_eq!(range(1, 2).negotiate(range(3, 4)), None);
        assert_eq!(range(5, 9).negotiate(range(1, 4)), None);
    }

    #[test]
    fn handshake_payload_round_trips() {
        let r = range(1, 7);
        assert_eq!(WireVersionRange::decode(&r.encode()), Some(r));
    }

    #[test]
    fn malformed_handshake_is_rejected_never_panics() {
        assert_eq!(WireVersionRange::decode(&[]), None); // too short
        assert_eq!(WireVersionRange::decode(&[0, 0, 0]), None); // wrong length
        assert_eq!(WireVersionRange::decode(&[0, 0, 0, 0, 0]), None); // too long
                                                                      // min > max: bytes for min=5, max=2 → rejected.
        assert_eq!(WireVersionRange::decode(&[5, 0, 2, 0]), None);
    }
}
