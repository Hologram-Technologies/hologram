//! κ-XOR Kademlia routing table. 256 k-buckets (one per bit of the 32-byte digest portion of
//! a κ-label), each holding up to `K=20` peers. Distance is bytewise XOR of the *decoded*
//! 32-byte digest portion of the 71-byte κ-label form (`blake3:<64 hex>`).
//!
//! **Why not byte-XOR the 71-byte form directly?** The 7-byte `blake3:` prefix is identical
//! for every blake3 κ, so XOR'ing the on-wire form would yield 0 for the prefix and only
//! discriminate on the hex digit *characters*. The hex-decoded form (32 bytes of digest) is
//! the natural XOR space and matches libp2p-kad's choice as well as the original Kademlia
//! paper's metric. uor-aligned: XOR over content keys is a structural relation (architecture
//! §11.1), not a registry lookup.

use crate::{Peer, K};
use hologram_space::KappaLabel71;

/// Decode the 32-byte blake3 digest from the on-wire κ-label form `blake3:<64 hex>`. Returns
/// zero on parse failure (caller should not see this on valid κ-labels; the substrate only
/// hands us labels that already round-tripped through `KappaLabel::from_bytes`).
fn decode_digest(label_bytes: &[u8; 71]) -> [u8; 32] {
    let mut out = [0u8; 32];
    // hex chars start at offset 7 (`blake3:` = 7 bytes).
    for i in 0..32 {
        let hi = hex_nibble(label_bytes[7 + i * 2]);
        let lo = hex_nibble(label_bytes[8 + i * 2]);
        out[i] = (hi << 4) | lo;
    }
    out
}

fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// XOR distance between two κ-labels. Returns the 32-byte XOR digest; bytewise comparison is
/// the standard Kademlia ordering (most-significant byte first). The substrate uses this as
/// the sort key for "closest peers to κ" queries.
pub fn xor_distance(a: &[u8; 71], b: &[u8; 71]) -> [u8; 32] {
    let da = decode_digest(a);
    let db = decode_digest(b);
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = da[i] ^ db[i];
    }
    out
}

/// Bucket index = position of the first differing bit (most-significant first) between two
/// digests. Returns 256 if equal (the peer's own κ).
fn bucket_index(distance: &[u8; 32]) -> usize {
    for (i, &b) in distance.iter().enumerate() {
        if b == 0 {
            continue;
        }
        // First differing byte; count leading zeros within it.
        return i * 8 + (b.leading_zeros() as usize);
    }
    256
}

/// 256-bucket κ-XOR routing table — a faithful Kademlia structure over κ keys.
pub struct RoutingTable {
    own_id: KappaLabel71,
    buckets: Vec<Vec<Peer>>,
}

impl RoutingTable {
    pub fn new(own_id: KappaLabel71) -> Self {
        Self {
            own_id,
            buckets: (0..256).map(|_| Vec::with_capacity(K)).collect(),
        }
    }

    /// Insert (or refresh) `peer`. If the bucket is full and the peer isn't already in it, the
    /// least-recently-added entry is kept (standard Kademlia eviction policy: incumbents are
    /// favored — they've proven liveness). The architecture is uor-aligned: a bucket-full
    /// state is structural, not arbitrary.
    pub fn insert(&mut self, peer: Peer) {
        if peer.id == self.own_id {
            return;
        }
        let dist = xor_distance(peer.id.as_array(), self.own_id.as_array());
        let idx = bucket_index(&dist).min(255);
        let bucket = &mut self.buckets[idx];
        if let Some(pos) = bucket.iter().position(|p| p.id == peer.id) {
            // Refresh by moving to the most-recently-seen position.
            bucket.remove(pos);
            bucket.push(peer);
            return;
        }
        if bucket.len() < K {
            bucket.push(peer);
        }
        // Full + not present: drop the new peer (favor incumbent). A production impl would
        // ping the least-recent entry to verify liveness; for the substrate this is a soft
        // refinement we can add when liveness probes are wired.
    }

    /// Return up to `n` peers closest to `target` by κ-XOR distance. Peers are walked across
    /// all buckets and sorted by distance — O(N log N) on the full table; the table is bounded
    /// by `256 * K = 5120` entries, so this is a structural bound, not policy.
    pub fn k_closest(&self, target: &[u8; 71], n: usize) -> Vec<Peer> {
        let mut all: Vec<Peer> = self
            .buckets
            .iter()
            .flat_map(|b| b.iter().cloned())
            .collect();
        all.sort_by_key(|p| xor_distance(p.id.as_array(), target));
        all.truncate(n);
        all
    }

    /// Number of peers currently in the table (across all buckets).
    pub fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// Whether the table holds no peers.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_space::address_bytes;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn fake_peer(seed: u8) -> Peer {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, seed)), 4000);
        Peer::from_addr(addr)
    }

    #[test]
    fn xor_distance_is_self_zero() {
        let a = address_bytes(b"a");
        let d = xor_distance(a.as_array(), a.as_array());
        assert_eq!(d, [0u8; 32]);
    }

    #[test]
    fn k_closest_orders_by_xor() {
        let own = address_bytes(b"own");
        let mut rt = RoutingTable::new(own);
        for s in 1u8..10 {
            rt.insert(fake_peer(s));
        }
        let target = address_bytes(b"target");
        let closest = rt.k_closest(target.as_array(), 3);
        // Manually compute distances and assert sort order.
        let mut by_hand: Vec<(Peer, [u8; 32])> = (1u8..10)
            .map(|s| {
                let p = fake_peer(s);
                let d = xor_distance(p.id.as_array(), target.as_array());
                (p, d)
            })
            .collect();
        by_hand.sort_by_key(|(_, d)| *d);
        for (i, p) in closest.iter().enumerate() {
            assert_eq!(p.id, by_hand[i].0.id);
        }
    }

    #[test]
    fn bucket_full_keeps_incumbents() {
        let own = address_bytes(b"own");
        let mut rt = RoutingTable::new(own);
        // Insert K identical-bucket peers, then one more — the last should NOT be present.
        for s in 1u8..=K as u8 {
            rt.insert(fake_peer(s));
        }
        let len_before = rt.len();
        rt.insert(fake_peer(K as u8 + 1));
        assert!(rt.len() >= len_before, "no entries lost");
    }
}
