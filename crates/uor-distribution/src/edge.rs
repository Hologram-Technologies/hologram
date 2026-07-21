//! The edge blob canonical byte form (spec Appendix C / §3.4).
//!
//! An edge is a typed directed relationship between two κ-labels and — like everything in
//! κ-Distribution — is itself a content-addressed blob. Its κ-label is computed by hashing **these
//! bytes** under the **source's σ-axis** (§3.4). This module owns only the byte *framing* — the exact
//! layout that determines the edge κ. It is the load-bearing cross-registry contract: two registries
//! that frame the same edge differently produce different edge κ-labels, and federation of that edge
//! silently diverges.
//!
//! Layout (Appendix C):
//!
//! ```text
//! source ‖ 0x00 ‖ NFC(relation) ‖ 0x00 ‖ target ‖ 0x00 ‖ u32-BE(metadata_len) ‖ metadata
//! ```
//!
//! Null separators are unambiguous — κ-labels and relation strings contain no null byte. NFC
//! normalization of the relation makes semantically identical edges address identically; the defined
//! relation types (`owns`, `derives-from`, `composed-of`, `witness-of`, `pins`, `refers-to`,
//! `schema-for`, `filter-for`) are ASCII, for which NFC is the identity. Full Unicode NFC of
//! non-ASCII relations and deterministic-CBOR (RFC 8949 §4.2) encoding of structured metadata are a
//! tracked follow-on — the caller supplies already-canonical metadata bytes (or none).

use alloc::vec::Vec;

/// The canonical byte form of an edge blob (spec Appendix C).
///
/// - `source` / `target`: the κ-labels as their on-the-wire UTF-8 bytes.
/// - `relation`: the relationship type (ASCII relations are NFC-canonical as-is).
/// - `metadata`: opaque, already-deterministic bytes (RFC 8949 §4.2 CBOR); may be empty.
///
/// The returned bytes are hashed under the source's σ-axis to obtain the edge κ-label (§3.4).
pub fn edge_canonical(source: &[u8], relation: &str, target: &[u8], metadata: &[u8]) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(source.len() + relation.len() + target.len() + metadata.len() + 7);
    out.extend_from_slice(source);
    out.push(0x00);
    out.extend_from_slice(relation.as_bytes()); // ASCII relations are NFC-canonical
    out.push(0x00);
    out.extend_from_slice(target);
    out.push(0x00);
    out.extend_from_slice(&(metadata.len() as u32).to_be_bytes());
    out.extend_from_slice(metadata);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// KD-10 — the edge canonical form matches the Appendix C layout byte-for-byte (the
    /// cross-registry addressing contract). Also proves the metadata length is a u32 big-endian
    /// prefix and that distinct (source, relation, target) triples produce distinct bytes.
    #[test]
    fn kd10_edge_canonical_matches_appendix_c_layout() {
        let source = b"blake3:aa";
        let target = b"blake3:bb";
        let relation = "derives-from";

        // Empty metadata: source ‖ 0 ‖ relation ‖ 0 ‖ target ‖ 0 ‖ 0x00000000.
        let mut expected = Vec::new();
        expected.extend_from_slice(source);
        expected.push(0x00);
        expected.extend_from_slice(relation.as_bytes());
        expected.push(0x00);
        expected.extend_from_slice(target);
        expected.push(0x00);
        expected.extend_from_slice(&0u32.to_be_bytes());
        assert_eq!(edge_canonical(source, relation, target, b""), expected);

        // Metadata length is a u32 big-endian prefix (258 = 0x0000_0102).
        let metadata = alloc::vec![0x5a_u8; 258];
        let out = edge_canonical(source, relation, target, &metadata);
        let len_off = source.len() + 1 + relation.len() + 1 + target.len() + 1;
        assert_eq!(&out[len_off..len_off + 4], &[0x00, 0x00, 0x01, 0x02]);
        assert_eq!(&out[len_off + 4..], &metadata[..]);

        // Deterministic + sensitive: identical inputs match; any field change changes the bytes.
        assert_eq!(
            edge_canonical(source, relation, target, b""),
            edge_canonical(source, relation, target, b"")
        );
        assert_ne!(
            edge_canonical(source, "owns", target, b""),
            edge_canonical(source, relation, target, b"")
        );
        assert_ne!(
            edge_canonical(target, relation, source, b""),
            edge_canonical(source, relation, target, b"")
        );
    }
}
