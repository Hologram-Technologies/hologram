//! Content addressing for hologram via UOR-ADDR (wiki ADR-031 / ADR-061).
//!
//! hologram is an application of Prism (ADR-031); its content identities
//! are uor-addr **κ-labels** — typed, σ-projection-grounded, replayable
//! 71-byte content addresses — not bare hashes. Two leverage points:
//!
//! * **Decomposition.** Each addressable part folds to a κ-label through
//!   the uor-addr realization that matches its type: [`ring`] for the
//!   Amendment-43 ring elements hologram is built on, and (under the
//!   `model-formats` feature) `gguf` / `onnx` for the model files
//!   hologram-ai ingests. Structurally-equivalent content collapses to
//!   one κ-label, so a κ-label is the canonical dedup / cache key.
//! * **Composition.** A composite object's κ-label is the E₈ categorical
//!   composition (ADR-061) of its parts' κ-labels. [`compose_model`]
//!   left-folds parts under the commutative CS-G2 product, so a model's
//!   identity is a deterministic, axis-homogeneous function of its parts'
//!   identities — independent of the order they were assembled in.
//!
//! Every κ-label minted here carries a replayable [`AddressWitness`]
//! (TC-05); `witness.verify()` re-certifies the derivation through
//! `prism::replay::certify_from_trace` without re-running the σ-axis.
//!
//! ## σ-axis
//!
//! hologram's canonical hash axis is BLAKE3 (ADR-052), so the
//! hologram-default entry points ([`address_ring`], [`compose_model`])
//! bind the `blake3` σ-axis. The full per-axis surface (sha256, sha3-256,
//! keccak256, sha512) is re-exported via [`composition`] and [`ring`] for
//! arbitrary use cases.

use hologram_host::HologramHasher;
use prism::vocabulary::Hasher;
use uor_addr::composition::compose_g2_product_blake3;

pub use uor_addr::composition::{
    self, compose_e6_filtration, compose_e7_augmentation, compose_e8_embedding,
    compose_f4_quotient, compose_g2_product, CompositionFailure,
};
pub use uor_addr::ring::{self, AddressFailure, AddressOutcome, AddressWitness, VerifyError};
pub use uor_addr::KappaLabel;

/// Hologram's content-address type: a BLAKE3-σ-axis κ-label (71 bytes,
/// `blake3:<64 hex>`). Every value the runtime operates on is identified
/// by one of these — leaves by their content ([`address_bytes`]), derived
/// values by their witnessed derivation ([`derive_label_witnessed`]).
pub type ContentLabel = KappaLabel<71>;

/// Render a 32-byte BLAKE3 digest as the canonical `blake3:<64 hex>`
/// κ-label. The 7-byte `blake3:` prefix plus 64 lowercase-hex digest
/// bytes is exactly the 71-byte κ-label width, so the [`KappaLabel`]
/// constructor's length + ASCII checks always pass.
fn blake3_kappa(digest: &[u8; 32]) -> ContentLabel {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 71];
    buf[..7].copy_from_slice(b"blake3:");
    for (i, &byte) in digest.iter().enumerate() {
        buf[7 + 2 * i] = HEX[(byte >> 4) as usize];
        buf[7 + 2 * i + 1] = HEX[(byte & 0x0f) as usize];
    }
    KappaLabel::from_bytes(&buf).expect("71-byte ASCII blake3 κ-label by construction")
}

/// Content-address opaque bytes on the BLAKE3 σ-axis (ADR-052/ADR-060).
///
/// This is the *leaf* identity: inputs, weights, and any value whose
/// identity is its content. Equal bytes yield an equal κ-label, so it is
/// the canonical dedup key. Unlike [`address_ring`] (which parses the
/// Amendment-43 ring grammar) this folds *arbitrary* bytes through
/// hologram's canonical `Hasher<32>` (`prism::crypto::Blake3Hasher`),
/// matching the streaming σ-axis pattern uor-addr's gguf/onnx
/// realizations use for opaque tensor-data leaves.
#[must_use]
pub fn address_bytes(bytes: &[u8]) -> ContentLabel {
    let digest: [u8; 32] = HologramHasher::initial().fold_bytes(bytes).finalize();
    blake3_kappa(&digest)
}

/// Derive a computed value's κ-label from its derivation — the operation
/// identity (`opcode` ‖ scalar `params`) and the **ordered** content labels
/// of its operands — by folding them on the σ-axis, without hashing the
/// result bytes. The fold is order-sensitive (operands are absorbed in
/// sequence), so `f(A,B)` and `f(B,A)` get distinct labels.
///
/// This is the runtime's *internal* content-address for execution
/// memoization: it is the per-node reuse key the content-addressed
/// executor folds on the hot path. The cost is `O(operands)` (a single
/// streaming BLAKE3 over a few 71-byte labels) — orders of magnitude
/// cheaper than grounding a prism pipeline — so addressing every node adds
/// no measurable overhead to the compute path (TC-01). Determinism of the
/// kernels (ADR-030) makes an identical derivation a sound cache key.
///
/// The replay-grounded form is [`derive_label_witnessed`], which composes
/// the same derivation *through the ψ-tower* to carry a TC-05 witness. Use
/// the witnessed form for an address that crosses the runtime boundary and
/// may be independently replayed; use this for the high-frequency internal
/// reuse key.
#[must_use]
pub fn derive_label(opcode: u16, params: &[u8], inputs: &[ContentLabel]) -> ContentLabel {
    let mut h = HologramHasher::initial()
        .fold_bytes(&opcode.to_le_bytes())
        .fold_bytes(params);
    for label in inputs {
        h = h.fold_bytes(label.as_bytes());
    }
    blake3_kappa(&h.finalize())
}

#[cfg(feature = "model-formats")]
pub use uor_addr::{gguf, onnx};

/// Mint hologram's canonical κ-label for an Amendment-43 ring element,
/// grounding the σ-projection on the BLAKE3 axis. The input is the ring
/// element's canonical bytes (`[witt_level] || le_coefficient`).
///
/// # Errors
///
/// [`AddressFailure`] if `canonical_bytes` is not a well-formed ring
/// element (bad Witt level or coefficient width).
pub fn address_ring(canonical_bytes: &[u8]) -> Result<AddressOutcome<71>, AddressFailure> {
    ring::address_blake3(canonical_bytes)
}

/// Assemble a composite κ-label from its constituent part labels by
/// left-folding the CS-G2 commutative product (ADR-061) on the BLAKE3
/// axis. Because CS-G2 commutativity is structural, the result is
/// independent of operand order; every part must carry the `blake3`
/// σ-axis (CA-3 σ-axis homogeneity).
///
/// This is hologram's decomposition→composition primitive: address each
/// part with [`address_ring`] (or a model-format realization), then fold
/// the part labels into one model identity. A single-part input yields
/// that part's label unchanged.
///
/// # Errors
///
/// - [`CompositionFailure::MalformedOperand`] if `parts` is empty or a
///   part is not a well-formed κ-label.
/// - [`CompositionFailure::OperandSigmaAxisMismatch`] if a part does not
///   carry the `blake3` axis.
pub fn compose_model(parts: &[KappaLabel<71>]) -> Result<KappaLabel<71>, CompositionFailure> {
    let (first, rest) = parts
        .split_first()
        .ok_or(CompositionFailure::MalformedOperand)?;
    let mut acc = *first;
    for part in rest {
        acc = compose_g2_product_blake3(&acc, part)?.address;
    }
    Ok(acc)
}

/// **Witnessed** derivation address — the canonical way to address a
/// *computed* value. A computed value's κ-label is built *by ordered
/// composition* of its operands' addresses (as prism-btc demonstrates),
/// not a re-hash of its result bytes, and the returned [`AddressOutcome`]
/// carries a replayable TC-05 witness. The cost is `O(operands)` (a fold
/// over 71-byte labels), independent of tensor size, so content addressing
/// the execution graph never scales with the data it moves. Determinism of
/// the kernels (ADR-030) makes it a sound memoization key: an identical
/// derivation (same op, params, operand addresses) always yields identical
/// content, so the executor can elide the recompute.
///
/// The op identity (`opcode` ‖ scalar `params`) is content-addressed to an
/// operand-0 base, then each input label is folded in **in order** through
/// [`crate::compose::compose_ordered_blake3`] — hologram's *non-commutative*
/// BLAKE3 ordered product (the operator uor-addr does not ship). Order is
/// captured by the operator's algebra, so `matmul(A,B)` and `matmul(B,A)`
/// get distinct addresses; the V&V suite proves it.
///
/// # Errors
///
/// [`CompositionFailure`] if a composition step rejects an operand
/// (unreachable for well-formed `blake3` κ-labels from [`address_bytes`]).
pub fn derive_label_witnessed(
    opcode: u16,
    params: &[u8],
    inputs: &[ContentLabel],
) -> Result<AddressOutcome<71>, CompositionFailure> {
    use crate::compose::compose_ordered_blake3;
    use alloc::vec::Vec;
    // Operand-0 base = content address of the operation identity.
    let mut pre: Vec<u8> = Vec::with_capacity(2 + params.len());
    pre.extend_from_slice(&opcode.to_le_bytes());
    pre.extend_from_slice(params);
    let op_label = address_bytes(&pre);

    // Fold inputs in order through the witnessed non-commutative product.
    let mut acc = op_label;
    let mut outcome: Option<AddressOutcome<71>> = None;
    for inp in inputs {
        let o = compose_ordered_blake3(&acc, inp)?;
        acc = o.address;
        outcome = Some(o);
    }
    match outcome {
        Some(o) => Ok(o),
        // No inputs (constant-only graph): ground the op identity alone.
        None => compose_ordered_blake3(&op_label, &op_label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_blake3_kappa(l: &ContentLabel) -> bool {
        let s = l.as_str();
        s.len() == 71
            && s.starts_with("blake3:")
            && s[7..]
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    }

    #[test]
    fn address_bytes_is_deterministic_and_well_formed() {
        let a = address_bytes(b"the quick brown fox");
        let b = address_bytes(b"the quick brown fox");
        assert_eq!(a, b, "equal content ⇒ equal κ-label");
        assert!(is_blake3_kappa(&a));
    }

    #[test]
    fn address_bytes_is_collision_sensitive() {
        assert_ne!(address_bytes(b"alpha"), address_bytes(b"beta"));
        // single-bit difference
        assert_ne!(
            address_bytes(&[0u8; 8]),
            address_bytes(&[0, 0, 0, 0, 0, 0, 0, 1])
        );
    }

    #[test]
    fn derive_label_is_deterministic_and_well_formed() {
        let i0 = address_bytes(b"x");
        let i1 = address_bytes(b"y");
        let a = derive_label(45, &[1, 2, 3], &[i0, i1]);
        let b = derive_label(45, &[1, 2, 3], &[i0, i1]);
        assert_eq!(a, b);
        assert!(is_blake3_kappa(&a));
    }

    #[test]
    fn derive_label_separates_opcode_params_and_ordered_operands() {
        let i0 = address_bytes(b"x");
        let i1 = address_bytes(b"y");
        let base = derive_label(45, &[1, 2, 3], &[i0, i1]);
        // different opcode
        assert_ne!(base, derive_label(46, &[1, 2, 3], &[i0, i1]));
        // different params (e.g. different shape)
        assert_ne!(base, derive_label(45, &[1, 2, 4], &[i0, i1]));
        // operand order matters (non-commutative reuse key)
        assert_ne!(base, derive_label(45, &[1, 2, 3], &[i1, i0]));
        // different operand content
        let i2 = address_bytes(b"z");
        assert_ne!(base, derive_label(45, &[1, 2, 3], &[i0, i2]));
    }
}
