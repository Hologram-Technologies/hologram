//! ed25519 reference `SignatureVerifier` for the R3 attestation seam (spec 07).
//!
//! `hologram-space`'s no_std core carries only the portable `SignatureVerifier` trait + the signed
//! bytes; a space supplies the concrete verifier for its platform. This is that reference impl
//! (native, via the `ed25519-dalek` dev-dependency): it proves a `SessionAttestation` signed over
//! its bound facts verifies end-to-end, that tampering any fact breaks it, and that the signing
//! key's κ-identity is exactly its published `AttestationKey` content (GV-3 single surface).

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hologram_space::{address_bytes, AttestationKey, SessionAttestation, SignatureVerifier};

/// The reference ed25519 verifier — a space's platform crypto behind the portable seam.
struct Ed25519;
impl SignatureVerifier for Ed25519 {
    fn verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> bool {
        let Ok(pk): Result<[u8; 32], _> = public_key.try_into() else {
            return false;
        };
        let Ok(vk) = VerifyingKey::from_bytes(&pk) else {
            return false;
        };
        let Ok(sig) = Signature::from_slice(signature) else {
            return false;
        };
        vk.verify(message, &sig).is_ok()
    }
}

fn signed_attestation(seed: &[u8; 32]) -> (SessionAttestation, Vec<u8>) {
    let signing = SigningKey::from_bytes(seed);
    let public_key = signing.verifying_key().to_bytes().to_vec();
    // The signing key's identity IS its published AttestationKey content (κ), never a second surface.
    let signing_key = AttestationKey::new(0, public_key.clone()).kappa();
    let mut att = SessionAttestation {
        app: address_bytes(b"app-kappa"),
        caps: address_bytes(b"caps-kappa"),
        space: address_bytes(b"space-impl-kappa"),
        engine: address_bytes(b"engine-kappa"),
        signing_key,
        signature: Vec::new(),
    };
    let sig: Signature = signing.sign(&att.signable_bytes());
    att.signature = sig.to_bytes().to_vec();
    (att, public_key)
}

#[test]
fn valid_session_attestation_verifies() {
    let (att, public_key) = signed_attestation(&[7u8; 32]);
    assert!(
        att.verify(&Ed25519, &public_key),
        "a session attestation signed over its bound facts must verify under the signing key"
    );
    // The signing key κ is exactly the published AttestationKey content — one identity surface.
    assert_eq!(
        att.signing_key,
        AttestationKey::new(0, public_key).kappa(),
        "the attestation names the key by its κ-identity, not a separate certificate"
    );
}

#[test]
fn tampering_any_bound_fact_breaks_verification() {
    let (att, public_key) = signed_attestation(&[7u8; 32]);
    // Re-point the engine κ: the signed message (which embeds every bound κ) changes, so the
    // detached signature no longer verifies — "where and how it ran" is tamper-evident.
    let tampered = SessionAttestation {
        app: att.app,
        caps: att.caps,
        space: att.space,
        engine: address_bytes(b"DIFFERENT-engine"),
        signing_key: att.signing_key,
        signature: att.signature.clone(),
    };
    assert!(
        !tampered.verify(&Ed25519, &public_key),
        "tampering a bound fact must break the signature"
    );
}

#[test]
fn verifying_under_the_wrong_key_fails() {
    let (att, _public_key) = signed_attestation(&[7u8; 32]);
    let other = SigningKey::from_bytes(&[9u8; 32])
        .verifying_key()
        .to_bytes()
        .to_vec();
    assert!(
        !att.verify(&Ed25519, &other),
        "an attestation must not verify under a key that did not sign it"
    );
}

#[test]
fn malformed_key_or_signature_is_rejected_never_panics() {
    let (att, _pk) = signed_attestation(&[7u8; 32]);
    // Short/oversized/garbage key and signature bytes must be a clean `false`, never a panic.
    assert!(!att.verify(&Ed25519, &[]));
    assert!(!att.verify(&Ed25519, &[0u8; 5]));
    assert!(!att.verify(&Ed25519, &[0u8; 64]));
    let empty_sig = SessionAttestation {
        signature: Vec::new(),
        ..signed_attestation(&[7u8; 32]).0
    };
    assert!(!empty_sig.verify(&Ed25519, &[0u8; 32]));
}
