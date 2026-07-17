//! ChaCha20-Poly1305 reference `PayloadCipher` for the Private network tier (spec 04 §Private).
//!
//! `hologram-space`'s no_std core carries only the portable `PayloadCipher` trait + the convergent
//! nonce derivation; a space supplies the concrete cipher for its platform. This is that reference
//! impl (native, via the `chacha20poly1305` dev-dependency). It proves: a private payload seals +
//! opens round-trip; identical content under one key produces identical ciphertext (Law L3 dedup
//! survives encryption — the spec's documented tension); a wrong key or tampered ciphertext fails
//! to authenticate (AEAD, fail-loud); and distinct keys give distinct ciphertext (no nonce reuse).

use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use hologram_space::{convergent_nonce, seal_private, PayloadCipher};

/// The reference AEAD cipher — a space's platform crypto behind the portable seam. Guards ill-sized
/// key/nonce so it never panics on network-supplied bytes.
struct ChaCha;
impl PayloadCipher for ChaCha {
    fn seal(&self, key: &[u8], nonce: &[u8], plaintext: &[u8]) -> Vec<u8> {
        if key.len() != 32 || nonce.len() != 12 {
            return Vec::new();
        }
        ChaCha20Poly1305::new(Key::from_slice(key))
            .encrypt(Nonce::from_slice(nonce), plaintext)
            .unwrap_or_default()
    }
    fn open(&self, key: &[u8], nonce: &[u8], ciphertext: &[u8]) -> Option<Vec<u8>> {
        if key.len() != 32 || nonce.len() != 12 {
            return None;
        }
        ChaCha20Poly1305::new(Key::from_slice(key))
            .decrypt(Nonce::from_slice(nonce), ciphertext)
            .ok()
    }
}

const KEY: [u8; 32] = [0x11; 32];
const OTHER_KEY: [u8; 32] = [0x22; 32];
const PLAINTEXT: &[u8] = b"secret payload published on a private (restricted + encrypted) network";

#[test]
fn private_payload_seals_and_opens_round_trip() {
    let cipher = ChaCha;
    let (nonce, ciphertext) = seal_private(&cipher, &KEY, PLAINTEXT);
    assert_ne!(
        ciphertext, PLAINTEXT,
        "the payload must actually be encrypted"
    );
    assert_eq!(
        cipher.open(&KEY, &nonce, &ciphertext).as_deref(),
        Some(PLAINTEXT),
        "a member holding the network key opens the payload"
    );
}

#[test]
fn convergent_encryption_preserves_dedup() {
    let cipher = ChaCha;
    // Same key + same plaintext ⇒ identical nonce ⇒ identical ciphertext: the store dedups one copy
    // of shared private content (Law L3 survives encryption).
    let (n1, c1) = seal_private(&cipher, &KEY, PLAINTEXT);
    let (n2, c2) = seal_private(&cipher, &KEY, PLAINTEXT);
    assert_eq!((n1, &c1), (n2, &c2));
    // The nonce is the documented convergent derivation, not random.
    assert_eq!(n1, convergent_nonce(&KEY, PLAINTEXT));
    // Distinct plaintexts under one key get distinct nonces (no AEAD nonce reuse across contents).
    let (n_other, _) = seal_private(&cipher, &KEY, b"a different payload");
    assert_ne!(n1, n_other);
}

#[test]
fn wrong_key_or_tampered_ciphertext_fails_to_authenticate() {
    let cipher = ChaCha;
    let (nonce, ciphertext) = seal_private(&cipher, &KEY, PLAINTEXT);
    // A non-member (wrong key) cannot open it — AEAD authentication fails.
    assert!(cipher.open(&OTHER_KEY, &nonce, &ciphertext).is_none());
    // Any tamper of the ciphertext or tag fails authentication (never returns wrong plaintext).
    for i in 0..ciphertext.len() {
        let mut bad = ciphertext.clone();
        bad[i] ^= 0x01;
        assert!(
            cipher.open(&KEY, &nonce, &bad).is_none(),
            "tampering byte {i} must fail authentication"
        );
    }
}

#[test]
fn distinct_keys_yield_distinct_ciphertext_no_cross_key_reuse() {
    let cipher = ChaCha;
    let (n1, c1) = seal_private(&cipher, &KEY, PLAINTEXT);
    let (n2, c2) = seal_private(&cipher, &OTHER_KEY, PLAINTEXT);
    // The convergent nonce mixes the key, so different networks never reuse a (key, nonce) pair.
    assert_ne!(n1, n2);
    assert_ne!(c1, c2);
}

#[test]
fn ill_sized_key_or_nonce_never_panics() {
    let cipher = ChaCha;
    // The reference impl guards lengths — hostile/short inputs are a clean empty/None, never a panic.
    assert!(cipher.seal(&[0u8; 8], &[0u8; 12], PLAINTEXT).is_empty());
    assert!(cipher.open(&KEY, &[0u8; 4], b"garbage").is_none());
    assert!(cipher.open(&[0u8; 3], &[0u8; 12], b"garbage").is_none());
}
