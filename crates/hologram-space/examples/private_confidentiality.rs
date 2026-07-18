//! **Exit-criteria demo — P6** (spec 06 §P6): private-network confidentiality + dedup.
//!
//! A member seals a payload with the network's symmetric key; a non-member (wrong key) cannot open
//! it; and two members sealing the SAME payload converge on ONE ciphertext — so Law L3 dedup
//! survives encryption. The dedup-preserving convergent nonce is what makes that hold; the
//! equality-leak it implies is the documented tradeoff of any dedup-preserving cipher (spec 04
//! §Private).
//!
//! Uses the ChaCha20-Poly1305 reference `PayloadCipher` (the `chacha20poly1305` dev-dependency) — the
//! no_std core carries only the portable trait + the convergent nonce; a space supplies the cipher.
//!
//! Run: `cargo run -p hologram-space --example private_confidentiality`

use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use hologram_space::{address_bytes, convergent_nonce, seal_private, PayloadCipher};

/// The reference AEAD — a space's platform crypto behind the portable seam. Guards ill-sized inputs
/// so it never panics on network-supplied bytes.
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

fn main() {
    let cipher = ChaCha;
    let network_key = [0x11u8; 32]; // the Private network's symmetric key (content at its key_ref κ)
    let outsider_key = [0x22u8; 32]; // a non-member's (wrong) key
    let payload = b"confidential model weights published on a private network";

    // A member seals the payload with the network key (convergent, dedup-preserving nonce).
    let (nonce, ciphertext) = seal_private(&cipher, &network_key, payload);
    println!("member sealed {} bytes → {} bytes ciphertext", payload.len(), ciphertext.len());

    // Another member (same key) opens it.
    assert_eq!(
        cipher.open(&network_key, &nonce, &ciphertext).as_deref(),
        Some(&payload[..]),
        "a member with the network key opens the payload"
    );
    println!("  a fellow member opened it");

    // A non-member (wrong key) CANNOT open it — the AEAD tag fails to authenticate.
    assert!(
        cipher.open(&outsider_key, &nonce, &ciphertext).is_none(),
        "a non-member (wrong key) cannot open the sealed payload"
    );
    println!("  a non-member (wrong key) was refused — AEAD authentication fails");

    // Dedup survives encryption: two members sealing the SAME payload converge on one ciphertext,
    // hence one κ (Law L3 — the store holds a single copy of shared private content).
    let (nonce2, ciphertext2) = seal_private(&cipher, &network_key, payload);
    assert_eq!(convergent_nonce(&network_key, payload), nonce, "the nonce is convergent");
    assert_eq!(nonce, nonce2);
    assert_eq!(
        address_bytes(&ciphertext),
        address_bytes(&ciphertext2),
        "identical content under one key → one κ (L3 dedup survives encryption)"
    );
    println!("  two members sealing the same payload converged on one κ — dedup preserved");

    println!("\nP6 private-network confidentiality demo: OK");
}
