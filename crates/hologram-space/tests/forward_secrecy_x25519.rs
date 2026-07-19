//! X25519 sealed-box reference `KeyWrapper` for the P6 forward-secrecy seam (spec 04 §Private).
//!
//! `hologram-space`'s no_std core carries only the portable `KeyWrapper` trait and the `KeyEpoch`
//! realization (the membership-epoch key chain); a space supplies the concrete asymmetric scheme.
//! This is that reference impl (native, via `x25519-dalek` + `chacha20poly1305` dev-deps): an
//! ephemeral-static ECDH "sealed box" that wraps an epoch's fresh symmetric key to a member's
//! enrollment public key. It proves the property a convergent shared-key *cannot* provide —
//! **forward secrecy on membership change**: a removed member has no wrap in the new epoch and
//! cannot unwrap anyone else's, so they can never obtain the new key nor open post-revocation
//! content, while the remaining members re-derive the new key and read on.

use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use hologram_space::{
    address_bytes, references, seal_private, KappaLabel71, KappaStore, KeyEpoch, KeyWrap,
    KeyWrapper, MemKappaStore, MembershipChange, PayloadCipher, Realization, REGISTRY,
};
use x25519_dalek::{PublicKey, StaticSecret};

/// The reference AEAD payload cipher (as in `private_tier_chacha20.rs`) — used both to seal payloads
/// and, inside the sealed box, to encrypt the wrapped key.
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

/// Derive the sealed-box symmetric key + nonce from the ECDH shared secret and the two public keys.
fn sym_and_nonce(
    shared: &[u8; 32],
    eph_pub: &[u8; 32],
    recipient_pub: &[u8; 32],
) -> ([u8; 32], [u8; 12]) {
    let mut kh = blake3::Hasher::new();
    kh.update(b"hologram-wrap-key");
    kh.update(shared);
    let key = *kh.finalize().as_bytes();
    let mut nh = blake3::Hasher::new();
    nh.update(b"hologram-wrap-nonce");
    nh.update(eph_pub);
    nh.update(recipient_pub);
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&nh.finalize().as_bytes()[..12]);
    (key, nonce)
}

/// An X25519 ephemeral-static sealed box: `wrapped = eph_pub(32) ‖ AEAD(key_material)`. The ephemeral
/// secret is *derived deterministically* (from the key material + recipient) so the test is
/// reproducible — a production wrapper would use a random ephemeral secret; the exclusion property
/// under test is identical either way.
struct SealedBox;
impl KeyWrapper for SealedBox {
    fn wrap(&self, recipient_public_key: &[u8], key_material: &[u8]) -> Vec<u8> {
        let Ok(rpk): Result<[u8; 32], _> = recipient_public_key.try_into() else {
            return Vec::new();
        };
        let recipient_pub = PublicKey::from(rpk);
        let mut eh = blake3::Hasher::new();
        eh.update(b"hologram-wrap-eph");
        eh.update(key_material);
        eh.update(&rpk);
        let eph_secret = StaticSecret::from(*eh.finalize().as_bytes());
        let eph_pub = PublicKey::from(&eph_secret);
        let shared = eph_secret.diffie_hellman(&recipient_pub);
        let (key, nonce) = sym_and_nonce(shared.as_bytes(), eph_pub.as_bytes(), &rpk);
        let ct = ChaCha20Poly1305::new(Key::from_slice(&key))
            .encrypt(Nonce::from_slice(&nonce), key_material)
            .unwrap_or_default();
        if ct.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(32 + ct.len());
        out.extend_from_slice(eph_pub.as_bytes());
        out.extend_from_slice(&ct);
        out
    }

    fn unwrap(&self, recipient_private_key: &[u8], wrapped: &[u8]) -> Option<Vec<u8>> {
        let rsk: [u8; 32] = recipient_private_key.try_into().ok()?;
        if wrapped.len() < 32 {
            return None;
        }
        let recipient_secret = StaticSecret::from(rsk);
        let recipient_pub = PublicKey::from(&recipient_secret);
        let eph_pub_bytes: [u8; 32] = wrapped[..32].try_into().ok()?;
        let eph_pub = PublicKey::from(eph_pub_bytes);
        let shared = recipient_secret.diffie_hellman(&eph_pub);
        let (key, nonce) =
            sym_and_nonce(shared.as_bytes(), &eph_pub_bytes, recipient_pub.as_bytes());
        ChaCha20Poly1305::new(Key::from_slice(&key))
            .decrypt(Nonce::from_slice(&nonce), &wrapped[32..])
            .ok()
    }
}

/// A member: an X25519 enrollment keypair whose identity κ *is* its published public key content
/// (the GV-3 single-surface principle, as with `AttestationKey`).
fn member(seed: u8) -> (StaticSecret, PublicKey, KappaLabel71) {
    let secret = StaticSecret::from([seed; 32]);
    let public = PublicKey::from(&secret);
    let kappa = address_bytes(public.as_bytes());
    (secret, public, kappa)
}

#[test]
fn removed_member_cannot_open_the_new_epoch_key_or_payload() {
    let w = SealedBox;
    let store = MemKappaStore::new();
    let network = address_bytes(b"a-private-network");

    let (alice_sk, alice_pk, alice_k) = member(1);
    let (bob_sk, bob_pk, bob_k) = member(2);
    let (carol_sk, carol_pk, carol_k) = member(3);

    // Genesis epoch — key K0 wrapped to all three members.
    let k0 = [0x10u8; 32];
    let e0 = KeyEpoch::genesis(
        network,
        vec![
            KeyWrap {
                member: alice_k,
                wrapped: w.wrap(alice_pk.as_bytes(), &k0),
            },
            KeyWrap {
                member: bob_k,
                wrapped: w.wrap(bob_pk.as_bytes(), &k0),
            },
            KeyWrap {
                member: carol_k,
                wrapped: w.wrap(carol_pk.as_bytes(), &k0),
            },
        ],
    );
    let e0_kappa = store.put("blake3", &e0.canonicalize()).unwrap();

    // Every genesis member recovers K0 from their own wrap.
    for (sk, k) in [(&alice_sk, alice_k), (&bob_sk, bob_k), (&carol_sk, carol_k)] {
        let wrap = e0.wrap_for(&k).expect("genesis member has a wrap");
        assert_eq!(
            w.unwrap(&sk.to_bytes(), wrap).unwrap(),
            k0,
            "a genesis member recovers the epoch-0 key"
        );
    }

    // Membership change: remove carol → epoch 1 with a FRESH key K1 wrapped to alice + bob only.
    let k1 = [0x21u8; 32];
    let e1 = KeyEpoch::next(
        network,
        e0_kappa,
        1,
        MembershipChange::MemberRemoved,
        vec![
            KeyWrap {
                member: alice_k,
                wrapped: w.wrap(alice_pk.as_bytes(), &k1),
            },
            KeyWrap {
                member: bob_k,
                wrapped: w.wrap(bob_pk.as_bytes(), &k1),
            },
        ],
    );
    let e1_kappa = store.put("blake3", &e1.canonicalize()).unwrap();

    // Structural forward secrecy: carol is absent from epoch 1's wraps.
    assert!(
        e1.wrap_for(&carol_k).is_none(),
        "removed member has no wrap in the new epoch"
    );
    assert!(!e1.is_member(&carol_k));
    assert!(e1.is_member(&alice_k) && e1.is_member(&bob_k));

    // Cryptographic forward secrecy: carol's key unwraps NONE of epoch 1's per-member wraps (each is
    // sealed to alice/bob), so she cannot obtain K1.
    for kw in &e1.wraps {
        assert!(
            w.unwrap(&carol_sk.to_bytes(), &kw.wrapped).is_none(),
            "a removed member's key unwraps none of the new epoch's wraps"
        );
    }

    // Remaining members re-derive K1 and open post-revocation content; the stale key K0 cannot.
    let cipher = ChaCha;
    let payload = b"post-revocation private content";
    let (nonce, ct) = seal_private(&cipher, &k1, payload);
    let alice_k1 = w
        .unwrap(&alice_sk.to_bytes(), e1.wrap_for(&alice_k).unwrap())
        .unwrap();
    assert_eq!(alice_k1, k1);
    assert_eq!(
        cipher.open(&alice_k1, &nonce, &ct).unwrap(),
        payload,
        "a remaining member opens new-epoch content"
    );
    assert!(
        cipher.open(&k0, &nonce, &ct).is_none(),
        "the stale epoch-0 key cannot open epoch-1 content"
    );

    // The chain-head predicate agrees, resolved over the store.
    assert!(!KeyEpoch::member_can_open_current(&carol_k, &e1_kappa, &store).unwrap());
    assert!(KeyEpoch::member_can_open_current(&alice_k, &e1_kappa, &store).unwrap());
    assert_eq!(
        KeyEpoch::current(&e1_kappa, &store).unwrap().unwrap().epoch,
        1
    );
}

#[test]
fn rekey_helper_cuts_forward_secure_epochs_on_add_and_remove() {
    let w = SealedBox;
    let cipher = ChaCha;
    let network = address_bytes(b"net-rekey");
    let (alice_sk, alice_pk, alice_k) = member(11);
    let (bob_sk, bob_pk, bob_k) = member(12);
    let (carol_sk, carol_pk, carol_k) = member(13);

    let all_three = [
        (alice_k, alice_pk.as_bytes().to_vec()),
        (bob_k, bob_pk.as_bytes().to_vec()),
        (carol_k, carol_pk.as_bytes().to_vec()),
    ];
    let alice_bob = [
        (alice_k, alice_pk.as_bytes().to_vec()),
        (bob_k, bob_pk.as_bytes().to_vec()),
    ];

    // Genesis via the helper — K0 to all three.
    let k0 = [0x40u8; 32];
    let e0 = KeyEpoch::genesis(network, KeyEpoch::wrap_for_members(&w, &all_three, &k0));
    assert_eq!(e0.epoch, 0);
    assert_eq!(e0.members().len(), 3);

    // Remove carol via `rekey` — fresh K1 sealed to {alice, bob}. The primitive excludes carol by
    // construction (she is not in the passed membership); the epoch increments and links e0.
    let k1 = [0x41u8; 32];
    let e1 = e0.rekey(&w, &alice_bob, MembershipChange::MemberRemoved, &k1);
    assert_eq!(e1.epoch, 1);
    assert_eq!(e1.predecessor, Some(e0.kappa()));
    assert_eq!(e1.change, MembershipChange::MemberRemoved);
    assert!(!e1.is_member(&carol_k), "rekey excluded the removed member");
    for kw in &e1.wraps {
        assert!(
            w.unwrap(&carol_sk.to_bytes(), &kw.wrapped).is_none(),
            "removed member unwraps none of the rekeyed epoch's wraps"
        );
    }
    assert_eq!(
        w.unwrap(&alice_sk.to_bytes(), e1.wrap_for(&alice_k).unwrap())
            .unwrap(),
        k1,
        "a remaining member recovers the rekeyed key"
    );
    let _ = &bob_sk; // (bob symmetric to alice — asserted via the wrap set above)

    // Content sealed during epoch 1 stays inaccessible to carol forever — even after she rejoins,
    // she never held K1 (no retroactive access to the gap epoch).
    let gap_payload = b"content only epoch-1 members can read";
    let (gap_nonce, gap_ct) = seal_private(&cipher, &k1, gap_payload);

    // Re-add carol via `rekey` — fresh K2 to all three. Epoch increments again; carol is re-enrolled
    // for NEW content, but the epoch-1 gap remains closed to her.
    let k2 = [0x42u8; 32];
    let e2 = e1.rekey(&w, &all_three, MembershipChange::MemberAdded, &k2);
    assert_eq!(e2.epoch, 2);
    assert_eq!(e2.change, MembershipChange::MemberAdded);
    assert!(e2.is_member(&carol_k), "re-added member is enrolled again");
    let carol_k2 = w
        .unwrap(&carol_sk.to_bytes(), e2.wrap_for(&carol_k).unwrap())
        .unwrap();
    assert_eq!(carol_k2, k2, "re-added member recovers the current key");
    assert!(
        cipher.open(&carol_k2, &gap_nonce, &gap_ct).is_none(),
        "the re-added member's new key cannot open the epoch-1 gap content"
    );
}

#[test]
fn key_epoch_canonical_form_round_trips_and_dispatches() {
    let w = SealedBox;
    let network = address_bytes(b"net");
    let (_a_sk, a_pk, a_k) = member(7);
    let k = [0x33u8; 32];
    let e = KeyEpoch::genesis(
        network,
        vec![KeyWrap {
            member: a_k,
            wrapped: w.wrap(a_pk.as_bytes(), &k),
        }],
    );
    let bytes = e.canonicalize();

    let decoded = KeyEpoch::decode(&bytes).unwrap();
    assert_eq!(decoded.network, network);
    assert_eq!(decoded.epoch, 0);
    assert_eq!(decoded.change, MembershipChange::Genesis);
    assert_eq!(decoded.predecessor, None);
    assert_eq!(decoded.wraps.len(), 1);
    assert_eq!(decoded.wraps[0].member, a_k);
    assert_eq!(decoded.wraps[0].wrapped, e.wraps[0].wrapped);

    // Generic registry dispatch recovers the operand κs (network + member) without knowing the type.
    let refs = references(&bytes, REGISTRY).unwrap();
    assert!(refs.contains(&network) && refs.contains(&a_k));

    // A next-epoch form (with a predecessor) round-trips, including the empty-wraps edge.
    let e2 = KeyEpoch::next(network, e.kappa(), 1, MembershipChange::MemberAdded, vec![]);
    let d2 = KeyEpoch::decode(&e2.canonicalize()).unwrap();
    assert_eq!(d2.predecessor, Some(e.kappa()));
    assert_eq!(d2.epoch, 1);
    assert_eq!(d2.change, MembershipChange::MemberAdded);
    assert!(d2.wraps.is_empty());
}

#[test]
fn malformed_key_epoch_bytes_are_rejected_never_panic() {
    assert!(KeyEpoch::decode(&[]).is_err());
    assert!(KeyEpoch::decode(b"not-a-realization-at-all").is_err());
    // A truncated tail after a valid header must be a clean error, not a panic.
    let e = KeyEpoch::genesis(address_bytes(b"n"), Vec::new());
    let mut bytes = e.canonicalize();
    bytes.truncate(bytes.len().saturating_sub(2));
    assert!(KeyEpoch::decode(&bytes).is_err());
}
