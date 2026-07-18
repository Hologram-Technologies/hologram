//! **Exit-criteria demo — P5** (spec 06 §P5): a two-node restricted network with a non-member
//! refused at the protocol boundary.
//!
//! A member peer is admitted; a non-member peer is refused. The gate's only inputs are
//! `(tier, is_member)` — no payload, no store state, no business data — so the refusal is
//! *structurally* at the protocol boundary (`NetworkTier::admits`), not buried in application logic.
//! (The live browser-peer ↔ native-peer variant over a real transport is the heavy-CI demo; this is
//! the always-green in-process witness of the same admission rule.)
//!
//! Run: `cargo run -p hologram-space --example restricted_network`

use hologram_space::{address_bytes, Network, NetworkOp, NetworkTier};

fn admits(network: &Network, peer: &hologram_space::KappaLabel71, op: NetworkOp) -> bool {
    // The protocol boundary: admitted iff the peer is a member (for restricted/private tiers).
    network.tier.admits(op, network.membership.contains(peer))
}

fn main() {
    let alice = address_bytes(b"peer-alice");
    let bob = address_bytes(b"peer-bob");
    let mallory = address_bytes(b"peer-mallory"); // never enrolled

    // A restricted "team" network with two members.
    let team = Network {
        membership: vec![alice, bob],
        policy: address_bytes(b"team-policy-caps"),
        parent: None,
        tier: NetworkTier::Restricted,
        key_ref: None,
    };
    println!("restricted network: {} members", team.membership.len());

    // Every network verb is gated the same way — members pass, the non-member is refused.
    for op in [NetworkOp::Fetch, NetworkOp::Announce, NetworkOp::Store] {
        assert!(
            admits(&team, &alice, op),
            "member alice admitted for {op:?}"
        );
        assert!(admits(&team, &bob, op), "member bob admitted for {op:?}");
        assert!(
            !admits(&team, &mallory, op),
            "NON-MEMBER mallory refused for {op:?}"
        );
    }
    println!("  members alice + bob: admitted for every op");
    println!("  non-member mallory: REFUSED at the protocol boundary for every op");

    // Contrast: a public network is a pure tier decision — it admits anyone, member or not.
    let public = Network {
        membership: Vec::new(),
        policy: address_bytes(b"open-policy"),
        parent: None,
        tier: NetworkTier::Public,
        key_ref: None,
    };
    assert!(
        admits(&public, &mallory, NetworkOp::Fetch),
        "public admits a non-member"
    );
    println!("  (a public network, by contrast, admits anyone)");

    println!("\nP5 two-node restricted-network (non-member refused) demo: OK");
}
