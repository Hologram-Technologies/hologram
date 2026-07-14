# 04 — Networks: Distributed Content — Public, Restricted, Private

Decisions: D11, D12 (see `00-overview.md`).

## Principle

The network is content-addressed all the way down. **uor-native is the contract**:
`KappaSync` (`fetch` / `announce` / `discover`) over SPINE-4 frames with
verify-on-receipt at every hop; κ-XOR distance for routing; **κ is the only identity**
(a peer *is* the κ of its PeerEndpoint realization — no PeerIds, no Multiaddrs, no
second naming surface). This is today's working stack (`substrate/hologram-net-tcp`
DHT + holospaces `BareNetSync`/`PacketLink`), relocated per `01-crate-map.md`:
protocol + DHT logic in `crates/hologram-net` (no_std core), transport pumps in each
space.

## Transports (D11)

A transport is a dumb frame pump behind `PacketLink`/`TransportEndpoint` (or the HAL
`NetworkInterface` on bare metal). It carries SPINE-4 frames; it holds no identity, no
naming, no policy.

| Space | Transports |
|-------|-----------|
| holospaces-browser | WebRTC data channel (p2p, out-of-band signaling), WebSocket egress relay |
| holospaces-native | TCP (existing κ-XOR DHT wire), **iroh** (QUIC, NAT traversal, relays), **WebRTC endpoint + WebSocket listener** (browser interop) |
| holospaces-bare | HAL `NetworkInterface` pump (NIC/radio) |

**Interop rule: every pair of space kinds must share at least one transport.** The
matrix above satisfies it: browser↔browser (WebRTC), browser↔native (WebRTC or WS —
native carries a native WebRTC stack and a WS listener precisely so browser peers can
reach it directly), native↔native (TCP/iroh), bare↔native (TCP over NIC). A future
space's transport set is checked against this rule at spec time — a space no peer can
reach is not a peer.

**iroh's position**: strictly a transport pump for native spaces, adopted for what
TCP-plus-manual-signaling lacks (hole punching, relays, QUIC streams). iroh NodeIds and
keys are transport-internal plumbing — they MUST NOT appear in any realization, stored
form, contract type, or log-as-identity. If a mapping table (κ → current iroh address
hints) is needed, it is ephemeral routing state, never content. Wholesale adoption of
iroh's blobs/gossip/docs layers was considered and **rejected** (second identity model,
browser-sandbox mismatch).

## Networks as content (D12)

A **Network** — the VPC analogue — is itself a κ-addressed realization:

```
Network (realization)
├─ membership: [operator κ / peer-endpoint κ, …]
├─ policy:     CapabilitySet κ   (admission, fetch/announce/discover rights, quotas)
└─ meta:       parent-network κ (optional — networks nest by attenuation, like apps)
```

- Consistent with control-plane-as-content (holospaces ADR-018): creating or changing a
  network is publishing a new realization; peers resolve and apply it. No server, no RPC.
- **Terminology ladder (deliberate, honest naming)**:
  - **Public** — open policy; anyone may fetch/announce/discover.
  - **Restricted** (ships in P5) — capability-gated: `fetch`/`announce`/`discover`
    require a capability proof derived from membership; non-members refused at the
    protocol layer. Access control, **not** confidentiality.
  - **Private** (ships in Phase B/P6) — restricted **plus** payload encryption. The word
    "private" is reserved until encryption exists; docs, CLI, and API names use
    "restricted" for the capability-only tier so no user assumes confidentiality that
    isn't there.
- Delegation uses the existing `Delegation` realization; attenuation-only (law 5)
  applies to network capabilities exactly as to app capabilities.
- **Nesting is reserved, not implemented**: the `parent-network κ` field stays in the
  canonical form (so adding semantics later is not a format break — κ-invariance), but
  P5 implements **flat networks only**. Subnet policy-composition semantics (how child
  policy attenuates fetch/announce/membership across levels) get their own design when a
  real use case arrives.
- "Distributed OPFS" falls out of this: multiple holospaces on one restricted network share
  a κ-store view — any member resolves any member-announced κ, verify-on-receipt, dedup
  by construction. There is no separate "distributed filesystem" component to build; it
  is the KappaStore + KappaSync + Network policy composition.

## Creating & joining networks (commands)

A network is created by *publishing a realization*, not by provisioning a server — so
"create a custom network" is a local, offline act that produces a κ others resolve. CLI
verbs live in `05-tooling.md` (`hologram network …`); the mechanism:

```sh
# PUBLIC — open policy, anyone may fetch/announce/discover
hologram network create commons --public          # → network κ=b3:1a4f…
hologram net announce <app κ> --network commons    # content reachable to all

# RESTRICTED (P5) — capability-gated; only members resolve
hologram network create team --restricted          # → network κ; creator is founding member
hologram network delegate team --to <operator κ>   # grant membership (attenuated capability)
#   the invitee joins by resolving the network κ (control-plane-as-content — no RPC):
hologram network join <network κ>                  # adopt membership from the realization
hologram net announce <app κ> --network team       # non-members are refused at the protocol
hologram network show team                          # membership + policy (resolved from κ)

# PRIVATE (P6) — restricted + payload encryption; membership = capability + key access
hologram network create vault --private            # also provisions network keys (P6)
```

Notes that keep this consistent with the laws:

- **No server, no account.** `create` publishes a Network realization to your store; you
  or anyone shares its κ; joiners `resolve` that κ and adopt the policy. Membership
  changes are new realizations (append-only), not mutations.
- **Delegation is attenuation-only** (law 5): a member may grant a subset of their own
  rights via the `Delegation` realization — never more. Revoking is an append-only
  revocation event (see `07` R3 key lifecycle), not a delete.
- **Tier availability tracks the phases**: `--public` and `--restricted` ship in P5;
  `--private` (encryption) in P6. The CLI accepts `--private` earlier only to error
  clearly ("encryption lands in P6"), never to silently downgrade to restricted.
- **A network κ is shareable like any κ** — including at the URL rung: a boot link can
  name both an app κ and a network κ, so "open this link" can mean "join this team and
  run this app," all content-addressed (`08-form-factor.md`).

## Two enforcement layers, phased

### Phase A — capability gating → **restricted** networks (lands in migration P5)

Admission and routing control as described above. **Honest caveat, stated as spec**: a κ
that leaks outside a restricted network still names readable bytes if any member serves
it without checking policy, and bytes obtained out-of-band are readable. Capability
gating is access control, not confidentiality — which is exactly why this tier is named
"restricted," never "private."

### Phase B — payload encryption → **private** networks (lands after P5; requirements fixed now)

- Private networks are restricted networks that additionally encrypt payloads;
  membership = capability + key access.
- Requirements the design must satisfy (deferred design, see also `07`):
  1. Key distribution and rotation as κ-addressed content (no key server), building on
     `Delegation`; rotation must not orphan pinned content.
  2. Dedup semantics under encryption made explicit: ciphertext κ ≠ plaintext κ; the
     design must state where convergent encryption is used (dedup within a network) vs
     forbidden (confirmation-attack surfaces), and never silently degrade Law L3.
  3. Verify-on-receipt (SPINE-4) must hold on ciphertext without requiring decryption at
     relay hops.
  4. Bare-metal/no_std spaces must be able to participate (cipher choices with no_std
     impls; ChaCha20 machinery already exists in the runtime's entropy class).

## Protocol hardening (requirements, not full design)

- **Wire versioning**: SPINE-4 frames carry a protocol version; peers negotiate at
  session start and **refuse cleanly** on mismatch (typed error, no silent
  misinterpretation). Required before P5 — browser peers auto-update via Pages while
  native peers lag, so cross-release contact is the normal case, not the edge case.
- **Bounded resolution**: `resolve_closure` enforces caller-supplied depth/size/count
  budgets (capability-scoped, like all budgets — no hardcoded caps, but no unbounded
  walks of hostile manifests either). Frame and payload sizes are bounds-checked at the
  codec; fuzz targets for the frame codec and DHT message parser are CI-permanent from
  P5 onward.
- **Abuse economics (open item, pre-GA of public networks)**: PeerEndpoint identities
  are free to mint, so κ-grinding toward a target's DHT neighborhood (eclipse) and
  announce-spam are cheap on a public network. Restricted networks are inherently
  protected (membership gate). For public networks, the design owed before they're
  promoted beyond experimental: admission cost or rate discipline per peer, bucket
  diversity policy, and announce quotas — expressed as Network policy, consistent with
  everything else.
- **Bootstrap & signaling ownership (P5 scope)**: the P5 demo stands on a concrete
  story, specced with it: initial peer discovery via a published PeerEndpoint κ
  (out-of-band exchange or well-known gateway), and browser WebRTC signaling via the
  existing WS gateway or manual SDP exchange. "Out-of-band" is an accepted answer for
  P5; an *unowned* answer is not.

## Open item: durability & replication policy

What is designed above is **access** distribution (any member can resolve any member's
content, verify-on-receipt). Deliberately not yet designed: **durability** — who is
obligated to keep content alive, replication factor, what `pin` means network-wide (a
member pinning κ on the network vs. locally), and how GC interacts with remote
reachability. Requirements to carry into that design: policy must be expressible per
Network (capability-based, like everything else); no silent data loss when the sole
holder of a κ leaves; bare/edge spaces must be able to opt out of holding obligations.
Single-peer durability is part of the same design: **browser storage is evictable** (the
UA may reclaim OPFS under pressure) — the browser space must request persistent storage
where available and must *detect* eviction (κs that stop resolving locally) rather than
discover it as corruption; an evicted browser peer degrades to re-fetching from its
networks. This is a post-P5 design doc alongside encryption (Phase B); nothing in P1–P5
may foreclose it (same rule as `07-governance-requirements.md`).

## Conformance

`hologram-tck` gains a sync/network battery: verify-on-receipt (reject corrupt frames),
κ-XOR routing correctness, capability refusal on restricted networks, and (Phase B)
ciphertext verification. Every space's transport pump must pass it, same rule as storage.
