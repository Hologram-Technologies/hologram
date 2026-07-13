# 04 — Networks: Distributed Content, Private/Public Meshes

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
| holospaces-native | TCP (existing κ-XOR DHT wire), **iroh** (QUIC, NAT traversal, relays) |
| holospaces-bare | HAL `NetworkInterface` pump (NIC/radio) |

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
- **Public network**: open policy — anyone may fetch/announce/discover.
- **Private network**: `fetch`/`announce`/`discover` require a capability proof derived
  from membership; non-members' requests are refused at the protocol layer. Delegation
  uses the existing `Delegation` realization; attenuation-only (law 5) applies to network
  capabilities exactly as to app capabilities.
- "Distributed OPFS" falls out of this: multiple holospaces on one private network share
  a κ-store view — any member resolves any member-announced κ, verify-on-receipt, dedup
  by construction. There is no separate "distributed filesystem" component to build; it
  is the KappaStore + KappaSync + Network policy composition.

## Two enforcement layers, phased

### Phase A — capability gating (lands in migration P5)

Admission and routing control as described above. **Honest caveat, stated as spec**: a κ
that leaks outside a private network still names readable bytes if any member serves it
without checking policy, and bytes obtained out-of-band are readable. Capability gating
is access control, not confidentiality.

### Phase B — payload encryption (lands after P5; requirements fixed now)

- Private networks additionally encrypt payloads; membership = capability + key access.
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

## Conformance

`hologram-tck` gains a sync/network battery: verify-on-receipt (reject corrupt frames),
κ-XOR routing correctness, capability refusal on private networks, and (Phase B)
ciphertext verification. Every space's transport pump must pass it, same rule as storage.
