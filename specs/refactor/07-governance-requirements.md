# 07 — Governance Requirements: Traceability, Auditability, Attestation, Data Governance

Decision: D19 (see `00-overview.md`). **Requirements only** — the full design is a
post-P5 spec. This document exists so no boundary drawn by the refactor forecloses these
capabilities.

## Requirements

### R1 — Traceability
Every artifact must be traceable to its inputs by κ alone: a `.holo`'s manifest
references its layers; a compiled plan's certificates reference its source terms; a
snapshot references the manifest + capability set that produced it. **Boundary rule**:
every new realization introduced by this refactor (AppManifest, Network, hoisted
Roster/Configuration) must embed its operand κs (SPINE-2/3) so `references()` yields the
full provenance closure. No side tables.

### R2 — Auditability
The audit trail is a κ-chained, append-only event log (SPINE-5 gives tamper-evidence for
free): lifecycle transitions (spawn/suspend/resume/terminate), capability delegations,
network membership changes, configuration applications. **Boundary rule**: the runtime's
lifecycle seams (hologram-runtime) and the network's policy decisions (hologram-net +
space transports) must emit events through one seam that can later be pointed at the
κ-chain — no lifecycle path may bypass it.
The audit trail's own access control needs no new mechanism: audit events are κ-content
like everything else, governed by the network tiers of `04-networks.md`
(public/restricted/private). No bespoke ACL system may be invented for logs.

### R3 — Attestation
`.holo` already carries per-node certificates; v3 adds per-layer certificates (03).
Attestation extends this to *where and how* something ran: a space must be able to sign
"session S booted app κ under capability set κ on space-impl κ at engine κ".
Signing introduces keys, and keys are not κs — so the design MUST bind signing keys to
κ-addressed identities the way Operator identity already works (self-sovereign key
material published/referenced as content), never as a second identity surface smuggled
in through certificates (law 2 applies to attestation too).
**Boundary rules**:
- The space contract keeps space-impl identity expressible as a κ (a space build is
  itself content).
- The FFI/Client surface must not strip certificates — inspection APIs expose them.
- Snapshot realizations must leave room (extension, not format break) for an attestation
  section.

### R4 — Data governance
Governance = capability policies on networks (04): who may store, fetch, announce which
content, with quotas. **Boundary rules**:
- Capability checks stay at the import/protocol boundary (never sprinkled in business
  logic), so policy can tighten without code motion.
- Resource accounting seams (the runtime's storage-quota ledger, fuel budgets) remain
  per-capability, not global, so per-network/per-operator accounting composes later.
- Retention/erasure reality is stated honestly: content-addressed append-only stores do
  not "delete"; governance operates via unpin + GC reachability + network policy, and the
  future design must document this model rather than promise erasure semantics κ-stores
  cannot give.

## Non-requirements (for now)

Compliance regimes (SOC2/GDPR mappings), key-escrow, identity federation, policy
languages. None may be designed yet; none may be made impossible by P1–P6 either — any
phase PR that would foreclose R1–R4 must call it out and amend this doc first.
