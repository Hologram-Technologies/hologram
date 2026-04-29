# ADR-052: uor-foundation 0.3.0 Domain Decisions

**Status:** Accepted
**Date:** 2026-04-28 (accepted 2026-04-29)
**Deciders:** Ari (project lead)
**Related:** ADR-001 (BLAKE3 archive checksums), Plan 074 (uor-foundation 0.3.0 upgrade)

## Context

Plan 074 documented that the `uor-foundation 0.1.4 → 0.3.0` upgrade
expands beyond mechanical renames. Four decisions cannot be answered
by reading the compiler errors — they require project policy:

1. Which **digest algorithm** does hologram standardise on for
   `Element::digest_algorithm`?
2. What **canonical-bytes encoding** does each `Element` impl produce?
3. What do **`Datum::stratum`** and **`Datum::spectrum`** mean for
   `ByteDatum`, `RingDatum`, and the `q1`/`q2`/`q3` datum types?
4. Which concrete types back **`HostTypes::HostString`** and
   **`HostTypes::WitnessBytes`** for `HoloPrimitives` and
   `PrismPrimitives`?

This ADR proposes a default for each so the migration can proceed
without re-litigating policy mid-port.

## Decision

### 1. Digest algorithm: BLAKE3

Pin `Element::digest_algorithm` to **`"blake3"`** for every `Element`
impl in this repository.

**Rationale.** ADR-001 already established BLAKE3 as hologram's archive
checksum (replacing CRC32 in format v2). Weight deduplication uses
BLAKE3. Adopting it here keeps a single hash function across the
ecosystem — no second algorithm to provision keys for, no second test
matrix to maintain. `uor-foundation 0.3.0` lists BLAKE3 as the primary
acceptable algorithm; `"sha256"` is the fallback secondary, which we
explicitly do not need.

**Implementation note.** Provide a single `BlakeDigest` helper that
hashes canonical bytes once at construction and caches the 32-byte
output, so repeated `digest()` calls are O(1).

### 2. Canonical-bytes format: Amendment 43 §2 verbatim

Adopt Amendment 43 §2 — `header(k) || le_bytes(x, k+1)` — as the
canonical-bytes encoding for every `Element` impl. Specifically:

- `header(k)` = single byte encoding the witt level index
  (`W8 → 0x00, W16 → 0x01, W24 → 0x02, W32 → 0x03`).
- `le_bytes(x, k+1)` = the underlying ring element value as
  `(k+1) * 8`-bit little-endian bytes (1, 2, 3, or 4 bytes
  respectively).

**Rationale.** This is the format the foundation layer already expects
for cross-implementation interop. Hologram's `RingLevel` enum already
uses the same 0/1/2/3 numbering (verified in `engine.rs:407` archive
encoding), so the header byte is a direct cast — no translation table
required.

**Archive format impact.** The existing `.holo` archive format v2
stores `RingLevel` as `u8` and ring values as their natural byte width.
The on-disk layout already matches Amendment 43 §2 byte-for-byte. **No
archive format migration is required.** The round-trip test added in
Plan 074 §3b should be extended to assert canonical-bytes byte equality
against this layout.

### 3. `Datum::stratum` and `Datum::spectrum` semantics

Adopt the foundation docstring interpretation directly:

- **`stratum() -> u64`**: the ring-layer index, `k ∈ {0, 1, 2, 3}` for
  `Q_k`. Implementation: cast `RingLevel as u64`. Identical to the
  archive's level encoding.

- **`spectrum() -> u64`**: the bit-pattern representation of the
  datum's value in `Z / (2^n) Z`, where `n = (k + 1) * 8`. For
  hologram's existing datum types this is just the underlying integer
  reinterpreted as `u64` (zero-extended for `Q_0`/`Q_1`/`Q_2`,
  identity-cast for `Q_3`).

**Rationale.** The foundation defines the abstract semantics; hologram's
existing storage representation is already a faithful encoding. Mapping
`stratum/spectrum` to these existing fields means no new state — a
helper trait `RingDatumExt` in `hologram-ring` can provide the default
impls for any `Datum` that already exposes `(level, value)`.

**`ByteDatum` (the `Q_0` byte datum).** `stratum() = 0`,
`spectrum() = self.byte() as u64`.

**Generic `RingDatum<H, k>`.** `stratum() = k as u64`,
`spectrum() = self.value().into()`.

### 4. `HostTypes` binding: `DefaultHostTypes`

Bind `HoloPrimitives` and `PrismPrimitives` to:

- `Decimal = f64` (or whatever `DefaultHostTypes::Decimal` resolves to
  in 0.3.0 — verify at port time).
- `HostString = str` (unsized, borrowed).
- `WitnessBytes = [u8]` (unsized, borrowed).

**Rationale.** Hologram does not distinguish a project-specific host
string type today — every site that previously consumed
`Primitives::String` was already converting to/from `&str`. Same for
witness bytes: `&[u8]` is the universal currency. Picking the unsized
defaults avoids forcing every call site through an owned-type
allocation.

**Old-trait fields with no replacement.** The 0.1.4 `Primitives` trait
had `Integer`, `NonNegativeInteger`, `PositiveInteger`, `Boolean`
slots. These have no `HostTypes` analogue and must be sourced from
hologram-side types instead:

- `Integer` / `NonNegativeInteger` / `PositiveInteger` — replace with
  concrete `i64` / `u64` / `NonZeroU64` at usage sites. Audit the
  ~120 `<P: Primitives>` trait bound sites in `hologram-ring` and
  `hologram-core` for any that load-bear on those slots.
- `Boolean` — replace with native `bool`.

If audit reveals a load-bearing dependency on the old generic slots,
re-open this ADR to adopt a per-repo associated type instead.

## Consequences

### Positive

- Each migration step has a single agreed answer, so the actual port
  is mechanical compiler-driven work.
- Reusing BLAKE3 + Amendment 43 §2 means no archive format break.
- `stratum/spectrum` map onto existing fields — no new state in datum
  types.
- `HostString = str` / `WitnessBytes = [u8]` keeps the migration
  zero-allocation at borders.

### Negative

- Locks hologram to BLAKE3 across rings. If a future contract demands
  SHA-256 for some interop case, that's a per-`Element`-impl override
  (not a hologram-wide flip).
- The `Integer`/`Boolean` slot disappearance forces an audit pass;
  some `<P: Primitives>` bounds may turn into concrete-type
  signatures rather than generic ones.
- Spectrum-as-value-cast assumes hologram never wants a richer
  spectral encoding (e.g. Walsh-Hadamard projection). If we later
  change our minds, every persisted `spectrum()` consumer would need
  re-derivation — but spectrum isn't currently persisted, so this is
  reversible.

### Alternatives considered

1. **SHA-256 for digest.** Rejected — second algorithm to provision,
   no consistency benefit, contradicts ADR-001.
2. **Per-datum custom canonical-bytes (e.g. variable-length).**
   Rejected — would force an archive-format migration with no
   demonstrated benefit over the foundation's chosen layout.
3. **Owned `String`/`Vec<u8>` for `HostTypes`.** Rejected — forces
   allocations at every API boundary that has no current allocation,
   and provides no information that `&str`/`&[u8]` wouldn't carry.
4. **Defer one or more decisions and gate them on later audit.**
   Rejected — Plan 074 is already on a dedicated branch; deferring
   means the branch lingers indefinitely. Better to commit to defaults
   in this ADR and revise if audit findings demand it.

## References

- ADR-001 — BLAKE3 archive checksums (origin of hologram's hash choice).
- Plan 074 — `specs/plans/074-uor-foundation-0.3.0-upgrade.md`,
  scope addendum dated 2026-04-28.
- `uor_foundation::kernel::address::Element` (0.3.0).
- `uor_foundation::kernel::HostTypes` (0.3.0).
- Amendment 43 §2 (canonical-bytes layout).
