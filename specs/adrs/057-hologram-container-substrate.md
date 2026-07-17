# ADR-057: Hologram container substrate (deployment layer)

**Status:** Accepted 2026-05-27
**Relates to:** ADR-031 (hologram is a Prism application), ADR-052 (BLAKE3 σ-axis),
ADR-018 (zero-movement pool), ADR-055/056 (UOR-native realizations)
**Full architecture:** [container-substrate-architecture.md](../docs/container-substrate-architecture.md)
**Normative source:** *Hologram Container Specifications* + *Hologram Bare-Metal Substrate Specification*

## Context

The container specs propose a **deployment substrate** — Container Runtime + Storage Layer
(`KappaStore`) + Network Layer (`KappaSync`) over a κ-label graph, across browser / WASI-native /
bare-metal — that hosts arbitrary Wasm containers. `hologram-ai` is one consumer; other platforms
will be others; containers must support **arbitrary workloads**.

The existing `hologram` workspace is the **compute substrate** (Prism tensor runtime). The specs
call the deployment substrate "independent of Prism," but `uor-addr` — which mints every κ-label —
depends transitively on `uor-prism`. The architecture must reconcile this honestly and stay
**uor-native with no traditional fallback and no arbitrary structural limits**.

## Decision

1. **New crate family** under `substrate/` (same workspace) that **reuses hologram's optimal,
   externally-validated κ-native primitives** — `hologram-host` (σ-axis) + `uor-addr` (witnessed
   composition) — directly. **Not** `hologram-archive` (it depends on `hologram-compute`, the tensor
   kernel engine; G-E1). `address_bytes`/`derive_label` are byte-identical reimpls over the same
   `HologramHasher`. The compute engine (`hologram-exec`/`-backend`/`-archive`/`-ops`/`-graph`/
   `-compiler`) is a *container* dependency, never in the store/route host path — `cargo tree` CI
   gate (RZ class).

2. **The spec's "Hologram does not link Prism" is rejected.** It is already false (`uor-addr →
   uor-prism`) and, more importantly, hologram is *optimal* and should be used where it makes sense.
   The boundary is **reuse hologram's κ-native primitives (encouraged) vs. embed tensor compute in
   the host path (no — a container's job)**, and it is bounded by decision 6: everything upholds
   hologram's performance contract. There is no non-UOR identity, addressing, or dedup anywhere
   (SPINE-1); addressing is the *same* identity layer the compute substrate proved.

3. **The uor-native spine is the architecture** (SPINE-1..6): canonical-bytes-or-nothing,
   realization-IRI on every artifact, `references()` as the one structural relation, verify-by-
   re-derivation, append-only + eviction-tolerant, and **no fallback / no arbitrary cap**.

4. **Async per the specs**, core executor-agnostic (`KappaStore` sync; `KappaSync`/`ContainerRuntime`
   async via `async-trait`). Backends bring the runtime (tokio / embassy / wasm-futures).

5. **Bare-metal is first-class from Phase 0** (HAL traits + `*-bare` skeleton crates compile no_std),
   hardened in Phase 5 (UEFI, drivers, no_std libp2p/rustls/Wasmtime forks).

6. **Efficiency held to PV-parity:** content-addressing is the efficiency mechanism (zero-copy,
   idempotent dedup, warm-start, bounded walks), measured by a new **SP** conformance class under
   `just perf`. Containers and their parts uphold hologram's benchmarks.

7. **First slice = all-three skeleton:** every §8 trait + supporting types + `Realization`/
   `references()` registry + real `verify_kappa`, with stub backends across all three substrates and
   `MemKappaStore` as the one working impl + conformance fixture.

## Rejected alternatives

- **Renaming the Prism crates / reassigning the `hologram` name.** Out; compute crates stay as-is.
- **Pretending zero Prism linkage / firewalling the substrate from hologram.** Rejected: dishonest
  (`uor-addr → uor-prism`) *and* wasteful — hologram's addressing/σ-axis/store primitives are
  optimal and validated; reimplementing them would be weaker and unvalidated.
- **A "convenience" non-canonical serialization or addressing fast-path.** Violates SPINE-6.
- **Reimplementing addressing over raw `uor-addr` to avoid `hologram-archive`.** Rejected: we
  *reuse* `hologram-archive`'s `address_bytes`/`derive_label`/`compose_*_blake3` + witnesses — they
  are the externally-validated κ-native path. The only thing kept out of the host path is the tensor
  compute engine, bounded by the performance contract.

## Consequences

- A **grounding review** (architecture §9) checked the spec against `uor-addr 0.2.0`/repo reality and
  found several non-UOR-native mechanisms and spec defects — corrected toward hologram's proven
  patterns: `references()` regrounded as the inverse projection of a **witnessed composition** (not a
  byte-scan); identities are **witnessed compositions** of operand labels (not blake3-of-concat);
  capability containment is grounded in the foundation's **`TypeInclusion`/`SubtypingLattice`/`Grounded`**
  subtyping lattice (the spec's "E₈ filtration" is misnamed — `uor-foundation-0.5.2/src/user/type_.rs:293–309`,
  `enforcement.rs:8928`; Capability Set = `ConstrainedType`, delegation = constraint-addition = lattice
  descent, **no fallback**); `spawn` takes a Capability Set **κ-label** (not a struct); `sha256d` removed
  (not in uor-addr); κ-label width is per-axis so the substrate's artifacts are blake3 `<71>`, stored-content
  keys axis-polymorphic. Decisions G-B1/G-A3 resolved; remaining items (G-C/G-D) tracked per phase.
- New conformance classes SPINE / ST / NW / CR / RZ / TR / SP (CONFORMANCE.md).
- Justfile `wasm`/`embedded` recipes gain the new crates immediately; portability cannot silently
  regress.
- The bare-metal no_std forks (libp2p/rustls/Wasmtime) are an explicit, tracked dependency risk.
