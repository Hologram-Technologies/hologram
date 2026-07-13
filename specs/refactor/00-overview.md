# 00 — Refactor Overview: The Hologram Consolidation

Status: **Accepted direction** (interrogation session 2026-07-13). Implementation not started.
Supersedes: the raw notes in `specs/scratch.md`.
Companion docs: `01-crate-map.md` … `07-governance-requirements.md`.

## Vision

A full end-to-end experience of loading a holospaces runtime: a fully native, secure
execution environment — initially browser-focused, ultimately distributed. Holospaces is a
browser-native, bare-metal-capable, model-centric runtime environment that cross-compiles
arbitrary workloads across architectures and manages their lifecycle.

This refactor consolidates the ecosystem (`hologram`, `holospaces`, and the transitional
`substrate/` tree) into **one repository with one contract layer**, leaving the project
clean, clear, and simplified, with explicit division lines between responsibilities.

## Definitions (ubiquitous language)


| Term                  | Definition                                                                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **workload**          | A binary with an exit code — any executable: a shell (`/bin/sh`), a container machine, a runtime (e.g. an AI model).                                        |
| **host**              | An environment that includes emulator support (abstraction) over the hardware.                                                                              |
| **space**             | A place hologram executes: a host that implements the hologram space contract (`hologram-space`). Examples: browser, native desktop, bare-metal/esp32, iOS. |
| **space contract**    | The trait set + laws + TCK conformance battery that every space implements identically. Defined by hologram, never by a space.                              |
| **application (app)** | A `.holo` archive of κ-addressed layers, each with an entrypoint and exit-code semantics, composable with other apps (see `03-holo-format.md`).             |
| **κ (kappa)**         | A content-addressed label (BLAKE3 σ-axis derivation via uor-addr). The **only** identity in the system — for bytes, apps, peers, operators, networks.       |
| **realization**       | A canonical, IRI-tagged byte form embedding operand κ-labels (SPINE-2/3), e.g. ContainerManifest, CapabilitySet, Network.                                   |
| **peer**              | One composed environment: a store + runtime (+ optional sync) for a given space.                                                                            |
| **operator**          | Self-sovereign identity owning a roster of holospaces; no server accounts.                                                                                  |




## The laws

These are repo-wide invariants. Any code or spec that violates one is wrong by definition.

1. **SPINE-1..6** (inherited from the substrate, now global): canonical-bytes-or-nothing;
  IRI-tagged realizations; composition = identity; verify by re-derivation; append-only;
   no fallback / no arbitrary caps.
2. **κ-only identity.** No second naming surface — no UUIDs, PeerIds, Multiaddrs, paths, or
  hostnames as identities. Transport-internal identifiers (e.g. iroh NodeIds) never leak
   into contracts or stored forms.
3. **Contracts are hologram's; spaces are anyone's.** The space contract is defined once
   in `hologram-space`; every space implements the identical surface. Platform differences
   live *behind* the traits, never in them. Conformance = passing `hologram-tck`. The
   contract is open: a space may live in any repository — no sealed traits, no
   crate-private seams, no in-tree privilege. (Decisions D2, D21.)
4. **Async contracts, sync compute.** I/O-shaped contract traits are async; the tensor
  execution hot path is synchronous, zero-allocation, `no_std`-capable. The session
   boundary is the only async↔sync seam. (Decision D14.)
5. **Capability attenuation only.** Delegated capabilities are always a subset of the
  grantor's; amplification is unrepresentable. Enforced at every spawn/import boundary.
6. **One programmatic surface.** All human/foreign-language entry points (CLI, FFI, SDKs)
  are thin shells over the `Client` facade. Behavior lives in exactly one place.
7. **Rust best practice throughout**: thiserror in libraries / anyhow only in binaries; no
  `unwrap()`/`expect()` on production paths; minimal `pub` surface (`pub(crate)` default);
   newtypes over bare primitives; exhaustive matching on business enums; workspace-level
   lints; zero unsafe outside documented FFI/SIMD boundaries with `# Safety` contracts.



## Decision record

Resolved 2026-07-13. Each decision's owning spec is listed; details live there, not here.


| #   | Decision                 | Resolution                                                                                                                                      | Owning doc |
| --- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| D1  | Repo scope               | holospaces merges into this repo under `spaces/`; **hologram-ai and hologram-apps stay external** as flagship consumers of the published facade | 01         |
| D2  | Contract ownership       | hologram defines the space contracts + TCK; `holospaces-*` implement the identical surface                                                      | 02         |
| D3  | "backend" disambiguation | tensor kernels → `hologram-compute`; environment contract → `hologram-space`; the word "backend" is retired                                     | 01         |
| D4  | Facade                   | single `hologram` facade crate, feature-gated; users never import subcrates                                                                     | 01, 05     |
| D5  | Impl grouping            | group by space — each `holospaces-x` owns its full stack; wasm engines shared internally as `hologram-runtime` features                         | 01, 02     |
| D6  | FFI                      | hologram owns FFI over the Client facade; uniffi (py/swift), wasm-bindgen (TS), napi-rs (node, optional), cbindgen C header                     | 05         |
| D7  | Platform manager         | Manager/Operator/Roster/Configuration + Peer/Session models move into `hologram-runtime`; views stay per-space                                  | 01, 02     |
| D8  | .holo format             | one format: `.holo` v3 is the application container; tensor-only archives are the degenerate single-layer case                                  | 03         |
| D9  | App composition          | capability-attenuated nesting (parent→child κ refs + delegated CapabilitySet); siblings share only the KappaStore                               | 03         |
| D10 | App views                | portable surface capability + optional per-space native view layers; design-system SDKs out of scope                                            | 03         |
| D11 | P2P stack                | uor-native (KappaSync/SPINE-4/κ-XOR DHT) is the contract; iroh is one transport pump for native spaces                                          | 04         |
| D12 | Networks                 | Network = κ-realization (membership + policy); capability gating first, payload encryption as a following phase                                 | 04         |
| D13 | Tooling                  | one `hologram` binary; `Client` facade under everything                                                                                         | 05         |
| D14 | Async posture            | contracts async, compute sync — codified as law 4                                                                                               | 02         |
| D15 | Crate map                | ~15 core crates + 4 space crates; `substrate/` eliminated                                                                                       | 01         |
| D16 | Releases                 | publish all crates to crates.io, lockstep workspace version                                                                                     | 01         |
| D17 | Migration                | phased, always-green; holospaces V&V passes at every phase boundary                                                                             | 06         |
| D18 | UOR crates               | remain external crates.io deps; hologram is the canonical integrator                                                                            | 01         |
| D19 | Governance               | requirements captured now; full design deferred post-P5                                                                                         | 07         |
| D20 | Rust practice            | best-practice Rust per law 7                                                                                                                    | all        |
| D21 | Space extensibility      | spaces are external-repo capable: the contract + TCK + engines are fully usable from outside this workspace; in-tree spaces are a convenience, not a privilege, and remain extractable to their own repos by a path→version dep swap | 02, 01     |




## Non-goals of this refactor

- Design-system / view SDKs (Material-style, SwiftUI-style) — future separate projects
that plug into the surface seam defined in `03-holo-format.md`.
- Bringing `hologram-ai` or `hologram-apps` into this repository — they migrate to the
published facade after the first lockstep release (P3), from their own repos.
- Vendoring the UOR crates (`uor-foundation`, `uor-foundation-sdk`, `uor-prism`,
`uor-prism-tensor`, `uor-addr`) — they stay external, published by the UOR Foundation.
- Full governance/attestation design and network payload-encryption key management —
requirements only (see `07`), designed after P5.



## Success criteria

1. `substrate/` no longer exists; its contracts and implementations live in the target
  crates of `01-crate-map.md`.
2. All previously working holospaces functionality is intact and verified: CC conformance
  catalog, QEMU differential oracles (RISC-V/AArch64/x86-64), Playwright browser tests,
   substrate TCK.
3. Exactly one binary named `hologram`; exactly one public crate (`hologram`) that users
  import with features.
4. hologram-ai builds against the published `hologram` facade with no git-pinned deps.
5. A new space can be brought up by implementing `hologram-space` traits and passing
  `hologram-tck` — nothing else.
6. That new space can live in an **external repository**, depending only on published
  crates (D21): the TCK runs there as a dev-dependency, and `Client` accepts the space
  with no facade change. Proof obligation: extracting any in-tree space to its own repo
  must require nothing beyond swapping path deps for version deps.

