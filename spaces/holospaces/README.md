# holospaces

> UOR-native boot layer over the hologram substrate — provisions and runs bootable, content-addressed environments and ships the Hologram Platform Manager.

holospaces is a Space implementation: a thin composition of the hologram substrate, consumed by reference (ADR-003, ADR-006) and never re-implemented. Its building blocks map to the architecture (arc42 chapter 5), each upholding the laws — L1 identity is content; L2 operate on canonical forms; L3 the store is memory, RAM a cache; L4 everything goes through the substrate; L5 verify by re-derivation. The `docs/` tree is authoritative; this code traces back to it.

## What it provides

- `realizations` — the canonical-form layer: `Kappa` (`hologram_space::KappaLabel71`) minted and verified by re-derivation, and the `Holospace` realization composing a `ContainerManifest` and a `CapabilitySet` (CC-1).
- `boot` — the environment-agnostic core: ingest, `provision`, resolve, and a `Session` driving the container lifecycle through hologram's `ContainerRuntime`.
- `engine` — the *.holo Engine*, running a `.holo` compute artifact via the hologram executor (CC-2, `std` only).
- `surface` / `disk` — the Execution Surface (the κ-addressed Wasm code-module form, CC-6) and the κ-disk (a `KappaStore`-backed `BlockDevice`, CC-7).
- `emulator` / `machine` — the system emulator core with two ISA targets (ADR-021) over a shared substrate-backed device bus: a RISC-V RV64GC machine (CC-9) and an AArch64 machine booting real `arm64` Linux to userspace (CC-35/36/37).
- `peer` / `identity` / `manager` / `projection` — the `Peer` composing storage · network · runtime; self-sovereign identity and the operator `Roster` (R5); the Platform Manager `View` + Intent surface; and the Workspace Projection over a running holospace (CC-11).
- `assembly` / `oci` / `content_net` / `config` — the Layer Assembler (OCI layers → bootable ext4, CC-10→CC-7), OCI image ingestion, the uor-native content network (`BareNetSync`, CC-38), and content-addressed reconfiguration of a running holospace (CC-28).
- `substrate` — the hologram seams (`KappaStore`, `KappaSync`, `ContainerRuntime`, the `.holo` executor) re-exported for convenience.

## Features

- `std` (default) — the host-side provisioning surfaces: the Dev Container ingestor (CC-4), the Wasm validator (CC-5), the `.holo` engine, Compose/Dockerfile resolution, and Zstandard layers. With `--no-default-features` the crate is `no_std` + `alloc`: the portable boot core the browser and bare-metal peers compile.
- `net` — the internet import boundary (ADR-013): the Repository Fetcher and OCI Image Fetcher over a blocking HTTP(S) client with rustls (CC-20). Host-only, kept out of `default` so the portable peer never links a TCP/TLS stack.
- `cc44-trace` — dev-only x86-64 guest serial/`rip`/TSC tracing for the CC-44 long-mode bring-up; never enabled in CI.

Part of the [hologram](../../README.md) workspace.
