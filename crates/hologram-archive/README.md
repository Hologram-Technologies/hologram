# hologram-archive

> The `.holo` archive format: kernel calls, schedule, deduped weights, certificates, and a footer fingerprint.

`hologram-archive` defines the `.holo` archive format described in spec Part X.
It wraps a compiler output for distribution — the kernel-call sequence, the
schedule, BLAKE3-deduped weights, the shape and dtype registries, per-node
certificates, trace, and metadata — sealed with a footer fingerprint.

Content identities are σ-projection-grounded, replayable κ-labels composed via
the E₈ categorical operations (ADR-061), not bare hashes: the archive routes its
footer and weight fingerprints through prism's canonical BLAKE3 `HashAxis` (via
`hologram-types`' `HologramHasher`) and holds no direct blake3 runtime
dependency.

The crate is `no_std` + `alloc` by default (matching prism / uor-addr) so the
reader/writer runs in wasm and on embedded targets; the `std` feature adds
host-only amenities and the optional `memmap2` loader path.

## What it provides

- `HoloWriter` / `HoloLoader` — write and load `.holo` archives; `LoadedPlan` and `ContentBlobs` are the loaded result.
- `HoloHeader`, `SectionKind`, `FORMAT_VERSION`, `MAGIC` — the on-disk format primitives.
- `ContentLabel`, `KappaLabel`, `derive_label`, `derive_label_witnessed`, `compose_model`, `address_ring`, `AddressWitness`, `AddressOutcome` — κ-label derivation and composition.
- `WeightStore`, `WeightProvider`, `WeightFingerprint` — BLAKE3-deduped weight storage and lookup.
- `WarmEntry`, `derive_cone_lattice` — warm-store codec support.
- `PortDescriptor` and the `decode_exec_plan` / `decode_ports` / `decode_weights` decoders.
- `ArchiveError` — the crate error type.

## Features

- `std` (default) — host-only amenities plus the `memmap2` loader path.
- `model-formats` — model-format addressing for hologram-ai (the gguf / onnx `uor-addr` realizations).
- `compression` — compression support for archive sections.

## Targets & build notes

`no_std` + `alloc` by default; opt into `std` for the memory-mapped loader and
host amenities.

Part of the [hologram](../../README.md) workspace.
