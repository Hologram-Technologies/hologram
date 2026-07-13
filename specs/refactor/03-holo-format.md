# 03 — `.holo` v3: The Application Container

Decisions: D8, D9, D10 (see `00-overview.md`).

## Principle

**One format.** `.holo` is hologram's native executable/archive — the ELF analogue —
covering everything from a compiled tensor plan to a full multi-layer application. A
tensor-only archive is simply the degenerate single-layer case. One codec, one loader,
one verifier, one identity scheme (κ).

Today's `.holo` (FORMAT_VERSION 2 in `crates/hologram-archive`: magic `HOLO`, SectionKind
0–14, BLAKE3-deduped weights, per-node certificates, footer fingerprint) is the substrate
this builds on — v3 is additive.

## v3 structure

```
.holo (FORMAT_VERSION 3)
├─ app manifest            AppManifest realization (IRI-tagged, κ-addressed)
│   ├─ layer[0]: { kind: wasm-codemodule, κ, entry: "_start",   exit: required }
│   ├─ layer[1]: { kind: tensor-plan,     κ, entry: session,    exit: n/a      }
│   ├─ layer[2]: { kind: rootfs-image,    κ (κ-disk sector set), entry: boot   }
│   ├─ layer[3]: { kind: view, surface: portable, κ }            # D10
│   ├─ layer[4]: { kind: view, surface: native(ios), κ }         # optional override
│   └─ child[0]: { app: κ, caps: CapabilitySet κ }               # D9 composition
├─ payload sections        existing v2 sections: KernelCalls, schedule, registries,
│                          weights/blobs (BLAKE3-deduped, SHARED across layers)
├─ certificates            per-node certs (existing) + per-layer certs (new)
└─ footer                  BLAKE3 fingerprint (existing)
```

- **AppManifest is a realization** (SPINE-2/3): IRI-tagged canonical bytes embedding the
  operand κ-labels of every layer and child; `references()` gives the inverse projection,
  so `resolve_closure(app κ)` fetches a whole application transitively — migration of an
  app between peers is the same operation as migrating any content.
- **Layer kinds** are a closed enum (exhaustive matching, no catch-all): wasm-codemodule,
  tensor-plan, rootfs-image, view, (extensible only by format version bump). Each layer
  carries an entrypoint and, where meaningful, exit-code semantics — an application, like
  any workload, is "a binary with an exit code."
- **Weights/blob dedup spans layers**: a model shared by two layers is stored once; two
  apps sharing content share κs, so a store holding both holds one copy (Law L3).
- **Degenerate case**: a v2-style compiled tensor graph is a v3 archive whose manifest has
  exactly one tensor-plan layer. The compiler emits this by default; nothing about the
  compile-only workflow gets heavier.

## Execution semantics

- **Entrypoint per layer**, like `main`: the runtime (`hologram-runtime`) boots layers
  per the manifest — wasm codemodules via the engine seam, tensor plans via an
  `InferenceSession`, rootfs images via the emulator + κ-disk in `spaces/holospaces`.
- **Exit codes**: the application's exit code is defined by the manifest's designated
  primary layer; child exit codes propagate parent←child (D9). ErrorEvent realizations
  capture failures append-only.

## Composition (D9): capability-attenuated nesting

- **Underneath**: a parent app's manifest lists `child[i] = { app κ, delegated
  CapabilitySet κ }`. The delegated set MUST be a subset of the parent's effective set —
  attenuation only, amplification unrepresentable. Enforced at `spawn_child` in the
  runtime (the seam already exists in the wasmtime engine's host import surface).
- **Alongside**: sibling apps are separate Sessions under one Peer. They share exactly
  one thing: the `KappaStore` (and hence content dedup). Channels between siblings use
  the existing `Channel` realization + subscribe/publish intents — no ambient authority.
- Composition therefore forms a **DAG of apps** (children may be shared by κ), with the
  capability lattice guaranteeing a child can never exceed its parent.

## Views (D10): portable surface + native override

- An app ships **one portable view layer** targeting the space contract's surface
  capability (02 §5) — it runs on every space unmodified.
- It MAY ship additional `view` layers tagged for specific spaces (`native(ios)`,
  `native(android)`, …). At boot, the space picks its native view if present, else the
  portable view. A chat app can thus ship one portable UI on day one and add an iPad
  view later without touching other layers — layers are κ-addressed, so this is an
  append, not a rewrite.
- Design-system SDKs (Material-like, SwiftUI-like) are **out of scope**: they are future
  projects that compile *down to* view layers; the format only fixes the slots.

## Relationship to OCI

OCI materialization (image pull → κ-disk sectors, in `spaces/holospaces` boot machinery,
e.g. today's DevcontainerProvision) is an **ingest boundary**, not a format concern: an
OCI image becomes a rootfs-image layer inside a `.holo` app. We do not adopt OCI's
manifest structure — κ-realizations already provide stronger identity/verification — but
the ingest path remains first-class so any container image is one provision step away
from being a hologram application.

## Compatibility & migration

- FORMAT_VERSION 2 archives remain loadable (read-side compatibility) through at least
  the first published release cycle; the loader wraps them as single-layer apps in
  memory. Writers emit v3 only.
- Format work lands in migration phase P4 (`06-migration.md`), after the crate moves are
  stable, so codec changes never interleave with tree moves.
- First-party applications (hologram-apps repo) and hologram-ai adopt v3 from their own
  repos once P4 ships in a published release (D1).
