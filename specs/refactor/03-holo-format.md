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
│   ├─ primary: 0          index of the layer whose exit code IS the app's exit code
│   ├─ requires: κ         CapabilitySet the app needs — checked at provision (fail fast)
│   ├─ layer[0]: { kind: wasm-codemodule, κ, entry: "_start",   exit: required }
│   ├─ layer[1]: { kind: tensor-plan,     κ, entry: session,    exit: n/a      }
│   ├─ layer[2]: { kind: rootfs-image,    κ, arch: riscv64,      entry: boot   }
│   ├─ layer[3]: { kind: view, surface: portable, κ }            # D10
│   ├─ layer[4]: { kind: view, surface: native(ios), κ }         # optional override
│   └─ child[0]: { app: κ, caps: CapabilitySet κ }               # D9 composition
├─ payload sections        existing v2 sections: KernelCalls, schedule, registries,
│                          weights/blobs (BLAKE3-deduped, SHARED across layers)
│                          (present in FAT archives; absent in THIN — see §Fat and thin)
├─ certificates            per-node certs (existing) + per-layer certs (new)
└─ footer                  BLAKE3 fingerprint (existing)
```

Manifest fields fixed here (they were prose-only before):

- **`primary`** — the layer whose exit code is the application's exit code. Exactly one.
- **`requires`** — the CapabilitySet κ the app needs to run. Provision checks
  `granted ⊇ requires` and fails fast; runtime enforcement is unchanged (the grant, not
  the request, is what's enforced — `requires` is a declaration, never an entitlement).
- **`arch`** — mandatory on `rootfs-image` layers (ISA fixed at provision, holospaces
  ADR-021); absent on portable kinds (wasm-codemodule, tensor-plan, view). A manifest may
  carry sibling rootfs layers for different arches; the space selects at boot exactly
  like native views.

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
- **Boot order = manifest order.** Layers boot sequentially in index order; a layer's
  boot completes (its readiness, not its exit) before the next begins. No dependency
  graph between layers in v3 — apps needing richer orchestration express it as child
  apps (the composition DAG already exists; don't build a second one inside the
  manifest). View layers are exempt: the space attaches the selected view when its
  surface is ready.
- **Exit codes**: the application's exit code is the `primary` layer's exit code; child
  exit codes propagate parent←child (D9). ErrorEvent realizations capture failures
  append-only. When the primary layer exits, the runtime terminates remaining layers
  (reverse boot order).
- **Suspend/resume is per-app and composite.** An app snapshot is itself a realization:
  `AppSnapshot = { manifest κ, per-layer snapshot κs }`, produced by suspending layers in
  reverse boot order and resumed in boot order. `resolve_closure(AppSnapshot κ)` therefore
  migrates a *running* multi-layer app between peers exactly like any other content —
  the single-container Session semantics are the one-layer degenerate case.

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

## Fat and thin archives

Layers are κ-references, so the file form has two profiles — **the app's identity is the
manifest κ in both**; fat vs thin is a packaging choice, never an identity change:

- **Fat** (self-contained): manifest + all referenced payloads embedded in the payload
  sections. One file to copy anywhere; ideal for distribution, air-gap, first install.
- **Thin** (manifest-only): payload sections absent; layers resolve through the
  KappaStore/KappaSync path (`resolve_closure`). Ideal once content is in a store or a
  network — a GB-scale rootfs never travels twice (Law L3 dedup does the work).

The loader treats them identically: "load" = resolve the manifest's closure, from the
file's own sections first, then the store, then sync. `hologram app` tooling converts
between profiles (`--fat`/`--thin`) without touching κs.

## Explicitly deferred (recorded so they aren't forgotten)

- **argv/env/stdio conventions** for workload entrypoints — P4 detail spec'd with the
  loader implementation; goes in the layer entry descriptor, not new sections.
- **Human naming & trust** ("install `chat-app` from whom?") — names/registries are a
  roster/manager concern (κ is the identity; names are content pointing at κs), and
  publisher trust is the attestation track (`07-governance-requirements.md` R3). The
  format stays name-free.

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
