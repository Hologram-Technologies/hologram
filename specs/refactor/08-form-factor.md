# 08 — Form Factor: The Front Door (and the On-Ramp)

Decision: D25 (see `00-overview.md`). This doc names the *access story* — the single
easiest way a stranger reaches and uses the hologram runtime — **and** the *on-ramp story*
— the single easiest way a producer's existing work becomes a hologram app. It adds no new
runtime mechanism; it chooses which existing mechanisms are the heroes and orders the rest
as one funnel, in both directions.

> **Change from prior draft.** The earlier version specced only the *access* funnel
> (consumption-first: a stranger opens a URL). It presupposed the app was already a `.holo`
> and gave the producer one verb (`app publish`). That silently narrowed the vision:
> every app should be *authored where users already work*, and existing work (Docker
> images, repos) should be *importable*, not left stranded. This version adds the missing
> left edge — the **BRING** on-ramp — and reframes **BUILD** from "embed our SDK" to
> "meet the build tool."

## The two-sided funnel

The runtime has a producer side and an accessor side. They share one identity (κ) and meet
at the `.holo`.

```
PRODUCER (on-ramp)                              ACCESSOR (funnel)
BRING → wrap what you already have → .holo      TRY   → a URL       open a link, boots in-browser   ← HERO
BUILD → author inside your existing tool        RUN   → a binary    curl | sh → `hologram run <κ>`
        → .holo + κ                             BUILD → a one-line embed (Client + SDKs)
SHIP  → `hologram app publish` → a κ            SHIP  → an app; carries its runtime; hologram invisible
```

Two heroes, one for each side:

- **Accessor hero: a URL.** Open a link → runtime boots in-browser, nothing installed, no
  account. Widest funnel; truest expression of "browser-native" (`00-overview.md`).
- **Producer hero: your existing workflow.** A stranger should reach a published,
  serverless, content-addressed app *without leaving the tool they already build in* —
  their editor, their AI coding agent, their Docker image, their repo. No new IDE, no
  rewrite.

Each rung is a strictly larger commitment and a strictly smaller audience. Optimize the
top of both funnels; let each rung hand down without a cliff.

## Producer side — BRING (the on-ramp)

The core promise: **work stranded on a laptop or an expensive server becomes a κ that
boots anywhere.** This is the backward-compatibility contract, and it is mostly codec
work, not new runtime — `.holo` v3 already carries a `rootfs-image` layer (`03-holo-format.md`).

| Source | Verb | Becomes | Backed by |
|--------|------|---------|-----------|
| Docker / OCI image | `hologram import docker://…` | `.holo` (rootfs-image layer) | 03 `rootfs-image`, holospaces Linux env |
| Local dir / GitHub repo | `hologram import ./repo` / `gh:…` | `.holo` (wasm or rootfs layer) | 03, 05 `compile` |
| Compiled tensor plan | `hologram compile` | `.holo` (tensor-plan layer) | 03, 05 |
| Existing `.holo` | (already there) | κ | 03 |

```sh
# an existing Docker image → a serverless, content-addressed app
hologram import docker://myorg/api:latest -o api.holo      # → κ=b3:1a4e…
# an existing repo → the same
hologram import ./my-app -o app.holo                       # → κ=b3:9f2c…
```

Once it is a `.holo`, it is a κ, and the rest of the funnel is uniform. **Import is the
one feature that turns "runs only on my machine / my $200/mo server" into "runs on any
accessor's space, addressed by a hash."**

**Open sub-item:** import fidelity is a spectrum — a static site is lossless day one; a
Docker image needs the emulation space (bare/native) and is bounded by what the engine
supports (`02-space-contract.md` ContainerEngine). Scope the supported source matrix in
P4 alongside `.holo` v3; ship the lossless cases first.

## Producer side — BUILD (meet the tool, not just the package manager)

The prior draft's BUILD rung was `cargo add / npm i / pip install hologram`. That meets
*embedders* in their package manager — necessary, but it is not meeting *app-builders*
where they actually work today. The 2026 reality: people build apps in AI generators
(Lovable, Bolt, v0), agent IDEs (Cursor, Windsurf), and agent CLIs (Claude Code, Codex).
Hologram integrates into those workflows rather than replacing them.

| Where users build | Integration surface | Output verb |
|-------------------|---------------------|-------------|
| Agent CLIs / IDEs (Claude Code, Cursor, Codex) | **`hologram` MCP server** exposing `compile` / `import` / `run` / `publish` over `Client` | agent ends its run with `publish` → a κ-link |
| App generators (Lovable, Bolt, v0) | **npm package with bundled browser space**; drop-in for the sandboxed run + a "Publish" that replaces "Deploy to Vercel/Netlify" | generated app → κ |
| Embedders (any language) | `hologram = { features = [...] }` / npm / wheel / SwiftPM (`Client` + `hologram-ffi`) | `client.open(κ)` |

The wedge: **do not build a fourth front door.** Every AI build tool already terminates in
the same painful step — provision Supabase + Vercel, then get a URL. Hologram replaces that
step with `publish` → a κ. Build stays where the user already is; only *run* and *share*
change, and they change by getting simpler. The MCP server rides directly on the
single-`Client` surface (`05-tooling.md`), so it is one more `Client` consumer, not new
mechanism.

## Accessor side — the funnel (unchanged hero, tightened)

| Rung | Face | Backed by | Spec |
|------|------|-----------|------|
| TRY (URL) | hosted entry page + browser space | `holospaces-browser` (OPFS, WebRTC, wasm-bindgen); browser SDK packaging crate | 01, 02, 05 |
| RUN (binary) | the one `hologram` binary | `hologram-cli` over `Client`, default native space | 05 |
| BUILD (embed) | `hologram = { features = [...] }` / npm / wheel / SwiftPM | `Client` facade + `hologram-ffi` | 05 |
| SHIP (app) | a `.holo` app the user installs | `.holo` v3 + per-platform packaging | 03, 05 |

## Deploying (producer side — content, not compute)

**Hologram deploys content, not compute.** There is no app server a producer pushes code
to. The app runs on the *accessor's* space (their browser, laptop, device). Deploy collapses
to two primitives plus one entry point:

1. **`put`** the `.holo` into a store → its κ (identity from content).
2. **`announce`** that κ to a network so others can `resolve` it (public, or a restricted
   `network` for private/paid audiences; `04-networks.md`). The network *is* the delivery
   — no upload-to-origin.
3. **(URL rung only)** publish a static entry page that boots the browser space at the κ —
   static-hostable anywhere (object store / CDN / Pages), or served by the HTTP-CAS
   gateway (`hologram-net-http`) for the first visitor.

```sh
# one convenience verb: put + announce (+ optional entry page)
hologram app publish app.holo --page ./site
#   → stored κ=b3:9f2c…   announced on: public   entry page: ./site/

# …or the primitives, for control:
hologram store put app.holo                        # → κ=b3:9f2c…
hologram network create team --restricted          # → network κ (private/paid audience)
hologram net announce b3:9f2c… --network team       # only members can resolve
```

`app publish` is the only verb most producers touch; it decomposes to `store put` +
`net announce` (+ `network`) for anyone who needs the seams. One command deploys to every
rung at once because the κ is the same everywhere (law 2). **No Supabase, no Vercel** — the
store is the backend, the network is the CDN, the κ is the deploy.

**Open sub-item (durability, from 04):** deploy answers *reachability*, not *liveness*.
"Who keeps content alive after the producer goes offline" is the replication policy flagged
in 04 (restricted network with a holding member, pinning gateway, or public replica),
designed post-P5. Until then, deploy is honestly "reachable while a holder is online."

## Monetization (deferred, but designed-for)

Not built here, but the primitives must not foreclose it, so it is named:

- A **κ gated by a capability on a restricted network** is the natural paid-access unit —
  selling access is granting a capability, not running a billing server
  (`04-networks.md` restricted tier, `07-governance-requirements.md` capability policy).
- Attestation (R3) and the audit log (R2) give paid access provable provenance and usage
  without a central intermediary.

Scope: a payment/entitlement design is post-P5. This section exists only to keep the
capability model shaped so that door stays open.

## What the heroes demand (requirements on the rest of the suite)

1. **The browser space is a first-class, permanently-green target, not a demo.** Its TCK
   pass (02) and Pages deployment (06 P2) are release gates — it is the accessor front door.
2. **Cold-load budget is a real number.** The browser SDK wasm size budget (05) is a
   release gate; bloat is a funnel leak. Set at first real build, defended in CI.
3. **Zero-account, zero-server to first workload.** The TRY rung reaches a running workload
   with no sign-up and no origin beyond static hosting. Networks/persistence/payment are
   opt-in *after* the hook, never gates before it.
4. **The link is content-addressed.** A shared URL resolves to a specific app/version by κ
   — "try this exact thing" is reproducible.
5. **Progressive disclosure, no cliff.** What a user built at BRING/BUILD (a `.holo`, its κ)
   carries unchanged through TRY/RUN/SHIP — same bytes, same κ, different host.
6. **Import is honest about fidelity.** `import` states what layer a source maps to and
   which spaces can run it; lossless cases ship first, emulation-bound cases are labeled.

## DX success tests (both funnels, measured)

- **BRING**: an existing Docker image or repo becomes a bootable `.holo` + κ in one command,
  with no code change to the source.
- **BUILD (meet-the-tool)**: from inside Claude Code / Cursor (via MCP) or a generator, a
  user's app reaches a published κ-link without touching Supabase or Vercel.
- **TRY**: a stranger with only a browser goes from *clicking a link* to *a running
  workload* in one page load, zero installs, no account.
- **RUN**: `curl … | sh` then one command runs a `.holo` — under a minute, one binary.
- **BUILD (embed)**: one dependency line + the `05` sample compiles and runs in
  rust/python/typescript/swift — no subcrate archaeology.
- **SHIP**: a first-party app installs and launches on browser + one native platform from
  the *same* `.holo` bytes (same κ) — the P4 multi-space demo.
- **No cliff**: the κ produced at BRING/TRY resolves and runs identically at RUN and SHIP.

## Sequencing (which rungs land when)

- **P3** (Client + first publish): TRY, RUN, BUILD(embed), and the `app publish` verb.
- **P3.5 / P4**: the **MCP server** (meet-the-tool BUILD) and **`hologram import`**
  (BRING) — both are `Client`/codec consumers, not new runtime, so they follow the facade.
- **P4**: `.holo` v3 multi-layer → SHIP multi-space demo; import fidelity matrix.
- **P5**: restricted networks → private/paid audiences.
- **Post-P5**: durability/replication and the monetization/entitlement design.

## Non-goals

- Does not choose a hosting provider, landing-page design, or go-to-market surface — it
  fixes the *technical form factor* and its DX contract.
- Does not add runtime capability; if a rung needs something the crate map lacks, that is a
  gap for the owning doc (01–05), surfaced here but fixed there.
- Does not build the monetization system — only keeps the capability model shaped for it.
