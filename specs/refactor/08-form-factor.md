# 08 — Form Factor: The Front Door

Decision: D25 (see `00-overview.md`). This doc names the *access story* — the single
easiest way a stranger reaches and uses the hologram runtime — and makes the tooling of
`05-tooling.md` serve it. It adds no new mechanism; it chooses which existing mechanism
is the hero and orders the rest as one funnel.

## The funnel (and the hero)

**Hero: a URL.** Open a link → the hologram runtime boots in the browser, nothing
installed, no account required. This is the widest possible funnel and the truest
expression of the "browser-native, initially browser-focused" vision (`00-overview.md`).
The deeper rungs exist for people who have already been hooked by the link.

```
TRY   →  a URL            open a link, runtime boots in-browser, zero install   ← HERO
RUN   →  a binary         curl | sh  →  `hologram run app.holo`
BUILD →  a one-line embed cargo add / npm i / pip install hologram  (Client + SDKs)
SHIP  →  an app           install a .holo app; it carries its runtime; hologram invisible
```

Each rung is a strictly larger commitment and a strictly smaller audience. Optimize the
top; let each rung hand down to the next without a cliff.

## Rung → mechanism (all already specced)

| Rung | Face | Backed by | Spec |
|------|------|-----------|------|
| TRY (URL) | hosted entry page + browser space | `holospaces-browser` (OPFS, WebRTC, wasm-bindgen); the browser SDK packaging crate | 01, 02, 05 |
| RUN (binary) | the one `hologram` binary | `hologram-cli` over `Client`, default native space | 05 |
| BUILD (embed) | `hologram = { features = [...] }` / npm / wheel / SwiftPM | `Client` facade + `hologram-ffi` packaging crates | 05 |
| SHIP (app) | a `.holo` app the user installs | `.holo` v3 + per-platform packaging | 03, 05 |

Nothing new is built for this doc — it is a **lens** on the crate map, not an addition to
it. Its value is forcing every DX decision to answer "does this make the URL rung
easier?" first.

## Deploying (the producer side — the mirror of the funnel)

The funnel above is the *accessor's* path in. Its mirror is: how does a producer make
their app reachable at all? The answer is a deliberate inversion:

**Hologram deploys content, not compute.** There is no app server a producer pushes code
to that then executes it — the app runs on the *accessor's* space (their browser, laptop,
device). Deploying therefore collapses to two primitives that already exist, plus one
entry point for the URL rung:

1. **`put`** the `.holo` into a store → its κ (identity from content; `05-tooling.md`
   `store put`).
2. **`announce`** that κ to a network so others can `resolve` it (`net announce`, or a
   restricted `network` for private audiences; `04-networks.md`). The network *is* the
   delivery — no upload-to-origin.
3. **(URL rung only)** publish a static entry page that boots the browser space pointed
   at the κ. The page is just the browser-SDK wasm + a κ pointer — static-hostable
   anywhere (object store / CDN / Pages), or served by an HTTP-CAS gateway
   (`hologram-net-http` already exists) which can hand back both the page and the
   initial content over plain HTTP for the very first visitor.

Per rung, "deploy" is the same `put` + `announce`, differing only in the entry point:

| Rung | Deploy = | Entry the accessor uses |
|------|----------|-------------------------|
| TRY | put + announce + static boot page | a URL |
| RUN | put + announce (or just share the file) | `hologram run <κ>` resolves it |
| BUILD | publish the *platform* to registries (crates.io/npm/pip); user apps ship as κ | `import hologram` then resolve the κ |
| SHIP | put + announce + per-platform app packaging | install-from-κ / app channel |

A single convenience verb — `hologram app publish <app.holo>` — does put + announce and,
with `--page`, emits the static entry bundle. One command deploys to every rung at once
because the κ is the same everywhere (law 2).

### Deploy mechanism (concrete commands)

```sh
# 1. build an app from source → app.holo (prints the app's κ)
hologram compile ./my-app -o app.holo
#   → app.holo   κ=b3:9f2c…

# 2. deploy — one convenience verb (put + announce + optional entry page)
hologram app publish app.holo --page ./site
#   → stored κ=b3:9f2c…   announced on: public   entry page: ./site/

#    …or the primitives explicitly, for control:
hologram store put app.holo                    # → κ=b3:9f2c…
hologram net announce b3:9f2c…                 # public network
#    restricted audience instead of public:
hologram network create team --restricted      # → network κ
hologram net announce b3:9f2c… --network team  # only members can resolve

# 3. serve the URL rung (browser boot). Either:
hologram node serve --gateway --page ./site    # HTTP-CAS gateway: page + first-fetch
#    …or copy ./site to any static host / CDN (page is just wasm + a κ pointer);
#    content is then resolved peer-to-peer or from a gateway.
```

The accessor side needs no producer coordination — the κ is the contract:

```sh
# TRY   open https://host/                # boots the browser space at κ=b3:9f2c…
# RUN   hologram run b3:9f2c…             # resolves from the network, then boots
# BUILD  (in code)  client.open("b3:9f2c…").await?      # same κ, embedder's space
```

`app publish` is the only verb most producers ever touch; `store put` + `net announce`
(+ `network`) are what it decomposes to, exposed for anyone who needs the seams.

**Open sub-item (connects to `04-networks.md` durability):** deploy answers *reachability*,
not *liveness*. "Who keeps the content alive after the producer's peer goes offline" is
the durability/replication policy already flagged in 04 — a restricted network with a
holding member, a pinning gateway, or a public-network replica are the candidate answers,
designed post-P5. Until then, deploy is honestly "reachable while a holder is online."

## What the hero rung demands (requirements on the rest of the suite)

The URL-first choice is not free; it constrains earlier docs, and those constraints are
recorded here so they are not discovered late:

1. **The browser space is a first-class, permanently-green target, not a demo.** Its TCK
   pass (02) and Pages deployment (06 P2) are release gates, because the browser space
   *is* the front door — if it breaks, the product's face is down.
2. **Cold-load budget is a real number.** The browser SDK wasm size budget (05) exists
   precisely because the hero rung is "boots from a link" — bloat is a funnel leak, not a
   nicety. The budget is set when the first real build exists and defended in CI.
3. **Zero-account, zero-server to first workload.** Operator identity is self-sovereign
   (00); the TRY rung must reach a running workload with no sign-up and no origin server
   beyond static hosting. Networks/persistence are opt-in *after* the hook, never gates
   before it.
4. **The link is content-addressed too.** A shared URL should resolve to a specific
   app/version by κ where possible — "try this exact thing" is reproducible, not
   "whatever main is today." (Fits κ-only identity, law 2.)
5. **Progressive disclosure, no cliff.** The thing a user built at the TRY rung (a
   workload, its κ) must carry unchanged to RUN/BUILD/SHIP — the same `.holo`, same κ,
   different host. This is already guaranteed by content-addressing; the form-factor doc
   makes it a *product* promise, not just an architectural fact.

## DX success tests (the funnel, measured)

The form factor is real only if these pass — they are the acceptance criteria for the
access story, checked as the browser/CLI/SDK work matures (P2–P4):

- **TRY**: a stranger with only a browser goes from *clicking a link* to *a running
  workload* in one page load and zero installs. No account, no CLI, no config.
- **RUN**: `curl … | sh` then one command runs a `.holo` — under a minute, one binary,
  no runtime prerequisites.
- **BUILD**: one dependency line + the sample from `05-tooling.md` §Principle compiles
  and runs in each of rust/python/typescript/swift — no subcrate archaeology.
- **SHIP**: a first-party app installs and launches on browser + one native platform from
  the *same* `.holo` bytes (same κ) — the P4 multi-space demo, seen as a product, is this
  test.
- **No cliff**: the κ produced at TRY resolves and runs identically at RUN and SHIP.

## Non-goals

- This doc does not choose a hosting provider, a landing-page design, or a marketing
  surface — it fixes the *technical form factor* and its DX contract, not the go-to-market.
- It does not add runtime capability; if a rung needs something the crate map lacks, that
  is a gap for the owning doc (01–05), surfaced here but fixed there.
