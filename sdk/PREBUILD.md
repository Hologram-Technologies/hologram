# SDK Prebuild and Smoke Matrix

The SDK packages are layered so package tests can prove the install surface
without making the browser-safe package import native code:

| Package | Artifact | Platforms | Smoke check |
|---|---|---|---|
| `hologram` | Python wheel bundling `hologram-ffi` | macOS arm64/x64, Linux x64/aarch64, Windows x64 | Install wheel, import `hologram`, compile/load/execute a small graph through `_hologram` |
| `@tryhologram/sdk` | Pure ESM npm package | Any Node/browser target with ESM | Install packed tarball, import metadata, build a small graph |
| `@tryhologram/native` | N-API addon package | macOS arm64/x64, Linux glibc x64, Linux musl x64, Windows x64 | Install packed tarball with `@tryhologram/sdk`, compile/load/execute a small graph |
| `@tryhologram/wasm` | WASM adapter plus driver crate | Browser and unsupported native platforms | Typecheck adapter, compile driver for `wasm32-unknown-unknown`, install packed tarball with compile/load/execute driver-shaped smoke |

`.github/workflows/sdk-packages.yml` is the source of truth for the current
automation. It builds local platform native artifacts, packs them, installs the
packed tarballs into temporary projects, and runs smoke checks from outside the
source tree.

## Release Policy

Developer experience should default to bundled native artifacts. The implemented
shape bundles every supported per-platform addon into `@tryhologram/native`:

- `scripts/copy-native.mjs` names each build's addon
  `dist/hologram-<platform>-<arch>.node`, so every target's binary ships
  side-by-side in the one package, and
- the loader (`src/index.ts` `targetTag`) resolves `<platform>-<arch>` at runtime
  (distinguishing linux musl from glibc) and loads the matching
  `./hologram-<tag>.node`, falling back to the legacy single-platform
  `hologram.node`.

The rejected alternative — publishing small optional per-platform packages and
letting `@tryhologram/native` load the installed one — remains possible later; it
would re-add an external-package candidate to the loader.

Release artifacts must include checksums before external publication. Signing
policy remains open until package registries are selected.
