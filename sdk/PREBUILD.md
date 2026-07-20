# SDK Prebuild and Smoke Matrix

The SDK packages are layered so package tests can prove the install surface
without making the browser-safe package import native code:

| Package | Artifact | Platforms | Smoke check |
|---|---|---|---|
| `hologram` | Python wheel bundling `hologram-ffi` | macOS arm64/x64, Linux x64/aarch64, Windows x64 | Install wheel, import `hologram`, compile/load/execute a small graph through `_hologram` |
| `@uor-foundation/sdk` | Pure ESM npm package | Any Node/browser target with ESM | Install packed tarball, import metadata, build a small graph |
| `@uor-foundation/native` | N-API addon package | macOS arm64/x64, Linux glibc x64, Linux musl x64, Windows x64 | Install packed tarball with `@uor-foundation/sdk`, compile/load/execute a small graph |
| `@uor-foundation/wasm` | WASM adapter plus driver crate | Browser and unsupported native platforms | Typecheck adapter, compile driver for `wasm32-unknown-unknown`, install packed tarball with compile/load/execute driver-shaped smoke |

`.github/workflows/sdk-packages.yml` is the source of truth for the current
automation. It builds local platform native artifacts, packs them, installs the
packed tarballs into temporary projects, and runs smoke checks from outside the
source tree.

## Release Policy

Developer experience should default to bundled native artifacts. Before npm
publication, choose one distribution shape:

- bundle all supported `hologram.node` binaries into `@uor-foundation/native` and
  select at load time, or
- publish small optional platform packages and let `@uor-foundation/native` load the
  installed platform package.

The current adapter already probes the colocated `dist/hologram.node` first and
then a future `@uor-foundation/native-bin` package. That keeps the loader compatible
with either release shape.

Release artifacts must include checksums before external publication. Signing
policy remains open until package registries are selected.
