# Plan 076: SDK / FFI Builder and Language Packages

**Status:** Active
**Created:** 2026-06-01
**Depends on:** [Plan 075](075-source-ir-language-frontends.md),
[External Tensor References](../docs/external-tensor-references.md)
**Primary files:** `crates/hologram-ffi/`, future `sdk/python/`, future
`sdk/node/`, `crates/hologram-compiler/src/source/`

## Problem

Plan 075 gives Hologram source parsers a common source IR, but parser frontends
are not enough for application SDKs. Python, TypeScript, C, and other users
also need a direct builder API for notebooks, services, generated graphs,
external weights, and package integrations where parsing host source files is
the wrong abstraction.

The current FFI surface is sufficient for compile/load/execute basics, but it
is not yet a publishable SDK foundation:

- `hologram_compile_source` compiles text source, but does not expose a stable
  graph/source builder.
- The checked-in C header advertises graph-builder symbols that are not the
  implemented/tested Rust FFI surface and must be reconciled before publishing.
- Language packages need ownership, errors, versioning, packaging, and
  generated op metadata policies before they are shipped.

The root design point is that SDKs and parsers should converge before graph
lowering:

```text
Python / TypeScript / Rust parser -> SourceProgram -> Graph -> Compiler
Python / TypeScript / Rust SDK    -> SourceProgram -> Graph -> Compiler
C / FFI builder API               -> SourceProgram -> Graph -> Compiler
```

## Goals

- Add a stable, versioned FFI builder API that can construct a `SourceProgram`
  or equivalent graph contract directly without source parsing.
- Publish ergonomic Python and TypeScript SDKs on top of generated low-level
  bindings.
- Generate SDK op surfaces from the canonical op catalog so new ops do not
  require hand-editing every language package.
- Support chainable human SDKs above generated bindings.
- Make external tensor references a first-class SDK path for large weights.
- Preserve runtime source agnosticism: SDK handles, parser spans, source names,
  and host-language metadata must not survive into execution dispatch.
- Keep source parsing as an optional convenience for extracting explicit graph
  regions from existing host-language files.

## Non-Goals

- Executing arbitrary Python, TypeScript, or Rust code to build graphs.
- Making SDK objects part of the `.holo` runtime archive format.
- Introducing dynamic op dispatch or callback registries into
  `hologram-compute` or `hologram-exec`.
- Requiring Node native addons in browsers; browser support must use a WASM
  package with documented limitations.

## Architecture

The SDK stack has three layers:

```text
Rust compiler/runtime crates
  -> stable C/WASM ABI in hologram-ffi
  -> generated low-level bindings
  -> human Python / TypeScript SDKs
```

The generated layer should be thin and mechanical:

- C ABI declarations.
- Python native extension bindings.
- TypeScript native/WASM bindings and `.d.ts`.
- Generated op method stubs from canonical op metadata.
- Generated dtype constants and attribute type definitions.

The human SDK layer should be ergonomic and stable:

- Chainable `Graph` and `Tensor` APIs.
- Friendly exceptions and diagnostics.
- External tensor/file handling.
- Async Node APIs and sync Python APIs where appropriate.
- Documentation examples and package-level compatibility checks.

## Target SDK Shape

Python:

```python
import hologram as hg

g = hg.Graph("encoder")

x = g.input("x", dtype=hg.f32, shape=[2, 3])
w = g.const_ref(
    "w",
    dtype=hg.f32,
    shape=[3, 2],
    file="weights.bin",
    blake3="...",
)

y = x.matmul(w, shape=[2, 2]).relu()
archive = g.output("y", y).compile()
```

TypeScript:

```ts
import * as hg from "@hologram/sdk";

const g = new hg.Graph("encoder");

const x = g.input("x", { dtype: hg.f32, shape: [2, 3] });
const w = g.constRef("w", {
  dtype: hg.f32,
  shape: [3, 2],
  file: "weights.bin",
  blake3: "...",
});

const y = x.matmul(w, { shape: [2, 2] }).relu();
const archive = await g.output("y", y).compile();
```

Both SDKs should also expose a lower-level escape hatch:

```python
archive = (
    hg.Graph("encoder")
    .input("x", dtype=hg.f32, shape=[2, 3])
    .const("w", shape=[3, 2], values=[1, 2, 3, 4, 5, 6])
    .op("matmul", ["x", "w"], shape=[2, 2], as_="y")
    .output("y")
    .compile()
)
```

Source parsing remains separate:

```python
archive = hg.compile_source_file("graph.txt")
```

```ts
const archive = await compileSourceFile("graph.txt", nativeBinding);
```

## Package Model

| Layer | Python | TypeScript / JavaScript |
|---|---|---|
| Generated native binding | `_hologram` wheel extension | `@hologram/native` N-API package |
| WASM binding | optional `_hologram_wasm` | `@hologram/wasm` |
| Human SDK | `hologram` on PyPI | `@hologram/sdk` on npm |
| Browser package | not primary | `@hologram/browser` |
| C ABI | `hologram-ffi` header/library | consumed by native package generation |

Python packaging should use `maturin` or `setuptools-rust`. Node packaging
should use `napi-rs` for native builds and a WASM fallback for browser and
unsupported native platforms.

## Required Design Areas

### ABI Stability

- [x] Define `hologram_abi_version`.
- [x] Define `hologram_feature_supported`.
- [ ] Use explicit opaque handle types for builders, graphs, archives,
  sessions, errors, and external tensors.
- [x] Keep ABI additions additive; never repurpose existing symbols.
- [x] Add exported-symbol snapshot tests so published symbols do not disappear.

### Error Model

- [x] Define stable source-builder error categories and reserve SDK-facing
  categories for parse error, graph error, unsupported op, bad attr, shape
  mismatch, external tensor error, archive load error, execution error, and ABI
  mismatch.
- [x] Add `hologram_last_error_code`.
- [x] Add `hologram_last_error_message`.
- [x] Add structured source/graph location fields where available.
- [x] Map errors to Python exceptions and TypeScript error classes.

### Memory Ownership

- [x] Define who owns every returned pointer and buffer.
- [ ] Add explicit free functions for archive bytes, error strings if owned,
  builder handles, tensor handles, session handles, and output buffers.
- [x] Avoid returning borrowed pointers across calls unless lifetime is
  explicitly documented and tested.
- [x] Add Python finalizers for source builder and session handles.
- [x] Add Node finalizers for all native handles.

### Generated Bindings Contract

- [x] Extend the canonical op metadata beyond `OpKind::ALL` names to include
  arity, attrs, dtype constraints, shape requirements, and docs.
- [x] Define SDK op metadata generated from the canonical op catalog.
- [x] Generate low-level Python op metadata and helper functions.
- [x] Generate low-level TypeScript op metadata and helper declarations.
- [x] Generate dtype constants and attribute-name metadata.
- [x] Add a check that all canonical ops have generated SDK metadata.
- [x] Generate Python method wrappers for each supported op.
- [x] Generate TypeScript methods and `.d.ts` types for each supported op.
- [x] Generate rich attribute option types from typed op metadata.

### Human SDK Layer

- [x] Implement chainable `Graph` and `Tensor` APIs in Python.
- [x] Implement chainable `Graph` and `Tensor` APIs in TypeScript.
- [x] Keep `Graph.op(kind, inputs, **attrs)` / `Graph.op(kind, inputs, attrs)`
  as an escape hatch.
- [x] Delay Python operator overloads except for obvious future candidates
  (`+`, `-`, `*`, `/`, `@`) after diagnostics are solid.
- [x] Provide Python `Session.load(...).execute(...)` wrapper with named
  input/output byte-buffer helpers.
- [x] Provide TypeScript `Session.load(...).execute(...)` wrapper with typed
  input/output helpers.

### External Tensor Security

- [x] Implement `const_ref` / `constRef` through the shared
  `ExternalTensor` contract.
- [x] Require dtype, shape, byte length, and content hash for production
  external tensors.
- [x] Decide path policy: source-file-relative, explicit root, or current
  working directory.
- [x] Support allowlists or compile roots for service use.
- [x] Validate hashes before archive insertion or backend weight packing.
- [x] Ensure runtime execution never opens source paths.

### Reproducibility

- [x] Same graph expressed through native `.txt`, Python SDK, TypeScript SDK,
  Python parser, TypeScript parser, and Rust parser must produce equivalent
  graph structure.
- [x] Equivalent source/SDK graphs should produce byte-identical archives when
  metadata policy allows.
- [x] Ensure deterministic symbol ordering, attrs, constants, external tensor
  resolution, and generated names.

### Schema and Introspection

- [x] Expose archive input/output names, dtypes, shapes, and metadata through
  SDKs.
- [ ] Expose supported ops, attrs, dtypes, backends, and feature flags.
- [x] Expose archive format version and SDK/native ABI version.

### Version Compatibility

- [x] Define compatibility rules for SDK package version, native library ABI
  version, and `.holo` archive format version.
- [x] Fail loudly on ABI mismatch at import/load time.
- [x] Document supported native package/platform combinations.

### Async and Cancellation

- [ ] Make TypeScript compile/load/execute async by default.
- [ ] Decide whether Python exposes blocking APIs only or optional background
  execution.
- [ ] Define thread-safety guarantees for builders and sessions.
- [ ] Define cancellation behavior for long compile/load/execute operations.

### Packaging Matrix

- [x] Add Python SDK wheel build configuration.
- [x] Add npm `@hologram/sdk` package configuration.
- [x] Add npm `@hologram/native` adapter package configuration.
- [x] Add npm `@hologram/wasm` adapter package configuration.
- [x] Keep the TypeScript SDK browser-safe by avoiding direct Node/native
  imports.
- [x] Add the `@hologram/native` N-API addon crate consumed by the native
  adapter.
- [x] Add the `@hologram/wasm` wasm-bindgen driver crate consumed by the WASM
  adapter.
- [x] Add SDK package CI/prebuild workflow and installed wheel/tarball smoke
  scripts.
- [ ] Python wheels: macOS arm64/x64, Linux glibc x64/aarch64, Linux musl
  where feasible, Windows x64.
- [ ] Node native packages: N-API prebuilds for macOS arm64/x64, Linux glibc,
  Linux musl, Windows x64.
- [ ] WASM package for browser and unsupported Node platforms.
- [x] Decide whether packages bundle native libraries or require a system
  install. Default is bundled for developer experience.
- [ ] Add signing/checksum policy for release artifacts.

### Browser Story

- [x] Document browser-safe SDK boundaries and future `@hologram/wasm` /
  `@hologram/browser` package roles.
- [x] Add browser-safe `@hologram/wasm` adapter package path.
- [x] Publish `@hologram/browser` or document browser usage through
  implemented `@hologram/wasm`.
- [x] Document limitations: filesystem access for `const_ref`, threading,
  SIMD availability, memory caps, and async loading.
- [x] Provide explicit browser examples that do not rely on native paths.

### Documentation Generation

- [ ] Generate API reference stubs from op metadata.
- [x] Layer human workflow docs on top of generated references.
- [ ] Generate Python and TypeScript examples from shared golden graphs where
  practical.
- [x] Document parser frontends and SDK builders as separate entry points.

### Testing Strategy

- [x] FFI ABI smoke tests for every exported builder function.
- [x] Python SDK unit tests for graph building and package-surface imports.
- [x] Python SDK unit tests for native compile, execute wrappers, and SDK-side
  native errors.
- [x] Python SDK unit tests for external tensor reference native compilation.
- [x] TypeScript SDK/native/WASM tests for graph building, compile, execute
  wrappers, and external tensor references.
- [x] TypeScript SDK/native/WASM package type checks, public declaration export
  smoke tests, and dry-run package checks.
- [x] Cross-language golden tests proving Python SDK, TypeScript SDK, native
  source, and parser frontends produce equivalent graph/archive outputs.
- [x] Packaging smoke tests that import installed wheels/npm packages on each
  supported platform.

## Phases

### Phase 1: FFI Builder Contract

- [x] Reconcile `crates/hologram-ffi/include/hologram.h` with the implemented
  Rust FFI surface.
- [x] Add opaque source builder handles.
- [x] Add initial builder functions for input, op, output, compile, and free.
- [x] Add builder functions for inline const and external const ref.
- [x] Add structured error-code APIs.
- [x] Add basic thread-local error message and ABI-version APIs.
- [x] Add C ABI tests covering graph/source builder compile round-trips.

**Acceptance:** A C caller can build a small graph without source text, compile
it to `.holo`, load it, and execute it.

**Current slice (2026-06-01):** `hologram-ffi` now exposes a narrow source
builder ABI (`input`, inline `const`, file-backed `const_ref`, `op`, `output`,
`compile`, `free`) over `SourceProgram`, plus `hologram_abi_version`,
`hologram_last_error`, `hologram_last_error_code`,
`hologram_error_message`, and `hologram_last_error_message`. The checked-in C
header now matches the implemented exports. External refs use
`SourceExternalTensor` and are lowered under `std` into ordinary graph constants
after byte-length and BLAKE3 validation. Source-builder failures now map to
stable SDK-facing categories. SDK package finalizers, language exception
mapping, generated bindings, and package compatibility checks remain before the
SDK bindings should treat the ABI as publishable.

**8.2 slice (2026-06-01):** The FFI now exposes
`hologram_archive_format_version` and `hologram_feature_supported`, documents
ownership/lifetime/version rules in `specs/docs/ffi-abi-contract.md`, and
snapshots required header symbols/constants in
`crates/hologram-ffi/tests/abi_contract.rs`. SDK import code should check the
native ABI version plus required feature strings before calling optional
builder APIs.

### Phase 2: Generated Metadata and Bindings

- [x] Define SDK op metadata generated from the canonical op catalog.
- [x] Generate low-level Python binding metadata and helper functions.
- [x] Generate low-level TypeScript declarations and helpers.
- [x] Add CI checks that generated bindings are current.

**Acceptance:** Adding op metadata updates generated Python/TypeScript binding
surfaces without hand-editing each language.

**8.3 slice (2026-06-01):** `hologram-ffi::sdk` now derives SDK op metadata
from `OpKind::ALL` plus the compiler source attribute metadata, generates
checked-in Python and TypeScript low-level SDK files under `sdk/`, and snapshots
those generated files in `crates/hologram-ffi/tests/sdk_generated.rs`.
The generated layer exposes dtype constants, required FFI feature strings,
canonical op names, accepted attribute names, and a low-level `op_call` /
`opCall` escape hatch.

### Phase 3: Human Python SDK

- [x] Add `hologram` Python package scaffold.
- [x] Add chainable `Graph` and `Tensor` APIs.
- [x] Add `const_ref`.
- [x] Add `compile_source_file` convenience wrapper.
- [x] Add PyPI wheel build configuration.

**Acceptance:** Python users can build, compile, load, and execute a graph from
the chainable SDK without parsing source.

**8.4 Python slice (2026-06-01):** `sdk/python/hologram` now exposes a human
`Graph` / `Tensor` API over generated op metadata. It supports
`input`, inline `const`, `const_ref`, chainable tensor calls such as
`x.matmul(w).relu()`, the `Graph.op(...)` escape hatch, output aliases, native
feature checks, and a low-level builder protocol. The slice is covered by
stdlib Python tests with a fake native binding. Actual wheel/native extension
packaging remains Phase 5.

### Phase 4: Human TypeScript SDK

- [x] Add `@hologram/sdk` source scaffold.
- [x] Add `@hologram/native` and `@hologram/wasm` low-level adapter packages.
- [x] Add chainable `Graph` and `Tensor` APIs.
- [x] Add `constRef`.
- [x] Add `compileSourceFile` convenience wrapper.

**Acceptance:** Node users can build, compile, load, and execute a graph from
the chainable SDK without parsing source.

**8.4 TypeScript slice (2026-06-01):** `sdk/typescript/src` now exports the
generated metadata plus a human `Graph` / `Tensor` API. Tensor values use a
generated-op proxy for calls like `x.matmul(w, { shape: [2, 2] }).relu()`, while
`graph.op(...)` remains the typed escape hatch. The source compiles under
`tsc --strict`; native N-API/WASM binding packages remain Phase 5.

### Phase 5: Packaging and Release

- [x] Add Python SDK wheel build configuration.
- [x] Add npm `@hologram/sdk` package metadata and TypeScript build config.
- [x] Document native/WASM package boundaries and browser-safe SDK constraints.
- [x] Add npm `@hologram/native` and `@hologram/wasm` adapter package
  metadata, TypeScript build configs, and dry-run package checks.
- [ ] Add Python wheel matrix.
- [ ] Add npm native prebuild matrix.
- [x] Add N-API binary and WASM driver implementations.
- [x] Add installed-package import smoke tests across supported platforms.
- [x] Add version compatibility checks.

**Acceptance:** Published package artifacts can be installed and imported on
supported platforms without a local Rust toolchain.

**8.5 packaging slice (2026-06-01):** The pure Python SDK is now packageable as
`hologram` through `sdk/python/pyproject.toml`, includes `py.typed`, and has
stdlib package-surface smoke tests. The browser-safe TypeScript SDK is now
packageable as `@hologram/sdk` through `sdk/typescript/package.json`, emits ESM
and `.d.ts` files through `sdk/typescript/tsconfig.json`, and documents that
native N-API and WASM bindings are separate packages implementing the exported
`NativeBinding` protocol. Platform prebuilds, WASM implementation, artifact
signing, and cross-platform installed-package smoke tests remain open.

**8.5d adapter slice (2026-06-01):** The FFI source-builder contract now has an
additive `hologram_source_builder_output_alias` function plus a
`source-builder.output-alias` feature probe, and source lowering now preserves
semantic input/output port names. `@hologram/native` and `@hologram/wasm`
adapter packages wrap the SDK `NativeBinding` protocol over future N-API/WASM
drivers, validate ABI/features on load, preserve output aliases, convert inline
f32 constants to bytes, derive f32 `constRef` byte lengths from shape when
omitted, and fail loudly for op attributes that the FFI builder cannot carry
yet. Actual N-API binary and WASM driver implementations remain open.

**8.5e/8.5f driver and prebuild slice (2026-06-01):** `@hologram/native` now
has a `napi-rs` addon crate that wraps the stable source-builder ABI, exports
the adapter-consumed functions, builds to `dist/hologram.node`, and has local
plus installed-package smoke coverage that compiles a small graph. `@hologram/wasm`
now has a wasm-bindgen driver crate exposing the same source-builder surface for
browser drivers and a script that checks it with the rustup stable
`wasm32-unknown-unknown` toolchain. `.github/workflows/sdk-packages.yml` defines
the package/prebuild smoke matrix, while `sdk/PREBUILD.md` captures the current
artifact policy and leaves the final npm native-binary distribution shape plus
artifact signing/checksums as release decisions.

### Phase 6: Cross-Language Conformance

- [x] Add golden graph shared across native `.txt`, Python SDK, TypeScript SDK,
  Python parser, TypeScript parser, and Rust parser.
- [x] Assert graph equivalence.
- [x] Assert archive equivalence where metadata policy permits.
- [x] Assert external tensor reference behavior is identical across SDKs.

**Acceptance:** SDK and parser entry points are proven to converge before graph
lowering.

**8.6 conformance slice (2026-06-02):** `source_ir` now includes a golden
external-weight matmul witness proving file-backed `SourceExternalConst`
lowers to the same graph constant and byte-identical archive as an inline
native `.txt` constant. `hologram-ffi` now has the same witness through the
SDK-facing source-builder ABI. The existing parser conformance suite runs the
native, Python, TypeScript, and Rust frontends against equivalent graph/archive
fixtures when their features are enabled, and the Python/TypeScript human SDKs
now have golden builder-contract tests for the external-ref graph.

**8.7 Python native wheel slice (2026-06-02):** The Python package now includes
a real `_hologram.py` ctypes binding over the `hologram-ffi` C ABI. The wheel
build runs `cargo build -p hologram-ffi --release`, bundles the platform
library as `_hologram_ffi.*`, marks the wheel platform-specific, checks ABI /
archive-format / required-feature compatibility before builder use, and has
native Python plus installed-wheel smoke coverage that compiles a graph.

**8.8 Python session wrapper slice (2026-06-02):** `hologram.Session.load`
now wraps the FFI session ABI from Python, exposes input/output counts, names,
shapes, output byte lengths, kernel count, archive fingerprint, context-managed
close/finalization, and `execute(...)` over either named input mappings or
ordered byte-buffer sequences. The Python native and installed-wheel smoke tests
now compile, load, execute, and validate a small graph through the bundled
library.

**8.9 TypeScript session wrapper slice (2026-06-02):** `@hologram/sdk` now
exports `Session.load(...)` over an optional `NativeBinding.sessionLoad`
contract, accepts named input maps / ordered input arrays / single byte buffers,
and returns output-name keyed `Uint8Array`s. `@hologram/native` and
`@hologram/wasm` now expose matching source-builder plus session driver
surfaces, including load, port introspection, archive fingerprint, execute, and
close. SDK golden tests and native/WASM package smokes now cover
compile/load/execute.

**8.10 SDK error taxonomy slice (2026-06-02):** Python now exposes a shared
`hologram.errors` taxonomy and maps native plus SDK-side failures to stable
classes such as `ArchiveLoadError`, `ExternalTensorError`, `ExecutionError`,
`AbiMismatchError`, and `InvalidArgumentError`. TypeScript now exports matching
`@hologram/sdk` error classes, `ERROR_*` constants, and `errorFromCode(...)`;
the native and WASM adapters translate driver `lastErrorCode()` /
`lastErrorMessage()` into those classes. Python tests cover bad archives,
bad external tensor hashes, missing inputs, and execution failures. TypeScript
golden and installed-package smokes cover SDK-side invalid inputs plus native
/ WASM archive-load and external-tensor mappings.

**8.11 SDK/FFI diagnostics and metadata slice (2026-06-02):** The FFI now
exposes source-position diagnostic fields (`line`, `column`, `rejected`) plus
session input/output dtype accessors. Python, N-API, and WASM adapters preserve
those diagnostics in SDK error objects and expose dtype / extension
introspection through loaded sessions. Node source-builder and session handles
now have explicit finalizer backstops. The SDK generator now emits richer op
metadata (`arity`, dtype policy, shape policy, docs), Python op wrapper
functions, and TypeScript `GeneratedTensorMethods` / `OpOptionsFor` types.
File-backed `const_ref` now has an opt-in
`HOLOGRAM_EXTERNAL_TENSOR_ROOT` compile root policy, tests prove outside-root
rejection and compile-time embedding, and docs cover the SDK/FFI usage path.

**8.12 SDK source helper slice (2026-06-02):** Python now exposes
`hg.compile_source(...)` and `hg.compile_source_file("graph.txt")` over
`hologram_compile_source`. `@hologram/sdk` exports browser-safe
`compileSource(source, binding)`, while `@hologram/native` adds Node-only
`compileSourceFile("graph.txt", binding)` and both N-API / WASM drivers expose
the direct `compileSource` method. Tests cover Python source strings/files,
TypeScript public types, SDK golden byte forwarding, native addon direct source
compilation, native installed-package file compilation, and WASM adapter
forwarding. Docs now show `.txt` source compilation separately from
host-language frontend extraction.

## Risks and Decisions

- **FFI vs parser boundary:** Do not expose parser internals through FFI.
  Parser frontends extract graph regions from files; SDKs build the common
  contract directly.
- **Native packaging complexity:** Bundled native libraries are the best
  developer experience but require release automation and platform testing.
- **Browser constraints:** Browser packages cannot assume filesystem access,
  native threads, or native SIMD.
- **Generated API drift:** Generated SDK methods must be checked in or verified
  in CI so package APIs do not silently diverge from canonical ops.
- **External tensor policy:** File path roots, hash requirements, and substrate
  store integration need explicit decisions before large-weight examples ship.

## Completion Criteria

- [x] Stable FFI builder API exists and is tested.
- [x] Python and TypeScript SDKs can build graphs without parsing source.
- [x] Generated op bindings are derived from canonical op metadata.
- [x] External tensor references are available through SDKs.
- [x] SDK/parser cross-language golden tests pass.
- [x] Package release matrix and import smoke tests exist for supported
  platforms.
- [x] Docs clearly separate source parsing from SDK graph construction.
