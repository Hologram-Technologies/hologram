# Hologram SDKs

This directory contains package code that sits above the stable
`hologram-ffi` ABI.

## Layers

- `python/hologram/_generated.py` and `typescript/src/generated.ts` are
  generated from Rust metadata in `hologram-ffi::sdk`.
- `python/hologram/graph.py` and `typescript/src/graph.ts` are the human
  chainable SDK layers.
- `python/pyproject.toml` packages the Python SDK as `hologram`.
- `typescript/package.json` packages the browser-safe TypeScript SDK as
  `@uor-foundation/sdk`.
- `typescript/native/package.json` packages the Node native adapter as
  `@uor-foundation/native`.
- `typescript/wasm/package.json` packages the browser-safe WASM adapter as
  `@uor-foundation/wasm`.
- Native binary/WASM driver implementations remain separate from
  `@uor-foundation/sdk`. The human SDKs accept a binding object that exposes the
  source-builder ABI.

Refresh generated files with:

```bash
cargo run -p hologram-ffi --example generate_sdk_bindings
```

The Rust test `cargo test -p hologram-ffi --test sdk_generated` fails if the
checked-in generated files drift from canonical op/dtype/attribute metadata.

## Native and WASM Boundaries

The package split is intentional:

- Python `hologram`: Python graph builder/session wrapper plus generated op metadata.
  `Graph.compile()` imports `hologram._hologram` only when no binding is passed
  explicitly. Wheels bundle the `hologram-ffi` cdylib and `_hologram.py`
  exposes it through the same source-builder and session ABIs.
- TypeScript `@uor-foundation/sdk`: pure ESM graph builder plus generated op metadata.
  It is browser-safe because it does not import `node:`, N-API, or filesystem
  modules.
- `@uor-foundation/native`: Node adapter implementing the exported `NativeBinding`
  protocol over a N-API addon for host source-building and session execution.
- `@uor-foundation/wasm`: browser-safe adapter implementing the same protocol over a
  WASM driver.
- Future `@uor-foundation/browser`: convenience package that composes
  `@uor-foundation/sdk` with the WASM binding and browser-safe filesystem policy.

Native Hologram `.txt` source compilation is exposed through the same binding
contract:

```python
archive = hg.compile_source("input x\nop relu x as=y\noutput y\n")
archive = hg.compile_source_file("graph.txt")
```

```ts
import { compileSource } from "@uor-foundation/sdk";
import { compileSourceFile, createNativeBinding } from "@uor-foundation/native";

const native = createNativeBinding();
const archive = await compileSource("input x\nop relu x as=y\noutput y\n", native);
const fileArchive = await compileSourceFile("graph.txt", native);
```

Package checks:

```bash
PYTHONPATH=sdk/python python3 -B -m unittest discover -s sdk/python/tests -p 'test_*.py'
python3 -m pip wheel sdk/python --no-deps -w /tmp/hologram-wheel
python3 -m pip install /tmp/hologram-wheel/*.whl
python3 sdk/python/scripts/smoke-installed.py
npm run --prefix sdk/typescript typecheck
npm run --prefix sdk/typescript test:types
npm run --prefix sdk/typescript test:golden
npm run --prefix sdk/typescript pack:check
npm run --prefix sdk/typescript/native typecheck
npm run --prefix sdk/typescript/native smoke:native
npm run --prefix sdk/typescript/wasm typecheck
npm run --prefix sdk/typescript/wasm check:driver
```

The platform prebuild and installed-package smoke matrix is tracked in
[`PREBUILD.md`](PREBUILD.md) and `.github/workflows/sdk-packages.yml`.

## Error Model

The native FFI reports stable numeric categories through
`hologram_last_error_code()`. SDKs map those categories to language-native
exceptions/classes and preserve the numeric code on the thrown object.
Source-positioned failures also carry optional `line`, `column`, and
`rejected` fields from `hologram_last_error_line()`,
`hologram_last_error_column()`, and `hologram_last_error_rejected()`.

| Code | Category | Python | TypeScript |
|---:|---|---|---|
| 1 | Parse | `ParseError` | `ParseError` |
| 2 | Graph | `GraphError` | `GraphError` |
| 3 | Unsupported op | `UnsupportedOpError` / `UnknownOpError` | `UnsupportedOpError` / `UnknownOpError` |
| 4 | Bad attr | `BadAttrError` | `BadAttrError` |
| 5 | Shape | `ShapeError` | `ShapeError` |
| 6 | External tensor | `ExternalTensorError` | `ExternalTensorError` |
| 7 | Archive load | `ArchiveLoadError` | `ArchiveLoadError` |
| 8 | Execution | `ExecutionError` | `ExecutionError` |
| 9 | ABI mismatch | `AbiMismatchError` | `AbiMismatchError` |
| 10 | Invalid argument | `InvalidArgumentError` | `InvalidArgumentError` |
| 11 | Unsupported dtype | `UnsupportedDTypeError` | `UnsupportedDTypeError` |
| 12 | Compile | `CompileError` | `CompileError` |

## Session Metadata

Loaded sessions expose archive and port metadata through both SDK families:

- input/output counts, names, shapes, and dtype IDs
- kernel count
- output byte lengths for caller-owned output buffers
- canonical archive fingerprint
- producer metadata extensions by key, returning `None` / `null` when absent

Python uses `session.input_dtype(0)` / `session.output_dtype(0)` and
`session.extension("tokenizer")`. TypeScript uses
`session.inputDType(0)` / `session.outputDType(0)` and
`session.extension("tokenizer")`.

## External Tensors

`const_ref` reads file-backed tensor bytes at compile time, validates the
declared byte range and BLAKE3 digest, and embeds the bytes into the archive.
Runtime session execution never opens those source paths.

By default, relative `const_ref` paths resolve from the compiler process'
current directory and absolute paths are allowed. Set
`HOLOGRAM_EXTERNAL_TENSOR_ROOT=/path/to/root` to require every resolved
external tensor path, relative or absolute, to canonicalize under that root.
