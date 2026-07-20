# @tryhologram/wasm

WASM adapter for the Hologram TypeScript SDK.

This package implements the SDK `NativeBinding` protocol over the
wasm-bindgen driver in `driver/`. It is browser-safe, but the driver must use
an application-provided filesystem policy for `constRef`.

```ts
import { loadWasmBinding } from "@tryhologram/wasm";
import { Graph, Session, compileSource, f32 } from "@tryhologram/sdk";
import initHologram from "./hologram_wasm.js";

const binding = await loadWasmBinding(initHologram);
const g = new Graph("encoder");
const x = g.input("x", { dtype: f32, shape: [2, 3] });
const y = x.relu();
const archive = await g.output("y", y).compile(binding);
const session = await Session.load(archive, binding);
console.log(session.inputDType(0), session.outputDType(0));
const outputs = await session.execute({ x: inputBytes });
await session.close();

const sourceArchive = await compileSource(
  "input x\nop relu x as=y\noutput y\n",
  binding,
);
```

The WASM driver should expose the same source-builder and session shape as
`@tryhologram/native`:

- `abiVersion()`
- `archiveFormatVersion()`
- `featureSupported(feature)`
- `lastErrorCode()`
- `lastErrorMessage()`
- `lastErrorLine()`
- `lastErrorColumn()`
- `lastErrorRejected()`
- `sourceBuilderNew()`
- `sourceBuilderFree(handle)`
- `sourceBuilderInput(handle, desc)`
- `sourceBuilderConst(handle, desc)`
- `sourceBuilderConstRef(handle, desc)`
- `sourceBuilderOp(handle, desc)`
- `sourceBuilderOutput(handle, name)`
- `sourceBuilderOutputAlias(handle, name, source)`
- `sourceBuilderCompile(handle)`
- `compileSource(source)`
- `sessionLoad(archive)`
- `sessionInputCount(handle)`
- `sessionOutputCount(handle)`
- `sessionKernelCount(handle)`
- `sessionOutputByteLen(handle, index)`
- `sessionInputDType(handle, index)`
- `sessionOutputDType(handle, index)`
- `sessionArchiveFingerprint(handle)`
- `sessionInputName(handle, index)`
- `sessionOutputName(handle, index)`
- `sessionInputShape(handle, index)`
- `sessionOutputShape(handle, index)`
- `sessionExtension(handle, key)`
- `sessionExecute(handle, inputs)`
- `sessionClose(handle)`

Browser constraints are explicit: `constRef` cannot read arbitrary host paths.
Browser drivers should use an application-provided virtual filesystem,
`File`/`Blob` registry, OPFS, or content-addressed store.
WASM package users should also assume async module loading, browser memory caps,
and platform-dependent thread/SIMD availability. Native path access and Node
worker assumptions belong in `@tryhologram/native`, not in this package.

Build and check:

```bash
npm run --prefix sdk/typescript/wasm typecheck
npm run --prefix sdk/typescript/wasm check:driver
npm run --prefix sdk/typescript/wasm pack:check
```

The driver scripts force the rustup stable toolchain's `rustc` so a system Rust
installation on `PATH` does not hide the installed wasm stdlib.

The adapter converts driver `lastErrorCode()` / `lastErrorMessage()` values
into the `@tryhologram/sdk` error classes. Browser drivers should use the same
numeric categories as `hologram-ffi` so application code can catch
`ArchiveLoadError`, `ExternalTensorError`, `ExecutionError`, and the other
SDK-level classes consistently across native and WASM builds.

`WasmSourceBuilder` and `WasmSession` call the driver free/close hooks
explicitly on `compile()` / `close()` and register finalizers as a leak
backstop. Browser code should still close sessions deterministically.
