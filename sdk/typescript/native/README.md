# @hologram/native

Node adapter and N-API driver for the Hologram TypeScript SDK.

This package is intentionally split from `@hologram/sdk`: the SDK is pure ESM
and browser-safe, while this package loads a Node native binary and implements
the SDK `NativeBinding` protocol.

```ts
import { compileSourceFile, createNativeBinding } from "@hologram/native";
import { Graph, Session, f32 } from "@hologram/sdk";

const native = createNativeBinding();
const g = new Graph("encoder");
const x = g.input("x", { dtype: f32, shape: [2, 3] });
const y = x.relu();
const archive = await g.output("y", y).compile(native);
const session = await Session.load(archive, native);
console.log(session.inputDType(0), session.outputDType(0));
const outputs = await session.execute({ x: inputBytes });
await session.close();

const sourceArchive = await compileSourceFile("graph.txt", native);
```

The native addon under `native/` wraps the stable `hologram-ffi`
source-builder and session ABIs and exports:

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

Build and check:

```bash
npm run --prefix sdk/typescript/native typecheck
npm run --prefix sdk/typescript/native smoke:native
npm run --prefix sdk/typescript/native pack:check
```

The adapter converts `lastErrorCode()` / `lastErrorMessage()` into the
`@hologram/sdk` error classes. For example, a bad archive load rejects with
`ArchiveLoadError`, invalid execute inputs reject with `InvalidArgumentError`,
and failed kernels reject with `ExecutionError`.

`NativeSourceBuilder` and `NativeSession` close their native handles explicitly
on `compile()` / `close()` and register finalizers as a leak backstop. Call
`session.close()` when done so release timing is deterministic.
