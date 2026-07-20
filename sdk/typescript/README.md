# Hologram TypeScript SDK

`@tryhologram/sdk` is the browser-safe graph builder package. It does not import
Node native modules directly; callers pass a native or WASM binding object into
`Graph.compile(...)`.

```ts
import * as hg from "@tryhologram/sdk";

const g = new hg.Graph("encoder");
const x = g.input("x", { dtype: hg.f32, shape: [2, 3] });
const w = g.constRef("w", {
  dtype: hg.f32,
  shape: [3, 2],
  file: "weights.bin",
  blake3: "0".repeat(64),
});

const y = x.matmul(w, { shape: [2, 2] }).relu();
const archive = await g.output("y", y).compile(nativeBinding);

const session = await hg.Session.load(archive, nativeBinding);
console.log(session.inputDType(0), session.outputDType(0));
console.log(session.extension("missing")); // null
const outputs = await session.execute({ x: inputBytes });
await session.close();
```

Native Hologram `.txt` source can be compiled through any binding that
implements `compileSource`:

```ts
const archive = await hg.compileSource(
  "input x\nop relu x as=y\noutput y\n",
  nativeBinding,
);
```

Native and SDK-side failures throw `HologramError` subclasses with a stable
`code` field:

```ts
try {
  await hg.Session.load(new Uint8Array([1, 2, 3]), nativeBinding);
} catch (error) {
  if (error instanceof hg.ArchiveLoadError) {
    console.log(error.code, error.line, error.column, error.rejected);
  }
}
```

`@tryhologram/sdk` exports `ParseError`, `GraphError`, `UnsupportedOpError`,
`UnknownOpError`, `BadAttrError`, `ShapeError`, `ExternalTensorError`,
`ArchiveLoadError`, `ExecutionError`, `AbiMismatchError`,
`InvalidArgumentError`, `UnsupportedDTypeError`, `CompileError`,
`errorFromCode(code, message)`, and the matching `ERROR_*` constants.
It also exports generated metadata types such as `OpName`, `OpSpec`,
`OpArity`, `OpAttrName`, `OpOptionsFor`, `GeneratedTensorMethods`,
`TensorRef`, and `LowLevelGraphBuilder`.

`constRef` is resolved at compile time by the selected binding. Native builds
read the declared file range, verify the BLAKE3 digest, and embed the bytes in
the archive. Set `HOLOGRAM_EXTERNAL_TENSOR_ROOT` to require every resolved
external tensor path to live under an explicit compile root.

Build and check the package:

```bash
npm run --prefix sdk/typescript build
npm run --prefix sdk/typescript test:types
npm run --prefix sdk/typescript pack:check
```

`@tryhologram/native` and `@tryhologram/wasm` implement the `NativeBinding` protocol
exported by this package. The native package carries the Node N-API driver; the
WASM package carries a browser-safe adapter and the wasm driver crate.
