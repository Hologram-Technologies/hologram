import { join } from "node:path";
import { pathToFileURL } from "node:url";

const sdk = await import(installedEntry("@uor-foundation/sdk"));
const wasm = await import(installedEntry("@uor-foundation/wasm"));

let lastErrorCode = 0;
let lastErrorMessage = null;

const driver = {
  abiVersion: () => 1,
  archiveFormatVersion: () => 2,
  featureSupported: (feature) => sdk.REQUIRED_FEATURES.includes(feature),
  lastErrorCode: () => lastErrorCode,
  lastErrorMessage: () => lastErrorMessage,
  lastErrorLine: () => 3,
  lastErrorColumn: () => 5,
  lastErrorRejected: () => "bad",
  sourceBuilderNew: () => 1,
  sourceBuilderFree: () => undefined,
  sourceBuilderInput: () => 0,
  sourceBuilderConst: () => 0,
  sourceBuilderConstRef: (_handle, desc) => {
    if (desc.blake3 === "not-hex") {
      return setError(sdk.ERROR_EXTERNAL_TENSOR, "const_ref blake3 must be hex");
    }
    return clearError(0);
  },
  sourceBuilderOp: () => 0,
  sourceBuilderOutput: () => 0,
  sourceBuilderOutputAlias: () => 0,
  sourceBuilderCompile: () => new Uint8Array([0x48, 0x4f, 0x4c, 0x4f]),
  compileSource: () => new Uint8Array([0x48, 0x4f, 0x4c, 0x4f]),
  sessionLoad: (archive) => {
    if (archive[0] !== 0x48 || archive[1] !== 0x4f || archive[2] !== 0x4c || archive[3] !== 0x4f) {
      return setError(sdk.ERROR_ARCHIVE_LOAD, "archive is missing HOLO magic");
    }
    return clearError(1);
  },
  sessionInputCount: () => 1,
  sessionOutputCount: () => 1,
  sessionKernelCount: () => 1,
  sessionOutputByteLen: () => 4,
  sessionInputDType: () => sdk.f32,
  sessionOutputDType: () => sdk.f32,
  sessionArchiveFingerprint: () => new Uint8Array(32),
  sessionInputName: () => "x",
  sessionOutputName: () => "y",
  sessionInputShape: () => [1],
  sessionOutputShape: () => [1],
  sessionExtension: () => null,
  sessionExecute: () => [new Uint8Array(new Float32Array([0]).buffer)],
  sessionClose: () => 0,
};

const binding = wasm.createWasmBinding(driver);
const graph = new sdk.Graph("installed_wasm_smoke");
const x = graph.input("x", { shape: [1] });
const archive = await graph.output("y", x.relu({ shape: [1] })).compile(binding);

if (archive[0] !== 0x48 || archive[1] !== 0x4f || archive[2] !== 0x4c || archive[3] !== 0x4f) {
  throw new Error("compiled archive is missing HOLO magic");
}

const sourceArchive = await sdk.compileSource("input x\nop relu x as=y\noutput y\n", binding);
if (sourceArchive[0] !== 0x48 || sourceArchive[1] !== 0x4f || sourceArchive[2] !== 0x4c || sourceArchive[3] !== 0x4f) {
  throw new Error("compiled source archive is missing HOLO magic");
}

const session = await sdk.Session.load(archive, binding);
if (session.inputDType(0) !== sdk.f32 || session.outputDType(0) !== sdk.f32) {
  throw new Error("session dtype introspection did not report f32");
}
if (session.extension("missing") !== null) {
  throw new Error("missing session extension should be null");
}
const outputs = await session.execute({ x: new Uint8Array(new Float32Array([-1]).buffer) });
if (new Float32Array(outputs.y.buffer, outputs.y.byteOffset, 1)[0] !== 0) {
  throw new Error("session output is not relu(-1)");
}
await assertRejects(sdk.InvalidArgumentError, () => session.execute({}));
await session.close();

await assertRejects(sdk.ArchiveLoadError, () => sdk.Session.load(new Uint8Array([1, 2, 3]), binding), {
  line: 3,
  column: 5,
  rejected: "bad",
});
await assertRejects(sdk.ExternalTensorError, () => badConstRefGraph().compile(binding));

function badConstRefGraph() {
  const graph = new sdk.Graph("bad_const_ref");
  graph.constRef("w", { shape: [1], file: "weights.bin", blake3: "not-hex" });
  return graph;
}

function clearError(value) {
  lastErrorCode = 0;
  lastErrorMessage = null;
  return value;
}

function setError(code, message) {
  lastErrorCode = code;
  lastErrorMessage = message;
  return -1;
}

async function assertRejects(errorClass, fn, diagnostic = undefined) {
  try {
    await fn();
  } catch (error) {
    if (error instanceof errorClass) {
      if (diagnostic !== undefined) {
        if (error.line !== diagnostic.line || error.column !== diagnostic.column || error.rejected !== diagnostic.rejected) {
          throw new Error("diagnostic fields were not preserved");
        }
      }
      return;
    }
    throw error;
  }
  throw new Error(`expected ${errorClass.name}`);
}

function installedEntry(packageName) {
  return pathToFileURL(
    join(process.cwd(), "node_modules", ...packageName.split("/"), "dist", "index.js"),
  ).href;
}
