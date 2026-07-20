import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const sdk = await import(installedEntry("@uor-foundation/sdk"));
const native = await import(installedEntry("@uor-foundation/native"));

const binding = native.createNativeBinding();
const graph = new sdk.Graph("installed_native_smoke");
const x = graph.input("x", { shape: [1] });
const archive = await graph.output("y", x.relu({ shape: [1] })).compile(binding);

if (archive[0] !== 0x48 || archive[1] !== 0x4f || archive[2] !== 0x4c || archive[3] !== 0x4f) {
  throw new Error("compiled archive is missing HOLO magic");
}

const sourceArchive = await compileSourceFile(native);
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
const y = outputs.y;
if (!(y instanceof Uint8Array) || y.byteLength !== 4) {
  throw new Error("session output is not one f32");
}
if (new Float32Array(y.buffer, y.byteOffset, 1)[0] !== 0) {
  throw new Error("session output is not relu(-1)");
}
await assertRejects(sdk.InvalidArgumentError, () => session.execute({}));
await session.close();

await assertRejects(sdk.ArchiveLoadError, () => sdk.Session.load(new Uint8Array([1, 2, 3]), binding));
await assertRejects(sdk.ExternalTensorError, () => badConstRefGraph().compile(binding));

function badConstRefGraph() {
  const graph = new sdk.Graph("bad_const_ref");
  graph.constRef("w", { shape: [1], file: "weights.bin", blake3: "not-hex" });
  return graph;
}

async function compileSourceFile(nativePackage) {
  const dir = await mkdtemp(join(tmpdir(), "hologram-native-smoke-"));
  const path = join(dir, "graph.txt");
  try {
    await writeFile(path, "input x :1\nop relu x :1 as=y\noutput y\n");
    return await nativePackage.compileSourceFile(path);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

async function assertRejects(errorClass, fn) {
  try {
    await fn();
  } catch (error) {
    if (error instanceof errorClass) {
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
