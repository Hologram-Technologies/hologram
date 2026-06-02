import { deepStrictEqual, strictEqual } from "node:assert";
import { Graph, InvalidArgumentError, Session, compileSource, f32, REQUIRED_FEATURES } from "../dist/index.js";

class Recorder {
  events = [];

  featureSupported(feature) {
    return REQUIRED_FEATURES.includes(feature);
  }

  sourceBuilder() {
    return this;
  }

  sessionLoad(archive) {
    this.events.push(["sessionLoad", Array.from(archive)]);
    return new RecordedSession(this);
  }

  compileSource(source) {
    this.events.push(["compileSource", Array.from(source)]);
    return new Uint8Array([0x48, 0x4f, 0x4c, 0x4f]);
  }

  input(name, desc) {
    this.events.push(["input", name, desc.dtype, desc.shape ?? null]);
    return name;
  }

  constRef(name, desc) {
    this.events.push(["constRef", name, desc.dtype, desc.shape, desc.file, desc.blake3, desc.byteLen, desc.byteOffset]);
    return name;
  }

  const(name, desc) {
    this.events.push(["const", name, desc.dtype, desc.shape, desc.values]);
    return name;
  }

  op(output, op, inputs, attrs = {}) {
    this.events.push(["op", output, op, inputs, attrs]);
    return output;
  }

  output(name, source) {
    this.events.push(["output", name, source]);
  }

  compile() {
    return new Uint8Array([0x48, 0x4f, 0x4c, 0x4f]);
  }
}

class RecordedSession {
  constructor(recorder) {
    this.recorder = recorder;
  }

  inputCount() { return 1; }
  outputCount() { return 1; }
  kernelCount() { return 1; }
  archiveFingerprint() { return new Uint8Array(32); }
  inputName() { return "x"; }
  outputName() { return "y"; }
  inputShape() { return [2, 3]; }
  outputShape() { return [2, 2]; }
  outputByteLen() { return 4; }
  inputDType() { return f32; }
  outputDType() { return f32; }
  extension() { return null; }

  execute(inputs) {
    this.recorder.events.push(["execute", inputs.map((input) => Array.from(input))]);
    return [new Uint8Array([0, 0, 0, 0])];
  }

  close() {
    this.recorder.events.push(["close"]);
  }
}

const native = new Recorder();
const graph = new Graph("encoder");
const x = graph.input("x", { dtype: f32, shape: [2, 3] });
const w = graph.constRef("w", {
  dtype: f32,
  shape: [3, 2],
  file: "weights.bin",
  blake3: "0".repeat(64),
  byteLen: 24,
});
const y = x.matmul(w, { shape: [2, 2] });
const archive = await graph.output("y", y).compile(native);
const sourceArchive = await compileSource("input x\nop relu x as=y\noutput y\n", native);
const session = await Session.load(archive, native);
strictEqual(session.inputDType(0), f32);
strictEqual(session.outputDType(0), f32);
strictEqual(session.extension("missing"), null);
const outputs = await session.execute({ x: new Uint8Array([1, 2, 3, 4]) });
await assertRejects(InvalidArgumentError, () => session.execute({}));
await session.close();

strictEqual(String.fromCharCode(...archive), "HOLO");
strictEqual(String.fromCharCode(...sourceArchive), "HOLO");
deepStrictEqual(outputs.y, new Uint8Array([0, 0, 0, 0]));
deepStrictEqual(native.events, [
  ["input", "x", f32, [2, 3]],
  ["constRef", "w", f32, [3, 2], "weights.bin", "0".repeat(64), 24, 0],
  ["op", "_t0", "matmul", ["x", "w"], { shape: [2, 2] }],
  ["output", "y", "_t0"],
  ["compileSource", [105, 110, 112, 117, 116, 32, 120, 10, 111, 112, 32, 114, 101, 108, 117, 32, 120, 32, 97, 115, 61, 121, 10, 111, 117, 116, 112, 117, 116, 32, 121, 10]],
  ["sessionLoad", [0x48, 0x4f, 0x4c, 0x4f]],
  ["execute", [[1, 2, 3, 4]]],
  ["close"],
]);

async function assertRejects(errorClass, fn) {
  try {
    await fn();
  } catch (error) {
    strictEqual(error instanceof errorClass, true);
    return;
  }
  throw new Error("expected rejection");
}
