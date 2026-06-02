import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const addon = require(join(here, "..", "dist", "hologram.node"));

assert(addon.abiVersion() === 1, "ABI version");
assert(addon.featureSupported("source-builder.output-alias") === 1, "output alias feature");

const builder = addon.sourceBuilderNew();
assert(builder >= 0, "builder handle");
assert(addon.sourceBuilderInput(builder, { name: "x", dtype: 8, shape: [1] }) >= 0, "input");
assert(
  addon.sourceBuilderOp(builder, {
    output: "hidden",
    op: "relu",
    inputs: ["x"],
    shape: [1],
  }) >= 0,
  "op",
);
assert(addon.sourceBuilderOutputAlias(builder, "y", "hidden") >= 0, "output alias");

const archive = addon.sourceBuilderCompile(builder);
assert(archive instanceof Uint8Array, "archive bytes");
assert(archive.length > 4, "archive length");
assert(String.fromCharCode(...archive.slice(0, 4)) === "HOLO", "archive magic");

const sourceArchive = addon.compileSource(Buffer.from("input x :1\nop relu x :1 as=y\noutput y\n"));
assert(sourceArchive instanceof Uint8Array, "source archive bytes");
assert(String.fromCharCode(...sourceArchive.slice(0, 4)) === "HOLO", "source archive magic");

const session = addon.sessionLoad(archive);
assert(session >= 0, "session handle");
assert(addon.sessionInputCount(session) === 1, "input count");
assert(addon.sessionOutputCount(session) === 1, "output count");
assert(addon.sessionKernelCount(session) > 0, "kernel count");
assert(addon.sessionInputName(session, 0) === "x", "input name");
assert(addon.sessionOutputName(session, 0) === "y", "output name");
assert(addon.sessionOutputByteLen(session, 0) === 4, "output byte length");
assert(addon.sessionArchiveFingerprint(session).length === 32, "archive fingerprint");

const input = new Uint8Array(new Float32Array([-1]).buffer);
const outputs = addon.sessionExecute(session, [input]);
assert(outputs.length === 1, "execute output count");
assert(new Float32Array(outputs[0].buffer, outputs[0].byteOffset, 1)[0] === 0, "relu output");
assert(addon.sessionClose(session) === 0, "session close");

function assert(condition, label) {
  if (!condition) {
    throw new Error(`smoke failed: ${label}`);
  }
}
