// Runs the wasm32 kernel conformance digests with Node's built-in WebAssembly
// API and prints them in the native example's format:
//
//   node run.mjs <hologram_compute_conformance_wasm.wasm>
//
// Comparison with the native digests is done by scripts/wasm-conformance.sh.
import { readFile } from "node:fs/promises";

const path = process.argv[2];
if (!path) {
  console.error("usage: node run.mjs <module.wasm>");
  process.exit(2);
}
const { instance } = await WebAssembly.instantiate(await readFile(path), {});
const hex = (v) => BigInt.asUintN(64, v).toString(16).padStart(16, "0");
console.log(`dot_sum ${hex(instance.exports.hologram_dot_sum_digest())}`);
console.log(`matmul ${hex(instance.exports.hologram_matmul_digest())}`);
