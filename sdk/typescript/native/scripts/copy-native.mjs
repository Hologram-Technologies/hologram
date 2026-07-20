import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const root = join(pkg, "..", "..", "..");

// Bundled multi-platform distribution: name the copied addon per target (e.g.
// `hologram-darwin-arm64.node`) so several platforms' binaries can live side-by-side in dist/ and
// the runtime loader (src/index.ts `targetTag`) picks the matching one. The build platform is
// self-detected; override via NATIVE_TARGET_TAG where process can't tell (e.g. linux musl built on
// a gnu runner → `linux-x64-musl`). Must stay in lockstep with the loader's `targetTag()`.
const tag = process.env.NATIVE_TARGET_TAG || `${process.platform}-${process.arch}`;
const source = candidateSources().find((path) => existsSync(path));
const destination = join(pkg, "dist", `hologram-${tag}.node`);

if (source === undefined) {
  throw new Error(`native addon not found in ${candidateSources().join(" or ")}`);
}

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.error(`copied ${source} -> ${destination}`);

function candidateSources() {
  const lib = libraryName();
  return [join(pkg, "native", "target", "release", lib), join(root, "target", "release", lib)];
}

function libraryName() {
  if (process.platform === "darwin") {
    return "libhologram_node_native.dylib";
  }
  if (process.platform === "win32") {
    return "hologram_node_native.dll";
  }
  return "libhologram_node_native.so";
}
