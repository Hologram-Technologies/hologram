import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const root = join(pkg, "..", "..", "..");

const source = candidateSources().find((path) => existsSync(path));
const destination = join(pkg, "dist", "hologram.node");

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
