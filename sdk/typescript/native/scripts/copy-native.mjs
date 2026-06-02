import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const root = join(pkg, "..", "..", "..");

const source = join(root, "target", "release", libraryName());
const destination = join(pkg, "dist", "hologram.node");

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.log(`copied ${source} -> ${destination}`);

function libraryName() {
  if (process.platform === "darwin") {
    return "libhologram_node_native.dylib";
  }
  if (process.platform === "win32") {
    return "hologram_node_native.dll";
  }
  return "libhologram_node_native.so";
}
