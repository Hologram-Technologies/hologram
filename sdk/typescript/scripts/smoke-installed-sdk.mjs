import { join } from "node:path";
import { pathToFileURL } from "node:url";

const sdk = await import(installedEntry("@hologram/sdk"));

if (sdk.f32 !== 8) {
  throw new Error("unexpected f32 dtype id");
}

if (!Object.hasOwn(sdk.OPS, "matmul")) {
  throw new Error("matmul missing from installed SDK metadata");
}

const graph = new sdk.Graph("installed_smoke");
const x = graph.input("x", { shape: [1] });
graph.output("y", x.relu());

function installedEntry(packageName) {
  return pathToFileURL(
    join(process.cwd(), "node_modules", ...packageName.split("/"), "dist", "index.js"),
  ).href;
}
