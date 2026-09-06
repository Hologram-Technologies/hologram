import { spawnSync } from "node:child_process";

const mode = process.argv[2] ?? "check";
const cargoArgs = [
  mode,
  "--manifest-path",
  "driver/Cargo.toml",
  "--target",
  "wasm32-unknown-unknown",
];

if (mode === "build") {
  cargoArgs.push("--release");
}

// Resolve the workflow/devcontainer's active pinned toolchain. Hardcoding the floating `stable`
// alias bypasses both the selected compiler and its installed wasm target.
const rustc = spawnSync("rustup", ["which", "rustc"], {
  encoding: "utf8",
});

if (rustc.status !== 0) {
  process.stderr.write(rustc.stderr);
  process.exit(rustc.status ?? 1);
}

const result = spawnSync("cargo", cargoArgs, {
  env: { ...process.env, RUSTC: rustc.stdout.trim() },
  stdio: "inherit",
});

process.exit(result.status ?? 1);
