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

const rustc = spawnSync("rustup", ["which", "rustc", "--toolchain", "stable"], {
  encoding: "utf8",
});

if (rustc.status !== 0) {
  process.stderr.write(rustc.stderr);
  process.exit(rustc.status ?? 1);
}

const result = spawnSync("rustup", ["run", "stable", "cargo", ...cargoArgs], {
  env: { ...process.env, RUSTC: rustc.stdout.trim() },
  stdio: "inherit",
});

process.exit(result.status ?? 1);
