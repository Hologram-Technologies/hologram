#!/usr/bin/env node
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const PACKAGE_DIRS = ["sdk/typescript", "sdk/typescript/native", "sdk/typescript/wasm"];

export function sha512Integrity(bytes) {
  return `sha512-${createHash("sha512").update(bytes).digest("base64")}`;
}

export function requireMatchingIntegrity(name, version, expected, observed) {
  if (observed !== null && observed !== expected) {
    throw new Error(
      `refusing ${name}@${version}: public integrity ${observed} differs from packed ${expected}`,
    );
  }
}

// npm 11 emits an array for `npm pack --json`; npm 12 emits an object keyed by
// package name. Accept both documented shapes while still requiring exactly one
// artifact so a CLI format change can never select an arbitrary tarball.
export function singlePackRow(output, directory) {
  const parsed = JSON.parse(output);
  const rows = Array.isArray(parsed) ? parsed : Object.values(parsed);
  if (rows.length !== 1 || typeof rows[0]?.filename !== "string") {
    throw new Error(`npm pack returned an unexpected manifest for ${directory}`);
  }
  return rows[0];
}

async function registryIntegrity(name, version) {
  const base = process.env.NPM_REGISTRY_URL ?? "https://registry.npmjs.org";
  const response = await fetch(`${base}/${encodeURIComponent(name)}/${encodeURIComponent(version)}`, {
    headers: { "user-agent": "hologram-release/0.13" },
    signal: AbortSignal.timeout(30_000),
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`npm registry preflight for ${name}@${version}: HTTP ${response.status}`);
  const body = await response.json();
  const value = body?.dist?.integrity;
  if (typeof value !== "string" || !value.startsWith("sha512-")) {
    throw new Error(`npm registry returned no SHA-512 integrity for ${name}@${version}`);
  }
  return value;
}

function packAll(stage) {
  return PACKAGE_DIRS.map((directory) => {
    const manifest = JSON.parse(readFileSync(join(directory, "package.json"), "utf8"));
    const output = execFileSync(
      "npm",
      ["pack", "--json", "--ignore-scripts", "--pack-destination", stage, `./${directory}`],
      { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
    );
    const row = singlePackRow(output, directory);
    const artifact = resolve(stage, row.filename);
    const integrity = sha512Integrity(readFileSync(artifact));
    if (row.integrity !== integrity) {
      throw new Error(`npm pack integrity disagreement for ${manifest.name}@${manifest.version}`);
    }
    return { directory, artifact, name: manifest.name, version: manifest.version, integrity };
  });
}

async function waitForIntegrity(pkg) {
  for (let attempt = 1; attempt <= 30; attempt += 1) {
    const observed = await registryIntegrity(pkg.name, pkg.version);
    requireMatchingIntegrity(pkg.name, pkg.version, pkg.integrity, observed);
    if (observed === pkg.integrity) return;
    process.stderr.write(`waiting for ${pkg.name}@${pkg.version} registry propagation (${attempt}/30)\n`);
    await new Promise((done) => setTimeout(done, 10_000));
  }
  throw new Error(`${pkg.name}@${pkg.version} did not become downloadable within 300 seconds`);
}

async function main() {
  const dryRun = process.env.DRY_RUN === "true" || process.env.DRY_RUN === "1";
  const stage = mkdtempSync(join(tmpdir(), "hologram-npm-release-"));
  try {
    // Produce every immutable tarball and compare all existing versions before the first publish.
    const packages = packAll(stage);
    const versions = new Set(packages.map(({ version }) => version));
    if (versions.size !== 1) throw new Error(`npm package versions disagree: ${[...versions].join(", ")}`);
    for (const pkg of packages) {
      pkg.observed = await registryIntegrity(pkg.name, pkg.version);
      requireMatchingIntegrity(pkg.name, pkg.version, pkg.integrity, pkg.observed);
      process.stdout.write(
        `${pkg.name}@${pkg.version} ${pkg.integrity} ${pkg.observed === null ? "missing" : "already-public"}\n`,
      );
    }
    if (dryRun) {
      process.stdout.write("DRY_RUN — complete npm pack/checksum preflight passed; not publishing.\n");
      return;
    }

    for (const pkg of packages) {
      if (pkg.observed === pkg.integrity) {
        process.stdout.write(`${pkg.name}@${pkg.version} already public with accepted integrity; skipping.\n`);
        continue;
      }
      try {
        execFileSync("npm", ["publish", pkg.artifact, "--access", "public", "--ignore-scripts"], {
          stdio: "inherit",
        });
      } catch (error) {
        // Treat a concurrent/retry publish as success only when the immutable bytes agree.
        const observed = await registryIntegrity(pkg.name, pkg.version);
        requireMatchingIntegrity(pkg.name, pkg.version, pkg.integrity, observed);
        if (observed !== pkg.integrity) throw error;
      }
      await waitForIntegrity(pkg);
    }
  } finally {
    rmSync(stage, { recursive: true, force: true });
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
