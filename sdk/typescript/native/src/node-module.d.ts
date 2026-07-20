declare module "node:module" {
  export function createRequire(url: string): (id: string) => unknown;
}

declare module "node:buffer" {
  export class Buffer extends Uint8Array {
    static from(source: Uint8Array): Buffer;
  }
}

declare module "node:fs/promises" {
  export function readFile(path: string): Promise<Uint8Array>;
}

interface ImportMeta {
  readonly url: string;
}

// Minimal `process` shim — the loader reads `platform`/`arch` to pick the bundled per-platform
// addon (see `targetTag` in index.ts). The optional `report` (glibc-vs-musl probe) is accessed
// through an inline cast there, so it is intentionally not declared here.
declare const process: {
  readonly platform: string;
  readonly arch: string;
};
