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
