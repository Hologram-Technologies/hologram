declare module "@tryhologram/sdk" {
  export const REQUIRED_FEATURES: readonly string[];
  export const f32: 8;
  export const ERROR_ABI_MISMATCH: 9;
  export const ERROR_ARCHIVE_LOAD: 7;
  export const ERROR_BAD_ATTR: 4;
  export const ERROR_COMPILE: 12;
  export const ERROR_EXECUTION: 8;
  export const ERROR_EXTERNAL_TENSOR: 6;
  export const ERROR_GRAPH: 2;
  export const ERROR_INVALID_ARGUMENT: 10;
  export const ERROR_PARSE: 1;
  export const ERROR_SHAPE: 5;
  export const ERROR_UNSUPPORTED_DTYPE: 11;
  export const ERROR_UNSUPPORTED_OP: 3;
  export type DType = typeof f32;
  export type Shape = readonly number[];
  export type ByteInput = Uint8Array | ArrayBuffer | ArrayBufferView;
  export type SourceInput = string | ByteInput;
  export type OpAttrs = Record<string, unknown>;
  export type OpName = string;

  export interface ConstOptions {
    readonly dtype?: DType;
    readonly shape: Shape;
    readonly values: readonly number[];
  }

  export interface ConstRefOptions {
    readonly dtype?: DType;
    readonly shape: Shape;
    readonly file: string;
    readonly blake3: string;
    readonly byteLen?: number;
    readonly byteOffset?: number;
  }

  export interface LowLevelBuilder {
    input(name: string, desc: { readonly dtype: DType; readonly shape?: Shape }): string;
    const(name: string, desc: ConstOptions): string;
    constRef(name: string, desc: ConstRefOptions): string;
    op(output: string, op: OpName, inputs: readonly string[], attrs?: OpAttrs): string;
    output(name: string, source?: string): void;
    compile(): Uint8Array | Promise<Uint8Array>;
  }

  export interface LowLevelSession {
    inputCount(): number;
    outputCount(): number;
    kernelCount(): number;
    archiveFingerprint(): Uint8Array;
    inputName(index: number): string;
    outputName(index: number): string;
    inputShape(index: number): Shape;
    outputShape(index: number): Shape;
    outputByteLen(index: number): number;
    inputDType(index: number): DType;
    outputDType(index: number): DType;
    extension(key: string): Uint8Array | null;
    execute(inputs: readonly Uint8Array[]): readonly Uint8Array[] | Promise<readonly Uint8Array[]>;
    close(): void | Promise<void>;
  }

  export interface NativeBinding {
    compileSource?(source: Uint8Array): Uint8Array | Promise<Uint8Array>;
    sourceBuilder(): LowLevelBuilder;
    sessionLoad?(archive: Uint8Array): LowLevelSession | Promise<LowLevelSession>;
    featureSupported(feature: string): boolean;
  }

  export interface ErrorDiagnostic {
    readonly line?: number;
    readonly column?: number;
    readonly rejected?: string;
  }

  export class HologramError extends Error {
    readonly code?: number;
    readonly line?: number;
    readonly column?: number;
    readonly rejected?: string;
  }

  export class NativeError extends HologramError {
    constructor(code: number, message: string, diagnostic?: ErrorDiagnostic);
  }

  export function compileSource(source: SourceInput, native: NativeBinding): Promise<Uint8Array>;
  export function errorFromCode(code: number, message: string, diagnostic?: ErrorDiagnostic): NativeError;
}
