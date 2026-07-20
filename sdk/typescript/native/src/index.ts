import {
  ERROR_ABI_MISMATCH,
  ERROR_ARCHIVE_LOAD,
  ERROR_BAD_ATTR,
  ERROR_COMPILE,
  ERROR_EXECUTION,
  ERROR_EXTERNAL_TENSOR,
  ERROR_GRAPH,
  ERROR_INVALID_ARGUMENT,
  ERROR_SHAPE,
  ERROR_UNSUPPORTED_OP,
  REQUIRED_FEATURES,
  f32,
  compileSource,
  errorFromCode,
  NativeError,
  type ConstOptions,
  type ConstRefOptions,
  type DType,
  type LowLevelBuilder,
  type LowLevelSession,
  type NativeBinding,
  type OpAttrs,
  type OpName,
  type Shape,
} from "@uor-foundation/sdk";
import { Buffer } from "node:buffer";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

export type NativeBuilderHandle = object | number | bigint;

export interface NativeAddon {
  abiVersion(): number;
  archiveFormatVersion(): number;
  featureSupported(feature: string): boolean;
  lastErrorCode(): number;
  lastErrorMessage(): string | null;
  lastErrorLine(): number;
  lastErrorColumn(): number;
  lastErrorRejected(): string | null;
  sourceBuilderNew(): NativeBuilderHandle;
  sourceBuilderFree(handle: NativeBuilderHandle): void;
  sourceBuilderInput(handle: NativeBuilderHandle, desc: TensorDesc): number;
  sourceBuilderConst(handle: NativeBuilderHandle, desc: ConstDesc): number;
  sourceBuilderConstRef(handle: NativeBuilderHandle, desc: ConstRefDesc): number;
  sourceBuilderOp(handle: NativeBuilderHandle, desc: OpDesc): number;
  sourceBuilderOutput(handle: NativeBuilderHandle, name: string): number;
  sourceBuilderOutputAlias(handle: NativeBuilderHandle, name: string, source: string): number;
  sourceBuilderCompile(handle: NativeBuilderHandle): Uint8Array;
  compileSource(source: Uint8Array): Uint8Array;
  sessionLoad(archive: Uint8Array): number;
  sessionInputCount(handle: number): number;
  sessionOutputCount(handle: number): number;
  sessionKernelCount(handle: number): number;
  sessionOutputByteLen(handle: number, index: number): number;
  sessionInputDType(handle: number, index: number): number;
  sessionOutputDType(handle: number, index: number): number;
  sessionArchiveFingerprint(handle: number): Uint8Array;
  sessionInputName(handle: number, index: number): string;
  sessionOutputName(handle: number, index: number): string;
  sessionInputShape(handle: number, index: number): number[];
  sessionOutputShape(handle: number, index: number): number[];
  sessionExtension(handle: number, key: string): Uint8Array | null;
  sessionExecute(handle: number, inputs: readonly Uint8Array[]): Uint8Array[];
  sessionClose(handle: number): number;
}

export interface TensorDesc {
  readonly name: string;
  readonly dtype: number;
  readonly shape?: Shape;
}

export interface ConstDesc {
  readonly tensor: TensorDesc;
  readonly bytes: Uint8Array;
}

export interface ConstRefDesc {
  readonly tensor: TensorDesc;
  readonly file: string;
  readonly byteOffset: number;
  readonly byteLen: number;
  readonly blake3: string;
}

export interface OpDesc {
  readonly output: string;
  readonly op: OpName;
  readonly inputs: readonly string[];
  readonly shape?: Shape;
}

export class HologramNativeError extends NativeError {}

const builderFinalizer = new FinalizationRegistry<{
  readonly addon: NativeAddon;
  readonly handle: NativeBuilderHandle;
}>(({ addon, handle }) => {
  try {
    addon.sourceBuilderFree(handle);
  } catch {
    // Finalizers cannot report errors to callers.
  }
});

const sessionFinalizer = new FinalizationRegistry<{
  readonly addon: NativeAddon;
  readonly handle: number;
}>(({ addon, handle }) => {
  try {
    addon.sessionClose(handle);
  } catch {
    // Finalizers cannot report errors to callers.
  }
});

export function createNativeBinding(addon: NativeAddon = loadNativeAddon()): NativeBinding {
  checkNative(addon);
  return {
    compileSource: (source) => nativeCall(addon, ERROR_COMPILE, "compileSource", () => addon.compileSource(nodeBuffer(source))),
    featureSupported: (feature) => addon.featureSupported(feature),
    sourceBuilder: () => new NativeSourceBuilder(addon),
    sessionLoad: (archive) => {
      const handle = nativeCall(addon, ERROR_ARCHIVE_LOAD, "sessionLoad", () => addon.sessionLoad(archive));
      if (handle < 0) {
        throw nativeError(addon, ERROR_ARCHIVE_LOAD, "sessionLoad");
      }
      return new NativeSession(addon, handle);
    },
  };
}

export async function compileSourceFile(path: string, binding: NativeBinding = createNativeBinding()): Promise<Uint8Array> {
  return compileSource(await readFile(path), binding);
}

export function loadNativeAddon(): NativeAddon {
  const require = createRequire(import.meta.url);
  for (const path of addonCandidates()) {
    try {
      return require(path) as NativeAddon;
    } catch {
      continue;
    }
  }
  throw new HologramNativeError(0, "unable to load @uor-foundation/native binary");
}

class NativeSourceBuilder implements LowLevelBuilder {
  private readonly handle: NativeBuilderHandle;
  private readonly finalizerToken = {};
  private freed = false;

  constructor(private readonly addon: NativeAddon) {
    this.handle = nativeCall(addon, ERROR_INVALID_ARGUMENT, "sourceBuilderNew", () => addon.sourceBuilderNew());
    builderFinalizer.register(this, { addon, handle: this.handle }, this.finalizerToken);
  }

  input(name: string, desc: { readonly dtype: number; readonly shape?: Shape }): string {
    this.call(ERROR_INVALID_ARGUMENT, "input", () => this.addon.sourceBuilderInput(this.handle, tensor(name, desc)));
    return name;
  }

  const(name: string, desc: ConstOptions): string {
    this.call(ERROR_SHAPE, "const", () => this.addon.sourceBuilderConst(this.handle, constant(name, desc)));
    return name;
  }

  constRef(name: string, desc: ConstRefOptions): string {
    this.call(ERROR_EXTERNAL_TENSOR, "constRef", () => this.addon.sourceBuilderConstRef(this.handle, constRef(name, desc)));
    return name;
  }

  op(output: string, op: OpName, inputs: readonly string[], attrs: OpAttrs = {}): string {
    this.call(ERROR_UNSUPPORTED_OP, "op", () => this.addon.sourceBuilderOp(this.handle, operation(output, op, inputs, attrs)));
    return output;
  }

  output(name: string, source?: string): void {
    const actual = source ?? name;
    const result = nativeCall(this.addon, ERROR_GRAPH, "output", () => actual === name
      ? this.addon.sourceBuilderOutput(this.handle, name)
      : this.addon.sourceBuilderOutputAlias(this.handle, name, actual));
    this.check(result, ERROR_GRAPH, "output");
  }

  compile(): Uint8Array {
    try {
      return nativeCall(this.addon, ERROR_COMPILE, "compile", () => this.addon.sourceBuilderCompile(this.handle));
    } finally {
      this.free();
    }
  }

  free(): void {
    if (!this.freed) {
      this.addon.sourceBuilderFree(this.handle);
      this.freed = true;
      builderFinalizer.unregister(this.finalizerToken);
    }
  }

  private call(code: number, context: string, fn: () => number): void {
    this.check(nativeCall(this.addon, code, context, fn), code, context);
  }

  private check(result: number, code: number, context: string): void {
    if (result >= 0) {
      return;
    }
    throw nativeError(this.addon, code, context);
  }
}

class NativeSession implements LowLevelSession {
  private readonly finalizerToken = {};
  private closed = false;

  constructor(private readonly addon: NativeAddon, private readonly handle: number) {
    sessionFinalizer.register(this, { addon, handle }, this.finalizerToken);
  }

  inputCount(): number {
    return this.value(this.addon.sessionInputCount(this.handle), ERROR_INVALID_ARGUMENT, "inputCount");
  }

  outputCount(): number {
    return this.value(this.addon.sessionOutputCount(this.handle), ERROR_INVALID_ARGUMENT, "outputCount");
  }

  kernelCount(): number {
    return this.value(this.addon.sessionKernelCount(this.handle), ERROR_INVALID_ARGUMENT, "kernelCount");
  }

  archiveFingerprint(): Uint8Array {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "archiveFingerprint", () => this.addon.sessionArchiveFingerprint(this.handle));
  }

  inputName(index: number): string {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "inputName", () => this.addon.sessionInputName(this.handle, index));
  }

  outputName(index: number): string {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "outputName", () => this.addon.sessionOutputName(this.handle, index));
  }

  inputShape(index: number): Shape {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "inputShape", () => this.addon.sessionInputShape(this.handle, index));
  }

  outputShape(index: number): Shape {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "outputShape", () => this.addon.sessionOutputShape(this.handle, index));
  }

  outputByteLen(index: number): number {
    return this.value(this.addon.sessionOutputByteLen(this.handle, index), ERROR_INVALID_ARGUMENT, "outputByteLen");
  }

  inputDType(index: number): DType {
    return this.value(this.addon.sessionInputDType(this.handle, index), ERROR_INVALID_ARGUMENT, "inputDType") as DType;
  }

  outputDType(index: number): DType {
    return this.value(this.addon.sessionOutputDType(this.handle, index), ERROR_INVALID_ARGUMENT, "outputDType") as DType;
  }

  extension(key: string): Uint8Array | null {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_INVALID_ARGUMENT, "extension", () => this.addon.sessionExtension(this.handle, key));
  }

  execute(inputs: readonly Uint8Array[]): readonly Uint8Array[] {
    this.requireOpen();
    return nativeCall(this.addon, ERROR_EXECUTION, "execute", () => this.addon.sessionExecute(this.handle, inputs));
  }

  close(): void {
    if (!this.closed) {
      this.check(this.addon.sessionClose(this.handle), ERROR_INVALID_ARGUMENT, "sessionClose");
      this.closed = true;
      sessionFinalizer.unregister(this.finalizerToken);
    }
  }

  private value(result: number, code: number, context: string): number {
    this.check(result, code, context);
    return result;
  }

  private check(result: number, code: number, context: string): void {
    this.requireOpen();
    if (result < 0) {
      throw nativeError(this.addon, code, context);
    }
  }

  private requireOpen(): void {
    if (this.closed) {
      throw errorFromCode(ERROR_INVALID_ARGUMENT, "session is closed");
    }
  }
}

function checkNative(addon: NativeAddon): void {
  if (addon.abiVersion() !== 1) {
    throw errorFromCode(ERROR_ABI_MISMATCH, `unsupported Hologram ABI ${addon.abiVersion()}`);
  }
  if (addon.archiveFormatVersion() !== 2 && addon.archiveFormatVersion() !== 3) {
    throw errorFromCode(ERROR_ABI_MISMATCH, `unsupported Hologram archive format ${addon.archiveFormatVersion()}`);
  }
  for (const feature of REQUIRED_FEATURES) {
    if (!addon.featureSupported(feature)) {
      throw errorFromCode(ERROR_ABI_MISMATCH, `native binding missing feature: ${feature}`);
    }
  }
}

function tensor(name: string, desc: { readonly dtype?: number; readonly shape?: Shape }): TensorDesc {
  return { name, dtype: desc.dtype ?? f32, shape: desc.shape };
}

function constant(name: string, desc: ConstOptions): ConstDesc {
  return { tensor: tensor(name, desc), bytes: f32Bytes(desc.values) };
}

function constRef(name: string, desc: ConstRefOptions): ConstRefDesc {
  return {
    tensor: tensor(name, desc),
    file: desc.file,
    byteOffset: desc.byteOffset ?? 0,
    byteLen: desc.byteLen ?? f32ByteLen(desc.shape),
    blake3: desc.blake3,
  };
}

function operation(output: string, op: OpName, inputs: readonly string[], attrs: OpAttrs): OpDesc {
  rejectUnsupportedAttrs(attrs);
  return { output, op, inputs, shape: shapeAttr(attrs) };
}

function rejectUnsupportedAttrs(attrs: OpAttrs): void {
  const unsupported = Object.keys(attrs).filter((name) => name !== "shape");
  if (unsupported.length > 0) {
    throw errorFromCode(ERROR_BAD_ATTR, `native builder does not support op attrs: ${unsupported.join(", ")}`);
  }
}

function shapeAttr(attrs: OpAttrs): Shape | undefined {
  const shape = attrs.shape;
  if (shape === undefined) {
    return undefined;
  }
  if (Array.isArray(shape) && shape.every((dim) => Number.isInteger(dim))) {
    return shape as number[];
  }
  throw errorFromCode(ERROR_BAD_ATTR, "op shape attr must be an integer array");
}

function f32Bytes(values: readonly number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function nodeBuffer(source: Uint8Array): Buffer {
  return Buffer.from(source);
}

function f32ByteLen(shape: Shape): number {
  return shape.reduce((total, dim) => total * dim, 1) * 4;
}

function nativeCall<T>(addon: NativeAddon, code: number, context: string, fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    throw nativeError(addon, code, message(error, context));
  }
}

function nativeError(addon: NativeAddon, code: number, context: string): NativeError {
  const nativeCode = addon.lastErrorCode() || code;
  return errorFromCode(nativeCode, addon.lastErrorMessage() ?? context, {
    line: positive(addon.lastErrorLine()),
    column: positive(addon.lastErrorColumn()),
    rejected: addon.lastErrorRejected() ?? undefined,
  });
}

function message(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function positive(value: number): number | undefined {
  return value > 0 ? value : undefined;
}

function addonCandidates(): readonly string[] {
  return ["./hologram.node", "../hologram.node", "@uor-foundation/native-bin"];
}
