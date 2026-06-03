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
} from "@hologram/sdk";

export type WasmBuilderHandle = object | number;

export interface WasmDriver {
  abiVersion(): number;
  archiveFormatVersion(): number;
  featureSupported(feature: string): boolean;
  lastErrorCode(): number;
  lastErrorMessage(): string | null;
  lastErrorLine(): number;
  lastErrorColumn(): number;
  lastErrorRejected(): string | null;
  sourceBuilderNew(): WasmBuilderHandle;
  sourceBuilderFree(handle: WasmBuilderHandle): void;
  sourceBuilderInput(handle: WasmBuilderHandle, desc: TensorDesc): number;
  sourceBuilderConst(handle: WasmBuilderHandle, desc: ConstDesc): number;
  sourceBuilderConstRef(handle: WasmBuilderHandle, desc: ConstRefDesc): number;
  sourceBuilderOp(handle: WasmBuilderHandle, desc: OpDesc): number;
  sourceBuilderOutput(handle: WasmBuilderHandle, name: string): number;
  sourceBuilderOutputAlias(handle: WasmBuilderHandle, name: string, source: string): number;
  sourceBuilderCompile(handle: WasmBuilderHandle): Uint8Array;
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

export class HologramWasmError extends NativeError {}

const builderFinalizer = new FinalizationRegistry<{
  readonly driver: WasmDriver;
  readonly handle: WasmBuilderHandle;
}>(({ driver, handle }) => {
  try {
    driver.sourceBuilderFree(handle);
  } catch {
    // Finalizers cannot report errors to callers.
  }
});

const sessionFinalizer = new FinalizationRegistry<{
  readonly driver: WasmDriver;
  readonly handle: number;
}>(({ driver, handle }) => {
  try {
    driver.sessionClose(handle);
  } catch {
    // Finalizers cannot report errors to callers.
  }
});

export async function loadWasmBinding(load: () => Promise<WasmDriver>): Promise<NativeBinding> {
  return createWasmBinding(await load());
}

export function createWasmBinding(driver: WasmDriver): NativeBinding {
  checkDriver(driver);
  return {
    compileSource: (source) => wasmCall(driver, ERROR_COMPILE, "compileSource", () => driver.compileSource(source)),
    featureSupported: (feature) => driver.featureSupported(feature),
    sourceBuilder: () => new WasmSourceBuilder(driver),
    sessionLoad: (archive) => {
      const handle = wasmCall(driver, ERROR_ARCHIVE_LOAD, "sessionLoad", () => driver.sessionLoad(archive));
      if (handle < 0) {
        throw wasmError(driver, ERROR_ARCHIVE_LOAD, "sessionLoad");
      }
      return new WasmSession(driver, handle);
    },
  };
}

class WasmSourceBuilder implements LowLevelBuilder {
  private readonly handle: WasmBuilderHandle;
  private readonly finalizerToken = {};
  private freed = false;

  constructor(private readonly driver: WasmDriver) {
    this.handle = wasmCall(driver, ERROR_INVALID_ARGUMENT, "sourceBuilderNew", () => driver.sourceBuilderNew());
    builderFinalizer.register(this, { driver, handle: this.handle }, this.finalizerToken);
  }

  input(name: string, desc: { readonly dtype: number; readonly shape?: Shape }): string {
    this.call(ERROR_INVALID_ARGUMENT, "input", () => this.driver.sourceBuilderInput(this.handle, tensor(name, desc)));
    return name;
  }

  const(name: string, desc: ConstOptions): string {
    this.call(ERROR_SHAPE, "const", () => this.driver.sourceBuilderConst(this.handle, constant(name, desc)));
    return name;
  }

  constRef(name: string, desc: ConstRefOptions): string {
    this.call(ERROR_EXTERNAL_TENSOR, "constRef", () => this.driver.sourceBuilderConstRef(this.handle, constRef(name, desc)));
    return name;
  }

  op(output: string, op: OpName, inputs: readonly string[], attrs: OpAttrs = {}): string {
    this.call(ERROR_UNSUPPORTED_OP, "op", () => this.driver.sourceBuilderOp(this.handle, operation(output, op, inputs, attrs)));
    return output;
  }

  output(name: string, source?: string): void {
    const actual = source ?? name;
    const result = wasmCall(this.driver, ERROR_GRAPH, "output", () => actual === name
      ? this.driver.sourceBuilderOutput(this.handle, name)
      : this.driver.sourceBuilderOutputAlias(this.handle, name, actual));
    this.check(result, ERROR_GRAPH, "output");
  }

  compile(): Uint8Array {
    try {
      return wasmCall(this.driver, ERROR_COMPILE, "compile", () => this.driver.sourceBuilderCompile(this.handle));
    } finally {
      this.free();
    }
  }

  free(): void {
    if (!this.freed) {
      this.driver.sourceBuilderFree(this.handle);
      this.freed = true;
      builderFinalizer.unregister(this.finalizerToken);
    }
  }

  private call(code: number, context: string, fn: () => number): void {
    this.check(wasmCall(this.driver, code, context, fn), code, context);
  }

  private check(result: number, code: number, context: string): void {
    if (result >= 0) {
      return;
    }
    throw wasmError(this.driver, code, context);
  }
}

class WasmSession implements LowLevelSession {
  private readonly finalizerToken = {};
  private closed = false;

  constructor(private readonly driver: WasmDriver, private readonly handle: number) {
    sessionFinalizer.register(this, { driver, handle }, this.finalizerToken);
  }

  inputCount(): number {
    return this.value(this.driver.sessionInputCount(this.handle), ERROR_INVALID_ARGUMENT, "inputCount");
  }

  outputCount(): number {
    return this.value(this.driver.sessionOutputCount(this.handle), ERROR_INVALID_ARGUMENT, "outputCount");
  }

  kernelCount(): number {
    return this.value(this.driver.sessionKernelCount(this.handle), ERROR_INVALID_ARGUMENT, "kernelCount");
  }

  archiveFingerprint(): Uint8Array {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "archiveFingerprint", () => this.driver.sessionArchiveFingerprint(this.handle));
  }

  inputName(index: number): string {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "inputName", () => this.driver.sessionInputName(this.handle, index));
  }

  outputName(index: number): string {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "outputName", () => this.driver.sessionOutputName(this.handle, index));
  }

  inputShape(index: number): Shape {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "inputShape", () => this.driver.sessionInputShape(this.handle, index));
  }

  outputShape(index: number): Shape {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "outputShape", () => this.driver.sessionOutputShape(this.handle, index));
  }

  outputByteLen(index: number): number {
    return this.value(this.driver.sessionOutputByteLen(this.handle, index), ERROR_INVALID_ARGUMENT, "outputByteLen");
  }

  inputDType(index: number): DType {
    return this.value(this.driver.sessionInputDType(this.handle, index), ERROR_INVALID_ARGUMENT, "inputDType") as DType;
  }

  outputDType(index: number): DType {
    return this.value(this.driver.sessionOutputDType(this.handle, index), ERROR_INVALID_ARGUMENT, "outputDType") as DType;
  }

  extension(key: string): Uint8Array | null {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_INVALID_ARGUMENT, "extension", () => this.driver.sessionExtension(this.handle, key));
  }

  execute(inputs: readonly Uint8Array[]): readonly Uint8Array[] {
    this.requireOpen();
    return wasmCall(this.driver, ERROR_EXECUTION, "execute", () => this.driver.sessionExecute(this.handle, inputs));
  }

  close(): void {
    if (!this.closed) {
      this.check(this.driver.sessionClose(this.handle), ERROR_INVALID_ARGUMENT, "sessionClose");
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
      throw wasmError(this.driver, code, context);
    }
  }

  private requireOpen(): void {
    if (this.closed) {
      throw errorFromCode(ERROR_INVALID_ARGUMENT, "session is closed");
    }
  }
}

function checkDriver(driver: WasmDriver): void {
  if (driver.abiVersion() !== 1) {
    throw errorFromCode(ERROR_ABI_MISMATCH, `unsupported Hologram ABI ${driver.abiVersion()}`);
  }
  if (driver.archiveFormatVersion() !== 2) {
    throw errorFromCode(ERROR_ABI_MISMATCH, `unsupported Hologram archive format ${driver.archiveFormatVersion()}`);
  }
  for (const feature of REQUIRED_FEATURES) {
    if (!driver.featureSupported(feature)) {
      throw errorFromCode(ERROR_ABI_MISMATCH, `WASM binding missing feature: ${feature}`);
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
    throw errorFromCode(ERROR_BAD_ATTR, `WASM builder does not support op attrs: ${unsupported.join(", ")}`);
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

function f32ByteLen(shape: Shape): number {
  return shape.reduce((total, dim) => total * dim, 1) * 4;
}

function wasmCall<T>(driver: WasmDriver, code: number, context: string, fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    throw wasmError(driver, code, message(error, context));
  }
}

function wasmError(driver: WasmDriver, code: number, context: string): NativeError {
  const nativeCode = driver.lastErrorCode() || code;
  return errorFromCode(nativeCode, driver.lastErrorMessage() ?? context, {
    line: positive(driver.lastErrorLine()),
    column: positive(driver.lastErrorColumn()),
    rejected: driver.lastErrorRejected() ?? undefined,
  });
}

function message(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function positive(value: number): number | undefined {
  return value > 0 ? value : undefined;
}
