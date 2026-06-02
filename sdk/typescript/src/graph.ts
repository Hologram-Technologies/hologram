import {
  OPS,
  REQUIRED_FEATURES,
  f32,
  opCall,
  type DType,
  type GeneratedTensorMethods,
  type OpAttrs,
  type OpName,
} from "./generated.js";

export type Shape = readonly number[];
export type TensorInput = Tensor | string;
export type TensorHandle = Tensor & TensorOpMethods;

export interface TensorOpMethods extends GeneratedTensorMethods<TensorHandle, TensorInput, OpOptions> {}

export interface LowLevelBuilder {
  input(name: string, desc: { readonly dtype: DType; readonly shape?: Shape }): string;
  const(name: string, desc: ConstOptions): string;
  constRef(name: string, desc: ConstRefOptions): string;
  op(output: string, op: OpName, inputs: readonly string[], attrs?: OpAttrs): string;
  output(name: string, source?: string): void;
  compile(): Uint8Array | Promise<Uint8Array>;
}

export interface NativeBinding {
  compileSource?(source: Uint8Array): Uint8Array | Promise<Uint8Array>;
  sourceBuilder(): LowLevelBuilder;
  sessionLoad?(archive: Uint8Array): LowLevelSession | Promise<LowLevelSession>;
  featureSupported(feature: string): boolean;
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

export type ByteInput = Uint8Array | ArrayBuffer | ArrayBufferView;
export type SourceInput = string | ByteInput;
export type SessionInputs = ByteInput | readonly ByteInput[] | Record<string, ByteInput>;

export interface TensorOptions {
  readonly dtype?: DType;
  readonly shape?: Shape;
}

export interface ConstOptions extends TensorOptions {
  readonly shape: Shape;
  readonly values: readonly number[];
}

export interface ConstRefOptions extends TensorOptions {
  readonly shape: Shape;
  readonly file: string;
  readonly blake3: string;
  readonly byteLen?: number;
  readonly byteOffset?: number;
}

export interface OpOptions extends OpAttrs {
  readonly dtype?: DType;
  readonly shape?: Shape;
  readonly as?: string;
}

export const ERROR_PARSE = 1;
export const ERROR_GRAPH = 2;
export const ERROR_UNSUPPORTED_OP = 3;
export const ERROR_BAD_ATTR = 4;
export const ERROR_SHAPE = 5;
export const ERROR_EXTERNAL_TENSOR = 6;
export const ERROR_ARCHIVE_LOAD = 7;
export const ERROR_EXECUTION = 8;
export const ERROR_ABI_MISMATCH = 9;
export const ERROR_INVALID_ARGUMENT = 10;
export const ERROR_UNSUPPORTED_DTYPE = 11;
export const ERROR_COMPILE = 12;

export interface ErrorDiagnostic {
  readonly line?: number;
  readonly column?: number;
  readonly rejected?: string;
}

type Item =
  | { readonly kind: "input"; readonly name: string; readonly options: RequiredTensorOptions }
  | { readonly kind: "const"; readonly name: string; readonly options: ConstOptions }
  | { readonly kind: "constRef"; readonly name: string; readonly options: ConstRefOptions }
  | { readonly kind: "op"; readonly name: string; readonly op: OpName; readonly inputs: readonly string[]; readonly attrs: OpAttrs };

interface RequiredTensorOptions {
  readonly dtype: DType;
  readonly shape?: Shape;
}

export class HologramError extends Error {
  readonly line?: number;
  readonly column?: number;
  readonly rejected?: string;

  constructor(message: string, readonly code?: number, diagnostic: ErrorDiagnostic = {}) {
    super(message);
    this.line = diagnostic.line;
    this.column = diagnostic.column;
    this.rejected = diagnostic.rejected;
  }
}

export class NativeError extends HologramError {
  constructor(code: number, message: string, diagnostic: ErrorDiagnostic = {}) {
    super(message, code, diagnostic);
  }
}

export class ParseError extends NativeError {}
export class GraphError extends NativeError {}
export class UnsupportedOpError extends NativeError {}
export class ShapeError extends NativeError {}
export class ExternalTensorError extends NativeError {}
export class ArchiveLoadError extends NativeError {}
export class ExecutionError extends NativeError {}
export class AbiMismatchError extends NativeError {}
export class UnsupportedDTypeError extends NativeError {}
export class CompileError extends NativeError {}

export class UnknownOpError extends UnsupportedOpError {
  constructor(message: string) {
    super(ERROR_UNSUPPORTED_OP, message);
  }
}

export class BadAttrError extends NativeError {
  constructor(message: string, diagnostic: ErrorDiagnostic = {}) {
    super(ERROR_BAD_ATTR, message, diagnostic);
  }
}

export class InvalidArgumentError extends NativeError {
  constructor(message: string, diagnostic: ErrorDiagnostic = {}) {
    super(ERROR_INVALID_ARGUMENT, message, diagnostic);
  }
}

export class Tensor {
  constructor(readonly graph: Graph, readonly name: string) {}

  call(op: OpName, inputs: readonly TensorInput[] = [], attrs: OpOptions = {}): TensorHandle {
    return this.graph.op(op, [this, ...inputs], attrs);
  }
}

export class Graph {
  readonly name?: string;
  private readonly items: Item[] = [];
  private readonly outputs: Array<readonly [string, string]> = [];
  private readonly names = new Set<string>();
  private nextTmp = 0;

  constructor(name?: string) {
    this.name = name;
  }

  input(name: string, options: TensorOptions = {}): TensorHandle {
    this.add({ kind: "input", name, options: tensorOptions(options) });
    return tensor(this, name);
  }

  const(name: string, options: ConstOptions): TensorHandle {
    this.add({ kind: "const", name, options: normalizeConst(options) });
    return tensor(this, name);
  }

  constRef(name: string, options: ConstRefOptions): TensorHandle {
    this.add({ kind: "constRef", name, options: normalizeConstRef(options) });
    return tensor(this, name);
  }

  op(op: OpName, inputs: readonly TensorInput[], options: OpOptions = {}): TensorHandle {
    const name = options.as ?? this.tmp();
    const attrs = opAttrs(op, options);
    this.add({ kind: "op", name, op, inputs: inputs.map(tensorName), attrs });
    return tensor(this, name);
  }

  output(name: string, value?: TensorInput): this {
    this.outputs.push([name, value === undefined ? name : tensorName(value)]);
    return this;
  }

  emit(builder: LowLevelBuilder): LowLevelBuilder {
    for (const item of this.items) {
      emitItem(builder, item);
    }
    for (const [name, source] of this.outputs) {
      builder.output(name, source);
    }
    return builder;
  }

  async compile(native: NativeBinding): Promise<Uint8Array> {
    checkFeatures(native);
    return await this.emit(native.sourceBuilder()).compile();
  }

  private add(item: Item): void {
    if (this.names.has(item.name)) {
      throw new GraphError(ERROR_GRAPH, `duplicate tensor name: ${item.name}`);
    }
    this.names.add(item.name);
    this.items.push(item);
  }

  private tmp(): string {
    return `_t${this.nextTmp++}`;
  }
}

export class Session {
  private closed = false;

  private constructor(private readonly session: LowLevelSession) {}

  static async load(archive: ByteInput, native: NativeBinding): Promise<Session> {
    checkFeatures(native);
    if (native.sessionLoad === undefined) {
      throw new InvalidArgumentError("native binding does not support sessions");
    }
    return new Session(await native.sessionLoad(bytes(archive)));
  }

  get inputCount(): number {
    this.requireOpen();
    return this.session.inputCount();
  }

  get outputCount(): number {
    this.requireOpen();
    return this.session.outputCount();
  }

  get kernelCount(): number {
    this.requireOpen();
    return this.session.kernelCount();
  }

  get archiveFingerprint(): Uint8Array {
    this.requireOpen();
    return this.session.archiveFingerprint();
  }

  inputName(index: number): string {
    this.requireOpen();
    return this.session.inputName(index);
  }

  outputName(index: number): string {
    this.requireOpen();
    return this.session.outputName(index);
  }

  inputShape(index: number): Shape {
    this.requireOpen();
    return this.session.inputShape(index);
  }

  outputShape(index: number): Shape {
    this.requireOpen();
    return this.session.outputShape(index);
  }

  outputByteLen(index: number): number {
    this.requireOpen();
    return this.session.outputByteLen(index);
  }

  inputDType(index: number): DType {
    this.requireOpen();
    return this.session.inputDType(index);
  }

  outputDType(index: number): DType {
    this.requireOpen();
    return this.session.outputDType(index);
  }

  extension(key: string): Uint8Array | null {
    this.requireOpen();
    return this.session.extension(key);
  }

  async execute(inputs: SessionInputs): Promise<Record<string, Uint8Array>> {
    this.requireOpen();
    return namedOutputs(await this.session.execute(this.orderedInputs(inputs)), this.outputNames());
  }

  async close(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      await this.session.close();
    }
  }

  private orderedInputs(inputs: SessionInputs): readonly Uint8Array[] {
    if (isByteInput(inputs)) {
      return checkedInputs([bytes(inputs)], this.inputCount);
    }
    if (isInputArray(inputs)) {
      return checkedInputs(inputs.map(bytes), this.inputCount);
    }
    return Array.from({ length: this.inputCount }, (_, i) => namedInput(inputs, this.inputName(i)));
  }

  private outputNames(): readonly string[] {
    return Array.from({ length: this.outputCount }, (_, i) => this.outputName(i) || String(i));
  }

  private requireOpen(): void {
    if (this.closed) {
      throw new InvalidArgumentError("session is closed");
    }
  }
}

export async function compileSource(source: SourceInput, native: NativeBinding): Promise<Uint8Array> {
  checkFeatures(native);
  if (native.compileSource === undefined) {
    throw new InvalidArgumentError("native binding does not support source compilation");
  }
  return await native.compileSource(sourceBytes(source));
}

function tensor(graph: Graph, name: string): TensorHandle {
  const base = new Tensor(graph, name);
  return new Proxy(base, { get: tensorProperty }) as TensorHandle;
}

function tensorProperty(target: Tensor, prop: string | symbol, receiver: unknown): unknown {
  if (typeof prop !== "string" || !(prop in OPS)) {
    return Reflect.get(target, prop, receiver);
  }
  return (...args: readonly unknown[]) => target.call(prop as OpName, methodInputs(args), methodAttrs(args));
}

function methodInputs(args: readonly unknown[]): readonly TensorInput[] {
  if (args.length === 0 || isAttrs(args[0])) {
    return [];
  }
  return args.slice(0, isAttrs(args[args.length - 1]) ? -1 : undefined) as TensorInput[];
}

function methodAttrs(args: readonly unknown[]): OpOptions {
  const last = args[args.length - 1];
  return isAttrs(last) ? (last as OpOptions) : {};
}

function isAttrs(value: unknown): boolean {
  return typeof value === "object" && value !== null && !(value instanceof Tensor) && !Array.isArray(value);
}

function tensorOptions(options: TensorOptions): RequiredTensorOptions {
  return { dtype: options.dtype ?? f32, shape: options.shape };
}

function normalizeConst(options: ConstOptions): ConstOptions {
  return { ...options, dtype: options.dtype ?? f32 };
}

function normalizeConstRef(options: ConstRefOptions): ConstRefOptions {
  return { ...options, dtype: options.dtype ?? f32, byteOffset: options.byteOffset ?? 0 };
}

function opAttrs(op: OpName, options: OpOptions): OpAttrs {
  const attrs: OpAttrs = { ...options };
  const dtype = attrs.dtype;
  delete attrs.as;
  delete attrs.dtype;
  if (dtype !== undefined && dtype !== f32) {
    attrs.dtype = dtype;
  }
  validateAttrs(op, attrs);
  return attrs;
}

function validateAttrs(op: OpName, attrs: OpAttrs): void {
  const allowed = new Set<string>(OPS[op].attrs);
  const unknown = Object.keys(attrs).filter((name) => name !== "shape" && !allowed.has(name));
  if (unknown.length > 0) {
    throw new BadAttrError(`${op}: unsupported attrs: ${unknown.join(", ")}`);
  }
}

function tensorName(value: TensorInput): string {
  return typeof value === "string" ? value : value.name;
}

function emitItem(builder: LowLevelBuilder, item: Item): void {
  if (item.kind === "input") {
    builder.input(item.name, item.options);
  } else if (item.kind === "const") {
    builder.const(item.name, item.options);
  } else if (item.kind === "constRef") {
    builder.constRef(item.name, item.options);
  } else {
    opCall(builder, item.name, item.op, item.inputs, item.attrs);
  }
}

function checkFeatures(native: NativeBinding): void {
  const missing = REQUIRED_FEATURES.filter((feature) => !native.featureSupported(feature));
  if (missing.length > 0) {
    throw new AbiMismatchError(ERROR_ABI_MISMATCH, `native binding missing features: ${missing.join(", ")}`);
  }
}

function bytes(value: ByteInput): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

function sourceBytes(value: SourceInput): Uint8Array {
  return typeof value === "string" ? utf8(value) : bytes(value);
}

function utf8(value: string): Uint8Array {
  const out: number[] = [];
  for (const char of value) {
    pushUtf8(out, char.codePointAt(0) ?? 0);
  }
  return new Uint8Array(out);
}

function pushUtf8(out: number[], code: number): void {
  if (code <= 0x7f) {
    out.push(code);
  } else if (code <= 0x7ff) {
    out.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
  } else if (code <= 0xffff) {
    out.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
  } else {
    out.push(0xf0 | (code >> 18), 0x80 | ((code >> 12) & 0x3f), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
  }
}

function isByteInput(value: SessionInputs): value is ByteInput {
  return value instanceof ArrayBuffer || ArrayBuffer.isView(value);
}

function isInputArray(value: SessionInputs): value is readonly ByteInput[] {
  return Array.isArray(value);
}

function checkedInputs(values: readonly Uint8Array[], expected: number): readonly Uint8Array[] {
  if (values.length !== expected) {
    throw new InvalidArgumentError("input count mismatch");
  }
  return values;
}

function namedInput(inputs: Record<string, ByteInput>, name: string): Uint8Array {
  if (!(name in inputs)) {
    throw new InvalidArgumentError(`missing input: ${name}`);
  }
  return bytes(inputs[name]);
}

function namedOutputs(outputs: readonly Uint8Array[], names: readonly string[]): Record<string, Uint8Array> {
  return Object.fromEntries(outputs.map((output, index) => [names[index], output]));
}

export function errorFromCode(code: number, message: string, diagnostic: ErrorDiagnostic = {}): NativeError {
  const factory = ERROR_TYPES.get(code);
  return factory === undefined ? new NativeError(code, message, diagnostic) : factory(code, message, diagnostic);
}

type ErrorFactory = (code: number, message: string, diagnostic: ErrorDiagnostic) => NativeError;

const ERROR_TYPES = new Map<number, ErrorFactory>([
  [ERROR_PARSE, (code, message, diagnostic) => new ParseError(code, message, diagnostic)],
  [ERROR_GRAPH, (code, message, diagnostic) => new GraphError(code, message, diagnostic)],
  [ERROR_UNSUPPORTED_OP, (code, message, diagnostic) => new UnsupportedOpError(code, message, diagnostic)],
  [ERROR_BAD_ATTR, (_code, message, diagnostic) => new BadAttrError(message, diagnostic)],
  [ERROR_SHAPE, (code, message, diagnostic) => new ShapeError(code, message, diagnostic)],
  [ERROR_EXTERNAL_TENSOR, (code, message, diagnostic) => new ExternalTensorError(code, message, diagnostic)],
  [ERROR_ARCHIVE_LOAD, (code, message, diagnostic) => new ArchiveLoadError(code, message, diagnostic)],
  [ERROR_EXECUTION, (code, message, diagnostic) => new ExecutionError(code, message, diagnostic)],
  [ERROR_ABI_MISMATCH, (code, message, diagnostic) => new AbiMismatchError(code, message, diagnostic)],
  [ERROR_INVALID_ARGUMENT, (_code, message, diagnostic) => new InvalidArgumentError(message, diagnostic)],
  [ERROR_UNSUPPORTED_DTYPE, (code, message, diagnostic) => new UnsupportedDTypeError(code, message, diagnostic)],
  [ERROR_COMPILE, (code, message, diagnostic) => new CompileError(code, message, diagnostic)],
]);
