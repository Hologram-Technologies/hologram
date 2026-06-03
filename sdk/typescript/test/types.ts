import type {
  ByteInput,
  ConstOptions,
  ConstRefOptions,
  DType,
  ErrorDiagnostic,
  Feature,
  GeneratedTensorMethods,
  LowLevelBuilder,
  LowLevelGraphBuilder,
  LowLevelSession,
  NativeBinding,
  OpArity,
  OpAttrName,
  OpAttrs,
  OpName,
  OpOptionsFor,
  OpOptions,
  OpSpec,
  SessionInputs,
  Shape,
  SourceInput,
  TensorHandle,
  TensorInput,
  TensorRef,
} from "../dist/index.js";
import { compileSource } from "../dist/index.js";

const dtype: DType = 8;
const feature: Feature = "source-builder";
const shape: Shape = [1, 2];
const tensorRef: TensorRef = "x";
const attrs: OpAttrs = { shape };
const diagnostic: ErrorDiagnostic = { line: 1, column: 2, rejected: "bad" };
const opName: OpName = "relu";
const opArity: OpArity<"matmul"> = 2;
const opAttrName: OpAttrName<"gemm"> = "alpha";
const opOptionsFor: OpOptionsFor<"gemm"> = { alpha: 1.0 };
const opSpec: OpSpec = {
  name: opName,
  arity: 1,
  attrs: [],
  dtypePolicy: "f32-source-builder",
  shapePolicy: "optional-output-shape",
  doc: "doc",
};
const opOptions: OpOptions = { shape };
const constOptions: ConstOptions = { dtype, shape, values: [1, 2] };
const constRefOptions: ConstRefOptions = {
  dtype,
  shape,
  file: "weights.bin",
  blake3: "0".repeat(64),
};
const byteInput: ByteInput = new Uint8Array();
const sourceInput: SourceInput = "input x\noutput x\n";
const sessionInputs: SessionInputs = { x: byteInput };
const tensorInput: TensorInput = tensorRef;
const tensorHandle = {} as TensorHandle;

const lowLevelGraphBuilder: LowLevelGraphBuilder = {
  op: (output) => output,
};

type GeneratedMethods = GeneratedTensorMethods<TensorHandle, TensorInput, OpOptions>;
const generatedMethods = {} as GeneratedMethods;
const generatedTensor: TensorHandle = generatedMethods.matmul(tensorInput, opOptions);

const builder: LowLevelBuilder = {
  input: (name) => name,
  const: (name) => name,
  constRef: (name) => name,
  op: (output) => output,
  output: () => undefined,
  compile: () => new Uint8Array(),
};

const session: LowLevelSession = {
  inputCount: () => 1,
  outputCount: () => 1,
  kernelCount: () => 1,
  archiveFingerprint: () => new Uint8Array(32),
  inputName: () => "x",
  outputName: () => "y",
  inputShape: () => shape,
  outputShape: () => shape,
  outputByteLen: () => 4,
  inputDType: () => dtype,
  outputDType: () => dtype,
  extension: () => null,
  execute: () => [new Uint8Array()],
  close: () => undefined,
};

const native: NativeBinding = {
  compileSource: () => new Uint8Array(),
  sourceBuilder: () => builder,
  sessionLoad: () => session,
  featureSupported: (name) => name === feature,
};

const sourceArchive: Promise<Uint8Array> = compileSource(sourceInput, native);

void [
  attrs,
  constOptions,
  constRefOptions,
  diagnostic,
  lowLevelGraphBuilder,
  native,
  generatedTensor,
  generatedMethods,
  opArity,
  opAttrName,
  opOptions,
  opOptionsFor,
  opSpec,
  sessionInputs,
  sourceArchive,
  tensorHandle,
  tensorInput,
];
