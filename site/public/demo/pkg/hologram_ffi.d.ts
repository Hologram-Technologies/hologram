/* tslint:disable */
/* eslint-disable */

/**
 * WASM graph builder wrapping the core pipeline.
 */
export class WasmGraphBuilder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Add an edge between two builder indices.
     */
    add_edge(source: number, target: number): void;
    /**
     * Add a named graph input. Returns the input index.
     */
    add_input(name: string): number;
    /**
     * Add a node. Returns node index or negative error.
     */
    add_node(op_kind: number, op_param: number): number;
    /**
     * Add a node wired to a graph-level input.
     */
    add_node_from_input(op_kind: number, op_param: number, graph_input_idx: number): number;
    /**
     * Add a node with input edges from given builder indices.
     */
    add_node_with_inputs(op_kind: number, op_param: number, inputs: Uint32Array): number;
    /**
     * Add a named output referencing a builder index.
     */
    add_output(name: string, node_index: number): number;
    /**
     * Build and compile the graph. Returns the archive as bytes.
     */
    compile(): Uint8Array;
    /**
     * Create a new graph builder.
     */
    constructor();
}

/**
 * Benchmark batch of values: processes entire array inside WASM.
 * Returns `[holo_ns_total, native_ns_total, max_err, mean_err]`.
 * Times are total nanoseconds for one pass of `count` values
 * (averaged over `iters` repetitions).
 */
export function wasm_bench_batch(embed_encoding: number, lut_op: number, lift_encoding: number, values: Float64Array, iters: number): Float64Array;

/**
 * Benchmark fused LUT composition vs chained native transcendentals.
 * Processes `count` byte values through op1 then op2.
 * Returns `[fused_lut_ns, chained_native_ns]` — total time for one
 * pass of all values (averaged over `iters`).
 */
export function wasm_bench_composition(op1: number, op2: number, count: number, iters: number): Float64Array;

/**
 * Benchmark a single operation: runs `iters` repetitions of one value
 * entirely inside WASM. Returns `[holo_ns_per_op, native_ns_per_op,
 * holo_result, native_result]`.
 */
export function wasm_bench_single(embed_encoding: number, lut_op: number, lift_encoding: number, value: number, iters: number): Float64Array;

/**
 * Embed a value using the given encoding.
 */
export function wasm_encoding_embed(encoding_id: number, value: number): number;

/**
 * Lift a byte back to a continuous value.
 */
export function wasm_encoding_lift(encoding_id: number, byte: number): number;

/**
 * Execute a compiled `.holo` archive with the given inputs.
 *
 * `archive`: the `.holo` bytes.
 * `input_data`: a flat array of input byte data (one input assumed).
 *
 * Returns the output bytes.
 */
export function wasm_execute(archive: Uint8Array, input_data: Uint8Array): Uint8Array;

/**
 * Full hologram pipeline in a single WASM call: embed → LUT → lift.
 *
 * Eliminates JS/WASM boundary overhead for fair comparison with native.
 */
export function wasm_hologram_compute(embed_encoding: number, lut_op: number, lift_encoding: number, value: number): number;

/**
 * Apply a LUT operation to a byte.
 */
export function wasm_lut_apply(lut_op: number, byte: number): number;

/**
 * Native f64 math operation in a single WASM call.
 *
 * Uses the same LutOp index as `wasm_lut_apply` but computes via
 * actual f64 transcendentals. This ensures both hologram and native
 * paths have the same WASM boundary-crossing overhead.
 */
export function wasm_native_compute(lut_op: number, value: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgraphbuilder_free: (a: number, b: number) => void;
    readonly hologram_compilation_archive_len: (a: number) => number;
    readonly hologram_compilation_archive_ptr: (a: number) => number;
    readonly hologram_compilation_free: (a: number) => void;
    readonly hologram_compilation_stats_levels: (a: number) => number;
    readonly hologram_compilation_stats_nodes: (a: number) => number;
    readonly hologram_compilation_stats_workspace_slots: (a: number) => number;
    readonly hologram_compile: (a: number) => number;
    readonly hologram_compile_no_fuse: (a: number) => number;
    readonly hologram_encoding_embed: (a: number, b: number) => number;
    readonly hologram_encoding_lift: (a: number, b: number) => number;
    readonly hologram_error_message: () => number;
    readonly hologram_execute_bytes: (a: number, b: number, c: number) => number;
    readonly hologram_graph_builder_build: (a: number) => number;
    readonly hologram_graph_builder_edge: (a: number, b: number, c: number) => number;
    readonly hologram_graph_builder_free: (a: number) => void;
    readonly hologram_graph_builder_input: (a: number, b: number) => number;
    readonly hologram_graph_builder_new: () => number;
    readonly hologram_graph_builder_node: (a: number, b: number, c: number) => number;
    readonly hologram_graph_builder_node_from_input: (a: number, b: number, c: number, d: number) => number;
    readonly hologram_graph_builder_node_with_inputs: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly hologram_graph_builder_output: (a: number, b: number, c: number) => number;
    readonly hologram_graph_free: (a: number) => void;
    readonly hologram_graph_node_count: (a: number) => number;
    readonly hologram_inputs_free: (a: number) => void;
    readonly hologram_inputs_new: () => number;
    readonly hologram_inputs_set: (a: number, b: number, c: number, d: number) => number;
    readonly hologram_last_error: () => number;
    readonly hologram_lut_apply: (a: number, b: number) => number;
    readonly hologram_outputs_by_name: (a: number, b: number, c: number, d: number) => number;
    readonly hologram_outputs_free: (a: number) => void;
    readonly hologram_outputs_get: (a: number, b: number, c: number, d: number) => number;
    readonly hologram_outputs_len: (a: number) => number;
    readonly hologram_outputs_name: (a: number, b: number) => number;
    readonly hologram_prim_apply_binary: (a: number, b: number, c: number) => number;
    readonly hologram_prim_apply_unary: (a: number, b: number) => number;
    readonly wasm_bench_batch: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly wasm_bench_composition: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasm_bench_single: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasm_encoding_embed: (a: number, b: number) => number;
    readonly wasm_encoding_lift: (a: number, b: number) => number;
    readonly wasm_execute: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasm_hologram_compute: (a: number, b: number, c: number, d: number) => number;
    readonly wasm_lut_apply: (a: number, b: number) => number;
    readonly wasmgraphbuilder_add_edge: (a: number, b: number, c: number) => void;
    readonly wasmgraphbuilder_add_input: (a: number, b: number, c: number) => number;
    readonly wasmgraphbuilder_add_node: (a: number, b: number, c: number) => number;
    readonly wasmgraphbuilder_add_node_from_input: (a: number, b: number, c: number, d: number) => number;
    readonly wasmgraphbuilder_add_node_with_inputs: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly wasmgraphbuilder_add_output: (a: number, b: number, c: number, d: number) => number;
    readonly wasmgraphbuilder_compile: (a: number, b: number) => void;
    readonly wasmgraphbuilder_new: () => number;
    readonly wasm_native_compute: (a: number, b: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
