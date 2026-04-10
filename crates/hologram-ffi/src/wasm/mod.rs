//! WASM bindings via `wasm-bindgen`.
//!
//! Feature-gated behind `wasm`. Provides JavaScript-friendly wrappers
//! around the core hologram pipeline.

use wasm_bindgen::prelude::*;

/// WASM graph builder wrapping the core pipeline.
#[wasm_bindgen]
pub struct WasmGraphBuilder {
    builder: crate::graph::FfiGraphBuilder,
}

#[wasm_bindgen]
impl WasmGraphBuilder {
    /// Create a new graph builder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: crate::graph::FfiGraphBuilder::new_internal(),
        }
    }

    /// Add a named graph input. Returns the input index.
    pub fn add_input(&mut self, name: &str) -> i32 {
        self.builder.graph.add_input(name) as i32
    }

    /// Add a node. Returns node index or negative error.
    pub fn add_node(&mut self, op_kind: i32, op_param: i32) -> i32 {
        match crate::graph::make_graph_op(op_kind, op_param) {
            Ok(op) => self.builder.add_node(op) as i32,
            Err(code) => code,
        }
    }

    /// Add a node wired to a graph-level input.
    pub fn add_node_from_input(
        &mut self,
        op_kind: i32,
        op_param: i32,
        graph_input_idx: u32,
    ) -> i32 {
        use hologram_ir::graph::edge;
        let op = match crate::graph::make_graph_op(op_kind, op_param) {
            Ok(op) => op,
            Err(code) => return code,
        };
        let id = self.builder.graph.add_node(op);
        self.builder.index_to_id.push(id);
        edge::connect_graph_input(&mut self.builder.graph, graph_input_idx, id, 0);
        (self.builder.index_to_id.len() - 1) as i32
    }

    /// Add a node with input edges from given builder indices.
    pub fn add_node_with_inputs(&mut self, op_kind: i32, op_param: i32, inputs: &[usize]) -> i32 {
        match crate::graph::make_graph_op(op_kind, op_param) {
            Ok(op) => self.builder.add_node_with_inputs(op, inputs) as i32,
            Err(code) => code,
        }
    }

    /// Add an edge between two builder indices.
    pub fn add_edge(&mut self, source: usize, target: usize) {
        self.builder.add_edge(source, target);
    }

    /// Add a named output referencing a builder index.
    pub fn add_output(&mut self, name: &str, node_index: usize) -> i32 {
        if let Some(&id) = self.builder.index_to_id.get(node_index) {
            self.builder.graph.add_output(name, id);
            0
        } else {
            -1
        }
    }

    /// Build and compile the graph. Returns the archive as bytes.
    pub fn compile(&mut self) -> Result<Vec<u8>, JsValue> {
        let graph = std::mem::replace(&mut self.builder.graph, hologram_ir::Graph::new());
        let output =
            hologram_compiler::compile(graph).map_err(|e| JsValue::from_str(&format!("{e}")))?;
        Ok(output.archive)
    }
}

/// Execute a compiled `.holo` archive with the given inputs.
///
/// `archive`: the `.holo` bytes.
/// `input_data`: a flat array of input byte data (one input assumed).
///
/// Returns the output bytes.
#[wasm_bindgen]
pub fn wasm_execute(archive: &[u8], input_data: &[u8]) -> Result<Vec<u8>, JsValue> {
    let mut inputs = hologram_fused_component::GraphInputs::new();
    inputs.set(0, input_data.to_vec());
    let plan = hologram_archive::load_from_bytes(archive)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let tape = hologram_fused_component::mmap::build_tape_from_plan(&plan)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let outputs = hologram_fused_component::mmap::execute_tape(&tape, &plan, &inputs)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    match outputs.get(0) {
        Some((_, data)) => Ok(data.to_vec()),
        None => Err(JsValue::from_str("no outputs")),
    }
}

/// Apply a LUT operation to a byte.
#[wasm_bindgen]
pub fn wasm_lut_apply(lut_op: i32, byte: u8) -> u8 {
    crate::encoding::hologram_lut_apply(lut_op, byte)
}

/// Embed a value using the given encoding.
#[wasm_bindgen]
pub fn wasm_encoding_embed(encoding_id: i32, value: f64) -> u8 {
    crate::encoding::hologram_encoding_embed(encoding_id, value)
}

/// Lift a byte back to a continuous value.
#[wasm_bindgen]
pub fn wasm_encoding_lift(encoding_id: i32, byte: u8) -> f64 {
    crate::encoding::hologram_encoding_lift(encoding_id, byte)
}

/// Full hologram pipeline in a single WASM call: embed → LUT → lift.
///
/// Eliminates JS/WASM boundary overhead for fair comparison with native.
#[wasm_bindgen]
pub fn wasm_hologram_compute(
    embed_encoding: i32,
    lut_op: i32,
    lift_encoding: i32,
    value: f64,
) -> f64 {
    let byte_in = crate::encoding::hologram_encoding_embed(embed_encoding, value);
    let byte_out = crate::encoding::hologram_lut_apply(lut_op, byte_in);
    crate::encoding::hologram_encoding_lift(lift_encoding, byte_out)
}

/// Native f64 math operation in a single WASM call.
///
/// Uses the same LutOp index as `wasm_lut_apply` but computes via
/// actual f64 transcendentals. This ensures both hologram and native
/// paths have the same WASM boundary-crossing overhead.
#[wasm_bindgen]
pub fn wasm_native_compute(lut_op: i32, value: f64) -> f64 {
    native_compute_inner(lut_op, value)
}

fn native_compute_inner(lut_op: i32, value: f64) -> f64 {
    match lut_op {
        0 => 1.0 / (1.0 + (-value).exp()), // Sigmoid
        1 => value.tanh(),                 // Tanh
        2 => value.exp(),                  // Exp
        3 => value.ln(),                   // Log
        4 => value.max(0.0),               // ReLU
        5 => value.sqrt(),                 // Sqrt
        6 => value.abs(),                  // Abs
        7 => {
            // GELU
            let k = (2.0_f64 / std::f64::consts::PI).sqrt();
            0.5 * value * (1.0 + (k * (value + 0.044715 * value.powi(3))).tanh())
        }
        8 => value / (1.0 + (-value).exp()), // SiLU
        9 => value.sin(),                    // Sin
        10 => value.cos(),                   // Cos
        11 => value.tan(),                   // Tan
        12 => value.asin(),                  // Asin
        13 => value.acos(),                  // Acos
        14 => value.atan(),                  // Atan
        15 => value.log2(),                  // Log2
        16 => value.log10(),                 // Log10
        17 => value.exp2(),                  // Exp2
        18 => (10.0_f64).powf(value),        // Exp10
        19 => value * value,                 // Square
        20 => value * value * value,         // Cube
        _ => f64::NAN,
    }
}

/// Prevent the compiler from optimizing away a value.
#[inline(never)]
fn black_box<T>(x: T) -> T {
    // Read-volatile trick: force the value to exist in memory.
    // On WASM this is the most reliable way to prevent DCE without
    // requiring nightly `std::hint::black_box`.
    let ptr = &x as *const T;
    unsafe { core::ptr::read_volatile(ptr) }
}

/// Benchmark a single operation: runs `iters` repetitions of one value
/// entirely inside WASM. Returns `[holo_ns_per_op, native_ns_per_op,
/// holo_result, native_result]`.
#[wasm_bindgen]
pub fn wasm_bench_single(
    embed_encoding: i32,
    lut_op: i32,
    lift_encoding: i32,
    value: f64,
    iters: u32,
) -> Vec<f64> {
    // Warm up
    for _ in 0..1000 {
        black_box(crate::encoding::hologram_encoding_lift(
            lift_encoding,
            crate::encoding::hologram_lut_apply(
                lut_op,
                crate::encoding::hologram_encoding_embed(embed_encoding, black_box(value)),
            ),
        ));
        black_box(native_compute_inner(lut_op, black_box(value)));
    }

    // Benchmark hologram
    let start = perf_now();
    let mut holo_result = 0.0_f64;
    for _ in 0..iters {
        let v = black_box(value);
        let b = crate::encoding::hologram_encoding_embed(embed_encoding, v);
        let b2 = crate::encoding::hologram_lut_apply(lut_op, b);
        holo_result = crate::encoding::hologram_encoding_lift(lift_encoding, b2);
        black_box(holo_result);
    }
    let holo_ms = perf_now() - start;
    let holo_ns = (holo_ms * 1_000_000.0) / iters as f64;

    // Benchmark native
    let start = perf_now();
    let mut native_result = 0.0_f64;
    for _ in 0..iters {
        native_result = native_compute_inner(lut_op, black_box(value));
        black_box(native_result);
    }
    let native_ms = perf_now() - start;
    let native_ns = (native_ms * 1_000_000.0) / iters as f64;

    vec![holo_ns, native_ns, holo_result, native_result]
}

/// Benchmark batch of values: processes entire array inside WASM.
/// Returns `[holo_ns_total, native_ns_total, max_err, mean_err]`.
/// Times are total nanoseconds for one pass of `count` values
/// (averaged over `iters` repetitions).
#[wasm_bindgen]
pub fn wasm_bench_batch(
    embed_encoding: i32,
    lut_op: i32,
    lift_encoding: i32,
    values: &[f64],
    iters: u32,
) -> Vec<f64> {
    let n = values.len();
    let mut holo_results = vec![0.0_f64; n];
    let mut native_results = vec![0.0_f64; n];

    // Warm up both paths
    for v in values {
        black_box(crate::encoding::hologram_encoding_lift(
            lift_encoding,
            crate::encoding::hologram_lut_apply(
                lut_op,
                crate::encoding::hologram_encoding_embed(embed_encoding, *v),
            ),
        ));
        black_box(native_compute_inner(lut_op, *v));
    }

    // Benchmark hologram
    let start = perf_now();
    for _ in 0..iters {
        for (i, v) in values.iter().enumerate() {
            let b = crate::encoding::hologram_encoding_embed(embed_encoding, black_box(*v));
            let b2 = crate::encoding::hologram_lut_apply(lut_op, b);
            holo_results[i] = crate::encoding::hologram_encoding_lift(lift_encoding, b2);
        }
        black_box(&holo_results);
    }
    let holo_ns = ((perf_now() - start) * 1_000_000.0) / iters as f64;

    // Benchmark native
    let start = perf_now();
    for _ in 0..iters {
        for (i, v) in values.iter().enumerate() {
            native_results[i] = native_compute_inner(lut_op, black_box(*v));
        }
        black_box(&native_results);
    }
    let native_ns = ((perf_now() - start) * 1_000_000.0) / iters as f64;

    // Error analysis
    let mut max_err = 0.0_f64;
    let mut sum_err = 0.0_f64;
    for i in 0..n {
        let e = (holo_results[i] - native_results[i]).abs();
        if e > max_err {
            max_err = e;
        }
        sum_err += e;
    }

    vec![holo_ns, native_ns, max_err, sum_err / n as f64]
}

/// Benchmark fused LUT composition vs chained native transcendentals.
/// Processes `count` byte values through op1 then op2.
/// Returns `[fused_lut_ns, chained_native_ns]` — total time for one
/// pass of all values (averaged over `iters`).
#[wasm_bindgen]
pub fn wasm_bench_composition(op1: i32, op2: i32, count: u32, iters: u32) -> Vec<f64> {
    // Warm up
    for b in 0..=255u8 {
        black_box(crate::encoding::hologram_lut_apply(
            op2,
            crate::encoding::hologram_lut_apply(op1, b),
        ));
        let v = b as f64 / 255.0;
        black_box(native_compute_inner(op2, native_compute_inner(op1, v)));
    }

    // Benchmark fused LUT: two table lookups (one byte each)
    let start = perf_now();
    for _ in 0..iters {
        for b in 0..count as u8 {
            black_box(crate::encoding::hologram_lut_apply(
                op2,
                crate::encoding::hologram_lut_apply(op1, black_box(b)),
            ));
        }
    }
    let fused_ns = ((perf_now() - start) * 1_000_000.0) / iters as f64;

    // Benchmark chained native: two f64 transcendentals
    let start = perf_now();
    for _ in 0..iters {
        for b in 0..count as u8 {
            let v = black_box(b) as f64 / 255.0;
            black_box(native_compute_inner(op2, native_compute_inner(op1, v)));
        }
    }
    let native_ns = ((perf_now() - start) * 1_000_000.0) / iters as f64;

    vec![fused_ns, native_ns]
}

// ── Compression functions ──────────────────────────────────────────

/// Compress data using the specified mode (0=Generic, 1=Stratum, 2=Float, 3=Quantized, -1=Auto).
#[cfg(feature = "compression")]
#[wasm_bindgen]
pub fn compress(data: &[u8], mode: i32) -> Vec<u8> {
    use hologram_compression::CompressionMode;

    let m = match mode {
        0 => CompressionMode::Generic,
        1 => CompressionMode::Stratum,
        2 => CompressionMode::Float,
        3 => CompressionMode::Quantized,
        _ => hologram_compression::pipeline::auto_select_mode(data),
    };
    hologram_compression::compress(data, m).data
}

/// Decompress a previously compressed block.
#[cfg(feature = "compression")]
#[wasm_bindgen]
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>, JsValue> {
    hologram_compression::decompress(compressed)
        .ok_or_else(|| JsValue::from_str("decompression failed"))
}

/// Get compression statistics for data with a given mode.
/// Returns: [original_size, compressed_size, ratio, stratum_histogram[0..9]].
#[cfg(feature = "compression")]
#[wasm_bindgen]
pub fn compression_stats(data: &[u8], mode: i32) -> Vec<f64> {
    use hologram_compression::CompressionMode;

    let m = match mode {
        0 => CompressionMode::Generic,
        1 => CompressionMode::Stratum,
        2 => CompressionMode::Float,
        3 => CompressionMode::Quantized,
        _ => hologram_compression::pipeline::auto_select_mode(data),
    };
    let block = hologram_compression::compress(data, m);
    let original = data.len() as f64;
    let comp = block.data.len() as f64;
    let ratio = if comp > 0.0 { original / comp } else { 0.0 };

    let mut result = vec![original, comp, ratio];
    let hist = stratum_histogram(data);
    for v in hist {
        result.push(v as f64);
    }
    result
}

/// Compute stratum (popcount) histogram for a byte slice.
/// Returns 9 values: count of bytes in each stratum 0..=8.
#[cfg(feature = "compression")]
#[wasm_bindgen]
pub fn stratum_histogram(data: &[u8]) -> Vec<u32> {
    use hologram_core::lut::q0::stratum_q0;
    let mut hist = vec![0u32; 9];
    for &b in data {
        hist[stratum_q0(b) as usize] += 1;
    }
    hist
}

/// Ring algebra operations on two bytes.
/// Returns: [add, sub, mul, neg_a, bnot_a, stratum_a, stratum_b].
#[wasm_bindgen]
pub fn ring_algebra(a: u8, b: u8) -> Vec<u8> {
    use hologram_core::lut::q0::stratum_q0;
    use hologram_core::ring::byte_ring::ByteRing;
    vec![
        ByteRing::add(a, b),
        ByteRing::sub(a, b),
        ByteRing::mul(a, b),
        ByteRing::neg(a),
        !a, // bnot
        stratum_q0(a),
        stratum_q0(b),
    ]
}

/// Float byte-plane transposition for f32 data.
/// Input: raw f32 bytes (len must be multiple of 4).
/// Returns: 4 concatenated byte planes (each of len/4 bytes).
#[cfg(feature = "compression")]
#[wasm_bindgen]
pub fn float_plane_transpose(f32_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    hologram_compression::float_plane::transpose_f32(f32_bytes)
        .ok_or_else(|| JsValue::from_str("input length must be multiple of 4"))
}

/// High-resolution timer via `performance.now()` (DOMHighResTimeStamp).
/// Returns milliseconds with microsecond resolution.
fn perf_now() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|p| js_sys::Reflect::get(&p, &"now".into()).ok())
            .and_then(|f| {
                let f = js_sys::Function::from(f);
                let perf = js_sys::Reflect::get(&js_sys::global(), &"performance".into()).unwrap();
                f.call0(&perf).ok()
            })
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| js_sys::Date::now())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}
