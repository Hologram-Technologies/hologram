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
        use holo_graph::graph::edge;
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
        let graph = std::mem::replace(&mut self.builder.graph, holo_graph::Graph::new());
        let output = holo_compiler::CompilerBuilder::new(graph)
            .build()
            .map_err(|e| JsValue::from_str(&format!("{e}")))?;
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
    let mut inputs = holo_exec::GraphInputs::new();
    inputs.set(0, input_data.to_vec());
    let outputs = holo_exec::mmap::execute_bytes(archive, &inputs)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    match outputs.get(0) {
        Some((_, data)) => Ok(data.to_vec()),
        None => Err(JsValue::from_str("no outputs")),
    }
}

/// Apply a LUT operation to a byte.
#[wasm_bindgen]
pub fn wasm_lut_apply(lut_op: i32, byte: u8) -> u8 {
    crate::encoding::holo_lut_apply(lut_op, byte)
}

/// Embed a value using the given encoding.
#[wasm_bindgen]
pub fn wasm_encoding_embed(encoding_id: i32, value: f64) -> u8 {
    crate::encoding::holo_encoding_embed(encoding_id, value)
}

/// Lift a byte back to a continuous value.
#[wasm_bindgen]
pub fn wasm_encoding_lift(encoding_id: i32, byte: u8) -> f64 {
    crate::encoding::holo_encoding_lift(encoding_id, byte)
}
