//! Graph construction FFI functions.

use crate::error::{set_last_error, FfiStatus};
use crate::handle::{borrow_handle, borrow_handle_mut, free_handle, into_handle};
use hologram_graph::graph::edge;
use hologram_graph::graph::node::NodeId;
use hologram_graph::{Graph, GraphOp};
use std::ffi::CStr;
use std::os::raw::c_char;

/// FFI-friendly graph builder (mutable, not consuming).
pub struct FfiGraphBuilder {
    pub(crate) graph: Graph,
    pub(crate) index_to_id: Vec<NodeId>,
}

impl FfiGraphBuilder {
    pub(crate) fn new_internal() -> Self {
        Self {
            graph: Graph::new(),
            index_to_id: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, op: GraphOp) -> usize {
        let id = self.graph.add_node(op);
        self.index_to_id.push(id);
        self.index_to_id.len() - 1
    }

    pub(crate) fn add_node_with_inputs(&mut self, op: GraphOp, inputs: &[usize]) -> usize {
        let id = self.graph.add_node(op);
        self.index_to_id.push(id);
        wire_inputs(&mut self.graph, &self.index_to_id, id, inputs);
        self.index_to_id.len() - 1
    }

    pub(crate) fn add_edge(&mut self, source: usize, target: usize) {
        if let (Some(&s), Some(&t)) = (self.index_to_id.get(source), self.index_to_id.get(target)) {
            self.graph.add_edge(s, t);
        }
    }
}

/// Wire input edges from builder indices to a target node.
fn wire_inputs(graph: &mut Graph, id_map: &[NodeId], target: NodeId, inputs: &[usize]) {
    for (slot, &src_idx) in inputs.iter().enumerate() {
        if let Some(&src_id) = id_map.get(src_idx) {
            edge::connect(graph, src_id, target, slot);
        }
    }
}

// ── FFI functions ──

/// Create a new graph builder.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_new() -> *mut FfiGraphBuilder {
    into_handle(FfiGraphBuilder::new_internal())
}

/// Add a named input. Returns input index (>= 0) or negative error.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_input(
    builder: *mut FfiGraphBuilder,
    name: *const c_char,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    let name_str = match parse_c_str(name) {
        Ok(s) => s,
        Err(code) => return code,
    };
    b.graph.add_input(name_str) as i32
}

/// Add a node. Returns node index (>= 0) or negative error.
///
/// `op_kind`: 0=Input, 1=Output, 2=Prim, 3=Lut.
/// `op_param`: discriminant for Prim/Lut (ignored for Input/Output).
#[no_mangle]
pub extern "C" fn hologram_graph_builder_node(
    builder: *mut FfiGraphBuilder,
    op_kind: i32,
    op_param: i32,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    match make_graph_op(op_kind, op_param) {
        Ok(op) => b.add_node(op) as i32,
        Err(code) => code,
    }
}

/// Add a node wired to a graph-level input. Returns node index or error.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_node_from_input(
    builder: *mut FfiGraphBuilder,
    op_kind: i32,
    op_param: i32,
    graph_input_idx: u32,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    let op = match make_graph_op(op_kind, op_param) {
        Ok(op) => op,
        Err(code) => return code,
    };
    let id = b.graph.add_node(op);
    b.index_to_id.push(id);
    edge::connect_graph_input(&mut b.graph, graph_input_idx, id, 0);
    (b.index_to_id.len() - 1) as i32
}

/// Add a node with input edges. Returns node index or negative error.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_node_with_inputs(
    builder: *mut FfiGraphBuilder,
    op_kind: i32,
    op_param: i32,
    inputs_ptr: *const usize,
    inputs_len: usize,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    let op = match make_graph_op(op_kind, op_param) {
        Ok(op) => op,
        Err(code) => return code,
    };
    let inputs = read_usize_slice(inputs_ptr, inputs_len);
    b.add_node_with_inputs(op, &inputs) as i32
}

/// Add an edge between two builder indices.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_edge(
    builder: *mut FfiGraphBuilder,
    source: usize,
    target: usize,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    b.add_edge(source, target);
    FfiStatus::Ok as i32
}

/// Set the N-D output shape for a node by builder index.
///
/// Per ADR-053, v3 archives require shape coverage for every
/// dispatch-producing node. Callers building graphs through this
/// FFI must populate shapes before compilation.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_set_node_shape(
    builder: *mut FfiGraphBuilder,
    node_index: usize,
    shape_ptr: *const usize,
    shape_len: usize,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    let shape = read_usize_slice(shape_ptr, shape_len).to_vec();
    if let Some(&id) = b.index_to_id.get(node_index) {
        b.graph.set_node_shape(id, shape);
        FfiStatus::Ok as i32
    } else {
        set_last_error(format!("node index {node_index} out of range"));
        FfiStatus::InvalidArgument as i32
    }
}

/// Add a named output referencing a builder index.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_output(
    builder: *mut FfiGraphBuilder,
    name: *const c_char,
    node_index: usize,
) -> i32 {
    let Some(b) = borrow_handle_mut(builder) else {
        return FfiStatus::NullPointer as i32;
    };
    let name_str = match parse_c_str(name) {
        Ok(s) => s,
        Err(code) => return code,
    };
    if let Some(&id) = b.index_to_id.get(node_index) {
        b.graph.add_output(name_str, id);
        FfiStatus::Ok as i32
    } else {
        set_last_error(format!("node index {node_index} out of range"));
        FfiStatus::InvalidArgument as i32
    }
}

/// Build the graph, consuming the builder. Returns graph handle or null.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_build(builder: *mut FfiGraphBuilder) -> *mut Graph {
    if builder.is_null() {
        set_last_error("null builder handle");
        return std::ptr::null_mut();
    }
    let b = unsafe { Box::from_raw(builder) };
    into_handle(b.graph)
}

/// Return the number of nodes in a graph.
#[no_mangle]
pub extern "C" fn hologram_graph_node_count(graph: *const Graph) -> i32 {
    match borrow_handle(graph) {
        Some(g) => g.node_count() as i32,
        None => FfiStatus::NullPointer as i32,
    }
}

/// Free a graph builder handle.
#[no_mangle]
pub extern "C" fn hologram_graph_builder_free(builder: *mut FfiGraphBuilder) {
    unsafe { free_handle(builder) };
}

/// Free a graph handle.
#[no_mangle]
pub extern "C" fn hologram_graph_free(graph: *mut Graph) {
    unsafe { free_handle(graph) };
}

// ── Helpers ──

/// Parse a C string to `&str`.
fn parse_c_str<'a>(ptr: *const c_char) -> Result<&'a str, i32> {
    if ptr.is_null() {
        set_last_error("null string pointer");
        return Err(FfiStatus::NullPointer as i32);
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().map_err(|e| {
        set_last_error(format!("invalid UTF-8: {e}"));
        FfiStatus::InvalidArgument as i32
    })
}

/// Read a C array of usize into a Vec.
fn read_usize_slice(ptr: *const usize, len: usize) -> Vec<usize> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

/// Map (op_kind, op_param) to a GraphOp.
pub(crate) fn make_graph_op(kind: i32, param: i32) -> Result<GraphOp, i32> {
    match kind {
        0 => Ok(GraphOp::Input),
        1 => Ok(GraphOp::Output),
        2 => prim_from_param(param).map(GraphOp::Prim),
        3 => lut_from_param(param).map(GraphOp::Lut),
        _ => {
            set_last_error(format!("unknown op kind: {kind}"));
            Err(FfiStatus::InvalidArgument as i32)
        }
    }
}

/// Map a discriminant to PrimOp.
fn prim_from_param(p: i32) -> Result<hologram_core::op::PrimOp, i32> {
    use hologram_core::op::PrimOp;
    let ops = [
        PrimOp::Neg,
        PrimOp::Bnot,
        PrimOp::Succ,
        PrimOp::Pred,
        PrimOp::Add,
        PrimOp::Sub,
        PrimOp::Mul,
        PrimOp::Xor,
        PrimOp::And,
        PrimOp::Or,
    ];
    ops.get(p as usize).copied().ok_or_else(|| {
        set_last_error(format!("unknown PrimOp: {p}"));
        FfiStatus::InvalidArgument as i32
    })
}

/// Map a discriminant to LutOp.
fn lut_from_param(p: i32) -> Result<hologram_core::op::LutOp, i32> {
    hologram_core::op::LutOp::ALL
        .get(p as usize)
        .copied()
        .ok_or_else(|| {
            set_last_error(format!("unknown LutOp: {p}"));
            FfiStatus::InvalidArgument as i32
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn build_empty_graph() {
        let b = hologram_graph_builder_new();
        let g = hologram_graph_builder_build(b);
        assert_eq!(hologram_graph_node_count(g), 0);
        hologram_graph_free(g);
    }

    #[test]
    fn build_linear_chain() {
        let b = hologram_graph_builder_new();
        let n0 = hologram_graph_builder_node(b, 0, 0);
        assert_eq!(n0, 0);
        let inputs = [0usize];
        let n1 = hologram_graph_builder_node_with_inputs(b, 3, 0, inputs.as_ptr(), 1);
        assert_eq!(n1, 1);
        let inputs2 = [1usize];
        let n2 = hologram_graph_builder_node_with_inputs(b, 1, 0, inputs2.as_ptr(), 1);
        assert_eq!(n2, 2);
        let g = hologram_graph_builder_build(b);
        assert_eq!(hologram_graph_node_count(g), 3);
        hologram_graph_free(g);
    }

    #[test]
    fn named_input_output() {
        let b = hologram_graph_builder_new();
        let name_x = CString::new("x").unwrap();
        let idx = hologram_graph_builder_input(b, name_x.as_ptr());
        assert_eq!(idx, 0);
        hologram_graph_builder_node(b, 0, 0);
        let name_y = CString::new("y").unwrap();
        let rc = hologram_graph_builder_output(b, name_y.as_ptr(), 0);
        assert_eq!(rc, 0);
        let g = hologram_graph_builder_build(b);
        let graph = borrow_handle(g as *const Graph).unwrap();
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
        hologram_graph_free(g);
    }

    #[test]
    fn null_builder_returns_error() {
        let rc = hologram_graph_builder_node(std::ptr::null_mut(), 0, 0);
        assert!(rc < 0);
    }

    #[test]
    fn invalid_op_kind() {
        let b = hologram_graph_builder_new();
        let rc = hologram_graph_builder_node(b, 99, 0);
        assert!(rc < 0);
        hologram_graph_builder_free(b);
    }

    #[test]
    fn invalid_prim_param() {
        let b = hologram_graph_builder_new();
        let rc = hologram_graph_builder_node(b, 2, 99);
        assert!(rc < 0);
        hologram_graph_builder_free(b);
    }

    #[test]
    fn invalid_lut_param() {
        let b = hologram_graph_builder_new();
        let rc = hologram_graph_builder_node(b, 3, 99);
        assert!(rc < 0);
        hologram_graph_builder_free(b);
    }

    #[test]
    fn edge_between_nodes() {
        let b = hologram_graph_builder_new();
        hologram_graph_builder_node(b, 0, 0);
        hologram_graph_builder_node(b, 1, 0);
        let rc = hologram_graph_builder_edge(b, 0, 1);
        assert_eq!(rc, 0);
        let g = hologram_graph_builder_build(b);
        let graph = borrow_handle(g as *const Graph).unwrap();
        assert_eq!(graph.edges().len(), 1);
        hologram_graph_free(g);
    }

    #[test]
    fn output_invalid_index() {
        let b = hologram_graph_builder_new();
        let name = CString::new("y").unwrap();
        let rc = hologram_graph_builder_output(b, name.as_ptr(), 99);
        assert!(rc < 0);
        hologram_graph_builder_free(b);
    }

    #[test]
    fn null_string_returns_error() {
        let b = hologram_graph_builder_new();
        let rc = hologram_graph_builder_input(b, std::ptr::null());
        assert!(rc < 0);
        hologram_graph_builder_free(b);
    }

    #[test]
    fn build_null_builder_returns_null() {
        let g = hologram_graph_builder_build(std::ptr::null_mut());
        assert!(g.is_null());
    }

    #[test]
    fn free_null_is_safe() {
        hologram_graph_builder_free(std::ptr::null_mut());
        hologram_graph_free(std::ptr::null_mut());
    }
}
