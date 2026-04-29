//! Compilation FFI functions.

use crate::error::set_last_error;
use crate::handle::{borrow_handle, free_handle, into_handle};
use hologram_compiler::{CompilationOutput, CompilerBuilder};
use hologram_graph::Graph;

/// Compile a graph into a `.holo` archive. Consumes the graph handle.
/// Returns a compilation output handle or null on error.
#[no_mangle]
pub extern "C" fn hologram_compile(graph: *mut Graph) -> *mut CompilationOutput {
    if graph.is_null() {
        set_last_error("null graph handle");
        return std::ptr::null_mut();
    }
    let g = unsafe { Box::from_raw(graph) };
    match CompilerBuilder::new(*g).build() {
        Ok(output) => into_handle(output),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Compile a graph with fusion disabled. Consumes the graph handle.
#[no_mangle]
pub extern "C" fn hologram_compile_no_fuse(graph: *mut Graph) -> *mut CompilationOutput {
    if graph.is_null() {
        set_last_error("null graph handle");
        return std::ptr::null_mut();
    }
    let g = unsafe { Box::from_raw(graph) };
    match CompilerBuilder::new(*g).fuse(false).build() {
        Ok(output) => into_handle(output),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Get pointer to the compiled archive bytes.
#[no_mangle]
pub extern "C" fn hologram_compilation_archive_ptr(output: *const CompilationOutput) -> *const u8 {
    match borrow_handle(output) {
        Some(o) => o.archive.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Get length of the compiled archive bytes.
#[no_mangle]
pub extern "C" fn hologram_compilation_archive_len(output: *const CompilationOutput) -> usize {
    match borrow_handle(output) {
        Some(o) => o.archive.len(),
        None => 0,
    }
}

/// Get total node count from compilation stats.
#[no_mangle]
pub extern "C" fn hologram_compilation_stats_nodes(output: *const CompilationOutput) -> u32 {
    match borrow_handle(output) {
        Some(o) => o.stats.total_nodes as u32,
        None => 0,
    }
}

/// Get schedule level count from compilation stats.
#[no_mangle]
pub extern "C" fn hologram_compilation_stats_levels(output: *const CompilationOutput) -> u32 {
    match borrow_handle(output) {
        Some(o) => o.stats.schedule_levels as u32,
        None => 0,
    }
}

/// Get workspace slot count from compilation stats.
#[no_mangle]
pub extern "C" fn hologram_compilation_stats_workspace_slots(
    output: *const CompilationOutput,
) -> u32 {
    match borrow_handle(output) {
        Some(o) => o.stats.workspace_slots as u32,
        None => 0,
    }
}

/// Free a compilation output handle.
#[no_mangle]
pub extern "C" fn hologram_compilation_free(output: *mut CompilationOutput) {
    unsafe { free_handle(output) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::*;
    use std::ffi::CString;

    /// Build a simple graph via FFI: Input → Sigmoid → Output.
    fn build_test_graph() -> *mut Graph {
        let b = hologram_graph_builder_new();
        let name_x = CString::new("x").unwrap();
        hologram_graph_builder_input(b, name_x.as_ptr());
        hologram_graph_builder_node_from_input(b, 0, 0, 0); // Input wired to graph input 0
        let inputs = [0usize];
        hologram_graph_builder_node_with_inputs(b, 3, 0, inputs.as_ptr(), 1); // Lut(Sigmoid)
                                                                              // ADR-053: v3 archives require shape coverage. Sigmoid (idx 1) gets [1].
        let shape = [1usize];
        hologram_graph_builder_set_node_shape(b, 1, shape.as_ptr(), shape.len());
        let inputs2 = [1usize];
        hologram_graph_builder_node_with_inputs(b, 1, 0, inputs2.as_ptr(), 1); // Output
        let name_y = CString::new("y").unwrap();
        hologram_graph_builder_output(b, name_y.as_ptr(), 2);
        hologram_graph_builder_build(b)
    }

    #[test]
    fn compile_simple_graph() {
        let g = build_test_graph();
        let out = hologram_compile(g);
        assert!(!out.is_null());
        assert!(hologram_compilation_archive_len(out) > 0);
        assert!(!hologram_compilation_archive_ptr(out).is_null());
        assert_eq!(hologram_compilation_stats_nodes(out), 3);
        assert!(hologram_compilation_stats_levels(out) > 0);
        hologram_compilation_free(out);
    }

    #[test]
    fn compile_no_fuse() {
        let g = build_test_graph();
        let out = hologram_compile_no_fuse(g);
        assert!(!out.is_null());
        assert!(hologram_compilation_archive_len(out) > 0);
        hologram_compilation_free(out);
    }

    #[test]
    fn compile_null_graph() {
        let out = hologram_compile(std::ptr::null_mut());
        assert!(out.is_null());
    }

    #[test]
    fn stats_on_null_return_zero() {
        let null: *const CompilationOutput = std::ptr::null();
        assert_eq!(hologram_compilation_stats_nodes(null), 0);
        assert_eq!(hologram_compilation_stats_levels(null), 0);
        assert_eq!(hologram_compilation_stats_workspace_slots(null), 0);
        assert_eq!(hologram_compilation_archive_len(null), 0);
        assert!(hologram_compilation_archive_ptr(null).is_null());
    }

    #[test]
    fn free_null_is_safe() {
        hologram_compilation_free(std::ptr::null_mut());
    }

    #[test]
    fn workspace_slots_present() {
        let g = build_test_graph();
        let out = hologram_compile(g);
        // Workspace slots should be > 0 for a graph with nodes
        assert!(hologram_compilation_stats_workspace_slots(out) > 0);
        hologram_compilation_free(out);
    }
}
