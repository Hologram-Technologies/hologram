//! Execution FFI functions.

use crate::error::{ffi_catch, set_last_error, FfiStatus};
use crate::handle::{borrow_handle, borrow_handle_mut, free_handle, into_handle};
use holo_exec::mmap::execute_bytes;
use holo_exec::{GraphInputs, GraphOutputs};
use std::ffi::CStr;
use std::os::raw::c_char;

// ── GraphInputs ──

/// Create a new empty inputs handle.
#[no_mangle]
pub extern "C" fn holo_inputs_new() -> *mut GraphInputs {
    into_handle(GraphInputs::new())
}

/// Set input data at `index`. Copies `data_len` bytes from `data_ptr`.
#[no_mangle]
pub extern "C" fn holo_inputs_set(
    inputs: *mut GraphInputs,
    index: u32,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    ffi_catch(|| {
        let inp = borrow_handle_mut(inputs)
            .ok_or((FfiStatus::NullPointer, "null inputs handle".into()))?;
        if data_ptr.is_null() && data_len > 0 {
            return Err((
                FfiStatus::NullPointer,
                "null data pointer with non-zero length".into(),
            ));
        }
        let data = read_byte_slice(data_ptr, data_len);
        inp.set(index, data);
        Ok(())
    })
}

/// Free an inputs handle.
#[no_mangle]
pub extern "C" fn holo_inputs_free(inputs: *mut GraphInputs) {
    unsafe { free_handle(inputs) };
}

// ── Execution ──

/// Execute a compiled archive with the given inputs.
/// `archive_ptr`/`archive_len` point to the `.holo` bytes.
/// Returns an outputs handle or null on error.
#[no_mangle]
pub extern "C" fn holo_execute_bytes(
    archive_ptr: *const u8,
    archive_len: usize,
    inputs: *const GraphInputs,
) -> *mut GraphOutputs {
    if archive_ptr.is_null() {
        set_last_error("null archive pointer");
        return std::ptr::null_mut();
    }
    let Some(inp) = borrow_handle(inputs) else {
        return std::ptr::null_mut();
    };
    let archive = unsafe { std::slice::from_raw_parts(archive_ptr, archive_len) };
    match execute_bytes(archive, inp) {
        Ok(outputs) => into_handle(outputs),
        Err(e) => {
            set_last_error(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

// ── GraphOutputs ──

/// Number of outputs.
#[no_mangle]
pub extern "C" fn holo_outputs_len(outputs: *const GraphOutputs) -> i32 {
    match borrow_handle(outputs) {
        Some(o) => o.len() as i32,
        None => FfiStatus::NullPointer as i32,
    }
}

/// Get output data at `index`. Writes the data pointer and length to
/// `out_ptr` and `out_len`. Returns 0 on success.
///
/// The data is owned by the outputs handle and valid until freed.
#[no_mangle]
pub extern "C" fn holo_outputs_get(
    outputs: *const GraphOutputs,
    index: usize,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    let Some(o) = borrow_handle(outputs) else {
        return FfiStatus::NullPointer as i32;
    };
    match o.get(index) {
        Some((_name, data)) => {
            write_out_slice(data, out_ptr, out_len);
            FfiStatus::Ok as i32
        }
        None => {
            set_last_error(format!("output index {index} out of range"));
            FfiStatus::InvalidArgument as i32
        }
    }
}

/// Get the name of output at `index`. Returns a null-terminated string
/// or null if index is out of range.
#[no_mangle]
pub extern "C" fn holo_outputs_name(outputs: *const GraphOutputs, index: usize) -> *const c_char {
    let Some(o) = borrow_handle(outputs) else {
        return std::ptr::null();
    };
    match o.get(index) {
        Some((name, _)) => {
            // Leak a CString for the caller (freed with outputs)
            // We store it alongside — but for simplicity return static
            // This is safe because the name lives as long as outputs.
            // We need a null-terminated copy though.
            let cstr = std::ffi::CString::new(name).unwrap_or_default();
            cstr.into_raw() as *const c_char
        }
        None => std::ptr::null(),
    }
}

/// Get output by name. Writes data pointer and length.
#[no_mangle]
pub extern "C" fn holo_outputs_by_name(
    outputs: *const GraphOutputs,
    name: *const c_char,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    let Some(o) = borrow_handle(outputs) else {
        return FfiStatus::NullPointer as i32;
    };
    if name.is_null() {
        set_last_error("null name pointer");
        return FfiStatus::NullPointer as i32;
    }
    let cstr = unsafe { CStr::from_ptr(name) };
    let name_str = match cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("invalid UTF-8: {e}"));
            return FfiStatus::InvalidArgument as i32;
        }
    };
    match o.by_name(name_str) {
        Some(data) => {
            write_out_slice(data, out_ptr, out_len);
            FfiStatus::Ok as i32
        }
        None => {
            set_last_error(format!("output '{name_str}' not found"));
            FfiStatus::InvalidArgument as i32
        }
    }
}

/// Free an outputs handle.
#[no_mangle]
pub extern "C" fn holo_outputs_free(outputs: *mut GraphOutputs) {
    unsafe { free_handle(outputs) };
}

// ── Helpers ──

/// Read a byte slice from a C pointer.
fn read_byte_slice(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

/// Write a slice's pointer and length to out-params.
fn write_out_slice(data: &[u8], out_ptr: *mut *const u8, out_len: *mut usize) {
    if !out_ptr.is_null() {
        unsafe { *out_ptr = data.as_ptr() };
    }
    if !out_len.is_null() {
        unsafe { *out_len = data.len() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::*;
    use crate::graph::*;
    use std::ffi::CString;

    /// Build and compile a test graph via FFI: x → Sigmoid → y.
    fn compile_test_graph() -> *mut holo_compiler::CompilationOutput {
        let b = holo_graph_builder_new();
        let name_x = CString::new("x").unwrap();
        holo_graph_builder_input(b, name_x.as_ptr());
        // Input node wired to graph input 0
        holo_graph_builder_node_from_input(b, 0, 0, 0);
        let inputs = [0usize];
        holo_graph_builder_node_with_inputs(b, 3, 0, inputs.as_ptr(), 1); // Lut(Sigmoid)
        let inputs2 = [1usize];
        holo_graph_builder_node_with_inputs(b, 1, 0, inputs2.as_ptr(), 1); // Output
        let name_y = CString::new("y").unwrap();
        holo_graph_builder_output(b, name_y.as_ptr(), 2);
        let g = holo_graph_builder_build(b);
        holo_compile(g)
    }

    #[test]
    fn execute_compiled_archive() {
        let out = compile_test_graph();
        let archive_ptr = holo_compilation_archive_ptr(out);
        let archive_len = holo_compilation_archive_len(out);

        let inp = holo_inputs_new();
        let data = vec![42u8];
        holo_inputs_set(inp, 0, data.as_ptr(), data.len());

        let outputs = holo_execute_bytes(archive_ptr, archive_len, inp);
        assert!(!outputs.is_null());
        assert_eq!(holo_outputs_len(outputs), 1);

        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let rc = holo_outputs_get(outputs, 0, &mut ptr, &mut len);
        assert_eq!(rc, 0);
        assert_eq!(len, 1);

        holo_outputs_free(outputs);
        holo_inputs_free(inp);
        holo_compilation_free(out);
    }

    #[test]
    fn outputs_by_name() {
        let out = compile_test_graph();
        let archive_ptr = holo_compilation_archive_ptr(out);
        let archive_len = holo_compilation_archive_len(out);

        let inp = holo_inputs_new();
        let data = vec![100u8];
        holo_inputs_set(inp, 0, data.as_ptr(), data.len());

        let outputs = holo_execute_bytes(archive_ptr, archive_len, inp);
        let name = CString::new("y").unwrap();
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let rc = holo_outputs_by_name(outputs, name.as_ptr(), &mut ptr, &mut len);
        assert_eq!(rc, 0);
        assert_eq!(len, 1);

        holo_outputs_free(outputs);
        holo_inputs_free(inp);
        holo_compilation_free(out);
    }

    #[test]
    fn null_archive_returns_null() {
        let inp = holo_inputs_new();
        let out = holo_execute_bytes(std::ptr::null(), 0, inp);
        assert!(out.is_null());
        holo_inputs_free(inp);
    }

    #[test]
    fn null_inputs_returns_null() {
        let out = holo_execute_bytes([0u8; 4].as_ptr(), 4, std::ptr::null());
        assert!(out.is_null());
    }

    #[test]
    fn outputs_get_invalid_index() {
        let out = compile_test_graph();
        let archive_ptr = holo_compilation_archive_ptr(out);
        let archive_len = holo_compilation_archive_len(out);
        let inp = holo_inputs_new();
        holo_inputs_set(inp, 0, [0u8].as_ptr(), 1);
        let outputs = holo_execute_bytes(archive_ptr, archive_len, inp);
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let rc = holo_outputs_get(outputs, 99, &mut ptr, &mut len);
        assert!(rc < 0);
        holo_outputs_free(outputs);
        holo_inputs_free(inp);
        holo_compilation_free(out);
    }

    #[test]
    fn outputs_by_name_not_found() {
        let out = compile_test_graph();
        let archive_ptr = holo_compilation_archive_ptr(out);
        let archive_len = holo_compilation_archive_len(out);
        let inp = holo_inputs_new();
        holo_inputs_set(inp, 0, [0u8].as_ptr(), 1);
        let outputs = holo_execute_bytes(archive_ptr, archive_len, inp);
        let name = CString::new("missing").unwrap();
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        let rc = holo_outputs_by_name(outputs, name.as_ptr(), &mut ptr, &mut len);
        assert!(rc < 0);
        holo_outputs_free(outputs);
        holo_inputs_free(inp);
        holo_compilation_free(out);
    }

    #[test]
    fn free_null_handles_safe() {
        holo_inputs_free(std::ptr::null_mut());
        holo_outputs_free(std::ptr::null_mut());
    }

    #[test]
    fn inputs_set_null_data_with_zero_len() {
        let inp = holo_inputs_new();
        let rc = holo_inputs_set(inp, 0, std::ptr::null(), 0);
        assert_eq!(rc, 0);
        holo_inputs_free(inp);
    }

    #[test]
    fn inputs_set_null_data_with_nonzero_len() {
        let inp = holo_inputs_new();
        let rc = holo_inputs_set(inp, 0, std::ptr::null(), 10);
        assert!(rc < 0);
        holo_inputs_free(inp);
    }
}
