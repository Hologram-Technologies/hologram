//! Hologram C ABI + WASM bindings.
//!
//! Exposes the full compile / load / introspect / execute / close surface
//! against the CPU backend. WASM bindings can be enabled via the `wasm`
//! feature.

#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_int, c_uchar};
use std::sync::Mutex;
use std::sync::OnceLock;

use hologram_backend::CpuBackend;
use hologram_compiler::{Compiler, BackendKind};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::Graph;
use prism::vocabulary::WittLevel;

type Session = InferenceSession<CpuBackend<BufferArena>>;

fn sessions() -> &'static Mutex<Vec<Option<Session>>> {
    static SESSIONS: OnceLock<Mutex<Vec<Option<Session>>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Compile an empty graph for the CPU backend at WittLevel::W32.
/// Writes the resulting `.holo` bytes into the caller's buffer.
/// Returns the number of bytes written, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_compile_empty(
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    if out.is_null() { return -1; }
    let graph = Graph::new();
    let compiled = match Compiler::new(graph, BackendKind::Cpu, WittLevel::W32).compile() {
        Ok(o) => o,
        Err(_) => return -1,
    };
    let n = compiled.archive.len().min(out_capacity);
    std::slice::from_raw_parts_mut(out, n).copy_from_slice(&compiled.archive[..n]);
    n as c_int
}

/// Compile a textual hologram-source program for the CPU backend.
/// `source_ptr`/`source_len` carry the input bytes (UTF-8). Returns
/// the archive length on success or -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn hologram_compile_source(
    source_ptr: *const c_uchar,
    source_len: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    if source_ptr.is_null() || out.is_null() { return -1; }
    let bytes = std::slice::from_raw_parts(source_ptr, source_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let compiled = match hologram_compiler::compile_from_source(s, WittLevel::W32, BackendKind::Cpu) {
        Ok(o) => o,
        Err(_) => return -1,
    };
    let n = compiled.archive.len().min(out_capacity);
    std::slice::from_raw_parts_mut(out, n).copy_from_slice(&compiled.archive[..n]);
    n as c_int
}

/// Load an `.holo` archive and return a session handle, or -1 on error.
/// The handle is an opaque integer index into a global session table.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_load(
    archive_ptr: *const c_uchar,
    archive_len: usize,
) -> c_int {
    if archive_ptr.is_null() { return -1; }
    let bytes = std::slice::from_raw_parts(archive_ptr, archive_len);
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let sess = match InferenceSession::load(bytes, backend) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mut tab = match sessions().lock() { Ok(t) => t, Err(_) => return -1 };
    tab.push(Some(sess));
    (tab.len() - 1) as c_int
}

/// Number of input ports declared by a session, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_input_count(handle: c_int) -> c_int {
    with_session(handle, |s| s.input_count() as c_int)
}

/// Number of output ports declared by a session, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_output_count(handle: c_int) -> c_int {
    with_session(handle, |s| s.output_count() as c_int)
}

/// Number of kernel calls in the loaded archive, or -1 on error.
/// Useful for cost estimation / progress reporting.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_kernel_count(handle: c_int) -> c_int {
    with_session(handle, |s| s.kernel_count() as c_int)
}

/// Execute a session against caller-provided input buffers, writing each
/// declared output port's bytes into the matching `out_ptrs[i]` (capacity
/// `out_caps[i]`). Returns 0 on success or -1 on error.
///
/// `in_ptrs` and `in_lens` carry pointers to and byte lengths of each
/// input port's bytes (length = `in_count`, which must equal the session's
/// input port count). `out_ptrs` and `out_caps` carry per-output capacity
/// pointers. Output port byte lengths can be queried via
/// `hologram_session_output_byte_len`.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_execute(
    handle: c_int,
    in_ptrs: *const *const c_uchar,
    in_lens: *const usize,
    in_count: usize,
    out_ptrs: *const *mut c_uchar,
    out_caps: *const usize,
    out_count: usize,
) -> c_int {
    if handle < 0 { return -1; }
    let mut tab = match sessions().lock() { Ok(t) => t, Err(_) => return -1 };
    let sess = match tab.get_mut(handle as usize).and_then(|s| s.as_mut()) {
        Some(s) => s, None => return -1,
    };

    if in_count != sess.input_count() { return -1; }
    if out_count != sess.output_count() { return -1; }

    let mut input_storage: Vec<&[c_uchar]> = Vec::with_capacity(in_count);
    if in_count > 0 {
        if in_ptrs.is_null() || in_lens.is_null() { return -1; }
        let ptrs = std::slice::from_raw_parts(in_ptrs, in_count);
        let lens = std::slice::from_raw_parts(in_lens, in_count);
        for i in 0..in_count {
            if ptrs[i].is_null() && lens[i] != 0 { return -1; }
            let s = if lens[i] == 0 { &[][..] } else { std::slice::from_raw_parts(ptrs[i], lens[i]) };
            input_storage.push(s);
        }
    }
    let inputs: Vec<InputBuffer> = input_storage.iter()
        .map(|b| InputBuffer { bytes: b }).collect();

    let outputs = match sess.execute(&inputs) {
        Ok(o) => o,
        Err(_) => return -1,
    };

    if out_count > 0 {
        if out_ptrs.is_null() || out_caps.is_null() { return -1; }
        let ptrs = std::slice::from_raw_parts(out_ptrs, out_count);
        let caps = std::slice::from_raw_parts(out_caps, out_count);
        for (i, out) in outputs.iter().enumerate() {
            if i >= out_count { break; }
            if ptrs[i].is_null() { continue; }
            let n = out.bytes.len().min(caps[i]);
            std::slice::from_raw_parts_mut(ptrs[i], n).copy_from_slice(&out.bytes[..n]);
        }
    }
    0
}

/// Drop a previously-loaded session.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_close(handle: c_int) -> c_int {
    if handle < 0 { return -1; }
    let mut tab = match sessions().lock() { Ok(t) => t, Err(_) => return -1 };
    if let Some(slot) = tab.get_mut(handle as usize) {
        *slot = None;
        return 0;
    }
    -1
}

fn with_session<F: FnOnce(&Session) -> c_int>(handle: c_int, f: F) -> c_int {
    if handle < 0 { return -1; }
    let tab = match sessions().lock() { Ok(t) => t, Err(_) => return -1 };
    match tab.get(handle as usize).and_then(|s| s.as_ref()) {
        Some(sess) => f(sess),
        None => -1,
    }
}
