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
use hologram_compiler::{BackendKind, Compiler};
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
///
/// Returns the **total archive length** (snprintf-style), or -1 on error. At
/// most `out_capacity` bytes are written; a return value greater than
/// `out_capacity` means the output was truncated and the caller must retry
/// with a buffer of at least the returned size — truncation is never reported
/// as success.
#[no_mangle]
pub unsafe extern "C" fn hologram_compile_empty(out: *mut c_uchar, out_capacity: usize) -> c_int {
    if out.is_null() {
        return -1;
    }
    let graph = Graph::new();
    let compiled = match Compiler::new(graph, BackendKind::Cpu, WittLevel::W32).compile() {
        Ok(o) => o,
        Err(_) => return -1,
    };
    let n = compiled.archive.len().min(out_capacity);
    std::slice::from_raw_parts_mut(out, n).copy_from_slice(&compiled.archive[..n]);
    compiled.archive.len() as c_int
}

/// Compile a textual hologram-source program for the CPU backend.
/// `source_ptr`/`source_len` carry the input bytes (UTF-8).
///
/// Returns the **total archive length** (snprintf-style), or -1 on failure. At
/// most `out_capacity` bytes are written; a return value greater than
/// `out_capacity` means the output was truncated and the caller must retry
/// with a larger buffer — truncation is never reported as success.
#[no_mangle]
pub unsafe extern "C" fn hologram_compile_source(
    source_ptr: *const c_uchar,
    source_len: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    if source_ptr.is_null() || out.is_null() {
        return -1;
    }
    let bytes = std::slice::from_raw_parts(source_ptr, source_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let compiled = match hologram_compiler::compile_from_source(s, WittLevel::W32, BackendKind::Cpu)
    {
        Ok(o) => o,
        Err(_) => return -1,
    };
    let n = compiled.archive.len().min(out_capacity);
    std::slice::from_raw_parts_mut(out, n).copy_from_slice(&compiled.archive[..n]);
    compiled.archive.len() as c_int
}

/// Load an `.holo` archive and return a session handle, or -1 on error.
/// The handle is an opaque integer index into a global session table.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_load(
    archive_ptr: *const c_uchar,
    archive_len: usize,
) -> c_int {
    if archive_ptr.is_null() {
        return -1;
    }
    let bytes = std::slice::from_raw_parts(archive_ptr, archive_len);
    let backend: CpuBackend<BufferArena> = CpuBackend::new();
    let sess = match InferenceSession::load(bytes, backend) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let mut tab = match sessions().lock() {
        Ok(t) => t,
        Err(_) => return -1,
    };
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

/// Byte length the i-th declared output port will produce, or -1 on error.
/// Callers use this to pre-size their output buffers before
/// `hologram_session_execute`. Returns 0 (not -1) when `i` is in range
/// but the port has zero element count.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_output_byte_len(handle: c_int, i: usize) -> c_int {
    with_session(handle, |s| {
        if i >= s.output_count() {
            -1
        } else {
            s.output_byte_len(i) as c_int
        }
    })
}

/// Copy the archive's canonical 32-byte BLAKE3 content fingerprint
/// (spec X.1) into `out` (must point to at least 32 writable bytes).
/// Returns 0 on success or -1 on error. This is the per-content anchor
/// that distinguishes one model from another — pair with the prism
/// attestation surface when auditing.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_archive_fingerprint(
    handle: c_int,
    out: *mut c_uchar,
) -> c_int {
    if out.is_null() {
        return -1;
    }
    with_session(handle, |s| {
        let fp = s.archive_fingerprint();
        std::slice::from_raw_parts_mut(out, 32).copy_from_slice(&fp);
        0
    })
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
    if handle < 0 {
        return -1;
    }
    let mut tab = match sessions().lock() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    let sess = match tab.get_mut(handle as usize).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };

    if in_count != sess.input_count() {
        return -1;
    }
    if out_count != sess.output_count() {
        return -1;
    }

    let mut input_storage: Vec<&[c_uchar]> = Vec::with_capacity(in_count);
    if in_count > 0 {
        if in_ptrs.is_null() || in_lens.is_null() {
            return -1;
        }
        let ptrs = std::slice::from_raw_parts(in_ptrs, in_count);
        let lens = std::slice::from_raw_parts(in_lens, in_count);
        for i in 0..in_count {
            if ptrs[i].is_null() && lens[i] != 0 {
                return -1;
            }
            let s = if lens[i] == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(ptrs[i], lens[i])
            };
            input_storage.push(s);
        }
    }
    let inputs: Vec<InputBuffer> = input_storage
        .iter()
        .map(|b| InputBuffer { bytes: b })
        .collect();

    let outputs = match sess.execute(&inputs) {
        Ok(o) => o,
        Err(_) => return -1,
    };

    if out_count > 0 {
        if out_ptrs.is_null() || out_caps.is_null() {
            return -1;
        }
        let ptrs = std::slice::from_raw_parts(out_ptrs, out_count);
        let caps = std::slice::from_raw_parts(out_caps, out_count);
        // Fail loud *before* writing anything if any output buffer is too small
        // (callers pre-size via `hologram_session_output_byte_len`). Silently
        // truncating an output and still returning success would surface a
        // wrong result as a correct one.
        for (i, out) in outputs.iter().enumerate() {
            if !ptrs[i].is_null() && caps[i] < out.bytes.len() {
                return -1;
            }
        }
        for (i, out) in outputs.iter().enumerate() {
            if ptrs[i].is_null() {
                continue;
            }
            std::slice::from_raw_parts_mut(ptrs[i], out.bytes.len()).copy_from_slice(&out.bytes);
        }
    }
    0
}

/// Drop a previously-loaded session.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_close(handle: c_int) -> c_int {
    if handle < 0 {
        return -1;
    }
    let mut tab = match sessions().lock() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    if let Some(slot) = tab.get_mut(handle as usize) {
        *slot = None;
        return 0;
    }
    -1
}

/// Copy input port `i`'s semantic name (UTF-8, e.g. `"input_ids"`) into `out`.
/// Returns the **total name length** (snprintf-style; a value > `out_capacity`
/// means truncated, retry with a larger buffer), or -1 on error / `i` out of
/// range. An unnamed port returns 0.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_input_name(
    handle: c_int,
    i: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    with_session(handle, |s| {
        copy_port_name(s.input_ports(), i, out, out_capacity)
    })
}

/// Copy output port `i`'s semantic name into `out` (see
/// [`hologram_session_input_name`]).
#[no_mangle]
pub unsafe extern "C" fn hologram_session_output_name(
    handle: c_int,
    i: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    with_session(handle, |s| {
        copy_port_name(s.output_ports(), i, out, out_capacity)
    })
}

/// Write input port `i`'s shape dims into `out_dims` (up to `max_dims`) and
/// return the **rank** (snprintf-style; a value > `max_dims` means the shape
/// was truncated). Returns -1 on error / `i` out of range.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_input_shape(
    handle: c_int,
    i: usize,
    out_dims: *mut u64,
    max_dims: usize,
) -> c_int {
    with_session(handle, |s| {
        copy_port_shape(s.input_ports(), i, out_dims, max_dims)
    })
}

/// Write output port `i`'s shape dims into `out_dims` (see
/// [`hologram_session_input_shape`]).
#[no_mangle]
pub unsafe extern "C" fn hologram_session_output_shape(
    handle: c_int,
    i: usize,
    out_dims: *mut u64,
    max_dims: usize,
) -> c_int {
    with_session(handle, |s| {
        copy_port_shape(s.output_ports(), i, out_dims, max_dims)
    })
}

/// Copy the producer-metadata extension stored under the UTF-8 key
/// `key_ptr`/`key_len` into `out`. Returns the **total byte length**
/// (snprintf-style), or -1 if the key is absent / on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_extension(
    handle: c_int,
    key_ptr: *const c_uchar,
    key_len: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    if key_ptr.is_null() {
        return -1;
    }
    let key = match std::str::from_utf8(std::slice::from_raw_parts(key_ptr, key_len)) {
        Ok(k) => k,
        Err(_) => return -1,
    };
    with_session(handle, |s| match s.extension(key) {
        Some(bytes) => {
            if !out.is_null() {
                let n = bytes.len().min(out_capacity);
                std::slice::from_raw_parts_mut(out, n).copy_from_slice(&bytes[..n]);
            }
            bytes.len() as c_int
        }
        None => -1,
    })
}

unsafe fn copy_port_name(
    ports: &[hologram_archive::PortDescriptor],
    i: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    let Some(p) = ports.get(i) else { return -1 };
    let name = p.name.as_bytes();
    if !out.is_null() {
        let n = name.len().min(out_capacity);
        std::slice::from_raw_parts_mut(out, n).copy_from_slice(&name[..n]);
    }
    name.len() as c_int
}

unsafe fn copy_port_shape(
    ports: &[hologram_archive::PortDescriptor],
    i: usize,
    out_dims: *mut u64,
    max_dims: usize,
) -> c_int {
    let Some(p) = ports.get(i) else { return -1 };
    if !out_dims.is_null() {
        let n = p.shape.len().min(max_dims);
        std::slice::from_raw_parts_mut(out_dims, n).copy_from_slice(&p.shape[..n]);
    }
    p.shape.len() as c_int
}

fn with_session<F: FnOnce(&Session) -> c_int>(handle: c_int, f: F) -> c_int {
    if handle < 0 {
        return -1;
    }
    let tab = match sessions().lock() {
        Ok(t) => t,
        Err(_) => return -1,
    };
    match tab.get(handle as usize).and_then(|s| s.as_ref()) {
        Some(sess) => f(sess),
        None => -1,
    }
}
