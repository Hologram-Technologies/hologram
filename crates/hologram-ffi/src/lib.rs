//! Hologram C ABI + WASM bindings.
//!
//! Exposes the full compile / load / introspect / execute / close surface
//! against the CPU backend. WASM bindings can be enabled via the `wasm`
//! feature.

#![allow(clippy::missing_safety_doc)]

pub mod sdk;

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uchar};
use std::sync::Mutex;
use std::sync::OnceLock;

use hologram_archive::FORMAT_VERSION;
use hologram_backend::CpuBackend;
use hologram_compiler::source::{
    self, SourceBinding, SourceConst, SourceExternalConst, SourceExternalTensor, SourceInput,
    SourceItem, SourceOpCall, SourceOutput, SourceProgram, SourceTensorLiteral, SourceType,
};
use hologram_compiler::{BackendKind, CompileError, Compiler};
use hologram_exec::{BufferArena, InferenceSession, InputBuffer};
use hologram_graph::registry::{DTypeId, ShapeDescriptor};
use hologram_graph::{Graph, OpKind};
use prism::vocabulary::WittLevel;

type Session = InferenceSession<CpuBackend<BufferArena>>;
const ABI_VERSION: u32 = 1;
const DTYPE_F32: u8 = 8;
pub const HOLOGRAM_ABI_VERSION: u32 = ABI_VERSION;
pub const HOLOGRAM_ERROR_NONE: c_int = 0;
pub const HOLOGRAM_ERROR_PARSE: c_int = 1;
pub const HOLOGRAM_ERROR_GRAPH: c_int = 2;
pub const HOLOGRAM_ERROR_UNSUPPORTED_OP: c_int = 3;
pub const HOLOGRAM_ERROR_BAD_ATTR: c_int = 4;
pub const HOLOGRAM_ERROR_SHAPE: c_int = 5;
pub const HOLOGRAM_ERROR_EXTERNAL_TENSOR: c_int = 6;
pub const HOLOGRAM_ERROR_ARCHIVE_LOAD: c_int = 7;
pub const HOLOGRAM_ERROR_EXECUTION: c_int = 8;
pub const HOLOGRAM_ERROR_ABI_MISMATCH: c_int = 9;
pub const HOLOGRAM_ERROR_INVALID_ARGUMENT: c_int = 10;
pub const HOLOGRAM_ERROR_UNSUPPORTED_DTYPE: c_int = 11;
pub const HOLOGRAM_ERROR_COMPILE: c_int = 12;

thread_local! {
    static LAST_ERROR: RefCell<Option<FfiErrorRecord>> = const { RefCell::new(None) };
}

struct FfiErrorRecord {
    code: c_int,
    message: CString,
    line: usize,
    column: usize,
    rejected: Option<CString>,
}

struct FfiError {
    code: c_int,
    message: &'static str,
    line: usize,
    column: usize,
    rejected: Option<String>,
}

impl FfiError {
    const fn new(code: c_int, message: &'static str) -> Self {
        Self {
            code,
            message,
            line: 0,
            column: 0,
            rejected: None,
        }
    }

    fn diagnostic(code: c_int, diagnostic: source::SourceDiagnostic) -> Self {
        Self {
            code,
            message: diagnostic.kind,
            line: diagnostic.line,
            column: diagnostic.column,
            rejected: Some(diagnostic.rejected),
        }
    }
}

/// FFI byte string. Bytes must be valid UTF-8 for name/op fields.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramString {
    pub ptr: *const c_uchar,
    pub len: usize,
}

/// FFI tensor shape. `rank == 0` means shape omitted where allowed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramShape {
    pub dims: *const u64,
    pub rank: usize,
}

/// FFI tensor descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramTensorDesc {
    pub name: HologramString,
    pub dtype_id: u8,
    pub shape: HologramShape,
}

/// FFI inline constant descriptor. Bytes must already match dtype endianness.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramConstDesc {
    pub tensor: HologramTensorDesc,
    pub bytes: *const c_uchar,
    pub byte_len: usize,
}

/// FFI external tensor reference descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramExternalTensorDesc {
    pub tensor: HologramTensorDesc,
    pub path: HologramString,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub content_hash: [c_uchar; 32],
}

/// FFI source op descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HologramSourceOp {
    pub output: HologramString,
    pub op: HologramString,
    pub inputs: *const HologramString,
    pub input_count: usize,
    pub shape: HologramShape,
}

/// Opaque source builder handle for SDKs.
pub struct HologramSourceBuilder {
    program: SourceProgram,
}

fn sessions() -> &'static Mutex<Vec<Option<Session>>> {
    static SESSIONS: OnceLock<Mutex<Vec<Option<Session>>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Return the C ABI version implemented by this library.
#[no_mangle]
pub extern "C" fn hologram_abi_version() -> u32 {
    ABI_VERSION
}

/// Return the `.holo` archive format version accepted by this library.
#[no_mangle]
pub extern "C" fn hologram_archive_format_version() -> u32 {
    u32::from(FORMAT_VERSION)
}

/// Return 1 when an additive FFI feature is available, 0 when it is not.
#[no_mangle]
pub unsafe extern "C" fn hologram_feature_supported(feature: HologramString) -> c_int {
    clear_error();
    match ffi_str(feature) {
        Ok(feature) => i32::from(feature_supported(&feature)),
        Err(error) => error_code(error),
    }
}

/// Return 0 when no FFI error is recorded on this thread, otherwise an error code.
#[no_mangle]
pub extern "C" fn hologram_last_error() -> c_int {
    hologram_last_error_code()
}

/// Return the stable error category for the last FFI error on this thread.
#[no_mangle]
pub extern "C" fn hologram_last_error_code() -> c_int {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(error) => error.code,
        None => HOLOGRAM_ERROR_NONE,
    })
}

/// Return the last FFI error message for this thread, or null.
#[no_mangle]
pub extern "C" fn hologram_error_message() -> *const c_char {
    hologram_last_error_message()
}

/// Return the last FFI error message for this thread, or null.
#[no_mangle]
pub extern "C" fn hologram_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(error) => error.message.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Return the 1-based line for the last source-positioned FFI error, or 0.
#[no_mangle]
pub extern "C" fn hologram_last_error_line() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map_or(0, |error| error.line))
}

/// Return the 1-based column for the last source-positioned FFI error, or 0.
#[no_mangle]
pub extern "C" fn hologram_last_error_column() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().as_ref().map_or(0, |error| error.column))
}

/// Return the rejected source token/fragment for the last FFI error, or null.
#[no_mangle]
pub extern "C" fn hologram_last_error_rejected() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(error) => error
            .rejected
            .as_ref()
            .map_or(std::ptr::null(), |rejected| rejected.as_ptr()),
        None => std::ptr::null(),
    })
}

/// Create a source builder for SDK graph construction.
#[no_mangle]
pub extern "C" fn hologram_source_builder_new() -> *mut HologramSourceBuilder {
    clear_error();
    Box::into_raw(Box::new(HologramSourceBuilder {
        program: SourceProgram::new(),
    }))
}

/// Free a source builder. Null is accepted.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_free(builder: *mut HologramSourceBuilder) {
    if !builder.is_null() {
        drop(Box::from_raw(builder));
    }
}

/// Add an input declaration. Returns the source symbol id, or -1.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_input(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramTensorDesc,
) -> c_int {
    source_builder_input(builder, desc).unwrap_or_else(error_code)
}

/// Add an inline constant declaration. Returns the source symbol id, or -1.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_const(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramConstDesc,
) -> c_int {
    source_builder_const(builder, desc).unwrap_or_else(error_code)
}

/// Add a file-backed constant reference. Returns the source symbol id, or -1.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_const_ref(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramExternalTensorDesc,
) -> c_int {
    source_builder_const_ref(builder, desc).unwrap_or_else(error_code)
}

/// Add a named op call. Returns the output source symbol id, or -1.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_op(
    builder: *mut HologramSourceBuilder,
    op: *const HologramSourceOp,
) -> c_int {
    source_builder_op(builder, op).unwrap_or_else(error_code)
}

/// Add a graph output referencing a prior source symbol name.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_output(
    builder: *mut HologramSourceBuilder,
    name: HologramString,
) -> c_int {
    source_builder_output(builder, name).unwrap_or_else(error_code)
}

/// Add a graph output with a semantic port name referencing a prior source symbol.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_output_alias(
    builder: *mut HologramSourceBuilder,
    name: HologramString,
    source: HologramString,
) -> c_int {
    source_builder_output_alias(builder, name, source).unwrap_or_else(error_code)
}

/// Compile a source builder to a `.holo` archive.
#[no_mangle]
pub unsafe extern "C" fn hologram_source_builder_compile(
    builder: *const HologramSourceBuilder,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    source_builder_compile(builder, out, out_capacity).unwrap_or_else(error_code)
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
    clear_error();
    if source_ptr.is_null() || out.is_null() {
        return error_code(invalid_arg_error("compile source null"));
    }
    let bytes = std::slice::from_raw_parts(source_ptr, source_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return error_code(parse_error("source: utf8")),
    };
    let program = match source::parse_ir_diagnostic(s, source::SourceLanguage::Hologram) {
        Ok(program) => program,
        Err(diagnostic) => return error_code(source_diagnostic_error(diagnostic)),
    };
    let graph = match source::lower_ir(&program) {
        Ok(graph) => graph,
        Err(err) => return error_code(source_error(err)),
    };
    let compiled = match Compiler::new(graph, BackendKind::Cpu, WittLevel::W32).compile() {
        Ok(output) => output,
        Err(err) => return error_code(compile_error(err)),
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

/// Return input port `i`'s dtype tag, or -1 on error / `i` out of range.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_input_dtype(handle: c_int, i: usize) -> c_int {
    with_session(handle, |s| port_dtype(s.input_ports(), i))
}

/// Return output port `i`'s dtype tag, or -1 on error / `i` out of range.
#[no_mangle]
pub unsafe extern "C" fn hologram_session_output_dtype(handle: c_int, i: usize) -> c_int {
    with_session(handle, |s| port_dtype(s.output_ports(), i))
}

fn port_dtype(ports: &[hologram_archive::PortDescriptor], i: usize) -> c_int {
    ports.get(i).map_or(-1, |port| c_int::from(port.dtype))
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

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn set_error(error: FfiError) {
    let message = CString::new(error.message).ok();
    let rejected = error
        .rejected
        .and_then(|rejected| CString::new(rejected).ok());
    let record = message.map(|message| FfiErrorRecord {
        code: error.code,
        message,
        line: error.line,
        column: error.column,
        rejected,
    });
    LAST_ERROR.with(|slot| *slot.borrow_mut() = record);
}

fn error_code(error: FfiError) -> c_int {
    set_error(error);
    -1
}

fn feature_supported(feature: &str) -> bool {
    matches!(
        feature,
        "abi.v1"
            | "archive.v2"
            | "compile.empty"
            | "compile.source"
            | "session"
            | "source-builder"
            | "source-builder.const"
            | "source-builder.const-ref"
            | "source-builder.output-alias"
            | "errors.structured"
            | "errors.locations"
    )
}

fn invalid_arg_error(message: &'static str) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_INVALID_ARGUMENT, message)
}

fn null_error(message: &'static str) -> FfiError {
    invalid_arg_error(message)
}

fn unsupported_dtype_error() -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_UNSUPPORTED_DTYPE, "dtype unsupported")
}

fn unsupported_op_error() -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_UNSUPPORTED_OP, "source op unknown")
}

fn shape_error(message: &'static str) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_SHAPE, message)
}

fn graph_error(message: &'static str) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_GRAPH, message)
}

fn external_tensor_error(message: &'static str) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_EXTERNAL_TENSOR, message)
}

fn parse_error(message: &'static str) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_PARSE, message)
}

fn compile_error(_err: CompileError) -> FfiError {
    FfiError::new(HOLOGRAM_ERROR_COMPILE, "source builder compile")
}

fn source_error(err: CompileError) -> FfiError {
    match err {
        CompileError::SourceParse(message) => source_parse_error(message),
        CompileError::GraphValidation(_) => graph_error("source builder graph validation"),
        CompileError::ShapeViolation { .. } => shape_error("source builder shape violation"),
        CompileError::UnsupportedOp(_) => unsupported_op_error(),
        _ => compile_error(err),
    }
}

fn source_parse_error(message: &'static str) -> FfiError {
    if message.contains("unknown op") {
        return unsupported_op_error();
    }
    if message.starts_with("external const") {
        return external_tensor_error(message);
    }
    if message.contains("shape") || message.contains("byte") || message.contains("value count") {
        return shape_error(message);
    }
    if message.contains("duplicate")
        || message.contains("unresolved")
        || message.contains("unknown/!node")
    {
        return graph_error(message);
    }
    parse_error(message)
}

fn source_diagnostic_error(diagnostic: source::SourceDiagnostic) -> FfiError {
    let code = source_parse_error(diagnostic.kind).code;
    FfiError::diagnostic(code, diagnostic)
}

unsafe fn source_builder_input(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramTensorDesc,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let desc = desc.as_ref().ok_or(null_error("tensor descriptor null"))?;
    let name = ffi_str(desc.name)?;
    let symbol = builder.program.intern(&name);
    let ty = tensor_type(desc)?;
    builder
        .program
        .push(SourceItem::Input(SourceInput::new(symbol, ty)));
    Ok(symbol.0 as c_int)
}

unsafe fn source_builder_const(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramConstDesc,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let desc = desc.as_ref().ok_or(null_error("const descriptor null"))?;
    let name = ffi_str(desc.tensor.name)?;
    let symbol = builder.program.intern(&name);
    let literal = source_literal(desc)?;
    let ty = tensor_type(&desc.tensor)?;
    builder
        .program
        .push(SourceItem::Const(SourceConst::new(symbol, ty, literal)));
    Ok(symbol.0 as c_int)
}

unsafe fn source_builder_const_ref(
    builder: *mut HologramSourceBuilder,
    desc: *const HologramExternalTensorDesc,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let desc = desc
        .as_ref()
        .ok_or(null_error("const ref descriptor null"))?;
    let name = ffi_str(desc.tensor.name)?;
    let path = ffi_str(desc.path)?;
    let symbol = builder.program.intern(&name);
    let ty = tensor_type(&desc.tensor)?;
    let reference =
        SourceExternalTensor::file(path, desc.byte_offset, desc.byte_len, desc.content_hash);
    builder
        .program
        .push(SourceItem::ExternalConst(SourceExternalConst::new(
            symbol, ty, reference,
        )));
    Ok(symbol.0 as c_int)
}

unsafe fn source_builder_op(
    builder: *mut HologramSourceBuilder,
    op: *const HologramSourceOp,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let op = op.as_ref().ok_or(null_error("source op null"))?;
    let output_name = ffi_str(op.output)?;
    let op_name = ffi_str(op.op)?;
    let output = builder.program.intern(&output_name);
    let kind = op_kind(&op_name).ok_or(unsupported_op_error())?;
    let inputs = source_inputs(&mut builder.program, op)?;
    let ty = optional_tensor_type(op.shape)?;
    builder.program.push(SourceItem::Binding(SourceBinding::op(
        Some(output),
        SourceOpCall::new(kind, inputs, ty),
    )));
    Ok(output.0 as c_int)
}

unsafe fn source_builder_output(
    builder: *mut HologramSourceBuilder,
    name: HologramString,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let name = ffi_str(name)?;
    let symbol = builder.program.intern(&name);
    builder
        .program
        .push(SourceItem::Output(SourceOutput::new(symbol)));
    Ok(symbol.0 as c_int)
}

unsafe fn source_builder_output_alias(
    builder: *mut HologramSourceBuilder,
    name: HologramString,
    source: HologramString,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_mut().ok_or(null_error("source builder null"))?;
    let name = builder.program.intern(&ffi_str(name)?);
    let source = builder.program.intern(&ffi_str(source)?);
    builder
        .program
        .push(SourceItem::Output(SourceOutput::with_port_name(
            source, name,
        )));
    Ok(name.0 as c_int)
}

unsafe fn source_builder_compile(
    builder: *const HologramSourceBuilder,
    out: *mut c_uchar,
    out_capacity: usize,
) -> Result<c_int, FfiError> {
    clear_error();
    let builder = builder.as_ref().ok_or(null_error("source builder null"))?;
    let graph = source::lower_ir(&builder.program).map_err(source_error)?;
    let compiled = Compiler::new(graph, BackendKind::Cpu, WittLevel::W32)
        .compile()
        .map_err(compile_error)?;
    copy_archive(&compiled.archive, out, out_capacity)
}

unsafe fn source_inputs(
    program: &mut SourceProgram,
    op: &HologramSourceOp,
) -> Result<Vec<source::SourceSymbol>, FfiError> {
    if op.input_count == 0 {
        return Ok(Vec::new());
    }
    if op.inputs.is_null() {
        return Err(null_error("source op inputs null"));
    }
    let inputs = std::slice::from_raw_parts(op.inputs, op.input_count);
    inputs
        .iter()
        .map(|input| source_symbol(program, *input))
        .collect()
}

unsafe fn source_symbol(
    program: &mut SourceProgram,
    name: HologramString,
) -> Result<source::SourceSymbol, FfiError> {
    ffi_str(name).map(|name| program.intern(&name))
}

fn op_kind(name: &str) -> Option<OpKind> {
    OpKind::ALL.iter().copied().find(|kind| kind.name() == name)
}

unsafe fn ffi_str(value: HologramString) -> Result<String, FfiError> {
    if value.len == 0 {
        return Ok(String::new());
    }
    if value.ptr.is_null() {
        return Err(null_error("string null"));
    }
    let bytes = std::slice::from_raw_parts(value.ptr, value.len);
    let text = std::str::from_utf8(bytes).map_err(|_| invalid_arg_error("string utf8"))?;
    Ok(text.to_owned())
}

unsafe fn source_literal(desc: &HologramConstDesc) -> Result<SourceTensorLiteral, FfiError> {
    let dtype = dtype_id(desc.tensor.dtype_id)?;
    let width = dtype_width(dtype)?;
    if !desc.byte_len.is_multiple_of(width) {
        return Err(shape_error("const byte len"));
    }
    let bytes = ffi_bytes(desc.bytes, desc.byte_len)?;
    Ok(SourceTensorLiteral::new(bytes, desc.byte_len / width))
}

unsafe fn ffi_bytes(ptr: *const c_uchar, len: usize) -> Result<Vec<u8>, FfiError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(null_error("bytes null"));
    }
    Ok(std::slice::from_raw_parts(ptr, len).to_vec())
}

unsafe fn tensor_type(desc: &HologramTensorDesc) -> Result<SourceType, FfiError> {
    let dtype = dtype_id(desc.dtype_id)?;
    let shape = required_shape(desc.shape)?;
    Ok(SourceType {
        dtype,
        shape: Some(shape),
    })
}

unsafe fn optional_tensor_type(shape: HologramShape) -> Result<Option<SourceType>, FfiError> {
    let shape = optional_shape(shape)?;
    Ok(shape.map(|shape| SourceType::f32(Some(shape))))
}

unsafe fn required_shape(shape: HologramShape) -> Result<ShapeDescriptor, FfiError> {
    optional_shape(shape)?.ok_or(shape_error("shape missing"))
}

unsafe fn optional_shape(shape: HologramShape) -> Result<Option<ShapeDescriptor>, FfiError> {
    if shape.rank == 0 {
        return Ok(None);
    }
    shape_descriptor(shape).map(Some)
}

unsafe fn shape_descriptor(shape: HologramShape) -> Result<ShapeDescriptor, FfiError> {
    if shape.rank > 8 || shape.dims.is_null() {
        return Err(shape_error("shape bad"));
    }
    let mut dims = [0; 8];
    dims[..shape.rank].copy_from_slice(std::slice::from_raw_parts(shape.dims, shape.rank));
    Ok(ShapeDescriptor {
        rank: shape.rank as u8,
        dims,
        dims_overflow: None,
    })
}

fn dtype_id(dtype: u8) -> Result<DTypeId, FfiError> {
    match dtype {
        0 | DTYPE_F32 => Ok(DTypeId(DTYPE_F32)),
        _ => Err(unsupported_dtype_error()),
    }
}

fn dtype_width(dtype: DTypeId) -> Result<usize, FfiError> {
    match dtype.0 {
        DTYPE_F32 => Ok(4),
        _ => Err(unsupported_dtype_error()),
    }
}

unsafe fn copy_archive(
    archive: &[u8],
    out: *mut c_uchar,
    out_capacity: usize,
) -> Result<c_int, FfiError> {
    if out.is_null() {
        return Err(null_error("archive output null"));
    }
    let n = archive.len().min(out_capacity);
    std::slice::from_raw_parts_mut(out, n).copy_from_slice(&archive[..n]);
    Ok(archive.len() as c_int)
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
