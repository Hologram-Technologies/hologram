//! `hologram-ai` C ABI surface (cargo feature `ai`, **default off**) — compile /
//! download / load / invoke `.holo` v4 inference-model applications (spec
//! `refactor/03` §v4) with JSON request/response payloads for binding simplicity.
//!
//! Every function here is fully wired Rust that delegates to the sibling
//! `hologram-ai` crate — hologram itself stays engine-agnostic (no engine code,
//! no engine dependency in the default build). Enabling `ai` without the
//! (currently commented-out) `hologram-ai` dependency is a deliberate compile
//! error — there is no stub fallback. See `crates/hologram-ffi/Cargo.toml` for
//! the wiring steps; they mirror `crates/hologram-cli/src/ai.rs`.
//!
//! # Assumed `hologram-ai` API (the contract this surface is written against)
//!
//! ```ignore
//! // hologram_ai::ffi — the engine-side implementation of this surface. `AiError.code`
//! // is one of the HOLOGRAM_ERROR_AI_* categories below.
//! pub struct AiError { pub code: i32, pub message: String }
//! pub struct AiApp { /* opaque: a loaded `.holo` v4 AI application */ }
//! pub struct AiSession { /* opaque: an invocation session over one service entry; 'static */ }
//! pub fn compile_huggingface(repository: &str, revision: &str, entry: &str, output_path: &str) -> Result<(), AiError>;
//! pub fn compile_source(source_dir: &str, entry: &str, output_path: &str) -> Result<(), AiError>;
//! pub fn download(repository: &str, revision: &str, offline: bool) -> Result<(), AiError>;
//! pub fn app_load_path(path: &str) -> Result<AiApp, AiError>;
//! pub fn app_load_bytes(bytes: &[u8]) -> Result<AiApp, AiError>;
//! impl AiApp { pub fn model_entries(&self) -> Vec<String>; }
//! pub fn session_open(app: &AiApp, entry: &str) -> Result<AiSession, AiError>;
//! pub fn session_invoke_json(session: &mut AiSession, request_json: &str) -> Result<String, AiError>;
//! ```

use std::os::raw::{c_int, c_uchar};
use std::sync::{Mutex, OnceLock};

use crate::{clear_error, error_code, ffi_bytes, ffi_str, set_error_message, HologramString};
use hologram_ai::ffi::{AiApp, AiError, AiSession};

// AI error categories — appended to the core 0..=12 range. The 100+ band is
// reserved for AI-surface failures so the core codes never renumber and SDK
// bindings can branch on category ranges.
/// Invalid or unpinnable model source (repository / revision / source tree).
pub const HOLOGRAM_ERROR_AI_SOURCE: c_int = 100;
/// Registry authentication/authorization failed.
pub const HOLOGRAM_ERROR_AI_AUTH: c_int = 101;
/// The model / operation / engine is unsupported by this build.
pub const HOLOGRAM_ERROR_AI_UNSUPPORTED: c_int = 102;
/// Model → `.holo` v4 compilation failed.
pub const HOLOGRAM_ERROR_AI_COMPILE: c_int = 103;
/// The compiled bundle is malformed.
pub const HOLOGRAM_ERROR_AI_BUNDLE: c_int = 104;
/// The `.holo` archive failed to load.
pub const HOLOGRAM_ERROR_AI_ARCHIVE: c_int = 105;
/// Content integrity (κ / certificate verification) failed.
pub const HOLOGRAM_ERROR_AI_INTEGRITY: c_int = 106;
/// The requested service entry does not exist in the app.
pub const HOLOGRAM_ERROR_AI_SELECTION: c_int = 107;
/// The layer's named engine is unavailable or rejected the model.
pub const HOLOGRAM_ERROR_AI_ENGINE: c_int = 108;
/// Inference itself failed.
pub const HOLOGRAM_ERROR_AI_INFERENCE: c_int = 109;
/// The operation was cancelled.
pub const HOLOGRAM_ERROR_AI_CANCELLED: c_int = 110;
/// AI-surface ABI misuse (null handle, poisoned table, capacity abuse).
pub const HOLOGRAM_ERROR_AI_ABI: c_int = 111;

/// Convert an FFI string, or return its recorded error code from the extern fn.
macro_rules! cstr {
    ($s:expr) => {
        match ffi_str($s) {
            Ok(s) => s,
            Err(e) => return error_code(e),
        }
    };
}

/// Parsed compile request shared by both compile verbs.
struct CompileRequest {
    source: CompileSource,
    entry: String,
    output_path: String,
}

enum CompileSource {
    HuggingFace {
        repository: String,
        revision: String,
    },
    SourceDir(String),
}

fn apps() -> &'static Mutex<Vec<Option<AiApp>>> {
    static APPS: OnceLock<Mutex<Vec<Option<AiApp>>>> = OnceLock::new();
    APPS.get_or_init(|| Mutex::new(Vec::new()))
}

fn ai_sessions() -> &'static Mutex<Vec<Option<AiSession>>> {
    static SESSIONS: OnceLock<Mutex<Vec<Option<AiSession>>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Compile a pinned Hugging Face repository revision into a `.holo` v4 AI
/// application at `output_path`, with `entry` as the inference-model layer's
/// service name. Returns 0 on success, -1 on error (see `hologram_last_error*`).
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_compile_huggingface(
    repository: HologramString,
    revision: HologramString,
    entry: HologramString,
    output_path: HologramString,
) -> c_int {
    clear_error();
    let request = CompileRequest {
        source: CompileSource::HuggingFace {
            repository: cstr!(repository),
            revision: cstr!(revision),
        },
        entry: cstr!(entry),
        output_path: cstr!(output_path),
    };
    compile(request)
}

/// Compile a local source tree into a `.holo` v4 AI application (see
/// [`hologram_ai_compile_huggingface`]).
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_compile_source(
    source_dir: HologramString,
    entry: HologramString,
    output_path: HologramString,
) -> c_int {
    clear_error();
    let request = CompileRequest {
        source: CompileSource::SourceDir(cstr!(source_dir)),
        entry: cstr!(entry),
        output_path: cstr!(output_path),
    };
    compile(request)
}

/// Download a pinned model repository revision into the local cache (`offline !=
/// 0` resolves from the cache only). Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_download(
    repository: HologramString,
    revision: HologramString,
    offline: c_int,
) -> c_int {
    clear_error();
    status(hologram_ai::ffi::download(
        &cstr!(repository),
        &cstr!(revision),
        offline != 0,
    ))
}

/// Load a `.holo` v4 AI application from a file path. Returns an opaque app
/// handle (a session-table index), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_app_load_path(path: HologramString) -> c_int {
    clear_error();
    app_handle(hologram_ai::ffi::app_load_path(&cstr!(path)))
}

/// Load a `.holo` v4 AI application from bytes. Returns an opaque app handle,
/// or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_app_load_bytes(
    bytes_ptr: *const c_uchar,
    bytes_len: usize,
) -> c_int {
    clear_error();
    let bytes = match ffi_bytes(bytes_ptr, bytes_len) {
        Ok(b) => b,
        Err(e) => return error_code(e),
    };
    app_handle(hologram_ai::ffi::app_load_bytes(&bytes))
}

/// Number of inference-model service entries in a loaded app, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_model_count(app: c_int) -> c_int {
    clear_error();
    with_app(app, |a| a.model_entries().len() as c_int).unwrap_or(-1)
}

/// Copy the service entry name at `index` into `out`. Returns the **total name
/// length** (snprintf-style; a value > `out_capacity` means truncated, retry
/// with a larger buffer), or -1 on error / out of range.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_model_entry(
    app: c_int,
    index: usize,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    clear_error();
    with_app(app, |a| unsafe { copy_entry(a, index, out, out_capacity) }).unwrap_or(-1)
}

/// Open an invocation session over one of the app's service entries. Returns an
/// opaque session handle, or -1 on error (unknown entry ⇒
/// [`HOLOGRAM_ERROR_AI_SELECTION`]).
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_session_open(app: c_int, entry: HologramString) -> c_int {
    clear_error();
    let entry = cstr!(entry);
    match with_app(app, |a| hologram_ai::ffi::session_open(a, &entry)) {
        Ok(Ok(session)) => push_slot(ai_sessions(), session),
        Ok(Err(e)) => ai_fail(e),
        Err(code) => code,
    }
}

/// Invoke a session with a JSON request document, writing the JSON response
/// into `out`. Returns the **total response length** (snprintf-style; a value >
/// `out_capacity` means truncated, retry with a larger buffer), or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_session_invoke_json(
    session: c_int,
    request: HologramString,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    clear_error();
    let request = cstr!(request);
    with_session_mut(session, |s| unsafe {
        invoke_json(s, &request, out, out_capacity)
    })
}

/// Drop a previously-opened AI session. Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn hologram_ai_session_close(session: c_int) -> c_int {
    clear_error();
    close_slot(ai_sessions(), session)
}

fn compile(request: CompileRequest) -> c_int {
    let (entry, output_path) = (request.entry.as_str(), request.output_path.as_str());
    let result = match &request.source {
        CompileSource::HuggingFace {
            repository,
            revision,
        } => hologram_ai::ffi::compile_huggingface(repository, revision, entry, output_path),
        CompileSource::SourceDir(dir) => hologram_ai::ffi::compile_source(dir, entry, output_path),
    };
    status(result)
}

/// 0 on success; on failure records the AI error and returns -1.
fn status(result: Result<(), AiError>) -> c_int {
    match result {
        Ok(()) => 0,
        Err(e) => ai_fail(e),
    }
}

fn app_handle(result: Result<AiApp, AiError>) -> c_int {
    match result {
        Ok(app) => push_slot(apps(), app),
        Err(e) => ai_fail(e),
    }
}

fn push_slot<T>(table: &Mutex<Vec<Option<T>>>, value: T) -> c_int {
    match table.lock() {
        Ok(mut tab) => {
            tab.push(Some(value));
            (tab.len() - 1) as c_int
        }
        Err(_) => abi_fail("handle table poisoned"),
    }
}

fn close_slot<T>(table: &Mutex<Vec<Option<T>>>, handle: c_int) -> c_int {
    if handle < 0 {
        return abi_fail("negative handle");
    }
    match table.lock() {
        Ok(mut tab) => match tab.get_mut(handle as usize) {
            Some(slot) => {
                *slot = None;
                0
            }
            None => abi_fail("unknown handle"),
        },
        Err(_) => abi_fail("handle table poisoned"),
    }
}

fn with_app<R, F: FnOnce(&AiApp) -> R>(handle: c_int, f: F) -> Result<R, c_int> {
    if handle < 0 {
        return Err(abi_fail("negative handle"));
    }
    match apps().lock() {
        Ok(tab) => match tab.get(handle as usize).and_then(|s| s.as_ref()) {
            Some(app) => Ok(f(app)),
            None => Err(abi_fail("unknown app handle")),
        },
        Err(_) => Err(abi_fail("handle table poisoned")),
    }
}

fn with_session_mut<F: FnOnce(&mut AiSession) -> c_int>(handle: c_int, f: F) -> c_int {
    if handle < 0 {
        return abi_fail("negative handle");
    }
    match ai_sessions().lock() {
        Ok(mut tab) => match tab.get_mut(handle as usize).and_then(|s| s.as_mut()) {
            Some(session) => f(session),
            None => abi_fail("unknown session handle"),
        },
        Err(_) => abi_fail("handle table poisoned"),
    }
}

unsafe fn copy_entry(app: &AiApp, index: usize, out: *mut c_uchar, out_capacity: usize) -> c_int {
    let entries = app.model_entries();
    let Some(name) = entries.get(index) else {
        return ai_fail_static(HOLOGRAM_ERROR_AI_SELECTION, "model entry out of range");
    };
    copy_out(name.as_bytes(), out, out_capacity)
}

unsafe fn invoke_json(
    session: &mut AiSession,
    request: &str,
    out: *mut c_uchar,
    out_capacity: usize,
) -> c_int {
    match hologram_ai::ffi::session_invoke_json(session, request) {
        Ok(response) => copy_out(response.as_bytes(), out, out_capacity),
        Err(e) => ai_fail(e),
    }
}

/// snprintf-style copy: writes at most `out_capacity` bytes, returns the total.
unsafe fn copy_out(bytes: &[u8], out: *mut c_uchar, out_capacity: usize) -> c_int {
    if !out.is_null() {
        let n = bytes.len().min(out_capacity);
        std::slice::from_raw_parts_mut(out, n).copy_from_slice(&bytes[..n]);
    }
    bytes.len() as c_int
}

fn ai_fail(error: AiError) -> c_int {
    ai_fail_static(error.code, &error.message)
}

fn abi_fail(message: &'static str) -> c_int {
    ai_fail_static(HOLOGRAM_ERROR_AI_ABI, message)
}

fn ai_fail_static(code: c_int, message: &str) -> c_int {
    set_error_message(code, message);
    -1
}
