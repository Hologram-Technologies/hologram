//! FFI error handling with thread-local last-error propagation.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

/// FFI result codes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiStatus {
    /// Operation succeeded.
    Ok = 0,
    /// Null pointer argument.
    NullPointer = -1,
    /// Invalid handle (freed or wrong type).
    InvalidHandle = -2,
    /// Graph validation failed.
    ValidationError = -3,
    /// Compilation failed.
    CompilationError = -4,
    /// Execution failed.
    ExecutionError = -5,
    /// Invalid argument value.
    InvalidArgument = -6,
    /// Archive load/parse failed.
    ArchiveError = -7,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message for later retrieval via `hologram_error_message`.
pub(crate) fn set_last_error(msg: impl Into<String>) {
    let s = msg.into();
    let c = CString::new(s).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Clear the last error.
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Return the last error code (0 if none).
///
/// # Safety
/// Safe to call from any thread.
#[no_mangle]
pub extern "C" fn hologram_last_error() -> i32 {
    LAST_ERROR.with(|e| if e.borrow().is_some() { -1 } else { 0 })
}

/// Return a pointer to the last error message (null-terminated UTF-8).
///
/// The returned pointer is valid until the next FFI call on this thread.
/// Returns null if no error is set.
///
/// # Safety
/// The caller must not free the returned pointer.
#[no_mangle]
pub extern "C" fn hologram_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow().as_ref() {
        Some(c) => c.as_ptr(),
        None => std::ptr::null(),
    })
}

/// Clear any previous error, then call the closure.
/// On `Err`, stores the message and returns the status code.
/// On `Ok`, returns 0.
pub(crate) fn ffi_catch<F>(f: F) -> i32
where
    F: FnOnce() -> Result<(), (FfiStatus, String)>,
{
    clear_last_error();
    match f() {
        Ok(()) => FfiStatus::Ok as i32,
        Err((status, msg)) => {
            set_last_error(msg);
            status as i32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_no_error() {
        clear_last_error();
        assert_eq!(hologram_last_error(), 0);
        assert!(hologram_error_message().is_null());
    }

    #[test]
    fn set_and_retrieve_error() {
        set_last_error("test error");
        assert_ne!(hologram_last_error(), 0);
        let ptr = hologram_error_message();
        assert!(!ptr.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "test error");
    }

    #[test]
    fn clear_removes_error() {
        set_last_error("oops");
        clear_last_error();
        assert_eq!(hologram_last_error(), 0);
        assert!(hologram_error_message().is_null());
    }

    #[test]
    fn ffi_catch_success() {
        let code = ffi_catch(|| Ok(()));
        assert_eq!(code, 0);
        assert_eq!(hologram_last_error(), 0);
    }

    #[test]
    fn ffi_catch_failure_sets_error() {
        let code = ffi_catch(|| Err((FfiStatus::InvalidArgument, "bad arg".into())));
        assert_eq!(code, FfiStatus::InvalidArgument as i32);
        let ptr = hologram_error_message();
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "bad arg");
    }

    #[test]
    fn ffi_status_codes_are_negative() {
        assert_eq!(FfiStatus::Ok as i32, 0);
        assert!((FfiStatus::NullPointer as i32) < 0);
        assert!((FfiStatus::InvalidHandle as i32) < 0);
        assert!((FfiStatus::ValidationError as i32) < 0);
        assert!((FfiStatus::CompilationError as i32) < 0);
        assert!((FfiStatus::ExecutionError as i32) < 0);
        assert!((FfiStatus::InvalidArgument as i32) < 0);
        assert!((FfiStatus::ArchiveError as i32) < 0);
    }

    #[test]
    fn ffi_catch_clears_previous_error() {
        set_last_error("old error");
        let code = ffi_catch(|| Ok(()));
        assert_eq!(code, 0);
        assert!(hologram_error_message().is_null());
    }
}
