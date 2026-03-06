//! Opaque handle system for FFI.
//!
//! Rust objects are heap-allocated and returned as raw pointers.
//! Consumers must call the matching `*_free()` function to release.

use crate::error::set_last_error;

/// Convert a Rust value into a heap-allocated opaque pointer.
pub(crate) fn into_handle<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// Reclaim and drop a handle. Returns true if the pointer was valid.
///
/// # Safety
/// The pointer must have been created by `into_handle` and not
/// previously freed.
pub(crate) unsafe fn free_handle<T>(ptr: *mut T) -> bool {
    if ptr.is_null() {
        return false;
    }
    drop(unsafe { Box::from_raw(ptr) });
    true
}

/// Borrow a handle as a shared reference.
///
/// Sets last error and returns `None` if null.
pub(crate) fn borrow_handle<'a, T>(ptr: *const T) -> Option<&'a T> {
    if ptr.is_null() {
        set_last_error("null handle");
        return None;
    }
    Some(unsafe { &*ptr })
}

/// Borrow a handle as a mutable reference.
///
/// Sets last error and returns `None` if null.
pub(crate) fn borrow_handle_mut<'a, T>(ptr: *mut T) -> Option<&'a mut T> {
    if ptr.is_null() {
        set_last_error("null handle");
        return None;
    }
    Some(unsafe { &mut *ptr })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_handle() {
        let h = into_handle(42u64);
        assert!(!h.is_null());
        let val = borrow_handle(h).unwrap();
        assert_eq!(*val, 42);
        assert!(unsafe { free_handle(h) });
    }

    #[test]
    fn null_borrow_returns_none() {
        let result = borrow_handle::<u64>(std::ptr::null());
        assert!(result.is_none());
    }

    #[test]
    fn null_borrow_mut_returns_none() {
        let result = borrow_handle_mut::<u64>(std::ptr::null_mut());
        assert!(result.is_none());
    }

    #[test]
    fn free_null_returns_false() {
        assert!(!unsafe { free_handle::<u64>(std::ptr::null_mut()) });
    }

    #[test]
    fn mutable_borrow_can_modify() {
        let h = into_handle(10u32);
        let val = borrow_handle_mut(h).unwrap();
        *val = 20;
        let val = borrow_handle(h as *const u32).unwrap();
        assert_eq!(*val, 20);
        assert!(unsafe { free_handle(h) });
    }
}
