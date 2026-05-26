//! C ABI integration test — exercises the full surface from Rust as if
//! through the C-stable signatures.

use hologram_ffi::*;

#[test]
fn compile_empty_round_trip() {
    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_compile_empty(buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    let archive = &buf[..n as usize];
    assert_eq!(&archive[..4], b"HOLO");

    let handle = unsafe { hologram_session_load(archive.as_ptr(), archive.len()) };
    assert!(handle >= 0);

    let kernel_count = unsafe { hologram_session_kernel_count(handle) };
    assert_eq!(kernel_count, 0);

    let inputs = unsafe { hologram_session_input_count(handle) };
    let outputs = unsafe { hologram_session_output_count(handle) };
    assert_eq!(inputs, 0);
    assert_eq!(outputs, 0);

    let rv = unsafe {
        hologram_session_execute(
            handle,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    assert_eq!(rv, 0);

    let close_rv = unsafe { hologram_session_close(handle) };
    assert_eq!(close_rv, 0);
}

#[test]
fn compile_source_round_trip() {
    let src = b"input x\nop relu x as=y\noutput y\n";
    let mut buf = vec![0u8; 16 * 1024];
    let n =
        unsafe { hologram_compile_source(src.as_ptr(), src.len(), buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);

    let archive = &buf[..n as usize];
    let handle = unsafe { hologram_session_load(archive.as_ptr(), archive.len()) };
    assert!(handle >= 0);

    let inputs = unsafe { hologram_session_input_count(handle) };
    assert_eq!(inputs, 1);

    let zeros = vec![0u8; 1024];
    let in_ptrs = [zeros.as_ptr()];
    let in_lens = [zeros.len()];

    let mut out_buf = vec![0u8; 1024];
    let out_ptrs = [out_buf.as_mut_ptr()];
    let out_caps = [out_buf.len()];

    let rv = unsafe {
        hologram_session_execute(
            handle,
            in_ptrs.as_ptr(),
            in_lens.as_ptr(),
            1,
            out_ptrs.as_ptr(),
            out_caps.as_ptr(),
            1,
        )
    };
    assert_eq!(rv, 0);

    unsafe {
        hologram_session_close(handle);
    }
}

#[test]
fn compile_signals_truncation_via_required_length() {
    // A too-small buffer must not report success: the return value is the full
    // required archive length, which exceeds the capacity we passed.
    let mut tiny = vec![0u8; 8];
    let needed = unsafe { hologram_compile_empty(tiny.as_mut_ptr(), tiny.len()) };
    assert!(
        needed > tiny.len() as i32,
        "return signals full length > capacity"
    );

    // Retrying with exactly the required size succeeds and round-trips.
    let mut buf = vec![0u8; needed as usize];
    let n = unsafe { hologram_compile_empty(buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, needed);
    assert_eq!(&buf[..4], b"HOLO");
}

#[test]
fn execute_fails_loud_on_undersized_output_buffer() {
    let src = b"input x :4\nop relu x :4 as=y\noutput y\n";
    let mut buf = vec![0u8; 16 * 1024];
    let n =
        unsafe { hologram_compile_source(src.as_ptr(), src.len(), buf.as_mut_ptr(), buf.len()) };
    let handle = unsafe { hologram_session_load(buf.as_ptr(), n as usize) };
    assert!(handle >= 0);

    let needed = unsafe { hologram_session_output_byte_len(handle, 0) };
    assert!(needed > 0);

    let zeros = vec![0u8; 1024];
    let in_ptrs = [zeros.as_ptr()];
    let in_lens = [zeros.len()];

    // Output buffer one byte short of what the port produces.
    let mut out_buf = vec![0u8; needed as usize - 1];
    let out_ptrs = [out_buf.as_mut_ptr()];
    let out_caps = [out_buf.len()];
    let rv = unsafe {
        hologram_session_execute(
            handle,
            in_ptrs.as_ptr(),
            in_lens.as_ptr(),
            1,
            out_ptrs.as_ptr(),
            out_caps.as_ptr(),
            1,
        )
    };
    assert_eq!(rv, -1, "undersized output must fail loud, not truncate");

    unsafe {
        hologram_session_close(handle);
    }
}

#[test]
fn negative_handles_return_error() {
    assert_eq!(unsafe { hologram_session_input_count(-1) }, -1);
    assert_eq!(unsafe { hologram_session_output_count(-1) }, -1);
    assert_eq!(unsafe { hologram_session_kernel_count(-1) }, -1);
    assert_eq!(unsafe { hologram_session_close(-1) }, -1);
}

#[test]
fn execute_with_wrong_input_count_errors() {
    let src = b"input x\nop relu x as=y\noutput y\n";
    let mut buf = vec![0u8; 16 * 1024];
    let n =
        unsafe { hologram_compile_source(src.as_ptr(), src.len(), buf.as_mut_ptr(), buf.len()) };
    let handle = unsafe { hologram_session_load(buf.as_ptr(), n as usize) };
    assert!(handle >= 0);

    // Session expects 1 input; pass 0.
    let rv = unsafe {
        hologram_session_execute(
            handle,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
        )
    };
    assert_eq!(rv, -1);

    unsafe {
        hologram_session_close(handle);
    }
}
