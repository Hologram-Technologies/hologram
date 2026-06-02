//! C ABI integration test — exercises the full surface from Rust as if
//! through the C-stable signatures.

use hologram_ffi::*;
use hologram_host::HologramHasher;
use prism::vocabulary::Hasher;

fn ffi_str(bytes: &[u8]) -> HologramString {
    HologramString {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    HologramHasher::initial().fold_bytes(bytes).finalize()
}

fn temp_tensor(bytes: &[u8]) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "hologram-ffi-external-{}-{}-{}.bin",
        std::process::id(),
        nanos,
        bytes.len()
    ));
    std::fs::write(&path, bytes).expect("write temp tensor");
    path
}

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
fn reports_versions_and_supported_features() {
    assert_eq!(hologram_abi_version(), HOLOGRAM_ABI_VERSION);
    assert_eq!(hologram_archive_format_version(), 2);
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"source-builder")) },
        1
    );
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"source-builder.const-ref")) },
        1
    );
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"source-builder.output-alias")) },
        1
    );
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"errors.structured")) },
        1
    );
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"not-a-feature")) },
        0
    );
}

#[test]
fn feature_probe_rejects_invalid_string_argument() {
    let invalid = HologramString {
        ptr: std::ptr::null(),
        len: 1,
    };
    assert_eq!(unsafe { hologram_feature_supported(invalid) }, -1);
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        unsafe { hologram_feature_supported(ffi_str(b"source-builder")) },
        1
    );
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_NONE);
    assert!(hologram_last_error_message().is_null());
}

#[test]
fn source_builder_compile_round_trip() {
    assert_eq!(hologram_abi_version(), 1);
    let builder = hologram_source_builder_new();
    assert!(!builder.is_null());

    let shape = [4u64];
    let input = HologramTensorDesc {
        name: ffi_str(b"x"),
        dtype_id: 8,
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert!(unsafe { hologram_source_builder_input(builder, &input) } >= 0);

    let inputs = [ffi_str(b"x")];
    let op = HologramSourceOp {
        output: ffi_str(b"y"),
        op: ffi_str(b"relu"),
        inputs: inputs.as_ptr(),
        input_count: inputs.len(),
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert!(unsafe { hologram_source_builder_op(builder, &op) } >= 0);
    assert!(unsafe { hologram_source_builder_output(builder, ffi_str(b"y")) } >= 0);

    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    unsafe { hologram_source_builder_free(builder) };

    assert_archive_executes_one_input(&buf[..n as usize], 16);
}

#[test]
fn source_builder_output_alias_preserves_port_name() {
    let builder = hologram_source_builder_new();
    let shape = [1u64];
    add_f32_input(builder, b"x", &shape);
    add_unary_op(builder, b"relu", b"x", b"hidden", &shape);
    assert!(
        unsafe { hologram_source_builder_output_alias(builder, ffi_str(b"y"), ffi_str(b"hidden")) }
            >= 0
    );

    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    unsafe { hologram_source_builder_free(builder) };

    let archive = &buf[..n as usize];
    let handle = unsafe { hologram_session_load(archive.as_ptr(), archive.len()) };
    assert!(handle >= 0);
    assert_eq!(session_name(handle, true, 0), "x");
    assert_eq!(session_name(handle, false, 0), "y");
    unsafe { hologram_session_close(handle) };
}

#[test]
fn source_builder_inline_const_round_trip() {
    let builder = hologram_source_builder_new();
    let shape = [1u64];
    add_f32_input(builder, b"x", &shape);
    add_inline_f32_const(builder, b"w", &shape, &[1.0]);
    add_binary_op(builder, b"add", b"x", b"w", b"y", &shape);
    assert!(unsafe { hologram_source_builder_output(builder, ffi_str(b"y")) } >= 0);

    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    unsafe { hologram_source_builder_free(builder) };

    assert_archive_executes_one_input(&buf[..n as usize], 4);
}

#[test]
fn source_builder_const_ref_round_trip() {
    let weight_bytes = 1.0f32.to_le_bytes();
    let path = temp_tensor(&weight_bytes);
    let builder = hologram_source_builder_new();
    let shape = [1u64];
    add_f32_input(builder, b"x", &shape);
    add_external_f32_const(builder, b"w", &shape, &path, &weight_bytes);
    add_binary_op(builder, b"add", b"x", b"w", b"y", &shape);
    assert!(unsafe { hologram_source_builder_output(builder, ffi_str(b"y")) } >= 0);

    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    unsafe { hologram_source_builder_free(builder) };
    std::fs::remove_file(path).expect("remove temp tensor");

    assert_archive_executes_one_input(&buf[..n as usize], 4);
}

#[test]
fn source_builder_external_const_matches_native_source_archive() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weight_bytes = f32_bytes(&values);
    let path = temp_tensor(&weight_bytes);
    let native = compile_source_archive(golden_native_matmul_source());
    let builder = external_matmul_builder(&path, &weight_bytes);
    let external = compile_and_free_builder(builder);

    std::fs::remove_file(path).expect("remove temp tensor");
    assert_eq!(native, external);
}

#[test]
fn source_builder_const_ref_rejects_hash_mismatch() {
    let weight_bytes = 1.0f32.to_le_bytes();
    let path = temp_tensor(&weight_bytes);
    let builder = hologram_source_builder_new();
    let shape = [1u64];
    add_f32_input(builder, b"x", &shape);
    add_external_f32_const_with_hash(builder, b"w", &shape, &path, weight_bytes.len(), [0u8; 32]);
    add_binary_op(builder, b"add", b"x", b"w", b"y", &shape);
    assert!(unsafe { hologram_source_builder_output(builder, ffi_str(b"y")) } >= 0);

    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, -1);
    assert_eq!(hologram_last_error(), HOLOGRAM_ERROR_EXTERNAL_TENSOR);
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_EXTERNAL_TENSOR);
    assert!(!hologram_error_message().is_null());
    assert!(!hologram_last_error_message().is_null());
    unsafe { hologram_source_builder_free(builder) };
    std::fs::remove_file(path).expect("remove temp tensor");
}

#[test]
fn source_builder_reports_bad_op_error() {
    let builder = hologram_source_builder_new();
    let shape = [1u64];
    let inputs = [ffi_str(b"x")];
    let op = HologramSourceOp {
        output: ffi_str(b"y"),
        op: ffi_str(b"not_an_op"),
        inputs: inputs.as_ptr(),
        input_count: inputs.len(),
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert_eq!(unsafe { hologram_source_builder_op(builder, &op) }, -1);
    assert_eq!(hologram_last_error(), HOLOGRAM_ERROR_UNSUPPORTED_OP);
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_UNSUPPORTED_OP);
    assert!(!hologram_error_message().is_null());
    unsafe { hologram_source_builder_free(builder) };
}

#[test]
fn source_builder_reports_argument_dtype_and_shape_error_codes() {
    let builder = hologram_source_builder_new();
    assert_eq!(
        unsafe { hologram_source_builder_input(builder, std::ptr::null()) },
        -1
    );
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_INVALID_ARGUMENT);

    let shape = [1u64];
    let bad_dtype = HologramTensorDesc {
        name: ffi_str(b"x"),
        dtype_id: 99,
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert_eq!(
        unsafe { hologram_source_builder_input(builder, &bad_dtype) },
        -1
    );
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_UNSUPPORTED_DTYPE);

    let bad_shape = HologramTensorDesc {
        name: ffi_str(b"x"),
        dtype_id: 8,
        shape: HologramShape {
            dims: std::ptr::null(),
            rank: 1,
        },
    };
    assert_eq!(
        unsafe { hologram_source_builder_input(builder, &bad_shape) },
        -1
    );
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_SHAPE);
    unsafe { hologram_source_builder_free(builder) };
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
    assert_eq!(session_name(handle, true, 0), "x");
    assert_eq!(session_name(handle, false, 0), "y");
    assert_eq!(unsafe { hologram_session_input_dtype(handle, 0) }, 8);
    assert_eq!(unsafe { hologram_session_output_dtype(handle, 0) }, 8);
    assert_eq!(
        unsafe {
            hologram_session_extension(handle, b"missing".as_ptr(), 7, std::ptr::null_mut(), 0)
        },
        -1
    );

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
fn compile_source_reports_structured_diagnostic() {
    let src = b"input x\nop not_an_op x as=y\noutput y\n";
    let mut buf = vec![0u8; 16 * 1024];
    let n =
        unsafe { hologram_compile_source(src.as_ptr(), src.len(), buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, -1);
    assert_eq!(hologram_last_error_code(), HOLOGRAM_ERROR_UNSUPPORTED_OP);
    assert_eq!(hologram_last_error_line(), 2);
    assert!(hologram_last_error_column() > 0);
    assert!(!hologram_last_error_rejected().is_null());
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

fn add_f32_input(builder: *mut HologramSourceBuilder, name: &[u8], shape: &[u64]) {
    let input = tensor_desc(name, shape);
    assert!(unsafe { hologram_source_builder_input(builder, &input) } >= 0);
}

fn assert_archive_executes_one_input(archive: &[u8], input_len: usize) {
    let handle = unsafe { hologram_session_load(archive.as_ptr(), archive.len()) };
    assert!(handle >= 0);
    assert_eq!(unsafe { hologram_session_input_count(handle) }, 1);
    assert_eq!(unsafe { hologram_session_output_count(handle) }, 1);
    let output_len = unsafe { hologram_session_output_byte_len(handle, 0) };
    assert!(output_len > 0);

    let input = vec![0u8; input_len];
    let in_ptrs = [input.as_ptr()];
    let in_lens = [input.len()];
    let mut output = vec![0u8; output_len as usize];
    let out_ptrs = [output.as_mut_ptr()];
    let out_caps = [output.len()];
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
    unsafe { hologram_session_close(handle) };
}

fn add_inline_f32_const(
    builder: *mut HologramSourceBuilder,
    name: &[u8],
    shape: &[u64],
    values: &[f32],
) {
    let bytes = f32_bytes(values);
    let desc = HologramConstDesc {
        tensor: tensor_desc(name, shape),
        bytes: bytes.as_ptr(),
        byte_len: bytes.len(),
    };
    assert!(unsafe { hologram_source_builder_const(builder, &desc) } >= 0);
}

fn add_external_f32_const(
    builder: *mut HologramSourceBuilder,
    name: &[u8],
    shape: &[u64],
    path: &std::path::Path,
    bytes: &[u8],
) {
    add_external_f32_const_with_hash(builder, name, shape, path, bytes.len(), digest(bytes));
}

fn add_external_f32_const_with_hash(
    builder: *mut HologramSourceBuilder,
    name: &[u8],
    shape: &[u64],
    path: &std::path::Path,
    byte_len: usize,
    content_hash: [u8; 32],
) {
    let path = path.to_string_lossy();
    let desc = HologramExternalTensorDesc {
        tensor: tensor_desc(name, shape),
        path: ffi_str(path.as_bytes()),
        byte_offset: 0,
        byte_len: byte_len as u64,
        content_hash,
    };
    assert!(unsafe { hologram_source_builder_const_ref(builder, &desc) } >= 0);
}

fn add_binary_op(
    builder: *mut HologramSourceBuilder,
    op: &[u8],
    left: &[u8],
    right: &[u8],
    output: &[u8],
    shape: &[u64],
) {
    let inputs = [ffi_str(left), ffi_str(right)];
    let op = HologramSourceOp {
        output: ffi_str(output),
        op: ffi_str(op),
        inputs: inputs.as_ptr(),
        input_count: inputs.len(),
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert!(unsafe { hologram_source_builder_op(builder, &op) } >= 0);
}

fn add_unary_op(
    builder: *mut HologramSourceBuilder,
    op: &[u8],
    input: &[u8],
    output: &[u8],
    shape: &[u64],
) {
    let inputs = [ffi_str(input)];
    let op = HologramSourceOp {
        output: ffi_str(output),
        op: ffi_str(op),
        inputs: inputs.as_ptr(),
        input_count: inputs.len(),
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    };
    assert!(unsafe { hologram_source_builder_op(builder, &op) } >= 0);
}

fn session_name(handle: i32, input: bool, i: usize) -> String {
    let mut out = vec![0u8; 64];
    let n = if input {
        unsafe { hologram_session_input_name(handle, i, out.as_mut_ptr(), out.len()) }
    } else {
        unsafe { hologram_session_output_name(handle, i, out.as_mut_ptr(), out.len()) }
    };
    assert!(n >= 0);
    String::from_utf8(out[..n as usize].to_vec()).expect("port name utf8")
}

fn tensor_desc(name: &[u8], shape: &[u64]) -> HologramTensorDesc {
    HologramTensorDesc {
        name: ffi_str(name),
        dtype_id: 8,
        shape: HologramShape {
            dims: shape.as_ptr(),
            rank: shape.len(),
        },
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn golden_native_matmul_source() -> &'static [u8] {
    b"
        input x: f32[2, 3]
        const w: f32[3, 2] = [1, 2, 3, 4, 5, 6]
        let y: f32[2, 2] = matmul(x, w)
        output y
    "
}

fn compile_source_archive(source: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe {
        hologram_compile_source(source.as_ptr(), source.len(), buf.as_mut_ptr(), buf.len())
    };
    assert!(n > 0);
    buf.truncate(n as usize);
    buf
}

fn external_matmul_builder(path: &std::path::Path, bytes: &[u8]) -> *mut HologramSourceBuilder {
    let builder = hologram_source_builder_new();
    add_f32_input(builder, b"x", &[2, 3]);
    add_external_f32_const(builder, b"w", &[3, 2], path, bytes);
    add_binary_op(builder, b"matmul", b"x", b"w", b"y", &[2, 2]);
    assert!(unsafe { hologram_source_builder_output(builder, ffi_str(b"y")) } >= 0);
    builder
}

fn compile_and_free_builder(builder: *mut HologramSourceBuilder) -> Vec<u8> {
    let mut buf = vec![0u8; 16 * 1024];
    let n = unsafe { hologram_source_builder_compile(builder, buf.as_mut_ptr(), buf.len()) };
    unsafe { hologram_source_builder_free(builder) };
    assert!(n > 0);
    buf.truncate(n as usize);
    buf
}
