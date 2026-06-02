//! Snapshot checks for the public C ABI header.

const HEADER: &str = include_str!("../include/hologram.h");

const REQUIRED_SYMBOLS: &[&str] = &[
    "hologram_abi_version",
    "hologram_archive_format_version",
    "hologram_feature_supported",
    "hologram_last_error",
    "hologram_last_error_code",
    "hologram_error_message",
    "hologram_last_error_message",
    "hologram_last_error_line",
    "hologram_last_error_column",
    "hologram_last_error_rejected",
    "hologram_source_builder_new",
    "hologram_source_builder_free",
    "hologram_source_builder_input",
    "hologram_source_builder_const",
    "hologram_source_builder_const_ref",
    "hologram_source_builder_op",
    "hologram_source_builder_output",
    "hologram_source_builder_output_alias",
    "hologram_source_builder_compile",
    "hologram_compile_empty",
    "hologram_compile_source",
    "hologram_session_load",
    "hologram_session_input_count",
    "hologram_session_output_count",
    "hologram_session_kernel_count",
    "hologram_session_output_byte_len",
    "hologram_session_input_dtype",
    "hologram_session_output_dtype",
    "hologram_session_archive_fingerprint",
    "hologram_session_execute",
    "hologram_session_close",
    "hologram_session_input_name",
    "hologram_session_output_name",
    "hologram_session_input_shape",
    "hologram_session_output_shape",
    "hologram_session_extension",
];

const REQUIRED_CONSTANTS: &[&str] = &[
    "HOLOGRAM_ABI_VERSION",
    "HOLOGRAM_DTYPE_DEFAULT",
    "HOLOGRAM_DTYPE_F32",
    "HOLOGRAM_ERROR_NONE",
    "HOLOGRAM_ERROR_PARSE",
    "HOLOGRAM_ERROR_GRAPH",
    "HOLOGRAM_ERROR_UNSUPPORTED_OP",
    "HOLOGRAM_ERROR_BAD_ATTR",
    "HOLOGRAM_ERROR_SHAPE",
    "HOLOGRAM_ERROR_EXTERNAL_TENSOR",
    "HOLOGRAM_ERROR_ARCHIVE_LOAD",
    "HOLOGRAM_ERROR_EXECUTION",
    "HOLOGRAM_ERROR_ABI_MISMATCH",
    "HOLOGRAM_ERROR_INVALID_ARGUMENT",
    "HOLOGRAM_ERROR_UNSUPPORTED_DTYPE",
    "HOLOGRAM_ERROR_COMPILE",
];

#[test]
fn header_retains_required_symbols() {
    for symbol in REQUIRED_SYMBOLS {
        assert!(HEADER.contains(symbol), "missing ABI symbol {symbol}");
    }
}

#[test]
fn header_retains_required_constants() {
    for constant in REQUIRED_CONSTANTS {
        assert!(HEADER.contains(constant), "missing ABI constant {constant}");
    }
}

#[test]
fn header_uses_opaque_builder_handle() {
    assert!(HEADER.contains("typedef struct HologramSourceBuilder HologramSourceBuilder;"));
    assert!(!HEADER.contains("struct HologramSourceBuilder {"));
}
