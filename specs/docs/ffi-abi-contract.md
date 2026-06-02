# FFI ABI Contract

Status: initial SDK-facing contract for Plan 076 Phase 8.2.

## Version Checks

- `hologram_abi_version()` returns the native C ABI version. SDKs must check it
  at import/load time before calling any other optional entry point.
- `hologram_archive_format_version()` returns the `.holo` archive format version
  this library writes and accepts.
- `hologram_feature_supported(feature)` returns `1` for supported additive
  features, `0` for unsupported features, and `-1` for invalid arguments. The
  current feature names are:
  - `abi.v1`
  - `archive.v2`
  - `compile.empty`
  - `compile.source`
  - `session`
  - `source-builder`
  - `source-builder.const`
  - `source-builder.const-ref`
  - `source-builder.output-alias`
  - `errors.structured`
  - `errors.locations`

ABI additions must be additive within an ABI version. Removing or repurposing an
exported symbol, struct field, constant value, or feature string requires an ABI
version bump.

## Ownership

- `HologramString`, `HologramShape`, `HologramTensorDesc`,
  `HologramConstDesc`, `HologramExternalTensorDesc`, and `HologramSourceOp`
  point to caller-owned memory. The library copies names and inline constant
  bytes it needs during the call.
- `HologramSourceBuilder *` is owned by the caller and must be released with
  `hologram_source_builder_free`. Passing null to free is accepted.
- `hologram_source_builder_output(builder, name)` exposes an existing source
  symbol as an output port named by that same symbol.
- `hologram_source_builder_output_alias(builder, name, source)` exposes an
  existing source symbol as a semantically named output port. SDKs should use it
  for `Graph.output("port_name", tensor)` when `port_name` differs from the
  source tensor's internal symbol.
- `hologram_compile_empty`, `hologram_compile_source`, and
  `hologram_source_builder_compile` write archive bytes into caller-owned
  buffers. They return the full required length, `snprintf`-style. If the return
  value exceeds the provided capacity, the output was truncated and the caller
  must retry with a larger buffer.
- Session handles returned by `hologram_session_load` are integer handles owned
  by the process-local session table. They must be released with
  `hologram_session_close`.
- Name, shape, extension, and fingerprint accessors write into caller-owned
  buffers. They never return owned heap allocations.
- `hologram_last_error_message()` and `hologram_error_message()` return borrowed
  thread-local pointers valid until the next FFI call on the same thread. The
  caller must not free them.
- `hologram_last_error_rejected()` follows the same borrowed thread-local
  lifetime. It returns null when the last failure has no rejected source token.
- File-backed `const_ref` paths are consumed at compile time. Runtime execution
  never opens source paths.

## Error Contract

FFI functions return negative values on failure. After failure:

- `hologram_last_error_code()` returns a stable `HOLOGRAM_ERROR_*` category.
- `hologram_last_error_message()` returns a short diagnostic string.
- `hologram_last_error_line()` and `hologram_last_error_column()` return
  1-based source positions, or `0` when unavailable.
- `hologram_last_error_rejected()` returns the rejected token or source
  fragment, or null when unavailable.
- `hologram_last_error()` is retained as a compatibility alias for the code.
- Error state is thread-local and cleared by successful source-builder and
  feature-probe calls.

SDKs should map these categories to language-native exceptions without parsing
message text.

| Code | C constant | SDK category |
|---:|---|---|
| 1 | `HOLOGRAM_ERROR_PARSE` | parse error |
| 2 | `HOLOGRAM_ERROR_GRAPH` | graph construction / validation error |
| 3 | `HOLOGRAM_ERROR_UNSUPPORTED_OP` | unsupported or unknown op |
| 4 | `HOLOGRAM_ERROR_BAD_ATTR` | unsupported or malformed op attribute |
| 5 | `HOLOGRAM_ERROR_SHAPE` | shape validation error |
| 6 | `HOLOGRAM_ERROR_EXTERNAL_TENSOR` | external tensor reference error |
| 7 | `HOLOGRAM_ERROR_ARCHIVE_LOAD` | archive load error |
| 8 | `HOLOGRAM_ERROR_EXECUTION` | session execution error |
| 9 | `HOLOGRAM_ERROR_ABI_MISMATCH` | ABI / archive-format mismatch |
| 10 | `HOLOGRAM_ERROR_INVALID_ARGUMENT` | invalid SDK or FFI argument |
| 11 | `HOLOGRAM_ERROR_UNSUPPORTED_DTYPE` | unsupported dtype |
| 12 | `HOLOGRAM_ERROR_COMPILE` | compile error |

## Session Introspection

The session ABI exposes the metadata needed by SDKs to bind user inputs and
pre-size output buffers without parsing archive internals:

- `hologram_session_input_count(handle)`
- `hologram_session_output_count(handle)`
- `hologram_session_kernel_count(handle)`
- `hologram_session_input_name(handle, i, out, out_capacity)`
- `hologram_session_output_name(handle, i, out, out_capacity)`
- `hologram_session_input_shape(handle, i, out_dims, max_dims)`
- `hologram_session_output_shape(handle, i, out_dims, max_dims)`
- `hologram_session_input_dtype(handle, i)`
- `hologram_session_output_dtype(handle, i)`
- `hologram_session_output_byte_len(handle, i)`
- `hologram_session_archive_fingerprint(handle, out)`
- `hologram_session_extension(handle, key_ptr, key_len, out, out_capacity)`

Name, shape, and extension copy functions return the full required length or
rank, so callers can retry with a larger buffer. `hologram_session_extension`
returns `-1` for an absent key.

## Threading

The error record is thread-local. Builder handles are mutable authoring handles;
SDKs should not mutate the same builder concurrently. Session handles live in a
process-local table guarded by the native library; SDKs should still serialize
calls that mutate or close the same handle.

## Compatibility Tests

`crates/hologram-ffi/tests/abi_contract.rs` snapshots the required C header
symbols and constants. Removing a published symbol or constant must either fail
that test or be accompanied by an explicit ABI-version bump and migration note.
