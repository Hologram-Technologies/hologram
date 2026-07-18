# hologram-ffi

> The hologram C ABI and WASM bindings.

`hologram-ffi` exposes hologram's full compile / load / introspect / execute /
close surface across a C ABI, running against the CPU backend. It builds as both
a `cdylib` (a host shared library) and an `rlib`, and can emit WASM bindings when
the `wasm` feature is enabled.

The C surface is an application-author façade over the prism-backed hologram
crates (compiler, exec, archive, compute); it opts those library crates back into
`std` + the CPU backend for the host shared library.

## What it provides

- Versioning / capability probes — `hologram_abi_version`, `hologram_archive_format_version`, `hologram_feature_supported`, and the `HOLOGRAM_ABI_VERSION` / `HOLOGRAM_ERROR_*` constants.
- Thread-local error reporting — `hologram_last_error`, `hologram_last_error_message`, `hologram_last_error_line/column`, `hologram_last_error_rejected`.
- Source builder + compile — `hologram_source_builder_*` (input / const / op / output / compile) and `hologram_compile_source` / `hologram_compile_empty`.
- Sessions — `hologram_session_load`, `..._execute`, `..._close`, plus introspection (`..._input_count`, `..._output_byte_len`, `..._output_dtype`, `..._archive_fingerprint`, `..._input_shape`, and related accessors).
- FFI structs — `HologramString`, `HologramShape`, `HologramTensorDesc`, `HologramConstDesc` (all `#[repr(C)]`).
- `sdk` module — generated SDK binding metadata (`SdkDType`, `SdkOp`, `DTYPES`, `FEATURES`, `ops`) plus `generate_python` / `generate_typescript` codegen.

## Features

- `wasm` — enables the WASM bindings.

## Targets & build notes

Builds as a `cdylib` + `rlib`. This is a `std` host crate: it depends on the
hologram library crates with `std` (and the CPU backend) enabled. Enable `wasm`
for the WebAssembly binding surface.

Part of the [hologram](../../README.md) workspace.
