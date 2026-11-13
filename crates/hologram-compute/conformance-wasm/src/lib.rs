//! wasm32 driver for the kernel conformance digest.
//!
//! Exports the same body the native test suite runs
//! (`hologram_compute::cpu::conformance`) as plain `extern "C"` functions
//! returning `u64`, so a host needs nothing but a WebAssembly engine to call
//! them: `run.mjs` loads the module with Node's built-in `WebAssembly` API and
//! compares its digests with the native ones. No bindings generator, no
//! imports.

/// Digest of `simd_f32_dot` and `simd_f32_sum`; must equal the native value.
#[no_mangle]
pub extern "C" fn hologram_dot_sum_digest() -> u64 {
    hologram_compute::cpu::conformance::dot_sum_digest()
}

/// Digest of `matmul_f32_blocked`; reported, not claimed equal to native.
#[no_mangle]
pub extern "C" fn hologram_matmul_digest() -> u64 {
    hologram_compute::cpu::conformance::matmul_digest()
}
