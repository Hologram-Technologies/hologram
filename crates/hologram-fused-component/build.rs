fn main() {
    // Register custom cfg names so rustc doesn't warn about them.
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_webgpu)");

    #[cfg(feature = "accelerate")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }

    // Enable native CPU features for SIMD autovectorization when building
    // for the host target (not cross-compiling).
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == std::env::consts::ARCH {
        println!("cargo:rustc-env=HOLOGRAM_NATIVE_CPU=1");
    }

    // ── Auto-detect GPU backends ──────────────────────────────────────────

    // Metal: always available on macOS 10.14+.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-cfg=has_metal");
    }

    // WebGPU: available when the `webgpu` feature is active (native via wgpu),
    // or on wasm32 targets (browser WebGPU).
    if std::env::var("CARGO_FEATURE_WEBGPU").is_ok()
        || std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32"
    {
        println!("cargo:rustc-cfg=has_webgpu");
    }
}
