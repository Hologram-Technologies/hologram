fn main() {
    // Register custom cfg names.
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_webgpu)");
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // Metal: enabled by feature flag OR auto-detected on macOS targets.
    // Feature flag allows explicit control for cross-compiled releases.
    if std::env::var("CARGO_FEATURE_METAL_BACKEND").is_ok()
        || std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos"
        || std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "ios"
    {
        println!("cargo:rustc-cfg=has_metal");
    }

    // WebGPU: enabled by feature flag OR on wasm32 targets.
    if std::env::var("CARGO_FEATURE_WEBGPU").is_ok()
        || std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32"
    {
        println!("cargo:rustc-cfg=has_webgpu");
    }

    // CUDA: only via explicit feature flag.
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-cfg=has_cuda");
    }
}
