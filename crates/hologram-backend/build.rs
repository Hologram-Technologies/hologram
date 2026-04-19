fn main() {
    // Register custom cfg names.
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_webgpu)");
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");
    println!("cargo::rustc-check-cfg=cfg(has_accelerate)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Accelerate BLAS on macOS — critical for CPU matmul performance.
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=has_accelerate");
    }

    // Metal: enabled by feature flag OR auto-detected on macOS/iOS targets.
    if std::env::var("CARGO_FEATURE_METAL_BACKEND").is_ok()
        || target_os == "macos"
        || target_os == "ios"
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
