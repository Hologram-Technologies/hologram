fn main() {
    // Register custom cfg names.
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_webgpu)");
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // Metal: always available on macOS 10.14+.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-cfg=has_metal");
    }

    // WebGPU: available when the `webgpu` feature is active or on wasm32.
    if std::env::var("CARGO_FEATURE_WEBGPU").is_ok()
        || std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32"
    {
        println!("cargo:rustc-cfg=has_webgpu");
    }

    // CUDA: available when the `cuda` feature is active.
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-cfg=has_cuda");
    }
}
