fn main() {
    // Register custom cfg names so rustc doesn't warn about them.
    println!("cargo::rustc-check-cfg=cfg(has_metal)");
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");
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

    // CUDA: detect via CUDA_HOME env var or nvcc on PATH.
    let has_cuda = std::env::var("CUDA_HOME").is_ok()
        || std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .is_ok();
    if has_cuda {
        println!("cargo:rustc-cfg=has_cuda");
    }

    // WebGPU: available on wasm32 targets.
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default() == "wasm32" {
        println!("cargo:rustc-cfg=has_webgpu");
    }
}
