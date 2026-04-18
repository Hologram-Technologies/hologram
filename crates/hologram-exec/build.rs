fn main() {
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
}
