//! Generate checked-in low-level SDK binding files.

use std::path::Path;

fn main() {
    write(
        "sdk/python/hologram/_generated.py",
        &hologram_ffi::sdk::generate_python(),
    );
    write(
        "sdk/typescript/src/generated.ts",
        &hologram_ffi::sdk::generate_typescript(),
    );
}

fn write(path: &str, contents: &str) {
    let path = Path::new(path);
    let parent = path.parent().expect("generated file has parent");
    std::fs::create_dir_all(parent).expect("create generated SDK directory");
    std::fs::write(path, contents).expect("write generated SDK file");
}
