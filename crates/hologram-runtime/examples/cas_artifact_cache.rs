//! Real-world use-case: a **content-addressed artifact / build cache** (Nix/Bazel/CI-style).
//!
//! Identical artifacts dedup to one κ; a reproducible build is the κ of its inputs; GC reclaims
//! everything not reachable from a pinned build. Run: `cargo run -p hologram-runtime-wasmtime
//! --example cas_artifact_cache`.

use hologram_space::{ContainerManifest, REGISTRY};
use hologram_space::{KappaStore, Realization};
use hologram_tck::MemKappaStore;

fn main() {
    let cache = MemKappaStore::new();

    // Two independent builds emit a byte-identical object → one κ, stored once (content dedup).
    let obj_a = cache.put("blake3", b"<compiled: libfoo.o>").unwrap();
    let obj_b = cache.put("blake3", b"<compiled: libfoo.o>").unwrap();
    assert_eq!(obj_a, obj_b);
    println!(
        "dedup     : 2 identical artifacts → 1 κ ({} object in cache)",
        cache.approximate_count()
    );

    // A reproducible build = the κ of (object, source, toolchain) — same inputs ⇒ same build κ.
    let src = cache.put("blake3", b"<source tree>").unwrap();
    let toolchain = cache.put("blake3", b"<toolchain v1>").unwrap();
    let build = cache
        .put(
            "blake3",
            &ContainerManifest {
                code: obj_a,
                initial_state: src,
                parameters: toolchain,
            }
            .canonicalize(),
        )
        .unwrap();
    println!("build κ   : {}", build.as_str());

    // A stale artifact from an old build is unreachable; pin the current build and GC.
    let stale = cache.put("blake3", b"<stale: libfoo.o.old>").unwrap();
    cache.pin(&build).unwrap();
    let evicted = cache.gc(REGISTRY);
    println!(
        "gc        : reclaimed {} stale artifact(s); inputs retained = {}",
        evicted,
        cache.contains(&src) && cache.contains(&toolchain) && cache.contains(&obj_a)
    );
    assert!(!cache.contains(&stale) && cache.contains(&build));

    println!("OK — content-addressed cache: dedup + reproducible build κ + reachability GC");
}
